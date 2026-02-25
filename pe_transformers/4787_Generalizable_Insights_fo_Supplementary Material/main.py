import torch
from ogb_code_utils import decode_arr_to_seq
from pe import load_pe, load_transform
import argparse
import random
import torch
import copy
import math
import torch.nn as nn
from loguru import logger
from contextlib import nullcontext
from data import apply_edge_transform, load_tasks
from model import Transformer
import os
import pandas as pd
from torch_geometric.seed import seed_everything
import time


class RuntimeMemProfile:

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.runs = 0
        self.max_mem = 0
        self.avg_mem = 0

    def activate(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device=None)

    def stop(self, total_steps):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print("total time:" + str(self.end_time - self.start_time))
        self.avg_time = (self.end_time - self.start_time) / total_steps
        print("avg time" + str(self.avg_time))
        self.max_mem = torch.cuda.max_memory_allocated(device=None)
        print("mem allocation in MB:" + str(self.max_mem / (1024**2)))


def setup_run():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator 🚀: {device}")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"
    )
    logger.info(f"Data type: {dtype}")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    return ctx, ptdtype, device


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            master_process = device_id == 0
        else:
            device_id = 0
            master_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        master_process = True

    return device, device_id, device_count, master_process


MODEL_SIZE = {
    "XS": (4, 192, 12),
    "S": (6, 384, 12),
    "M": (8, 768, 16),
    "PCBA": (8, 384, 16),
    "PCQ": (16, 384, 16),
    "PCQ_large": (24, 384, 16),
    "PCBA_large": (24, 384, 32),
    "COCO": (12, 256, 16),
    "L": (12, 768, 16),
    "XL": (24, 1024, 16),
    "PCQ_XL": (24, 768, 16),
    "PCQ_32_heads": (12, 768, 32),
    "PCBA_XL": (24, 768, 32),
    "PCBA_512": (10, 512, 32),
    "PCBA_S": (12, 384, 32),
    "FINAL_SMALL": (16, 384, 32),
}


MODEL = {
    "transformer": Transformer,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--keep_task_modules", action="store_true")
    parser.add_argument("--pe", type=str, default="none")
    parser.add_argument("--path", type=str, default="./results.csv")
    parser.add_argument("--tasks", type=str, nargs="+", default=["pcqm4mv2"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_size", type=str, choices=list(MODEL_SIZE.keys()))
    parser.add_argument("--model", type=str, choices=list(MODEL.keys()), default="transformer"),
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--gradient_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_feat_graph", action="store_true")
    parser.add_argument("--test_every_iter", action="store_true")
    parser.add_argument("--warmup_iters", type=int, default=20000)
    parser.add_argument("--preserve_graph", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--max_rw_steps", type=int, default=32)
    parser.add_argument("--max_eigvals", type=int, default=32)
    parser.add_argument("--rwse_steps", type=int, default=16)
    parser.add_argument("--rrwp_steps", type=int, default=16)
    parser.add_argument("--spe_num_eigvals", type=int, default=8)
    parser.add_argument("--spe_hidden_dim", type=int, default=384)
    parser.add_argument("--spe_inner_dim", type=int, default=384)
    parser.add_argument("--spe_phi_dim", type=int, default=384)
    parser.add_argument("--spe_num_layers_phi", type=int, default=2)
    parser.add_argument("--spe_num_layers_rho", type=int, default=2)
    parser.add_argument("--lower_rank", type=bool, default=True)
    parser.add_argument("--normalized", type=bool, default=True)
    parser.add_argument("--large_graph", type=bool, default=True)
    parser.add_argument("--lpe_num_eigvals", type=int, default=32)
    parser.add_argument("--lpe_inner_dim", type=int, default=16)
    parser.add_argument("--lpe_position_aware", type=bool, default=True)
    parser.add_argument("--lpe_bias", type=bool, default=True)
    parser.add_argument("--ogb_max_seq_len", type=int, default=5)
    parser.add_argument("--ogb_max_vocab", type=int, default=5000)
    parser.add_argument("--edge_enc", type=str, default="MLP")
    parser.add_argument("--resource_management", action="store_true")
    args = parser.parse_args()

    logger.info(vars(args))
    # creating default dict for each task:
    if args.tasks[0] in ["pcqm4mv2"]:
        args.batch_sizes = [256]
        args.num_steps = 2000000
        args.edge_feat_graph = False
        args.warmup_iters = 20000
        args.log_every = 1000
    elif args.tasks[0] in ["algo_reas_flow"]:
        args.num_steps = math.ceil(3000000 / args.batch_sizes[0])
        args.edge_feat_graph = False
        args.warmup_iters = int(0.01 * args.num_steps)
        args.log_every = 200
    elif args.tasks[0] in ["coco"]:
        args.batch_sizes = [32]
        args.num_steps = 1000000
        args.edge_feat_graph = False
        args.warmup_iters = 10000
        args.log_every = 1000
    elif args.tasks[0] in ["algo_reas_edge", "algo_reas_mst"]:
        args.num_steps = math.ceil(3000000 / args.batch_sizes[0])
        args.edge_feat_graph = True
        args.preserve_graph = True
        args.warmup_iters = int(0.01 * args.num_steps)
        args.log_every = 200
    elif args.tasks[0] in ["ogb-code2"]:
        args.batch_sizes = [32]
        args.num_steps = 200000
        args.edge_feat_graph = False
        args.warmup_iters = 2000
        args.log_every = 1000
    seed_everything(args.seed)
    ctx, dtype, device = setup_run()

    batch_sizes = {k: v for k, v in zip(args.tasks, args.batch_sizes)}
    grad_accum = {k: v for k, v in zip(args.tasks, args.grad_accum)}
    logger.info(f"Batch sizes and grad accumulation plan: {batch_sizes} | {grad_accum}")
    transforms = load_transform(vars(args))
    num_layers, embed_dim, num_heads = MODEL_SIZE[args.model_size]
    encoder = load_pe(
        args.pe, vars(args), embed_dim=embed_dim, device=device, num_heads=num_heads
    )
    embedding_edge, transform_run = apply_edge_transform(
        args.edge_feat_graph, embed_dim
    )
    data, modules, funcs = load_tasks(
        args.tasks,
        args.root,
        batch_sizes,
        embed_dim,
        args.bias,
        transforms,
        transform_run,
        args,
    )

    orig_model = MODEL[args.model](
        modules,
        funcs,
        num_layers,
        embed_dim,
        num_heads,
        args.dropout,
        args.bias,
        encoder,
        transform=None,
        embedding_edge=embedding_edge,
        device=device,
        edge_enc=args.edge_enc,
    )
    logger.info(orig_model)
    orig_model.reset_parameters()

    model = orig_model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model params: " + str(param_count))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def get_loss(task):
        batch = data[task]["train"].sample().to(device)
        with ctx:
            preds = model(batch, task, device)
        if task == "ogb-code2":
            return funcs[task]["loss"](preds, batch.y_arr)
        elif task == "pcba":
            is_labeled = batch.y == batch.y

            return funcs[task]["loss"](preds[is_labeled], batch.y[is_labeled])
        else:
            return funcs[task]["loss"](preds, batch.y)

    @torch.inference_mode()
    def evaluate(split):
        metrics = {}

        for task, loaders in data.items():
            if task == "ogb-code2":
                arr_to_seq = lambda arr: decode_arr_to_seq(
                    arr, funcs[task]["idx2vocab"]
                )
            y_true = []
            y_pred = []
            is_labeled = []
            metric_name, metric_func = funcs[task]["metric"]
            if task == "ogb-code2":
                for batch in loaders[split]:
                    batch = batch.to(device)
                    if batch.x.shape[0] == 1:
                        pass
                    else:
                        with torch.no_grad():
                            pred_list = model(batch, task, device)

                        mat = []
                        for i in range(len(pred_list)):
                            mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
                        mat = torch.cat(mat, dim=1)

                        seq_pred = [arr_to_seq(arr) for arr in mat]

                        # PyG = 1.4.3
                        # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

                        # PyG >= 1.5.0
                        seq_ref = [batch.y[i] for i in range(len(batch.y))]

                        y_true.extend(seq_ref)
                        y_pred.extend(seq_pred)
                metrics[f"{task}_{split}_{metric_name}"] = metric_func(y_pred, y_true)
                metrics[f"{task}_{split}_loss"] = metric_func(y_pred, y_true)

            else:
                for batch in loaders[split]:
                    batch = batch.to(device)
                    with ctx:
                        preds = model(batch, task, device)
                    y_true.append(batch.y.detach().cpu())
                    y_pred.append(preds.detach().cpu())

                y_true = torch.cat(y_true)
                y_pred = torch.cat(y_pred)

                metrics[f"{task}_{split}_loss"] = funcs[task]["loss"](
                    y_pred, y_true
                ).item()
                metrics[f"{task}_{split}_{metric_name}"] = metric_func(y_pred, y_true)
        return metrics

    logger.info("Starting training 🍿")
    warmup_iters = args.warmup_iters

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / warmup_iters)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps - warmup_iters
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [warmup_iters],
    )
    logger.info(f"Cosine schedule with {warmup_iters} warm-up iters.")

    best_loss = None
    metrics = []
    state_dict = {}
    best_model = None

    if args.num_steps == 0:
        test_metrics = evaluate("test")
        results = [{"best_val_loss": best_loss, **test_metrics, **vars(args)}]

    if args.resource_management:
        Runtime_calc = RuntimeMemProfile()
        args.log_every = 1e9
        args.num_steps = 1000
    for step in range(args.num_steps):
        if step == 1 and args.resource_management:
            Runtime_calc.activate()
        _loss = 0.0
        for _ in range(len(args.tasks)):
            if len(list(data.keys())) == 1:
                task = list(data.keys())[0]
            else:
                task = random.choice(list(data.keys()))
            for _ in range(grad_accum[task]):

                train_loss = (
                    get_loss(task) * grad_accum[task] / sum(grad_accum.values())
                )
                train_loss.backward()
                _loss += train_loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if (step + 1) % args.log_every == 0:
            logger.info("Running validation")
            model.eval()
            val_metrics = evaluate("valid")

            val_loss = sum(
                [
                    val
                    for task_metric, val in val_metrics.items()
                    if task_metric.endswith("loss")
                ]
            )
            better, score = funcs[args.tasks[0]]["is_better"](best_loss, val_metrics)
            if score == 1.0 and args.tasks[0] == "algo_reas":
                break
            if better and args.test_every_iter:
                best_loss = score
                logger.info(f"New best_score: {best_loss}")
                test_metrics = evaluate("test")
                results = [
                    {
                        "best_val_score": best_loss,
                        **val_metrics,
                        **test_metrics,
                        **vars(args),
                    }
                ]
                lr = optimizer.param_groups[0]["lr"]
                metrics = {"lr": lr, **val_metrics, **test_metrics}
            elif better:
                best_loss = score
                logger.info(f"New best_score: {best_loss}")
                state_dict = copy.deepcopy(model.state_dict())
                best_model = copy.deepcopy(model)
                if args.save_checkpoint:
                    state_dict_path = (
                        "./"
                        + args.tasks[0]
                        + "_"
                        + str(args.model_size)
                        + "_"
                        + str(args.learning_rate)
                        + "_"
                        + str(args.seed)
                        + "_"
                        + str(args.pe)
                        + "_"
                        + str(args.rwse_steps)
                        + "_"
                        + str(args.lpe_num_eigvals)
                        + "_state_dict.pt"
                    )
                    torch.save(state_dict, state_dict_path)
                lr = optimizer.param_groups[0]["lr"]
                metrics = {"lr": lr, **val_metrics}
            else:
                lr = optimizer.param_groups[0]["lr"]
                metrics = {"lr": lr, **val_metrics}
            logger.info(f"Steps: {step} | Metrics: {metrics}")
            model.train()
    if args.resource_management:
        Runtime_calc.stop(step - 1)
    logger.info("Training complete ✨")

    if args.path is not None and not args.resource_management:

        if not args.test_every_iter:
            model = best_model
            test_metrics = evaluate("test")
            results = [
                {
                    "best_val_score": best_loss,
                    **val_metrics,
                    **test_metrics,
                    **vars(args),
                }
            ]
            metrics = {"lr": lr, **val_metrics, **test_metrics}
        logger.info(f"Steps: {step} | Metrics: {metrics}")
        logger.info(f"Logging results to {args.path}")
        if os.path.exists(path := args.path):
            pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
        else:
            pd.DataFrame(results).to_csv(path, header=True, index=False)
