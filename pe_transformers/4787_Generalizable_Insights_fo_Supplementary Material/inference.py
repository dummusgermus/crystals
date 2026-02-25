from functools import partial
import torch
import tqdm
from pe import load_pe, load_transform
import argparse
import torch
from loguru import logger
from data import apply_edge_transform, load_tasks
from generate_data import ALGORITHMS, CONFIG
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from main import setup_run, MODEL_SIZE, MODEL
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--inference_task", type=str, default="cycles")
    parser.add_argument("--pe", type=str, default="none")
    parser.add_argument("--path", type=str, default="./results/results.csv")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["pcqm4mv2"]
    )
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1024])
    parser.add_argument("--model_size", type=str, choices=list(MODEL_SIZE.keys()))
    parser.add_argument("--model", type=str, choices=list(MODEL.keys()), default="transformer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_feat_graph", action="store_true")
    parser.add_argument("--preserve_graph", action="store_true")
    parser.add_argument("--max_rw_steps", type=int, default=32)
    parser.add_argument("--max_eigvals", type=int, default=32)
    parser.add_argument("--rwse_steps", type=int, default=16)
    parser.add_argument("--rrwp_steps", type=int, default=16)
    parser.add_argument("--spe_num_eigvals", type=int, default=8)
    parser.add_argument("--spe_hidden_dim", type=int, default=32)
    parser.add_argument("--spe_inner_dim", type=int, default=16)
    parser.add_argument("--spe_phi_dim", type=int, default=32)
    parser.add_argument("--spe_num_layers_phi", type=int, default=8)
    parser.add_argument("--spe_num_layers_rho", type=int, default=8)
    parser.add_argument("--lower_rank", type=bool, default=False)
    parser.add_argument("--normalized", type=bool, default=False)
    parser.add_argument("--large_graph", type=bool, default=False)
    parser.add_argument("--lpe_num_eigvals", type=int, default=32)
    parser.add_argument("--lpe_inner_dim", type=int, default=32)
    parser.add_argument("--lpe_position_aware", type=bool, default=False)
    parser.add_argument("--lpe_bias", type=bool, default=False)
    parser.add_argument("--ogb_max_seq_len", type=int, default=5)
    parser.add_argument("--ogb_max_vocab", type=int, default=5000)
    parser.add_argument("--edge_enc", type=str, default="MLP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/algo_reas_edge_S_0.0005_54_lpe_state_dict.pt",
    )
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--extrapolate_start", type=int, default=16)
    parser.add_argument("--extrapolate_end", type=int, default=32)
    parser.add_argument("--evaluated_size", type=int, default=16)
    args = parser.parse_args()

    logger.info(vars(args))

    if args.tasks[0] in ["algo_reas_edge", "algo_reas_mst"]:
        args.edge_feat_graph = True
    elif args.tasks[0] in ["pascal"]:
        args.batch_sizes = [2]

    seed_everything(args.seed)
    ctx, dtype, device = setup_run()

    print(f"Evaluating checkpoint {args.checkpoint}")

    batch_sizes = {k: v for k, v in zip(args.tasks, args.batch_sizes)}
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
        fast_inference=True,
    )
    logger.info(orig_model)
    orig_model.reset_parameters()

    state_dict = torch.load(args.checkpoint, "cpu", weights_only=True)

    if args.tasks[0] in ["pascal"]:
        upd_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("task_modules")
        }
        for k, v in orig_model.state_dict().items():
            if k.startswith("task_modules"):
                upd_state_dict[k] = v
                if k.startswith("task_modules.pascal") and not "decoder" in k:
                    upd_state_dict[k] = state_dict[k.replace("pascal", "coco")]
            elif k not in upd_state_dict:
                upd_state_dict[k] = v
        state_dict = upd_state_dict

    orig_model.load_state_dict(state_dict)

    model = orig_model.to(device)
    model.eval()

    if args.tasks[0] == "algo_reas_edge":
        task_name = "bridges"
        transform_run = partial(
            transform_run,
            preserve_nodes=True,
            undirected=True,
            edge_level=True,
            preserve_graph=True,
        )
    elif args.tasks[0] == "algo_reas_mst":
        task_name = "mst"
        transform_run = partial(
            transform_run,
            preserve_nodes=True,
            undirected=True,
            edge_level=True,
            preserve_graph=True,
        )
    elif args.tasks[0] == "algo_reas_flow":
        task_name = "flow"
        transform_run = lambda x: x

    parameters = dict(
        checkpoint=args.checkpoint,
        shots=args.shots,
        evaluated_size=args.evaluated_size,
        pe=args.pe,
        seed=args.seed,
    )

    if args.inference_task == "extrapolation":
        metric_name, metric_func = funcs[args.tasks[0]]["metric"]
        results = []
        for num_nodes in range(args.extrapolate_start, args.extrapolate_end, 16):
            y_true = []
            y_pred = []
            failed = False
            for _ in tqdm.tqdm(range(25)):
                batch = Batch.from_data_list(
                    [
                        Compose([transform_run, transforms])(
                            ALGORITHMS[task_name](num_nodes, *CONFIG[task_name])
                        )
                        for _ in range(4)
                    ]
                ).to(device)
                try:
                    with torch.inference_mode():
                        with ctx:
                            preds = model(batch, args.tasks[0], device)
                except torch.OutOfMemoryError:
                    failed = True
                    break
                y_true.append(batch.y.detach().cpu())
                y_pred.append(preds.detach().cpu())
            if not failed:
                y_true = torch.cat(y_true)
                y_pred = torch.cat(y_pred)
                score = metric_func(y_pred, y_true)
                print(num_nodes, metric_name, score)
                results.append(dict(**parameters, num_nodes=num_nodes, score=score))
            else:
                break

    elif args.inference_task == "cycles":
        metric_name, metric_func = funcs[args.tasks[0]]["metric"]
        y_true = []
        y_pred = []
        for _ in range(16):
            batch1 = Batch.from_data_list(
                [
                    Compose([transform_run, transforms])(
                        ALGORITHMS["cycles"](16, *CONFIG[task_name])
                    )
                    for _ in range(args.shots)
                ]
            ).to(device)
            with torch.inference_mode():
                with ctx:
                    train = model(
                        batch1, args.tasks[0], device, return_node_embeddings=True
                    )

            batch2 = Batch.from_data_list(
                [
                    Compose([transform_run, transforms])(
                        ALGORITHMS["cycles"](args.evaluated_size, *CONFIG[task_name])
                    )
                    for _ in range(16)
                ]
            ).to(device)
            with torch.inference_mode():
                with ctx:
                    query = model(
                        batch2, args.tasks[0], device, return_node_embeddings=True
                    )

            idx = (
                torch.linalg.norm((query[:, None] - train), dim=2)
                .topk(3, largest=False)
                .indices
            )
            preds = batch1.y[idx].mode(1).values
            y_true.append(batch2.y.detach().cpu())
            y_pred.append(preds.detach().cpu())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        score = metric_func(y_pred, y_true)
        print(metric_name, score)
        results = [dict(**parameters, score=score)]
    elif args.inference_task == "pascal":
        metric_name, metric_func = funcs[args.tasks[0]]["metric"]

        train_set = []
        y = []
        for _ in tqdm.tqdm(range(args.shots)):
            batch1 = data[args.tasks[0]]["train"].sample().to(device)
            with torch.inference_mode():
                with ctx:
                    train = model(
                        batch1, args.tasks[0], device, return_node_embeddings=True
                    )
                    train_set.append(train)
                    y.append(batch1.y)
        train_set = torch.cat(train_set)
        print(train_set.shape)
        y = torch.cat(y)
        y_true = []
        y_pred = []
        for batch2 in tqdm.tqdm(data[args.tasks[0]]["test"]):
            batch2 = batch2.to(device)
            with torch.inference_mode():
                with ctx:
                    query = model(
                        batch2, args.tasks[0], device, return_node_embeddings=True
                    )

            num_processed = 0
            to_process = 1000
            dists = []
            while num_processed < train_set.size(0):
                train = train_set[num_processed : num_processed + to_process]
                dist = torch.linalg.norm((query[:, None] - train), dim=2)
                dists.append(dist)
                num_processed += to_process
            dist = torch.cat(dists, 1)
            idx = dist.topk(5, largest=False).indices
            preds = y[idx].mode(1).values

            y_true.append(batch2.y.detach().cpu())
            y_pred.append(preds.detach().cpu())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        score = metric_func(y_pred, y_true)
        print(metric_name, score)
        results = [dict(**parameters, score=score)]

    path = f"{args.inference_task}.csv"
    if path is not None:
        if os.path.exists(path):
            pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
        else:
            pd.DataFrame(results).to_csv(path, header=True, index=False)
