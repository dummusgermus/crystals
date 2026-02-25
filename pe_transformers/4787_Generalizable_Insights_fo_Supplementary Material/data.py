import os
import numpy as np
from functools import partial
from torcheval.metrics.functional import multiclass_f1_score
from torch_geometric.datasets import LRGBDataset, ZINC
from generate_data import AlgoReaso
from ogb_code_utils import (
    ASTNodeEncoder,
    get_vocab_mapping,
    encode_y_to_arr,
    augment_edge,
)
from torch_geometric import transforms
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from loguru import logger
import torch
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


def center_features(dataset, x_mu=None, x_sigma=None, e_mu=None, e_sigma=None):
    if x_mu is None:
        x_mu, x_sigma = dataset.x.mean(0, keepdim=True), dataset.x.std(0, keepdim=True)
        e_mu, e_sigma = dataset.edge_attr.mean(0, keepdim=True), dataset.edge_attr.std(
            0, keepdim=True
        )
    dataset.x = (dataset.x - x_mu) / x_sigma
    dataset.edge_attr = (dataset.edge_attr - e_mu) / e_sigma
    return dataset, x_mu, x_sigma, e_mu, e_sigma


def ogb_code_loss(pred_list, ref_list):
    loss = 0
    multicls_criterion = torch.nn.CrossEntropyLoss()
    for i in range(len(pred_list)):
        loss += multicls_criterion(pred_list[i].to(torch.float32), ref_list[:, i])
    return loss / len(pred_list)


def ogb_code_node_enc(dataset, split_idx, args):

    nodetypes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz")
    )
    nodeattributes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz")
    )

    return nodetypes_mapping, nodeattributes_mapping


def mae(preds: torch.Tensor, targets: torch.Tensor):
    return (preds - targets).abs().mean(dtype=float).item()


def accuracy(preds: torch.Tensor, targets: torch.Tensor):
    return (preds.softmax(-1).argmax(-1) == targets).mean(dtype=float).item()


def f1(preds: torch.Tensor, targets: torch.Tensor):
    if preds.dtype != torch.long:
        preds = preds.softmax(-1).argmax(-1)
    return float(
        f1_score(
            targets.to(dtype=torch.float32).detach().cpu().numpy(),
            preds.to(dtype=torch.float32).detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def ap(preds: torch.Tensor, targets: torch.Tensor):
    preds = torch.sigmoid(preds)
    return sum(
        [
            float(
                average_precision_score(
                    targets[:, i].to(dtype=torch.float32).detach().cpu().numpy(),
                    preds[:, i].to(dtype=torch.float32).detach().cpu().numpy(),
                )
            )
            for i in range(preds.size(1))
        ]
    ) / preds.size(1)


def rocauc(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.softmax(-1).argmax(-1)
    return float(
        roc_auc_score(
            targets.to(dtype=torch.float32).detach().cpu().numpy(),
            preds.to(dtype=torch.float32).detach().cpu().numpy(),
        )
    )


def first_pool(data):
    x, _ = to_dense_batch(data.x, data.batch)
    return x[torch.arange(x.size(0), device=x.device), data.mapping]


def dataset_stats(dataset):
    return {
        "avg nodes": sum([data.x.shape[0] for data in dataset]) / len(dataset),
        "avg edges": sum([data.edge_index.shape[1] for data in dataset]) / len(dataset),
        "max nodes": max([data.x.shape[0] for data in dataset]),
        "max edges": max([data.edge_index.shape[1] for data in dataset]),
    }


class MLP(nn.Module):
    def __init__(self, embed_dim, output_dim, bias, squeeze=False):
        super().__init__()
        self.nn = nn.Sequential(
            *[
                nn.Linear(embed_dim, embed_dim, bias=bias),
                nn.GELU(),
                nn.LayerNorm(embed_dim, bias),
                nn.Linear(embed_dim, output_dim, bias=bias),
            ]
        )
        self.squeeze = squeeze

    def forward(self, x):
        x = self.nn(x)
        if self.squeeze:
            x = x.squeeze()
        return x


def load_pcqm4mv2(
    root,
    embed_dim,
    bias,
    max_rw_steps,
    max_eigvals,
    transform,
    transform_run,
    *args,
    **kwargs,
):  # 4M graphs
    dataset = PygPCQM4Mv2Dataset(root, pre_transform=transform, transform=transform_run)
    splits = dataset.get_idx_split()

    train_dataset = dataset[splits["train"]]
    valid_dataset = dataset[splits["valid"]]
    test_dataset = dataset[splits["valid"]]

    node_encoder = AtomEncoder(embed_dim)
    edge_encoder = BondEncoder(embed_dim)

    decoder = MLP(embed_dim, 1, bias, squeeze=True)

    prepare = lambda x: x

    evaluator = PCQM4Mv2Evaluator()
    metric = (
        "mae",
        lambda y_pred, y_true: evaluator.eval({"y_pred": y_pred, "y_true": y_true})[
            "mae"
        ],
    )

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score > val_metrics["pcqm4mv2_valid_mae"]
        return better, val_metrics["pcqm4mv2_valid_mae"]

    def pool(data):
        return global_mean_pool(data.x, data.batch)

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.l1_loss,
        metric,
        None,
        is_better,
    )


def load_pascal(root, embed_dim, bias, transform, *args, **kwargs):  # 10K graphs

    train_dataset, x_mu, x_sigma, e_mu, e_sigma = center_features(
        LRGBDataset(root, "PascalVOC-SP", split="train", pre_transform=transform)
    )
    valid_dataset, _, _, _, _ = center_features(
        LRGBDataset(root, "PascalVOC-SP", split="val", pre_transform=transform),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )
    test_dataset, _, _, _, _ = center_features(
        LRGBDataset(root, "PascalVOC-SP", split="test", pre_transform=transform),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )

    node_encoder = nn.Linear(train_dataset.num_node_features, embed_dim)
    edge_encoder = nn.Linear(train_dataset.num_edge_features, embed_dim)
    decoder = MLP(embed_dim, train_dataset.num_classes, bias)

    prepare = lambda x: x

    def pool(data):
        return data.x

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["pascal_valid_f1"]
        return better, val_metrics["pascal_valid_f1"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.cross_entropy,
        ("f1", f1),
        None,
        is_better,
    )


def load_coco(
    root, embed_dim, bias, max_rw_steps, max_eigvals, transform=None, **kwargs
):  # 120K graphs

    train_dataset, x_mu, x_sigma, e_mu, e_sigma = center_features(
        LRGBDataset(root, "COCO-SP", split="train", pre_transform=transform)
    )
    valid_dataset, _, _, _, _ = center_features(
        LRGBDataset(root, "COCO-SP", split="val", pre_transform=transform)
    )
    test_dataset, _, _, _, _ = center_features(
        LRGBDataset(root, "COCO-SP", split="test", pre_transform=transform)
    )
    node_encoder = nn.Linear(train_dataset.num_node_features, embed_dim)
    edge_encoder = nn.Linear(train_dataset.num_edge_features, embed_dim)
    decoder = MLP(embed_dim, train_dataset.num_classes, bias)

    prepare = lambda x: x

    def pool(data):
        return data.x

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["coco_valid_f1"]
        return better, val_metrics["coco_valid_f1"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.cross_entropy,
        ("f1", f1),
        None,
        is_better,
    )


def load_algo_reas_edge(
    root,
    embed_dim,
    bias,
    max_rw_steps,
    max_eigvals,
    transform=None,
    transform_run=None,
    preserve_graph=False,
    **kwargs,
):
    transform_run = partial(
        transform_run,
        preserve_nodes=True,
        undirected=True,
        edge_level=True,
        preserve_graph=preserve_graph,
    )
    train_dataset = AlgoReaso(
        root, "bridges", 16, 1000000, pre_transform=Compose([transform_run, transform])
    )  # 16 1mio
    valid_dataset = AlgoReaso(
        root, "bridges", 16, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 16
    test_dataset = AlgoReaso(
        root, "bridges", 64, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 24
    node_encoder = None
    edge_encoder = None
    decoder = MLP(embed_dim, 2, bias)

    prepare = lambda x: x

    def pool(data):
        return data.x

    def is_better(best_score, val_metrics):
        better = (
            best_score is None or best_score < val_metrics["algo_reas_edge_valid_f1"]
        )
        return better, val_metrics["algo_reas_edge_valid_f1"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.cross_entropy,
        ("f1", f1),
        None,
        is_better,
    )


def load_algo_reas_mst(
    root,
    embed_dim,
    bias,
    max_rw_steps,
    max_eigvals,
    transform=None,
    transform_run=None,
    preserve_graph=False,
    **kwargs,
):
    transform_run = partial(
        transform_run,
        preserve_nodes=True,
        undirected=True,
        edge_level=True,
        preserve_graph=preserve_graph,
    )
    train_dataset = AlgoReaso(
        root, "mst", 16, 1000000, pre_transform=Compose([transform_run, transform])
    )  # 16 1mio
    valid_dataset = AlgoReaso(
        root, "mst", 16, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 16
    test_dataset = AlgoReaso(
        root, "mst", 64, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 24
    node_encoder = None
    edge_encoder = nn.Linear(1, embed_dim)
    decoder = MLP(embed_dim, 2, bias)

    prepare = lambda x: x

    def pool(data):
        return data.x

    def is_better(best_score, val_metrics):
        better = (
            best_score is None or best_score < val_metrics["algo_reas_mst_valid_f1"]
        )
        return better, val_metrics["algo_reas_mst_valid_f1"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.cross_entropy,
        ("f1", f1),
        None,
        is_better,
    )


def load_algo_reas_flow(
    root,
    embed_dim,
    bias,
    max_rw_steps,
    max_eigvals,
    transform=None,
    transform_run=None,
    preserve_graph=False,
    **kwargs,
):
    train_dataset = AlgoReaso(
        root, "flow", 16, 1000000, pre_transform=Compose([transform_run, transform])
    )  # 16 1mio
    valid_dataset = AlgoReaso(
        root, "flow", 16, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 16
    test_dataset = AlgoReaso(
        root, "flow", 64, 10000, pre_transform=Compose([transform_run, transform])
    )  # 1000 samples, 24
    node_encoder = nn.Embedding(3, embed_dim)
    edge_encoder = nn.Linear(1, embed_dim)
    decoder = MLP(embed_dim, 1, bias, squeeze=True)  # binary classification problem

    prepare = lambda x: x

    evaluator = PCQM4Mv2Evaluator()
    metric = (
        "mae",
        lambda y_pred, y_true: evaluator.eval({"y_pred": y_pred, "y_true": y_true})[
            "mae"
        ],
    )

    def is_better(best_score, val_metrics):
        better = (
            best_score is None or best_score > val_metrics["algo_reas_flow_valid_mae"]
        )
        return better, val_metrics["algo_reas_flow_valid_mae"]

    def pool(data):
        return global_mean_pool(data.x, data.batch)

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.l1_loss,
        metric,
        None,
        is_better,
    )


def load_algo_reas_add(
    root, embed_dim, bias, transform=None, transform_run=None, **kwargs
):
    train_dataset = AlgoReaso(
        root, "addition", 4, 1000000, pre_transform=transform, transform=transform_run
    )  # 16 1mio
    valid_dataset = AlgoReaso(
        root, "addition", 4, 10000, pre_transform=transform, transform=transform_run
    )  # 1000 samples, 16
    test_dataset = AlgoReaso(
        root, "addition", 8, 10000, pre_transform=transform, transform=transform_run
    )  # 1000 samples, 24
    node_encoder = nn.Embedding(10, embed_dim)
    edge_encoder = nn.Embedding(3, embed_dim)
    decoder = MLP(embed_dim, 10, bias)  # binary classification problem

    prepare = lambda x: x

    def pool(data):
        return data.x

    def is_better(best_score, val_metrics):
        better = (
            best_score is None or best_score < val_metrics["algo_reas_add_valid_ap"]
        )
        return better, val_metrics["algo_reas_add_valid_ap"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        F.cross_entropy,
        ("ap", multiclass_f1_score),
        None,
        is_better,
    )


def load_ogb_code(
    root,
    embed_dim,
    bias,
    ogb_max_seq_len,
    ogb_max_vocab,
    transform,
    max_rw_steps,
    max_eigvals,
    **kwargs,
):
    dataset = PygGraphPropPredDataset(
        name="ogbg-code2", root=root, pre_transform=transform
    )

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print(
        "Target seqence less or equal to {} is {}%.".format(
            ogb_max_seq_len, np.sum(seq_len_list <= ogb_max_seq_len) / len(seq_len_list)
        )
    )

    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = get_vocab_mapping(
        [dataset.data.y[i] for i in split_idx["train"]], ogb_max_vocab
    )

    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, ogb_max_seq_len)]
    )

    evaluator = Evaluator("ogbg-code2")

    filter_mask_train = (
        np.array([dataset[i].num_nodes for i in split_idx["train"]]) <= 1000
    )
    filter_mask_valid = (
        np.array([dataset[i].num_nodes for i in split_idx["valid"]]) <= 1000
    )
    filter_mask_test = (
        np.array([dataset[i].num_nodes for i in split_idx["test"]]) <= 1000
    )

    train_dataset = dataset[split_idx["train"][filter_mask_train]]
    valid_dataset = dataset[split_idx["valid"][filter_mask_valid]]
    test_dataset = dataset[split_idx["test"][filter_mask_test]]

    nodetypes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz")
    )
    nodeattributes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz")
    )

    node_encoder = ASTNodeEncoder(
        embed_dim,
        num_nodetypes=len(nodetypes_mapping["type"]),
        num_nodeattributes=len(nodeattributes_mapping["attr"]),
        max_depth=20,
    )

    edge_encoder = torch.nn.Linear(2, embed_dim)
    decoder = torch.nn.ModuleList(
        [torch.nn.Linear(embed_dim, len(vocab2idx)) for _ in range(ogb_max_seq_len)]
    )

    prepare = lambda x: x

    def pool(data):
        return data.x

    metric = (
        "F1",
        lambda seq_pred_list, seq_ref_list: evaluator.eval(
            {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
        )["F1"],
    )

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["ogb-code2_valid_F1"]
        return better, val_metrics["ogb-code2_valid_F1"]

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        node_encoder,
        edge_encoder,
        decoder,
        prepare,
        pool,
        ogb_code_loss,
        metric,
        idx2vocab,
        is_better,
    )


class TrainingDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.loader)

    def sample(self):
        minibatch = next(self.iterator, None)
        if minibatch is None:
            self.iterator = iter(self.loader)
            minibatch = next(self.iterator, None)
        return minibatch


def extract_task(batch_size, task_data):
    return (
        {
            "train": TrainingDataLoader(task_data[0], batch_size),
            "valid": DataLoader(task_data[1], batch_size),
            "test": DataLoader(task_data[2], 8),
        },
        nn.ModuleDict(
            {
                "node": task_data[3],
                "edge": task_data[4],
                "decoder": task_data[5],
            }
        ),
        {
            "prepare": task_data[6],
            "pooling": task_data[7],
            "loss": task_data[8],
            "metric": task_data[9],
            "idx2vocab": task_data[10],
            "is_better": task_data[11],
        },
    )


TASKS = {
    "pascal": load_pascal,
    "coco": load_coco,
    "pcqm4mv2": load_pcqm4mv2,
    "ogb-code2": load_ogb_code,
    "algo_reas_edge": load_algo_reas_edge,
    "algo_reas_mst": load_algo_reas_mst,
    "algo_reas_add": load_algo_reas_add,
    "algo_reas_flow": load_algo_reas_flow,
}


def load_tasks(
    tasks, root, batch_sizes, embed_dim, bias, transform, transform_run, args
):
    tasks = {k: v for k, v in TASKS.items() if k in tasks}
    data, modules, funcs = {}, {}, {}
    for task, _load in tasks.items():
        logger.info(f"Loading task {task}")
        _data, _modules, _funcs = extract_task(
            batch_sizes[task],
            _load(
                root=root,
                embed_dim=embed_dim,
                bias=bias,
                max_rw_steps=args.max_rw_steps,
                max_eigvals=args.max_eigvals,
                ogb_max_seq_len=args.ogb_max_seq_len,
                transform=transform,
                ogb_max_vocab=args.ogb_max_vocab,
                transform_run=transform_run,
                preserve_graph=args.preserve_graph,
            ),
        )
        data[task] = _data
        modules[task] = _modules
        funcs[task] = _funcs
    return data, modules, funcs


def load_pre_process_tasks(tasks, root, embed_dim, bias, transform, args):
    tasks = {k: v for k, v in TASKS.items() if k in tasks}
    data, modules, funcs = {}, {}, {}
    for task, _load in tasks.items():
        _load(root, embed_dim, bias, transform, args)
    return data, modules, funcs


def remap_features(num_nodes, edge_index, tuple_index, features):
    outer_shape = features.shape[1:]
    adj = torch.zeros((num_nodes, num_nodes, *outer_shape), dtype=features.dtype)
    adj[edge_index[0], edge_index[1]] = features
    return adj[tuple_index[:, 0], tuple_index[:, 1]]


def edge_transform(
    data, preserve_nodes=True, preserve_graph=False, undirected=True, edge_level=False
):
    if undirected:
        adj = torch.zeros((data.num_nodes, data.num_nodes))
        adj[data.edge_index[0], data.edge_index[1]] = 1
        tuple_index = torch.triu(adj).nonzero()

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            orig_edge_attr = remap_features(
                data.num_nodes, data.edge_index, tuple_index, data.edge_attr
            )
        else:
            orig_edge_attr = None

        if edge_level and len(data.y) == data.edge_index.size(1):

            y = remap_features(data.num_nodes, data.edge_index, tuple_index, data.y)
        else:
            y = data.y
    else:
        tuple_index = data.edge_index.T
        orig_edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        y = data.y

    mask = [True] * tuple_index.size(0)

    if preserve_nodes:
        tuple_index = torch.cat(
            [torch.arange(data.num_nodes)[:, None].repeat(1, 2), tuple_index], 0
        )

        mask = [False] * data.num_nodes + mask

    edge_mask = tuple_index[:, None] == tuple_index[None]
    adj = torch.zeros(tuple_index.size(0), tuple_index.size(0), dtype=torch.long)

    self_loop_mask = torch.logical_and(edge_mask[:, :, 0], edge_mask[:, :, 1])
    edge_index1 = (~self_loop_mask & edge_mask[:, :, 0]).nonzero().T
    edge_index2 = (~self_loop_mask & edge_mask[:, :, 1]).nonzero().T

    adj[edge_index1[0], edge_index1[1]] = 1
    adj[edge_index2[0], edge_index2[1]] = 1 if undirected else 2

    if preserve_graph:
        assert preserve_nodes, "preserve_graph requires preserve_nodes to be True"
        adj[data.edge_index[0], data.edge_index[1]] = adj.max() + 1

    edge_index = adj.nonzero().T
    edge_attr = adj[edge_index[0], edge_index[1]] - 1
    x = torch.tensor(mask).to(torch.long)

    data_dict = dict(
        token_mask=torch.tensor(mask),
        x=x,
        edge_index=edge_index,
        num_nodes=tuple_index.size(0),
        y=y,
    )

    if (edge_attr > 0).any():
        data_dict["edge_attr"] = edge_attr

    if hasattr(data, "x") and data.x is not None:
        data_dict["x_node"] = data.x

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_dict["x_edge"] = orig_edge_attr

    return Data(**data_dict)


def apply_edge_transform(flag, embed_dim):
    if flag:
        embedding = torch.nn.Embedding(3, embed_dim)
        return embedding, edge_transform
    else:
        return None, None
