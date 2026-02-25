import networkx as nx
import random
import torch
import tqdm
import os
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from networkx.algorithms import tree


def generate_graph(num_nodes, num_connect=1, p=0.1):
    graph = nx.erdos_renyi_graph(num_nodes, p=p)
    cc = list(nx.connected_components(graph))
    if len(cc) >= 2:
        for i in range(len(cc)):
            for _ in range(num_connect):
                j = random.choice([j for j in range(len(cc)) if j != i])
                a = random.choice(list(cc[i]))
                b = random.choice(list(cc[j]))
                graph.add_edge(a, b)
    return graph


def mst_graph(num_nodes, num_connect, p):
    while True:
        graph = generate_graph(num_nodes, num_connect, p)
        if len(graph.edges) < 1:
            continue

        weight_dict = {
            e: {"edge_attr": round(random.uniform(0, 10), 6)} for e in graph.edges
        }

        weights = [v["edge_attr"] for k, v in weight_dict.items()]
        if len(weights) == len(set(weights)):
            nx.set_edge_attributes(graph, weight_dict)
            edges = [
                (u, v)
                for u, v, _ in tree.minimum_spanning_edges(graph, weight="edge_attr")
            ]
            return graph, edges


def mst(num_nodes, num_connect=1, p=0.1):
    graph, edges = mst_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges:
            data.y[i] = 1
    data.edge_attr = data.edge_attr[:, None]
    return data


def addition_graph(num_nodes):
    graph = nx.empty_graph(3 * num_nodes + 1).to_directed()
    for i in range(3 * num_nodes):
        if i == (num_nodes - 1):
            edge_attr = 1
        elif i == (2 * num_nodes - 1):
            edge_attr = 2
        else:
            edge_attr = 0
        graph.add_edge(i, i + 1, edge_attr=edge_attr)
    return graph


def addition(num_nodes):
    graph = addition_graph(num_nodes)
    data = from_networkx(graph)
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)

    num1 = random.randint(10 ** (num_nodes - 1), 10**num_nodes - 1)
    num2 = random.randint(10 ** (num_nodes - 1), 10**num_nodes - 1)
    res = list(str(num1 + num2).zfill(num_nodes + 1))

    digits = [int(elem) for elem in list(str(num1))]
    digits += [int(elem) for elem in list(str(num2))]
    data.x[: 2 * num_nodes] = torch.tensor(digits)
    data.y = torch.tensor([int(elem) for elem in res])
    data.target_filter = torch.tensor(
        ([False] * 2 * num_nodes) + ([True] * (num_nodes + 1))
    )

    return data


def bridges_graph(num_nodes, num_connect, p):
    graph = generate_graph(num_nodes, num_connect, p)
    edges = list(nx.bridges(graph))
    return graph, edges


def bridges(num_nodes, num_connect=1, p=0.1):
    graph, edges = bridges_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges:
            data.y[i] = 1

    return data


def cycles(num_nodes, num_connect=1, p=0.1):
    graph, _ = bridges_graph(num_nodes, num_connect, p)
    cycle_nodes = list(set([el for cycle in nx.simple_cycles(graph) for el in cycle]))
    cycle_idx = torch.tensor(cycle_nodes, dtype=torch.long)
    data = from_networkx(graph)
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.y[cycle_idx] = 1
    return data


def flow_graph(num_nodes, num_connect, p):
    graph = generate_graph(num_nodes, num_connect, p).to_directed()
    weight_dict = {
        e: {"edge_attr": round(random.uniform(0, 3), 2)} for e in graph.edges
    }
    nx.set_edge_attributes(graph, weight_dict)
    nodes = list(range(num_nodes))
    source = random.choice(nodes)
    nodes = [n for n in nodes if n != source]
    sink = random.choice(nodes)
    value = nx.flow.maximum_flow_value(graph, source, sink, capacity="edge_attr")
    return graph, source, sink, value


def flow(num_nodes, num_connect=1, p=0.1):
    graph, source, sink, value = flow_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    data.y = torch.tensor(value)
    data.x[source] = 1
    data.x[sink] = 2
    data.edge_attr = data.edge_attr[:, None]
    return data


ALGORITHMS = {
    "bridges": bridges,
    "cycles": cycles,
    "mst": mst,
    "flow": flow,
}


CONFIG = {"bridges": (1, 0.05), "cycles": (1, 0.05), "mst": (3, 0.1), "flow": (2, 0.05)}


class AlgoReaso(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        num_nodes,
        num_samples,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        sp_edge=False,
    ):
        root = os.path.join(root, f"{name}_{num_nodes}")
        self.name = name
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.sp_edge = sp_edge
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):

        data_list = [
            ALGORITHMS[self.name](self.num_nodes, *CONFIG[self.name])
            for _ in tqdm.tqdm(range(self.num_samples))
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    for task in ALGORITHMS.keys():
        dataset = AlgoReaso("./datasets", task, 64, 10000)
        dataset = AlgoReaso("./datasets", task, 64, 10)
