import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_dense_adj
#from xformers.ops import memory_efficient_attention
import torch.nn.functional as F

STDDEV = 0.02


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=STDDEV)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=STDDEV)


class MLP(nn.Module):
    def __init__(self, embed_dim, output_dim, bias, squeeze=False):
        super().__init__()
        self.nn = nn.Sequential(
            *[
                nn.Linear(embed_dim, 2 * embed_dim, bias=bias),
                nn.ReLU(),
                # nn.LayerNorm(embed_dim, bias),
                nn.Linear(2 * embed_dim, output_dim, bias=bias),
            ]
        )
        self.squeeze = squeeze

    def forward(self, x):
        x = self.nn(x)
        if self.squeeze:
            x = x.squeeze()
        return x


class FastInferenceLayer(nn.TransformerEncoderLayer):
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal: bool = False):
        B, T, C = x.size()

        result = F.linear(x, self.self_attn.in_proj_weight)
        q, k, v = torch.chunk(result, 3, dim=-1)
        q = q.view(B, T, self.self_attn.num_heads, C // self.self_attn.num_heads)
        k = k.view(B, T, self.self_attn.num_heads, C // self.self_attn.num_heads)
        v = v.view(B, T, self.self_attn.num_heads, C // self.self_attn.num_heads)

        x = memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_mask.reshape(B, self.self_attn.num_heads, T, T).to(q.dtype),
        )
        x = x.view(B, T, C)
        return self.self_attn.out_proj(x)


class Transformer(nn.Module):
    def __init__(
        self,
        task_modules,
        funcs,
        num_layers,
        embed_dim,
        num_heads,
        dropout,
        bias,
        pe_encoder,
        transform,
        embedding_edge,
        edge_enc,
        device,
        fast_inference=False,
    ):
        super().__init__()
        self.task_modules = nn.ModuleDict(task_modules)
        self.funcs = funcs
        self.device = device
        self.embedding_edge = embedding_edge
        if fast_inference:
            layer = FastInferenceLayer(
                embed_dim, num_heads, embed_dim, dropout, batch_first=True, bias=False
            )  # , activation='gelu')
        else:
            layer = nn.TransformerEncoderLayer(
                embed_dim, num_heads, embed_dim, dropout, batch_first=True, bias=bias
            )
        self.encoder = torch.compile(nn.TransformerEncoder(layer, num_layers))
        if edge_enc == "MLP":
            self.edge_enc = MLP(embed_dim, num_heads, bias=False)
        else:
            self.edge_enc = nn.Linear(embed_dim, num_heads)
        self.num_heads = num_heads
        self.pe_encoder = pe_encoder
        self.ln = nn.LayerNorm(embed_dim)
        self.context_size = {
            "algo_reas_edge": [48, 192, 512, 1024, 2048, 4096, 8192],
            "algo_reas_mst": [16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384],
            "pcqm4mv2": [64, 96, 128],
            "pascal": [256, 512],
            "coco": [256, 512],
            "ogb-code2": [8, 16, 32, 64, 96, 128, 256, 512, 1024],
            "algo_reas_flow": [16, 32, 64, 128, 512, 1024],
        }
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim)))
        self.cls_loop = nn.Parameter(torch.randn((1, 1, 1, num_heads)))
        self.cls_in = nn.Parameter(torch.randn((1, 1, 1, num_heads)))
        self.cls_out = nn.Parameter(torch.randn((1, 1, 1, num_heads)))
        self.node_feature = nn.Parameter(torch.randn((1, embed_dim)))
        self.edge_feature = nn.Parameter(torch.randn((1, embed_dim)))
        self.transform = transform

    def embed_nodes(self, loc, data, task, task_modules, node, num_nodes):
        if task_modules[task]["node"] is not None and not (
            task == "zinc"
            or task == "pcqm4mv2"
            or task == "ogb-code2"
            or task == "algo_reas_flow"
        ):
            x = task_modules[task]["node"](data[loc].to(torch.bfloat16))
        elif task == "zinc" or task == "pcqm4mv2" or task == "algo_reas_flow":
            x = task_modules[task]["node"](data[loc])
        elif task == "ogb-code2":
            x = task_modules[task]["node"](data[loc], data.node_depth)
        else:
            if loc in data and data[loc] is not None:
                x = node.repeat(data[loc].size(0), 1)
            else:
                x = node.repeat(num_nodes, 1)
        return x

    def embed_edges(self, edge_encoder, loc, data, edge, num_edges):
        if edge_encoder is not None:
            if loc in data:
                if data[loc] is not None:
                    return edge_encoder(data[loc])
        return edge.repeat(num_edges, 1)

    def prepare_data(
        self, data, task, funcs, task_modules, node, edge, is_training, perturb=None
    ):
        data = funcs[task]["prepare"](data)
        if hasattr(data, "token_mask"):
            x = torch.zeros(
                (data.token_mask.size(0), node.size(1)),
                dtype=node.dtype,
                device=node.device,
            )
            if (num_node_tokens := torch.sum(~data.token_mask)) > 0:

                x[~data.token_mask] = self.embed_nodes(
                    "x_node", data, task, task_modules, node, num_node_tokens
                ).to(x.dtype)
            x[data.token_mask] = self.embed_edges(
                task_modules[task]["edge"],
                "x_edge",
                data,
                edge,
                data.num_nodes - num_node_tokens,
            )
            edge_attr = self.embed_edges(
                self.embedding_edge, "edge_attr", data, edge, data.edge_index.size(1)
            )
        else:
            x = self.embed_nodes("x", data, task, task_modules, node, data.num_nodes)
            edge_attr = self.embed_edges(
                task_modules[task]["edge"],
                "edge_attr",
                data,
                edge,
                data.edge_index.size(1),
            )

        ctx_size = self.context_size[task][-1]
        max_num_nodes = data.batch.unique(return_counts=True)[1].max().item()
        for size in self.context_size[task]:
            if max_num_nodes < size:
                ctx_size = size
                break

        encoded_x, encoded_e = self.pe_encoder(
            data,
            is_training=is_training,
            ctx_size=ctx_size,
            num_heads=self.num_heads,
            device=self.device,
        )
        e = self.edge_enc(edge_attr)
        if encoded_x is not None:
            x = x + encoded_x

        x, mask = to_dense_batch(
            x, data.batch, max_num_nodes=ctx_size - 1
        )  # B, (N - 1), D
        attn_mask = to_dense_adj(
            data.edge_index, data.batch, e, max_num_nodes=ctx_size - 1
        )  # B, (N - 1), (N - 1), H
        if encoded_e is not None and encoded_e.dim() == 4:
            attn_mask = attn_mask + encoded_e
        attn_mask[~mask] = float("-inf")
        attn_mask.permute(0, 2, 1, 3)[~mask] = float("-inf")

        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], 1)  # B, N, D
        attn_col = self.cls_out.repeat(
            x.size(0), 1, ctx_size - 1, 1
        )  # B, 1, (N - 1), H
        attn_row = torch.cat(
            [self.cls_loop, self.cls_in.repeat(1, ctx_size - 1, 1, 1)], 1
        )  # 1, N, 1, H
        attn_row = attn_row.repeat(x.size(0), 1, 1, 1)  # B, N, 1, H

        attn_mask = torch.cat([attn_col, attn_mask], 1)  # B, N, (N - 1), H
        attn_mask = torch.cat([attn_row, attn_mask], 2)  # B, N, N, H

        attn_mask = attn_mask.permute(0, 3, 1, 2).flatten(0, 1)

        return x, mask, attn_mask

    def forward(
        self,
        data,
        task,
        device=None,
        perturb=None,
        return_graph_embeddings=False,
        return_node_embeddings=False,
    ):

        x, mask, attn_mask = self.prepare_data(
            data,
            task,
            self.funcs,
            self.task_modules,
            self.node_feature.to(torch.bfloat16),
            self.edge_feature.to(torch.bfloat16),
            self.training,
            perturb=perturb,
        )

        x = self.ln(x)
        x = self.encoder(x, attn_mask)
        if return_graph_embeddings:
            return x[:, 0]
        if return_node_embeddings:
            if hasattr(data, "token_mask") and data.token_mask is not None:
                return x[:, 1:][mask][~data.token_mask]
            else:
                return x[:, 1:][mask]
        if hasattr(data, "token_mask") and task in ["algo_reas_flow"]:
            return self.task_modules[task]["decoder"](x[:, 0])
        if hasattr(data, "token_mask"):
            return self.task_modules[task]["decoder"](x[:, 1:][mask][data.token_mask])
        if task in ["pascal", "coco"]:
            return self.task_modules[task]["decoder"](x[:, 1:][mask])
        if task in ["ogb-code2"]:
            pred_list = []
            for i, l in enumerate(self.task_modules[task]["decoder"]):
                pred_list.append(
                    self.task_modules[task]["decoder"][i](x[:, 0]).squeeze(1)
                )

            return pred_list

        return self.task_modules[task]["decoder"](x[:, 0])

    def reset_parameters(self):
        self.apply(init_weights)
