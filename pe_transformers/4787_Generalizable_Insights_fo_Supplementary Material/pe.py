from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.transforms import Compose
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, get_laplacian, to_scipy_sparse_matrix
import torch_geometric
import numpy as np


class RWSE_encoder(torch.nn.Module):

    def __init__(self, rwse_steps, embed_dim, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.rwse_steps = rwse_steps
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(rwse_steps, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, **kwargs):
        pos_enc: torch.Tensor = self.encoder(
            data.rwse[:, : self.rwse_steps].to(self.device)
        )
        return pos_enc, None


def load_RWSE(embed_dim, device, rwse_steps, **kwargs):
    return RWSE_encoder(rwse_steps, embed_dim, device)


class RRWP_encoder(torch.nn.Module):
    def __init__(self, rrwp_steps, num_heads, embed_dim, device, **kwargs):
        super().__init__()
        self.edge_enc = nn.Linear(embed_dim, num_heads)
        self.rrwp_steps = rrwp_steps
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(rrwp_steps, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, ctx_size, **kwargs):
        data_adj = to_dense_adj(
            data.edge_index, data.batch, max_num_nodes=ctx_size - 1
        ).to(
            self.device
        )  # B x N x N
        deg_inv_sum = torch.diag_embed(data_adj.sum(dim=2)).to(self.device)
        deg_inv = torch.nan_to_num(1 / deg_inv_sum, posinf=0).to(
            self.device
        )  # B x N x N

        P = torch.bmm(deg_inv, data_adj)
        P_0 = P.clone()

        probs = torch.empty(
            (data_adj.size(0), data_adj.size(1), data_adj.size(1), self.rrwp_steps)
        ).to(self.device)

        for k in range(self.rrwp_steps):
            probs[:, :, :, k] = P  # B x N x N x k
            P = torch.bmm(P, P_0)

        return None, self.edge_enc(self.encoder(probs))


class RRWP_encoder_pretrans(torch.nn.Module):
    def __init__(self, rrwp_steps, num_heads, embed_dim, device, **kwargs):
        super().__init__()
        self.edge_enc = nn.Linear(embed_dim, num_heads)
        self.rrwp_steps = rrwp_steps
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(rrwp_steps, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, ctx_size, **kwargs):

        return None, self.edge_enc(self.encoder(data.rrwp))


def load_RRWP(embed_dim, device, rrwp_steps, num_heads, **kwargs):
    return RRWP_encoder(rrwp_steps, num_heads, embed_dim, device=device)


def load_RRWP_pretrans(embed_dim, device, rrwp_steps, num_heads, **kwargs):
    return RRWP_encoder_pretrans(rrwp_steps, num_heads, embed_dim, device)


class SPE_encoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        spe_num_layers_phi,
        spe_num_eigvals,
        spe_phi_dim,
        spe_inner_dim,
        spe_hidden_dim,
        spe_num_layers_rho,
        lower_rank,
        device,
    ):
        super().__init__()
        self.psi = EigenMLP(
            spe_num_layers_phi, spe_phi_dim, spe_inner_dim, dropout=0
        ).to(device)
        self.gin = GIN_SPE(
            input_dim=spe_inner_dim,
            hidden_dim=spe_hidden_dim,
            output_dim=embed_dim,
            num_layers=spe_num_layers_rho,
        ).to(device)
        self.device = device
        self.lower_rank = lower_rank
        self.num_eigvals = spe_num_eigvals

    def forward(self, data, **kwargs):
        eigvals = nan_to_zero(data.eigvals[:, : self.num_eigvals]).to(
            self.device
        )  # N_sum, M
        eigvecs = nan_to_zero(data.eigvecs[:, : self.num_eigvals]).to(
            self.device
        )  # N_sum, M

        eigvals, _ = to_dense_batch(eigvals, data.batch)  # B, N, M
        eigvals = eigvals[:, 0, :].unsqueeze(-1)  # B, M, 1

        eigvals = self.psi(eigvals)  # B, M, N_psi

        eigvecs, mask = to_dense_batch(eigvecs, data.batch)  # B, N, M

        if self.lower_rank is not None:
            outer_eigvecs = eigvecs[:, : self.lower_rank]
        else:
            outer_eigvecs = eigvecs

        eigen_tensor = torch.einsum(
            "bijk,blj->bilk",
            eigvecs.unsqueeze(-1) * eigvals.unsqueeze(1),  # B, N, M, N_psi
            outer_eigvecs,
        )  # B, N, LN, N_psi
        pe = eigen_tensor[mask]  # N_sum, LN, N_psi
        pe = self.gin(pe, data.edge_index).sum(1)
        return pe, None


def load_SPE(
    embed_dim,
    device,
    spe_num_eigvals,
    spe_hidden_dim,
    spe_inner_dim,
    spe_phi_dim,
    spe_num_layers_phi,
    spe_num_layers_rho,
    lower_rank,
    **kwargs
):
    return SPE_encoder(
        embed_dim,
        spe_num_layers_phi,
        spe_num_eigvals,
        spe_phi_dim,
        spe_inner_dim,
        spe_hidden_dim,
        spe_num_layers_rho,
        lower_rank,
        device,
    )


class LPE_encoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        lpe_num_eigvals,
        position_aware,
        lpe_bias,
        device,
        lpe_inner_dim,
    ):
        super().__init__()
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(2, embed_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim, bias=False),
        )
        if position_aware:
            self.eps = torch.nn.Parameter(
                1e-12 * torch.arange(lpe_inner_dim).unsqueeze(0)
            ).to(device)
        self.position_aware = position_aware
        self.lpe_num_eigvals = lpe_num_eigvals
        self.device = device

    def forward(self, data, is_training, **kwargs):
        eigvecs = data.eigvecs[:, : self.lpe_num_eigvals].to(self.device)
        eigvals = data.eigvals[:, : self.lpe_num_eigvals].to(self.device)

        if is_training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        if self.position_aware:
            eigvals = eigvals + self.eps[:, : self.lpe_num_eigvals]

        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x).to(self.device)
        x[empty_mask] = 0

        eigen_embed = self.phi(x)
        lpe_pe = self.rho(eigen_embed.sum(1))
        return lpe_pe, None


def load_LPE(
    embed_dim,
    device,
    lpe_num_eigvals,
    lpe_inner_dim=None,
    position_aware=False,
    lpe_bias=False,
    **kwargs
):
    return LPE_encoder(
        embed_dim, lpe_num_eigvals, position_aware, lpe_bias, device, lpe_inner_dim
    )


class WL_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, **kwargs):
        adj_mat = torch.zeros((data.num_nodes, data.num_nodes, data.num_heads))
        adj_mat[data.edge_index[0], data.edge_index[1], :] = 1
        return None, adj_mat


def load_1WL_PE(embed_dim, device, **kwargs):
    return WL_encoder()


class NoPE_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, **kwargs):
        return None, None


def load_no_PE(embed_dim, device, **kwargs):
    return NoPE_encoder()


def nan_to_zero(mat):
    empty_mask = torch.isnan(mat)
    mat[empty_mask] = 0
    return mat


class EigenMLPLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_proj = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.bn(x.reshape(-1, x.size(-1))).reshape(x.size())
        return self.dropout(F.relu(x))


class EigenMLP(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, out_dim, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                EigenMLPLayer(1 if i == 0 else embed_dim, embed_dim, dropout)
                for i in range(num_layers)
            ]
        )
        self.out_proj = torch.nn.Linear(embed_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return self.dropout(x)


class GIN_SPE(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch_geometric.nn.conv.GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            input_dim if _ == 0 else hidden_dim, hidden_dim
                        ),
                        torch.nn.ReLU(),
                    ),
                    node_dim=0,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.out_proj = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            z = layer(x, edge_index)
            if i > 0:
                x = z + x
            else:
                x = z
        return self.out_proj(x)


ENCODINGS = {
    "rwse": load_RWSE,
    "rrwp": load_RRWP,
    "spe": load_SPE,
    "lpe": load_LPE,
    "none": load_no_PE,
    "1wl": load_1WL_PE,
    "rrwp_pretrans": load_RRWP_pretrans,
}


def transforms(
    max_rw_steps, max_eigvals, normalized, large_graph, **kwargs
):
    def random_walk_transform(data, **kwargs):
        edge_index, num_nodes = data.edge_index, data.num_nodes

        edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)
        adj: torch.Tensor = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1

        deg_inv = torch.nan_to_num(1 / adj.sum(1), posinf=0)
        P = torch.diag(deg_inv) @ adj
        P_0 = torch.diag(deg_inv) @ adj
        probs = torch.empty((num_nodes, max_rw_steps))
        for k in range(max_rw_steps):
            probs[:, k] = torch.diag(P)
            P = P @ P_0
        data.rwse = probs
        return data

    def rrwp_transform(data, **kwargs):
        data_adj = to_dense_adj(
            data.edge_index, data.batch, max_num_nodes=1
        )  # B x N x N
        deg_inv_sum = torch.diag_embed(data_adj.sum(dim=2))
        deg_inv = torch.nan_to_num(1 / deg_inv_sum, posinf=0)  # B x N x N

        P = torch.bmm(deg_inv, data_adj)
        P_0 = P.clone()

        probs = torch.empty(
            (data_adj.size(0), data_adj.size(1), data_adj.size(1), max_rw_steps)
        )

        for k in range(max_rw_steps):
            probs[:, :, :, k] = P  # B x N x N x k
            P = torch.bmm(P, P_0)

        data.rrwp = probs
        return data

    def eigen_transform(data, **kwargs):

        A = torch.zeros((data.num_nodes, data.num_nodes))
        A[data.edge_index[0], data.edge_index[1]] = 1

        if normalized:
            D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
            I = torch.eye(A.size(0))
            L = I - D12 @ A @ D12
        else:
            D = torch.diag(A.sum(1))
            L = D - A

        if large_graph:

            L_edge_index, L_edge_weight = get_laplacian(
                data.edge_index, num_nodes=data.num_nodes, normalization="sym"
            )
            L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight)

            eigvals, eigvecs = np.linalg.eigh(L.todense())
            eigvals = torch.from_numpy(np.real(eigvals)).clamp_min(0)
            eigvecs = torch.from_numpy(eigvecs).float()
        else:
            eigvals, eigvecs = torch.linalg.eigh(L)
        eigvals_size = eigvals.size(0)
        if large_graph:
            idx1 = torch.argsort(eigvals)[: max_eigvals // 2]
            idx2 = torch.argsort(eigvals, descending=True)[: max_eigvals // 2]
            idx = torch.cat([idx1, idx2])
        else:
            idx = torch.argsort(eigvals)[:max_eigvals]

        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        eigvals, eigvecs = eigvals[:eigvals_size], eigvecs[:, :eigvals_size]

        eigvals = torch.real(eigvals).clamp_min(0)
        eigvals = eigvals.unsqueeze(0).repeat(data.num_nodes, 1)
        if data.num_nodes < max_eigvals:
            eigvals = F.pad(
                eigvals, (0, max_eigvals - data.num_nodes), value=float("nan")
            )
            eigvecs = F.pad(
                eigvecs, (0, max_eigvals - data.num_nodes), value=float("nan")
            )

        data.eigvals = eigvals
        data.eigvecs = eigvecs

        return data

    return random_walk_transform, rrwp_transform, eigen_transform


def load_transform(args):
    transform_list = Compose(transforms(**args))
    return transform_list


def load_pe(encoding, args, device, embed_dim, num_heads):
    pe = {k: v for k, v in ENCODINGS.items() if k in encoding}
    for encoding, _load in pe.items():
        encoder = _load(**args, device=device, num_heads=num_heads, embed_dim=embed_dim)

    return encoder
