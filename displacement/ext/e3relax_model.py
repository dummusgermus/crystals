# Vendored from the authors' public repository; edits marked '# [disp-study]'.
# %%
import math
import torch
from torch import nn
from torch_geometric.utils import scatter  # [disp-study] torch_scatter not available
from .graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis
from .graph_utils import vector_norm
from torch_geometric.nn import global_mean_pool

class LatticeNode(nn.Module):
    def __init__(self, in_feats, out_feats, num_rbf):
        super(LatticeNode, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.mlp_scalar_global = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())
        self.mlp_scalar_local = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU())

        self.mlp_vector_global = nn.Linear(in_feats, out_feats, bias=False)
        self.mlp_vector_local = nn.Linear(in_feats, out_feats, bias=False)

        self.x_local_proj = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            ScaledSiLU(),
            nn.Linear(in_feats, in_feats*3),
        )
        self.edge_local_proj = nn.Linear(num_rbf, in_feats*3)

        self.vector_l_proj = nn.Linear(
            out_feats, out_feats * 2, bias=False
        )
        self.scalar_l_proj = nn.Sequential(
            nn.Linear(out_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats * 3),
        )

        self.mlp_l = nn.Linear(out_feats, 1, bias=False)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(in_feats)
 
    def update_local_emb(self, x, scalar_l, vec, vector_l, edge_feat, edge_udiff, batch):

        x_p = self.x_local_proj(x)
        edge_p = self.edge_local_proj(edge_feat)
        x_1, x_2, x_3 = torch.split(x_p * edge_p * self.inv_sqrt_3, self.in_feats, dim=-1)

        x = x_3 + x
        vec = x_1.unsqueeze(1) * vec + x_2.unsqueeze(1) * edge_udiff.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        hx = self.mlp_scalar_local(torch.cat([x, scalar_l[batch]], dim=1)) + x
        hvec = self.mlp_vector_local(vec + vector_l[batch]) + vec

        return hx, scalar_l, hvec, vector_l
    
    def update_global_emb(self, x, scalar_l, vec, vector_l, batch):

        scalar_l_temp = self.mlp_scalar_global(torch.cat([global_mean_pool(x, batch), scalar_l], dim=-1)) 
        vector_l_temp = self.mlp_vector_global(scatter(vec, batch, dim=0, reduce='mean', dim_size=vector_l.size(0)) + vector_l) 
        scalar_l = scalar_l + scalar_l_temp
        vector_l = vector_l + vector_l_temp

        vector_l1_h1, vector_l1_h2 = torch.split(
            self.vector_l_proj(vector_l), self.out_feats, dim=-1
        )
        scalar_l_h = self.scalar_l_proj(
            torch.cat(
                [scalar_l, torch.sqrt(torch.sum(vector_l1_h2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        scalar_l_h1, scalar_l_h2, scalar_l_h3 = torch.split(
            scalar_l_h, self.out_feats, dim=-1
        )
        gate = torch.tanh(scalar_l_h3)
        scalar_l = scalar_l_h2  + scalar_l * gate
        vector_l = scalar_l_h1.unsqueeze(1) * vector_l1_h1 + vector_l

        return scalar_l, vector_l

    def update_lattice_vector(self, vector_l):

        l_delta = self.mlp_l(vector_l)

        return l_delta

class E3Relax(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83
    ):
        super(E3Relax, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        
        self.scalar_l1_emb = nn.Embedding(1, hidden_channels)   # Embedding for lattice node's scalar (dimension l1)
        self.scalar_l2_emb = nn.Embedding(1, hidden_channels)   # Embedding for lattice node's scalar (dimension l2)
        self.scalar_l3_emb = nn.Embedding(1, hidden_channels)   # Embedding for lattice node's scalar (dimension l3)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_passing_layers = nn.ModuleList()
        self.message_update_layers = nn.ModuleList()
        self.structure_update_layers = nn.ModuleList()

        self.lattice_l1_layers = nn.ModuleList()
        self.lattice_l2_layers = nn.ModuleList()
        self.lattice_l3_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassing(hidden_channels, num_rbf)) # Layer for message passing between atoms
            self.message_update_layers.append(SelfUpdating(hidden_channels)) # Layer for self-updating 
            self.structure_update_layers.append(StructureUpdating(hidden_channels)) # Layer for updating atomic structure

            self.lattice_l1_layers.append(LatticeNode(hidden_channels, hidden_channels, num_rbf))  # Layer for updating lattice in dimension l1
            self.lattice_l2_layers.append(LatticeNode(hidden_channels, hidden_channels, num_rbf))  # Layer for updating lattice in dimension l2
            self.lattice_l3_layers.append(LatticeNode(hidden_channels, hidden_channels, num_rbf))  # Layer for updating lattice in dimension l3

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):

        pos = data.pos_u
        cell = data.cell_u
        
        cell_offsets = data.cell_offsets.float()
        edge_index = data.edge_index

        neighbors = data.neighbors
        batch = data.batch
        z = data.x.long()
        assert z.dim() == 1 and z.dtype == torch.long

        x = self.atom_emb(z)
        if getattr(self, 'feat_emb', None) is not None:  # [disp-study] extra node features
            x = x + self.feat_emb(data.xf)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        if getattr(self, 'f_init', None) is not None:  # [disp-study] initial force as start vector
            vec = vec + self.f_init(data.force0.unsqueeze(-1))
 
        scalar_l1 = self.scalar_l1_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)) # Initial lattice node's scalar
        scalar_l2 = self.scalar_l2_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)) # Initial lattice node's scalar
        scalar_l3 = self.scalar_l3_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)) # Initial lattice node's scalar

        vector_l1 = torch.zeros(
                batch[-1].item() + 1, 3, x.size(1)).to(edge_index.dtype).to(edge_index.device) # Initial lattice node's vector
        vector_l2 = torch.zeros(
                batch[-1].item() + 1, 3, x.size(1)).to(edge_index.dtype).to(edge_index.device) # Initial lattice node's vector
        vector_l3 = torch.zeros(
                batch[-1].item() + 1, 3, x.size(1)).to(edge_index.dtype).to(edge_index.device) # Initial lattice node's vector
        
        #### Interaction blocks ###############################################
        pos_list = []
        cell_list = []

        for n in range(self.num_layers):
            # Atom coordinates and lattice parameters are updated after each graph convolution
            # Therefore, relative positions and interatomic distances must be recalculated
            j, i = edge_index
            abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0) 
            pos_diff = pos[j] + torch.einsum("a p, a p v -> a v", cell_offsets, abc_unsqueeze) - pos[i]
            edge_dist = vector_norm(pos_diff, dim=-1)
            edge_udiff = -pos_diff / edge_dist.unsqueeze(-1)
            edge_feat = self.radial_basis(edge_dist)  

            # Calculate the relative position and distance between each atom and lattice nodes
            vector_l1_diff = cell[:, 0, :][batch] - pos   
            vector_l2_diff = cell[:, 1, :][batch] - pos
            vector_l3_diff = cell[:, 2, :][batch] - pos
            vector_l1_dist = vector_norm(vector_l1_diff, dim=-1)
            vector_l2_dist = vector_norm(vector_l2_diff, dim=-1)
            vector_l3_dist = vector_norm(vector_l3_diff, dim=-1)
            vector_l1_udiff = -vector_l1_diff / vector_l1_dist.unsqueeze(-1)
            vector_l2_udiff = -vector_l2_diff / vector_l2_dist.unsqueeze(-1)
            vector_l3_udiff = -vector_l3_diff / vector_l3_dist.unsqueeze(-1)
            vector_l1_feat = self.radial_basis(vector_l1_dist)
            vector_l2_feat = self.radial_basis(vector_l2_dist) 
            vector_l3_feat = self.radial_basis(vector_l3_dist)

            # Atom collects messages from the lattice nodes
            x, scalar_l1, vec, vector_l1 = self.lattice_l1_layers[n].update_local_emb(x, scalar_l1, vec, vector_l1, vector_l1_feat, vector_l1_udiff, batch)
            x, scalar_l2, vec, vector_l2 = self.lattice_l2_layers[n].update_local_emb(x, scalar_l2, vec, vector_l2, vector_l2_feat, vector_l2_udiff, batch)
            x, scalar_l3, vec, vector_l3 = self.lattice_l3_layers[n].update_local_emb(x, scalar_l3, vec, vector_l3, vector_l3_feat, vector_l3_udiff, batch)

            # Message passing
            dx, dvec = self.message_passing_layers[n](
                x, vec, edge_index, edge_feat, edge_udiff
            )
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            # Self-updating
            dx, dvec = self.message_update_layers[n](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            # Structure updating
            pos_delta = self.structure_update_layers[n](x, vec)
            pos = pos + pos_delta

            # Lattice nodes collect messages from all atoms
            scalar_l1, vector_l1 = self.lattice_l1_layers[n].update_global_emb(x, scalar_l1, vec, vector_l1, batch)
            scalar_l2, vector_l2 = self.lattice_l2_layers[n].update_global_emb(x, scalar_l2, vec, vector_l2, batch)
            scalar_l3, vector_l3 = self.lattice_l3_layers[n].update_global_emb(x, scalar_l3, vec, vector_l3, batch)

            # Convert lattice node's vector representation into lattice delta
            vector_l1_delta = self.lattice_l1_layers[n].update_lattice_vector(vector_l1)
            vector_l2_delta = self.lattice_l2_layers[n].update_lattice_vector(vector_l2)
            vector_l3_delta = self.lattice_l3_layers[n].update_lattice_vector(vector_l3)

            cell_delta = torch.cat([vector_l1_delta, vector_l2_delta, vector_l3_delta], dim=-1)
            # [disp-study] fix: the deltas are columns here but the cell stores lattice vectors as rows
            cell_delta = cell_delta.transpose(1, 2)
            cell = cell + cell_delta

            pos_list.append(pos)
            cell_list.append(cell)

        return pos_list, cell_list

class StructureUpdating(nn.Module):
    def __init__(
        self,
        hidden_channels
    ):
        super(StructureUpdating, self).__init__()

        self.attention_pos = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.pos_mlp = nn.Linear(hidden_channels, 1, bias=False)


    def forward(self, x, vec):
        # pos attention
        a_pos = torch.softmax(self.attention_pos(x), dim=-1).unsqueeze(1)

        # update coordinate
        pos_delta = (self.pos_mlp(vec).squeeze(-1) + (a_pos * vec).sum(-1)) / 2

        return pos_delta

class MessagePassing(nn.Module):
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
    ):
        super(MessagePassing, self).__init__()

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels*3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels*3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec, edge_index, edge_rbf, edge_udiff):
        j, i = edge_index

        rbf_h = self.edge_proj(edge_rbf)

        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(x_h[j] * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1)

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_udiff.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0)) 
        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0))
        
        return d_x, d_vec
    
class SelfUpdating(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, x, vec):

        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        gate = torch.tanh(xvec3)
        dx = xvec2 * self.inv_sqrt_2 + x * gate

        dvec = xvec1.unsqueeze(1) * vec1

        return dx, dvec

# %%