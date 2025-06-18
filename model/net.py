# -*- coding: utf-8 -*-
# @Time: 2024/12/11
# @File: ssm_net.py
# @Author: fwb
import torch
import torch.nn as nn
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.pool import knn_graph, radius_graph
from layers.voxel_token import VoxelToken
from layers.shared_mlp import SharedMLP
from layers.local_feat_aggr import LocalFeatAggr
from layers.graph_pool import GraphPool
from layers.pos_embedding import sin_pos_embedding
from layers.gnn_mamba_layer import GNNMambaLayer

# Default activation and batch norm parameters.
l_relu02_kwargs = {'negative_slope': 0.2}
bn099_kwargs = {'momentum': 0.01, 'eps': 1e-6}


class BlockSequence(torch.nn.Module):
    def __init__(self,
                 depth,
                 in_chs,
                 embed_chs,
                 num_neighbors,
                 radius,
                 way: str
                 ):
        super(BlockSequence, self).__init__()
        self.num_neighbors = num_neighbors
        self.radius = radius
        self.way = way

        self.mlp_in = SharedMLP([in_chs, embed_chs // 2**(depth+1)])
        self.shortcut = SharedMLP([in_chs, embed_chs], act=None)
        self.mlp_out = SharedMLP([embed_chs // 2, embed_chs], act=None)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = LocalFeatAggr(embed_chs // 2**(depth-i))
            self.blocks.append(block)

        self.l_relu = torch.nn.LeakyReLU(**l_relu02_kwargs)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        if self.way.lower() == 'n':
            edge_index = knn_graph(x=pos, k=self.num_neighbors, batch=batch, loop=True)
        else:
            edge_index = radius_graph(x=pos, r=self.radius, batch=batch, loop=True, max_num_neighbors=64)
        shortcut_of_x = self.shortcut(x)
        x = self.mlp_in(x)
        for block in self.blocks:
            x = block(edge_index, x, pos)
        x = self.mlp_out(x)
        x = self.l_relu(x + shortcut_of_x)
        data.x, data.pos, data.batch = x, pos, batch
        return data


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_chs,
                 embed_chs,
                 num_neighbors,
                 radius,
                 way: str,
                 pool_ratio
                 ):
        super(Encoder, self).__init__()
        self.blocks = BlockSequence(
            depth=depth,
            in_chs=in_chs,
            embed_chs=embed_chs,
            num_neighbors=num_neighbors,
            radius=radius,
            way=way
        )
        self.pool = GraphPool(pool_ratio=pool_ratio)

    def forward(self, data):
        data = self.blocks(data)
        return self.pool(data)


class Net(torch.nn.Module):
    def __init__(self,
                 args,
                 feats_dim,
                 in_feats,
                 num_classes,
                 embed_dim=32,
                 enc_depth=(2, 2, 2, 2),
                 enc_chs=(64, 128, 256, 512),
                 enc_neighbours=(16, 16, 16, 16),
                 enc_radius=(1, 2, 3, 4),
                 enc_way=('n', 'n', 'n', 'n'),
                 pool_ratio=(0.5, 0.5, 0.25, 0.25),
                 is_feats_map=True,
                 is_gnn_mamba=True,
                 ):
        super().__init__()
        self.args = args
        self.feats_dim = feats_dim
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.enc_neighbours = enc_neighbours
        self.enc_radius = enc_radius
        self.enc_way = enc_way
        self.is_feats_map = is_feats_map
        self.num_stages = len(enc_depth)
        assert self.num_stages == len(enc_chs)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(enc_radius)
        assert self.num_stages == len(enc_way)
        assert self.num_stages == len(pool_ratio)
        # Learnable feature map.
        if is_feats_map:
            self.feats_map = VoxelToken(
                in_feats=in_feats,
                embed_dim=embed_dim
            )
        else:
            self.feats_map = Linear(in_feats, embed_dim)
        # Encoder.
        enc_chs = [embed_dim] + list(enc_chs)
        self.enc_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depth[i],
                in_chs=enc_chs[i],
                embed_chs=enc_chs[i + 1],
                num_neighbors=enc_neighbours[i],
                radius=enc_radius[i],
                way=enc_way[i],
                pool_ratio=pool_ratio[i]
            )
            self.enc_stages.append(enc)
        # Graph position embedding.
        edge_attr_dim = 3
        if args.is_node_pe:
            self.node_pe_dim = enc_chs[-1]
        if args.is_edge_pe:
            self.edge_learn_param = nn.Parameter(torch.ones(args.batch_size * args.Nv * enc_neighbours[-1],
                                                            edge_attr_dim))
            nn.init.trunc_normal_(self.edge_learn_param, std=0.02)
        # Combine local and global outputs.
        self.gnn_mamba = None
        if is_gnn_mamba:
            if args.sel_gnn in ['GCN', 'SAGE']:
                self.gnn_mamba = GNNMambaLayer(
                    args=args,
                    embed_chs=enc_chs[-1]
                )
            elif args.sel_gnn in ['GAT']:
                self.gnn_mamba = GNNMambaLayer(
                    args=args,
                    embed_chs=enc_chs[-1],
                    heads=2
                )
            elif args.sel_gnn in ['GMM', 'Spline']:
                self.gnn_mamba = GNNMambaLayer(
                    args=args,
                    embed_chs=enc_chs[-1],
                    dim=3,
                    kernel_size=3
                )
            else:
                self.gnn_mamba = None
        # Classify head.
        self.mlp = SharedMLP([enc_chs[-1], enc_chs[-1] // 2])
        self.max_aggr = MaxAggregation()
        self.mlp_cls = SharedMLP([enc_chs[-1] // 2, enc_chs[-1] // 4], dropout=[0.5])
        self.fc = Linear(enc_chs[-1] // 4, num_classes)

    def forward(self, data):
        # Learnable feature map.
        data.x = data.x[:, :self.feats_dim, :, :]
        if self.is_feats_map:
            data = self.feats_map(data)
        else:
            data.x = self.feats_map(data.x.flatten(1))
        # Encoder.
        for i in range(self.num_stages):
            data = self.enc_stages[i](data)
        # Graph position embedding.
        if self.enc_way[-1].lower() == 'n':
            data.edge_index = knn_graph(x=data.pos, k=self.enc_neighbours[-1], batch=data.batch, loop=True)
        else:
            data.edge_index = radius_graph(x=data.pos, r=self.enc_radius[-1], batch=data.batch, loop=True,
                                           max_num_neighbors=64)
        transform = T.Cartesian(cat=False)
        data = transform(data)
        if self.args.is_node_pe:
            node_pos_embedding = (sin_pos_embedding(data.pos[:, 0], self.node_pe_dim) +
                                  sin_pos_embedding(data.pos[:, 1], self.node_pe_dim) +
                                  sin_pos_embedding(data.pos[:, 2], self.node_pe_dim))
            data.x = data.x + node_pos_embedding[:, :data.x.size(1)]
        if self.args.is_edge_pe:
            data.edge_attr = data.edge_attr + self.edge_learn_param[:data.edge_attr.size(0), :]
        # Combine local and global outputs.
        if self.gnn_mamba is not None:
            data = self.gnn_mamba(data)
        # Classify head.
        x = self.mlp(data.x)
        x = self.max_aggr(x, data.batch)
        x = self.mlp_cls(x)
        out = self.fc(x)

        return out
