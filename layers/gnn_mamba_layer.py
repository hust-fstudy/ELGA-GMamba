# -*- coding: utf-8 -*-
# @Time: 2025/1/10
# @File: gnn_mamba.py
# @Author: fwb
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, GELU
from torch_geometric.nn import BatchNorm, LayerNorm, GCNConv, GATConv, SAGEConv, GMMConv, SplineConv
from torch_geometric.nn.inits import reset
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba


class GNNMambaLayer(nn.Module):
    def __init__(self,
                 args,
                 embed_chs: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 1,
                 **kwargs):
        super(GNNMambaLayer, self).__init__()
        self.args = args
        self.mlp_dropout = args.mlp_dropout
        self.gnn_dropout = args.gnn_dropout
        match args.sel_gnn:
            case 'GCN':
                self.conv = GCNConv(in_channels=embed_chs, out_channels=embed_chs)
            case 'GAT':
                self.conv = GATConv(in_channels=embed_chs, out_channels=embed_chs, concat=False, **kwargs)
            case 'SAGE':
                self.conv = SAGEConv(in_channels=embed_chs, out_channels=embed_chs)
            case 'GMM':
                self.conv = GMMConv(in_channels=embed_chs, out_channels=embed_chs, **kwargs)
            case 'Spline':
                self.conv = SplineConv(in_channels=embed_chs, out_channels=embed_chs, **kwargs)
            case _:
                print("The GNN does not exist!")
        self.attn = Mamba(d_model=embed_chs, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mlp = Sequential(
            Linear(embed_chs, embed_chs * 2),
            GELU(),
            Dropout(self.mlp_dropout),
            Linear(embed_chs * 2, embed_chs),
            Dropout(self.mlp_dropout),
        )
        match args.norm:
            case 'layer_norm':
                self.norm1 = LayerNorm(embed_chs)
                self.norm2 = LayerNorm(embed_chs)
                self.norm3 = LayerNorm(embed_chs)
            case _:
                self.norm1 = BatchNorm(embed_chs)
                self.norm2 = BatchNorm(embed_chs)
                self.norm3 = BatchNorm(embed_chs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        # Resets all learnable parameters of the module.
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(self, data):
        hs = []
        # Local info.
        if self.conv is not None:
            if self.args.sel_gnn in ['GCN', 'SAGE']:
                h = self.conv(x=data.x, edge_index=data.edge_index)
            elif self.args.sel_gnn in ['GAT', 'GMM', 'Spline']:
                h = self.conv(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            else:
                h = None
                print("The GNN layer does not exist!")
            h = F.dropout(h, p=self.gnn_dropout, training=self.training)
            h = h + data.x
            if self.norm1 is not None and self.args.num_norm in {1, 2, 3}:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=data.batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
        # Global attention.
        h, mask = to_dense_batch(data.x, data.batch)  # h->B x L x D
        if self.args.is_bidirectional:
            inv_x = torch.flip(h, [1])
            x_attention = self.attn(h)
            inv_x_attention = self.attn(inv_x)
            h = torch.add(x_attention, torch.flip(inv_x_attention, [1]))[mask]
        else:
            h = self.attn(h)[mask]
        h = F.dropout(h, p=self.mlp_dropout, training=self.training)
        h = h + data.x  # Residual connection.
        if self.norm2 is not None and self.args.num_norm in {2, 3}:
            if self.norm_with_batch:
                h = self.norm2(h, batch=data.batch)
            else:
                h = self.norm2(h)
        hs.append(h)
        data.x = sum(hs)  # combine local and global outputs
        data.x = data.x + self.mlp(data.x)
        if self.norm3 is not None and self.args.num_norm in {3}:
            if self.norm_with_batch:
                data.x = self.norm3(data.x, batch=data.batch)
            else:
                data.x = self.norm3(data.x)
        return data
