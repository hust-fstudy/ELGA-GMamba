# -*- coding: utf-8 -*-
# @Time: 2025/3/24
# @File: local_feat_aggr.py
# @Author: fwb
from abc import ABC
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from layers.shared_mlp import SharedMLP


class LocalFeatAggr(MessagePassing, ABC):
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.mlp_encoder = SharedMLP([16, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False,
                                       act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.mlp_post_attention(out)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                index: Tensor) -> Tensor:
        # Encode local neighbourhood structural information.
        pos_diff = pos_j - pos_i
        pos_std = torch.std(pos_diff, dim=1, unbiased=False, keepdim=True)
        euclidean_distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        manhattan_distance = torch.abs(pos_diff).sum(1, keepdim=True)
        chebyshev_distance = torch.max(torch.abs(pos_diff), dim=1, keepdim=True).values
        min_distance = torch.min(torch.abs(pos_diff), dim=1, keepdim=True).values
        mean_distance = torch.mean(torch.abs(pos_diff), dim=1, keepdim=True)
        cos_distance = 1 - F.cosine_similarity(pos_i[:, None, :], pos_j[:, None, :], dim=2)
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, pos_std,
                                    euclidean_distance, manhattan_distance, chebyshev_distance,
                                    min_distance, mean_distance, cos_distance],
                                   dim=1)
        local_spatial_encoding = self.mlp_encoder(relative_infos)
        local_features = torch.cat([x_j, local_spatial_encoding],
                                   dim=1)
        # Attention will weight the different features of x.
        att_features = self.mlp_attention(local_features)
        att_scores = softmax(att_features, index=index)

        return att_scores * local_features
