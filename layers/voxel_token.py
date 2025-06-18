# -*- coding: utf-8 -*-
# @Time: 2025/1/13
# @File: voxel_token.py
# @Author: fwb
import torch.nn as nn
from torch_geometric.nn import MLP


class VoxelToken(nn.Module):
    def __init__(self,
                 in_feats=3,
                 embed_dim=32
                 ):
        super(VoxelToken, self).__init__()
        self.mlp = MLP(
            channel_list=[in_feats, embed_dim],
            dropout=0.5,
            act="elu",
            norm="batch_norm"
        )

    def forward(self, data):
        """
        :param data: data.x shape (B*Nv) x feats_dim x vh x vw
        :return: (B*Nv) x embed_dim
        """
        data.x = data.x.flatten(1)  # (B*Nv) x (feats_dim*vh*vw)
        data.x = self.mlp(data.x, batch=data.batch)
        return data
