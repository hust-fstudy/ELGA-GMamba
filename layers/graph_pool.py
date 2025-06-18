# -*- coding: utf-8 -*-
# @Time: 2025/3/24
# @File: graph_pool.py
# @Author: fwb
import torch.nn as nn
from torch_geometric.nn import fps


class GraphPool(nn.Module):
    def __init__(self,
                 pool_ratio
                 ):
        super(GraphPool, self).__init__()
        self.pool_ratio = pool_ratio

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        idx = fps(pos, batch, ratio=self.pool_ratio)
        data.x, data.pos, data.batch = x[idx], pos[idx], batch[idx]
        return data
