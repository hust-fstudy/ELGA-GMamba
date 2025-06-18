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

