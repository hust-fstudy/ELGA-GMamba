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

