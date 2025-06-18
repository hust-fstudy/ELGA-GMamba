# -*- coding: utf-8 -*-
# @Time: 2025/3/24
# @File: shared_mlp.py
# @Author: fwb
from torch_geometric.nn import MLP

# Default activation and batch norm parameters.
l_relu02_kwargs = {'negative_slope': 0.2}
bn099_kwargs = {'momentum': 0.01, 'eps': 1e-6}


class SharedMLP(MLP):
    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs['plain_last'] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs['act'] = kwargs.get('act', 'LeakyReLU')
        kwargs['act_kwargs'] = kwargs.get('act_kwargs', l_relu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum and 1e-6 eps by default.
        kwargs['norm_kwargs'] = kwargs.get('norm_kwargs', bn099_kwargs)
        super().__init__(*args, **kwargs)
