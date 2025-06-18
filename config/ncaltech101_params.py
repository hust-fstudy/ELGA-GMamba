# -*- coding: utf-8 -*-
# @Time: 2025/3/4
# @File: ncaltech101_params.py
# @Author: fwb
import argparse


def ncaltech101_params():
    parser = argparse.ArgumentParser(description="Parameter Configuration of NCaltech101 Dataset.")
    # Split.
    parser.add_argument('--is_split', default=True, type=bool)
    parser.add_argument('--is_remove_split', default=True, type=bool)
    # To voxel.
    parser.add_argument('--is_filter', default=True, type=bool)
    parser.add_argument('--coord_range', default=[240, 180, 6], type=list)
    parser.add_argument('--grid_size', default=[8, 6, 1], type=list)
    parser.add_argument('--max_num_voxels', default=10000, type=int)
    parser.add_argument('--max_num_points_per_voxel', default=3000, type=int)
    parser.add_argument('--Nv', default=4096, type=int)
    parser.add_argument('--feats_dim', default=3, type=int)
    # To graph.
    parser.add_argument('--is_remove_graph', default=True, type=bool)
    parser.add_argument('--transform', default=True, type=bool)
    # Initialization model.
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--is_reg_loss', default=False, type=bool)
    parser.add_argument('--opt_name', default='sgd', type=str)
    parser.add_argument('--criterion_name', default='smooth', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=5e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lambda_reg', default=1e-5, type=float)
    # Lr scheduler.
    parser.add_argument('--is_lr_scheduler', default=True, type=bool)
    parser.add_argument('--lr_scheduler', default='cycle', type=str)
    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--decay_epochs', default=30, type=int)
    parser.add_argument('--min_lr', default=5e-6, type=float)
    parser.add_argument('--warmup_lr_init', default=5e-7, type=float)
    parser.add_argument('--decay_rate', default=0.1, type=float)
    # Learnable feature map.
    parser.add_argument('--is_feats_map', default=True, type=bool)
    parser.add_argument('--embed_dim', default=32, type=int)
    # Graph position embedding.
    parser.add_argument('--is_node_pe', default=True, type=bool)
    parser.add_argument('--is_edge_pe', default=True, type=bool)
    # Graph encoder.
    parser.add_argument('--enc_depth', default=(2, 2, 3, 2), type=tuple)
    parser.add_argument('--enc_chs', default=(64, 128, 256, 512), type=tuple)
    parser.add_argument('--enc_neighbours', default=(16, 16, 16, 16), type=tuple)
    parser.add_argument('--enc_radius', default=(1.25, 2.5, 5, 10), type=tuple)
    parser.add_argument('--enc_way', default=('r', 'r', 'n', 'n'), type=tuple)
    parser.add_argument('--pool_ratio', default=(0.5, 0.25, 0.25, 0.25), type=tuple)
    # GNN mamba.
    parser.add_argument('--is_gnn_mamba', default=True, type=bool)
    parser.add_argument('--is_bidirectional', default=True, type=bool)
    parser.add_argument('--sel_gnn', default='GAT', type=str)
    parser.add_argument('--norm', default='batch_norm', type=str)
    parser.add_argument('--num_norm', default=3, type=int)
    parser.add_argument('--gnn_dropout', default=0.3, type=float)
    parser.add_argument('--mlp_dropout', default=0.5, type=float)
    # Classify head.
    parser.add_argument('--num_classes', default=101, type=int)
    # Parsing command lines.
    args = parser.parse_args()
    return args
