# -*- coding: utf-8 -*-
# @Time: 2025/1/12
# @File: pos_embedding.py
# @Author: fwb
import torch


def sin_pos_embedding(positions, embedding_dim, n=10000.0):
    if embedding_dim % 2 != 0:
        raise ValueError(
            f"Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={embedding_dim})")
    max_len = positions.shape[0]
    d_model = embedding_dim
    positions = positions.unsqueeze(-1).expand(-1, d_model // 2)
    embeddings = torch.zeros((max_len, d_model), device=positions.device)
    denominators = torch.pow(n, 2 * torch.arange(0, d_model // 2) / d_model).to(positions.device)  # 10000^(2i/d_model)
    embeddings[:, 0::2] = torch.sin(positions / denominators)  # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions / denominators)  # cos(pos/10000^(2i/d_model))
    return embeddings
