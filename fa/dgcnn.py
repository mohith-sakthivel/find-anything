#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code adapted from https://github.com/antao97/dgcnn.pytorch
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from typing import Optional


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None, ignore_geometry_data: bool = False) -> torch.Tensor:
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if ignore_geometry_data == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (batch_size, 2 * num_dims, num_points, k)

    return feature


def build_conv_block_2d(in_dim: int, out_dim: int) -> nn.Module:
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

    return block


def build_conv_block_1d(in_dim: int, out_dim: int) -> nn.Module:
    block = nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

    return block


class DGCNNSeg(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        k: int = 20,
        embed_dim: int = 1024,
        dropout: float = 0.5,
        num_classes: Optional[int] = None,
        feat_dim: int = 256,
    ) -> None:
        super().__init__()

        self.k = k
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.conv1 = build_conv_block_2d(2 * pc_dim, 64)
        self.conv2 = build_conv_block_2d(64, 64)

        self.conv3 = build_conv_block_2d(64 * 2, 64)
        self.conv4 = build_conv_block_2d(64, 64)

        self.conv5 = build_conv_block_2d(64 * 2, 64)

        self.conv6 = build_conv_block_1d(192, embed_dim)

        self.conv7 = build_conv_block_1d(64 * 3 + embed_dim, 512)

        self.conv8 = build_conv_block_1d(512, 256)

        if num_classes is not None:
            self.dp = nn.Dropout(p=dropout)
            self.conv9 = nn.Conv1d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_pts, _ = x.size()
        x = x.transpose(-1, -2)

        x = get_graph_feature(x, self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.amax(dim=-1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.amax(dim=-1)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.amax(dim=-1)

        x = torch.cat([x1, x2, x3], dim=-2)
        x = self.conv6(x)
        x = x.amax(dim=-1, keepdim=True)

        x = x.repeat(1, 1, n_pts)
        x = torch.cat([x, x1, x2, x3], dim=-2)

        x = self.conv7(x)
        x = self.conv8(x)
        if self.num_classes is not None:
            x = self.dp(x)
            x = self.conv9(x)

        return x
