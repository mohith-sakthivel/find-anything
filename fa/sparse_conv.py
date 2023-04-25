"""
Code adapted from https://github.com/VinAIResearch/GeoFormer
"""
import functools
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, xyz):
        xyz1 = xyz.unsqueeze(1)  # N * 1 * 3
        xyz2 = xyz.unsqueeze(0)  # 1 * N * 3
        pairwise_dist = xyz1 - xyz2  # N * N * 3

        return pairwise_dist
    

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=64, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.N = N
        self.pe = PositionalEncoder(d_model)
        
        self.layers = nn.ModuleList([deepcopy(EncoderLayer(d_model, heads, d_ff=d_ff)) for _ in range(N)])
        self.norm = Norm(d_model)
        self.position_linear = nn.Linear(3, d_model)

    def forward(self, xyz, features, batch_ids):
        batch_size = batch_ids.max().item() + 1
        assert features.size(1) == self.d_model
        output = torch.zeros_like(features)
        for i in range(batch_size):
            batch_id = torch.nonzero((batch_ids == i)).squeeze(dim=1)
            if batch_id.size(0) == 0:
                continue
            start_id = int(batch_id.min().item())
            end_id = int(batch_id.max().item()) + 1
            batch_xyz = xyz[start_id:end_id].view(-1, 3)  # n' * 3
            batch_features = features[start_id:end_id].view(-1, self.d_model)  # n' * c

            pairwise_dist = self.pe(batch_xyz).float()  # n' * n' * 3
            pairwise_dist = pairwise_dist.mean(dim=1)
            position_embedding = self.position_linear(pairwise_dist)

            x = (batch_features + position_embedding).unsqueeze(dim=0)
            for i in range(self.N):
                x = self.layers[i](x, mask=None)

            x = self.norm(x)
            output[batch_id] = x.squeeze(dim=0)
        return output
    

class ResidualBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, use_backbone_transformer=False, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        blocks = {
            "block{}".format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key="subm{}".format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)
        if len(nPlanes) <= 2 and use_backbone_transformer:
            d_model = 128
            self.before_transformer_linear = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=4, d_ff=64)
            self.after_transformer_linear = nn.Linear(d_model, nPlanes[0])
        else:
            self.before_transformer_linear = None
            self.transformer = None
            self.after_transformer_linear = None
        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                nn.BatchNorm1d(nPlanes[0], eps=1e-4, momentum=0.1),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, use_backbone_transformer, indice_key_id=indice_key_id + 1
            )


class SPConvGeoFormer(nn.Module):

    def __init__(self, input_dim: int, m: int = 16)->None:

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_dim, m, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )
        self.unet = UBlock(
            [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
            functools.partial(nn.BatchNorm1d, eps=1e-5),
            2,
            ResidualBlock,
            use_backbone_transformer=True,
            indice_key_id=1,
        )
        self.output_layer = spconv.SparseSequential(
            nn.BatchNorm1d(m, eps=1e-5),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.input_conv(sparse_input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[p2v_map.long()]
        output_feats = output_feats.contiguous()
