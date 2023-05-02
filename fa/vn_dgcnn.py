"""
Code adapted from https://github.com/FlyingGiraffe/vnn-pc/blob/master/vn-dgcnn/model_equi.py
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

EPS = 1e-6


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:          # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0

def build_conv_block_1d(in_dim: int, out_dim: int) -> nn.Module:
    block = nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

    return block


class VN_DGCNNSeg(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        k: int = 20,
        embed_dim: int = 512,
        dropout: float = 0.5,
        num_classes: Optional[int] = None,
        feat_dim: int = 256,
        pooling: str = 'max',
    ) -> None:
        super().__init__()

        self.k = k
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.conv1 = VNLinearLeakyReLU(2*pc_dim//3, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)

        self.conv3 = VNLinearLeakyReLU(64//3 * 2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)

        self.conv5 = VNLinearLeakyReLU(64//3 * 2, 64//3)

        self.conv6 = VNLinearLeakyReLU(64//3, embed_dim//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(embed_dim//3*2, dim=4, normalize_frame=False)

        self.conv7 = build_conv_block_1d(2109, 512)

        self.conv8 = build_conv_block_1d(512, feat_dim)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        if num_classes is not None:
            self.dp = nn.Dropout(p=dropout)
            self.conv9 = nn.Conv1d(feat_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_pts, _ = x.size()

        x = x.transpose(-1, -2)

        x = x.unsqueeze(1)

        x = get_graph_feature(x, self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat([x1, x2, x3], dim=-2)
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(b, -1, n_pts)
        x = x.view(b, -1, n_pts)
        x = x.amax(dim=-1, keepdim=True)

        x = x.repeat(1, 1, n_pts)
        x = torch.cat([x, x123], dim=-2)

        x = self.conv7(x)
        x = self.conv8(x)
        if self.num_classes is not None:
            x = self.dp(x)
            x = self.conv9(x).squeeze()

        return x

class VN_Backbone(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        k: int = 20,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim
        
        self.bn8 = nn.BatchNorm1d(256)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 512//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(512//3*2, dim=4, normalize_frame=False)

        self.conv8 = nn.Sequential(nn.Conv1d(1209, self.feat_dim, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size, num_points, _ = x.size()
        x = x.transpose(-1, -2)

        x = x.unsqueeze(1)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = self.pool3(x)
        
        x123 = torch.cat((x1, x2, x3), dim=1)
        
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        
        return x
