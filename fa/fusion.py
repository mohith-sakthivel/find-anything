from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fa.dgcnn import build_conv_block_1d, get_graph_feature


class SelfAttention(nn.Module):
    """
    Code taken from https://github.com/Na-Z/attMPTI
    """

    def __init__(self, in_dim: int, out_dim: int = None, key_query_dim: Optional[int] = 64, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim

        self.out_dim = out_dim if out_dim is not None else in_dim
        self.key_query_dim = key_query_dim if key_query_dim is not None else self.out_dim

        self.temperature = self.key_query_dim**0.5

        self.q_map = nn.Conv1d(in_dim, self.key_query_dim, 1, bias=False)
        self.k_map = nn.Conv1d(in_dim, self.key_query_dim, 1, bias=False)
        self.v_map = nn.Conv1d(in_dim, self.out_dim, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1, 2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, v.transpose(1, 2))  # (batch_size, num_points, out_channel)

        return x.transpose(1, 2)


class SimpleAggregator(nn.Module):
    def __init__(
            self,
            scene_feat_dim: int,
            template_feat_dim: int,
            out_dim: Optional[int] = None,
            use_difference: bool = True,
            use_similarity: bool = True,
            use_self_attention: bool = True
        ) -> None:
        super().__init__()

        assert scene_feat_dim == template_feat_dim
        self.scene_feat_dim = scene_feat_dim
        self.template_feat_dim = template_feat_dim
        self.use_difference = use_difference
        self.use_similarity = use_similarity
        self.use_self_attention = use_self_attention

        if self.use_self_attention:
            self.template_sa = SelfAttention(self.template_feat_dim, self.template_feat_dim)

        upscale_factor = 2 + (self.use_difference and self.use_similarity)

        if out_dim is None:
            self.out_dim = upscale_factor * self.scene_feat_dim
            self.projection_head = None
        else:
            self.out_dim = out_dim
            self.projection_head = build_conv_block_1d(upscale_factor * self.scene_feat_dim, out_dim)

    def forward(self, scene_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        if self.use_self_attention:
            template_feat = self.template_sa(template_feat)
        template_feat = template_feat.amax(dim=-1, keepdims=True)

        feat_list = []
        if self.use_difference:
            feat_list.append(scene_feat - template_feat)
        if self.use_similarity:
            feat_list.append(scene_feat * template_feat)
        if not (self.use_difference or self.use_similarity):
            feat_list.append(template_feat.expand(-1, -1, scene_feat.shape[-1]))
        feat_list.append(scene_feat)

        aggr_feat = torch.cat(feat_list, dim=-2)

        if self.projection_head is not None:
            aggr_feat = self.projection_head(aggr_feat)

        return aggr_feat


class DynamicConvolution(nn.Module):
    def __init__(
        self,
        scene_feat_dim: int,
        template_feat_dim: int,
        cond_conv_num_layers: int = 2,
        cond_conv_feat_dim: int = 64,
        use_coords: bool = False,
        use_knn: bool = False,
        predict_mask: bool = False,
    ) -> None:
        super().__init__()

        self.scene_feat_dim = scene_feat_dim
        self.template_feat_dim = template_feat_dim
        self.cond_conv_num_layers = cond_conv_num_layers
        self.cond_conv_feat_dim = cond_conv_feat_dim
        self.use_coords = use_coords
        self.use_knn = use_knn
        self.predict_mask = predict_mask

        self.weight_dims = []
        self.bias_dims = []
        for i in range(cond_conv_num_layers):
            if i == 0 and use_coords:
                self.weight_dims.append((cond_conv_feat_dim + 3) * cond_conv_feat_dim * (2 if self.use_knn else 1))
                self.bias_dims.append(cond_conv_feat_dim)
            else:
                self.weight_dims.append(cond_conv_feat_dim * cond_conv_feat_dim * (2 if self.use_knn else 1))
                self.bias_dims.append(cond_conv_feat_dim)

        if predict_mask:
            self.weight_dims.append(cond_conv_feat_dim * 1)
            self.bias_dims.append(1)
            self.out_dim = None
        else:
            self.out_dim = cond_conv_feat_dim

        self.controller_head = nn.Linear(template_feat_dim, sum(self.weight_dims) + sum(self.bias_dims))
        torch.nn.init.normal_(self.controller_head.weight, std=0.01)
        torch.nn.init.constant_(self.controller_head.bias, 0)

        self.process_scene_feat = SimpleAggregator(scene_feat_dim, template_feat_dim, cond_conv_feat_dim)

    def parse_dynamic_params(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        assert params.dim() == 2
        assert len(self.weight_dims) == len(self.bias_dims)
        assert params.size(-1) == sum(self.weight_dims) + sum(self.bias_dims)

        batches = params.size(0)
        num_layers = len(self.weight_dims)
        params_splits = list(torch.split(params, self.weight_dims + self.bias_dims, dim=-1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            weight_splits[l] = weight_splits[l].reshape(batches * self.bias_dims[l], -1, 1)
            bias_splits[l] = bias_splits[l].reshape(batches * self.bias_dims[l])

        return weight_splits, bias_splits, batches

    def forward(self, scene_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        scene_feat = self.process_scene_feat(scene_feat, template_feat)

        if self.use_coords:
            raise NotImplementedError
        controllers = self.controller_head(template_feat.amax(dim=-1))
        weights, biases, batches = self.parse_dynamic_params(controllers)
        n_layers = len(weights)
        num_pts = scene_feat.shape[-1]

        x = scene_feat.reshape(1, -1, num_pts)
        for i, (w, b) in enumerate(zip(weights, biases)):
            if self.use_knn and (not self.predict_mask or (i + 1) != n_layers):
                x = get_graph_feature(x)
                x = F.conv2d(x, w.unsqueeze(dim=-1), bias=b, groups=batches).amax(dim=-1)
            else:
                x = F.conv1d(x, w, bias=b, groups=batches)

            if not self.predict_mask or ((i + 1) < n_layers):
                x = F.relu(x)

        if self.predict_mask:
            x = x.reshape(batches, num_pts)
        else:
            x = x.reshape(batches, -1, num_pts)

        return x
