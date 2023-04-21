from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fa.dgcnn import build_conv_block_1d, build_conv_block_2d, get_graph_feature


class SimpleAggregator(nn.Module):
    def __init__(self, scene_feat_dim: int, template_feat_dim: int, project_dim: Optional[int] = None) -> None:
        super().__init__()

        assert scene_feat_dim == template_feat_dim
        self.scene_feat_dim = scene_feat_dim
        self.template_feat_dim = 3 * template_feat_dim
        if project_dim is None:
            self.out_dim = 3 * self.scene_feat_dim
            self.projection_head = None
        else:
            self.out_dim = project_dim
            self.projection_head = build_conv_block_1d(3 * self.scene_feat_dim, project_dim)

    def forward(self, scene_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        template_feat = template_feat.amax(dim=-1, keepdims=True)
        aggr_feat = torch.cat([scene_feat - template_feat, scene_feat * template_feat, scene_feat], dim=-2)

        if self.projection_head is not None:
            aggr_feat = self.projection_head(aggr_feat)

        return aggr_feat


class DynamicConvolution(nn.Module):
    def __init__(
        self,
        scene_feat_dim: int,
        template_feat_dim: int,
        cond_conv_num_layers: int,
        cond_conv_feat_dim: int,
        use_coords: bool = True,
        predict_mask: bool = True,
    ) -> None:
        super().__init__()

        self.scene_feat_dim = scene_feat_dim
        self.template_feat_dim = template_feat_dim
        self.cond_conv_num_layers = cond_conv_num_layers
        self.cond_conv_feat_dim = cond_conv_feat_dim
        self.use_coords = use_coords
        self.predict_mask = predict_mask

        self.weight_dims = []
        self.bias_dims = []
        for i in range(cond_conv_num_layers):
            if i == 0 and use_coords:
                self.weight_dims.append((cond_conv_feat_dim + 3) * cond_conv_feat_dim)
                self.bias_dims.append(cond_conv_feat_dim)
            else:
                self.weight_dims.append(cond_conv_feat_dim * cond_conv_feat_dim)
                self.bias_dims.append(cond_conv_feat_dim)

        if predict_mask:
            self.weight_dims.append(cond_conv_feat_dim * 1)
            self.bias_dims.append(1)

        self.process_template_feat = build_conv_block_2d(template_feat_dim, template_feat_dim)

        self.controller_head = nn.Sequential(
            nn.Conv1d(template_feat_dim, template_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(template_feat_dim, sum(self.weight_dims) + sum(self.bias_dims)),
        )

        self.process_scene_feat = build_conv_block_2d(scene_feat_dim, cond_conv_feat_dim)

    def parse_dynamic_params(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        assert params.dim() == 2
        assert len(self.weight_dims) == len(self.bias_dims)
        assert params.size(1) == sum(self.weight_dims) + sum(self.bias_dims)

        batches = params.size(0)
        num_layers = len(self.weight_dims)
        params_splits = list(torch.split_with_sizes(params, self.weight_dims + self.bias_dims, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            weight_splits[l] = weight_splits[l].reshape(batches * self.bias_dims[l], -1, 1)
            bias_splits[l] = bias_splits[l].reshape(batches * self.bias_dims[l])

        return weight_splits, bias_splits, batches

    def forward(self, scene_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        scene_feat = get_graph_feature(scene_feat)
        scene_feat = self.process_scene_feat(scene_feat).amax(dim=-1)

        template_feat = get_graph_feature(template_feat)
        template_feat = self.process_template_feat(template_feat).amax(dim=-1)

        controllers = self.controller_head(template_feat.amax(dim=-1, keepdim=True))
        weights, biases, batches = self.parse_dynamic_params(controllers)
        n_layers = len(weights)
        num_pts = scene_feat.shape[-1]

        x = scene_feat.reshape(1, -1, scene_feat.shape[-1])
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=batches)
            if not self.predict_mask or (i < (n_layers - 1)):
                x = F.relu(x)
        x = x.reshape(batches, -1, num_pts)

        return x
