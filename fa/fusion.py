from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fa.dgcnn import build_conv_block_1d, get_graph_feature


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

            if self.use_knn and (not self.predict_mask or (i + 1) != n_layers) :
                x = get_graph_feature(x)
                x = F.conv2d(x, w.unsqueeze(dim=-1), bias=b, groups=batches).amax(dim=-1)
            else:
                x = F.conv1d(x, w, bias=b,  groups=batches)
                
            if not self.predict_mask or ((i + 1) < n_layers):
                x = F.relu(x)

        if self.predict_mask:
            x = x.reshape(batches, num_pts)
        else:
            x = x.reshape(batches, -1, num_pts)

        return x
