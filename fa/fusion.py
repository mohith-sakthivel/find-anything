import torch 
import torch.nn as nn


class SimpleAggregator(nn.Module):

    def __init__(self, in_dim: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = 3 * in_dim

    def forward(self, scene_feat, template_feat) -> torch.Tensor:
        template_feat = template_feat.unsqueeze(dim=-1)
        aggr_feat = torch.cat([scene_feat - template_feat, scene_feat * template_feat, scene_feat], dim=-2)
        return aggr_feat
