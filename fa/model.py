import torch
import torch.nn as nn

from fa.fusion import SimpleAggregator
from fa.predictor import DGCNNPredHead


class FindAnything(nn.Module):
    def __init__(
        self,
        scene_feat_extractor: nn.Module,
        template_feat_extractor: nn.Module,
        fusion_module: nn.Module = SimpleAggregator,
        pred_head: nn.Module = DGCNNPredHead,
    ) -> None:
        super().__init__()

        self.scene_feat_extractor = scene_feat_extractor
        self.fusion_module = fusion_module
        self.pred_head = pred_head

        self.template_feat_extractor = template_feat_extractor

    def forward(self, scene_pointcloud: torch.Tensor, template_pointcloud: torch.Tensor) -> torch.Tensor:
        scene_feat = self.scene_feat_extractor(scene_pointcloud)
        template_feat = self.template_feat_extractor(template_pointcloud)

        fused_feat = self.fusion_module(scene_feat, template_feat)
        preds = self.pred_head(fused_feat)

        return preds
