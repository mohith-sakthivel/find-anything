import torch.nn as nn

from fa.dgcnn import DGCNNSeg
from fa.fusion import SimpleAggregator, DynamicConvolution
from fa.predictor import DGCNNPredHead

FEATURE_EXTRACTORS = {
    "dgcnn": DGCNNSeg,
}

AGGREGATORS = {
    "simple": SimpleAggregator,
    "dynamic_conv": DynamicConvolution,
}

PREDICTORS = {
    "none": nn.Identity,
    "dgcnn": DGCNNPredHead,
}
