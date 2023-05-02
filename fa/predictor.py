import torch
import torch.nn as nn

from fa.dgcnn import build_conv_block_1d, build_conv_block_2d, get_graph_feature


class DGCNNPredHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dropout: float = 0.5,
        use_knn: bool = True,
        k: int = 20,
    ) -> None:
        super().__init__()

        self.k = k
        self.use_knn = use_knn

        if self.use_knn:
            self.k_conv1 = build_conv_block_2d(2 * in_dim, 64)
            self.k_conv2 = build_conv_block_2d(64, 64)

        self.conv1 = build_conv_block_1d(64 if self.use_knn else in_dim, 256)
        self.conv2 = build_conv_block_1d(256, 128)
        self.dp = nn.Dropout(p=dropout)
        self.conv3 = nn.Conv1d(128, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_knn:
            x = get_graph_feature(x, self.k)
            x = self.k_conv1(x)
            x = self.k_conv2(x)
            x = x.amax(dim=-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dp(x)
        x = self.conv3(x)

        return x.squeeze(dim=-2)
