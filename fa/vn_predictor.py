import torch
import torch.nn as nn

from fa.vn_dgcnn import build_conv_block_1d, build_conv_block_2d, get_graph_feature, \
    VNMaxPool, VNLinearLeakyReLU, VNStdFeature, mean_pool


class VN_DGCNNPredHead(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        dropout: float = 0.5, 
        use_knn: bool = True, 
        k: int = 20,
        pooling: str = 'max',
    ) -> None:
        super().__init__()

        self.k = k
        self.use_knn = use_knn

        if self.use_knn:
            self.k_conv1 = build_conv_block_2d(2 * in_dim//3, 64//3)
            self.k_conv2 = build_conv_block_2d(64//3, 64//3)
            self.std_feature = VNStdFeature(64//3*2, dim=4, normalize_frame=False)

            if pooling == 'max':
                self.pool1 = VNMaxPool(64//3)
            elif pooling == 'mean':
                self.pool1 = mean_pool

        self.conv1 = build_conv_block_1d(64//3 * 3 if self.use_knn else in_dim, 256)
        self.conv2 = build_conv_block_1d(256, 128)
        self.dp = nn.Dropout(p=dropout)
        self.conv3 = nn.Conv1d(128, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n_pts = x.shape
        
        if self.use_knn:
            x = x.unsqueeze(1)
            x = get_graph_feature(x, self.k)
            x = self.k_conv1(x)
            x = self.k_conv2(x)
            x = self.pool1(x)
            x = x.view(b, -1, n_pts)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dp(x)
        x = self.conv3(x)

        return x.squeeze(dim=-2)

class VN_Head(nn.Module):
    def __init__(
        self, 
        in_dim: int = 256
    ) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(self.in_dim, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x
