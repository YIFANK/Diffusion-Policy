import torch
import torch.nn as nn
from torch.nn import functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = feature.view(B, C, H * W)
        softmax = F.softmax(feature, dim=-1)
        exp_x = torch.sum(self.pos_x * softmax, dim=-1)
        exp_y = torch.sum(self.pos_y * softmax, dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)

# class Resnet(nn.Module):
#     def __init__(self, out_dim=64):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
#             nn.GroupNorm(4, 32),
#             nn.ReLU(inplace=True)
#         )
#         self.res_blocks = nn.Sequential(
#             ResidualBlock(32, 64, stride=2),
#             ResidualBlock(64, 128, stride=2),
#             ResidualBlock(128, 256, stride=2)
#         )
#         self.spatial_softmax = SpatialSoftmax(6, 6)
#         self.fc = nn.Linear(512, out_dim)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.res_blocks(x)
#         x = self.spatial_softmax(x)
#         return self.fc(x)

class Resnet(nn.Module):
    def __init__(self, out_dim=64, k_obs=1):
        super().__init__()
        self.k_obs = k_obs
        in_channels = 3 * k_obs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )
        self.spatial_softmax = SpatialSoftmax(6, 6)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # Expect input shape: (B, k_obs, 3, H, W) => reshape to (B, 3*k_obs, H, W)
        B, k, C, H, W = x.shape
        x = x.view(B, k * C, H, W)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.spatial_softmax(x)
        return self.fc(x)