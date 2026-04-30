import torch
import torch.nn as nn


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.InstanceNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class ConvNeXt3DDenoiser(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_dim=32):
        super().__init__()

        self.stem = nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1)

        self.block1 = ConvNeXtBlock3D(base_dim)
        self.block2 = ConvNeXtBlock3D(base_dim)
        self.block3 = ConvNeXtBlock3D(base_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(128, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim)
        )

        self.head = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x, t_emb, class_labels=None):
        # class_labels accepted for registry uniformity; not yet used here.
        x = self.stem(x)

        t = self.time_mlp(t_emb).view(x.shape[0], -1, 1, 1, 1)
        x = x + t

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return self.head(x)