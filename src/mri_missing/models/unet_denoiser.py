import torch
import torch.nn as nn
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet3D(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, base_ch=16):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(128)

        self.time_mlp = nn.Sequential(
            nn.Linear(128, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 4)
        )

        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        t_emb = self.time_embed(t.float()).to(x.dtype)
        t_proj = self.time_mlp(t_emb).view(b.shape[0], -1, 1, 1, 1)
        b = b + t_proj

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)