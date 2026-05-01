import math
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SwinDenoiser(nn.Module):
    def __init__(
        self,
        in_channels=9,
        out_channels=1,
        img_size=(96, 96, 32),
        feature_size=48,           
        time_embed_dim=16,         
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4),     # <-- could be tuned, but start here
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        
        # 1. Process the Timestep
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # 2. The Core Swin Network
        self.net = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels + time_embed_dim, # Native MRI + Time Map
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            normalize=True,       
        )

        # Override MONAI's default window_size dynamically
        self.net.swinViT.window_size = window_size

    def forward(self, x, t):
        # 1. Embed time
        t_emb = self.time_mlp(t)  
        
        # 2. Expand time to match the 3D spatial dimensions of the MRI
        t_spatial = t_emb.view(x.shape[0], -1, 1, 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3], x.shape[4]
        )
        
        # 3. Concatenate Time + MRI
        x_in = torch.cat([x, t_spatial], dim=1)
        
        # 4. Predict Noise
        return self.net(x_in)