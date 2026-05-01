import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet

class MonaiDiffusionWrapper(nn.Module):
    def __init__(
        self,
        in_channels=9,
        out_channels=1,
        channels=(256, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=(0, 0, 512),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=False,
        transformer_num_layers=1,
        dropout_cattn=0.0,
        use_flash_attention=False, # Exposed to prevent 3D hardware crashes
        num_class_embeds=None,
    ):
        super().__init__()
        self.net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            with_conditioning=False, # True is only for cross-attention
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=None,
            num_class_embeds=None,
            upcast_attention=False,
            dropout_cattn=dropout_cattn,
            include_fc=True,
            use_combined_linear=True,
            use_flash_attention=use_flash_attention,
        )

    def forward(self, x, t, class_labels=None):
        return self.net(x, timesteps=t, class_labels=class_labels)