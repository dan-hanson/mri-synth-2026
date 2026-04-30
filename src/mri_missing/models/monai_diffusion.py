import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet


class MonaiDiffusionWrapper(nn.Module):
    """
    Wraps MONAI's DiffusionModelUNet.

    When num_class_embeds is set to N, this wrapper internally reserves one
    extra slot (index N) as a "null" / dropped-label token used for
    conditioning dropout during training. The model is built with
    num_class_embeds = N + 1 in that case, transparent to callers — pass
    class labels in [0, N) for real labels, or N for the null token.
    """

    def __init__(
        self,
        in_channels=9,
        out_channels=1,
        channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=False,
        transformer_num_layers=1,
        dropout_cattn=0.0,
        num_class_embeds=None,   # 4 for {t1, t1ce, t2, flair}; null token is added internally
    ):
        super().__init__()
        self.num_class_embeds = num_class_embeds

        # Reserve one extra slot for the null/dropped label
        if num_class_embeds is not None:
            internal_num_class_embeds = num_class_embeds + 1
            self.null_label_index = num_class_embeds  # last slot is null
        else:
            internal_num_class_embeds = None
            self.null_label_index = None

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
            with_conditioning=False,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=None,
            num_class_embeds=internal_num_class_embeds,
            upcast_attention=False,
            dropout_cattn=dropout_cattn,
            include_fc=True,
            use_combined_linear=True,
            use_flash_attention=True,
        )

    def forward(self, x, t, class_labels=None):
        if self.num_class_embeds is not None and class_labels is None:
            raise ValueError(
                "MonaiDiffusionWrapper was built with num_class_embeds, "
                "but forward() was called without class_labels."
            )
        return self.net(x, timesteps=t, class_labels=class_labels)