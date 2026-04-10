from mri_missing.models.unet_denoiser import TinyUNet3D


def build_model(name: str, **kwargs):
    name = name.lower()

    if name == "unet":
        return TinyUNet3D(**kwargs)

    elif name == "convnext3d":
        from mri_missing.models.convnext3d import ConvNeXt3DDenoiser
        return ConvNeXt3DDenoiser(**kwargs)

    elif name == "swin":
        from mri_missing.models.swinUNETR import SwinUNETRWrapper
        return SwinUNETRWrapper(**kwargs)
    
    elif name == "swin_ddpm":
        from mri_missing.models.swin_UNTR_DDPM import SwinUNETR
        return SwinUNETR(**kwargs)

    elif name == "monai_diffusion":
        from mri_missing.models.monai_diffusion import MonaiDiffusionWrapper
        return MonaiDiffusionWrapper(**kwargs)

    else:
        raise ValueError(f"Unknown model: {name}")