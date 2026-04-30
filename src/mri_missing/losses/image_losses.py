import torch
import torch.nn.functional as F

try:
    from monai.losses import SSIMLoss
except ImportError:
    SSIMLoss = None


class CompositeSynthesisLoss:
    """
    Composite loss for noise-prediction diffusion training.

    Components:
      - noise MSE      (full weight, applied uniformly across t)
      - raw MAE        (z-score space, NO clamping — preserves intensity scale)
      - clamped MAE    (mapped to [0,1] for stability, Min-SNR weighted)
      - SSIM           (mapped to [0,1], Min-SNR weighted)

    Tumor-region weighting:
      Where a tumor mask is provided, MAE losses inside the tumor get
      multiplied by `tumor_weight` (default 3.0). This pushes the model
      to invest gradient signal in lesion appearance instead of producing
      generic healthy-brain output.

    Min-SNR-gamma weighting (Hang et al. 2023):
      w(t) = min(SNR(t), gamma) / SNR(t)
      Used on the clamped image-space losses (MAE, SSIM) to balance learning
      across timesteps.
    """

    def __init__(
        self,
        use_noise_mse=True,
        noise_mse_weight=1.0,
        use_mae=True,
        mae_weight=1.0,
        use_ssim=True,
        ssim_weight=1.0,
        use_raw_mae=True,
        raw_mae_weight=0.5,
        tumor_weight=3.0,
        min_snr_gamma=0.5,
        **kwargs,
    ):
        self.use_noise_mse = use_noise_mse
        self.noise_mse_weight = noise_mse_weight
        self.use_mae = use_mae
        self.mae_weight = mae_weight
        self.use_ssim = use_ssim and SSIMLoss is not None
        self.ssim_weight = ssim_weight
        self.use_raw_mae = use_raw_mae
        self.raw_mae_weight = raw_mae_weight
        self.tumor_weight = tumor_weight
        self.min_snr_gamma = min_snr_gamma

        self.ssim_loss = (
            SSIMLoss(spatial_dims=3, data_range=1.0, reduction="none")
            if self.use_ssim
            else None
        )

    def __call__(
        self,
        pred_noise,
        true_noise,
        pred_x0,
        target_x0,
        snr=None,
        tumor_mask=None,
        **kwargs,
    ):
        total = 0.0
        parts = {}

        # 1. Noise MSE — full weight, uniform across t
        if self.use_noise_mse:
            mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        # 2. Raw MAE in z-score space — NO clamping. This lets the model
        # learn intensity scale including extreme values like edema (z >> 3)
        # and CSF (z << -1). Tumor-weighted if mask given.
        if self.use_raw_mae:
            raw_mae_per_voxel = F.l1_loss(pred_x0, target_x0, reduction="none")

            if tumor_mask is not None and self.tumor_weight != 1.0:
                # weight mask: 1.0 outside tumor, tumor_weight inside
                w_mask = 1.0 + (self.tumor_weight - 1.0) * tumor_mask
                raw_mae_per_voxel = raw_mae_per_voxel * w_mask

            raw_mae = raw_mae_per_voxel.mean()
            total = total + self.raw_mae_weight * raw_mae
            parts["raw_mae"] = raw_mae.item()

        # 3. Map z-scored data to [0, 1] for clamped image-space metrics
        pred_eval = torch.clamp((pred_x0 + 3.0) / 6.0, min=0.0, max=1.0)
        target_eval = torch.clamp((target_x0 + 3.0) / 6.0, min=0.0, max=1.0)

        # 4. Min-SNR-gamma weighting
        if snr is not None:
            snr_flat = snr.view(-1)
            aux_weight = torch.clamp(snr_flat, max=self.min_snr_gamma) / (snr_flat + 1e-8)
        else:
            aux_weight = torch.ones(pred_x0.shape[0], device=pred_x0.device)

        # 5. Clamped MAE with optional tumor weighting
        if self.use_mae:
            mae_per_voxel = F.l1_loss(pred_eval, target_eval, reduction="none")

            if tumor_mask is not None and self.tumor_weight != 1.0:
                w_mask = 1.0 + (self.tumor_weight - 1.0) * tumor_mask
                mae_per_voxel = mae_per_voxel * w_mask

            mae_per_batch = mae_per_voxel.mean(dim=(1, 2, 3, 4))
            mae_weighted = (mae_per_batch * aux_weight).mean()

            total = total + self.mae_weight * mae_weighted
            parts["mae_loss"] = mae_weighted.item()

        # 6. SSIM. SSIM is structural — tumor weighting is awkward because
        # SSIM operates on a window, not voxels. Leave unmodified.
        if self.use_ssim:
            ssim_per_batch = self.ssim_loss(pred_eval, target_eval).view(-1)
            ssim_weighted = (ssim_per_batch * aux_weight).mean()

            total = total + self.ssim_weight * ssim_weighted
            parts["ssim_loss"] = ssim_weighted.item()

        # Diagnostics
        parts["aux_w_mean"] = float(aux_weight.mean().item())
        if tumor_mask is not None:
            parts["tumor_frac"] = float(tumor_mask.mean().item())

        return total, parts