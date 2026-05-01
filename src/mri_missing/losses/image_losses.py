import torch
import torch.nn.functional as F

try:
    from monai.losses import SSIMLoss
except ImportError:
    SSIMLoss = None


# MONAI's default SSIM window is 11x11x11. A tumor-region crop smaller than
# that in any axis can't be SSIM'd; we fall back to the healthy term only.
SSIM_MIN_DIM = 11


def _tumor_bbox_or_none(tumor_mask, min_dim=SSIM_MIN_DIM):
    """
    Return (d0, d1, h0, h1, w0, w1) for the tight bounding box of the tumor,
    or None if no tumor or the box is too small for the SSIM window.

    Operates on the first sample only (batch size 1 is the common case).
    """
    if tumor_mask is None:
        return None
    bool_mask = tumor_mask[0, 0] > 0  # [D, H, W]
    if not bool_mask.any():
        return None

    coords = torch.where(bool_mask)
    d0, d1 = coords[0].min().item(), coords[0].max().item() + 1
    h0, h1 = coords[1].min().item(), coords[1].max().item() + 1
    w0, w1 = coords[2].min().item(), coords[2].max().item() + 1

    if (d1 - d0) < min_dim or (h1 - h0) < min_dim or (w1 - w0) < min_dim:
        return None

    return d0, d1, h0, h1, w0, w1


class CompositeSynthesisLoss:
    """
    Composite loss for noise-prediction diffusion training.

    Components:
      - noise MSE       (full weight, applied uniformly across t)
      - MAE on pred_x0  (Min-SNR weighted, healthy = whole brain)
      - SSIM split into:
          * healthy SSIM: whole-volume SSIM (BraSyn paper's healthy proxy)
          * tumor SSIM:   SSIM on tumor bounding-box crop
        Each with its own weight. When the patch has no tumor (or tumor is
        too small for the SSIM window), the tumor term is skipped.

    Min-SNR-gamma weighting (Hang et al. 2023):
        w(t) = min(SNR(t), gamma) / SNR(t)
    """

    def __init__(
        self,
        use_noise_mse=True,
        noise_mse_weight=1.0,
        use_mae=True,
        mae_weight=1.0,
        use_ssim=True,
        healthy_ssim_weight=1.0,
        tumor_ssim_weight=1.0,
        min_snr_gamma=0.5,
        **kwargs,
    ):
        self.use_noise_mse = use_noise_mse
        self.noise_mse_weight = noise_mse_weight
        self.use_mae = use_mae
        self.mae_weight = mae_weight
        self.use_ssim = use_ssim and SSIMLoss is not None
        self.healthy_ssim_weight = healthy_ssim_weight
        self.tumor_ssim_weight = tumor_ssim_weight
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

        # 1. Noise MSE
        if self.use_noise_mse:
            mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        # 2. Map z-scored data to [0, 1] for image-space metrics
        pred_eval = torch.clamp((pred_x0 + 3.0) / 6.0, min=0.0, max=1.0)
        target_eval = torch.clamp((target_x0 + 3.0) / 6.0, min=0.0, max=1.0)

        # 3. Min-SNR weighting
        if snr is not None:
            snr_flat = snr.view(-1)
            aux_weight = torch.clamp(snr_flat, max=self.min_snr_gamma) / (snr_flat + 1e-8)
        else:
            aux_weight = torch.ones(pred_x0.shape[0], device=pred_x0.device)

        # 4. MAE (whole volume, Min-SNR weighted)
        if self.use_mae:
            mae_per_voxel = F.l1_loss(pred_eval, target_eval, reduction="none")
            mae_per_batch = mae_per_voxel.mean(dim=(1, 2, 3, 4))
            mae_weighted = (mae_per_batch * aux_weight).mean()
            total = total + self.mae_weight * mae_weighted
            parts["mae_loss"] = mae_weighted.item()

        # 5. Region-split SSIM
        if self.use_ssim:
            # Healthy SSIM: whole-volume SSIM (BraSyn-style healthy proxy)
            healthy_ssim = self.ssim_loss(pred_eval, target_eval).view(-1)
            healthy_weighted = (healthy_ssim * aux_weight).mean()
            total = total + self.healthy_ssim_weight * healthy_weighted
            parts["healthy_ssim_loss"] = healthy_weighted.item()

            # Tumor SSIM: SSIM on tumor bounding-box crop, when available
            bbox = _tumor_bbox_or_none(tumor_mask)
            if bbox is not None and self.tumor_ssim_weight > 0.0:
                d0, d1, h0, h1, w0, w1 = bbox
                pred_tumor = pred_eval[:, :, d0:d1, h0:h1, w0:w1]
                target_tumor = target_eval[:, :, d0:d1, h0:h1, w0:w1]
                tumor_ssim = self.ssim_loss(pred_tumor, target_tumor).view(-1)
                tumor_weighted = (tumor_ssim * aux_weight).mean()
                total = total + self.tumor_ssim_weight * tumor_weighted
                parts["tumor_ssim_loss"] = tumor_weighted.item()
                parts["tumor_ssim_active"] = 1.0
            else:
                # Logged so you can see how often tumor SSIM is firing
                parts["tumor_ssim_loss"] = 0.0
                parts["tumor_ssim_active"] = 0.0

        # Diagnostics
        parts["aux_w_mean"] = float(aux_weight.mean().item())
        if tumor_mask is not None:
            parts["tumor_frac"] = float(tumor_mask.mean().item())

        return total, parts