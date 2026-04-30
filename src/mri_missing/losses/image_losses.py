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
      - noise MSE       (full weight, applied uniformly across t)
      - MAE on pred_x0  (Min-SNR weighted)
      - SSIM on pred_x0 (Min-SNR weighted)

    Min-SNR-gamma weighting (Hang et al. 2023):
        w(t) = min(SNR(t), gamma) / SNR(t)
    For epsilon-prediction this caps the influence of easy low-t samples
    and gives full weight to harder high-t samples, which is what we want
    when conditioning under-utilization is a problem.
    """

    def __init__(
        self,
        use_noise_mse=True,
        noise_mse_weight=1.0,
        use_mae=True,
        mae_weight=1.0,
        use_ssim=True,
        ssim_weight=1.0,
        min_snr_gamma=5.0,
        **kwargs,
    ):
        self.use_noise_mse = use_noise_mse
        self.noise_mse_weight = noise_mse_weight
        self.use_mae = use_mae
        self.mae_weight = mae_weight
        self.use_ssim = use_ssim and SSIMLoss is not None
        self.ssim_weight = ssim_weight
        self.min_snr_gamma = min_snr_gamma

        # reduction="none" so we can apply per-batch Min-SNR weights
        self.ssim_loss = (
            SSIMLoss(spatial_dims=3, data_range=1.0, reduction="none")
            if self.use_ssim
            else None
        )

    def __call__(self, pred_noise, true_noise, pred_x0, target_x0, snr=None, **kwargs):
        total = 0.0
        parts = {}

        # 1. Noise MSE — full weight, uniform across t
        if self.use_noise_mse:
            mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        # 2. Map z-scored data [-3, 3] to [0, 1] for image-space metrics
        pred_eval = torch.clamp((pred_x0 + 3.0) / 6.0, min=0.0, max=1.0)
        target_eval = torch.clamp((target_x0 + 3.0) / 6.0, min=0.0, max=1.0)

        # 3. Min-SNR-gamma weighting for aux losses
        # FIX: was `clamp(snr, max=gamma)` which up-weights low-t.
        # Correct form is min(snr, gamma) / snr, which gives full weight at
        # high-t (SNR < gamma) and down-weights low-t (SNR > gamma).
        if snr is not None:
            snr_flat = snr.view(-1)
            aux_weight = torch.clamp(snr_flat, max=self.min_snr_gamma) / (snr_flat + 1e-8)
        else:
            aux_weight = torch.ones(pred_x0.shape[0], device=pred_x0.device)

        # 4. Min-SNR-weighted MAE
        if self.use_mae:
            mae_per_voxel = F.l1_loss(pred_eval, target_eval, reduction="none")
            mae_per_batch = mae_per_voxel.mean(dim=(1, 2, 3, 4))
            mae_weighted = (mae_per_batch * aux_weight).mean()

            total = total + self.mae_weight * mae_weighted
            parts["mae_loss"] = mae_weighted.item()

        # 5. Min-SNR-weighted SSIM
        if self.use_ssim:
            ssim_per_batch = self.ssim_loss(pred_eval, target_eval).view(-1)
            ssim_weighted = (ssim_per_batch * aux_weight).mean()

            total = total + self.ssim_weight * ssim_weighted
            parts["ssim_loss"] = ssim_weighted.item()

        # Diagnostic: report mean Min-SNR weight so you can sanity-check it
        # in the training log. Should be < 1.0 on average (usually 0.3–0.5).
        parts["aux_w_mean"] = float(aux_weight.mean().item())

        return total, parts