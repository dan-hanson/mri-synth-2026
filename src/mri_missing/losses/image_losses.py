import torch
import torch.nn.functional as F

try:
    from monai.losses import SSIMLoss
except ImportError:
    SSIMLoss = None


def normalize_per_case_01(x):
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


class CompositeSynthesisLoss:
    def __init__(
        self,
        use_noise_mse=True,
        noise_mse_weight=1.0,
        use_mae=True,
        mae_weight=0.25,
        use_ssim=True,
        ssim_weight=0.25,
    ):
        self.use_noise_mse = use_noise_mse
        self.noise_mse_weight = noise_mse_weight

        self.use_mae = use_mae
        self.mae_weight = mae_weight

        self.use_ssim = use_ssim and SSIMLoss is not None
        self.ssim_weight = ssim_weight

        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0) if self.use_ssim else None

    def __call__(self, pred_noise, true_noise, pred_x0, target_x0):
        total = 0.0
        parts = {}

        if self.use_noise_mse:
            mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        pred_eval = normalize_per_case_01(pred_x0)
        target_eval = normalize_per_case_01(target_x0)

        if self.use_mae:
            mae = F.l1_loss(pred_eval, target_eval)
            total = total + self.mae_weight * mae
            parts["mae_loss"] = mae.item()

        if self.use_ssim:
            ssim = self.ssim_loss(pred_eval, target_eval)
            total = total + self.ssim_weight * ssim
            parts["ssim_loss"] = ssim.item()

        return total, parts