import torch
import torch.nn.functional as F

try:
    from monai.metrics import SSIMMetric
except ImportError:
    SSIMMetric = None


def normalize_per_case_01(x):
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


class ImageMetricBundle:
    def __init__(
        self,
        compute_mae=True,
        compute_psnr=True,
        compute_ssim=True,
        eval_norm="per_case_minmax",
        data_range=None,
    ):
        self.compute_mae = compute_mae
        self.compute_psnr = compute_psnr
        self.compute_ssim = compute_ssim and SSIMMetric is not None
        self.eval_norm = eval_norm

        if data_range is None:
            self.data_range = 1.0 if eval_norm == "per_case_minmax" else 2.0
        else:
            self.data_range = float(data_range)

        self.ssim = SSIMMetric(spatial_dims=3, data_range=self.data_range) if self.compute_ssim else None

    def _prep(self, pred, target):
        if self.eval_norm == "per_case_minmax":
            pred = normalize_per_case_01(pred)
            target = normalize_per_case_01(target)
        elif self.eval_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown eval_norm: {self.eval_norm}")
        return pred, target

    def __call__(self, pred, target, mask=None):
        pred, target = self._prep(pred, target)
        out = {}

        if mask is not None:
            bool_mask = mask > 0

            if self.compute_mae:
                mae = F.l1_loss(pred[bool_mask], target[bool_mask], reduction="mean")
                out["mae"] = mae.item()

            if self.compute_psnr:
                mse = F.mse_loss(pred[bool_mask], target[bool_mask], reduction="mean")
                if mse == 0:
                    out["psnr"] = 100.0
                else:
                    out["psnr"] = 10 * torch.log10((self.data_range ** 2) / mse).item()

            if self.compute_ssim:
                coords = torch.where(bool_mask)
                if len(coords[0]) > 0:
                    d0, d1 = coords[2].min(), coords[2].max() + 1
                    h0, h1 = coords[3].min(), coords[3].max() + 1
                    w0, w1 = coords[4].min(), coords[4].max() + 1

                    pred_crop = pred[:, :, d0:d1, h0:h1, w0:w1]
                    targ_crop = target[:, :, d0:d1, h0:h1, w0:w1]

                    ssim_val = self.ssim(pred_crop, targ_crop)
                    out["ssim"] = ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)
                else:
                    out["ssim"] = 0.0
        else:
            if self.compute_mae:
                out["mae"] = F.l1_loss(pred, target, reduction="mean").item()

            if self.compute_psnr:
                mse = F.mse_loss(pred, target, reduction="mean")
                out["psnr"] = 100.0 if mse == 0 else 10 * torch.log10((self.data_range ** 2) / mse).item()

            if self.compute_ssim:
                ssim_val = self.ssim(pred, target)
                out["ssim"] = ssim_val.mean().item() if torch.is_tensor(ssim_val) else float(ssim_val)

        return out