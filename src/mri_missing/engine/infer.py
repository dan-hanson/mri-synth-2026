import os
import sys
import time
import torch
import numpy as np

sys.path.append(r"C:\mri_synth_2026\src")

from mri_missing.config import load_config, ensure_dir
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.models.time_embedding import SinusoidalTimeEmbedding
from mri_missing.utils.io import load_checkpoint, save_json
from mri_missing.utils.cases import MOD_KEYS, resolve_case_dirs
from mri_missing.utils.nifti import load_case, save_prediction_nifti, normalize_zscore, normalize_per_case_01_np
from mri_missing.utils.visualization import save_slice_panel
from monai.inferers import sliding_window_inference
from mri_missing.utils.ema import EMA
from mri_missing.metrics.image_metrics import ImageMetricBundle


def prepare_condition(cfg, data_dict, missing_key):
    cond = []
    mask = []
    target = data_dict[missing_key].copy()

    fill_value = cfg["missing_policy"]["fill_value"]
    use_presence_mask = cfg["missing_policy"]["use_presence_mask"]

    for k in MOD_KEYS:
        if k == missing_key:
            cond.append(np.full_like(data_dict[k], fill_value, dtype=np.float32))
            mask.append(np.zeros_like(data_dict[k], dtype=np.float32))
        else:
            cond.append(data_dict[k].astype(np.float32))
            mask.append(np.ones_like(data_dict[k], dtype=np.float32))

    cond = np.stack(cond, axis=0)   # [4, D, H, W]

    if use_presence_mask:
        mask = np.stack(mask, axis=0)   # [4, D, H, W]
        cond = np.concatenate([cond, mask], axis=0)   # [8, D, H, W]

    return cond, target


def resolve_checkpoint_path(cfg):
    ckpt_path = cfg["inference"].get("checkpoint_path")
    run_dir = cfg["inference"].get("run_dir")
    ckpt_name = cfg["inference"].get("checkpoint_name", "best.pt")

    if ckpt_path:
        return ckpt_path

    if run_dir:
        return os.path.join(run_dir, "checkpoints", ckpt_name)

    raise ValueError("Set either inference.checkpoint_path or inference.run_dir in config.")


def build_infer_model(cfg, device):
    model_name = cfg["model"]["name"].lower()
    model_kwargs = {
        "in_channels": cfg["model"]["in_channels"],
        "out_channels": cfg["model"]["out_channels"],
    }

    if model_name == "unet":
        model_kwargs["base_ch"] = cfg["model"]["base_ch"]
    elif model_name == "convnext3d":
        model_kwargs["base_dim"] = cfg["model"]["base_dim"]

    model = build_model(model_name, **model_kwargs).to(device)
    return model


@torch.inference_mode()
def infer_single_case(cfg, model, diffusion, time_embed, case_dir, missing_key, device):
    data_dict, affine, header = load_case(case_dir)

    # training-style normalization
    for k in data_dict:
        data_dict[k] = normalize_zscore(data_dict[k])

    cond_np, target_np = prepare_condition(cfg, data_dict, missing_key)

    cond = torch.tensor(cond_np[None], dtype=torch.float32, device=device)  # [1,8,D,H,W]
    x = torch.randn((1, 1, *target_np.shape), device=device)

    reverse_steps = cfg["inference"]["reverse_steps"]
    total_train_steps = cfg["diffusion"]["timesteps"]

    if reverse_steps >= total_train_steps:
        t_schedule = list(reversed(range(total_train_steps)))
    else:
        idxs = np.linspace(0, total_train_steps - 1, reverse_steps, dtype=int)
        t_schedule = list(reversed(idxs.tolist()))

    use_sw = cfg["inference"]["sliding_window"]["enabled"]
    roi_size = tuple(int(x) for x in cfg["inference"]["sliding_window"]["roi_size"])
    sw_batch_size = cfg["inference"]["sliding_window"]["sw_batch_size"]
    overlap = cfg["inference"]["sliding_window"]["overlap"]

    start = time.time()

    for t_idx in t_schedule:
        t = torch.tensor([t_idx], device=device, dtype=torch.float32)
        t_emb = time_embed(t)

        model_in = torch.cat([cond, x], dim=1)  # [1,8,D,H,W]

        if use_sw:
            def predictor(patch_x):
                return model(patch_x, t_emb)

            pred_noise = sliding_window_inference(
                inputs=model_in,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=overlap,
            )
        else:
            pred_noise = model(model_in, t_emb)

        alpha = diffusion.alphas[t_idx].view(1, 1, 1, 1, 1)
        alpha_bar = diffusion.alpha_cumprod[t_idx].view(1, 1, 1, 1, 1)
        beta = diffusion.betas[t_idx].view(1, 1, 1, 1, 1)

        if t_idx > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
        ) + torch.sqrt(beta) * z

    elapsed = time.time() - start

    pred_full = x[0, 0].detach().cpu().numpy()
    pred_full = normalize_per_case_01_np(pred_full)

    return pred_full, target_np, cond_np, affine, header, elapsed


def main():
    cfg = load_config("configs/base.yaml")

    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = resolve_checkpoint_path(cfg)

    model = build_infer_model(cfg, device)
    ema = EMA(model, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None
    load_checkpoint(
        ckpt_path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        ema=ema,
        map_location=device,
    )
    infer_model = ema.shadow if (ema is not None and cfg["ema"]["infer_with_ema"]) else model
    infer_model.eval()

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    metric_bundle = ImageMetricBundle(
        compute_mae=cfg["metrics"]["compute_mae"],
        compute_psnr=cfg["metrics"]["compute_psnr"],
        compute_ssim=cfg["metrics"]["compute_ssim"],
        eval_norm=cfg.get("metrics", {}).get("eval_norm", "per_case_minmax"),
    )

    time_embed = SinusoidalTimeEmbedding(128).to(device)

    data_root = cfg["inference"]["data_root"]
    case_ids = cfg["inference"].get("case_ids", [])
    missing_keys = cfg["inference"].get("missing_keys", ["t1ce"])

    case_dirs = resolve_case_dirs(
        data_root,
        case_ids=case_ids,
        strict=False,
        random_case_count=cfg["inference"].get("random_case_count", 0),
        random_case_seed=cfg["inference"].get("random_case_seed", 42),
    )

    base_out = os.path.join(
        cfg["project"]["output_root"],
        cfg["inference"]["output_subdir"],
        cfg["model"]["name"],
    )
    ensure_dir(base_out)

    summary = []

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)

        for missing_key in missing_keys:
            pred, target, cond, affine, header, elapsed = infer_single_case(
                cfg, infer_model, diffusion, time_embed, case_dir, missing_key, device
            )

            pred_t = torch.tensor(pred[None, None], dtype=torch.float32)
            target_t = torch.tensor(target[None, None], dtype=torch.float32)
            infer_metrics = metric_bundle(pred_t, target_t)

            item_dir = os.path.join(base_out, f"{case_id}_{missing_key}")
            ensure_dir(item_dir)

            if cfg["inference"]["save_nifti"]:
                nifti_path = os.path.join(item_dir, f"{case_id}_pred_{missing_key}.nii.gz")
                save_prediction_nifti(pred, affine, header, nifti_path)

            if cfg["inference"]["save_png"]:
                png_path = os.path.join(item_dir, f"{case_id}_panel_{missing_key}.png")
                save_slice_panel(cond, target, pred, png_path, missing_key=missing_key, title=f"{case_id}")

            summary.append({
                "case_id": case_id,
                "missing_key": missing_key,
                "elapsed_sec": elapsed,
                "checkpoint": ckpt_path,
                **infer_metrics,
            })

            print(f"[infer] case={case_id} missing={missing_key} time={elapsed:.2f}s")
            

    save_json(os.path.join(base_out, "summary.json"), summary)
    print(f"Saved inference outputs to: {base_out}")


if __name__ == "__main__":
    main()