import os, sys
import time
import json
import torch
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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

    cond = np.stack(cond, axis=0)  # [4, D, H, W]

    if use_presence_mask:
        mask = np.stack(mask, axis=0)  # [4, D, H, W]
        cond = np.concatenate([cond, mask], axis=0)  # [8, D, H, W]

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
    elif model_name == "monai_diffusion_ssim":
        model_kwargs["channels"] = tuple(cfg["model"]["channels"])
        model_kwargs["attention_levels"] = tuple(cfg["model"]["attention_levels"])
        model_kwargs["num_res_blocks"] = cfg["model"]["num_res_blocks"]
        model_kwargs["num_head_channels"] = cfg["model"]["num_head_channels"]
        model_kwargs["norm_num_groups"] = cfg["model"].get("norm_num_groups", 32)
        model_kwargs["norm_eps"] = cfg["model"].get("norm_eps", 1e-6)
        model_kwargs["resblock_updown"] = cfg["model"].get("resblock_updown", False)
        model_kwargs["transformer_num_layers"] = cfg["model"].get("transformer_num_layers", 1)
        model_kwargs["dropout_cattn"] = cfg["model"].get("dropout_cattn", 0.0)

    model = build_model(model_name, **model_kwargs).to(device)
    return model


@torch.inference_mode()
def infer_single_case(cfg, model, diffusion, case_dir, missing_key, device):
    data_dict, affine, header = load_case(case_dir)

    # 1. Preserve the absolute native scanner intensities
    raw_data_dict = {k: v.copy() for k, v in data_dict.items()}
    raw_target = raw_data_dict[missing_key]
    raw_min, raw_max = raw_target.min(), raw_target.max()

    # 2. Apply Z-Score so the model gets the distribution it was trained on
    for k in data_dict:
        data_dict[k] = normalize_zscore(data_dict[k])

    cond_np, target_np = prepare_condition(cfg, data_dict, missing_key)

    mask = target_np > target_np.min()
    coords = np.argwhere(mask)
    if len(coords) > 0:
        d0, h0, w0 = coords.min(axis=0)
        d1, h1, w1 = coords.max(axis=0) + 1
    else:
        d0, h0, w0 = 0, 0, 0
        d1, h1, w1 = target_np.shape

    cond_crop = cond_np[:, d0:d1, h0:h1, w0:w1]
    target_crop = target_np[d0:d1, h0:h1, w0:w1]

    cond = torch.tensor(cond_crop[None], dtype=torch.float32, device=device)
    x = torch.randn((1, 1, *target_crop.shape), device=device)

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
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        for i, t_idx in enumerate(t_schedule):
            t = torch.tensor([t_idx], device=device, dtype=torch.long)
            model_in = torch.cat([cond, x], dim=1)

            if use_sw:
                def predictor(patch_x):
                    return model(patch_x, t)

                pred_noise = sliding_window_inference(
                    inputs=model_in,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=overlap,
                    mode="gaussian"
                )
            else:
                pred_noise = model(model_in, t)

            alpha_bar_t = diffusion.alpha_cumprod[t_idx].view(1, 1, 1, 1, 1)

            if i < len(t_schedule) - 1:
                t_prev = t_schedule[i + 1]
                alpha_bar_prev = diffusion.alpha_cumprod[t_prev].view(1, 1, 1, 1, 1)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0)

            x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * pred_noise

    elapsed = time.time() - start

    # --- THE INTENSITY FIX ---
    # 1. Grab the model's Z-scored prediction
    pred_cropped = x[0, 0].detach().cpu().numpy()

    # 2. Map the Z-score exactly to a [0, 1] range safely
    pred_cropped_01 = normalize_per_case_01_np(pred_cropped)

    # 3. Upscale the [0, 1] brain back to Native Scanner space (e.g., 0 to 2500)
    pred_native = (pred_cropped_01 * (raw_max - raw_min)) + raw_min

    # Create empty background using the native target shape
    pred_full = np.zeros_like(raw_target)
    pred_full[d0:d1, h0:h1, w0:w1] = pred_native

    # 4. Mask the native background (which is purely 0.0)
    brain_mask = raw_target > raw_target.min()
    pred_full = pred_full * brain_mask

    # CRITICAL: Return `raw_target` instead of `target_np`!
    return pred_full, raw_target, cond_np, affine, header, elapsed, raw_data_dict


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

    # --- COMPILER TRICK FOR BLACKWELL ---
    if cfg["inference"].get("use_compile", False):
        if os.name == "nt":
            print("Warning: torch.compile is not fully supported on Windows. Skipping compile.")
        else:
            print("Compiling model for Linux (max-autotune). The first volume will take a few extra minutes...")
            # max-autotune optimizes the graph specifically for inference speed
            infer_model = torch.compile(infer_model, mode="max-autotune")
    # ------------------------------------

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"].get("schedule", "linear"),
        cosine_s=cfg["diffusion"].get("cosine_s", 0.008),
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    metric_bundle = ImageMetricBundle(
        compute_mae=cfg["metrics"]["compute_mae"],
        compute_psnr=cfg["metrics"]["compute_psnr"],
        compute_ssim=cfg["metrics"]["compute_ssim"],
        eval_norm=cfg.get("metrics", {}).get("eval_norm", "per_case_minmax"),
    )

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

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_name = cfg["inference"].get("run_name")
    if not run_name:
        run_name = f"{run_tag}_{ckpt_stem}"

    base_out = os.path.join(
        cfg["project"]["output_root"],
        cfg["inference"]["output_subdir"],
        cfg["model"]["name"],
        run_name,
    )
    ensure_dir(base_out)

    # --- FIX: SAFE JSON LOADING ---
    summary_path = os.path.join(base_out, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = []
    # ------------------------------

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)

        for missing_key in missing_keys:
            pred, target, cond, affine, header, elapsed, raw_data_dict = infer_single_case(
                cfg, infer_model, diffusion, case_dir, missing_key, device
            )

            pred_t = torch.tensor(pred[None, None], dtype=torch.float32)
            target_t = torch.tensor(target[None, None], dtype=torch.float32)
            mask_t = torch.tensor((target > target.min())[None, None], dtype=torch.float32)

            infer_metrics = metric_bundle(pred_t, target_t, mask=mask_t)

            item_dir = os.path.join(base_out, f"{case_id}_{missing_key}")
            ensure_dir(item_dir)

            if cfg["inference"]["save_nifti"]:
                nifti_path = os.path.join(item_dir, f"{case_id}_pred_{missing_key}.nii.gz")
                save_prediction_nifti(pred, affine, header, nifti_path)

                if cfg["inference"].get("save_composite_nifti", True):
                    composite = []
                    for k in MOD_KEYS:
                        if k == missing_key:
                            composite.append(pred)
                        else:
                            composite.append(raw_data_dict[k])

                    composite = np.stack(composite, axis=0)
                    comp_path = os.path.join(item_dir, f"{case_id}_composite_{missing_key}.nii.gz")
                    save_prediction_nifti(composite, affine, header, comp_path)

            if cfg["inference"]["save_png"]:
                png_path = os.path.join(item_dir, f"{case_id}_panel_{missing_key}.png")
                save_slice_panel(cond, target, pred, png_path, missing_key=missing_key, title=f"{case_id}")

            # --- FIX: ADDED METADATA & APPEND ---
            summary.append({
                "case_id": case_id,
                "missing_key": missing_key,
                "elapsed_sec": elapsed,
                "checkpoint": ckpt_path,
                "reverse_steps": cfg["inference"]["reverse_steps"],
                "overlap": cfg["inference"]["sliding_window"]["overlap"],
                "roi_size": cfg["inference"]["sliding_window"]["roi_size"],
                **infer_metrics,
            })
            # ------------------------------------

            print(f"[infer] case={case_id} missing={missing_key} time={elapsed:.2f}s")

            # Save incrementally inside the loop so you don't lose data if it crashes
            save_json(summary_path, summary)

    print(f"Saved inference outputs to: {base_out}")


if __name__ == "__main__":
    main()