import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import _path_bootstrap  # noqa: F401  (must come before mri_missing.*)

from mri_missing.config import load_config, ensure_dir
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.utils.io import load_checkpoint
from mri_missing.utils.cases import MOD_KEYS
from mri_missing.utils.ema import EMA
from mri_missing.metrics.image_metrics import ImageMetricBundle
from diffusers import DDIMScheduler
from monai.inferers import sliding_window_inference
from skimage.exposure import match_histograms

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@torch.inference_mode()
def eval_cached_case(cfg, model, diffusion, pt_data, missing_key, device):
    """Highly optimized inference loop for cached .pt tensors."""
    target_t = pt_data[missing_key].to(device)

    # 1. Prepare Condition (Mask out the missing modality)
    cond_list = []
    mask_list = []
    fill_value = cfg["missing_policy"]["fill_value"]

    for k in MOD_KEYS:
        if k == missing_key:
            cond_list.append(torch.full_like(pt_data[k], fill_value))
            mask_list.append(torch.zeros_like(pt_data[k]))
        else:
            cond_list.append(pt_data[k])
            mask_list.append(torch.ones_like(pt_data[k]))

    cond = torch.stack(cond_list, dim=0)
    if cfg["missing_policy"]["use_presence_mask"]:
        mask = torch.stack(mask_list, dim=0)
        cond = torch.cat([cond, mask], dim=0)

    cond = cond.unsqueeze(0).to(device)  # [1, 8, D, H, W]

    # 2. Build Tissue Support Mask for Metrics
    available_idx = [i for i, k in enumerate(MOD_KEYS) if k != missing_key]
    support_masks = [(cond[0, i] > -0.999) for i in available_idx]
    eval_mask = torch.stack(support_masks, dim=0).any(dim=0)

    # 3. Setup Scheduler & CFG
    trained_betas_np = diffusion.betas.detach().cpu().numpy()
    scheduler = DDIMScheduler(
        trained_betas=trained_betas_np,
        beta_schedule="scaled_linear",
        clip_sample=True,
        clip_sample_range=5.0,
        thresholding=False,
    )
    scheduler.set_timesteps(cfg["inference"]["reverse_steps"])

    # --- FETCH PER-MODALITY CFG ---
    cfg_dict = cfg["inference"].get("cfg_guidance_scale", {})
    guidance_scale = cfg_dict.get(missing_key, 2.0) if isinstance(cfg_dict, dict) else cfg_dict

    roi_size = tuple(int(v) for v in cfg["inference"]["sliding_window"]["roi_size"])
    sw_batch_size = cfg["inference"]["sliding_window"]["sw_batch_size"]
    overlap = cfg["inference"]["sliding_window"]["overlap"]
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    mod_to_idx = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    target_idx = torch.tensor([mod_to_idx[missing_key]], device=device, dtype=torch.long)

    x = torch.randn((1, 1, *cond.shape[2:]), device=device)

    total_train_steps = cfg["diffusion"]["timesteps"]
    fade_threshold = total_train_steps * 0.20

    for t_idx in scheduler.timesteps:
        t_scalar = int(t_idx.item()) if torch.is_tensor(t_idx) else int(t_idx)
        t = torch.tensor([t_scalar], device=device, dtype=torch.long)
        current_guidance = guidance_scale if t_scalar > fade_threshold else 1.0

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if current_guidance > 1.0:
                cond_zero = torch.zeros_like(cond)
                cond_zero[:, :4] = -1.0
                cond_zero[:, 4:8] = 0.0

                def p_cond(patch):
                    return model(patch, t.expand(patch.shape[0]), class_labels=target_idx.expand(patch.shape[0]))

                def p_uncond(patch):
                    return model(patch, t.expand(patch.shape[0]), class_labels=target_idx.expand(patch.shape[0]))

                noise_cond = sliding_window_inference(torch.cat([cond, x], dim=1), roi_size, sw_batch_size, p_cond,
                                                      overlap, mode="gaussian")
                noise_uncond = sliding_window_inference(torch.cat([cond_zero, x], dim=1), roi_size, sw_batch_size,
                                                        p_uncond, overlap, mode="gaussian")

                pred_noise = noise_uncond + current_guidance * (noise_cond - noise_uncond)
            else:
                def predictor(patch):
                    return model(patch, t.expand(patch.shape[0]), class_labels=target_idx.expand(patch.shape[0]))

                pred_noise = sliding_window_inference(torch.cat([cond, x], dim=1), roi_size, sw_batch_size, predictor,
                                                      overlap, mode="gaussian")

        x = scheduler.step(pred_noise.float(), t_idx, x.float()).prev_sample

    pred_full = x[0, 0].detach().float().cpu().numpy()
    target_np = target_t.cpu().numpy()
    eval_mask_np = eval_mask.cpu().numpy()

    pred_full = np.where(eval_mask_np, pred_full, -1.0)

    # 4. Histogram Matching
    mask_bool = eval_mask_np.astype(bool)
    matched_tissue = match_histograms(pred_full[mask_bool], target_np[mask_bool])
    pred_full[mask_bool] = matched_tissue
    pred_full = np.clip(pred_full, -1.0, 1.0)

    # 5. Calculate Metrics
    pred_t = torch.tensor(pred_full[None, None], dtype=torch.float32)
    t_t = torch.tensor(target_np[None, None], dtype=torch.float32)
    m_t = torch.tensor(eval_mask_np[None, None], dtype=torch.float32)

    metric_bundle = ImageMetricBundle(compute_mae=True, compute_psnr=True, compute_ssim=True,
                                      eval_norm=cfg["metrics"]["eval_norm"])
    metrics = metric_bundle(pred_t, t_t, mask=m_t)

    # Parallel full volume metrics
    if cfg["metrics"].get("compute_full_volume", False):
        full_metrics = metric_bundle(pred_t, t_t, mask=None)
        for k, v in full_metrics.items():
            metrics[f"full_{k}"] = v

    return metrics


def build_eval_model(cfg, device):
    """
    Build the model from config for evaluation.

    NOTE: This must stay in sync with the kwarg-mapping in train.py and infer.py.
    Missing fields here (e.g. resblock_updown) silently change layer shapes and
    break checkpoint loading. The cleanest long-term fix is to lift this into a
    shared helper in models/registry.py and call it from train/infer/eval.
    """
    model_name = cfg["model"]["name"].lower()
    model_kwargs = {
        "in_channels": cfg["model"]["in_channels"],
        "out_channels": cfg["model"]["out_channels"],
    }

    if model_name == "unet":
        model_kwargs["base_ch"] = cfg["model"]["base_ch"]

    elif model_name == "convnext3d":
        model_kwargs["base_dim"] = cfg["model"]["base_dim"]

    elif model_name == "monai_diffusion":
        model_kwargs["channels"] = tuple(cfg["model"]["channels"])
        model_kwargs["attention_levels"] = tuple(cfg["model"]["attention_levels"])
        model_kwargs["num_res_blocks"] = cfg["model"]["num_res_blocks"]
        model_kwargs["num_head_channels"] = cfg["model"]["num_head_channels"]
        model_kwargs["norm_num_groups"] = cfg["model"].get("norm_num_groups", 32)
        model_kwargs["norm_eps"] = cfg["model"].get("norm_eps", 1e-6)
        model_kwargs["resblock_updown"] = cfg["model"].get("resblock_updown", False)
        model_kwargs["transformer_num_layers"] = cfg["model"].get("transformer_num_layers", 1)
        model_kwargs["dropout_cattn"] = cfg["model"].get("dropout_cattn", 0.0)
        model_kwargs["use_flash_attention"] = cfg["model"].get("use_flash_attention", False)
        # Match train.py's default (False). Config sets it true for this project.
        if cfg["model"].get("use_target_class_embed", False):
            model_kwargs["num_class_embeds"] = 5  # 0=T1, 1=T1ce, 2=T2, 3=FLAIR, 4=NULL

    elif model_name == "swin":
        model_kwargs["img_size"] = tuple(cfg["patch"]["size"])
        model_kwargs["feature_size"] = cfg["model"].get("feature_size", 48)
        model_kwargs["time_embed_dim"] = cfg["model"].get("time_embed_dim", 16)
        model_kwargs["depths"] = tuple(cfg["model"].get("depths", [2, 2, 2, 2]))
        model_kwargs["num_heads"] = tuple(cfg["model"].get("num_heads", [4, 4, 8, 16]))
        model_kwargs["window_size"] = tuple(cfg["model"].get("window_size", [4, 4, 4]))
        model_kwargs["drop_rate"] = cfg["model"].get("drop_rate", 0.0)
        model_kwargs["attn_drop_rate"] = cfg["model"].get("attn_drop_rate", 0.0)

    elif model_name == "swin_ddpm":
        model_kwargs["img_size"] = tuple(cfg["patch"]["size"])
        model_kwargs["feature_size"] = cfg["model"].get("feature_size", 48)
        model_kwargs["depths"] = tuple(cfg["model"].get("depths", [2, 2, 2, 2]))
        model_kwargs["num_heads"] = tuple(cfg["model"].get("num_heads", [4, 4, 8, 16]))
        model_kwargs["window_size"] = tuple(cfg["model"].get("window_size", [4, 4, 4]))
        model_kwargs["drop_rate"] = cfg["model"].get("drop_rate", 0.0)
        model_kwargs["attn_drop_rate"] = cfg["model"].get("attn_drop_rate", 0.0)
        model_kwargs["use_checkpoint"] = cfg["model"].get("use_checkpoint", True)
        model_kwargs["spatial_dims"] = cfg["model"].get("spatial_dims", 3)

    return build_model(model_name, **model_kwargs).to(device)


def main():
    cfg = load_config("configs/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup Output ---
    run_name = cfg["inference"].get("run_name", "eval_run")
    base_out = os.path.join(cfg["project"]["output_root"], "evaluations",
                            f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(base_out)

    # --- Load Model ---
    ckpt_path = cfg["inference"].get("checkpoint_path") or os.path.join(cfg["inference"]["run_dir"], "checkpoints",
                                                                        cfg["inference"]["checkpoint_name"])
    model = build_eval_model(cfg, device)

    ema = EMA(model, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None
    load_checkpoint(ckpt_path, model, ema=ema, map_location=device)
    infer_model = ema.shadow if ema else model
    infer_model.eval()

    diffusion = DiffusionScheduler(timesteps=cfg["diffusion"]["timesteps"]).to(device)

    # --- Load Validation Cache ---
    val_cache_dir = cfg["data"]["val_cache_root"]
    pt_files = [f for f in os.listdir(val_cache_dir) if f.endswith(".pt")]

    # --- DRY RUN LIMITER ---
    max_cases = cfg["inference"].get("max_eval_cases", 0)
    if max_cases > 0:
        pt_files = pt_files[:max_cases]
        print(f"Limiting evaluation to {max_cases} cases for dry-run testing.")

    summary_data = []

    print(f"Starting Evaluation on {len(pt_files)} cached cases...")
    for pt_file in tqdm(pt_files):
        pt_path = os.path.join(val_cache_dir, pt_file)
        data = torch.load(pt_path, map_location="cpu")
        case_id = data["case_id"]

        for mod in MOD_KEYS:
            metrics = eval_cached_case(cfg, infer_model, diffusion, data, mod, device)
            summary_data.append({
                "case_id": case_id,
                "missing_modality": mod,
                **metrics
            })

    # --- Generate Paper Outputs ---
    df = pd.DataFrame(summary_data)

    # 1. Save Full Raw Log
    df.to_csv(os.path.join(base_out, "full_eval_log.csv"), index=False)

    # 2. Generate Paper Summary Table (Means and STDs per modality)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_table = df.groupby("missing_modality")[numeric_cols].agg(['mean', 'std']).round(4)
    summary_table.to_csv(os.path.join(base_out, "paper_summary_table.csv"))

    print(f"\nEvaluation Complete! Results saved to {base_out}")
    print("\n--- Paper Summary Table Preview ---")
    print(summary_table)


if __name__ == "__main__":
    main()