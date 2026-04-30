import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import concurrent.futures

import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from mri_missing.config import load_config, ensure_dir, make_run_dir, save_config_copy
from mri_missing.seed import seed_everything
from mri_missing.data.dataset import BraTSDataset
from mri_missing.models.registry import build_model
from mri_missing.diffusion.scheduler import DiffusionScheduler
from mri_missing.utils.io import save_checkpoint, load_checkpoint, save_json
from mri_missing.utils.stats import StepTimer, RunTimer, get_vram_mb, reset_vram_stats, count_params
from mri_missing.metrics.image_metrics import ImageMetricBundle
from mri_missing.utils.visualization import save_history_plots
from mri_missing.losses.image_losses import CompositeSynthesisLoss
from mri_missing.utils.ema import EMA
from mri_missing.utils.cases import NUM_MODALITIES

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_optimizer(cfg, model):
    name = cfg["optim"]["name"].lower()
    lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer):
    name = cfg["scheduler"]["name"].lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["scheduler"]["t_max"],
            eta_min=cfg["scheduler"]["eta_min"],
        )
    elif name == "constant_with_warmup":
        warmup_steps = cfg["scheduler"].get("warmup_steps", 2500)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def should_stop(run_dir, cfg):
    stop_file = cfg.get("control", {}).get("stop_file", "STOP")
    stop_path = os.path.join(run_dir, stop_file)
    return os.path.exists(stop_path)


@torch.inference_mode()
def validate(model, golden_batches, scheduler, loss_fn, metric_bundle, device, use_amp, timesteps, compute_heavy=False):
    was_training = model.training
    model.eval()

    loss_accum = {}
    metric_accum = {}
    modality_metrics = {}

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if compute_heavy:
        reverse_steps = 20
        t_schedule = torch.linspace(0, timesteps - 1, reverse_steps, dtype=torch.long).tolist()
        t_schedule = list(reversed(t_schedule))

    # Each golden batch is now (cond, target, keys, target_idx, case_ids)
    for cond, target, keys, target_idx, case_ids in golden_batches:
        cond = cond.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)

        # -------------------------------------------------
        # 1. LIGHT VAL
        # -------------------------------------------------
        t = scheduler.sample_timesteps(cond.shape[0], device)
        x_t, noise = scheduler.q_sample(target, t)
        x_in = torch.cat([cond, x_t], dim=1)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            pred_noise = model(x_in, t, class_labels=target_idx)

        alpha_bar = scheduler.alpha_cumprod[t].view(-1, 1, 1, 1, 1).float()
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)

        pred_x0_1step = (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / (torch.sqrt(alpha_bar) + 1e-5)
        pred_x0_1step = torch.clamp(pred_x0_1step, min=-3.0, max=3.0)

        total_loss, loss_parts = loss_fn(
            pred_noise, noise, pred_x0_1step, target, snr=snr, t=t, timesteps=timesteps
        )

        loss_accum.setdefault("val_total_loss", []).append(total_loss.item())
        for k, v in loss_parts.items():
            loss_accum.setdefault(f"val_{k}", []).append(v)

        # -------------------------------------------------
        # 2. HEAVY VAL
        # -------------------------------------------------
        if compute_heavy:
            x_gen = torch.randn_like(target)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                for i, t_idx in enumerate(t_schedule):
                    t_tensor = torch.full((cond.shape[0],), t_idx, device=device, dtype=torch.long)
                    model_in = torch.cat([cond, x_gen], dim=1)

                    p_noise = model(model_in, t_tensor, class_labels=target_idx)

                    a_bar_t = scheduler.alpha_cumprod[t_idx].view(-1, 1, 1, 1, 1)
                    if i < len(t_schedule) - 1:
                        t_prev = t_schedule[i + 1]
                        a_bar_prev = scheduler.alpha_cumprod[t_prev].view(-1, 1, 1, 1, 1)
                    else:
                        a_bar_prev = torch.ones_like(a_bar_t)

                    p_x0 = (x_gen - torch.sqrt(1 - a_bar_t) * p_noise) / torch.sqrt(a_bar_t)
                    p_x0 = torch.clamp(p_x0, min=-3.0, max=3.0)

                    x_gen = torch.sqrt(a_bar_prev) * p_x0 + torch.sqrt(1 - a_bar_prev) * p_noise

            for b in range(target.shape[0]):
                mod = keys[b] if isinstance(keys, (list, tuple)) else keys

                p_b = x_gen[b:b + 1]
                t_b = target[b:b + 1]

                metrics = metric_bundle(p_b, t_b)

                for k, v in metrics.items():
                    metric_accum.setdefault(k, []).append(v)
                    modality_metrics.setdefault(mod, {}).setdefault(k, []).append(v)

    out = {k: sum(vals) / max(len(vals), 1) for k, vals in loss_accum.items()}

    if compute_heavy:
        for k, vals in metric_accum.items():
            out[f"val_{k}"] = sum(vals) / max(len(vals), 1)

        for mod, m_dict in modality_metrics.items():
            for k, vals in m_dict.items():
                out[f"val_{mod}_{k}"] = sum(vals) / max(len(vals), 1)

    if was_training:
        model.train()

    return out


def main():
    cfg = load_config("configs/base.yaml")

    seed_everything(cfg["seed"]["value"], cfg["seed"]["deterministic"])

    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if cfg["runtime"]["cudnn_benchmark"] and device == "cuda" and not cfg["seed"]["deterministic"]:
        torch.backends.cudnn.benchmark = True

    run_dir, run_name = make_run_dir(cfg)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    save_config_copy(cfg, os.path.join(run_dir, "config.yaml"))

    train_ds = BraTSDataset(
        cfg["data"]["train_cache_root"] if cfg["data"]["backend"] == "pt_cache" else cfg["data"]["train_root"],
        patch_size=tuple(cfg["patch"]["size"]),
        fill_value=cfg["missing_policy"]["fill_value"],
        use_presence_mask=cfg["missing_policy"]["use_presence_mask"],
        sampling_probs=cfg["missing_policy"].get("sampling_probs", None),
        backend=cfg["data"]["backend"],
        augmentation=cfg.get("augmentation", {}),
    )

    val_aug_cfg = dict(cfg.get("augmentation", {}))
    val_aug_cfg["enabled"] = False

    val_ds = BraTSDataset(
        cfg["data"]["val_cache_root"] if cfg["data"]["backend"] == "pt_cache" else cfg["data"]["val_root"],
        patch_size=tuple(cfg["patch"]["size"]),
        fill_value=cfg["missing_policy"]["fill_value"],
        use_presence_mask=cfg["missing_policy"]["use_presence_mask"],
        sampling_probs={"t1": 0.0, "t1ce": 0.5, "t2": 0.0, "flair": 0.5},
        backend=cfg["data"]["backend"],
        augmentation=val_aug_cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=cfg["loader"]["shuffle"],
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
        persistent_workers=cfg["loader"].get("persistent_workers", False),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=cfg["loader"]["pin_memory"],
        persistent_workers=cfg["loader"].get("persistent_workers", False),
    )

    print("\n" + "=" * 60)
    print("Extracting Validation Micro-Batch...")
    golden_batches = []

    # 5-tuple unpack now: cond, target, keys, target_idx, case_ids
    for cond, target, keys, target_idx, case_ids in val_loader:
        golden_batches.append((cond.clone(), target.clone(), keys, target_idx.clone(), case_ids))
        if len(golden_batches) >= 4:
            break

    print(f"Locked {len(golden_batches)} batches. Tracking the following volumes:")

    for b_idx, (_, _, keys, _, case_ids) in enumerate(golden_batches):
        for i in range(len(keys)):
            mod = keys[i] if isinstance(keys, (list, tuple)) else keys
            c_id = case_ids[i] if isinstance(case_ids, (list, tuple)) else case_ids
            print(f"  -> {c_id} [Missing: {mod.upper()}]")

    print("=" * 60 + "\n")

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

        # NEW: target-modality conditioning, opt-in via config
        if cfg["model"].get("use_target_class_embed", False):
            model_kwargs["num_class_embeds"] = NUM_MODALITIES

    model = build_model(model_name, **model_kwargs).to(device)

    if cfg["train"].get("use_compile", False):
        if os.name == "nt":
            print("Warning: torch.compile is not fully supported on Windows. Skipping compilation.")
        else:
            print("Compiling model for Linux (this will take a few minutes)...")
            model = torch.compile(model)

    ema = EMA(model, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None

    optimizer = build_optimizer(cfg, model)
    scheduler_lr = build_scheduler(cfg, optimizer)
    loss_fn = CompositeSynthesisLoss(
        use_noise_mse=cfg["loss"]["use_noise_mse"],
        noise_mse_weight=cfg["loss"]["noise_mse_weight"],
        use_mae=cfg["loss"]["use_mae"],
        mae_weight=cfg["loss"]["mae_weight"],
        use_ssim=cfg["loss"]["use_ssim"],
        ssim_weight=cfg["loss"]["ssim_weight"],
        min_snr_gamma=cfg["loss"].get("min_snr_gamma", 5.0),
    )

    # Conditioning dropout — fraction of training steps where conditioning is
    # zeroed and class label is replaced with the null index. Forces the model
    # to learn the value of conditioning explicitly. 0.0 disables.
    cond_dropout_prob = cfg["loss"].get("cond_dropout_prob", 0.0)
    use_class_embed = cfg["model"].get("use_target_class_embed", False)
    null_label_index = NUM_MODALITIES if use_class_embed else None
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["use_amp"] and device == "cuda")

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

    start_step = 0
    best_val = float("inf")
    best_ssim = -float("inf")

    if cfg["train"]["resume"]:
        ckpt = load_checkpoint(
            cfg["train"]["resume"],
            model,
            optimizer=optimizer,
            scheduler=scheduler_lr,
            scaler=scaler,
            ema=ema,
            map_location=device,
        )
        start_step = ckpt.get("step", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_ssim = ckpt.get("best_ssim", -float("inf"))

    print(f"Device: {device}")
    print(f"Run: {run_name}")
    print(f"Model: {model_name}")
    print(f"Trainable params: {count_params(model):,}")
    print(f"Target-modality embedding: {cfg['model'].get('use_target_class_embed', False)}")

    step_timer = StepTimer()
    run_timer = RunTimer()
    history = []

    model.train()
    step = start_step

    accum_steps = cfg["train"].get("accumulate_grad_batches", 1)
    optimizer.zero_grad(set_to_none=True)

    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    try:
        while step < cfg["train"]["max_steps"]:
            # 5-tuple unpack now
            for cond, target, key, target_idx, current_case_id in train_loader:
                if should_stop(run_dir, cfg):
                    print("\nStop file detected. Saving checkpoint...")
                    save_checkpoint(
                        os.path.join(ckpt_dir, "stop.pt"),
                        model, optimizer, scheduler_lr, scaler,
                        step, best_val, cfg, ema=ema
                    )
                    print("Saved stop checkpoint.")
                    return

                reset_vram_stats()
                step_timer.tic()

                cond = cond.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_idx = target_idx.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(cond.shape[0], device)
                x_t, noise = diffusion.q_sample(target, t)

                # Conditioning dropout: with prob `cond_dropout_prob`, zero
                # the cond channels (modalities + presence mask) AND swap the
                # class label to the null token. Per-sample, not per-batch,
                # so different samples in a batch may or may not be dropped.
                if cond_dropout_prob > 0.0:
                    drop_mask = (
                        torch.rand(cond.shape[0], device=device) < cond_dropout_prob
                    )
                    if drop_mask.any():
                        cond_view = cond.clone()
                        cond_view[drop_mask] = 0.0
                        cond = cond_view

                        if null_label_index is not None:
                            target_idx_view = target_idx.clone()
                            target_idx_view[drop_mask] = null_label_index
                            target_idx = target_idx_view

                x = torch.cat([cond, x_t], dim=1)

                with torch.amp.autocast("cuda", enabled=cfg["train"]["use_amp"] and device == "cuda"):
                    pred_noise = model(x, t, class_labels=target_idx)

                alpha_bar_f32 = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1, 1).float()
                snr = alpha_bar_f32 / (1.0 - alpha_bar_f32 + 1e-8)

                pred_x0 = (
                    x_t.float() - torch.sqrt(1.0 - alpha_bar_f32) * pred_noise.float()
                ) / (torch.sqrt(alpha_bar_f32) + 1e-5)

                pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0)

                loss, loss_parts = loss_fn(
                    pred_noise.float(),
                    noise.float(),
                    pred_x0,
                    target.float(),
                    snr=snr,
                    t=t,
                    timesteps=cfg["diffusion"]["timesteps"],
                )

                if not torch.isfinite(loss) or not torch.isfinite(pred_noise).all() or not torch.isfinite(pred_x0).all():
                    print(f"NaN detected at step={step}. Saving debug checkpoint...")
                    save_checkpoint(
                        os.path.join(ckpt_dir, "nan_debug.pt"),
                        model, optimizer, scheduler_lr, scaler,
                        step, best_val, cfg, ema=ema
                    )
                    break

                loss_scaled = loss / accum_steps
                scaler.scale(loss_scaled).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == cfg["train"]["max_steps"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if ema is not None:
                        ema.update(model)

                    if scheduler_lr is not None:
                        scheduler_lr.step()

                step_time = step_timer.toc()
                vram_mb = get_vram_mb() if device == "cuda" else 0.0
                lr = optimizer.param_groups[0]["lr"]
                missing_str = key[0] if isinstance(key, (list, tuple)) else key

                should_log = (step % cfg["train"]["log_every"] == 0)
                do_light_val = (step % cfg["validation"]["light_every"] == 0 and step > 0)
                do_heavy_val = (step % cfg["validation"]["heavy_every"] == 0 and step > 0)
                should_save = (step % cfg["train"]["save_every"] == 0 and step > 0)

                record = None
                if should_log or do_light_val or do_heavy_val or should_save:
                    record = {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "missing": str(missing_str),
                        "lr": float(lr),
                        "step_time_sec": float(step_time),
                        "vram_mb": float(vram_mb),
                        "run_time_sec": float(run_timer.elapsed()),
                    }
                    record.update(loss_parts)

                if should_log:
                    print(
                        f"step={step} "
                        f"loss={loss.item():.6f} "
                        f"noise_mse={loss_parts.get('noise_mse', 0):.6f} "
                        f"mae_loss={loss_parts.get('mae_loss', 0):.6f} "
                        f"ssim_loss={loss_parts.get('ssim_loss', 0):.6f} "
                        f"missing={missing_str} "
                        f"lr={lr:.6e} "
                        f"time={step_time:.3f}s "
                        f"vram={vram_mb:.1f}MB"
                    )

                if do_light_val or do_heavy_val:
                    val_model = ema.shadow if (ema is not None and cfg["ema"]["validate_with_ema"]) else model

                    val_stats = validate(
                        val_model,
                        golden_batches,
                        diffusion,
                        loss_fn,
                        metric_bundle,
                        device,
                        cfg["train"]["use_amp"],
                        timesteps=cfg["diffusion"]["timesteps"],
                        compute_heavy=do_heavy_val,
                    )

                    if record is None:
                        record = {
                            "step": step,
                            "train_loss": float(loss.item()),
                            "missing": str(missing_str),
                            "lr": float(lr),
                            "step_time_sec": float(step_time),
                            "vram_mb": float(vram_mb),
                            "run_time_sec": float(run_timer.elapsed()),
                        }
                        record.update(loss_parts)

                    record.update(val_stats)

                    prefix = "Val Heavy" if do_heavy_val else "Val Light"
                    print(f"\n{'=' * 12} [ {prefix} ] {'=' * 12}")

                    t_loss = val_stats.get('val_total_loss', 0)
                    n_mse = val_stats.get('val_noise_mse', 0)
                    m_loss = val_stats.get('val_mae_loss', 0)
                    s_loss = val_stats.get('val_ssim_loss', 0)

                    if do_heavy_val:
                        print(
                            f"Total Loss: {t_loss:>8.6f}  |  Noise MSE: {n_mse:>8.6f} ||  Global PSNR: {val_stats.get('val_psnr', 0):>8.4f}")
                        print(
                            f"MAE Loss:   {m_loss:>8.6f}  |  SSIM Loss: {s_loss:>8.6f} ||  Global SSIM: {val_stats.get('val_ssim', 0):>8.4f}")
                        print("-" * 65)

                        for mod in ["t1ce", "flair"]:
                            if f"val_{mod}_ssim" in val_stats:
                                print(
                                    f"[{mod.upper():>5}] PSNR: {val_stats[f'val_{mod}_psnr']:>7.4f}  |  SSIM: {val_stats[f'val_{mod}_ssim']:>7.4f}  |  MAE: {val_stats[f'val_{mod}_mae']:>7.4f}")
                        print(f"{'=' * 50}\n")
                    else:
                        print(f"Total Loss: {t_loss:>8.6f}  |  Noise MSE: {n_mse:>8.6f}")
                        print(f"MAE Loss:   {m_loss:>8.6f}  |  SSIM Loss: {s_loss:>8.6f}")
                        print(f"{'=' * 46}\n")

                    if val_stats["val_noise_mse"] < best_val:
                        best_val = val_stats["val_noise_mse"]
                        save_checkpoint(
                            os.path.join(ckpt_dir, "best_noise.pt"),
                            model, optimizer, scheduler_lr, scaler,
                            step, best_val, cfg, ema=ema
                        )

                    if "val_ssim" in val_stats and val_stats["val_ssim"] > best_ssim:
                        best_ssim = val_stats["val_ssim"]
                        save_checkpoint(
                            os.path.join(ckpt_dir, "best_ssim.pt"),
                            model, optimizer, scheduler_lr, scaler,
                            step, best_ssim, cfg, ema=ema
                        )
                        print(f"*** New Best Clinical SSIM: {best_ssim:.4f} saved to best_ssim.pt ***\n")

                    model.train()

                if record is not None:
                    history.append(record)

                if should_log or do_light_val or do_heavy_val or should_save:
                    io_executor.submit(save_json, os.path.join(log_dir, "history.json"), list(history))

                if do_light_val or do_heavy_val or should_save:
                    io_executor.submit(save_history_plots, list(history), os.path.join(run_dir, "plots"))

                if should_save:
                    save_checkpoint(
                        os.path.join(ckpt_dir, "latest.pt"),
                        model, optimizer, scheduler_lr, scaler,
                        step, best_val, cfg, ema=ema
                    )

                step += 1
                if step >= cfg["train"]["max_steps"]:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        save_checkpoint(
            os.path.join(ckpt_dir, "interrupt.pt"),
            model, optimizer, scheduler_lr, scaler,
            step, best_val, cfg, ema=ema
        )
        print("Saved interrupt checkpoint.")


if __name__ == "__main__":
    main()