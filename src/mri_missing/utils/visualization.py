import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _norm_for_plot(vol):
    """Safely squeezes any array into exactly [0, 1] exclusively for matplotlib."""
    v_min, v_max = vol.min(), vol.max()
    if v_max - v_min < 1e-6:
        return np.zeros_like(vol)
    return (vol - v_min) / (v_max - v_min)


def save_history_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Training loss
    # -------------------------
    steps = [h["step"] for h in history if "train_loss" in h]
    train_loss = [h["train_loss"] for h in history if "train_loss" in h]

    if steps:
        plt.figure()
        plt.plot(steps, train_loss, label="train_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_loss.png"))
        plt.close()

    # -------------------------
    # Validation Loss Components (Log Scale)
    # -------------------------
    val_steps = [h["step"] for h in history if "val_total_loss" in h]
    if val_steps:
        plt.figure(figsize=(10, 6))

        # Define all the possible loss parts we want to track
        loss_keys = ["val_total_loss", "val_noise_mse", "val_mae_loss", "val_ssim_loss", "val_texture_loss"]

        for key in loss_keys:
            # Check if this specific loss exists in the history at all
            if any(key in h for h in history):
                vals = [h.get(key, None) for h in history if "val_total_loss" in h]

                # Filter out None gaps if a metric was added mid-run
                valid_steps = [s for s, v in zip(val_steps, vals) if v is not None]
                valid_vals = [v for v in vals if v is not None]

                if valid_vals:
                    plt.plot(valid_steps, valid_vals, label=key)

        plt.xlabel("Step")
        plt.ylabel("Loss (Log Scale)")
        plt.yscale("log")  # Prevents small losses from being squashed by total_loss
        plt.title("Validation Loss Components")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_loss_components.png"))
        plt.close()

    # -------------------------
    # Validation Structural Metrics
    # -------------------------
    heavy_steps = [h["step"] for h in history if "val_mae" in h]
    mae_vals = [h["val_mae"] for h in history if "val_mae" in h]
    psnr_vals = [h["val_psnr"] for h in history if "val_psnr" in h]
    ssim_vals = [h["val_ssim"] for h in history if "val_ssim" in h]

    if heavy_steps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(heavy_steps, mae_vals, color="red")
        axs[0].set_title("Validation MAE (Lower is better)")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(heavy_steps, psnr_vals, color="green")
        axs[1].set_title("Validation PSNR (Higher is better)")
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(heavy_steps, ssim_vals, color="blue")
        axs[2].set_title("Validation SSIM (Higher is better)")
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_metrics.png"))
        plt.close()


MOD_LABELS = ["T1", "T1ce", "T2", "FLAIR"]


def _to_display_01(x):
    # For strict-bound normalized MRI data in [-1, 1]
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def save_slice_panel(cond, target, pred, out_path, missing_key=None, title="sample"):
    """
    cond: [4 or 8, D, H, W] in normalized space
    target: [D, H, W] in normalized space
    pred: [D, H, W] in normalized space
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Use the last spatial axis as slice index, matching existing convention
    z = target.shape[2] // 2

    cond_vis = cond[:4].copy()
    target_vis = target.copy()
    pred_vis = pred.copy()

    # ---Dynamically scale all arrays for safe rendering ---
    cond_vis = _norm_for_plot(cond[:4])
    target_vis = _norm_for_plot(target)
    pred_vis = _norm_for_plot(pred)

    diff = np.abs(target_vis[:, :, z] - pred_vis[:, :, z])
    diff_vmax = max(np.percentile(diff, 99), 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")

    for i in range(4):
        axes[0, i].imshow(
            cond_vis[i, :, :, z],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="none",
        )
        label = MOD_LABELS[i]
        if missing_key is not None:
            expected_missing_idx = {
                "t1": 0,
                "t1ce": 1,
                "t2": 2,
                "flair": 3,
            }[missing_key]
            if i == expected_missing_idx:
                label += " (missing slot)"
        axes[0, i].set_title(label)
        axes[0, i].axis("off")

    axes[1, 0].imshow(target_vis[:, :, z], cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_vis[:, :, z], cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    axes[1, 1].set_title("Prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(diff, cmap="hot", vmin=0.0, vmax=diff_vmax, interpolation="none")
    axes[1, 2].set_title("Abs Diff")
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    meta = f"Missing: {missing_key}" if missing_key is not None else ""
    axes[1, 3].text(0.05, 0.8, meta, fontsize=12)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def save_slice_panel(cond, target, pred, out_path, missing_key=None, title="sample"):
    """
    cond: [4 or 8, D, H, W]
    target: [D, H, W]
    pred: [D, H, W]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    z = target.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")

    MOD_LABELS = ["T1", "T1ce", "T2", "FLAIR"]

    # --- 1. Plot Conditions (Auto-scaled for visibility) ---
    for i in range(4):
        slice_2d = cond[i, :, :, z]
        axes[0, i].imshow(slice_2d, cmap="gray")

        label = MOD_LABELS[i]
        if missing_key is not None:
            expected_missing_idx = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}.get(missing_key, -1)
            if i == expected_missing_idx:
                label += " (missing slot)"
        axes[0, i].set_title(label)
        axes[0, i].axis("off")

    # --- 2. Lock Physical Bounds for Prediction & Target ---
    # Find the true physical bounds of the Target slice
    v_min = target[:, :, z].min()
    v_max = target[:, :, z].max()

    # Plot Target
    axes[1, 0].imshow(target[:, :, z], cmap="gray", vmin=v_min, vmax=v_max)
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    # Plot Prediction (LOCKED to Target bounds)
    axes[1, 1].imshow(pred[:, :, z], cmap="gray", vmin=v_min, vmax=v_max)
    axes[1, 1].set_title("Prediction")
    axes[1, 1].axis("off")

    # --- 3. Plot Absolute Difference ---
    diff = np.abs(target[:, :, z] - pred[:, :, z])
    # We lock the heat map vmax to a fraction of the physical max so small errors don't look like explosions
    axes[1, 2].imshow(diff, cmap="hot", vmin=0, vmax=v_max * 0.5)
    axes[1, 2].set_title("Abs Diff")
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    meta = f"Missing: {missing_key}" if missing_key is not None else ""
    axes[1, 3].text(0.05, 0.8, meta, fontsize=12)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def save_paper_grid_panel(targets_dict, preds_dict, out_path, title="sample"):
    """
    Creates a 2x4 paper-ready grid.
    Top row: Targets (T1, T1ce, T2, FLAIR). Middle row: Predictions. Bottom row: Absolute Differences.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor="white")
    MOD_LABELS = ["t1", "t1ce", "t2", "flair"]
    DISPLAY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

    for col, mod in enumerate(MOD_LABELS):
        # We assume the user passed a 3D volume, so we grab the middle Z-slice
        target_slice = targets_dict[mod][:, :, targets_dict[mod].shape[2] // 2]
        pred_slice = preds_dict[mod][:, :, preds_dict[mod].shape[2] // 2]

        v_min, v_max = target_slice.min(), target_slice.max()

        # --- Top Row: Target ---
        axes[0, col].imshow(target_slice, cmap="gray", vmin=v_min, vmax=v_max)
        axes[0, col].set_title(f"Target: {DISPLAY_NAMES[col]}", fontsize=14)
        axes[0, col].axis("off")

        # --- Middle Row: Prediction ---
        axes[1, col].imshow(pred_slice, cmap="gray", vmin=v_min, vmax=v_max)
        axes[1, col].set_title(f"Predicted: {DISPLAY_NAMES[col]}", fontsize=14)
        axes[1, col].axis("off")

        # --- Bottom Row (Absolute Difference) ---
        diff = np.abs(target_slice - pred_slice)

        # We lock the heat map vmax to a fraction (0.5) of the physical max
        # so small errors don't look like explosions, matching the slice panel logic
        axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=v_max * 0.5)
        axes[2, col].set_title(f"Abs Diff: {DISPLAY_NAMES[col]}", fontsize=14)
        axes[2, col].axis("off")

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_augmentation_panel(dataset, out_path, num_samples=5):
    """Generate a multi-row panel showing the same dataset entry sampled
    `num_samples` times. Because random augmentations (flips, rotations,
    intensity shifts, missing-modality selection) are re-rolled on every
    __getitem__ call, each row will look slightly different — which makes
    this a quick visual sanity check that augmentation is actually firing
    and not destroying the data.

    Args:
        dataset: a BraTSDataset (NOT a DataLoader). We index it directly so
            we control how many fresh draws we take.
        out_path: where to write the PNG.
        num_samples: number of rows in the panel (each row is one fresh
            __getitem__ call on the same index).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if len(dataset) == 0:
        print("[save_augmentation_panel] dataset is empty, skipping")
        return

    # Always pull the same case so what changes row-to-row is purely
    # the augmentation (and the random missing-modality choice).
    idx = 0

    fig, axes = plt.subplots(
        num_samples, 5,
        figsize=(20, 4 * num_samples),
        facecolor="white",
    )
    # Handle the num_samples=1 case where axes is 1D
    if num_samples == 1:
        axes = axes[None, :]

    case_id_seen = None

    for row in range(num_samples):
        item = dataset[idx]

        # Dataset returns (cond, target, target_key, target_idx, case_id).
        # Be defensive in case someone shortened it later.
        if len(item) == 5:
            cond, target, target_key, _target_idx, case_id = item
        elif len(item) == 4:
            cond, target, target_key, _target_idx = item
            case_id = "?"
        else:
            cond, target, target_key = item[:3]
            case_id = "?"

        # Convert to numpy and drop any leading singleton channel on target.
        cond_np = cond.cpu().numpy() if hasattr(cond, "cpu") else np.asarray(cond)
        target_np = target.cpu().numpy() if hasattr(target, "cpu") else np.asarray(target)
        if target_np.ndim == 4 and target_np.shape[0] == 1:
            target_np = target_np[0]

        if case_id_seen is None:
            case_id_seen = case_id

        # Pick the middle slice along whichever axis is the smallest spatial
        # dim (matches the rest of the file's convention).
        z = target_np.shape[-1] // 2

        # Show all four conditioning modalities + the target on each row.
        for col in range(4):
            slc = cond_np[col, :, :, z]
            slc = _norm_for_plot(slc)
            axes[row, col].imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
            label = MOD_LABELS[col]
            if target_key == ["t1", "t1ce", "t2", "flair"][col]:
                label += " (missing slot)"
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].axis("off")

        target_vis = _norm_for_plot(target_np)
        axes[row, 4].imshow(target_vis[:, :, z], cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        axes[row, 4].set_title(f"Target ({target_key})", fontsize=10)
        axes[row, 4].axis("off")

        # Row label on the left margin
        axes[row, 0].set_ylabel(f"Aug #{row + 1}", fontsize=11)

    fig.suptitle(
        f"Augmentation Preview — case {case_id_seen} (idx={idx})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()