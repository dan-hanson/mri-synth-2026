import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(history, key):
    """Pull a (steps, values) pair from history, dropping records that lack the key."""
    steps = []
    vals = []
    for r in history:
        if key in r and r[key] is not None:
            steps.append(r["step"])
            vals.append(r[key])
    return steps, vals


def _save_plot(out_path, title, ylabel, series_dict, ylim=None, xlabel="step"):
    """series_dict: {label: (steps, values)}. Skips empty series silently."""
    plt.figure(figsize=(8, 5))
    plotted_any = False
    for label, (steps, values) in series_dict.items():
        if len(steps) == 0:
            continue
        plt.plot(steps, values, label=label, marker="o", markersize=2, linewidth=1.0)
        plotted_any = True
    if not plotted_any:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def save_history_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Training loss --------------------------------------------------
    train_steps, train_loss = _series(history, "train_loss")
    _save_plot(
        os.path.join(out_dir, "train_loss.png"),
        title="Training Loss",
        ylabel="loss",
        series_dict={"train_loss": (train_steps, train_loss)},
    )

    # --- 2. Validation loss components ------------------------------------
    # Stack the loss components on a single plot.
    _save_plot(
        os.path.join(out_dir, "val_loss_components.png"),
        title="Validation Loss Components",
        ylabel="loss",
        series_dict={
            "val_total_loss": _series(history, "val_total_loss"),
            "val_noise_mse": _series(history, "val_noise_mse"),
            "val_mae_loss": _series(history, "val_mae_loss"),
            "val_healthy_ssim_loss": _series(history, "val_healthy_ssim_loss"),
            "val_tumor_ssim_loss": _series(history, "val_tumor_ssim_loss"),
        },
    )

    # --- 3. Validation MAE ------------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_mae.png"),
        title="Validation MAE (lower better)",
        ylabel="MAE",
        series_dict={
            "global": _series(history, "val_mae"),
            "t1ce": _series(history, "val_t1ce_mae"),
            "flair": _series(history, "val_flair_mae"),
            "t1": _series(history, "val_t1_mae"),
            "t2": _series(history, "val_t2_mae"),
        },
    )

    # --- 4. Validation PSNR -----------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_psnr.png"),
        title="Validation PSNR (higher better)",
        ylabel="PSNR (dB)",
        series_dict={
            "global": _series(history, "val_psnr"),
            "t1ce": _series(history, "val_t1ce_psnr"),
            "flair": _series(history, "val_flair_psnr"),
            "t1": _series(history, "val_t1_psnr"),
            "t2": _series(history, "val_t2_psnr"),
        },
    )

    # --- 5. Validation SSIM -----------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_ssim.png"),
        title="Validation SSIM (higher better)",
        ylabel="SSIM",
        ylim=(0.0, 1.0),
        series_dict={
            "global": _series(history, "val_ssim"),
            "t1ce": _series(history, "val_t1ce_ssim"),
            "flair": _series(history, "val_flair_ssim"),
            "t1": _series(history, "val_t1_ssim"),
            "t2": _series(history, "val_t2_ssim"),
        },
    )

    # --- 6. Tumor-region SSIM ---------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_tumor_ssim.png"),
        title="Validation Tumor SSIM (higher better)",
        ylabel="SSIM",
        ylim=(0.0, 1.0),
        series_dict={
            "global": _series(history, "val_tumor_ssim"),
            "t1ce": _series(history, "val_t1ce_tumor_ssim"),
            "flair": _series(history, "val_flair_tumor_ssim"),
        },
    )

    # --- 7. Healthy-region SSIM -------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_healthy_ssim.png"),
        title="Validation Healthy-Tissue SSIM (higher better)",
        ylabel="SSIM",
        ylim=(0.0, 1.0),
        series_dict={
            "global": _series(history, "val_healthy_ssim"),
            "t1ce": _series(history, "val_t1ce_healthy_ssim"),
            "flair": _series(history, "val_flair_healthy_ssim"),
        },
    )

    # --- 8. Tumor-region PSNR ---------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_tumor_psnr.png"),
        title="Validation Tumor PSNR (higher better)",
        ylabel="PSNR (dB)",
        series_dict={
            "global": _series(history, "val_tumor_psnr"),
            "t1ce": _series(history, "val_t1ce_tumor_psnr"),
            "flair": _series(history, "val_flair_tumor_psnr"),
        },
    )

    # --- 9. Healthy-region PSNR -------------------------------------------
    _save_plot(
        os.path.join(out_dir, "val_healthy_psnr.png"),
        title="Validation Healthy-Tissue PSNR (higher better)",
        ylabel="PSNR (dB)",
        series_dict={
            "global": _series(history, "val_healthy_psnr"),
            "t1ce": _series(history, "val_t1ce_healthy_psnr"),
            "flair": _series(history, "val_flair_healthy_psnr"),
        },
    )


# ---------------------------------------------------------------------------
# Slice panel for inference visualization (unchanged behavior)
# ---------------------------------------------------------------------------
MOD_LABELS = ["T1", "T1ce", "T2", "FLAIR"]


def save_slice_panel(cond, target, pred, out_path, missing_key=None, title="sample"):
    """
    cond: [4 or 8, D, H, W]  -- only first 4 channels are displayed
    target: [D, H, W]
    pred: [D, H, W]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    z = target.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        axes[0, i].imshow(cond[i, :, :, z], cmap="gray")
        label = MOD_LABELS[i]
        if missing_key is not None:
            expected_missing_idx = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}[missing_key]
            if i == expected_missing_idx:
                label += " (missing slot)"
        axes[0, i].set_title(label)
        axes[0, i].axis("off")

    axes[1, 0].imshow(target[:, :, z], cmap="gray")
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred[:, :, z], cmap="gray")
    axes[1, 1].set_title("Prediction")
    axes[1, 1].axis("off")

    diff = np.abs(target[:, :, z] - pred[:, :, z])
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Abs Diff")
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    meta = f"Missing: {missing_key}" if missing_key is not None else ""
    axes[1, 3].text(0.05, 0.8, meta, fontsize=12)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()