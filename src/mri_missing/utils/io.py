import os
import json
import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, best_val=None, cfg=None, ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "ema_state": ema.state_dict() if ema is not None else None,
            "step": step,
            "best_val": best_val,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if ema is not None and ckpt.get("ema_state") is not None:
        ema.load_state_dict(ckpt["ema_state"])

    return ckpt


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)