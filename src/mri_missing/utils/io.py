import os
import json
import torch
from collections.abc import Mapping


def _strip_orig_mod_prefix(state_dict):
    """
    Removes the '_orig_mod.' prefix that can appear when saving checkpoints
    from a torch.compile()-wrapped model.
    """
    if not isinstance(state_dict, Mapping):
        return state_dict

    cleaned = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        else:
            cleaned[k] = v
    return cleaned


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


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)

    model_state = ckpt.get("model_state")
    if model_state is None:
        raise KeyError(f"Checkpoint at {path} does not contain 'model_state'.")

    model_state = _strip_orig_mod_prefix(model_state)
    model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if ema is not None and ckpt.get("ema_state") is not None:
        ema_state = _strip_orig_mod_prefix(ckpt["ema_state"])
        ema.load_state_dict(ema_state)

    return ckpt


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)