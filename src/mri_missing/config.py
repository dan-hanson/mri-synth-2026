from pathlib import Path
from datetime import datetime
import os
import yaml


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
# We walk up from this file looking for a "marker" that identifies the repo
# root. Any of these are good signals: .git, pyproject.toml, setup.py, or a
# top-level "configs" folder. The first match wins. This means anyone who
# clones the repo gets the right paths regardless of where on disk the repo
# lives.
_PROJECT_ROOT_MARKERS = (".git", "pyproject.toml", "setup.py", "configs")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until we find a directory containing one of
    the marker files/folders. Falls back to the current working directory
    if nothing is found (so the function never raises during normal use).
    """
    start = (start or Path(__file__)).resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        for marker in _PROJECT_ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate

    # Fallback: current working directory. Better than raising and breaking
    # someone's notebook.
    return Path.cwd()


# Cached at import time so every consumer agrees on the same root.
PROJECT_ROOT = find_project_root()


def project_path(*parts: str) -> Path:
    """Join paths against the project root."""
    return PROJECT_ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
# Any value under one of these dotted keys that is a *relative* path will be
# resolved against PROJECT_ROOT during load_config(). Absolute paths are left
# alone, so power users can still point at data sitting on a separate drive.
_PATH_KEYS = {
    "project.output_root",
    "data.train_root",
    "data.val_root",
    "data.train_cache_root",
    "data.val_cache_root",
    "inference.data_root",
    "inference.run_dir",
    "inference.checkpoint_path",
    "train.resume",
}


def _resolve_path(value):
    """Turn a relative path string into an absolute path string against
    PROJECT_ROOT. Leaves None, empty strings, and absolute paths alone."""
    if value is None or value == "":
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _walk_and_resolve(cfg: dict, prefix: str = "") -> None:
    """Recursively walk the config dict, resolving any value whose dotted
    key matches one in _PATH_KEYS."""
    for key, val in list(cfg.items()):
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            _walk_and_resolve(val, dotted)
        elif dotted in _PATH_KEYS:
            cfg[key] = _resolve_path(val)


def load_config(path: str | None = None):
    """Load YAML config and auto-resolve any relative paths against the
    project root. If `path` is None or relative, it is resolved against the
    project root too — so `load_config()` works from anywhere."""
    if path is None:
        path = "configs/base.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _walk_and_resolve(cfg)
    return cfg


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_run_dir(cfg):
    output_root = cfg["project"]["output_root"]
    run_name = cfg["project"]["run_name"]

    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = cfg["model"]["name"]
        run_name = f"{stamp}_{model_name}"

    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir), run_name


def save_config_copy(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)