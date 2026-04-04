from pathlib import Path
from datetime import datetime
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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