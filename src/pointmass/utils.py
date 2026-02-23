from __future__ import annotations

import random
from pathlib import Path
from typing import Dict


def seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str):
    import torch

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def ensure_dirs(paths_cfg: Dict[str, str]) -> None:
    for key in ("data_dir", "ckpt_dir", "log_dir", "video_dir"):
        if key in paths_cfg:
            Path(paths_cfg[key]).mkdir(parents=True, exist_ok=True)


def path_from_cfg(base_dir: str, filename: str) -> Path:
    return Path(base_dir) / filename
