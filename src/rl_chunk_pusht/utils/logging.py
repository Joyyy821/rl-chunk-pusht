"""CSV and W&B logging helpers."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from PIL import Image, ImageEnhance


class CsvLogger:
    """Append-only CSV logger."""

    disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = None
        self.file = None

    def log(self, row: dict[str, Any], step: int) -> None:
        row = dict(row)
        row["step"] = step
        filtered = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        if self.file is None:
            self.file = self.path.open("w")
            self.header = list(filtered.keys())
            self.file.write(",".join(self.header) + "\n")
        self.file.write(",".join([str(filtered.get(k, "")) for k in self.header]) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class ExperimentLogger:
    """Log namespaced metrics to CSV and optionally W&B."""

    def __init__(
        self,
        log_dir: Path,
        *,
        project: str,
        group: str,
        name: str,
        config: dict[str, Any],
        wandb_mode: str,
        prefixes: tuple[str, ...] = ("train", "eval", "env"),
    ):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        self.csv_loggers = {
            prefix: CsvLogger(self.log_dir / f"{prefix}.csv") for prefix in prefixes
        }
        self.wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            config=config,
            dir=tempfile.mkdtemp(),
            mode=wandb_mode,
        )

    def log(self, data: dict[str, Any], prefix: str, step: int) -> None:
        if prefix not in self.csv_loggers:
            self.csv_loggers[prefix] = CsvLogger(self.log_dir / f"{prefix}.csv")
        self.csv_loggers[prefix].log(data, step=step)
        wandb.log({f"{prefix}/{k}": v for k, v in data.items()}, step=step)

    def close(self) -> None:
        for logger in self.csv_loggers.values():
            logger.close()
        wandb.finish()


def make_exp_name(seed: int, suffix: str | None = None) -> str:
    name = f"sd{seed:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if suffix:
        name += f"_{suffix}"
    return name


def reshape_video(v: np.ndarray, n_cols: int | None = None) -> np.ndarray:
    if v.ndim == 4:
        v = v[None]
    _, t, h, w, c = v.shape
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(v.shape[0])))
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c), dtype=v.dtype)), axis=0)
    n_rows = v.shape[0] // n_cols
    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    return np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))


def get_wandb_video(renders: list[np.ndarray], n_cols: int | None = None, fps: int = 15):
    max_length = max(len(render) for render in renders)
    padded = []
    for render in renders:
        assert render.dtype == np.uint8
        final_image = Image.fromarray(render[-1])
        final_frame = np.array(ImageEnhance.Brightness(final_image).enhance(0.5))
        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        video = np.concatenate([render, pad], axis=0)
        padded.append(np.pad(video, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=0))
    video = reshape_video(np.array(padded), n_cols)
    return wandb.Video(video, fps=fps, format="mp4")
