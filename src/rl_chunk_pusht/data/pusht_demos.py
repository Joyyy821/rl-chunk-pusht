"""Optional Push-T demonstration loading for warm-start ablations."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import zarr

from rl_chunk_pusht.envs.pusht import WORKSPACE_SIZE
from rl_chunk_pusht.utils.datasets import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


def download_pusht(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def normalize_state(states: np.ndarray) -> np.ndarray:
    angles = states[:, 4]
    return np.column_stack(
        [
            states[:, 0] / WORKSPACE_SIZE,
            states[:, 1] / WORKSPACE_SIZE,
            states[:, 2] / WORKSPACE_SIZE,
            states[:, 3] / WORKSPACE_SIZE,
            np.sin(angles),
            np.cos(angles),
        ]
    ).astype(np.float32)


def normalize_position_actions(actions: np.ndarray) -> np.ndarray:
    return np.clip(actions / (WORKSPACE_SIZE * 0.5) - 1.0, -1.0, 1.0).astype(np.float32)


def infer_velocity_actions(
    states: np.ndarray,
    episode_ends: np.ndarray,
    *,
    max_speed_px_s: float,
    control_hz: float = 10.0,
) -> np.ndarray:
    """Infer normalized end-effector velocities from adjacent demo states."""

    velocities = np.zeros((len(states), 2), dtype=np.float32)
    starts = np.concatenate(([0], episode_ends[:-1]))
    for start, end in zip(starts, episode_ends, strict=True):
        if end - start <= 1:
            continue
        delta = states[start + 1 : end, :2] - states[start : end - 1, :2]
        velocities[start : end - 1] = delta * control_hz / max_speed_px_s
        velocities[end - 1] = velocities[end - 2]
    return np.clip(velocities, -1.0, 1.0).astype(np.float32)


def transition_indices(episode_ends: np.ndarray) -> np.ndarray:
    starts = np.concatenate(([0], episode_ends[:-1]))
    idxs: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        idxs.extend(range(start, max(start, end - 1)))
    return np.asarray(idxs, dtype=np.int64)


def make_demo_dataset(
    zarr_path: Path,
    *,
    env_mode: Literal["position_target", "velocity_kinematic"],
    max_speed_px_s: float = 120.0,
) -> Dataset:
    """Create a Dataset compatible with the online replay buffer.

    For `velocity_kinematic`, actions are inferred from finite differences of
    the demonstrated agent position. This is intended only for optional prior
    ablations because the original demos were collected with position targets.
    """

    states, position_actions, episode_ends = load_pusht_zarr(zarr_path)
    observations = normalize_state(states)
    if env_mode == "position_target":
        actions = normalize_position_actions(position_actions)
    elif env_mode == "velocity_kinematic":
        actions = infer_velocity_actions(states, episode_ends, max_speed_px_s=max_speed_px_s)
    else:
        raise ValueError(f"Demo dataset does not support env_mode={env_mode}")

    idxs = transition_indices(episode_ends)
    terminals = np.zeros(len(idxs), dtype=np.float32)
    starts = np.concatenate(([0], episode_ends[:-1]))
    terminal_idx_set = {int(end - 2) for start, end in zip(starts, episode_ends, strict=True) if end - start > 1}
    for out_i, idx in enumerate(idxs):
        terminals[out_i] = float(int(idx) in terminal_idx_set)

    next_observations = observations[idxs + 1]
    rewards = np.zeros(len(idxs), dtype=np.float32)
    masks = 1.0 - terminals
    return Dataset.create(
        observations=observations[idxs].astype(np.float32),
        actions=actions[idxs].astype(np.float32),
        rewards=rewards,
        terminals=terminals,
        masks=masks.astype(np.float32),
        next_observations=next_observations.astype(np.float32),
    )
