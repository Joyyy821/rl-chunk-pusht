"""Trajectory overlay visualization for trained Push-T agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import imageio.v2 as imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw

from rl_chunk_pusht.agents import agents
from rl_chunk_pusht.configs import TrainConfig, build_agent_config
from rl_chunk_pusht.envs import make_pusht_env
from rl_chunk_pusht.utils.flax_utils import restore_agent_with_file


@dataclass
class VisualizeConfig:
    checkpoint: Path
    output_dir: Path = Path("figures")
    reset_seed: int = 0
    num_rollouts: int = 16
    max_steps: int | None = None
    save_videos: bool = False


def load_train_config(checkpoint: Path) -> TrainConfig:
    config_path = checkpoint.parent / "config.json"
    with config_path.open("r") as f:
        raw = json.load(f)
    raw["save_dir"] = Path(raw.get("save_dir", "exp"))
    raw["render_size"] = tuple(raw.get("render_size", (256, 256)))
    raw["actor_hidden_dims"] = tuple(raw.get("actor_hidden_dims", (512, 512, 512, 512)))
    raw["value_hidden_dims"] = tuple(raw.get("value_hidden_dims", (512, 512, 512, 512)))
    return TrainConfig(**raw)


def rollout(agent, env, *, rng, reset_seed: int, action_dim: int, max_steps: int | None):
    obs, _ = env.reset(seed=reset_seed)
    base_frame = env.render()
    done = False
    step = 0
    action_queue: list[np.ndarray] = []
    agent_path = []
    block_path = []
    frames = []

    while not done and (max_steps is None or step < max_steps):
        if len(action_queue) == 0:
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(agent.sample_actions(observations=obs, rng=action_rng))
            action_queue.extend([a.astype(np.float32) for a in action.reshape(-1, action_dim)])

        action = action_queue.pop(0)
        obs, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
        done = terminated or truncated
        step += 1
        agent_path.append(np.asarray(info["pos_agent"], dtype=np.float32))
        block_path.append(np.asarray(info["block_pose"][:2], dtype=np.float32))
        frames.append(env.render())

    return {
        "base_frame": base_frame,
        "agent_path": np.asarray(agent_path, dtype=np.float32),
        "block_path": np.asarray(block_path, dtype=np.float32),
        "frames": np.asarray(frames, dtype=np.uint8),
    }


def to_pixel(points: np.ndarray, image_size: tuple[int, int]) -> list[tuple[float, float]]:
    width, height = image_size
    scaled = points.copy()
    scaled[:, 0] = scaled[:, 0] / 512.0 * width
    scaled[:, 1] = scaled[:, 1] / 512.0 * height
    return [tuple(p) for p in scaled]


def draw_overlay(frame: np.ndarray, rollouts: list[dict[str, Any]]) -> Image.Image:
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        (31, 119, 180, 180),
        (255, 127, 14, 180),
        (44, 160, 44, 180),
        (214, 39, 40, 180),
        (148, 103, 189, 180),
        (140, 86, 75, 180),
        (227, 119, 194, 180),
        (127, 127, 127, 180),
    ]
    for idx, traj in enumerate(rollouts):
        color = palette[idx % len(palette)]
        agent_points = traj["agent_path"]
        block_points = traj["block_path"]
        if len(agent_points) >= 2:
            draw.line(to_pixel(agent_points, image.size), fill=color, width=3)
        if len(block_points) >= 2:
            block_color = (color[0], color[1], color[2], 90)
            draw.line(to_pixel(block_points, image.size), fill=block_color, width=2)
    return Image.alpha_composite(image, overlay).convert("RGB")


def run_visualization(config: VisualizeConfig) -> Path:
    train_config = load_train_config(config.checkpoint)
    env = make_pusht_env(
        train_config.env_mode,
        max_speed_px_s=train_config.max_speed_px_s,
        episode_length=train_config.episode_length,
        visualization_width=train_config.render_size[0],
        visualization_height=train_config.render_size[1],
    )
    example_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    example_action = np.zeros(env.action_space.shape, dtype=np.float32)
    agent_config = build_agent_config(train_config)
    agent = agents[train_config.agent].create(
        train_config.seed,
        example_obs,
        example_action,
        agent_config,
    )
    agent = restore_agent_with_file(agent, config.checkpoint)

    rng = jax.random.PRNGKey(train_config.seed + 10_000)
    rollouts = []
    action_dim = env.action_space.shape[-1]
    for idx in range(config.num_rollouts):
        rng, rollout_rng = jax.random.split(rng)
        rollouts.append(
            rollout(
                agent,
                env,
                rng=rollout_rng,
                reset_seed=config.reset_seed,
                action_dim=action_dim,
                max_steps=config.max_steps,
            )
        )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(rollouts[0]["base_frame"], rollouts)
    overlay_path = output_dir / f"overlay_seed{config.reset_seed}.png"
    overlay.save(overlay_path)

    if config.save_videos:
        video_dir = output_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for idx, traj in enumerate(rollouts):
            if len(traj["frames"]) > 0:
                imageio.mimsave(video_dir / f"rollout_{idx:03d}.mp4", traj["frames"], fps=10)

    env.close()
    return overlay_path


def main() -> None:
    config = tyro.cli(VisualizeConfig)
    output = run_visualization(config)
    print(f"Saved overlay to {output}")


if __name__ == "__main__":
    main()
