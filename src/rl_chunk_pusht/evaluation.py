"""Evaluation helpers for chunked Push-T agents."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jax
import numpy as np
from PIL import Image
from tqdm.auto import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Split a PRNG key before each stochastic actor call."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    return np.asarray(image.resize(size, resample=Image.BILINEAR))


def evaluate(
    *,
    agent,
    env,
    action_dim: int,
    num_eval_episodes: int = 20,
    num_video_episodes: int = 0,
    video_frame_skip: int = 3,
    video_size: tuple[int, int] = (256, 256),
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]], list[np.ndarray]]:
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))
    stats = defaultdict(list)
    trajectories: list[dict[str, Any]] = []
    renders: list[np.ndarray] = []

    for ep_idx in trange(num_eval_episodes + num_video_episodes, desc="eval", leave=False):
        should_render = ep_idx >= num_eval_episodes
        obs, info = env.reset(seed=seed + ep_idx)
        done = False
        action_queue: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        traj = defaultdict(list)
        ep_return = 0.0
        ep_len = 0
        max_reward = 0.0
        final_info = info

        while not done:
            if len(action_queue) == 0:
                action = np.asarray(actor_fn(observations=obs))
                action_chunk = action.reshape(-1, action_dim)
                action_queue.extend([a.astype(np.float32) for a in action_chunk])

            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            ep_return += float(reward)
            ep_len += 1
            max_reward = max(max_reward, float(reward))
            final_info = info

            traj["observations"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            traj["infos"].append(info)

            if should_render and (ep_len % video_frame_skip == 0 or done):
                frames.append(resize_frame(env.render(), video_size))

            obs = next_obs

        stats["return"].append(ep_return)
        stats["length"].append(ep_len)
        stats["max_reward"].append(max_reward)
        stats["success"].append(float(final_info.get("is_success", False)))
        stats["coverage"].append(float(final_info.get("coverage", 0.0)))
        if ep_idx < num_eval_episodes:
            trajectories.append({k: np.asarray(v, dtype=object) for k, v in traj.items()})
        elif frames:
            renders.append(np.asarray(frames, dtype=np.uint8))

    return {k: float(np.mean(v)) for k, v in stats.items()}, trajectories, renders
