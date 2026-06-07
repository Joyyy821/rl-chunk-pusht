"""Online RL training for Push-T action chunking experiments."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tyro
from tqdm.auto import trange

from rl_chunk_pusht.agents import agents
from rl_chunk_pusht.configs import (
    TrainConfig,
    build_agent_config,
    config_to_dict,
    normalize_bool_cli_args,
)
from rl_chunk_pusht.envs import make_pusht_env
from rl_chunk_pusht.evaluation import evaluate
from rl_chunk_pusht.utils.datasets import ReplayBuffer
from rl_chunk_pusht.utils.flax_utils import save_agent
from rl_chunk_pusht.utils.logging import ExperimentLogger, get_wandb_video, make_exp_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def tree_to_numpy_scalars(tree: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for key, value in tree.items():
        value_np = np.asarray(value)
        if value_np.shape == ():
            flat[key] = float(value_np)
    return flat


def reshape_for_utd(batch: dict[str, np.ndarray], utd_ratio: int, batch_size: int):
    return {
        k: v.reshape((utd_ratio, batch_size) + v.shape[1:])
        for k, v in batch.items()
    }


def make_replay_buffer(env, size: int) -> ReplayBuffer:
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    example_transition = dict(
        observations=np.zeros(obs_shape, dtype=np.float32),
        actions=np.zeros(action_shape, dtype=np.float32),
        rewards=np.array(0.0, dtype=np.float32),
        terminals=np.array(0.0, dtype=np.float32),
        masks=np.array(1.0, dtype=np.float32),
        next_observations=np.zeros(obs_shape, dtype=np.float32),
    )
    return ReplayBuffer.create(example_transition, size=size)


def sample_random_action(env, rng_key) -> np.ndarray:
    action = jax.random.uniform(
        rng_key,
        shape=env.action_space.shape,
        minval=-1.0,
        maxval=1.0,
    )
    return np.asarray(action, dtype=np.float32)


def run_training(config: TrainConfig) -> Path:
    set_seed(config.seed)

    env = make_pusht_env(
        config.env_mode,
        max_speed_px_s=config.max_speed_px_s,
        episode_length=config.episode_length,
        visualization_width=config.render_size[0],
        visualization_height=config.render_size[1],
    )
    eval_env = make_pusht_env(
        config.env_mode,
        max_speed_px_s=config.max_speed_px_s,
        episode_length=config.episode_length,
        visualization_width=config.render_size[0],
        visualization_height=config.render_size[1],
    )

    example_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    example_action = np.zeros(env.action_space.shape, dtype=np.float32)
    agent_config = build_agent_config(config)
    agent = agents[config.agent].create(
        config.seed,
        example_obs,
        example_action,
        agent_config,
    )
    replay_buffer = make_replay_buffer(env, config.replay_size)

    exp_name = make_exp_name(config.seed, config.exp_name)
    log_dir = config.save_dir / config.wandb_project / config.wandb_group / config.env_mode / exp_name
    logger = ExperimentLogger(
        log_dir,
        project=config.wandb_project,
        group=config.wandb_group,
        name=exp_name,
        config=config_to_dict(config),
        wandb_mode=config.wandb_mode,
        prefixes=("train", "eval", "env"),
    )

    with (log_dir / "agent_config.json").open("w") as f:
        json.dump(agent_config, f, indent=2, sort_keys=True, default=str)

    rng = jax.random.PRNGKey(config.seed)
    obs, _ = env.reset(seed=config.seed)
    action_queue: list[np.ndarray] = []
    action_dim = env.action_space.shape[-1]
    update_info: dict[str, float] = {}

    try:
        for step in trange(1, config.online_steps + 1, desc="train"):
            rng, action_rng = jax.random.split(rng)

            if len(action_queue) == 0:
                if step <= config.warmup_steps:
                    action = sample_random_action(env, action_rng)
                else:
                    action = np.asarray(agent.sample_actions(observations=obs, rng=action_rng))
                action_chunk = action.reshape(-1, action_dim)
                action_queue.extend([a.astype(np.float32) for a in action_chunk])

            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated

            replay_buffer.add_transition(
                dict(
                    observations=np.asarray(obs, dtype=np.float32),
                    actions=np.asarray(action, dtype=np.float32),
                    rewards=np.array(reward, dtype=np.float32),
                    terminals=np.array(float(done), dtype=np.float32),
                    masks=np.array(1.0 - float(terminated), dtype=np.float32),
                    next_observations=np.asarray(next_obs, dtype=np.float32),
                )
            )

            env_metrics = {
                "reward": float(reward),
                "coverage": float(info.get("coverage", 0.0)),
                "success": float(info.get("is_success", False)),
                "replay_size": float(replay_buffer.size),
            }
            if step % config.log_interval == 0:
                logger.log(env_metrics, "env", step)

            if done:
                obs, _ = env.reset()
                action_queue = []
            else:
                obs = next_obs

            can_update = replay_buffer.size >= max(config.chunk_size, config.batch_size)
            if step >= config.warmup_steps and can_update:
                batch = replay_buffer.sample_sequence(
                    config.batch_size * config.utd_ratio,
                    sequence_length=config.chunk_size,
                    discount=config.discount,
                )
                batch = reshape_for_utd(batch, config.utd_ratio, config.batch_size)
                agent, update_info = agent.batch_update(batch)

            if step % config.log_interval == 0 and update_info:
                logger.log(tree_to_numpy_scalars(update_info), "train", step)
                update_info = {}

            if config.eval_interval > 0 and step % config.eval_interval == 0:
                eval_info, _, renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=action_dim,
                    num_eval_episodes=config.eval_episodes,
                    num_video_episodes=config.video_episodes,
                    video_frame_skip=config.video_frame_skip,
                    video_size=config.render_size,
                    seed=config.seed + step,
                )
                if renders:
                    eval_info["video"] = get_wandb_video(renders, fps=15)
                logger.log(eval_info, "eval", step)

            if config.save_interval > 0 and step % config.save_interval == 0:
                save_agent(agent, log_dir, step)

        save_agent(agent, log_dir, config.online_steps)
    finally:
        env.close()
        eval_env.close()
        logger.close()

    return log_dir


def main() -> None:
    config = tyro.cli(TrainConfig, args=normalize_bool_cli_args(sys.argv[1:]))
    log_dir = run_training(config)
    print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
