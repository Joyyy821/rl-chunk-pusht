from __future__ import annotations

import numpy as np
from gymnasium.wrappers import TimeLimit

from gym_pusht.envs.pusht import PushTEnv
from rl_chunk_pusht.envs.pusht import (
    NormalizedPositionActionWrapper,
    make_pusht_env,
)


def test_position_action_wrapper_maps_to_workspace_corners():
    env = NormalizedPositionActionWrapper(TimeLimit(PushTEnv(obs_type="state"), max_episode_steps=10))
    np.testing.assert_allclose(env.action(np.array([-1.0, -1.0])), np.array([0.0, 0.0]))
    np.testing.assert_allclose(env.action(np.array([1.0, 1.0])), np.array([512.0, 512.0]))
    env.close()


def test_velocity_kinematic_moves_agent_directly():
    env = make_pusht_env(
        "velocity_kinematic",
        max_speed_px_s=120.0,
        episode_length=10,
        normalize_obs=False,
    )
    obs, _ = env.reset(seed=0)
    start = obs[:2].copy()
    obs, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obs[0] - start[0], 12.0, atol=1.0)
    np.testing.assert_allclose(obs[1] - start[1], 0.0, atol=1.0)
    env.close()


def test_velocity_kinematic_clamps_agent_inside_workspace():
    env = make_pusht_env(
        "velocity_kinematic",
        max_speed_px_s=120.0,
        episode_length=10,
        normalize_obs=False,
    )
    obs, _ = env.reset(seed=0, options={"reset_to_state": [490, 250, 256, 300, 0]})
    for _ in range(5):
        obs, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert obs[0] <= 497.0
    env.close()


def test_normalized_state_observation_shape():
    env = make_pusht_env("position_target", episode_length=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)
    env.close()
