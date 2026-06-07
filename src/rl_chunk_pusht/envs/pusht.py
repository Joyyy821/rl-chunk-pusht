"""Push-T environment variants used by the RL experiments."""

from __future__ import annotations

from typing import Literal

import gym_pusht  # noqa: F401  # Registers gym-pusht when users create envs manually.
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gym_pusht.envs.pusht import PushTEnv
from pymunk.vec2d import Vec2d

EnvMode = Literal["position_target", "velocity_kinematic", "velocity_delta_target"]

WORKSPACE_SIZE = 512.0
AGENT_RADIUS = 15.0


class NormalizedStateObservationWrapper(gym.ObservationWrapper):
    """Map Push-T state observations to compact numeric features.

    Stock Push-T state is [agent_x, agent_y, block_x, block_y, block_angle].
    The wrapped observation is [agent_x, agent_y, block_x, block_y, sin(theta),
    cos(theta)], with positions normalized by 512.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        angle = obs[4]
        return np.array(
            [
                obs[0] / WORKSPACE_SIZE,
                obs[1] / WORKSPACE_SIZE,
                obs[2] / WORKSPACE_SIZE,
                obs[3] / WORKSPACE_SIZE,
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )


class NormalizedPositionActionWrapper(gym.ActionWrapper):
    """Use normalized actions for the stock absolute-position Push-T controller."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def action(self, action):
        normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return ((normalized + 1.0) * 0.5 * WORKSPACE_SIZE).astype(np.float32)


class DeltaTargetActionWrapper(gym.ActionWrapper):
    """Diagnostic velocity-to-target wrapper that keeps the built-in PD controller."""

    def __init__(self, env: gym.Env, max_speed_px_s: float):
        super().__init__(env)
        self.max_speed_px_s = float(max_speed_px_s)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def action(self, action):
        normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        current_agent_pos = np.asarray(self.unwrapped.agent.position, dtype=np.float32)
        dt_outer = 1.0 / float(self.unwrapped.control_hz)
        target = current_agent_pos + normalized * self.max_speed_px_s * dt_outer
        return np.clip(target, 0.0, WORKSPACE_SIZE).astype(np.float32)


class VelocityPushTEnv(PushTEnv):
    """Push-T variant with direct kinematic end-effector velocity commands."""

    def __init__(self, max_speed_px_s: float = 120.0, clamp_agent: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.max_speed_px_s = float(max_speed_px_s)
        self.clamp_agent = bool(clamp_agent)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        velocity_np = normalized * self.max_speed_px_s

        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        velocity = Vec2d(float(velocity_np[0]), float(velocity_np[1]))
        self._last_action = np.asarray(self.agent.position, dtype=np.float32) + (
            velocity_np / float(self.control_hz)
        )

        for _ in range(n_steps):
            self.agent.velocity = velocity
            self.space.step(self.dt)
            if self.clamp_agent:
                self._clamp_agent_position()

        self.agent.velocity = Vec2d(0.0, 0.0)
        return self._finish_step()

    def _clamp_agent_position(self) -> None:
        x, y = self.agent.position
        self.agent.position = (
            float(np.clip(x, AGENT_RADIUS, WORKSPACE_SIZE - AGENT_RADIUS)),
            float(np.clip(y, AGENT_RADIUS, WORKSPACE_SIZE - AGENT_RADIUS)),
        )

    def _finish_step(self):
        coverage = float(self._get_coverage())
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        terminated = bool(coverage > self.success_threshold)
        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = terminated
        info["coverage"] = coverage
        return observation, reward, terminated, False, info


def make_pusht_env(
    env_mode: EnvMode = "velocity_kinematic",
    *,
    max_speed_px_s: float = 120.0,
    episode_length: int = 300,
    obs_type: str = "state",
    render_mode: str = "rgb_array",
    observation_width: int = 96,
    observation_height: int = 96,
    visualization_width: int = 680,
    visualization_height: int = 680,
    normalize_obs: bool = True,
) -> gym.Env:
    """Create a Push-T env with normalized action and observation spaces."""

    env_kwargs = dict(
        obs_type=obs_type,
        render_mode=render_mode,
        observation_width=observation_width,
        observation_height=observation_height,
        visualization_width=visualization_width,
        visualization_height=visualization_height,
    )

    if env_mode == "velocity_kinematic":
        env: gym.Env = VelocityPushTEnv(max_speed_px_s=max_speed_px_s, **env_kwargs)
    elif env_mode in {"position_target", "velocity_delta_target"}:
        env = PushTEnv(**env_kwargs)
    else:
        raise ValueError(f"Unsupported Push-T env mode: {env_mode}")

    env = TimeLimit(env, max_episode_steps=episode_length)

    if env_mode == "position_target":
        env = NormalizedPositionActionWrapper(env)
    elif env_mode == "velocity_delta_target":
        env = DeltaTargetActionWrapper(env, max_speed_px_s=max_speed_px_s)

    if normalize_obs:
        if obs_type != "state":
            raise ValueError("NormalizedStateObservationWrapper currently expects obs_type='state'.")
        env = NormalizedStateObservationWrapper(env)

    return env
