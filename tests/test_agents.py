from __future__ import annotations

import jax
import numpy as np

from rl_chunk_pusht.agents import agents
from rl_chunk_pusht.configs import TrainConfig, build_agent_config


def make_tiny_config(*, action_chunking: bool, chunk_size: int) -> TrainConfig:
    return TrainConfig(
        online_steps=1,
        warmup_steps=1,
        batch_size=4,
        chunk_size=chunk_size,
        action_chunking=action_chunking,
        actor_hidden_dims=(16, 16),
        value_hidden_dims=(16, 16),
        actor_num_samples=2,
        flow_steps=2,
        num_qs=2,
        wandb_mode="disabled",
    )


def make_batch(batch_size: int, chunk_size: int):
    return dict(
        observations=np.zeros((batch_size, 6), dtype=np.float32),
        actions=np.zeros((batch_size, chunk_size, 2), dtype=np.float32),
        rewards=np.zeros((batch_size, chunk_size), dtype=np.float32),
        masks=np.ones((batch_size, chunk_size), dtype=np.float32),
        terminals=np.zeros((batch_size, chunk_size), dtype=np.float32),
        valid=np.ones((batch_size, chunk_size), dtype=np.float32),
        next_observations=np.zeros((batch_size, chunk_size, 6), dtype=np.float32),
        next_actions=np.zeros((batch_size, chunk_size, 2), dtype=np.float32),
    )


def test_acfql_chunked_and_no_chunk_action_shapes():
    for action_chunking in (True, False):
        config = make_tiny_config(action_chunking=action_chunking, chunk_size=3)
        agent = agents["acfql"].create(
            config.seed,
            np.zeros(6, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            build_agent_config(config),
        )
        action = agent.sample_actions(np.zeros(6, dtype=np.float32), rng=jax.random.PRNGKey(0))
        expected_dim = 6 if action_chunking else 2
        assert action.shape == (expected_dim,)


def test_acfql_update_smoke():
    config = make_tiny_config(action_chunking=True, chunk_size=2)
    agent = agents["acfql"].create(
        config.seed,
        np.zeros(6, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        build_agent_config(config),
    )
    batch = make_batch(batch_size=4, chunk_size=2)
    agent, info = agent.update(batch)
    assert "critic/critic_loss" in info
    assert "actor/bc_flow_loss" in info
