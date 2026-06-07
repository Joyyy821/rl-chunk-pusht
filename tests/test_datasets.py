from __future__ import annotations

import numpy as np

from rl_chunk_pusht.utils.datasets import Dataset, ReplayBuffer


def make_dataset():
    n = 8
    return Dataset.create(
        observations=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        actions=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        rewards=np.ones(n, dtype=np.float32),
        terminals=np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        masks=np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32),
        next_observations=np.arange(n * 3, dtype=np.float32).reshape(n, 3) + 1,
    )


def test_sample_sequence_shapes():
    dataset = make_dataset()
    batch = dataset.sample_sequence(batch_size=4, sequence_length=3, discount=0.99)
    assert batch["observations"].shape == (4, 3)
    assert batch["actions"].shape == (4, 3, 2)
    assert batch["next_observations"].shape == (4, 3, 3)
    assert batch["valid"].shape == (4, 3)


def test_replay_buffer_wraps_without_growing_forever():
    example = dict(
        observations=np.zeros(3, dtype=np.float32),
        actions=np.zeros(2, dtype=np.float32),
        rewards=np.array(0.0, dtype=np.float32),
        terminals=np.array(0.0, dtype=np.float32),
        masks=np.array(1.0, dtype=np.float32),
        next_observations=np.zeros(3, dtype=np.float32),
    )
    buffer = ReplayBuffer.create(example, size=3)
    for _ in range(5):
        buffer.add_transition(example)
    assert buffer.size == 3
    assert buffer.pointer == 2
