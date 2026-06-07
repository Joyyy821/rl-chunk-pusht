"""Replay dataset utilities adapted from the Q-chunking implementation."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Frozen transition dataset with sequence sampling."""

    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []
            next_obs = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self["observations"]))
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self["observations"])
                    )
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"]))
            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self.augment(batch, ["observations", "next_observations"])
        return batch

    def sample_sequence(self, batch_size: int, sequence_length: int, discount: float):
        if self.size < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} transitions, but dataset has {self.size}."
            )

        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        data = {k: v[idxs] for k, v in self.items()}
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

        batch_observations = self["observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["observations"].shape[1:]
        )
        batch_next_observations = self["next_observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["next_observations"].shape[1:]
        )
        batch_actions = self["actions"][all_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )
        batch_rewards = self["rewards"][all_idxs].reshape(
            batch_size, sequence_length, *self["rewards"].shape[1:]
        )
        batch_masks = self["masks"][all_idxs].reshape(
            batch_size, sequence_length, *self["masks"].shape[1:]
        )
        batch_terminals = self["terminals"][all_idxs].reshape(
            batch_size, sequence_length, *self["terminals"].shape[1:]
        )

        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self["actions"][next_action_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        masks = np.ones((batch_size, sequence_length), dtype=np.float32)
        terminals = np.zeros((batch_size, sequence_length), dtype=np.float32)
        valid = np.ones((batch_size, sequence_length), dtype=np.float32)

        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i - 1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i - 1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i - 1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i - 1]

        if len(batch_observations.shape) == 5:
            observations = batch_observations.transpose(0, 2, 3, 1, 4)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else:
            observations = batch_observations
            next_observations = batch_next_observations

        return dict(
            observations=data["observations"].copy(),
            full_observations=observations,
            actions=batch_actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=batch_next_actions,
        )

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result["next_actions"] = self._dict["actions"][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Mutable replay buffer with the Dataset sequence-sampling API."""

    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def clear(self):
        self.size = self.pointer = 0
