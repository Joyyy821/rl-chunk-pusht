"""Small Flax helpers adapted from Q-chunking."""

from __future__ import annotations

import functools
import glob
import os
import pickle
from collections.abc import Mapping, Sequence
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def copy_params_with(params, replacements):
    """Return params with top-level replacements for dict or FrozenDict params."""

    try:
        return params.copy(add_or_replace=replacements)
    except TypeError:
        new_params = dict(params)
        new_params.update(replacements)
        return new_params


class ModuleDict(nn.Module):
    """Dictionary of named modules with convenient selection."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"Expected kwargs keys {self.modules.keys()}, got {kwargs.keys()}."
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Train state with a multi-module select helper."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        opt_state = tx.init(params) if tx is not None else None
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {"params": params}
        method_name = getattr(self.model_def, method) if method is not None else None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0
        )
        grad_min_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0
        )
        grad_norm_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0
        )

        info.update(
            {
                "grad/max": jnp.max(grad_max_flat),
                "grad/min": jnp.min(grad_min_flat),
                "grad/norm": jnp.linalg.norm(grad_norm_flat, ord=1),
            }
        )
        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, step):
    save_dict = dict(agent=flax.serialization.to_state_dict(agent))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"params_{step}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved to {save_path}")


def restore_agent_with_file(agent, file_path):
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    with open(file_path, "rb") as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
    print(f"Restored from {file_path}")
    return agent


def restore_agent(agent, restore_path, restore_epoch):
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f"Found {len(candidates)} candidates: {candidates}"
    restore_file = candidates[0] + f"/params_{restore_epoch}.pkl"
    return restore_agent_with_file(agent, restore_file)
