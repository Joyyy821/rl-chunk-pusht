"""SAC/RLPD-style baseline with optional action chunking."""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from rl_chunk_pusht.networks import Ensemble, MLP, StateActionValue, TanhNormal, Temperature
from rl_chunk_pusht.utils.flax_utils import (
    ModuleDict,
    TrainState,
    copy_params_with,
    nonpytree_field,
)


class ACRLPDAgent(flax.struct.PyTreeNode):
    """Soft actor-critic agent over single actions or flattened action chunks."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select("actor")(batch["next_observations"][..., -1, :])
        next_actions = next_dist.sample(seed=sample_rng)

        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :], next_actions
        )
        next_q = next_qs.min(axis=0) if self.config["q_agg"] == "min" else next_qs.mean(axis=0)
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_q
        )
        q = self.network.select("critic")(batch["observations"], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        dist = self.network.select("actor")(batch["observations"], params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        qs = self.network.select("critic")(batch["observations"], actions)
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select("alpha")() - q).mean()
        alpha = self.network.select("alpha")(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config["target_entropy"])).mean()
        bc_loss = (
            -dist.log_prob(jnp.clip(batch_actions, -1 + 1e-5, 1 - 1e-5)).mean()
            * self.config["bc_alpha"]
        )
        total_loss = actor_loss + alpha_loss + bc_loss
        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "bc_loss": bc_loss,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "q": q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        source_key = f"modules_{module_name}"
        target_key = f"modules_target_{module_name}"
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[source_key],
            self.network.params[target_key],
        )
        return network.replace(params=copy_params_with(network.params, {target_key: new_target_params}))

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        dist = self.network.select("actor")(observations)
        return jnp.clip(dist.sample(seed=rng), -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = jnp.asarray(ex_observations)
        ex_actions = jnp.asarray(ex_actions)
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        config = dict(config)
        if config["target_entropy"] is None:
            config["target_entropy"] = -config["target_entropy_multiplier"] * full_action_dim

        critic_base_cls = partial(
            MLP,
            hidden_dims=config["value_hidden_dims"],
            activate_final=True,
            use_layer_norm=config["layer_norm"],
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config["num_qs"])
        actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = TanhNormal(actor_base_cls, full_action_dim)
        alpha_def = Temperature(config["init_temp"])

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=optax.adam(config["lr"]))
        network = network.replace(
            params=copy_params_with(
                network.params,
                {"modules_target_critic": network.params["modules_critic"]},
            )
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
