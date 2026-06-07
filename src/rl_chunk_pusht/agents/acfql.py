"""Flow Q-learning agent with action chunking, adapted from Q-chunking."""

from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from rl_chunk_pusht.networks import ActorVectorField, Value
from rl_chunk_pusht.utils.flax_utils import (
    ModuleDict,
    TrainState,
    copy_params_with,
    nonpytree_field,
)


class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with optional action chunking."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"][..., -1, :], rng=sample_rng)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :], actions=next_actions
        )
        next_q = next_qs.min(axis=0) if self.config["q_agg"] == "min" else next_qs.mean(axis=0)
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_q
        )
        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
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
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, t, params=grad_params
        )

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]),
                )
                * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch["observations"], noises=noises)
            actor_actions = self.network.select("actor_onestep_flow")(
                batch["observations"], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            actor_actions = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select("critic")(batch["observations"], actions=actor_actions)
            q_loss = -jnp.mean(qs, axis=0).mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
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
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (*observations.shape[: -len(self.config["ob_dims"])], action_dim),
            )
            actions = self.network.select("actor_onestep_flow")(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            noises = jax.random.normal(
                rng,
                (*observations.shape[: -len(self.config["ob_dims"])], self.config["actor_num_samples"], action_dim),
            )
            observations = jnp.repeat(
                observations[..., None, :], self.config["actor_num_samples"], axis=-2
            )
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            q = self.network.select("critic")(observations, actions)
            q = q.mean(axis=0) if self.config["q_agg"] == "mean" else q.min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[
                jnp.arange(bsize), indices, :
            ].reshape(bshape + (action_dim,))
        else:
            raise ValueError(f"Unsupported actor_type: {self.config['actor_type']}")

        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = jnp.asarray(ex_observations)
        ex_actions = jnp.asarray(ex_actions)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=None,
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=None,
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=None,
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}
        if config["weight_decay"] > 0.0:
            network_tx = optax.adamw(learning_rate=config["lr"], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network = network.replace(
            params=copy_params_with(
                network.params,
                {"modules_target_critic": network.params["modules_critic"]},
            )
        )

        config = dict(config)
        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
