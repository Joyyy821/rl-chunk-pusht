"""Flax network modules adapted from the Q-chunking codebase."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def xavier_init():
    return nn.initializers.xavier_uniform()


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x, training: bool = False) -> jnp.ndarray:
        if isinstance(x, tuple):
            x = broadcast_concatenate(*x)
        use_layer_norm = self.layer_norm or self.use_layer_norm
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                if use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


def broadcast_concatenate(*arrs):
    shape = jnp.broadcast_shapes(*map(lambda x: x.shape[:-1], arrs))
    return jnp.concatenate(
        tuple(map(lambda x: jnp.broadcast_to(x, shape=shape + (x.shape[-1],)), arrs)),
        axis=-1,
    )


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module | None = None

    def setup(self):
        mlp_class = ensemblize(MLP, self.num_ensembles) if self.num_ensembles > 1 else MLP
        self.value_net = mlp_class(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, observations, actions=None):
        inputs = [self.encoder(observations) if self.encoder is not None else observations]
        if actions is not None:
            inputs.append(actions)
        v = self.value_net(jnp.concatenate(inputs, axis=-1)).squeeze(-1)
        return v


class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module | None = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)
        return self.mlp(inputs)


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


class StateActionValue(nn.Module):
    base_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)
        value = nn.Dense(1, kernel_init=xavier_init())(outputs)
        return jnp.squeeze(value, -1)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution,
            bijector=tfb.Tanh(),
            validate_args=validate_args,
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class TanhNormal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)
        means = nn.Dense(self.action_dim, kernel_init=xavier_init(), name="OutputDenseMean")(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=xavier_init(),
                name="OutputDenseLogStd",
            )(x)
        else:
            log_stds = self.param("OutputLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return TanhTransformedDistribution(distribution)


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)
