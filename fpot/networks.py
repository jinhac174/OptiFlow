"""Flax modules used by FPOT: MLP, Value (critic), FlowPolicy (teacher), NNPolicy (one-step student)."""
from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


# ---------------------------------------------------------------------------
# Critic / Value
# ---------------------------------------------------------------------------

class Value(nn.Module):
    """Value/critic network. Used for both V(s) and Q(s, a)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class FlowPolicy(nn.Module):
    """Flow-matching vector field policy for BC (the reference / teacher flow)."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, times, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        return self.mlp(jnp.concatenate([observations, actions, times], axis=-1))


class NNPolicy(nn.Module):
    """Noise-conditioned deterministic one-step policy: (obs, noise) -> action."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    tanh_squash: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.policy_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.action_head = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, noises, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, noises], axis=-1)
        actions = self.action_head(self.policy_net(inputs))
        if self.tanh_squash:
            actions = jnp.tanh(actions)
        return actions


