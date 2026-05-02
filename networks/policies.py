from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from .common import MLP, default_init

class FlowPolicy(nn.Module):
    """Flow-matching vector field policy for BC."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, times, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        return self.mlp(jnp.concatenate([observations, actions, times], axis=-1))
    


class NNPolicy(nn.Module):
    """
    Noise-conditioned deterministic one-step policy: (obs, noise) -> action."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    tanh_squash: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self) -> None:
        self.policy_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.action_head = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, noises, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        inputs = jnp.concatenate([observations, noises], axis=-1)
        actions = self.action_head(self.policy_net(inputs))
        if self.tanh_squash:
            actions = jnp.tanh(actions) # For training stabilization, default: False
        return actions


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GaussianPolicy(nn.Module):
    """Gaussian policy network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution
