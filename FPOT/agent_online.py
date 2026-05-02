"""
Online-capable FPOT agent.

Inherits all training logic from FPOTAgent and adds `sample_actions_explore`
for offline-to-online RL (App. C.2 of the paper).

The paper specifies that online transitions are collected with the deterministic
one-step student. We default exploration_noise_std/clip to 0 to match this; both
can be overridden in config to enable a small exploration noise budget.
"""
import jax
import jax.numpy as jnp

from .agent_fpot import FPOTAgent


class OnlineFPOTAgent(FPOTAgent):

    @jax.jit
    def sample_actions_explore(self, observations, seed):
        """One-step student action plus optional clipped Gaussian exploration noise."""
        seed, noise_seed = jax.random.split(seed)
        actions = self.sample_actions(observations, seed=seed)

        noise_std  = self.config.get('exploration_noise_std', 0.0)
        noise_clip = self.config.get('exploration_noise_clip', 0.0)

        noise = jax.random.normal(noise_seed, actions.shape) * noise_std
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        return jnp.clip(actions + noise, -1, 1)
