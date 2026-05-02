"""Shared helpers for FPOT agents."""
import jax
import jax.numpy as jnp


class FPOTCommonMixin:

    def _critic_agg(self, observations, actions, params, is_encoded=False):
        """Evaluate critic ensemble and aggregate (mean or min)."""
        qs = self.network.select('critic')(observations, actions=actions, params=params, is_encoded=is_encoded)
        if qs.ndim == 2:
            return qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)
        return qs

    def _critic_with_std(self, observations, actions, params):
        """Evaluate critic ensemble, return (aggregated, std)."""
        qs = self.network.select('critic')(observations, actions=actions, params=params)
        if qs.ndim == 2:
            agg = qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)
            std = qs.std(axis=0)
            return agg, std
        return qs, jnp.zeros_like(qs)
