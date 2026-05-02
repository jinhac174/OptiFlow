import d4rl
import numpy as np

from envs import d4rl_common
from utils.datasets import Dataset


ADROIT_ENV_PREFIXES = (
    'pen-',
    'door-',
    'hammer-',
    'relocate-',
)
ADROIT_DISCONTINUITY_THRESHOLD = 1e-6


def is_supported_env(env_name):
    """Return whether the env name belongs to the Adroit family."""
    return env_name.startswith(ADROIT_ENV_PREFIXES)


def make_env(env_name):
    """Make Adroit environment."""
    return d4rl_common.make_env(env_name)


def _compute_adroit_boundary(observations, next_observations, dones):
    """Compute trajectory boundaries for Adroit datasets."""
    terminals = np.zeros_like(dones, dtype=np.float32)
    discontinuity = (
        np.linalg.norm(observations[1:] - next_observations[:-1], axis=1) > ADROIT_DISCONTINUITY_THRESHOLD
    )

    terminals[:-1] = np.logical_or(discontinuity, dones[:-1] == 1.0).astype(np.float32)
    terminals[-1] = 1.0
    return terminals


def get_dataset(env, env_name):
    """Make Adroit dataset."""
    if not is_supported_env(env_name):
        raise ValueError(f'Unsupported Adroit environment: {env_name}')

    source_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    dataset = d4rl.qlearning_dataset(source_env)

    observations = dataset['observations'].astype(np.float32)
    next_observations = dataset['next_observations'].astype(np.float32)
    actions = dataset['actions'].astype(np.float32)
    rewards = dataset['rewards'].astype(np.float32).copy()
    dones = dataset['terminals'].astype(np.float32)

    terminals = _compute_adroit_boundary(observations, next_observations, dones)
    masks = 1.0 - terminals

    return Dataset.create(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        rewards=rewards,
        masks=masks,
    )
