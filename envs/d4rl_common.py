import d4rl
import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor
from envs.datasets import Dataset


def ensure_shimmy_gym_v21():
    """Ensure shimmy registers GymV21 compatibility envs."""
    try:
        import shimmy.openai_gym_compatibility  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'GymV21 compatibility requires shimmy[gym-v21]. '
            'Install with: pip install "shimmy[gym-v21]"'
        ) from exc


def make_env(env_name, filter_regexes=None):
    """Make a GymV21-backed D4RL environment."""
    ensure_shimmy_gym_v21()

    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env, filter_regexes=filter_regexes)
    return env


def qlearning_dataset_to_dataset(env, reward_transform=None, include_dones_in_terminals=True):
    """Convert a D4RL qlearning dataset into the repo Dataset format."""
    source_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    dataset = d4rl.qlearning_dataset(source_env)

    observations = dataset['observations'].astype(np.float32)
    next_observations = dataset['next_observations'].astype(np.float32)
    actions = dataset['actions'].astype(np.float32)
    rewards = dataset['rewards'].astype(np.float32).copy()
    dones = dataset['terminals'].astype(np.float32)

    if reward_transform is not None:
        rewards = reward_transform(rewards)

    terminals = np.zeros_like(rewards, dtype=np.float32)
    masks = np.zeros_like(rewards, dtype=np.float32)

    discontinuity = np.linalg.norm(observations[1:] - next_observations[:-1], axis=1) > 1e-6

    if include_dones_in_terminals:
        terminals[:-1] = np.logical_or(discontinuity, dones[:-1] == 1.0).astype(np.float32)
    else:
        terminals[:-1] = discontinuity.astype(np.float32)

    masks[:-1] = 1.0 - dones[:-1]
    masks[-1] = 1.0 - dones[-1]
    terminals[-1] = 1.0

    return Dataset.create(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        rewards=rewards,
        masks=masks,
    )
