from envs import d4rl_common


VALID_ANTMAZE_REWARD_MODES = ('minus_one', 'raw')
D4RL_ENV_PREFIXES = (
    'maze2d-',
    'antmaze-',
    'pointmaze-',
    'kitchen-',
    'halfcheetah-',
    'hopper-',
    'walker2d-',
    'D4RL/',
    'mujoco/',
)


def is_supported_env(env_name):
    """Return whether the env name belongs to the non-Adroit D4RL family."""
    return env_name.startswith(D4RL_ENV_PREFIXES)


def _transform_antmaze_rewards(rewards, mode):
    """Apply AntMaze reward transform for a given mode."""
    if mode == 'minus_one':
        # Sparse {0,1} -> {-1,0}
        return rewards - 1.0
    if mode == 'raw':
        return rewards
    raise ValueError(f'Unknown antmaze_reward_mode={mode!r}. Expected one of {VALID_ANTMAZE_REWARD_MODES}.')


def make_env(env_name):
    """Make D4RL environment."""
    return d4rl_common.make_env(env_name)


def get_dataset(
    env,
    env_name,
    antmaze_reward_mode='minus_one',
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    if 'antmaze' in env_name:
        return d4rl_common.qlearning_dataset_to_dataset(
            env,
            reward_transform=lambda rewards: _transform_antmaze_rewards(rewards, antmaze_reward_mode),
            include_dones_in_terminals=False,
        )

    return d4rl_common.qlearning_dataset_to_dataset(
        env,
        include_dones_in_terminals=True,
    )
