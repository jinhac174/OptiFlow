import collections
import re
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.datasets import Dataset


ADROIT_ENV_PREFIXES = (
    'pen-',
    'door-',
    'hammer-',
    'relocate-',
)

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

VISUAL_PLAY_DATASET_STEMS = (
    'visual-cube-single-',
    'visual-cube-double-',
    'visual-cube-triple-',
    'visual-cube-quadruple-',
    'visual-scene-',
    'visual-puzzle-3x3-',
    'visual-puzzle-4x4-',
    'visual-puzzle-4x5-',
    'visual-puzzle-4x6-',
)


def _resolve_ogbench_env_names(env_name):
    """Map a dataset-style env_name to (gymnasium-env name, dataset name).

    OGBench registers visual env IDs without the ``play-`` segment, while
    datasets are stored under names with ``play-``. The dataset-only API
    accepts the dataset name and the pre-built env. Returns identical names
    for non-visual ogbench envs.
    """
    actual_env_name = env_name
    dataset_env_name = env_name
    if env_name.startswith('antsoccer-arena-navigate-singletask-'):
        actual_env_name = env_name.replace(
            'antsoccer-arena-navigate-singletask-', 'antsoccer-arena-singletask-'
        )
    if env_name.startswith('visual-antmaze-') and '-singletask-' in env_name:
        dataset_env_name = env_name.replace('-singletask-', '-navigate-singletask-')
    elif env_name.startswith('visual-humanoidmaze-') and '-singletask-' in env_name:
        dataset_env_name = env_name.replace('-singletask-', '-navigate-singletask-')
    for stem in VISUAL_PLAY_DATASET_STEMS:
        base_token = f'{stem}singletask-'
        play_token = f'{stem}play-singletask-'
        if env_name.startswith(play_token):
            actual_env_name = env_name.replace(play_token, base_token)
            dataset_env_name = env_name
            break
        if env_name.startswith(base_token):
            dataset_env_name = env_name.replace(base_token, play_token)
            break
    if env_name.startswith('cube-triple-singletask-'):
        dataset_env_name = env_name.replace('cube-triple-singletask-', 'cube-triple-play-singletask-')
    elif env_name.startswith('cube-double-singletask-'):
        dataset_env_name = env_name.replace('cube-double-singletask-', 'cube-double-play-singletask-')
    elif env_name.startswith('cube-single-singletask-'):
        dataset_env_name = env_name.replace('cube-single-singletask-', 'cube-single-play-singletask-')
    return actual_env_name, dataset_env_name


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()
        self.success = 0.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        if 'success' in info:
            self.success = max(self.success, float(info['success']))
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
            info['episode']['success'] = self.success

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def _with_next_actions(dataset):
    if dataset is None:
        return None
    actions = np.asarray(dataset['actions'], dtype=np.float32)
    next_actions = np.concatenate([actions[1:], actions[-1:]], axis=0)
    out = dict(dataset)
    out['next_actions'] = next_actions
    return out


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, antmaze_reward_mode='minus_one'):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        antmaze_reward_mode: AntMaze reward transform mode ('minus_one', 'raw').

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        import ogbench

        actual_env_name, dataset_env_name = _resolve_ogbench_env_names(env_name)
        is_visual = env_name.startswith('visual-') or actual_env_name != env_name or dataset_env_name != env_name
        if is_visual:
            # Visual / antsoccer / cube remap: env ID and dataset ID differ.
            env = gymnasium.make(actual_env_name)
            eval_env = gymnasium.make(actual_env_name)
            train_dataset, val_dataset = ogbench.make_env_and_datasets(
                dataset_env_name, dataset_only=True, cur_env=env,
            )
        else:
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, success_timing='post')
            eval_env = ogbench.make_env_and_datasets(env_name, env_only=True, success_timing='post')
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**_with_next_actions(train_dataset))
        val_dataset_with_next_actions = _with_next_actions(val_dataset)
        val_dataset = Dataset.create(**val_dataset_with_next_actions) if val_dataset_with_next_actions is not None else None
    elif env_name.startswith(ADROIT_ENV_PREFIXES):
        if frame_stack is not None:
            raise ValueError('Adroit does not support frame_stack yet because offline dataset stacking is not wired.')

        from envs import adroit_utils

        env = adroit_utils.make_env(env_name)
        eval_env = adroit_utils.make_env(env_name)
        dataset = adroit_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith(D4RL_ENV_PREFIXES):
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name, antmaze_reward_mode=antmaze_reward_mode)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if 'next_actions' in train_dataset:
            train_dataset = train_dataset.copy(
                add_or_replace=dict(
                    next_actions=np.clip(train_dataset['next_actions'], -1 + action_clip_eps, 1 - action_clip_eps)
                )
            )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )
            if 'next_actions' in val_dataset:
                val_dataset = val_dataset.copy(
                    add_or_replace=dict(
                        next_actions=np.clip(val_dataset['next_actions'], -1 + action_clip_eps, 1 - action_clip_eps)
                    )
                )

    return env, eval_env, train_dataset, val_dataset
