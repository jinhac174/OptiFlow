"""
Gymnasium-based vectorized envs for online data collection.

Handles both OGBench singletask envs and D4RL envs (via shimmy GymV21 wrapper).
Uses the gymnasium 5-tuple step API: obs, reward, terminated, truncated, info.

Does NOT replace envs/vec_utils.py — that file still serves any old-gym code
paths that may exist.
"""
import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor


def _is_ogbench_env(env_name: str) -> bool:
    return 'singletask' in env_name


def _make_ogbench_env(env_name, seed=0):
    import ogbench
    env = ogbench.make_env_and_datasets(env_name, env_only=True, success_timing='post')
    env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    env.reset(seed=seed)
    return env


def _make_d4rl_env(env_name, antmaze_reward_mode=None, seed=0):
    """D4RL env via shimmy. MinusOneRewardWrapper applied for antmaze if requested."""
    from envs import d4rl_utils
    env = d4rl_utils.make_env(env_name)

    if antmaze_reward_mode == 'minus_one' and 'antmaze' in env_name.lower():
        class _MinusOne(gymnasium.RewardWrapper):
            def reward(self, reward):
                return reward - 1.0
        env = _MinusOne(env)

    env.reset(seed=seed)
    return env


def _make_single_env_factory(env_name, antmaze_reward_mode, seed):
    def _init():
        if _is_ogbench_env(env_name):
            return _make_ogbench_env(env_name, seed=seed)
        return _make_d4rl_env(env_name, antmaze_reward_mode=antmaze_reward_mode, seed=seed)
    return _init


def make_vec_collection_env_gym(env_name, num_envs, base_seed, antmaze_reward_mode=None):
    """K parallel envs using gymnasium.vector.SyncVectorEnv.

    reset(seed=[...]) returns (obs_batch, info_batch).
    step(actions) returns (obs, rewards, terminated, truncated, infos).
    """
    fns = [_make_single_env_factory(env_name, antmaze_reward_mode, base_seed + i)
           for i in range(num_envs)]
    return gymnasium.vector.SyncVectorEnv(fns)


def make_eval_env_gym(env_name, antmaze_reward_mode=None, seed=12345):
    """Single eval env (gymnasium API). Works with utils.evaluation.evaluate()."""
    if _is_ogbench_env(env_name):
        return _make_ogbench_env(env_name, seed=seed)
    return _make_d4rl_env(env_name, antmaze_reward_mode=antmaze_reward_mode, seed=seed)
