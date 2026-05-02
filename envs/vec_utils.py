"""Vectorized env utilities for online data collection."""
import gym
import numpy as np


class GymCompatWrapper(gym.Wrapper):
    """Makes old gym envs return new-style API for evaluation.py.
    Only used on single eval envs, NOT inside SyncVectorEnv."""

    def __init__(self, env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._ep_reward = 0.0
        self._ep_length = 0
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        self._ep_reward += reward
        self._ep_length += 1

        if done:
            info["episode.return"] = self._ep_reward
            info["episode.length"] = self._ep_length

        if len(result) == 4:
            return obs, reward, terminated, truncated, info
        return result


class MinusOneRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward - 1.0


def make_collection_env(env_name, antmaze_reward_mode=None, seed=0):
    """Raw old-gym env for data collection inside SyncVectorEnv."""
    env = gym.make(env_name)
    if antmaze_reward_mode == "minus_one" and "antmaze" in env_name.lower():
        env = MinusOneRewardWrapper(env)
    env.seed(seed)
    env.reset()
    return env


def make_vec_collection_env(env_name, num_envs, base_seed, antmaze_reward_mode=None):
    """K parallel envs. Old gym API (reset->obs, step->4 values)."""
    def _make(idx):
        def _init():
            return make_collection_env(
                env_name, antmaze_reward_mode=antmaze_reward_mode, seed=base_seed + idx,
            )
        return _init
    return gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])


def make_eval_env(env_name, antmaze_reward_mode=None):
    """Single eval env with compat wrapper for evaluation.py."""
    env = gym.make(env_name)
    if antmaze_reward_mode == "minus_one" and "antmaze" in env_name.lower():
        env = MinusOneRewardWrapper(env)
    return GymCompatWrapper(env)
