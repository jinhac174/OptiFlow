"""
Gymnasium env factories for online interaction.

Handles both OGBench singletask envs and D4RL envs (via shimmy GymV21 wrapper).
Uses the gymnasium 5-tuple step API: obs, reward, terminated, truncated, info.
"""
import gymnasium

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


def make_eval_env_gym(env_name, antmaze_reward_mode=None, seed=12345):
    """Single env (gymnasium API). Used for both online collection and eval."""
    if _is_ogbench_env(env_name):
        return _make_ogbench_env(env_name, seed=seed)
    return _make_d4rl_env(env_name, antmaze_reward_mode=antmaze_reward_mode, seed=seed)
