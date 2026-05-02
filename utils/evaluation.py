from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _mean_metric(values):
    arrays = [np.asarray(v) for v in values]
    try:
        stacked = np.stack(arrays, axis=0)
    except ValueError:
        stacked = np.asarray(arrays)
    mean_value = np.mean(stacked, axis=0)
    if np.asarray(mean_value).shape == ():
        item = np.asarray(mean_value).item()
        return item.item() if isinstance(item, np.generic) else item
    return mean_value


def expand_metrics(metrics):
    """Expand array-valued metrics into scalar entries with dotted indices."""
    expanded = {}
    for key, value in metrics.items():
        arr = np.asarray(value)
        if arr.shape == ():
            item = arr.item()
            expanded[key] = item.item() if isinstance(item, np.generic) else item
            continue

        for idx in np.ndindex(arr.shape):
            item = arr[idx]
            if isinstance(item, np.generic):
                item = item.item()
            suffix = '.'.join(str(i) for i in idx)
            expanded[f'{key}.{suffix}'] = item
    return expanded


def serialize_traj(traj, info, episode_index, episode_seed):
    """Convert a collected trajectory into a pickle-friendly dictionary."""
    serialized = {
        'episode_index': int(episode_index),
        'episode_seed': None if episode_seed is None else int(episode_seed),
        'stats': flatten(info),
    }
    for key, values in traj.items():
        serialized[key] = np.asarray(values)
    return serialized


def maybe_get_goal_xy(env, reset_info):
    base_env = getattr(env, 'unwrapped', env)
    goal_xy = getattr(base_env, 'cur_goal_xy', None)
    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=float)
        if goal_xy.shape == (2,):
            return goal_xy

    if isinstance(reset_info, dict) and 'goal_xy' in reset_info:
        goal_xy = np.asarray(reset_info['goal_xy'], dtype=float)
        if goal_xy.shape == (2,):
            return goal_xy

    return None


def maybe_get_xy(env, info):
    if isinstance(info, dict) and 'xy' in info:
        xy = np.asarray(info['xy'], dtype=float)
        if xy.shape == (2,):
            return xy

    base_env = getattr(env, 'unwrapped', env)
    get_xy = getattr(base_env, 'get_xy', None)
    if callable(get_xy):
        xy = np.asarray(get_xy(), dtype=float)
        if xy.shape == (2,):
            return xy

    return None


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    episode_seeds=None,
    actor_rng_seed=None,
    progress=True,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if episode_seeds is not None and len(episode_seeds) != num_eval_episodes:
        raise ValueError(
            f'episode_seeds must have length {num_eval_episodes}, got {len(episode_seeds)}'
        )

    if actor_rng_seed is None:
        actor_rng_seed = np.random.randint(0, 2**32)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_rng_seed))
    trajs = []
    stats = defaultdict(list)

    renders = []
    iterator = trange(num_eval_episodes + num_video_episodes) if progress else range(
        num_eval_episodes + num_video_episodes
    )
    for i in iterator:
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        episode_seed = None

        if episode_seeds is not None and i < num_eval_episodes:
            episode_seed = int(episode_seeds[i])
            observation, info = env.reset(seed=episode_seed)
        else:
            observation, info = env.reset()

        goal_xy = maybe_get_goal_xy(env, info)
        initial_xy = maybe_get_xy(env, info)
        initial_goal_dist = None
        min_goal_dist = None
        if goal_xy is not None and initial_xy is not None:
            initial_goal_dist = float(np.linalg.norm(initial_xy - goal_xy))
            min_goal_dist = initial_goal_dist

        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            current_xy = maybe_get_xy(env, info)
            if goal_xy is not None and current_xy is not None:
                current_goal_dist = float(np.linalg.norm(current_xy - goal_xy))
                min_goal_dist = (
                    current_goal_dist
                    if min_goal_dist is None
                    else min(min_goal_dist, current_goal_dist)
                )

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            episode_metrics = flatten(info)
            final_xy = maybe_get_xy(env, info)
            if goal_xy is not None:
                episode_metrics['goal_xy'] = goal_xy
            if initial_xy is not None:
                episode_metrics['initial_xy'] = initial_xy
            if final_xy is not None:
                episode_metrics['final_xy'] = final_xy
            if initial_goal_dist is not None:
                episode_metrics['goal_dist_init'] = initial_goal_dist
            if goal_xy is not None and final_xy is not None:
                final_goal_dist = float(np.linalg.norm(final_xy - goal_xy))
                episode_metrics['goal_dist_final'] = final_goal_dist
                if initial_goal_dist is not None:
                    episode_metrics['goal_dist_reduction'] = initial_goal_dist - final_goal_dist
            if min_goal_dist is not None:
                episode_metrics['goal_dist_min'] = min_goal_dist
                if initial_goal_dist is not None:
                    episode_metrics['goal_dist_best_reduction'] = initial_goal_dist - min_goal_dist

            add_to(stats, episode_metrics)
            trajs.append(serialize_traj(traj, episode_metrics, i, episode_seed))
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = _mean_metric(v)

    return stats, trajs, renders
