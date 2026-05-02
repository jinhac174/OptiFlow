"""
FPOT training script.

Hydra entry point. The agent class is selected by `cfg.agent.agent_file`:
    agent_fpot   -> FPOTAgent              (offline)
    agent_online -> OnlineFPOTAgent        (offline-to-online)

Usage:
    python scripts/train.py env=cube_single_play_task1 seed=1
    python scripts/train.py train=offline_to_online env=cube_double_play_task2 seed=1
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import json
import os
import random
import re
from typing import Any, Dict

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from FPOT.agent_fpot import FPOTAgent
from FPOT.agent_online import OnlineFPOTAgent
from envs.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.replay_buffer import ReplayBuffer

try:
    import wandb
except Exception:
    wandb = None

try:
    from utils.flax_utils import save_agent
except Exception:
    save_agent = None


# ------------------------------------------------------------------
# WandB metric whitelist
# ------------------------------------------------------------------
WANDB_METRICS = {
    "eval/episode.return":                          "eval/return",
    "eval/episode.normalized_return":               "eval/normalized_return",
    "eval/episode.success":                         "eval/success",
    # Critic
    "train/critic/critic_loss":                     "critic/loss",
    "train/critic/q_mean":                          "critic/q_mean",
    "train/critic/q_max":                           "critic/q_max",
    "train/critic/q_min":                           "critic/q_min",
    "train/critic/target_q_mean":                   "critic/target_q_mean",
    "train/critic/next_q_student_mean":             "critic/next_q_student_mean",
    "train/critic/next_q_teacher_mean":             "critic/next_q_teacher_mean",
    "train/critic/next_q_mean":                     "critic/next_q_mean",
    # Actor: VaBC reference flow
    "train/actor/loss":                             "actor/loss",
    "train/actor/vabc/loss":                        "actor/vabc_loss",
    "train/actor/vabc/guidance_mean":               "vabc/guidance_mean",
    "train/actor/vabc/guidance_max":                "vabc/guidance_max",
    "train/actor/vabc/guidance_min":                "vabc/guidance_min",
    "train/actor/vabc/q_data_mean":                 "vabc/q_data_mean",
    "train/actor/vabc/q_ref_mean":                  "vabc/q_ref_mean",
    "train/actor/vabc/q_scale":                     "vabc/q_scale",
    # Actor: distillation + transport
    "train/actor/distill/loss":                     "actor/distill_loss",
    "train/actor/transport/selected_mass_mean":     "transport/selected_mass",
    "train/actor/q/selected_mean":                  "q/selected_mean",
    "train/actor/q/student_mean":                   "q/student_mean",
    "train/actor/q/teacher_mean":                   "q/teacher_mean",
    "train/actor/q/teacher_minus_student":          "q/teacher_minus_student",
    # Actor: optional Q-max ablation
    "train/actor/qmax/loss":                        "actor/qmax_loss",
    "train/actor/qmax/lambda_q":                    "qmax/lambda_q",
    # Online phase
    "train/buffer_size":                            "online/buffer_size",
    "train/env_steps":                              "online/env_steps",
}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def to_scalar(value: Any):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key, sep=sep))
        else:
            items[key] = v
    return items


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list = []
        self.header: list = []

    def log(self, data: Dict, step: int):
        row = {"step": int(step)}
        for k, v in data.items():
            s = to_scalar(v)
            row[k] = (np.array2string(s, separator=" ", threshold=s.size)
                      if isinstance(s, np.ndarray) else s)
        for k in row:
            if k not in self.header:
                self.header.append(k)
        self.rows.append(row)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in self.header})


class NullLogger:
    """No-op stand-in for CsvLogger when logging.save_csv=false."""
    def log(self, data: Dict, step: int):
        pass


def make_logger(cfg: DictConfig, run_dir: Path):
    if bool(cfg.logging.get("save_csv", True)):
        return CsvLogger(run_dir / "metrics.csv")
    return NullLogger()


def write_json(path: Path, payload):
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Override parsing helpers (shared by run-dir and wandb init)
# ------------------------------------------------------------------

def _format_value(v):
    """Format a value for wandb name/group/tags.

    Floats: repr, then strip trailing zeros after the decimal point, but
    always keep at least one digit after the point (so 2.0 stays 2.0, not 2).
    Ints, bools, strings: return unchanged (as str for non-strings).
    """
    if isinstance(v, float):
        s = repr(v)
        # Only reformat simple decimal notation; leave 1e-5 style alone.
        if '.' in s and 'e' not in s.lower():
            s = s.rstrip('0')
            if s.endswith('.'):
                s += '0'
        return s
    if isinstance(v, str):
        return v
    return str(v)


def _parse_overrides(overrides):
    """Categorise Hydra task overrides.

    Returns:
        env_name, seed_str, exp_tag, ablation_parts.
    ablation_parts excludes any override whose key appears in
    +logging.fixed_keys=[...] (injected by sweep_runner.py for the
    sweep's ``fixed:`` block) so the wandb name/group reflects only
    the swept axes.
    """
    # First pass: extract fixed_keys list from the overrides themselves.
    fixed_keys = set()
    for override in overrides:
        clean = override.lstrip('+~')
        if clean.startswith('logging.fixed_keys='):
            raw = clean[len('logging.fixed_keys='):]
            raw = raw.strip().lstrip('[').rstrip(']')
            for tok in raw.split(','):
                tok = tok.strip()
                if tok:
                    fixed_keys.add(tok)
            break

    env_name = None
    seed_str = None
    exp_tag = None
    ablation_parts = []

    for override in overrides:
        clean = override.lstrip('+~')
        if '=' not in clean:
            continue
        key, _, value = clean.partition('=')

        if key == 'env':
            env_name = value
        elif key == 'seed':
            seed_str = value
        elif key == 'exp':
            exp_tag = value
        elif key == 'logging.fixed_keys':
            continue  # meta-override, never a real ablation
        elif key in fixed_keys:
            continue  # swept-yaml fixed override, skip for name/group/tags
        elif key.startswith('agent.') or key.startswith('train.'):
            prefix    = 'agent.' if key.startswith('agent.') else 'train.'
            short_key = key[len(prefix):]
            try:
                formatted = _format_value(float(value)) if '.' in value else value
            except (ValueError, TypeError):
                formatted = value
            ablation_parts.append((short_key, formatted))

    return env_name, seed_str, exp_tag, ablation_parts

# ------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------

def _build_run_base(cfg: DictConfig) -> Path:
    """Derive the base path for this run from Hydra overrides.

    Layout:
        {root_dir}/{env.name}/{sweep_name}/{hyperparam_subdir}/seed_{N}

    sweep_name
        Value of the ``+exp=...`` override injected by sweep_runner.py.
        Falls back to ``cfg.experiment_name`` for legacy sbatch scripts that
        never set ``+exp``.

    hyperparam_subdir
        All overrides that are not env/seed/experiment_name/exp, with the
        ``agent.``/``train.`` prefix stripped, joined as ``k=v`` pairs
        separated by ``_``.  "base" when no such overrides exist (single-
        config runs that only vary env and seed).
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        overrides = []

    _, _, exp_tag, ablation_parts = _parse_overrides(overrides)

    sweep_name        = exp_tag if exp_tag is not None else cfg.experiment_name
    hyperparam_subdir = (
        "_".join(f"{k}={v}" for k, v in ablation_parts)
        if ablation_parts else "base"
    )

    return (
        Path(cfg.logging.root_dir)
        / cfg.env.name
        / sweep_name
        / hyperparam_subdir
        / f"seed_{cfg.seed}"
    )


def ensure_run_dir(cfg: DictConfig) -> Path:
    base = _build_run_base(cfg)
    base.mkdir(parents=True, exist_ok=True)
    if cfg.logging.overwrite:
        run_dir = base / f"{cfg.logging.run_prefix}_00"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    if cfg.logging.auto_increment:
        idx = 0
        while True:
            run_dir = base / f"{cfg.logging.run_prefix}_{idx:02d}"
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_dir
            idx += 1
    raise RuntimeError("Set logging.overwrite=true or logging.auto_increment=true")


_TASK_RE = re.compile(r'^(.+)-singletask-task(\d+)$')


def _clean_task_name(env_name: str) -> str:
    """Convert raw env_name string to a clean task name for naming/tagging.

    Examples:
        cube-single-play-singletask-task1-v0 -> 'cube_single_play_task1'
        antmaze-large-navigate-singletask-task3-v0 -> 'antmaze_large_navigate_task3'
        antmaze-umaze-v2  -> 'antmaze_umaze'
        door-cloned-v1    -> 'door_cloned'
    """
    s = re.sub(r'-v\d+$', '', env_name)
    m = _TASK_RE.match(s)
    if m:
        return f"{m.group(1).replace('-', '_')}_task{m.group(2)}"
    return s.replace('-', '_')


def maybe_init_wandb(cfg: DictConfig, run_dir: Path):
    if wandb is None:
        return
    mode = str(cfg.logging.get("wandb_mode", "disabled"))
    if mode == "disabled":
        return

    # --- resolve Hydra task overrides (for sweep_name, exp tag, swept axes) ---
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        overrides = []

    _, seed_str, _, ablation_parts = _parse_overrides(overrides)
    if seed_str is None:
        seed_str = str(cfg.seed)

    # --- task name from cfg.env.env_name (the actual env id, not config-group) ---
    clean_task = _clean_task_name(str(cfg.env.env_name))

    # --- short keys for swept axes (used in run name + tags) ---
    _KEY_REMAP = {
        "w_temperature": "tau",
        "eta_temperature": "eta",
        "n_student": "N",
        "n_teacher": "M",
        "lambda_q": "lq",
    }
    ablation_parts = [(_KEY_REMAP.get(k, k), v) for k, v in ablation_parts]
    ablation_summary = " ".join(f"{k}={v}" for k, v in ablation_parts)

    # --- offline-to-online prefix ---
    is_o2o = (cfg.train.get("mode", "offline") == "offline_to_online")
    o2o_prefix = "O2O | " if is_o2o else ""

    # --- run name / group ---
    name_parts = [clean_task]
    if ablation_summary:
        name_parts.append(ablation_summary)
    name_parts.append(f"s{seed_str}")
    wandb_name = o2o_prefix + " | ".join(name_parts)

    group_parts = [clean_task]
    if ablation_summary:
        group_parts.append(ablation_summary)
    wandb_group = o2o_prefix + " | ".join(group_parts)

    wandb_job_type = cfg.agent.name

    # --- tags: minimal — task, seed, tau, eta, plus any swept axes ---
    def _tag_val(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, float):
            return _format_value(v)
        return str(v)

    tags = [
        f"task:{clean_task}",
        f"seed:{seed_str}",
    ]
    tau_val = cfg.agent.get("w_temperature", None)
    if tau_val is not None:
        tags.append(f"tau:{_tag_val(float(tau_val))}")
    eta_val = cfg.agent.get("eta_temperature", None)
    if eta_val is not None:
        tags.append(f"eta:{_tag_val(float(eta_val))}")
    # Swept-axis values (e.g. lambda_q, N, M for ablations) — skip if duplicate of tau/eta
    seen_keys = {t.split(":", 1)[0] for t in tags}
    for short_key, value in ablation_parts:
        if short_key not in seen_keys:
            tags.append(f"{short_key}:{value}")
    if is_o2o:
        tags.append("off2on")

    # --- project: cfg.logging.wandb_project (set per-yaml) ---
    project = cfg.logging.get("wandb_project", None) or "fpot"
    if cfg.logging.get("wandb_project_append_date", False):
        date_fmt = str(cfg.logging.get("wandb_project_date_format", "%Y%m%d"))
        project = f"{project}_{datetime.now().strftime(date_fmt)}"

    print(f"[wandb] project: {project}")
    print(f"[wandb] name   : {wandb_name}")
    print(f"[wandb] group  : {wandb_group}")
    print(f"[wandb] tags   : {tags}")

    wandb.init(
        project=project,
        entity=cfg.logging.get("wandb_entity", None),
        mode=mode,
        name=wandb_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=tags,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def save_resolved_config(cfg: DictConfig, run_dir: Path):
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    write_json(run_dir / "config.json", OmegaConf.to_container(cfg, resolve=True))


def save_checkpoint(agent, path: Path):
    if save_agent is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    save_agent(agent, str(path), epoch=0)


def build_agent_config(cfg: DictConfig) -> Dict:
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    assert isinstance(agent_cfg, dict)
    for key in ("actor_hidden_dims", "value_hidden_dims"):
        if key in agent_cfg and isinstance(agent_cfg[key], list):
            agent_cfg[key] = tuple(agent_cfg[key])
    return agent_cfg


def build_env_kwargs(cfg: DictConfig) -> Dict:
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    assert isinstance(env_cfg, dict)
    return {
        "env_name": env_cfg["env_name"],
        "frame_stack": env_cfg.get("frame_stack", None),
        "action_clip_eps": env_cfg.get("action_clip_eps", 1e-5),
        "antmaze_reward_mode": env_cfg.get("antmaze_reward_mode", "minus_one"),
    }


def load_agent_class(agent_file: str):
    if agent_file == "agent_fpot":
        return FPOTAgent
    if agent_file == "agent_online":
        return OnlineFPOTAgent
    raise ValueError(f"Unknown agent_file: {agent_file}")


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def log_to_wandb(row: Dict, step: int):
    if wandb is None or wandb.run is None:
        return
    out = {v: row[k] for k, v in WANDB_METRICS.items() if k in row}
    if out:
        wandb.log(out, step=step)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_and_log(*, agent, eval_env, cfg, run_dir, csv_logger, step, best_score):
    eval_stats, _, _ = evaluate(
        agent=agent,
        env=eval_env,
        num_eval_episodes=int(cfg.eval.num_eval_episodes),
        num_video_episodes=int(cfg.eval.num_video_episodes),
        video_frame_skip=int(cfg.eval.video_frame_skip),
        actor_rng_seed=int(cfg.eval.fixed_eval_actor_seed),
        progress=bool(cfg.eval.progress),
    )
    eval_row = {f"eval/{k}": to_scalar(v) for k, v in flatten_dict(dict(eval_stats)).items()}
    csv_logger.log(eval_row, step)
    log_to_wandb(eval_row, step)

    score = eval_stats.get(
        "episode.normalized_return",
        eval_stats.get("episode.return", -np.inf),
    )
    if score > best_score and cfg.logging.save_checkpoint:
        save_checkpoint(agent, run_dir / "best_agent")
    return score, eval_stats


# ------------------------------------------------------------------
# Offline training
# ------------------------------------------------------------------

def offline_train(cfg: DictConfig, run_dir: Path):
    env_kwargs = build_env_kwargs(cfg)
    _, eval_env, train_dataset, _ = make_env_and_datasets(**env_kwargs)

    seed = int(cfg.seed)
    if hasattr(train_dataset, "set_seed"):
        train_dataset.set_seed(seed)

    # Configure dataset for visual observations.
    if cfg.env.get('frame_stack', None) is not None:
        train_dataset.frame_stack = int(cfg.env.frame_stack)
    if cfg.agent.get('encoder', None) is not None:
        train_dataset.p_aug = float(cfg.agent.get('p_aug', 0.5))

    ex_batch = train_dataset.sample(2)
    AgentClass = load_agent_class(cfg.agent.get("agent_file", "agent_offline"))
    agent = AgentClass.create(
        seed=seed,
        ex_observations=ex_batch["observations"],
        ex_actions=ex_batch["actions"],
        config=build_agent_config(cfg),
    )

    csv_logger = make_logger(cfg, run_dir)
    best_score = -np.inf
    max_steps     = int(cfg.train.max_steps)
    batch_size    = int(cfg.train.batch_size)
    log_interval  = int(cfg.train.log_interval)
    eval_interval = int(cfg.train.eval_interval)
    save_interval = int(cfg.train.save_interval)

    for step in range(1, max_steps + 1):
        batch = train_dataset.sample(batch_size)
        batch["global_step"] = np.int32(step)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        agent, info = agent.update(batch)

        if step % log_interval == 0:
            row = {f"train/{k}": to_scalar(v) for k, v in flatten_dict(dict(info)).items()}
            csv_logger.log(row, step)
            log_to_wandb(row, step)

        if step % eval_interval == 0:
            score, stats = evaluate_and_log(
                agent=agent, eval_env=eval_env, cfg=cfg, run_dir=run_dir,
                csv_logger=csv_logger, step=step, best_score=best_score,
            )
            ret = stats.get("episode.normalized_return", stats.get("episode.return", float("nan")))
            print(f"[step {step:>7d}] return: {ret:.2f}  (best: {best_score:.2f})")
            if score > best_score:
                best_score = score

        if cfg.logging.save_checkpoint and step % save_interval == 0:
            save_checkpoint(agent, run_dir / f"agent_step_{step}")

    if cfg.logging.save_checkpoint:
        save_checkpoint(agent, run_dir / "final_agent")


# ------------------------------------------------------------------
# Online training (gymnasium API; handles both OGBench and D4RL)
# ------------------------------------------------------------------

def online_train(cfg: DictConfig, run_dir: Path):
    import jax
    from envs.vec_utils_gymnasium import make_vec_collection_env_gym, make_eval_env_gym

    env_kwargs = build_env_kwargs(cfg)
    seed = int(cfg.seed)
    num_envs = int(cfg.train.get("num_collection_envs", 8))

    vec_env = make_vec_collection_env_gym(
        env_name=env_kwargs["env_name"],
        num_envs=num_envs,
        base_seed=seed + 1000,
        antmaze_reward_mode=env_kwargs.get("antmaze_reward_mode", None),
    )
    obs, _ = vec_env.reset(seed=[seed + 1000 + i for i in range(num_envs)])

    eval_env = make_eval_env_gym(
        env_kwargs["env_name"],
        antmaze_reward_mode=env_kwargs.get("antmaze_reward_mode", None),
    )

    obs_dim    = obs.shape[1]
    action_dim = vec_env.single_action_space.shape[0]

    AgentClass = load_agent_class(cfg.agent.get("agent_file", "agent_online"))
    agent_cfg  = build_agent_config(cfg)

    agent = AgentClass.create(
        seed=seed,
        ex_observations=np.zeros((2, obs_dim), dtype=np.float32),
        ex_actions=np.zeros((2, action_dim), dtype=np.float32),
        config=agent_cfg,
    )

    buffer = ReplayBuffer(
        capacity=int(cfg.train.replay_buffer_capacity),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    csv_logger = make_logger(cfg, run_dir)
    best_score = -np.inf
    max_steps      = int(cfg.train.max_steps)
    batch_size     = int(cfg.train.batch_size)
    log_interval   = int(cfg.train.log_interval)
    eval_interval  = int(cfg.train.eval_interval)
    save_interval  = int(cfg.train.save_interval)
    warmup_steps   = int(cfg.train.warmup_random_steps)
    start_training = int(cfg.train.start_training_step)
    utd_ratio      = int(cfg.train.utd_ratio)

    explore_rng = jax.random.PRNGKey(seed + 100)
    env_steps = 0
    _just_reset = np.zeros(num_envs, dtype=bool)

    for step in range(1, max_steps + 1):
        if env_steps < warmup_steps:
            actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        else:
            explore_rng, act_rng = jax.random.split(explore_rng)
            actions = np.asarray(agent.sample_actions_explore(obs, seed=act_rng))

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions)

        # gymnasium 1.2 AutoresetMode.NEXT_STEP: when an episode ends, the
        # NEXT step's transition is fake (env auto-resets, action ignored).
        # Skip storing those fake transitions.
        valid = ~_just_reset
        if valid.any():
            buffer.add_batch(
                obs[valid],
                actions[valid],
                rewards[valid],
                next_obs[valid],
                terms[valid].astype(np.float32),
            )
            env_steps += int(valid.sum())

        _just_reset = np.logical_or(terms, truncs)
        obs = next_obs

        if env_steps >= start_training and len(buffer) >= batch_size:
            for _ in range(utd_ratio):
                batch = buffer.sample(batch_size)
                batch["global_step"] = np.int32(step)
                batch = {k: jnp.asarray(v) for k, v in batch.items()}
                agent, info = agent.update(batch)

            if step % log_interval == 0:
                row = {f"train/{k}": to_scalar(v) for k, v in flatten_dict(dict(info)).items()}
                row["train/buffer_size"] = len(buffer)
                row["train/env_steps"] = env_steps
                csv_logger.log(row, step)
                log_to_wandb(row, step)

        if step % eval_interval == 0:
            score, stats = evaluate_and_log(
                agent=agent, eval_env=eval_env, cfg=cfg, run_dir=run_dir,
                csv_logger=csv_logger, step=step, best_score=best_score,
            )
            ret = stats.get("episode.normalized_return", stats.get("episode.return", float("nan")))
            print(f"[step {step:>7d}] return: {ret:.2f}  (best: {best_score:.2f})  env_steps: {env_steps}", flush=True)
            if score > best_score:
                best_score = score

        if cfg.logging.save_checkpoint and step % save_interval == 0:
            save_checkpoint(agent, run_dir / f"agent_step_{step}")

    vec_env.close()
    if cfg.logging.save_checkpoint:
        save_checkpoint(agent, run_dir / "final_agent")


# ------------------------------------------------------------------
# Offline-to-online training (single process, continuous step axis)
# ------------------------------------------------------------------

def offline_to_online_train(cfg: DictConfig, run_dir: Path):
    """Run offline phase (1..T_off) and online phase (T_off+1..T_off+T_on) in one process.

    Supports two modes for online batch sampling:
      - balanced_sampling=False (default, matches FPOT paper): single unified
        replay buffer pre-filled with the offline dataset; online transitions
        are appended (circular). The critic and actor see one mixed
        distribution. This matches the paper's "unified replay buffer that
        already contains the offline dataset" description.
      - balanced_sampling=True: FQL-style — keep offline dataset and online
        buffer separate, sample 50/50 each train step. Useful when a strict
        guarantee of offline-distribution data per batch is desired, but
        deviates from the paper's offline-to-online setup.

    Supports resuming from a saved checkpoint to skip the offline phase:
      - cfg.train.resume_from: filesystem path to an agent checkpoint dir.
        When set, the offline phase is skipped and the agent is restored from
        the checkpoint, then the online phase begins from that state.
    """
    import jax
    from envs.vec_utils_gymnasium import make_vec_collection_env_gym, make_eval_env_gym

    env_kwargs = build_env_kwargs(cfg)
    seed = int(cfg.seed)

    # ========== Setup (offline dataset always loaded; needed for balanced sampling
    #            and for agent init, even when resuming) ==========
    _, eval_env_offline, train_dataset, _ = make_env_and_datasets(**env_kwargs)
    if hasattr(train_dataset, "set_seed"):
        train_dataset.set_seed(seed)

    if cfg.env.get("frame_stack", None) is not None:
        train_dataset.frame_stack = int(cfg.env.frame_stack)
    if cfg.agent.get("encoder", None) is not None:
        train_dataset.p_aug = float(cfg.agent.get("p_aug", 0.5))

    ex_batch = train_dataset.sample(2)
    AgentClass = load_agent_class(cfg.agent.get("agent_file", "agent_online"))
    agent = AgentClass.create(
        seed=seed,
        ex_observations=ex_batch["observations"],
        ex_actions=ex_batch["actions"],
        config=build_agent_config(cfg),
    )

    csv_logger = make_logger(cfg, run_dir)
    best_score = -np.inf
    batch_size     = int(cfg.train.batch_size)
    log_interval   = int(cfg.train.log_interval)
    eval_interval  = int(cfg.train.eval_interval)
    save_interval  = int(cfg.train.save_interval)
    T_off          = int(cfg.train.offline_max_steps)
    T_on           = int(cfg.train.online_max_steps)

    # ========== Resume from checkpoint? ==========
    resume_from = cfg.train.get("resume_from", None)
    if resume_from is not None and str(resume_from).strip() != "":
        from utils.flax_utils import restore_agent
        print(f"=== Resuming from checkpoint: {resume_from} ===", flush=True)
        # restore_agent uses glob; pass the parent dir, then it appends params_0.pkl
        agent = restore_agent(agent, str(resume_from), 0)
        print(f"=== Skipping offline phase, jumping to online phase ===", flush=True)
    else:
        # ---------- Offline phase ----------
        print(f"=== Offline phase: steps 1..{T_off} ===", flush=True)
        for step in range(1, T_off + 1):
            batch = train_dataset.sample(batch_size)
            batch["global_step"] = np.int32(step)
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            agent, info = agent.update(batch)

            if step % log_interval == 0:
                row = {f"train/{k}": to_scalar(v) for k, v in flatten_dict(dict(info)).items()}
                row["train/env_steps"] = 0
                row["train/phase"] = 0
                csv_logger.log(row, step)
                log_to_wandb(row, step)

            if step % eval_interval == 0:
                score, stats = evaluate_and_log(
                    agent=agent, eval_env=eval_env_offline, cfg=cfg, run_dir=run_dir,
                    csv_logger=csv_logger, step=step, best_score=best_score,
                )
                ret = stats.get("episode.normalized_return", stats.get("episode.return", float("nan")))
                print(f"[offline {step:>7d}] return: {ret:.2f}  (best: {best_score:.2f})", flush=True)
                if score > best_score:
                    best_score = score

            if cfg.logging.save_checkpoint and step % save_interval == 0:
                save_checkpoint(agent, run_dir / f"agent_step_{step}")

        # Always save offline endpoint (regardless of save_checkpoint flag)
        save_checkpoint(agent, run_dir / f"agent_step_{T_off}")

    # ========== Online phase setup ==========
    print(f"=== Online phase: steps {T_off+1}..{T_off+T_on} ===", flush=True)
    num_envs = int(cfg.train.num_collection_envs)
    vec_env = make_vec_collection_env_gym(
        env_name=env_kwargs["env_name"],
        num_envs=num_envs,
        base_seed=seed + 1000,
        antmaze_reward_mode=env_kwargs.get("antmaze_reward_mode", None),
    )
    obs, _ = vec_env.reset(seed=[seed + 1000 + i for i in range(num_envs)])

    eval_env_online = make_eval_env_gym(
        env_kwargs["env_name"],
        antmaze_reward_mode=env_kwargs.get("antmaze_reward_mode", None),
    )

    obs_dim    = obs.shape[1]
    action_dim = vec_env.single_action_space.shape[0]

    online_buffer = ReplayBuffer(
        capacity=int(cfg.train.replay_buffer_capacity),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    balanced_sampling = bool(cfg.train.get("balanced_sampling", False))

    if balanced_sampling:
        # FQL-style: keep offline dataset + online buffer separate,
        # sample 50/50 each train step. The online buffer is initially EMPTY.
        print(f"=== Balanced sampling enabled (50/50 offline_dataset + online_buffer) ===", flush=True)
    else:
        # Legacy: pre-fill the online buffer with offline transitions.
        if bool(cfg.train.get("seed_replay_with_offline", True)):
            print("Seeding online buffer with offline transitions...", flush=True)
            n_seed = min(len(train_dataset), online_buffer.capacity)
            online_buffer.add_batch(
                np.asarray(train_dataset["observations"][:n_seed], dtype=np.float32),
                np.asarray(train_dataset["actions"][:n_seed], dtype=np.float32),
                np.asarray(train_dataset["rewards"][:n_seed], dtype=np.float32),
                np.asarray(train_dataset["next_observations"][:n_seed], dtype=np.float32),
                1.0 - np.asarray(train_dataset["masks"][:n_seed], dtype=np.float32),
            )
            print(f"Seeded {len(online_buffer)} transitions", flush=True)

    warmup_steps   = int(cfg.train.warmup_random_steps)
    start_training = int(cfg.train.start_training_step)
    utd_ratio      = int(cfg.train.utd_ratio)
    explore_rng    = jax.random.PRNGKey(seed + 100)
    env_steps      = 0
    _just_reset    = np.zeros(num_envs, dtype=bool)

    # ---------- Online phase ----------
    for online_step in range(1, T_on + 1):
        global_step = T_off + online_step

        if env_steps < warmup_steps:
            actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        else:
            explore_rng, act_rng = jax.random.split(explore_rng)
            actions = np.asarray(agent.sample_actions_explore(obs, seed=act_rng))

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions)

        # gymnasium 1.2 AutoresetMode.NEXT_STEP: when an episode ends, the
        # NEXT step's transition is fake (env auto-resets, action ignored).
        # Skip storing those fake transitions.
        valid = ~_just_reset
        if valid.any():
            online_buffer.add_batch(
                obs[valid],
                actions[valid],
                rewards[valid],
                next_obs[valid],
                terms[valid].astype(np.float32),
            )
            env_steps += int(valid.sum())

        _just_reset = np.logical_or(terms, truncs)
        obs = next_obs

        # ---- Training step ----
        # In balanced mode: require online buffer to have batch_size//2 samples.
        # In legacy mode: require buffer to have batch_size samples.
        min_online = batch_size // 2 if balanced_sampling else batch_size
        ready = (env_steps >= start_training) and (len(online_buffer) >= min_online)

        if ready:
            for _ in range(utd_ratio):
                if balanced_sampling:
                    half = batch_size // 2
                    # Half from offline dataset (Dataset.sample returns dict with extra keys)
                    off_batch = train_dataset.sample(half)
                    # Half from online buffer
                    on_batch = online_buffer.sample(half)
                    # Concatenate the keys agent.update needs
                    keys = ['observations', 'actions', 'rewards', 'next_observations', 'masks']
                    batch = {
                        k: np.concatenate([
                            np.asarray(off_batch[k]), np.asarray(on_batch[k])
                        ], axis=0)
                        for k in keys
                    }
                else:
                    batch = online_buffer.sample(batch_size)

                batch["global_step"] = np.int32(global_step)
                batch = {k: jnp.asarray(v) for k, v in batch.items()}
                agent, info = agent.update(batch)

            if global_step % log_interval == 0:
                row = {f"train/{k}": to_scalar(v) for k, v in flatten_dict(dict(info)).items()}
                row["train/buffer_size"] = len(online_buffer)
                row["train/env_steps"] = env_steps
                row["train/phase"] = 1
                csv_logger.log(row, global_step)
                log_to_wandb(row, global_step)

        if global_step % eval_interval == 0:
            score, stats = evaluate_and_log(
                agent=agent, eval_env=eval_env_online, cfg=cfg, run_dir=run_dir,
                csv_logger=csv_logger, step=global_step, best_score=best_score,
            )
            ret = stats.get("episode.normalized_return", stats.get("episode.return", float("nan")))
            print(f"[online  {global_step:>7d}] return: {ret:.2f}  env_steps: {env_steps}", flush=True)
            if score > best_score:
                best_score = score

        if cfg.logging.save_checkpoint and global_step % save_interval == 0:
            save_checkpoint(agent, run_dir / f"agent_step_{global_step}")

    vec_env.close()
    if cfg.logging.save_checkpoint:
        save_checkpoint(agent, run_dir / "final_agent")



# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    set_global_seed(int(cfg.seed))

    run_dir = ensure_run_dir(cfg)
    if bool(cfg.logging.get("save_config", True)):
        save_resolved_config(cfg, run_dir)
    maybe_init_wandb(cfg, run_dir)

    if cfg.train.mode == "offline":
        offline_train(cfg, run_dir)
    elif cfg.train.mode == "online":
        online_train(cfg, run_dir)
    elif cfg.train.mode == "offline_to_online":
        offline_to_online_train(cfg, run_dir)
    else:
        raise ValueError(f"Unknown train.mode: {cfg.train.mode}")

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()