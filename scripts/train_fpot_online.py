"""
Offline-to-online fine-tuning for FPOT.

Loads an offline 1M-step checkpoint, pre-fills a unified replay buffer with
the offline dataset, then continues training with online environment interaction.
agent.update() is called unchanged — no new losses are introduced.

Usage example:
    python scripts/train_fpot_online.py \
        --checkpoint_path logs/runs/cube-single.../final_agent \
        --env_name cube-single-play-singletask-task1-v0 \
        --online_steps 500000 \
        --seed 1 \
        --wandb_project fpot-online-cube-single
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from FPOT.agent_fpot import FPOTAgent
from envs.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.replay_buffer import ReplayBuffer

try:
    import wandb
except ImportError:
    wandb = None


WANDB_METRICS = {
    "eval/episode.return":            "eval/return",
    "eval/episode.normalized_return": "eval/normalized_return",
    "eval/episode.success":           "eval/success",
    "eval/success":                   "eval/success",
    "train/critic/critic_loss":       "critic/loss",
    "train/critic/q_mean":            "critic/q_mean",
    "train/critic/q_max":             "critic/q_max",
    "train/critic/q_min":             "critic/q_min",
    "train/actor/loss":               "actor/loss",
    "train/actor/bc/loss":            "actor/bc_loss",
    "train/actor/distill/loss":       "actor/distill_loss",
    "train/actor/qmax/loss":          "actor/qmax_loss",
    "train/buffer_size":              "online/buffer_size",
    "train/env_steps":                "online/env_steps",
}


def parse_args():
    p = argparse.ArgumentParser(description="FPOT offline-to-online fine-tuning")
    p.add_argument("--checkpoint_path", required=True,
                   help="Directory containing params_0.pkl (offline checkpoint)")
    p.add_argument("--env_name", required=True,
                   help="OGBench or D4RL environment name")
    p.add_argument("--offline_dataset_path", default=None,
                   help="Unused; dataset is loaded via make_env_and_datasets")
    p.add_argument("--online_steps", type=int, default=500_000,
                   help="Number of online gradient update steps")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--replay_capacity", type=int, default=2_000_000,
                   help="Total replay buffer capacity (offline + online transitions)")
    p.add_argument("--eval_interval", type=int, default=50_000)
    p.add_argument("--save_interval", type=int, default=100_000,
                   help="Checkpoint save interval (0 to disable)")
    p.add_argument("--gradient_updates_per_env_step", type=int, default=1,
                   help="Number of gradient updates per environment step")
    p.add_argument("--deterministic_online_collection", action="store_true",
                   help="Use student one-step policy for collection (always true for FPOTAgent)")
    p.add_argument("--lambda_q_online", type=float, default=0.0,
                   help="Override lambda_q for online phase (default 0 = no Q-max loss)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_eval_episodes", type=int, default=50)
    p.add_argument("--run_dir", default="logs/fpot_online",
                   help="Output directory for logs and checkpoints")
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_mode", default="disabled",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def to_scalar(v):
    arr = np.asarray(v)
    return arr.item() if arr.shape == () else arr


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key, sep=sep))
        else:
            items[key] = v
    return items


def log_to_wandb(row, step):
    if wandb is None or wandb.run is None:
        return
    out = {v: row[k] for k, v in WANDB_METRICS.items() if k in row}
    if out:
        wandb.log(out, step=step)


def populate_buffer(buffer, dataset, max_transitions):
    """Pre-fill replay buffer from offline dataset in chunks."""
    n = min(dataset.size, max_transitions)
    obs   = np.asarray(dataset["observations"])
    acts  = np.asarray(dataset["actions"])
    rews  = np.asarray(dataset["rewards"])
    nobs  = np.asarray(dataset["next_observations"])
    if "masks" in dataset:
        dones = (1.0 - np.asarray(dataset["masks"])).astype(np.float32)
    else:
        dones = np.asarray(dataset["terminals"]).astype(np.float32)

    chunk = 4096
    loaded = 0
    with tqdm(total=n, desc="loading offline data", unit="trans", file=sys.stdout) as pbar:
        while loaded < n:
            end = min(loaded + chunk, n)
            buffer.add_batch(
                obs[loaded:end], acts[loaded:end], rews[loaded:end],
                nobs[loaded:end], dones[loaded:end],
            )
            pbar.update(end - loaded)
            loaded = end

    print(f"Replay buffer: {buffer.size} transitions loaded from offline dataset.")


def load_agent_config(checkpoint_path: Path) -> dict:
    """Load agent config from config.json adjacent to or above the checkpoint dir."""
    for candidate in [checkpoint_path / "config.json",
                      checkpoint_path.parent / "config.json",
                      checkpoint_path.parent.parent / "config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                cfg = json.load(f)
            agent_cfg = cfg["agent"]
            for key in ("actor_hidden_dims", "value_hidden_dims"):
                if key in agent_cfg and isinstance(agent_cfg[key], list):
                    agent_cfg[key] = tuple(agent_cfg[key])
            return agent_cfg
    raise FileNotFoundError(
        f"config.json not found near {checkpoint_path}. "
        "Ensure the offline run_dir is accessible."
    )


def save_ckpt(agent, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    save_agent(agent, str(path), epoch=0)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Environment & dataset ---
    collect_env, eval_env, train_dataset, _ = make_env_and_datasets(
        env_name=args.env_name
    )

    obs_dim    = train_dataset["observations"].shape[1]
    action_dim = train_dataset["actions"].shape[1]

    # --- Replay buffer: pre-fill with offline data ---
    buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    populate_buffer(buffer, train_dataset, args.replay_capacity)

    # --- Build agent from config + example batch ---
    ckpt_path = Path(args.checkpoint_path).resolve()
    agent_cfg = load_agent_config(ckpt_path)
    agent_cfg["lambda_q"] = args.lambda_q_online  # disable Q-max in online phase

    ex_obs = np.asarray(train_dataset["observations"][:2])
    ex_act = np.asarray(train_dataset["actions"][:2])
    agent = FPOTAgent.create(
        seed=args.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=agent_cfg,
    )

    # --- Restore offline checkpoint ---
    agent = restore_agent(agent, str(ckpt_path), restore_epoch=0)

    # --- WandB ---
    if wandb is not None and args.wandb_mode != "disabled" and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(run_dir),
        )

    # --- Online training loop ---
    collect_rng = jax.random.PRNGKey(args.seed + 1000)
    obs, _ = collect_env.reset()
    env_steps = 0
    best_score = -np.inf
    info_train = {}

    progress = tqdm(
        range(1, args.online_steps + 1),
        total=args.online_steps,
        desc="fpot online",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for step in progress:
        # --- Collect one transition using student one-step policy ---
        collect_rng, act_rng = jax.random.split(collect_rng)
        action = np.asarray(
            agent.sample_actions(np.asarray(obs, dtype=np.float32)[None], seed=act_rng)
        )[0]
        next_obs, reward, terminated, truncated, _ = collect_env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, float(reward), next_obs, done)
        obs = collect_env.reset()[0] if done else next_obs
        env_steps += 1

        # --- Gradient updates ---
        if len(buffer) >= args.batch_size:
            for _ in range(args.gradient_updates_per_env_step):
                batch = buffer.sample(args.batch_size)
                batch["global_step"] = np.int32(step)
                batch = {k: jnp.asarray(v) for k, v in batch.items()}
                agent, info_train = agent.update(batch)

        # --- Training log ---
        if step % args.log_interval == 0 and info_train:
            row = {f"train/{k}": to_scalar(v)
                   for k, v in flatten_dict(dict(info_train)).items()}
            row["train/buffer_size"] = len(buffer)
            row["train/env_steps"]   = env_steps
            log_to_wandb(row, step)
            actor_loss  = row.get("train/actor/loss", float("nan"))
            critic_loss = row.get("train/critic/critic_loss", float("nan"))
            progress.set_postfix(
                actor=f"{actor_loss:.4g}",
                critic=f"{critic_loss:.4g}",
                env_steps=env_steps,
            )
            tqdm.write(
                f"[step {step:>7d}] actor={actor_loss:.4g}  "
                f"critic={critic_loss:.4g}  env_steps={env_steps}",
                file=sys.stdout,
            )

        # --- Evaluation ---
        if step % args.eval_interval == 0:
            eval_stats, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                num_eval_episodes=args.num_eval_episodes,
                num_video_episodes=0,
                progress=False,
            )
            score = eval_stats.get(
                "episode.normalized_return",
                eval_stats.get("episode.return", -np.inf),
            )
            success = eval_stats.get("success", None)
            eval_row = {f"eval/{k}": to_scalar(v)
                        for k, v in flatten_dict(dict(eval_stats)).items()}
            log_to_wandb(eval_row, step)
            tqdm.write(
                f"[step {step:>7d}] eval return={score:.3f}"
                + (f"  success={100*success:.1f}%" if success is not None else ""),
                file=sys.stdout,
            )
            if score > best_score:
                best_score = score
                save_ckpt(agent, run_dir / "best_agent")

        # --- Periodic checkpoint ---
        if args.save_interval > 0 and step % args.save_interval == 0:
            save_ckpt(agent, run_dir / f"agent_step_{step}")

    save_ckpt(agent, run_dir / "final_agent")
    print(f"Done. Best score: {best_score:.4f}")

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
