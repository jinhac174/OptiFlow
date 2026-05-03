"""Resolve a sweep yaml + idx into a concrete train.py invocation.

Sweep keys iterate in declaration order: first key = outermost loop (slowest),
last key = innermost (fastest). Use `count` to query total jobs and `run` to
exec one (with `--dry-run` to print the command without executing).
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import yaml


# ---- Sweep loading & product -----------------------------------------------

def load_sweep(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_product(sweep_axes: dict):
    keys = list(sweep_axes.keys())
    value_lists = [sweep_axes[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield list(zip(keys, combo))


def count_jobs(sweep_axes: dict) -> int:
    total = 1
    for vals in sweep_axes.values():
        total *= len(vals)
    return total


def explicit_jobs(cfg: dict) -> list[dict]:
    jobs = cfg.get("jobs")
    if jobs is None:
        return []
    if not isinstance(jobs, list):
        raise TypeError("'jobs' must be a list of override mappings")
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(f"jobs[{i}] must be a mapping")
    return jobs


def total_jobs(cfg: dict) -> int:
    jobs = explicit_jobs(cfg)
    if jobs:
        return len(jobs)
    return count_jobs(cfg.get("sweep", {}))


def resolve_overrides(cfg: dict, idx: int) -> list[str]:
    jobs = explicit_jobs(cfg)
    n = total_jobs(cfg)
    if not (0 <= idx < n):
        raise ValueError(f"idx {idx} out of range [0, {n})")

    if jobs:
        sweep_pairs = list(jobs[idx].items())
    else:
        axes = cfg.get("sweep", {})
        product = list(iter_product(axes))
        sweep_pairs = product[idx]

    overrides = []
    for key, value in sweep_pairs:
        overrides.append(f"{key}={value}")

    for key, value in (cfg.get("fixed") or {}).items():
        overrides.append(f"{key}={value}")

    return overrides


def build_command(cfg: dict, idx: int, extra_overrides=None) -> list[str]:
    overrides = resolve_overrides(cfg, idx)

    overrides.append(f"+exp={cfg['name']}")

    wandb_project = cfg.get("wandb_project")
    if wandb_project:
        overrides.append(f"+logging.wandb_project={wandb_project}")

    # Tell train.py which keys are sweep-fixed (so wandb name/group skip them).
    fixed_keys = list((cfg.get("fixed") or {}).keys())
    if fixed_keys:
        overrides.append("+logging.fixed_keys=[" + ",".join(fixed_keys) + "]")

    if extra_overrides:
        overrides.extend(extra_overrides)

    return [sys.executable, "scripts/train.py"] + overrides


# ---- CLI -------------------------------------------------------------------

def cmd_count(args):
    cfg = load_sweep(args.sweep)
    print(total_jobs(cfg))


def cmd_run(args):
    cfg = load_sweep(args.sweep)
    cmd = build_command(cfg, args.idx, extra_overrides=args.override)
    print(" ".join(cmd))
    if not args.dry_run:
        os.execv(sys.executable, cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_count = sub.add_parser("count", help="Print total job count for a sweep")
    p_count.add_argument("--sweep", required=True, help="Path to sweep yaml")

    p_run = sub.add_parser("run", help="Resolve and execute one sweep job")
    p_run.add_argument("--sweep", required=True, help="Path to sweep yaml")
    p_run.add_argument("--idx", required=True, type=int, help="Array task index")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print resolved command without executing")
    p_run.add_argument("--override", action="append", default=[],
                       help="Extra Hydra override appended to the command "
                            "(repeatable, e.g. --override logging.save_csv=false)")

    args = parser.parse_args()
    if args.command == "count":
        cmd_count(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()