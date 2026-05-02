"""sweep_runner.py — translate a sweep yaml into concrete train.py invocations.

Cartesian-product iteration order
----------------------------------
Keys are taken from the ``sweep:`` block in **declaration order** (PyYAML
preserves insertion order for mappings in Python 3.7+).  The FIRST key is the
outermost loop (changes slowest) and the LAST key is the innermost loop
(changes fastest).  This matches numpy/itertools.product convention.

Example — sweep keys [env, agent.w_temperature, seed] with sizes [3, 2, 2]:
  idx 0  → env[0]  w_temperature[0]  seed[0]
  idx 1  → env[0]  w_temperature[0]  seed[1]
  idx 2  → env[0]  w_temperature[1]  seed[0]
  ...
  idx 11 → env[2]  w_temperature[1]  seed[1]

Commands
---------
  python scripts/sweep_runner.py count --sweep <yaml>
      Prints the integer total number of jobs (Cartesian product size).

  python scripts/sweep_runner.py run --sweep <yaml> --idx <int> [--dry-run]
      Resolves overrides for job <idx>, prints the full train.py command, then
      exec()s it (unless --dry-run is set, in which case it only prints).
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Sweep loading & product
# ---------------------------------------------------------------------------

def load_sweep(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_product(sweep_axes: dict):
    """Yield (key, value) lists for every cell in the Cartesian product."""
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
    """Return explicitly enumerated jobs, if the sweep yaml defines them."""
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
    """Return the override list for job index ``idx``."""
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


def build_command(cfg: dict, idx: int) -> list[str]:
    """Build the full argv list for one job (does NOT include 'python')."""
    overrides = resolve_overrides(cfg, idx)

    # Always inject +exp=<sweep_name> so wandb gets an exp: tag
    sweep_name = cfg["name"]
    overrides.append(f"+exp={sweep_name}")

    # Inject wandb_project from yaml if present (overrides default cfg.env.name routing)
    wandb_project = cfg.get("wandb_project")
    if wandb_project:
        overrides.append(f"+logging.wandb_project={wandb_project}")

    fixed_keys = list((cfg.get("fixed") or {}).keys())
    if fixed_keys:
        overrides.append("+logging.fixed_keys=[" + ",".join(fixed_keys) + "]")

    # Hydra needs the script path; we invoke scripts/train.py from repo root
    cmd = [sys.executable, "scripts/train.py"] + overrides
    return cmd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_count(args):
    cfg = load_sweep(args.sweep)
    print(total_jobs(cfg))


def cmd_run(args):
    cfg = load_sweep(args.sweep)
    cmd = build_command(cfg, args.idx)
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

    args = parser.parse_args()
    if args.command == "count":
        cmd_count(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()