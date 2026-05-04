#!/usr/bin/env python3
"""Resolve an offline sweep's outputs to an online-resume sweep yaml.

For each seed in the offline sweep, scan `outputs/<env>/<sweep_name>/base/seed_N/run_*/`
for the latest `final_agent/params_0.pkl`, and emit an online sweep yaml whose
`jobs:` list maps each found seed to a concrete `train.resume_from` path. The
`fixed:` block inherits the offline yaml's hyperparameters but switches `train`
to `offline_to_online` and sets explicit step budgets.

Seeds with no offline checkpoint are skipped (and reported on stderr).
"""
import argparse
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline-sweep", required=True, help="Offline sweep yaml (input)")
    ap.add_argument("--output", required=True, help="Online sweep yaml (output)")
    ap.add_argument("--root", default="outputs", help="Outputs root dir (default: outputs)")
    ap.add_argument("--offline-steps", type=int, default=1_000_000)
    ap.add_argument("--online-steps", type=int, default=1_000_000)
    ap.add_argument("--eval-interval", type=int, default=50_000)
    args = ap.parse_args()

    with open(args.offline_sweep) as f:
        offline = yaml.safe_load(f)

    fixed = offline.get("fixed") or {}
    env_template = fixed["env"]
    env_name = fixed["env.env_name"]
    seeds = offline["sweep"]["seed"]
    sweep_name = offline["name"]

    jobs = []
    missing = []
    for s in seeds:
        seed_dir = Path(args.root) / env_template / sweep_name / "base" / f"seed_{s}"
        ckpt = None
        if seed_dir.exists():
            for run_dir in sorted(seed_dir.glob("run_*")):
                if (run_dir / "final_agent" / "params_0.pkl").exists():
                    ckpt = run_dir / "final_agent"
        if ckpt is None:
            missing.append(s)
            continue
        jobs.append({"seed": s, "train.resume_from": str(ckpt)})

    if missing:
        print(f"WARNING: missing offline ckpt for seeds: {missing}", file=sys.stderr)
    if not jobs:
        print("ERROR: no offline ckpts found — refusing to write empty online yaml.",
              file=sys.stderr)
        sys.exit(1)

    online_name = sweep_name.replace("offline", "online", 1)
    online_project = (offline.get("wandb_project") or "fpot_online").replace(
        "offline", "online", 1)

    online_fixed = {k: v for k, v in fixed.items()
                    if not (k.startswith("train") or k.startswith("logging"))}
    online_fixed["train"] = "offline_to_online"
    online_fixed["train.offline_max_steps"] = args.offline_steps
    online_fixed["train.online_max_steps"] = args.online_steps
    online_fixed["train.eval_interval"] = args.eval_interval

    out = {
        "name": online_name,
        "description": f"Online resume from {sweep_name} ckpts ({len(jobs)} seeds)",
        "wandb_project": online_project,
        "jobs": jobs,
        "fixed": online_fixed,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {args.output} with {len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
