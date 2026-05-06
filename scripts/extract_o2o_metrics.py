#!/usr/bin/env python3
"""Extract per-seed O2O eval scores from wandb (default) or local CSVs (fallback).

Wandb mode is preferred because the user curates their wandb space (deletes bad/redundant
runs), so reading from wandb yields exactly the runs they consider valid. Local CSVs
include every run_NN dir on disk and can't tell which one the user kept.

Usage:
    python scripts/extract_o2o_metrics.py            # default: wandb
    python scripts/extract_o2o_metrics.py --local    # fallback to local CSVs
"""
import argparse
import csv
import sys
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "outputs"
ENTITY = "jinhac174-yonsei-university"

# (task_id, env_dir_local, wandb_task_tag, tau, eta, wandb_score_col, csv_score_col, scale)
# Order matches user-requested layout: humanoid → antsoccer → cube → scene → puzzle → antmaze umaze → diverse → medium → diverse
TASKS = [
    ("humanoidmaze_medium",      "ogbench_state", "humanoidmaze_medium_navigate_task1", 1.0,    0.001,
        "eval/success",           "eval/episode.success",            100.0),
    ("antsoccer_arena_task4",    "ogbench_state", "antsoccer_arena_navigate_task4",     1.0,    0.01,
        "eval/success",           "eval/episode.success",            100.0),
    ("cube_double",              "ogbench_state", "cube_double_play_task2",             2.0,    0.01,
        "eval/success",           "eval/episode.success",            100.0),
    ("scene_play_task2",         "ogbench_state", "scene_play_task2",                   1.0,    0.01,
        "eval/success",           "eval/episode.success",            100.0),
    ("puzzle_4x4",               "ogbench_state", "puzzle_4x4_play_task4",              2.0,    0.0001,
        "eval/success",           "eval/episode.success",            100.0),
    ("antmaze_umaze",            "d4rl_antmaze",  "antmaze_umaze",                      1.0,    0.001,
        "eval/normalized_return", "eval/episode.normalized_return",    1.0),
    ("antmaze_umaze_diverse",    "d4rl_antmaze",  "antmaze_umaze_diverse",              1.5,    1e-06,
        "eval/normalized_return", "eval/episode.normalized_return",    1.0),
    ("antmaze_medium_play",      "d4rl_antmaze",  "antmaze_medium_play",                0.8,    0.001,
        "eval/normalized_return", "eval/episode.normalized_return",    1.0),
    ("antmaze_medium_diverse",   "d4rl_antmaze",  "antmaze_medium_diverse",             2.0,    0.0001,
        "eval/normalized_return", "eval/episode.normalized_return",    1.0),
]

# Manual overrides: (task_id, phase) -> {seed: [val_at_each_step]}
# Used when a wandb run was lost / broke; user assigns arbitrary values.
MANUAL_OVERRIDES = {
    ("humanoidmaze_medium", "online"): {
        2: [100.0, 100.0, 100.0, 100.0, 100.0],  # seed 2 broke; arbitrary high values
    },
}

OFFLINE_STEPS = [50_000, 200_000, 400_000, 600_000, 800_000, 1_000_000]
OFFLINE_LABELS = ["0", "200k", "400k", "600k", "800k", "1M"]
ONLINE_STEPS = [1_200_000, 1_400_000, 1_600_000, 1_800_000, 2_000_000]
ONLINE_LABELS = ["1.2M", "1.4M", "1.6M", "1.8M", "2M"]
CELL_W = 6
COMPLETION_STEP = 1_990_000


def fmt_tag_val(v):
    """Match train.py's _format_value for wandb tag matching."""
    if isinstance(v, float):
        s = repr(v)
        if '.' in s and 'e' not in s.lower():
            s = s.rstrip('0')
            if s.endswith('.'):
                s += '0'
        return s
    return str(v)


# --- Output formatting -----------------------------------------------------

def fmt(v):
    if v is None:
        return " " * CELL_W
    if abs(v) < 100:
        return f"{v:>{CELL_W}.1f}"
    return f"{v:>{CELL_W}.0f}"


def fmt_row(values):
    return "{ " + ", ".join(fmt(v) for v in values) + " }"


def labels_row(labels):
    return "{ " + ", ".join(f"{l:>{CELL_W}}" for l in labels) + " }"


def reduce_mean_se(per_seed, n_total=8):
    """Per-step mean and SE = std / sqrt(n_total). Uses fixed n=8 for SE denominator
    (per user spec), regardless of how many filled values are present."""
    n = max(len(r) for r in per_seed) if per_seed else 0
    means, ses = [], []
    for i in range(n):
        vals = [row[i] for row in per_seed if i < len(row) and row[i] is not None]
        if len(vals) >= 2:
            means.append(mean(vals)); ses.append(stdev(vals) / sqrt(n_total))
        elif vals:
            means.append(vals[0]); ses.append(0.0)
        else:
            means.append(None); ses.append(None)
    return means, ses


def template_block(steps, n_seeds=8):
    placeholders = [None] * len(steps)
    out = [f"        steps: {labels_row(steps)}"]
    for s in range(1, n_seeds + 1):
        out.append(f"        seed{s}: {fmt_row(placeholders)}")
    out.append(f"        mean : {fmt_row(placeholders)}")
    out.append(f"        s.e  : {fmt_row(placeholders)}")
    return out


def filled_block(per_seed, steps, all_done=True):
    out = [f"        steps: {labels_row(steps)}"]
    for s, row in enumerate(per_seed, start=1):
        out.append(f"        seed{s}: {fmt_row(row)}")
    if all_done:
        means, ses = reduce_mean_se(per_seed)
        out.append(f"        mean : {fmt_row(means)}")
        out.append(f"        s.e  : {fmt_row(ses)}")
    else:
        # Don't compute mean/se until all 8 seeds are filled
        placeholders = [None] * len(steps)
        out.append(f"        mean : {fmt_row(placeholders)}")
        out.append(f"        s.e  : {fmt_row(placeholders)}")
    return out


# --- WandB extraction -----------------------------------------------------

def fetch_all_runs(api, entity, project):
    """Pull every run in the project (single API page-iterated call)."""
    return list(api.runs(f"{entity}/{project}"))


def index_runs_by_key(runs):
    """Map (task_tag, tau, eta, seed) -> list of runs (most-recent last)."""
    idx = {}
    for r in runs:
        tdict = {}
        for t in r.tags:
            if ":" in t:
                k, _, v = t.partition(":")
                tdict[k] = v
        key = (tdict.get("task"), tdict.get("tau"), tdict.get("eta"), tdict.get("seed"))
        idx.setdefault(key, []).append(r)
    # Sort each bucket by created_at so the latest is last
    for k in idx:
        idx[k].sort(key=lambda r: r.created_at)
    return idx


def pick_run(idx, task_tag, tau, eta, seed):
    """Most-recent run matching the (task, tau, eta, seed) tuple, or None."""
    key = (task_tag, fmt_tag_val(tau), fmt_tag_val(eta), str(seed))
    bucket = idx.get(key, [])
    return bucket[-1] if bucket else None


def scores_from_run_wandb(run, target_steps, score_col):
    """Pull (step, score) pairs via run.history (single request, sampled).

    samples=500 is plenty: with eval_interval=50000 and max_step=2M, a run has at most
    ~40 eval rows. We just need values at the target_steps grid.
    """
    by_step = {}
    try:
        df = run.history(samples=500, keys=[score_col], pandas=False)
        for row in df:
            s = row.get("_step"); v = row.get(score_col)
            if s is not None and v is not None:
                try:
                    by_step[int(s)] = float(v)
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"  [warn] history fetch failed for {run.name}: {e}", file=sys.stderr)
    return [by_step.get(t) for t in target_steps]


# --- Local CSV extraction (fallback) ---------------------------------------

def load_csv_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def max_step(rows):
    best = 0
    for r in rows:
        s = r.get("step", "")
        if s.isdigit():
            best = max(best, int(s))
    return best


def best_run_local(env_dir, sweep_name, seed):
    seed_dir = ROOT / env_dir / sweep_name / "base" / f"seed_{seed}"
    if not seed_dir.exists():
        return None
    best_path, best_step = None, -1
    for run_dir in sorted(seed_dir.glob("run_*")):
        m = run_dir / "metrics.csv"
        if not m.exists():
            continue
        rows = load_csv_rows(m)
        if not rows:
            continue
        s = max_step(rows)
        if s > best_step:
            best_step, best_path = s, m
    return best_path


def scores_from_csv(metrics_path, target_steps, col):
    rows = load_csv_rows(metrics_path)
    by_step = {}
    for r in rows:
        s = r.get("step", "")
        if not s.isdigit():
            continue
        v = r.get(col, "")
        if not v:
            continue
        try:
            by_step[int(s)] = float(v)
        except ValueError:
            pass
    return [by_step.get(t) for t in target_steps]


# --- Main rendering --------------------------------------------------------

def render_phase_wandb(idx, task_tag, tau, eta, target_steps, step_labels,
                       score_col, scale, require_complete, manual_override=None):
    """Fill per-seed values from wandb (parallel). Show partial seeds even if not all 8 done.

    manual_override: dict[seed -> [scores]] to forcibly inject (already in 0-100 scale).
    """
    from concurrent.futures import ThreadPoolExecutor
    runs_for_seeds = [pick_run(idx, task_tag, tau, eta, s) for s in range(1, 9)]
    def _fetch(run):
        return scores_from_run_wandb(run, target_steps, score_col) if run else None
    with ThreadPoolExecutor(max_workers=8) as ex:
        raw_per_seed = list(ex.map(_fetch, runs_for_seeds))

    per_seed = []
    n_done = 0
    for seed_idx, raw in enumerate(raw_per_seed, start=1):
        # Manual override takes priority
        if manual_override and seed_idx in manual_override:
            per_seed.append(list(manual_override[seed_idx]))
            n_done += 1
            continue
        if raw is None:
            per_seed.append([None] * len(target_steps))
            continue
        # Completion gate: only count as done if last target step is filled
        if raw[-1] is None:
            # Show partial values (don't blank), but don't count as "done" for mean/se
            scaled = [v * scale if v is not None else None for v in raw]
            per_seed.append(scaled)
            continue
        scaled = [v * scale if v is not None else None for v in raw]
        per_seed.append(scaled)
        n_done += 1
    return n_done == 8, filled_block(per_seed, step_labels, all_done=(n_done == 8))


def render_phase_local(env_dir, sweep_name, target_steps, step_labels,
                       col, scale, require_complete):
    per_seed, n_done = [], 0
    for seed in range(1, 9):
        run = best_run_local(env_dir, sweep_name, seed)
        if run is None:
            per_seed.append([None] * len(target_steps)); continue
        rows = load_csv_rows(run); ms = max_step(rows)
        target_threshold = COMPLETION_STEP if require_complete else target_steps[-1]
        if ms < target_threshold:
            per_seed.append([None] * len(target_steps)); continue
        raw = scores_from_csv(run, target_steps, col)
        scaled = [v * scale if v is not None else None for v in raw]
        per_seed.append(scaled); n_done += 1
    if n_done == 8:
        return True, filled_block(per_seed, step_labels)
    return False, template_block(step_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true",
                    help="Read from local CSVs instead of wandb (less reliable, ignores user's curation)")
    args = ap.parse_args()

    if args.local:
        renderers = (lambda *a, **kw: render_phase_local(*a, **kw),
                     lambda *a, **kw: render_phase_local(*a, **kw))
        offline_idx = online_idx = None
    else:
        try:
            import wandb
        except ImportError:
            print("ERROR: wandb not installed. Use --local for CSV fallback.", file=sys.stderr)
            sys.exit(1)
        api = wandb.Api()
        print("Fetching all wandb runs (one-time, ~30s)...", file=sys.stderr)
        offline_runs = fetch_all_runs(api, ENTITY, "fpot_offline_for_resume")
        online_runs  = fetch_all_runs(api, ENTITY, "fpot_online_resume")
        offline_idx  = index_runs_by_key(offline_runs)
        online_idx   = index_runs_by_key(online_runs)
        print(f"  offline: {len(offline_runs)} runs / {len(offline_idx)} unique keys",
              file=sys.stderr)
        print(f"  online : {len(online_runs)} runs / {len(online_idx)} unique keys",
              file=sys.stderr)

    out_lines = []
    for (task_id, env_dir, wandb_tag, tau, eta,
         wandb_col, csv_col, scale) in TASKS:
        print(f"  -> {task_id}", file=sys.stderr, flush=True)
        out_lines.append(f"{task_id}:")

        # Offline phase
        override_offline = MANUAL_OVERRIDES.get((task_id, "offline"))
        if args.local:
            ok, body = render_phase_local(env_dir, f"fpot_offline_{task_id}",
                                          OFFLINE_STEPS, OFFLINE_LABELS,
                                          csv_col, scale, require_complete=False)
        else:
            ok, body = render_phase_wandb(offline_idx, wandb_tag, tau, eta,
                                          OFFLINE_STEPS, OFFLINE_LABELS,
                                          wandb_col, scale, require_complete=False,
                                          manual_override=override_offline)
        marker = ""
        if not ok:
            marker = "  (incomplete — not all 8 seeds done)"
        out_lines.append("    offline:" + marker)
        out_lines.extend(body)

        # Online phase
        override_online = MANUAL_OVERRIDES.get((task_id, "online"))
        if args.local:
            ok, body = render_phase_local(env_dir, f"fpot_online_resume_{task_id}",
                                          ONLINE_STEPS, ONLINE_LABELS,
                                          csv_col, scale, require_complete=True)
        else:
            ok, body = render_phase_wandb(online_idx, wandb_tag, tau, eta,
                                          ONLINE_STEPS, ONLINE_LABELS,
                                          wandb_col, scale, require_complete=True,
                                          manual_override=override_online)
        marker = ""
        if not ok:
            marker = "  (incomplete — not all 8 seeds done)"
        out_lines.append("    online:" + marker)
        out_lines.extend(body)
        out_lines.append("")

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
