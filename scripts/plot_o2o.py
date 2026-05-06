#!/usr/bin/env python3
"""Plot O2O learning curves per task across methods (OptiFlow / IFQL / FQL).

Each figure: x = global step (0 → 2M), y = score (0 → 100). Offline phase (0–1M)
shaded light grey, online (1M–2M) white, to visually separate. One line per
method (mean over 8 seeds, ±1 SE band).

Methods other than OptiFlow are stubbed; if their wandb projects are not yet
populated, those lines simply don't appear on the plot.

Usage:
    python scripts/plot_o2o.py                      # all tasks
    python scripts/plot_o2o.py humanoidmaze_medium  # one task
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

# Publication-ready typography: serif body (Times-equivalent), STIX math.
# Sizes tuned so the figure remains legible at paper-column scale (~2 in wide).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.dpi": 600,
    "figure.dpi": 300,
})

# ----- Reuse the helpers from extract_o2o_metrics.py to avoid duplication -----
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_o2o_metrics import (  # noqa: E402
    ENTITY,
    TASKS,
    fetch_all_runs,
    fmt_tag_val,
    index_runs_by_key,
    pick_run,
    scores_from_run_wandb,
    MANUAL_OVERRIDES,
)
from parse_o2o_text import load_text_data  # noqa: E402

# Step grid: every 200k from 200K to 2M (10 measured points). The (0, 0) anchor is
# prepended at plot time, so the visible curve is at x ∈ {0, 200K, 400K, …, 2M}.
PLOT_STEPS = [200_000, 400_000, 600_000, 800_000, 1_000_000,
              1_200_000, 1_400_000, 1_600_000, 1_800_000, 2_000_000]

# Methods table. Each entry: project_offline, project_online, color.
# Only OptiFlow has data right now; IFQL and FQL placeholders for later.
# Order is the legend layout: OptiFlow → FQL → IFQL.
METHODS = {
    "OptiFlow": {
        "source": "wandb",
        "offline_project": "fpot_offline_for_resume",
        "online_project":  "fpot_online_resume",
        "color":           "#1f78b4",
    },
    "FQL": {
        "source": "wandb",
        "offline_project": "fpot_fql_offline",     # placeholder — update when runs exist
        "online_project":  "fpot_fql_online",
        "color":           "#33a02c",
    },
    "IFQL": {
        "source": "text",
        "data_file": "ifql.txt",
        "color":     "#e31a1c",
    },
}

# Maps TASKS[].task_id -> the key used in text-format data files (ifql.txt etc.).
# Most are identical; antsoccer/scene drop the task-suffix the wandb tag carries.
TEXT_KEY_MAP = {
    "antsoccer_arena_task4": "antsoccer_arena",
    "scene_play_task2":      "scene",
}

TASK_DISPLAY = {
    "humanoidmaze_medium":     "humanoidmaze-medium{1}",
    "antsoccer_arena_task4":   "antsoccer-arena{4}",
    "cube_double":             "cube-double{2}",
    "scene_play_task2":        "scene-play{2}",
    "puzzle_4x4":              "puzzle-4x4{4}",
    "antmaze_umaze":           "antmaze-umaze",
    "antmaze_umaze_diverse":   "antmaze-umaze-diverse",
    "antmaze_medium_play":     "antmaze-medium-play",
    "antmaze_medium_diverse":  "antmaze-medium-diverse",
    "antmaze_large_play":      "antmaze-large-play",
    "antmaze_large_diverse":   "antmaze-large-diverse",
    "pen_cloned":              "pen-cloned",
    "door_cloned":             "door-cloned",
    "hammer_cloned":           "hammer-cloned",
    "relocate_cloned":         "relocate-cloned",
}

OFFLINE_RANGE = (0, 1_000_000)


def _reduce_per_seed(per_seed):
    """Aggregate 8 per-seed rows (each len(PLOT_STEPS)) into (means, ses).
    Returns (None, None) if any seed is incomplete at the final step."""
    if any(row[-1] is None for row in per_seed):
        return None, None
    means, ses = [], []
    for i in range(len(PLOT_STEPS)):
        vals = [row[i] for row in per_seed if i < len(row) and row[i] is not None]
        if len(vals) >= 2:
            means.append(mean(vals)); ses.append(stdev(vals) / sqrt(8))
        elif vals:
            means.append(vals[0]); ses.append(0.0)
        else:
            means.append(None); ses.append(None)
    return means, ses


def collect_method_curve(api, method_indices, task_id, task_tag, tau, eta, score_col, scale):
    """Return (mean, se) per PLOT_STEPS for one wandb method, or (None, None) if no data."""
    if method_indices is None or method_indices == (None, None):
        return None, None
    overrides = MANUAL_OVERRIDES
    off_steps = [s for s in PLOT_STEPS if s <= 1_000_000]
    on_steps  = [s for s in PLOT_STEPS if s >  1_000_000]

    off_idx, on_idx = method_indices

    def fetch_seed(idx, target_steps, seed, override_seeds):
        run = pick_run(idx, task_tag, tau, eta, seed)
        raw = scores_from_run_wandb(run, target_steps, score_col) if run is not None else None
        # Override only applies if there's no real run reaching the final target step.
        has_real_run = raw is not None and raw[-1] is not None
        if override_seeds and seed in override_seeds and not has_real_run:
            return list(override_seeds[seed])  # already 0-100, no scale
        if raw is None:
            return [None] * len(target_steps)
        return [v * scale if v is not None else None for v in raw]

    override_off = overrides.get((task_id, "offline"))
    override_on  = overrides.get((task_id, "online"))

    with ThreadPoolExecutor(max_workers=16) as ex:
        off_seeds = list(ex.map(lambda s: fetch_seed(off_idx, off_steps, s, override_off), range(1, 9)))
        on_seeds  = list(ex.map(lambda s: fetch_seed(on_idx,  on_steps,  s, override_on),  range(1, 9)))

    per_seed = [list(off) + list(on) for off, on in zip(off_seeds, on_seeds)]
    return _reduce_per_seed(per_seed)


def collect_method_curve_text(text_data, task_id):
    """Build (mean, se) per PLOT_STEPS from a parsed text data file.
    Text offline has 6 cols (incl. step 0); we drop col 0 because PLOT_STEPS starts at 200k.
    Online has 5 cols (1.2M..2M). Concatenated => 10 values aligned with PLOT_STEPS.
    """
    if not text_data:
        return None, None
    key = TEXT_KEY_MAP.get(task_id, task_id)
    block = text_data.get(key)
    if block is None:
        return None, None
    off = block.get("offline", [])
    on  = block.get("online",  [])
    if len(off) != 8 or len(on) != 8:
        return None, None
    per_seed = []
    for off_row, on_row in zip(off, on):
        # Drop step-0 column (always 0.0); keep cols 1..5 of offline + cols 0..4 of online.
        if len(off_row) < 6 or len(on_row) < 5:
            return None, None
        per_seed.append(list(off_row[1:6]) + list(on_row[0:5]))
    return _reduce_per_seed(per_seed)


def plot_legend(out_dir):
    """Standalone horizontal legend (one row, all 3 methods, inline). Cropped tight."""
    fig, ax = plt.subplots(figsize=(3.6, 0.32))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=cfg["color"], linewidth=1.8)
               for cfg in METHODS.values()]
    labels = list(METHODS.keys())
    leg = ax.legend(
        handles, labels,
        loc="center", ncol=len(labels),
        frameon=True, fontsize=9,
        handlelength=1.6, handletextpad=0.45, columnspacing=1.6,
        borderpad=0.4, borderaxespad=0,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("#444444")

    # Save with the legend's exact bbox so there is no whitespace L/R/T/B of the box.
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    out_path = out_dir / "legend.png"
    fig.savefig(out_path, dpi=600, bbox_inches=bbox, pad_inches=0)
    plt.close(fig)
    print(f"  wrote {out_path}", file=sys.stderr)


def plot_task(api, task_entry, out_dir, fetched_indices, text_data, index):
    (task_id, env_dir, wandb_tag, tau, eta,
     wandb_col, csv_col, scale) = task_entry

    DASH_COLOR = "#d4d4d4"  # light grey for guideline + boundary dashes
    SHADE_COLOR = "#f4f4f4"  # very light shade for offline half

    # Fixed canvas (figsize) + fixed plot-grid placement (subplots_adjust) so all 15 panels
    # save at IDENTICAL dimensions and align in a 5-col × 3-row paper grid.
    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.85, bottom=0.20)

    # Axis bounds: small visual gap on each side.
    x_pad, y_pad = 60_000, 3.5
    ax.set_xlim(-x_pad, 2_000_000 + x_pad)
    ax.set_ylim(-y_pad, 100 + y_pad)

    # Offline shade — extends to the LEFT edge of the plot grid (covers the x_pad gap).
    ax.axvspan(-x_pad, OFFLINE_RANGE[1], color=SHADE_COLOR, zorder=0)

    for method_name, cfg in METHODS.items():
        try:
            if cfg.get("source") == "text":
                means, ses = collect_method_curve_text(
                    text_data.get(method_name), task_id,
                )
            else:
                method_indices = fetched_indices.get(method_name)
                if method_indices is None:
                    continue
                means, ses = collect_method_curve(
                    api, method_indices, task_id, wandb_tag, tau, eta, wandb_col, scale,
                )
        except Exception as e:
            print(f"  [{task_id} / {method_name}] error: {e}", file=sys.stderr)
            continue
        if means is None:
            continue

        # Anchor the curve at (0, 0); then plot at {200K, 400K, …, 2M} where data exists.
        steps = [0] + [s for s, m in zip(PLOT_STEPS, means) if m is not None]
        ms    = [0.0] + [m for m in means if m is not None]
        es    = [0.0] + [e for e, m in zip(ses, means) if m is not None]
        if len(steps) <= 1:
            continue

        ax.plot(steps, ms, color=cfg["color"], linewidth=1.4, zorder=3)
        ax.fill_between(steps,
                        [m - (e or 0) for m, e in zip(ms, es)],
                        [m + (e or 0) for m, e in zip(ms, es)],
                        color=cfg["color"], alpha=0.18, linewidth=0, zorder=2)

    # Light dashed guidelines at y=50 + x=1M (mid-perf / offline-online split).
    ax.axhline(50,        linestyle=(0, (4, 3)), color=DASH_COLOR, linewidth=0.55, zorder=1)
    ax.axvline(1_000_000, linestyle=(0, (4, 3)), color=DASH_COLOR, linewidth=0.55, zorder=1)

    # Dashed markers at the canonical 0/2M & 0/100 edges.
    for x in (0, 2_000_000):
        ax.axvline(x, linestyle=(0, (4, 3)), color=DASH_COLOR, linewidth=0.55, zorder=1)
    for y in (0, 100):
        ax.axhline(y, linestyle=(0, (4, 3)), color=DASH_COLOR, linewidth=0.55, zorder=1)

    ax.set_xticks([0, 1_000_000, 2_000_000])
    ax.set_xticklabels(["0", "1M", "2M"])
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"])

    # Axis labels only on the leftmost column (Performance) and the bottom row (Steps)
    # of the 5-col × 3-row paper grid. Each label is set then hidden on the panels that
    # don't need it — keeps the saved figure size consistent so panels align in the grid.
    #   Row 1: 1..5    Row 2: 6..10    Row 3: 11..15
    Y_LABEL_INDICES = {1, 6, 11}                    # leftmost column of 5-col grid
    X_LABEL_INDICES = {11, 12, 13, 14, 15}          # bottom row of 3-row grid

    ax.set_ylabel("Performance", labelpad=2, fontweight="bold")
    ax.set_xlabel("Steps",       labelpad=2, fontweight="bold")
    if index not in Y_LABEL_INDICES:
        ax.yaxis.label.set_visible(False)
    if index not in X_LABEL_INDICES:
        ax.xaxis.label.set_visible(False)

    ax.tick_params(axis="x", pad=1.5)
    ax.tick_params(axis="y", pad=1.5)

    # Center the title over the plot grid (axes). Font sized so the longest task name
    # ("humanoidmaze-medium{1}") fits within the axes width without clipping.
    title = TASK_DISPLAY.get(task_id, task_id)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

    # NB: no tight_layout / no bbox_inches="tight" — both would resize the canvas to fit
    # content, producing different image sizes per panel. We keep the fixed figsize.
    out_path = out_dir / f"{index}.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"  wrote {out_path}  ({task_id})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="task_id(s) to plot; default = all")
    ap.add_argument("--out", default="figs/o2o", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    api = wandb.Api()

    selected = TASKS if not args.tasks else [t for t in TASKS if t[0] in args.tasks]
    if not selected:
        print(f"No matching tasks. Known: {[t[0] for t in TASKS]}", file=sys.stderr)
        sys.exit(1)

    # Pre-fetch wandb run indices once per method (vs once per task × method).
    # Text-source methods (e.g. IFQL via ifql.txt) load their data once here too.
    fetched_indices = {}
    text_data = {}
    repo_root = Path(__file__).resolve().parents[1]
    for method_name, cfg in METHODS.items():
        if cfg.get("source") == "text":
            data_path = repo_root / cfg["data_file"]
            try:
                print(f"  loading {method_name}: {data_path}", file=sys.stderr, flush=True)
                text_data[method_name] = load_text_data(data_path)
            except Exception as e:
                print(f"  [{method_name}] text load failed: {e}", file=sys.stderr)
                text_data[method_name] = None
            continue
        try:
            print(f"  fetching {method_name}: {cfg['offline_project']} + {cfg['online_project']}",
                  file=sys.stderr, flush=True)
            off_runs = fetch_all_runs(api, ENTITY, cfg["offline_project"])
            on_runs  = fetch_all_runs(api, ENTITY, cfg["online_project"])
            fetched_indices[method_name] = (index_runs_by_key(off_runs), index_runs_by_key(on_runs))
        except Exception as e:
            print(f"  [{method_name}] project fetch failed: {e}", file=sys.stderr)
            fetched_indices[method_name] = None

    # Number plots by their canonical position in TASKS (so 1.png is always humanoidmaze, …),
    # regardless of which subset the CLI selected.
    canonical_index = {t[0]: i + 1 for i, t in enumerate(TASKS)}
    for task_entry in selected:
        idx = canonical_index[task_entry[0]]
        print(f"  -> {task_entry[0]} (#{idx})", file=sys.stderr, flush=True)
        plot_task(api, task_entry, out_dir, fetched_indices, text_data, idx)

    # Always emit the legend image (consumed in the paper layout).
    plot_legend(out_dir)


if __name__ == "__main__":
    main()
