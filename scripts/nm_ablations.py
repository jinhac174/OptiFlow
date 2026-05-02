"""
Pull N×M ablation results from WandB and emit a LaTeX table.

Usage:
    python nm_ablation.py \
        --projects entity/cube_double entity/scene_play \
        --task-labels cube-double-play-task2 scene-play-task2 \
        --metric eval/success \
        --out nm_ablation.tex

Assumes each run has:
  - tags containing "uot_n:<val>" and "uot_m:<val>"
  - seed identifiable from config or tags
  - the metric key logged in the run's history

Output: a single LaTeX table block with N×M sub-tables side-by-side per task.
Each cell shows mean ± s.e. over 8 seeds; seed count shown only if < 8.
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import wandb


N_VALUES = [4, 16, 32]
M_VALUES = [4, 16, 64, 256]
EXPECTED_SEEDS = 8


def parse_tag(tags, key):
    """Return float value for 'key:<val>' tag, else None."""
    pattern = re.compile(rf"^{re.escape(key)}:(.+)$")
    for t in tags:
        m = pattern.match(t)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def get_final_metric(run, metric_key):
    """Return the last logged value of metric_key, or None."""
    try:
        history = run.history(keys=[metric_key], pandas=True)
    except Exception:
        return None
    if history is None or metric_key not in history.columns:
        return None
    vals = history[metric_key].dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[-1])


def collect_runs(project, metric_key):
    """Return dict (N, M) -> list of final metric values across seeds."""
    api = wandb.Api()
    runs = api.runs(project)
    cells = defaultdict(list)
    for run in runs:
        if run.state != "finished":
            continue
        n = parse_tag(run.tags, "uot_n")
        m = parse_tag(run.tags, "uot_m")
        if n is None or m is None:
            continue
        n, m = int(n), int(m)
        if n not in N_VALUES or m not in M_VALUES:
            continue
        val = get_final_metric(run, metric_key)
        if val is None:
            continue
        cells[(n, m)].append(val)
    return cells


def format_cell(values):
    """Return 'mean ± s.e.' string, or '--' if empty."""
    if len(values) == 0:
        return "--"
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    # Scale to percent if values look like rates in [0, 1]
    if max(values) <= 1.0:
        mean *= 100
        se *= 100
    cell = rf"\val{{{mean:.1f}}}{{{se:.1f}}}"
    if len(values) < EXPECTED_SEEDS:
        cell += rf"$^{{({len(values)})}}$"
    return cell


def build_subtable(task_label, cells):
    """Build one 3x4 subtable for a single task."""
    lines = []
    lines.append(r"\begin{subtable}[t]{0.48\textwidth}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{task_label}}}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{@{}c|cccc@{}}")
    lines.append(r"\toprule")
    header = r"$N \backslash M$ & " + " & ".join(str(m) for m in M_VALUES) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for n in N_VALUES:
        row = [str(n)] + [format_cell(cells.get((n, m), [])) for m in M_VALUES]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{subtable}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", required=True,
                    help="WandB project paths (entity/project), one per task")
    ap.add_argument("--task-labels", nargs="+", required=True,
                    help="Display labels for each project, same order")
    ap.add_argument("--metric", default="eval/success")
    ap.add_argument("--out", default="nm_ablation.tex")
    args = ap.parse_args()

    assert len(args.projects) == len(args.task_labels), \
        "projects and task-labels must match in length"

    subtables = []
    for project, label in zip(args.projects, args.task_labels):
        print(f"Pulling runs from {project}...")
        cells = collect_runs(project, args.metric)
        n_runs = sum(len(v) for v in cells.values())
        print(f"  {n_runs} runs across {len(cells)} (N, M) combos")
        for (n, m), vals in sorted(cells.items()):
            if len(vals) < EXPECTED_SEEDS:
                print(f"  WARN: (N={n}, M={m}) has only {len(vals)} seeds")
        subtables.append(build_subtable(label, cells))

    preamble = [
        r"% ===== N x M ablation =====",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ablation on the number of student ($N$) and teacher ($M$) "
        r"samples used for transport. Each cell reports success rate "
        r"(mean $\pm$ s.e., \%) over 8 seeds unless noted; "
        r"superscript $(k)$ indicates $k<8$ seeds available.}",
        r"\label{tab:nm_ablation}",
    ]
    postamble = [r"\end{table}"]
    body = "\n\\hfill\n".join(subtables)
    out = "\n".join(preamble + [body] + postamble)

    with open(args.out, "w") as f:
        f.write(out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()