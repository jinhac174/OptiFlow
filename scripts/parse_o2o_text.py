#!/usr/bin/env python3
"""Parse a text-format O2O data file (O2O.txt / ifql.txt / fql.txt).

Format (one block per task):

    task_key:
        offline:
            steps: {     0,   200k,   400k,   600k,   800k,     1M }
            seed1: {   0.0,   12.0,   40.0,   66.0,   70.0,   58.0 }
            ...
            seed8: { ... }
            mean : { ... }
            s.e  : { ... }
        online:
            steps: {  1.2M,   1.4M,   1.6M,   1.8M,     2M }
            seed1: { ... }
            ...

Returns: dict[task_key] -> {"offline": [[s1_vals], ..., [s8_vals]],
                            "online":  [[s1_vals], ..., [s8_vals]]}
where each s_vals list aligns with the 'steps' columns. Note: offline includes
a leading step-0 column (always 0.0 in the user's file), since eval doesn't
start until step 50k+.
"""
import re
from pathlib import Path


_ROW_RE = re.compile(r"^\s*(seed\d+|mean|s\.e)\s*:\s*\{\s*(.*?)\s*\}\s*$")
_TASK_RE = re.compile(r"^([A-Za-z_0-9]+):\s*$")
_PHASE_RE = re.compile(r"^\s*(offline|online):\s*$")


def _parse_row(values_str):
    out = []
    for tok in values_str.split(","):
        tok = tok.strip()
        if not tok:
            out.append(None)
            continue
        try:
            out.append(float(tok))
        except ValueError:
            out.append(None)
    return out


def load_text_data(path):
    text = Path(path).read_text()
    data = {}
    cur_task = None
    cur_phase = None
    cur_seeds = {}  # seed_idx (1..8) -> list[float]
    def _flush():
        nonlocal cur_seeds
        if cur_task and cur_phase and cur_seeds:
            per_seed = [cur_seeds.get(i, []) for i in range(1, 9)]
            data.setdefault(cur_task, {})[cur_phase] = per_seed
        cur_seeds = {}

    for line in text.splitlines():
        m = _TASK_RE.match(line)
        if m:
            _flush()
            cur_task = m.group(1)
            cur_phase = None
            continue
        m = _PHASE_RE.match(line)
        if m:
            _flush()
            cur_phase = m.group(1)
            continue
        m = _ROW_RE.match(line)
        if m and cur_task and cur_phase:
            tag = m.group(1)
            if tag.startswith("seed"):
                idx = int(tag[4:])
                cur_seeds[idx] = _parse_row(m.group(2))
    _flush()
    return data


if __name__ == "__main__":
    import sys, json
    d = load_text_data(sys.argv[1])
    print(f"tasks: {len(d)}")
    for k, v in d.items():
        off = v.get("offline", [])
        on = v.get("online", [])
        print(f"  {k}: off seeds={len(off)} cols={len(off[0]) if off else 0}; "
              f"on seeds={len(on)} cols={len(on[0]) if on else 0}")
