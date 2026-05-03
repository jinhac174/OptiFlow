# Experiments

Each `*.yaml` is one logical experiment. It declares the Cartesian product of
overrides to run (`sweep:`) and a set of overrides applied to every job
(`fixed:`).

## Layout

```
experiments/
├── benchmarks/              # paper Sec. 5 / Tab. 1 (final results)
│   ├── ogbench/
│   │   ├── navigation/      # antmaze-{large,giant}, humanoidmaze-{med,large}, antsoccer
│   │   ├── manipulation/    # cube-{single,double,triple}, scene, puzzle-{3x3,4x4}
│   │   └── visual/          # pixel variants (4 seeds, 500K steps)
│   └── d4rl/
│       ├── antmaze/         # umaze, medium-{play,diverse}, large-{play,diverse}
│       └── adroit/          # pen / door / hammer / relocate × {human, cloned, expert}
└── online/                  # paper App. C.2 (offline-to-online fine-tuning, 1M+1M)
```

One yaml per task family for OGBench (5 task variants × 8 seeds in one sweep);
one yaml per env for D4RL (the 12 Adroit tasks and 6 AntMaze variants each have
distinct `(τ, η)` per paper Tab. 5).

## Hyperparameter source

| Source                                | Where it lives                                                                               |
|---------------------------------------|----------------------------------------------------------------------------------------------|
| Paper Tab. 2 (shared defaults)        | [`configs/agent/fpot.yaml`](../configs/agent/fpot.yaml) — `[512]×4` GeLU, lr 3e-4, etc.     |
| Paper Tab. 3 (per-task γ, TD target)  | Each yaml's `fixed:` block (`agent.discount`, `agent.use_vabc_td_target`).                  |
| Paper Tab. 4 (per-task τ, η, OGBench) | Each yaml's `fixed:` block (`agent.w_temperature`, `agent.eta_temperature`).                |
| Paper Tab. 5 (per-task τ, η, D4RL)    | Same.                                                                                       |

CDQL (`agent.q_agg: min`) is set in the yaml for `antmaze-{large, giant}` and
all D4RL tasks; `mean` elsewhere (paper Tab. 2).

## Running

```bash
# Inspect total job count
python scripts/sweep_runner.py count --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml

# Print one job's command without running
python scripts/sweep_runner.py run \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml \
    --idx 0 --dry-run

# Launch the full sweep across local GPUs (8 GPUs, 1 job per GPU by default)
bash scripts/run_sweep.sh experiments/benchmarks/ogbench/manipulation/cube_single.yaml
```

`run_sweep.sh` pins each job to GPU `idx % num_gpus` and keeps at most
`num_gpus * per_gpu` jobs in flight. Per-job logs land in
`logs/local/<sweep_name>_<idx>.log`.

## Yaml schema

```yaml
name: <sweep id>                  # used as the wandb +exp tag
description: <one line>
wandb_project: <project>          # routed to wandb

sweep:                            # Cartesian product (declaration order = loop order)
  env.env_name: [...]
  seed: [1, 2, 3, 4, 5, 6, 7, 8]

fixed:                            # appended to every job
  env: ogbench_state              # config-group selection
  agent: fpot
  agent.w_temperature: 2.0
  ...
```

Sweep job indices iterate the Cartesian product in declaration order: the
first key is the outermost loop (changes slowest), the last is the innermost.

For online fine-tuning, set `train: offline_to_online` and `train.offline_max_steps`
+ `train.online_max_steps` (see `experiments/online/*.yaml`).

## Reproducing paper tables

| Paper table                  | Yamls to run                                              |
|------------------------------|-----------------------------------------------------------|
| Tab. 1 OGBench Navigation    | `benchmarks/ogbench/navigation/*.yaml`                    |
| Tab. 1 OGBench Manipulation  | `benchmarks/ogbench/manipulation/*.yaml`                  |
| Tab. 1 OGBench Visual        | `benchmarks/ogbench/visual/*.yaml`                        |
| Tab. 1 D4RL AntMaze          | `benchmarks/d4rl/antmaze/*.yaml`                          |
| Tab. 1 D4RL Adroit           | `benchmarks/d4rl/adroit/*.yaml`                           |
| App. C.2 Offline-to-Online   | `online/*.yaml`                                           |

Each yaml sweeps all seeds reported in the paper (8 seeds for state-based and
D4RL tasks, 4 seeds for visual OGBench).
