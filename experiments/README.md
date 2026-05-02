# Experiments

Each `*.yaml` here is a single sweep specification: it lists the Cartesian product
of overrides to run, plus the Slurm submission settings. The submission script
`scripts/submit.sh` reads one such yaml and submits a Slurm array job whose size
equals the product of the `sweep:` axes.

## Layout

```
experiments/
├── benchmarks/          # paper main results (Sec. 5)
│   ├── ogbench/
│   │   ├── navigation/      # antmaze-large/giant, humanoidmaze-{med,large}, antsoccer
│   │   ├── manipulation/    # cube-{single,double,triple}, scene, puzzle-{3x3,4x4}
│   │   └── visual/          # pixel variants of cube-{single,double}, scene, puzzle-*
│   └── d4rl/
│       ├── antmaze/         # umaze, medium-{play,diverse}, large-{play,diverse}
│       └── adroit/          # pen / door / hammer / relocate × {human, cloned, expert}
├── ablations/           # paper appendices
│   ├── nm_*.yaml            # N (one-step samples) × M (reference samples) sweep
│   ├── sensitivity_*.yaml   # τ × η around the per-task optimum
│   └── qmax_*.yaml          # λ_q ablation (direct critic maximization)
└── online/              # offline-to-online fine-tuning (App. C.2)
```

## Hyperparameter source

Every sweep encodes the hyperparameters from the paper:

| Source                                            | Defaults                                                                                                                                                                       |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Table 2 (`tab:FPOT_shared_hparams`)               | MLP `[512]×4`, GeLU, LayerNorm critic only, Adam lr=3e-4, target τ=5e-3, batch=256, K=10 Euler steps, N=16 student / M=64 teacher, 30 Sinkhorn iters, ε=0.05, γ=0.99 default. |
| Table 3 (`tab:FPOT_hparams_override`)             | Per-task discount γ and TD target (`y^VaBC`) overrides — encoded in each yaml's `fixed:` block.                                                                                |
| Table 4 (`tab:hparams_ogbench`)                   | Per-task FPOT (τ, η) for OGBench — encoded as `agent.w_temperature` and `agent.eta_temperature`.                                                                              |
| Table 5 (`tab:hparams_d4rl`)                      | Per-task FPOT (τ, η) for D4RL — encoded per-yaml.                                                                                                                              |

The `agent` config group (`configs/agent/fpot.yaml`) provides the Table-2 shared
defaults; each sweep yaml overrides only the per-task fields from Tables 3–5.

## Running a sweep

```bash
# 1. Inspect what the array will run
python scripts/sweep_runner.py count --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml
python scripts/sweep_runner.py run   --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml --idx 0 --dry-run

# 2. Submit the full sweep to Slurm
bash scripts/submit.sh experiments/benchmarks/ogbench/manipulation/cube_single.yaml
```

`submit.sh` picks the right runner script (`slurm/ogbench_array.sh` or
`slurm/d4rl_array.sh`) based on the yaml's `env_set:` field.

## Sweep yaml schema

```yaml
name: <sweep id>                  # becomes the Slurm --job-name and +exp= tag
description: <one-line summary>
env_set: ogbench | d4rl           # selects the array runner / conda env
wandb_project: <project name>     # injected as +logging.wandb_project=...

sweep:                            # Cartesian product axes (declaration-order = loop order)
  env.env_name: [...]
  seed:         [1, 2, 3, 4, 5, 6, 7, 8]

fixed:                            # appended to every job (no product)
  env: ogbench_state              # config-group selection
  agent: fpot
  agent.w_temperature: 2.0
  agent.eta_temperature: 1e-1
  ...

slurm:                            # forwarded as sbatch CLI flags
  time: "72:00:00"
  mem: "64G"
  cpus_per_task: 20
  qos: big_qos
  partitions: [base_suma_rtx3090, big_suma_rtx3090, ...]
```

## Reproducing paper tables

| Paper table                | Yamls to run                                              |
|----------------------------|-----------------------------------------------------------|
| Table 1 OGBench Navigation | `benchmarks/ogbench/navigation/*.yaml`                    |
| Table 1 OGBench Manipulation | `benchmarks/ogbench/manipulation/*.yaml`                 |
| Table 1 OGBench Visual     | `benchmarks/ogbench/visual/*.yaml`                        |
| Table 1 D4RL AntMaze       | `benchmarks/d4rl/antmaze/*.yaml`                          |
| Table 1 D4RL Adroit        | `benchmarks/d4rl/adroit/*.yaml`                           |
| App. N×M ablation          | `ablations/nm_{cube_double,scene,antmaze}.yaml`           |
| App. τ × η sensitivity     | `ablations/sensitivity_{cube_double,scene,antmaze}.yaml`  |
| App. λ_q ablation          | `ablations/qmax_*.yaml`                                   |
| App. C.2 offline-to-online | `online/*.yaml`                                           |
