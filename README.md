# FPOT: Flow Policy via Optimal Transport

JAX/Flax implementation of **Flow Policy via Optimal Transport (FPOT)** — a
value-weighted optimal-transport method for learning efficient one-step flow
policies in offline reinforcement learning.

FPOT jointly learns:

- a **critic** `Q_φ(s, a)` from offline data,
- a **value-aware reference flow policy** `μ_ω(s, z)` (multi-step Euler integration), and
- a **one-step flow policy** `μ_θ(s, z)` deployed at inference.

For each state, FPOT samples actions from both policies, builds an entropic
optimal-transport coupling whose teacher-side marginal is critic-weighted, and
distills the one-step policy toward transport-selected reference actions. This
separates value guidance from direct critic maximization while preserving
multimodal action structure.

The algorithm is implemented in [`fpot/agent.py`](fpot/agent.py); training
details and per-task hyperparameters live in [`experiments/`](experiments/).

---

## Repository layout

```
fpot/                       # Algorithm package
  agent.py                  # FPOT agent (paper Sec. 4 + App. C.2 online)
  networks.py               # MLP, FlowPolicy, NNPolicy, Value (Q ensemble)
  encoders.py               # IMPALA visual encoders
  flax_utils.py             # TrainState, ModuleDict, save/restore helpers
  replay_buffer.py          # numpy circular replay buffer
  evaluation.py             # rollout-based eval loop
envs/                       # Env factories + offline-dataset wrappers
  env_utils.py              # dispatches OGBench vs D4RL by env_name
  vec_utils_gymnasium.py    # gymnasium-API single-env factories (online phase)
  d4rl_utils.py, d4rl_common.py, adroit_utils.py
  datasets.py               # FrozenDict-backed offline dataset wrapper
configs/                    # Hydra config groups
  config.yaml               # top-level defaults
  agent/fpot.yaml           # FPOT agent defaults (paper Tab. 2)
  env/                      # 5 base templates: ogbench_state, ogbench_visual,
                            # d4rl_locomotion, d4rl_antmaze, d4rl_adroit
  train/{offline,offline_to_online}.yaml
  eval/default.yaml
  logging/default.yaml
experiments/                # Sweep specifications (paper-aligned)
  benchmarks/
    ogbench/{navigation,manipulation,visual}/
    d4rl/{antmaze,adroit}/
  ablations/                # N×M, τ×η, λ_q sweeps (paper App.)
  online/                   # offline-to-online sweeps
scripts/
  train.py                  # Hydra entry point (one job)
  sweep_runner.py           # Resolve a sweep yaml + idx → train.py command
  run_sweep.sh              # Launch an entire sweep across local GPUs
requirements/               # Pinned dependency lists
```

---

## Installation

D4RL and OGBench require incompatible MuJoCo / Python combinations, so the
codebase uses two conda environments. The dispatch is automatic:
[`envs/env_utils.py`](envs/env_utils.py) routes any env name containing
`singletask` to OGBench and the rest to D4RL.

| Conda env | Python | Use for |
|---|---|---|
| `fpot-ogbench` | 3.12 | OGBench tasks (`*-singletask-*`) |
| `fpot-d4rl` | 3.10 | D4RL AntMaze / Adroit / locomotion |

### OGBench env

```bash
conda create -n fpot-ogbench python=3.12 -y
conda activate fpot-ogbench
pip install -U pip setuptools wheel
pip install -r requirements/requirements-ogbench.txt
```

### D4RL env

D4RL needs MuJoCo 2.1 on disk:

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz
```

Then create the env:

```bash
conda create -n fpot-d4rl python=3.10 -y
conda activate fpot-d4rl
pip install -U pip setuptools wheel
pip install -r requirements/requirements-d4rl.txt
```

D4RL also needs three runtime exports per shell session:

```bash
export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

Verify GPU detection:

```bash
python -c "import jax; print(jax.devices())"
```

### WandB (optional)

```bash
wandb login
```

If you do not want online logging, set `+logging.wandb_mode=disabled` on any
training command.

---

## Running experiments

Every experiment is a single yaml under [`experiments/`](experiments/). Each
yaml declares the Cartesian product of overrides to sweep (`sweep:`) and a set
of fixed Hydra overrides (`fixed:`). One yaml = one logical experiment in the
paper.

### Single foreground run

```bash
python scripts/train.py \
    env=ogbench_state \
    env.env_name=cube-single-play-singletask-task1-v0 \
    agent.w_temperature=2.0 \
    agent.eta_temperature=1e-1 \
    agent.use_vabc_td_target=true \
    seed=1
```

### Inspect a sweep before launching

```bash
# Number of jobs in the array
python scripts/sweep_runner.py count \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml

# Print the train.py command for one index without running it
python scripts/sweep_runner.py run \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml \
    --idx 0 --dry-run
```

### Launch an entire sweep across local GPUs

[`scripts/run_sweep.sh`](scripts/run_sweep.sh) takes a sweep yaml and runs every
job in it across the GPUs on the current host:

```bash
# 8 GPUs, 1 job per GPU (default). Per-job logs land in logs/local/<name>_<idx>.log.
bash scripts/run_sweep.sh experiments/benchmarks/ogbench/manipulation/cube_single.yaml

# Override GPU count and per-GPU concurrency
bash scripts/run_sweep.sh experiments/online/door_cloned.yaml 4 2
```

The launcher pins each job index to GPU `idx % num_gpus` deterministically; at
most `num_gpus * jobs_per_gpu` are in flight at any time. As each job
completes, the next pending index is launched on its slot.

### Reproduce the full paper, one command per group

```bash
# OGBench (state-based)
for f in experiments/benchmarks/ogbench/{navigation,manipulation}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# D4RL
for f in experiments/benchmarks/d4rl/{antmaze,adroit}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# Ablations (paper App.)
for f in experiments/ablations/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# Offline-to-online (paper App. C.2)
for f in experiments/online/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done
```

Each yaml sweeps all seeds and ablation cells reported in the paper. Activate
the matching conda env first (`fpot-ogbench` for OGBench yamls, `fpot-d4rl`
for D4RL yamls).

---

## Configuration reference

Hydra resolves configs in this order, lowest to highest priority:

1. Defaults from `configs/config.yaml` (`agent: fpot`, `env: ogbench_state`,
   `train: offline`, `eval: default`, `logging: default`).
2. Config-group overrides from the sweep yaml's `fixed:` block (e.g.
   `env: d4rl_adroit`, `train: offline_to_online`).
3. Field overrides from the same `fixed:` block (e.g. `agent.w_temperature: 2.0`).
4. Sweep-axis overrides from `sweep:` (one combination per array index).
5. Anything passed on the command line.

The agent's hyperparameters live in
[`configs/agent/fpot.yaml`](configs/agent/fpot.yaml). Mapping to paper notation:

| Config key | Paper symbol | Meaning |
|---|---|---|
| `agent.w_temperature` | τ | teacher-marginal softmax temperature |
| `agent.eta_temperature` | η | value-aware BC temperature |
| `agent.n_student` / `agent.n_teacher` | N / M | one-step / reference samples per state |
| `agent.sinkhorn_eps` / `agent.sinkhorn_iters` | ε / T | Sinkhorn regularization & iterations |
| `agent.lambda_vbc` / `agent.lambda_distill` | λ_VBC / λ_distill | loss weights |
| `agent.lambda_q` | λ_q | (ablation only; default 0) direct critic-max term |
| `agent.use_vabc_td_target` | y^VaBC | VaBC TD target variant |
| `agent.q_agg` | — | `mean` (default) or `min` (CDQL, used for adroit + antmaze-{large,giant}) |
| `agent.discount` | γ | per-task override |

Env templates ([`configs/env/`](configs/env/)) take an `env.env_name` override
to specialize a base template (e.g. `env=ogbench_state` +
`env.env_name=cube-double-play-singletask-task2-v0`). The five templates cover
all task families in the paper.

---

## Logging

WandB projects collect runs by experiment family:

| Project | Source |
|---|---|
| `fpot_ogbench_benchmark` | `experiments/benchmarks/ogbench/**/*.yaml` |
| `fpot_d4rl_benchmark`    | `experiments/benchmarks/d4rl/**/*.yaml` |
| `fpot_ablation_nm`       | `experiments/ablations/nm_*.yaml` |
| `fpot_ablation_sensitivity` | `experiments/ablations/sensitivity_*.yaml` |
| `fpot_ablation_qmax`     | `experiments/ablations/qmax_*.yaml` |
| `fpot_online`            | `experiments/online/*.yaml` |

Each run is named `<task> | <swept-axes> | s<seed>` (offline-to-online runs
prefix `O2O |`). Tags: `task:<name>`, `seed:<n>`, `tau:<v>`, `eta:<v>`, plus
any swept axes (e.g. `lq:0.1`, `N:16 M:64`, `off2on`).

Disable on-disk artifacts (csv / config / checkpoints) per-run by adding to
the sweep yaml's `fixed:` block:

```yaml
fixed:
  logging.save_csv: false
  logging.save_config: false
  logging.save_checkpoint: false
```

---

## Outputs

By default each run writes to:

```
outputs/<env_name>/<sweep_name>/<hp_subdir>/seed_<N>/run_NN/
```

containing `metrics.csv`, `config.yaml`, `config.json`, and (if
`logging.save_checkpoint=true`) `final_agent/params_0.pkl`. The output root is
`outputs/` relative to the working directory; change it via
`logging.root_dir=/path/to/outputs`.

---

## Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'jax'` | Wrong conda env. Activate `fpot-ogbench` or `fpot-d4rl`. |
| D4RL: `Missing path to your environment variable .../usr/lib/nvidia` | `mujoco_py` cannot find NVIDIA driver libs; add `/usr/lib/nvidia` to `LD_LIBRARY_PATH`. |
| `Unable to load cuSPARSE` | JAX cannot find its CUDA wheels. Verify `JAX_NVIDIA_DIR` is set as in [Installation](#installation). |
| D4RL: `OSError: Unable to synchronously open file ...` | Corrupted dataset under `~/.d4rl/datasets/`. Delete and re-download. (Race: when launching many seeds at once on a fresh machine, run one job to download first, then the rest.) |
| Hydra: `Could not override 'env.env_name'` | Ensure the yaml has a base template (`env: ogbench_state` or similar) in its `fixed:` block before any `env.env_name:` override. |

---

## Citation

```bibtex
@inproceedings{fpot2026,
  title  = {Learning Multimodal One-step Flow Policy via Value-weighted Optimal Transport},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026.}
}
```
