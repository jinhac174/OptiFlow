# OptiFlow

JAX/Flax implementation of **OptiFlow** — value-weighted optimal-transport
distillation for one-step flow policies in offline RL.

OptiFlow jointly trains a critic `Q_φ(s, a)`, a value-aware reference flow
policy `μ_ω(s, z)`, and a one-step flow policy `μ_θ(s, z)` deployed at
inference. For each state, an entropic optimal-transport coupling between
reference- and one-step-policy action samples uses a critic-weighted teacher
marginal; the one-step policy is then distilled toward transport-selected
reference actions. The algorithm lives in
[`optiflow/agent.py`](optiflow/agent.py); per-task hyperparameters (paper
Tables 4–5) are encoded in the `experiments/` yamls.

---

## Installation

A single Python 3.10 conda environment hosts both OGBench (gymnasium /
modern mujoco) and D4RL (legacy `gym==0.23` / `mujoco_py` 2.1).

### 1. MuJoCo 2.1 binary (required by D4RL via `mujoco_py`)

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz
```

### 2. Conda env + system libs

```bash
conda create -n optiflow python=3.10 -y
conda activate optiflow
# mujoco_py compiles a Cython extension on first import; it needs GLEW headers
# and patchelf. Install via conda-forge (no root required):
conda install -c conda-forge -y glew patchelf
```

### 3. Pre-pin Cython before pip install

`mujoco_py==2.1.2.14` only builds against `Cython<3`. Install it first so
pip's resolver can't upgrade it during the next step:

```bash
pip install "Cython<3" "numpy==1.26.4"
```

### 4. Install OptiFlow

```bash
pip install -e .
```

This pulls every JAX/Flax/OGBench/D4RL dependency from `pyproject.toml`.

### 5. Runtime exports (per shell)

JAX needs to find its bundled CUDA wheels, `mujoco_py` needs the MuJoCo 2.1
binary, and rendering uses OSMesa:

```bash
export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

### 6. Verify

```bash
python -c "import jax; print(jax.devices())"
python -c "from optiflow import OptiFlowAgent; print('OptiFlow OK')"
python -c "import ogbench; import d4rl; print('envs OK')"
```

---

## Quick start: train one job

A "job" is one `(task, seed)` configuration. Pick the matching benchmark
yaml, copy its `fixed:` overrides onto the command line, set `seed`, and
pass the explicit `env.env_name`. **Any task can be trained by adjusting
the overrides below** — set `env`, `env.env_name`, the four agent
hyperparameters, and `seed`.

Example — `cube-double-play-singletask-task2-v0`, seed 1, with the
hyperparameters from [`experiments/benchmarks/ogbench/manipulation/cube_double.yaml`](experiments/benchmarks/ogbench/manipulation/cube_double.yaml):

```bash
python scripts/train.py \
    env=ogbench_state \
    env.env_name=cube-double-play-singletask-task2-v0 \
    agent=optiflow \
    agent.w_temperature=2.0 \
    agent.eta_temperature=0.01 \
    agent.q_agg=mean \
    agent.use_vabc_td_target=true \
    agent.discount=0.99 \
    train.max_steps=1000000 \
    train.eval_interval=20000 \
    seed=1
```

Defaults from [`configs/agent/optiflow.yaml`](configs/agent/optiflow.yaml)
supply paper Tab. 2 (`[512]×4` MLP, lr 3e-4, batch 256, target-network
τ 0.005, N=16 student / M=64 teacher samples, 30 Sinkhorn iters, ε=0.05).
Per-task overrides set τ, η, `q_agg`, the VaBC TD-target flag, and γ.

Common command-line overrides:

| Override | Effect |
|---|---|
| `seed=<N>` | random seed |
| `train.max_steps=<N>` | gradient steps (1M state OGBench, 500K visual / D4RL) |
| `train.eval_interval=<N>` | eval cadence |
| `eval.num_eval_episodes=<N>` | episodes per eval (default 100) |
| `logging.save_checkpoint=true` | write `final_agent/params_0.pkl` at the end |
| `logging.wandb_mode=disabled` | turn off wandb for this run |
| `logging.root_dir=<path>` | redirect outputs (default `outputs/`) |

---

## Reproduce paper Table 1 (offline benchmark)

Each yaml under [`experiments/benchmarks/`](experiments/benchmarks/) holds
the per-task `fixed:` overrides (paper Tab. 4 / Tab. 5) plus the
`sweep:` block (env variants × seeds).

Run a whole yaml across local GPUs:

```bash
# 40 jobs: 5 task variants × 8 seeds
bash scripts/run_sweep.sh experiments/benchmarks/ogbench/manipulation/cube_double.yaml

# 8 jobs: 1 D4RL env × 8 seeds
bash scripts/run_sweep.sh experiments/benchmarks/d4rl/adroit/pen_cloned.yaml
```

Inspect before launching:

```bash
# Total job count
python scripts/sweep_runner.py count --sweep experiments/benchmarks/ogbench/manipulation/cube_double.yaml

# Print one job's resolved command
python scripts/sweep_runner.py run \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_double.yaml \
    --idx 0 --dry-run
```

Reproduce the full table:

```bash
# OGBench state (1M steps, 8 seeds, 5 task variants per family)
for f in experiments/benchmarks/ogbench/{navigation,manipulation}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# OGBench visual (500K steps, 4 seeds)
for f in experiments/benchmarks/ogbench/visual/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# D4RL AntMaze + Adroit (1M steps, 8 seeds)
for f in experiments/benchmarks/d4rl/{antmaze,adroit}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done
```

---

## Reproduce paper Appendix C.2 (offline-to-online fine-tuning)

Each yaml in [`experiments/online/`](experiments/online/) runs **1M offline
steps + 1M online steps in one process** (2M gradient updates per seed).
The replay buffer is pre-seeded with the offline dataset; online
transitions are appended uniformly. WandB step axis is continuous (offline
`1..1e6`, online `1e6+1..2e6`); runs are prefixed `O2O |` and tagged
`off2on`. Hyperparameters come from paper Tab. C.2 (App. C.2) — `q_agg`
and `(τ, η)` differ from the offline benchmark for adroit tasks.

Run all included tasks:

```bash
for f in experiments/online/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done
```

Run one:

```bash
bash scripts/run_sweep.sh experiments/online/cube_double.yaml
```

To O2O **any other task**, copy a yaml from `experiments/online/`, edit
`env.env_name` and the agent overrides to match the App. C.2 row for that
task, and run it. The structure is the same as the benchmark yamls plus:

```yaml
fixed:
  ...
  train: offline_to_online
  train.offline_max_steps: 1000000
  train.online_max_steps: 1000000
  train.eval_interval: 50000
```

Resume a crashed online phase from the saved offline checkpoint:

```bash
python scripts/train.py train=offline_to_online \
    env=ogbench_state env.env_name=cube-double-play-singletask-task2-v0 \
    agent=optiflow agent.w_temperature=2.0 agent.eta_temperature=0.01 \
    agent.use_vabc_td_target=true \
    train.resume_from=outputs/ogbench_state/<sweep>/<hp>/seed_1/run_NN/agent_step_1000000 \
    seed=1
```

---

## Repository layout

```
optiflow/              # Algorithm package
  agent.py             # OptiFlow agent (paper Sec. 4)
  networks.py          # MLP, FlowPolicy (teacher), NNPolicy (student), Value
  encoders.py          # IMPALA visual encoders
  flax_utils.py        # TrainState, ModuleDict, save/restore helpers
  replay_buffer.py     # numpy circular buffer (online phase)
  evaluation.py        # rollout-based eval loop
envs/                  # Env factories + offline-dataset wrappers
  env_utils.py         # dispatches OGBench vs D4RL by env_name
  vec_utils_gymnasium.py
  d4rl_utils.py, adroit_utils.py, datasets.py
configs/               # Hydra config groups
  agent/optiflow.yaml  # Tab. 2 shared defaults
  env/                 # 4 base templates: ogbench_{state,visual}, d4rl_{antmaze,adroit}
  train/               # offline.yaml, offline_to_online.yaml
  eval/, logging/
experiments/           # Sweep specifications (paper-aligned)
  benchmarks/          # paper Tab. 1
  online/              # paper App. C.2
scripts/
  train.py             # Hydra entry point (one job)
  sweep_runner.py      # yaml + idx → train.py command
  run_sweep.sh         # Launch a sweep across local GPUs
pyproject.toml         # Unified dependency spec
```

---

## Configuration reference

Hydra resolves configs in this order, lowest → highest priority:

1. Defaults from `configs/config.yaml` (`agent: optiflow`, `env: ogbench_state`,
   `train: offline`, `eval: default`, `logging: default`).
2. Config-group overrides from the sweep yaml's `fixed:` block (e.g.
   `env: d4rl_adroit`, `train: offline_to_online`).
3. Field overrides from the same `fixed:` block (e.g. `agent.w_temperature: 2.0`).
4. Sweep-axis overrides from `sweep:` (one combination per array index).
5. Anything passed on the command line.

Agent hyperparameters live in [`configs/agent/optiflow.yaml`](configs/agent/optiflow.yaml):

| Config key                        | Paper symbol | Meaning                                                          |
|-----------------------------------|--------------|------------------------------------------------------------------|
| `agent.w_temperature`             | τ            | teacher-marginal softmax temperature                             |
| `agent.eta_temperature`           | η            | value-aware BC temperature                                       |
| `agent.n_student` / `n_teacher`   | N / M        | one-step / reference samples per state                           |
| `agent.sinkhorn_eps` / `_iters`   | ε / T        | entropic regularization & Sinkhorn iterations                    |
| `agent.lambda_vbc` / `_distill`   | λ_VBC / λ_d  | loss weights                                                     |
| `agent.q_agg`                     | —            | `mean` (default) or `min` (CDQL — paper Tab. 2)                  |
| `agent.use_vabc_td_target`        | y^VaBC       | VaBC TD target variant                                           |
| `agent.discount`                  | γ            | per-task override                                                |
| `agent.critic_update_interval`    | —            | 1 (default); 5 for antsoccer-arena (paper Tab. 2)                |

---

## Logging

WandB projects collect runs by experiment family:

| Project                          | Source                                    |
|----------------------------------|-------------------------------------------|
| `optiflow_ogbench_benchmark`     | `experiments/benchmarks/ogbench/**`       |
| `optiflow_d4rl_benchmark`        | `experiments/benchmarks/d4rl/**`          |
| `optiflow_online`                | `experiments/online/*`                    |

Run name format: `<task> | <swept-axes> | s<seed>` (offline-to-online runs
prefix `O2O |`). Tags include `task:<name>`, `seed:<n>`, `tau:<v>`, `eta:<v>`.

By default each run writes to `outputs/<env_name>/<sweep>/<hp_subdir>/seed_<N>/run_NN/`
containing `metrics.csv`, `config.yaml`, `config.json`, and (if
`logging.save_checkpoint=true`) `final_agent/params_0.pkl`.

---

## Citation

```bibtex
@inproceedings{optiflow2026,
  title  = {Learning Multimodal One-step Flow Policy via Value-weighted Optimal Transport},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026.}
}
```
