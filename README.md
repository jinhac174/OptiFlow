# FPOT: Flow Policy via Optimal Transport

JAX/Flax implementation of **Flow Policy via Optimal Transport (FPOT)** — a
value-weighted optimal-transport method for learning efficient one-step flow
policies in offline reinforcement learning.

FPOT jointly trains:

- a critic `Q_φ(s, a)`,
- a value-aware reference flow policy `μ_ω(s, z)` (multi-step Euler integration),
- a one-step flow policy `μ_θ(s, z)` deployed at inference.

Per state, it constructs an entropic optimal-transport coupling between
reference- and one-step-policy action samples whose teacher marginal is
critic-weighted, then distills the one-step policy toward transport-selected
reference actions. The algorithm lives in
[`fpot/agent.py`](fpot/agent.py); per-task hyperparameters (paper Tables 4–5)
are encoded in the `experiments/` yamls.

---

## Installation

A single Python 3.10 conda environment hosts both OGBench (gymnasium /
modern mujoco) and D4RL (legacy `gym==0.23` / `mujoco_py` 2.1).

### 1. MuJoCo 2.1 binary (required by D4RL via `mujoco_py`)

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz
```

### 2. Conda env + pip install

```bash
conda create -n fpot python=3.10 -y
conda activate fpot
# mujoco_py compiles a Cython extension on first import; it needs GLEW headers
# and patchelf. Install via conda-forge (no root required):
conda install -c conda-forge -y glew patchelf
pip install -e .
```

`pip install -e .` reads `pyproject.toml` and pulls every
JAX/Flax/OGBench/D4RL dependency.

### 3. Runtime exports (per shell)

JAX needs to find its bundled CUDA wheels, `mujoco_py` needs the MuJoCo 2.1
binary, and rendering uses OSMesa:

```bash
export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

### 4. Verify

```bash
python -c "import jax; print(jax.devices())"
python -c "from fpot import FPOTAgent; print('FPOT OK')"
python -c "import ogbench; import d4rl; print('envs OK')"
```

---

## Benchmark experiments

Every paper experiment is one yaml under [`experiments/`](experiments/) — see
[`experiments/README.md`](experiments/README.md) for the full layout. Each
yaml's `fixed:` block holds the per-task hyperparameters from paper Tabs. 3–5
(`agent.w_temperature` = τ, `agent.eta_temperature` = η, `agent.q_agg`,
`agent.use_vabc_td_target`, `agent.discount`); the `sweep:` block enumerates
the seeds (and, for state-based OGBench, the 5 task variants per family).

### Launch a single job

A single job is one `(task, seed)` configuration. Pick the matching benchmark
yaml, copy its `fixed:` overrides onto the command line, set `seed`, and pass
the explicit `env.env_name`. Example — `cube-double-play-singletask-task2-v0`,
seed 1, with the hyperparameters from
[`experiments/benchmarks/ogbench/manipulation/cube_double.yaml`](experiments/benchmarks/ogbench/manipulation/cube_double.yaml):

```bash
python scripts/train.py \
    env=ogbench_state \
    env.env_name=cube-double-play-singletask-task2-v0 \
    agent=fpot \
    agent.w_temperature=2.0 \
    agent.eta_temperature=0.01 \
    agent.q_agg=mean \
    agent.use_vabc_td_target=true \
    agent.discount=0.99 \
    train.max_steps=1000000 \
    train.eval_interval=20000 \
    seed=1
```

The defaults in [`configs/agent/fpot.yaml`](configs/agent/fpot.yaml) supply
all of paper Tab. 2 (`[512]×4` MLP, lr 3e-4, batch 256, target-network τ
0.005, N=16 student / M=64 teacher samples, 30 Sinkhorn iters, ε=0.05);
overrides above only set the per-task fields.

Any Hydra override is fair game on the command line — common ones:

| Override | Effect |
|---|---|
| `seed=<N>` | random seed (re-creates JAX PRNGKey + dataset RNG) |
| `train.max_steps=<N>` | number of gradient steps (1M for state OGBench, 500K for D4RL/visual) |
| `train.eval_interval=<N>` | eval cadence in steps |
| `eval.num_eval_episodes=<N>` | episodes per eval (default 100) |
| `logging.save_checkpoint=true` | write `final_agent/params_0.pkl` at the end |
| `logging.wandb_mode=disabled` | turn off wandb for this run |
| `logging.root_dir=<path>` | redirect outputs (default `outputs/`) |

### Sweep one task or one family

[`scripts/run_sweep.sh`](scripts/run_sweep.sh) launches every job in a yaml
across local GPUs (`<yaml> [num_gpus=8] [per_gpu=1]`, pins each job to
GPU `idx % num_gpus`, per-job logs go to `logs/local/<name>_<idx>.log`):

```bash
# 40 jobs: 5 task variants × 8 seeds
bash scripts/run_sweep.sh experiments/benchmarks/ogbench/manipulation/cube_double.yaml

# 8 jobs: 1 D4RL env × 8 seeds
bash scripts/run_sweep.sh experiments/benchmarks/d4rl/adroit/pen_cloned.yaml
```

Inspect before launching:

```bash
# Total job count for one yaml
python scripts/sweep_runner.py count --sweep experiments/benchmarks/ogbench/manipulation/cube_double.yaml

# Print one job's resolved train.py command without executing
python scripts/sweep_runner.py run \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_double.yaml \
    --idx 0 --dry-run
```

### Reproduce paper Table 1

```bash
# OGBench state-based (1M steps, 8 seeds, 5 tasks per family)
for f in experiments/benchmarks/ogbench/{navigation,manipulation}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# OGBench visual (500K steps, 4 seeds)
for f in experiments/benchmarks/ogbench/visual/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# D4RL AntMaze + Adroit (500K steps, 8 seeds)
for f in experiments/benchmarks/d4rl/{antmaze,adroit}/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done
```

### Reproduce paper Appendix C.2 (offline-to-online fine-tuning)

Each yaml in `experiments/online/` runs a single process that does **1M
offline steps followed by 1M online steps** on the same task — total 2M
gradient updates per seed, 8 seeds per task. The buffer is pre-seeded with
the offline dataset and online transitions are appended uniformly. WandB
step axis is continuous across phases (offline `1..1e6`, online
`1e6+1..2e6`); runs are prefixed `O2O |` and tagged `off2on`.

```bash
# All 7 tasks (3 OGBench + 4 D4RL Adroit), 8 seeds each
for f in experiments/online/*.yaml; do
    bash scripts/run_sweep.sh "$f"
done

# Or one task at a time
bash scripts/run_sweep.sh experiments/online/cube_double.yaml
```

The offline-phase endpoint is always saved to
`outputs/<env>/<sweep>/<hp>/seed_<N>/run_<NN>/agent_step_1000000/`
(regardless of `logging.save_checkpoint`) so a crashed online phase can be
resumed without redoing the offline phase:

```bash
python scripts/train.py train=offline_to_online \
    env=ogbench_state env.env_name=cube-double-play-singletask-task2-v0 \
    agent=fpot agent.w_temperature=2.0 agent.eta_temperature=0.01 \
    agent.use_vabc_td_target=true \
    train.resume_from=outputs/ogbench_state/fpot_online_cube_double/<hp>/seed_1/run_00/agent_step_1000000 \
    seed=1
```

---

## Repository layout

```
fpot/                  # Algorithm package
  agent.py             # FPOT agent (paper Sec. 4)
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
  agent/fpot.yaml      # Tab. 2 shared defaults
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

Hydra resolves configs in this order, lowest to highest priority:

1. Defaults from `configs/config.yaml` (`agent: fpot`, `env: ogbench_state`,
   `train: offline`, `eval: default`, `logging: default`).
2. Config-group overrides from the sweep yaml's `fixed:` block (e.g.
   `env: d4rl_adroit`, `train: offline_to_online`).
3. Field overrides from the same `fixed:` block (e.g. `agent.w_temperature: 2.0`).
4. Sweep-axis overrides from `sweep:` (one combination per array index).
5. Anything passed on the command line.

Agent hyperparameters live in [`configs/agent/fpot.yaml`](configs/agent/fpot.yaml):

| Config key                        | Paper symbol | Meaning                                                          |
|-----------------------------------|--------------|------------------------------------------------------------------|
| `agent.w_temperature`             | τ            | teacher-marginal softmax temperature                             |
| `agent.eta_temperature`           | η            | value-aware BC temperature                                       |
| `agent.n_student` / `n_teacher`   | N / M        | one-step / reference samples per state                           |
| `agent.sinkhorn_eps` / `_iters`   | ε / T        | entropic regularization & Sinkhorn iterations                    |
| `agent.lambda_vbc` / `_distill`   | λ_VBC / λ_d  | loss weights                                                     |
| `agent.q_agg`                     | —            | `mean` (default) or `min` (CDQL — antmaze-{large,giant}, adroit) |
| `agent.use_vabc_td_target`        | y^VaBC       | VaBC TD target variant (paper Tab. 3)                            |
| `agent.discount`                  | γ            | per-task override                                                |

---

## Logging

WandB projects collect runs by experiment family:

| Project                  | Source                                    |
|--------------------------|-------------------------------------------|
| `fpot_ogbench_benchmark` | `experiments/benchmarks/ogbench/**`       |
| `fpot_d4rl_benchmark`    | `experiments/benchmarks/d4rl/**`          |
| `fpot_online`            | `experiments/online/*`                    |

Run name format: `<task> | <swept-axes> | s<seed>` (offline-to-online runs
prefix `O2O |`). Tags include `task:<name>`, `seed:<n>`, `tau:<v>`, `eta:<v>`.

By default each run writes to `outputs/<env_name>/<sweep>/<hp_subdir>/seed_<N>/run_NN/`
containing `metrics.csv`, `config.yaml`, `config.json`, and (if
`logging.save_checkpoint=true`) `final_agent/params_0.pkl`.

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
