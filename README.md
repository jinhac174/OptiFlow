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

The paper's algorithm and theoretical analysis are summarized in
[`FPOT/agent_fpot.py`](FPOT/agent_fpot.py); training details and per-task
hyperparameters live in [`experiments/`](experiments/).

---

## Repository layout

```
FPOT/                       # Algorithm code
  agent_fpot.py             # FPOT agent (paper Sec. 4 + App. C.2 online)
  common.py                 # shared critic helpers
configs/                    # Hydra config groups
  config.yaml               # top-level defaults
  agent/fpot.yaml           # FPOT agent defaults (paper Tab. 2)
  env/                      # 5 base templates: ogbench_state, ogbench_visual,
                            #   d4rl_locomotion, d4rl_antmaze, d4rl_adroit
  train/{offline,offline_to_online}.yaml
  eval/default.yaml
  logging/default.yaml
experiments/                # Sweep specifications (paper-aligned)
  benchmarks/
    ogbench/{navigation,manipulation,visual}/
    d4rl/{antmaze,adroit}/
  ablations/                # N×M, τ×η, λ_q sweeps (paper App.)
  online/                   # offline-to-online sweeps
networks/, envs/, utils/    # Networks, env wrappers, training utilities
scripts/
  train.py                  # Hydra entry point
  sweep_runner.py           # Resolve one sweep index → train.py command
  submit.sh                 # Submit a sweep as a Slurm array
  run_local_sweep.sh        # Round-robin a sweep across local GPUs (no Slurm)
  launch_smoke.sh           # Compact multi-yaml multi-GPU launcher
  check_sweep_status.sh     # Tally done / running / crashed / missing for a sweep
slurm/                      # Slurm runner shells (d4rl_array.sh, ogbench_array.sh)
requirements/               # Pinned dependency lists
```

---

## Installation

The codebase splits into two Python environments because D4RL and OGBench
require incompatible MuJoCo / Python combinations.

| Conda env | Python | Use for |
|---|---|---|
| `flowrl-ogbench` | 3.12 | OGBench tasks (`*-singletask-*`) |
| `flowrl-d4rl` | 3.10 | D4RL AntMaze / Adroit / locomotion |

The dispatch is automatic: [`envs/env_utils.py`](envs/env_utils.py) routes any
env name containing `singletask` to OGBench and the rest to D4RL.

### OGBench env

```bash
conda create -n flowrl-ogbench python=3.12 -y
conda activate flowrl-ogbench
pip install -U pip setuptools wheel
pip install -r requirements/requirements-ogbench.txt
```

### D4RL env

D4RL needs MuJoCo 2.1 on disk:

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz
```

Then create the env:

```bash
conda create -n flowrl-d4rl python=3.10 -y
conda activate flowrl-d4rl
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

### WandB

```bash
wandb login
```

Logs are organized into six projects (see [Logging](#logging)), so no per-task
project setup is needed.

---

## Running experiments

Every experiment is a single yaml under [`experiments/`](experiments/). Each
yaml specifies the Cartesian product of overrides to run, plus the Slurm
submission settings.

### Inspect what a sweep would launch

```bash
# Total number of jobs in the array
python scripts/sweep_runner.py count \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml

# Show the exact train.py command for one index, without running it
python scripts/sweep_runner.py run \
    --sweep experiments/benchmarks/ogbench/manipulation/cube_single.yaml \
    --idx 0 --dry-run
```

### Submit to Slurm

```bash
bash scripts/submit.sh experiments/benchmarks/ogbench/manipulation/cube_single.yaml
```

The submitter reads `env_set`, `slurm:`, `wandb_project`, and `name` from the
yaml, picks the right runner (`slurm/ogbench_array.sh` vs `slurm/d4rl_array.sh`,
which differ only in the conda env they activate), computes the array size,
builds the sbatch flags, and submits.

To run only a subset of indices (e.g. seeds 1 and 5 across 5 tasks
correspond to indices `0,4,8,12,16,20,24,28,32,36`):

```bash
bash scripts/submit.sh experiments/.../cube_single.yaml \
    --array=0,4,8,12,16,20,24,28,32,36
```

### Run on a single multi-GPU box (no Slurm)

```bash
# Round-robin one yaml across 8 GPUs, 1 job per GPU
bash scripts/run_local_sweep.sh \
    experiments/benchmarks/ogbench/manipulation/cube_single.yaml 8 1
```

### Single job, foreground

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    env=ogbench_state \
    env.env_name=cube-single-play-singletask-task1-v0 \
    agent.w_temperature=2.0 \
    agent.eta_temperature=1e-1 \
    agent.use_vabc_td_target=true \
    seed=1
```

---

## Configuration system

Hydra resolves configs in this order:
1. Defaults from `configs/config.yaml` (`agent: fpot`, `env: ogbench_state`,
   `train: offline`, `eval: default`, `logging: default`).
2. Config-group overrides from the sweep yaml's `fixed:` block (e.g.
   `env: ogbench_state`).
3. Field overrides from the same `fixed:` block (e.g. `agent.w_temperature: 2.0`).
4. Sweep-axis overrides from `sweep:` (one combination per array index).
5. Anything passed on the command line.

The full list of agent hyperparameters lives in
[`configs/agent/fpot.yaml`](configs/agent/fpot.yaml). Every field corresponds
to a paper notation, e.g.

| Config key | Paper symbol | Meaning |
|---|---|---|
| `agent.w_temperature` | τ | teacher-marginal softmax temperature |
| `agent.eta_temperature` | η | value-aware BC temperature |
| `agent.n_student` / `agent.n_teacher` | N / M | one-step / reference samples per state |
| `agent.sinkhorn_eps` / `agent.sinkhorn_iters` | ε / T | Sinkhorn regularization & iterations |
| `agent.lambda_vbc` / `agent.lambda_distill` | λ_VBC / λ_distill | loss weights |
| `agent.lambda_q` | λ_q | (ablation only; default 0) direct critic-max term |
| `agent.use_vabc_td_target` | y^VaBC | VaBC TD target variant |
| `agent.q_agg` | — | `mean` (default) or `min` (CDQL) |
| `agent.discount` | γ | per-task override |

The new env scheme uses **5 base templates** (`ogbench_state`, `ogbench_visual`,
`d4rl_locomotion`, `d4rl_antmaze`, `d4rl_adroit`) and a per-task `env.env_name`
override, instead of one yaml per task.

---

## Reproducing the paper

| Paper table / section | Yamls |
|---|---|
| Tab. 1, OGBench Navigation | `experiments/benchmarks/ogbench/navigation/*.yaml` |
| Tab. 1, OGBench Manipulation | `experiments/benchmarks/ogbench/manipulation/*.yaml` |
| Tab. 1, OGBench Visual | `experiments/benchmarks/ogbench/visual/*.yaml` |
| Tab. 1, D4RL AntMaze | `experiments/benchmarks/d4rl/antmaze/*.yaml` |
| Tab. 1, D4RL Adroit | `experiments/benchmarks/d4rl/adroit/*.yaml` |
| App., N × M ablation | `experiments/ablations/nm_*.yaml` |
| App., τ × η sensitivity | `experiments/ablations/sensitivity_*.yaml` |
| App., λ_q ablation | `experiments/ablations/qmax_*.yaml` |
| App. C.2, offline-to-online | `experiments/online/*.yaml` |

Hyperparameters in every yaml are taken directly from the paper's Tables 2–5;
each yaml lists the source line in its `description:` field. Each yaml sweeps
the full set of seeds reported in the paper (1–8 for state-based / D4RL,
1–4 for visual). See [`experiments/README.md`](experiments/README.md) for the
yaml schema.

---

## Logging

Six WandB projects collect everything; nothing fans out per-task:

| Project | Source |
|---|---|
| `fpot_ogbench_benchmark` | `experiments/benchmarks/ogbench/**/*.yaml` |
| `fpot_d4rl_benchmark` | `experiments/benchmarks/d4rl/**/*.yaml` |
| `fpot_ablation_nm` | `experiments/ablations/nm_*.yaml` |
| `fpot_ablation_sensitivity` | `experiments/ablations/sensitivity_*.yaml` |
| `fpot_ablation_qmax` | `experiments/ablations/qmax_*.yaml` |
| `fpot_online` | `experiments/online/*.yaml` |

Each run gets:

- **Name**: `<task> | <swept axes> | s<seed>` — e.g.
  `cube_single_play_task1 | s1`, or `cube_double_play_task2 | tau=2 eta=0.01 | s1`
  for a sensitivity run.
- **Group**: same as name without the seed (collects seeds for one cell).
- **Tags**: `task:<name>`, `seed:<n>`, `tau:<v>`, `eta:<v>`, plus the values
  of any swept axes (e.g. `lq:0.1` in λ_q ablations, `N:16 M:64` in N×M
  ablations). Offline-to-online runs also carry `off2on`.

Disable on-disk artifacts (e.g. on space-constrained machines) with:

```
--override logging.save_csv=false \
--override logging.save_config=false \
--override logging.save_checkpoint=false
```

passed to `sweep_runner.py run`.

---

## Outputs

By default each run writes to:

```
outputs/<env_name>/<sweep_name>/<hp_subdir>/seed_<N>/run_NN/
```

with `metrics.csv`, `config.yaml`, `config.json`, and (if `save_checkpoint=true`)
`final_agent/params_0.pkl`. The output root is `outputs/` in the project
directory; change it via `logging.root_dir`.

---

## Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'jax'` | Wrong conda env. `conda activate flowrl-{ogbench,d4rl}`. |
| D4RL: `Missing path to your environment variable .../usr/lib/nvidia` | `mujoco_py` cannot find NVIDIA driver libs; add `/usr/lib/nvidia` to `LD_LIBRARY_PATH`. |
| `Unable to load cuSPARSE` | JAX cannot find its CUDA wheels. Verify `JAX_NVIDIA_DIR` is set, or `pip install --upgrade "jax[cuda13]==0.9.1"`. |
| D4RL: `OSError: Unable to synchronously open file (truncated file)` | Corrupted dataset under `~/.d4rl/datasets/`. Delete and re-download (or fetch directly from `https://huggingface.co/datasets/imone/D4RL`). |
| Hydra: `Could not override 'env.env_name'` | Make sure the yaml has `env: ogbench_state` (or another template) in its `fixed:` block before the per-task `env.env_name:` overrides. |

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
