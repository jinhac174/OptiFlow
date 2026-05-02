# FlowRL-with-sUOT

JAX/Flax offline RL codebase for `sUOT`, `FQL`, and related agents.

## Running experiments

All sweeps live under `experiments/eNNN_<name>.yaml`.  Use `scripts/submit.sh`
to launch a Slurm array job, or `scripts/sweep_runner.py` to inspect individual
jobs locally.

```bash
# Submit a full sweep to Slurm (do not run on the login node)
./scripts/submit.sh experiments/e001_suot_temp_tau_d4rl.yaml

# Inspect what a specific job would run (no execution)
python scripts/sweep_runner.py run --sweep experiments/e001_suot_temp_tau_d4rl.yaml \
    --idx 0 --dry-run

# Count total jobs in a sweep
python scripts/sweep_runner.py count --sweep experiments/e001_suot_temp_tau_d4rl.yaml
```

Legacy per-agent sbatch scripts are preserved in `scripts/legacy/` for
reference but are no longer the primary way to launch experiments.

### Adding a new agent

Three steps are all it takes — no changes to the training loop or sweep
infrastructure needed:

1. **Agent class** — create `agents/baselines/<name>/agent.py` (and
   `__init__.py`) implementing the `.create()` / `.update()` /
   `.sample_actions()` interface.  See `agents/baselines/iql/agent.py` as the
   canonical example.

2. **Config** — create `configs/agent/<name>.yaml` with the agent's
   hyperparameters.  Set `name: <name>` and `agent_file: <name>`.

3. **Dispatch** — add one line to `load_agent_class()` in
   `scripts/train.py`:
   ```python
   if agent_file == "<name>":
       return MyNewAgent
   ```

After that, write a sweep yaml under `experiments/` and launch with
`./scripts/submit.sh`.

**Currently available agents** (`agent_file` value → class):

| `agent_file` | Class | Notes |
|---|---|---|
| `agent_offline` | `OfflineSUOTAgent` | sUOT main agent |
| `agent_gfot` | `GFOTAgent` | Value-aware BC + balanced OT |
| `agent_fqot` | `FQOTAgent` | Flow-matching Q-filter |
| `agent_online` | `OnlineSUOTAgent` | Online sUOT |
| `iql` | `IQLAgent` | IQL baseline (Kostrikov et al., 2021) |

이 저장소는 환경을 두 갈래로 나눠서 쓰는 걸 권장합니다.

- `D4RL`용: `antmaze`, `maze2d`, `Adroit(pen/door/hammer/relocate)` 등
- `OGBench`용: `singletask-*` 등 최신 스택

현재 파일 구조도 그 기준으로 정리돼 있습니다.

- D4RL requirements: [`requirements-d4rl.txt`](requirements-d4rl.txt)
- OGBench requirements: [`requirements-ogbench.txt`](requirements-ogbench.txt)

이 README는 우선 `conda` 기준으로 정리합니다.

## 0. 지금 이 리눅스 환경에서 먼저 알아둘 점

이 문서는 현재 이 워크스페이스가 올라가 있는 리눅스 셸 기준으로 썼습니다.

- 작업 디렉터리: `/home/manfromearth_11/FlowRL-with-sUOT`
- conda base: `/home/manfromearth_11/miniconda3`
- 현재 `base` 셸의 기본 Python: `3.13.12`

중요:

- `base`의 `python 3.13.12`에 바로 `pip install`하지 마세요.
- 이 저장소는 task에 따라 별도 conda env를 써야 합니다.
- 로그인 직후 `conda activate`가 안 먹으면 아래를 먼저 실행하면 됩니다.

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
cd /home/manfromearth_11/FlowRL-with-sUOT
```

## 1. 권장 환경 구조와 Python 버전

권장 conda env:

- `flowrl-d4rl`
- `flowrl-ogbench`

이유:

- D4RL은 `python 3.10 + mujoco-py + mujoco210 + old gym` 축이 필요합니다.
- OGBench는 최신 JAX 스택이라 D4RL과 한 env에 넣으면 충돌 가능성이 큽니다.

정리하면:

- `antmaze`, `maze2d`, `adroit` 계열 -> `flowrl-d4rl` / `python=3.10`
- `singletask-*` 같은 OGBench 계열 -> `flowrl-ogbench` / `python=3.12`

## 2. D4RL용 conda env

생성:

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
cd /home/manfromearth_11/FlowRL-with-sUOT

conda create -n flowrl-d4rl python=3.10 -y
conda activate flowrl-d4rl
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-d4rl.txt
```

`requirements-d4rl.txt`는 다음 계열을 위한 환경입니다.

- `antmaze-*`
- `maze2d-*`
- `pen-*`
- `door-*`
- `hammer-*`
- `relocate-*`
- 기타 D4RL prefix

## 3. OGBench용 conda env

생성:

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
cd /home/manfromearth_11/FlowRL-with-sUOT

conda create -n flowrl-ogbench python=3.12 -y
conda activate flowrl-ogbench
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-ogbench.txt
```

이 env는 `singletask-*` 같은 OGBench용입니다.

## 4. MuJoCo 2.1 설치

D4RL 쪽은 `mujoco-py`가 `~/.mujoco/mujoco210`를 기대합니다.

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -o mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
```

## 5. D4RL 실행 전 필수 env vars

매 세션마다 최소한 아래를 잡고 실행하는 걸 권장합니다.

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
conda activate flowrl-d4rl

export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

설명:

- `JAX_NVIDIA_DIR`: conda env 내부 CUDA wheel 라이브러리 경로
- `~/.mujoco/mujoco210/bin`: MuJoCo 2.1
- `/usr/lib/nvidia`: `mujoco_py`가 요구하는 NVIDIA driver libs
- `/opt/ohpc/pub/apps/cuda/12.8/lib64`: 클러스터 CUDA libs 예시

클러스터마다 CUDA 경로는 다를 수 있습니다. 필요하면 경로만 바꿔서 쓰면 됩니다.

## 6. JAX 확인

### D4RL env

```bash
conda activate flowrl-d4rl
python - <<'PY'
import jax
print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

### OGBench env

```bash
conda activate flowrl-ogbench
python - <<'PY'
import jax
print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

## 7. D4RL 데이터셋 이슈

처음 `antmaze-*`를 실행하면 dataset을 다운로드합니다.

가끔 다운로드가 중간에 끊겨서 아래 같은 에러가 납니다:

```text
OSError: Unable to synchronously open file (truncated file ...)
```

이 경우는 환경 문제가 아니라 `~/.d4rl/datasets/*.hdf5` 파일이 깨진 겁니다.

### AntMaze medium play/diverse 직접 복구

깨진 파일 삭제:

```bash
rm -f ~/.d4rl/datasets/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5
rm -f ~/.d4rl/datasets/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5
```

직접 다운로드:

```bash
mkdir -p ~/.d4rl/datasets

curl -L https://huggingface.co/datasets/imone/D4RL/resolve/main/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5 \
  -o ~/.d4rl/datasets/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5

curl -L https://huggingface.co/datasets/imone/D4RL/resolve/main/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5 \
  -o ~/.d4rl/datasets/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5
```

크기 확인:

```bash
ls -lh ~/.d4rl/datasets/Ant_maze_big-maze_noisy_multistart_True_multigoal_*_sparse_fixed.hdf5
```

정상이면 대략 `221M` 정도로 보여야 합니다.

## 8. D4RL smoke test

```bash
conda activate flowrl-d4rl
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

python scripts/train_pointmaze.py \
  --env-name antmaze-medium-play-v2 \
  --agent suot \
  --seed 0 \
  --w-temperature 0.3 \
  --smoke
```

## 9. AntMaze 실제 실행

예시:

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
cd /home/manfromearth_11/FlowRL-with-sUOT

conda activate flowrl-d4rl
export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

python scripts/train_pointmaze.py \
  --env-name antmaze-medium-play-v2 \
  --agent suot \
  --seed 0 \
  --w-temperature 0.3 \
  --wandb-project antmaze-medium-0316
```

8개 배치 실행 스크립트:

```bash
bash scripts/run_antmaze_suot_0316.sh
```

주의:

- 현재 [`scripts/run_antmaze_suot_0316.sh`](scripts/run_antmaze_suot_0316.sh)는 `.venv-d4rl/bin/python`을 가리키고 있습니다.
- conda만 쓸 거면 이 스크립트를 conda env 기준으로 바꾸거나, 직접 명령으로 실행하는 편이 더 안전합니다.

## 10. OGBench 실행

예시:

```bash
conda activate flowrl-ogbench

python scripts/train_pointmaze.py \
  --env-name singletask-xyz \
  --agent suot \
  --seed 0
```

실제 OGBench env name은 설치된 benchmark 이름에 맞게 넣으면 됩니다.

## 11. 현재 env 분기

현재 [`envs/env_utils.py`](envs/env_utils.py) 기준 분기:

- `singletask-*` -> OGBench
- 나머지 D4RL prefix (`antmaze-*`, `maze2d-*`, `pointmaze-*`, `pen-*`, `door-*`, `hammer-*`, `relocate-*`, `halfcheetah-*`, `hopper-*`, `walker2d-*`, `D4RL/*`, `mujoco/*`) -> D4RL

즉 지금 구조상 `minari_utils` 없이 D4RL 경로로 가도록 정리돼 있습니다.

## 12. 자주 막히는 문제

### `ModuleNotFoundError: No module named 'jax'`

대부분 `base` env에서 돌린 경우입니다.

```bash
conda activate flowrl-d4rl
```

또는

```bash
conda activate flowrl-ogbench
```

### `Missing path to your environment variable ... /usr/lib/nvidia`

`mujoco_py`가 NVIDIA driver lib 경로를 못 찾는 경우입니다.

아래를 `LD_LIBRARY_PATH`에 포함해야 합니다:

```bash
/usr/lib/nvidia
```

### `Unable to load cuSPARSE`

JAX CUDA runtime을 못 찾는 경우입니다.

확인:

```bash
python -m pip show jax jaxlib jax-cuda12-plugin nvidia-cusparse-cu12
```

필요하면 재설치:

```bash
pip install --upgrade "jax[cuda12]==0.6.2"
```

### `truncated file`

깨진 dataset 파일입니다. 위 7번 절차대로 다시 받으면 됩니다.

## 13. W&B

로그인이 필요하면 각 env에서:

```bash
wandb login
```

프로젝트 이름은 실행 시 `--wandb-project ...`로 지정합니다.

## 14. 이 리눅스 셸에서 바로 실행하는 예시

### D4RL / AntMaze 단일 실행

```bash
source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh
cd /home/manfromearth_11/FlowRL-with-sUOT
conda activate flowrl-d4rl

export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

CUDA_VISIBLE_DEVICES=0 python scripts/train_pointmaze.py \
  --env-name antmaze-medium-diverse-v2 \
  --agent suot \
  --seed 1 \
  --w-temperature 0.20 \
  --wandb-project antmaze-diverse
```

### tmux로 2개 띄우기 예시

```bash
tmux new-session -d -s flowrl
tmux send-keys -t flowrl:0.0 'source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh && cd /home/manfromearth_11/FlowRL-with-sUOT && conda activate flowrl-d4rl && export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" && export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}" && export MUJOCO_GL=osmesa && export D4RL_SUPPRESS_IMPORT_ERROR=1 && CUDA_VISIBLE_DEVICES=0 python scripts/train_pointmaze.py --env-name antmaze-medium-diverse-v2 --agent suot --seed 1 --w-temperature 0.20 --wandb-project antmaze-diverse' C-m
tmux split-window -h -t flowrl:0
tmux send-keys -t flowrl:0.1 'source /home/manfromearth_11/miniconda3/etc/profile.d/conda.sh && cd /home/manfromearth_11/FlowRL-with-sUOT && conda activate flowrl-d4rl && export JAX_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" && export LD_LIBRARY_PATH="$JAX_NVIDIA_DIR/cusparse/lib:$JAX_NVIDIA_DIR/cublas/lib:$JAX_NVIDIA_DIR/cuda_runtime/lib:$JAX_NVIDIA_DIR/cudnn/lib:$JAX_NVIDIA_DIR/cufft/lib:$JAX_NVIDIA_DIR/cusolver/lib:$JAX_NVIDIA_DIR/nccl/lib:$JAX_NVIDIA_DIR/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}" && export MUJOCO_GL=osmesa && export D4RL_SUPPRESS_IMPORT_ERROR=1 && CUDA_VISIBLE_DEVICES=1 python scripts/train_pointmaze.py --env-name antmaze-medium-play-v2 --agent suot --seed 1 --w-temperature 0.10 --wandb-project antmaze-play' C-m
tmux attach -t flowrl
```
