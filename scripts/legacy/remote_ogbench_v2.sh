#!/bin/bash
#SBATCH --job-name=og-v2
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node18,node31

ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

# ---- Build the job list ----
# cube_single_play: LOCKED at (2.0, 0.1) × 5 tasks = 5 jobs
# cube_double_play: 6 configs × 5 tasks = 30 jobs
# scene_play: tasks 1,3 LOCKED at (2.0,0.1) = 2 jobs + tasks 2,4,5 × 6 configs = 18 = 20 jobs
# puzzle_3x3_play: 6 × 5 = 30 jobs
# puzzle_4x4_play: 6 × 5 = 30 jobs
# total = 5 + 30 + 20 + 30 + 30 = 115

SWEEP_CONFIGS=(
  "0.5 0.01"
  "0.5 0.0001"
  "2.0 0.1"
  "2.0 0.001"
  "10.0 10.0"
  "1.0 0.01"
)

JOB_LIST=()

# cube_single (locked)
for t in 1 2 3 4 5; do
  JOB_LIST+=("cube_single_play_task${t} 2.0 0.1")
done

# cube_double (sweep all tasks)
for t in 1 2 3 4 5; do
  for cfg in "${SWEEP_CONFIGS[@]}"; do
    JOB_LIST+=("cube_double_play_task${t} $cfg")
  done
done

# scene (locked tasks 1,3 + sweep 2,4,5)
JOB_LIST+=("scene_play_task1 2.0 0.1")
JOB_LIST+=("scene_play_task3 2.0 0.1")
for t in 2 4 5; do
  for cfg in "${SWEEP_CONFIGS[@]}"; do
    JOB_LIST+=("scene_play_task${t} $cfg")
  done
done

# puzzle_3x3 (sweep all)
for t in 1 2 3 4 5; do
  for cfg in "${SWEEP_CONFIGS[@]}"; do
    JOB_LIST+=("puzzle_3x3_play_task${t} $cfg")
  done
done

# puzzle_4x4 (sweep all)
for t in 1 2 3 4 5; do
  for cfg in "${SWEEP_CONFIGS[@]}"; do
    JOB_LIST+=("puzzle_4x4_play_task${t} $cfg")
  done
done

read ENV_FULL TEMP E <<< "${JOB_LIST[$SLURM_ARRAY_TASK_ID]}"
SEED=1

# extract env_base and task_num for wandb tag
ENV_BASE=$(echo $ENV_FULL | sed 's/_task[0-9]//')
TASK_NUM=$(echo $ENV_FULL | grep -o 'task[0-9]' | grep -o '[0-9]')

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="task${TASK_NUM}_t${TEMP_TAG}_eta${ETA_TAG}_qmean"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $ENV_FULL seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$ENV_FULL \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=mean \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_mean,algo_gfot,task${TASK_NUM},${ENV_BASE}]"

echo "[$(date)] Done: $EXP_NAME $ENV_FULL seed=$SEED"
