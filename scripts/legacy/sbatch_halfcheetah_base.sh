#!/bin/bash
#SBATCH --job-name=suot-hc
#SBATCH --partition=base_suma_rtx3090
#SBATCH --array=36-71
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err

ENV=/home/danielc174/miniconda3/envs/flowrl-d4rl
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd /home/danielc174/projects/FPOT

TEMPS=(0.5 1.0 1.5 2.0)
TAUS=(1.2 1.5 1.8)
ENVS=(halfcheetah_medium halfcheetah_medium_replay halfcheetah_medium_expert)
SEEDS=(2 4)

NUM_TAUS=${#TAUS[@]}
NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=${#SEEDS[@]}

TEMP_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_TAUS * NUM_ENVS * NUM_SEEDS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_TAUS * NUM_ENVS * NUM_SEEDS)))
TAU_IDX=$((REMAINDER / (NUM_ENVS * NUM_SEEDS)))
REMAINDER2=$((REMAINDER % (NUM_ENVS * NUM_SEEDS)))
ENV_IDX=$((REMAINDER2 / NUM_SEEDS))
SEED_IDX=$((REMAINDER2 % NUM_SEEDS))

TEMP=${TEMPS[$TEMP_IDX]}
TAU=${TAUS[$TAU_IDX]}
TASK=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

TEMP_TAG=$(echo $TEMP | tr -d '.')
TAU_TAG=$(echo $TAU | tr -d '.')
EXP_NAME="t${TEMP_TAG}_tau${TAU_TAG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.w_temperature=$TEMP \
  agent.uot_tau=$TAU

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
