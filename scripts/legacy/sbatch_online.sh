#!/bin/bash
#SBATCH --job-name=suot-online
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err

ENV=/home/danielc174/miniconda3/envs/flowrl-d4rl
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd /home/danielc174/projects/FPOT

SEEDS=(1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: online hopper seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  train=online \
  +experiment=online \
  env=hopper \
  seed=$SEED

echo "[$(date)] Done: online hopper seed=$SEED"
