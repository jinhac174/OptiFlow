#!/bin/bash
#SBATCH --job-name=suot-offline
#SBATCH --partition=base_suma_rtx3090
#SBATCH --array=162-323
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

VARIANTS=(offline_t10_tau12 offline_t10_tau15 offline_t10_tau18 offline_t15_tau12 offline_t15_tau15 offline_t15_tau18 offline_t20_tau12 offline_t20_tau15 offline_t20_tau18)
ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert halfcheetah_medium halfcheetah_medium_replay halfcheetah_medium_expert walker2d_medium walker2d_medium_replay walker2d_medium_expert)
SEEDS=(2 4 6 8)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=${#SEEDS[@]}

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_ENVS * NUM_SEEDS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_ENVS * NUM_SEEDS)))
ENV_IDX=$((REMAINDER / NUM_SEEDS))
SEED_IDX=$((REMAINDER % NUM_SEEDS))

VARIANT=${VARIANTS[$VARIANT_IDX]}
TASK=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $VARIANT $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py +experiment=$VARIANT env=$TASK seed=$SEED

echo "[$(date)] Done: $VARIANT $TASK seed=$SEED"
