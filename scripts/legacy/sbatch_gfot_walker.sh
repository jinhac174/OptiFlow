#!/bin/bash
#SBATCH --job-name=gfot-wk
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-47
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node18,node31

ENV=/home/danielc174/miniconda3/envs/flowrl-d4rl
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd /home/danielc174/projects/FPOT

# TEMP ETA
CONFIGS=(
  "1.0 0.02"
  "1.0 0.05"
  "1.0 0.1"
  "1.5 0.02"
  "1.5 0.05"
  "1.5 0.07"
  "1.5 0.1"
  "1.5 0.15"
  "1.5 0.3"
  "1.5 1.0"
  "2.0 0.02"
  "2.0 0.05"
  "2.0 0.1"
  "2.0 0.3"
  "3.0 0.05"
  "3.0 0.1"
)

ENVS=(walker2d_medium walker2d_medium_replay walker2d_medium_expert)
SEED=2

NUM_ENVS=${#ENVS[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ENVS))
ENV_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ENVS))

read TEMP ETA <<< "${CONFIGS[$CFG_IDX]}"
TASK=${ENVS[$ENV_IDX]}

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $ETA | tr -d '.')
EXP_NAME="gfot_t${TEMP_TAG}_eta${ETA_TAG}_qmin"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$ETA \
  agent.q_agg=min

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
