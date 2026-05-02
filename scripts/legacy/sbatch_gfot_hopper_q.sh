#!/bin/bash
#SBATCH --job-name=gfot-hopq
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-11
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

# TEMP ETA LQ — best GFOT settings + q max sweep
CONFIGS=(
  "1.5 10.0 0.0001"
  "1.5 10.0 0.001"
  "1.5 10.0 0.01"
  "1.5 1.0  0.001"
)

ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert)
SEED=2

NUM_ENVS=${#ENVS[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ENVS))
ENV_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ENVS))

read TEMP ETA LQ <<< "${CONFIGS[$CFG_IDX]}"
TASK=${ENVS[$ENV_IDX]}

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $ETA | tr -d '.')
LQ_TAG=$(echo $LQ | tr -d '.')
EXP_NAME="gfot_t${TEMP_TAG}_eta${ETA_TAG}_qmin_q${LQ_TAG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$ETA \
  agent.q_agg=min \
  agent.lambda_q=$LQ

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
