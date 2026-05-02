#!/bin/bash
#SBATCH --job-name=d4rl-rest
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

declare -A WTEMP=(
  [hopper_medium]=5.0
  [hopper_medium_replay]=1.5
  [hopper_medium_expert]=5.0
  [halfcheetah_medium]=0.3
  [halfcheetah_medium_replay]=0.3
  [halfcheetah_medium_expert]=0.5
)
declare -A ETA=(
  [hopper_medium]=1.0
  [hopper_medium_replay]=0.1
  [hopper_medium_expert]=1.0
  [halfcheetah_medium]=0.1
  [halfcheetah_medium_replay]=0.1
  [halfcheetah_medium_expert]=0.1
)

ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert halfcheetah_medium halfcheetah_medium_replay halfcheetah_medium_expert)
SEEDS=(1 2 3 4 5 6 7 8)

NUM_SEEDS=${#SEEDS[@]}
ENV_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

TASK=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
TEMP=${WTEMP[$TASK]}
E=${ETA[$TASK]}

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="t${TEMP_TAG}_eta${ETA_TAG}_qmin"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=min \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_min,algo_gfot,final]"

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
