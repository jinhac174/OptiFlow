#!/bin/bash
#SBATCH --job-name=gfot
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --array=0-71
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

# Per-env best w_temperature from sUOT runs
declare -A WTEMP=(
  [hopper_medium]=1.5
  [hopper_medium_replay]=0.5
  [hopper_medium_expert]=1.5
  [halfcheetah_medium]=0.5
  [halfcheetah_medium_replay]=0.5
  [halfcheetah_medium_expert]=1.5
  [walker2d_medium]=1.5
  [walker2d_medium_replay]=1.5
  [walker2d_medium_expert]=1.5
)

ETAS=(0.05 0.1 0.3 1.0)
QAGGS=(mean min)
ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert halfcheetah_medium halfcheetah_medium_replay halfcheetah_medium_expert walker2d_medium walker2d_medium_replay walker2d_medium_expert)
SEEDS=(2)

NUM_ETAS=${#ETAS[@]}
NUM_QAGGS=${#QAGGS[@]}
NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=${#SEEDS[@]}

ETA_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_QAGGS * NUM_ENVS * NUM_SEEDS)))
REM=$((SLURM_ARRAY_TASK_ID % (NUM_QAGGS * NUM_ENVS * NUM_SEEDS)))
QAGG_IDX=$((REM / (NUM_ENVS * NUM_SEEDS)))
REM2=$((REM % (NUM_ENVS * NUM_SEEDS)))
ENV_IDX=$((REM2 / NUM_SEEDS))
SEED_IDX=$((REM2 % NUM_SEEDS))

ETA=${ETAS[$ETA_IDX]}
QAGG=${QAGGS[$QAGG_IDX]}
TASK=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
TEMP=${WTEMP[$TASK]}

ETA_TAG=$(echo $ETA | tr -d '.')
TEMP_TAG=$(echo $TEMP | tr -d '.')
EXP_NAME="gfot_t${TEMP_TAG}_eta${ETA_TAG}"
[ "$QAGG" = "min" ] && EXP_NAME="${EXP_NAME}_qmin"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$ETA \
  agent.q_agg=$QAGG

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
