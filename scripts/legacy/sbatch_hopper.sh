#!/bin/bash
#SBATCH --job-name=suot-hop
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090
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

# TEMP TAU APPEND QAGG LQ
CONFIGS=(
  "0.5 0.05 false min 0.0"
  "0.5 0.1  false min 0.0"
  "1.0 0.05 false min 0.0"
  "1.5 0.05 false min 0.0"
  "2.0 1.0  false min 0.0"
  "1.5 1.0  false min 0.0"
  "1.5 1.0  true  min 0.0"
  "0.5 0.05 false min 0.001"
)

ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert)
SEEDS=(2 4)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=${#SEEDS[@]}
JOBS_PER_CFG=$((NUM_ENVS * NUM_SEEDS))

CFG_IDX=$((SLURM_ARRAY_TASK_ID / JOBS_PER_CFG))
REMAINDER=$((SLURM_ARRAY_TASK_ID % JOBS_PER_CFG))
ENV_IDX=$((REMAINDER / NUM_SEEDS))
SEED_IDX=$((REMAINDER % NUM_SEEDS))

read TEMP TAU APPEND QAGG LQ <<< "${CONFIGS[$CFG_IDX]}"
TASK=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

TEMP_TAG=$(echo $TEMP | tr -d '.')
TAU_TAG=$(echo $TAU | tr -d '.')
EXP_NAME="t${TEMP_TAG}_tau${TAU_TAG}"
[ "$APPEND" = "false" ] && EXP_NAME="${EXP_NAME}_nostu"
[ "$QAGG" = "min" ] && EXP_NAME="${EXP_NAME}_qmin"
[ "$LQ" != "0.0" ] && EXP_NAME="${EXP_NAME}_q$(echo $LQ | tr -d '.')"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.w_temperature=$TEMP \
  agent.uot_tau=$TAU \
  agent.append_student_proposals=$APPEND \
  agent.q_agg=$QAGG \
  agent.lambda_q=$LQ

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
