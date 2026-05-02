#!/bin/bash
#SBATCH --job-name=d4rl-mix
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-99
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

JOB_LIST=()

WR_CONFIGS=("2.0 0.05 min" "1.5 0.05 min" "2.0 0.05 mean")
for cfg in "${WR_CONFIGS[@]}"; do
  for sd in 2 4 6 8; do
    JOB_LIST+=("walker2d_medium_replay $cfg $sd")
  done
done

for sd in 1 2 3 4 5 6 7 8; do
  JOB_LIST+=("walker2d_medium_expert 2.5 0.1 min $sd")
done

HCM_CONFIGS=("0.5 0.1 min" "0.5 0.001 mean" "1.0 0.1 min")
for cfg in "${HCM_CONFIGS[@]}"; do
  for sd in 2 4 6 8; do
    JOB_LIST+=("halfcheetah_medium $cfg $sd")
  done
done

HCR_CONFIGS=("0.5 0.1 min" "0.5 0.001 mean" "1.0 0.1 min")
for cfg in "${HCR_CONFIGS[@]}"; do
  for sd in 2 4 6 8; do
    JOB_LIST+=("halfcheetah_medium_replay $cfg $sd")
  done
done

for sd in 1 2 3 4 5 6 7 8; do
  JOB_LIST+=("halfcheetah_medium_expert 0.5 0.001 mean $sd")
done

HOPPER_CONFIGS=("10.0 10.0 min" "5.0 0.0001 min" "2.0 0.1 min" "5.0 10.0 mean")
HOPPER_ENVS=(hopper_medium hopper_medium_replay hopper_medium_expert)
for cfg in "${HOPPER_CONFIGS[@]}"; do
  for env in "${HOPPER_ENVS[@]}"; do
    for sd in 2 4 6 8; do
      JOB_LIST+=("$env $cfg $sd")
    done
  done
done

read TASK TEMP E QAGG SEED <<< "${JOB_LIST[$SLURM_ARRAY_TASK_ID]}"

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="t${TEMP_TAG}_eta${ETA_TAG}_q${QAGG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $TASK seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$TASK \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=$QAGG \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_${QAGG},algo_gfot,ablation]"

echo "[$(date)] Done: $EXP_NAME $TASK seed=$SEED"
