#!/bin/bash
#SBATCH --job-name=gfot-d4rl
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-85
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

# Configs: TEMP ETA ENV SEED
# ---- Hopper: high-temp exploration (what you asked for) ----
# ---- Halfcheetah medium/replay: push past BC ceiling ----
# ---- Walker: fine-tune around best known (t1.5, eta0.05) ----
# ---- Halfcheetah expert: lock in winner ----
CONFIGS=(
  # Hopper — 3 envs × 4 temps × 3 etas = 36 jobs
  "1.5  0.1  hopper_medium 2"
  "1.5  1.0  hopper_medium 2"
  "1.5  10.0 hopper_medium 2"
  "5.0  0.1  hopper_medium 2"
  "5.0  1.0  hopper_medium 2"
  "5.0  10.0 hopper_medium 2"
  "10.0 0.1  hopper_medium 2"
  "10.0 1.0  hopper_medium 2"
  "10.0 10.0 hopper_medium 2"
  "30.0 0.1  hopper_medium 2"
  "30.0 1.0  hopper_medium 2"
  "30.0 10.0 hopper_medium 2"
  "1.5  0.1  hopper_medium_replay 2"
  "1.5  1.0  hopper_medium_replay 2"
  "1.5  10.0 hopper_medium_replay 2"
  "5.0  0.1  hopper_medium_replay 2"
  "5.0  1.0  hopper_medium_replay 2"
  "5.0  10.0 hopper_medium_replay 2"
  "10.0 0.1  hopper_medium_replay 2"
  "10.0 1.0  hopper_medium_replay 2"
  "10.0 10.0 hopper_medium_replay 2"
  "30.0 0.1  hopper_medium_replay 2"
  "30.0 1.0  hopper_medium_replay 2"
  "30.0 10.0 hopper_medium_replay 2"
  "1.5  0.1  hopper_medium_expert 2"
  "1.5  1.0  hopper_medium_expert 2"
  "1.5  10.0 hopper_medium_expert 2"
  "5.0  0.1  hopper_medium_expert 2"
  "5.0  1.0  hopper_medium_expert 2"
  "5.0  10.0 hopper_medium_expert 2"
  "10.0 0.1  hopper_medium_expert 2"
  "10.0 1.0  hopper_medium_expert 2"
  "10.0 10.0 hopper_medium_expert 2"
  "30.0 0.1  hopper_medium_expert 2"
  "30.0 1.0  hopper_medium_expert 2"
  "30.0 10.0 hopper_medium_expert 2"

  # Halfcheetah medium — 4 temps × 3 etas = 12 jobs
  "0.3  0.1  halfcheetah_medium 2"
  "0.3  1.0  halfcheetah_medium 2"
  "0.3  10.0 halfcheetah_medium 2"
  "0.5  0.1  halfcheetah_medium 2"
  "0.5  1.0  halfcheetah_medium 2"
  "0.5  10.0 halfcheetah_medium 2"
  "1.5  0.1  halfcheetah_medium 2"
  "1.5  1.0  halfcheetah_medium 2"
  "1.5  10.0 halfcheetah_medium 2"
  "5.0  0.1  halfcheetah_medium 2"
  "5.0  1.0  halfcheetah_medium 2"
  "5.0  10.0 halfcheetah_medium 2"

  # Halfcheetah medium_replay — 4 temps × 3 etas = 12 jobs
  "0.3  0.1  halfcheetah_medium_replay 2"
  "0.3  1.0  halfcheetah_medium_replay 2"
  "0.3  10.0 halfcheetah_medium_replay 2"
  "0.5  0.1  halfcheetah_medium_replay 2"
  "0.5  1.0  halfcheetah_medium_replay 2"
  "0.5  10.0 halfcheetah_medium_replay 2"
  "1.5  0.1  halfcheetah_medium_replay 2"
  "1.5  1.0  halfcheetah_medium_replay 2"
  "1.5  10.0 halfcheetah_medium_replay 2"
  "5.0  0.1  halfcheetah_medium_replay 2"
  "5.0  1.0  halfcheetah_medium_replay 2"
  "5.0  10.0 halfcheetah_medium_replay 2"

  # Walker push — fine-tune around best (t1.5, eta0.05) with 2 seeds
  "1.5  0.02 walker2d_medium 2"
  "1.5  0.05 walker2d_medium 2"
  "1.5  0.1  walker2d_medium 2"
  "1.5  0.02 walker2d_medium 4"
  "1.5  0.05 walker2d_medium 4"
  "1.5  0.1  walker2d_medium 4"
  "1.5  0.02 walker2d_medium_replay 2"
  "1.5  0.05 walker2d_medium_replay 2"
  "1.5  0.1  walker2d_medium_replay 2"
  "1.5  0.02 walker2d_medium_replay 4"
  "1.5  0.05 walker2d_medium_replay 4"
  "1.5  0.1  walker2d_medium_replay 4"
  "1.5  0.02 walker2d_medium_expert 2"
  "1.5  0.05 walker2d_medium_expert 2"
  "1.5  0.1  walker2d_medium_expert 2"
  "1.5  0.02 walker2d_medium_expert 4"
  "1.5  0.05 walker2d_medium_expert 4"
  "1.5  0.1  walker2d_medium_expert 4"

  # Halfcheetah expert — lock in winner at 2 seeds
  "1.5  0.05 halfcheetah_medium_expert 2"
  "1.5  0.1  halfcheetah_medium_expert 2"
  "2.0  0.05 halfcheetah_medium_expert 2"
  "2.0  0.1  halfcheetah_medium_expert 2"
  "1.5  0.05 halfcheetah_medium_expert 4"
  "1.5  0.1  halfcheetah_medium_expert 4"
  "2.0  0.05 halfcheetah_medium_expert 4"
  "2.0  0.1  halfcheetah_medium_expert 4"
)

read TEMP ETA TASK SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

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