#!/bin/bash
#SBATCH --job-name=gfot-og
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node18,node31

# Use flowrl-ogbench venv on remote (adjust path if different)
ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

# NOT on coworker's list: humanoidmaze, cube-single, scene, puzzle-3x3
# Each env runs all 5 tasks (task1..task5)
# Per-family hyperparameters:
#   humanoidmaze: wt=1, eta=0.001   (coworker's humanoidmaze setting)
#   cube-single:  wt=2, eta=0.1     (coworker's cube setting)
#   scene:        wt=2, eta=0.1     (no info, guess from cube)
#   puzzle-3x3:   wt=2, eta=0.1     (no info, guess from cube)

CONFIGS=(
  # humanoidmaze-medium-navigate — 5 tasks
  "1.0 0.001 humanoidmaze_medium_navigate_task1"
  "1.0 0.001 humanoidmaze_medium_navigate_task2"
  "1.0 0.001 humanoidmaze_medium_navigate_task3"
  "1.0 0.001 humanoidmaze_medium_navigate_task4"
  "1.0 0.001 humanoidmaze_medium_navigate_task5"

  # cube-single-play — 5 tasks
  "2.0 0.1 cube_single_play_task1"
  "2.0 0.1 cube_single_play_task2"
  "2.0 0.1 cube_single_play_task3"
  "2.0 0.1 cube_single_play_task4"
  "2.0 0.1 cube_single_play_task5"

  # scene-play — 5 tasks
  "2.0 0.1 scene_play_task1"
  "2.0 0.1 scene_play_task2"
  "2.0 0.1 scene_play_task3"
  "2.0 0.1 scene_play_task4"
  "2.0 0.1 scene_play_task5"

  # puzzle-3x3-play — 5 tasks
  "2.0 0.1 puzzle_3x3_play_task1"
  "2.0 0.1 puzzle_3x3_play_task2"
  "2.0 0.1 puzzle_3x3_play_task3"
  "2.0 0.1 puzzle_3x3_play_task4"
  "2.0 0.1 puzzle_3x3_play_task5"
)

read TEMP ETA TASK <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
SEED=1

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