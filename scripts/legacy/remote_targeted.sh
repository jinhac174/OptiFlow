#!/bin/bash
#SBATCH --job-name=og-targ
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-125
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node09,node18,node31

ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

JOB_LIST=()

# === cube_single_play t=2.0 eta=0.1 qmean ===
# task1 seed4 x2
JOB_LIST+=("cube_single_play_task1 2.0 0.1 mean 4")
JOB_LIST+=("cube_single_play_task1 2.0 0.1 mean 4")
# task2 seeds 3,4,8 x1
for sd in 3 4 8; do JOB_LIST+=("cube_single_play_task2 2.0 0.1 mean $sd"); done
# task3 seeds 7,8
for sd in 7 8; do JOB_LIST+=("cube_single_play_task3 2.0 0.1 mean $sd"); done
# task4 seeds 1,2
for sd in 1 2; do JOB_LIST+=("cube_single_play_task4 2.0 0.1 mean $sd"); done
# task4 seed5 x2
JOB_LIST+=("cube_single_play_task4 2.0 0.1 mean 5")
JOB_LIST+=("cube_single_play_task4 2.0 0.1 mean 5")
# task5 seeds 2,3,4,8
for sd in 2 3 4 8; do JOB_LIST+=("cube_single_play_task5 2.0 0.1 mean $sd"); done

# === puzzle_4x4_play t=2.0 eta=0.0001 qmean tasks 1-5 seeds 1-8 ===
for t in 1 2 3 4 5; do
  for sd in 1 2 3 4 5 6 7 8; do
    JOB_LIST+=("puzzle_4x4_play_task${t} 2.0 0.0001 mean $sd")
  done
done

# === puzzle_4x4_play t=2.0 eta=0.1 qmean tasks 1-5 seeds 1-8 ===
for t in 1 2 3 4 5; do
  for sd in 1 2 3 4 5 6 7 8; do
    JOB_LIST+=("puzzle_4x4_play_task${t} 2.0 0.1 mean $sd")
  done
done

# === scene_play t=2.0 eta=0.1 qmean ===
JOB_LIST+=("scene_play_task1 2.0 0.1 mean 4")
for sd in 1 2 3 4 5 6 7 8; do JOB_LIST+=("scene_play_task2 2.0 0.1 mean $sd"); done
for sd in 1 4 5 6 7 8; do JOB_LIST+=("scene_play_task3 2.0 0.1 mean $sd"); done
for sd in 1 2 3 4 5 6 7 8; do JOB_LIST+=("scene_play_task4 2.0 0.1 mean $sd"); done
for sd in 1 2 3 4 5 6 7 8; do JOB_LIST+=("scene_play_task5 2.0 0.1 mean $sd"); done

# Total: 15 + 40 + 40 + 31 = 126

read ENV_FULL TEMP E QAGG SEED <<< "${JOB_LIST[$SLURM_ARRAY_TASK_ID]}"
[ -z "$ENV_FULL" ] && exit 0

ENV_BASE=$(echo $ENV_FULL | sed 's/_task[0-9]//')
TASK_NUM=$(echo $ENV_FULL | grep -o 'task[0-9]' | grep -o '[0-9]')

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="task${TASK_NUM}_t${TEMP_TAG}_eta${ETA_TAG}_q${QAGG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $ENV_FULL seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$ENV_FULL \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=$QAGG \
  train.max_steps=1000000 \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_${QAGG},algo_gfot,${ENV_BASE},targeted]"

echo "[$(date)] Done: $EXP_NAME $ENV_FULL seed=$SEED"
