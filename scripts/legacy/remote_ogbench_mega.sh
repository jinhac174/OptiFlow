#!/bin/bash
#SBATCH --job-name=og-mega
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-407
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

# === LOCKED: 6 envs × 5 tasks × 8 seeds × 1M steps = 240 ===
LOCKED_FAMS=(cube_single_play scene_play cube_double_play puzzle_3x3_play puzzle_4x4_play humanoidmaze_medium_navigate)
declare -A LOCKED_CFG=(
  [cube_single_play]="2.0 0.1 mean"
  [scene_play]="2.0 0.1 mean"
  [cube_double_play]="2.0 0.0001 mean"
  [puzzle_3x3_play]="0.5 0.001 mean"
  [puzzle_4x4_play]="0.5 0.001 mean"
  [humanoidmaze_medium_navigate]="1.0 0.0001 mean"
)
for fam in "${LOCKED_FAMS[@]}"; do
  cfg="${LOCKED_CFG[$fam]}"
  for t in 1 2 3 4 5; do
    for sd in 1 2 3 4 5 6 7 8; do
      JOB_LIST+=("${fam}_task${t} $cfg $sd 1000000")
    done
  done
done

# === SWEEP: 2 envs × 2 tasks × 3 configs × 2 seeds × 500k = 24 ===
SWEEP_CONFIGS=("1.5 0.01 mean" "1.5 0.001 mean" "2.0 0.001 mean")
for fam in antsoccer_arena_navigate humanoidmaze_large_navigate; do
  for t in 1 2; do
    for cfg in "${SWEEP_CONFIGS[@]}"; do
      for sd in 2 4; do
        JOB_LIST+=("${fam}_task${t} $cfg $sd 500000")
      done
    done
  done
done

# === NEW: 12 envs × 2 tasks × 3 configs × 2 seeds × 500k = 144 ===
NEW_FAMS=(antmaze_large_navigate antmaze_large_stitch antmaze_large_explore antmaze_giant_navigate humanoidmaze_medium_stitch antsoccer_arena_stitch cube_single_noisy cube_double_noisy cube_triple_play cube_triple_noisy puzzle_4x4_noisy scene_noisy)
NEW_CONFIGS=("2.0 0.1 mean" "1.0 0.001 mean" "0.5 0.001 mean")
for fam in "${NEW_FAMS[@]}"; do
  for t in 1 2; do
    for cfg in "${NEW_CONFIGS[@]}"; do
      for sd in 2 4; do
        JOB_LIST+=("${fam}_task${t} $cfg $sd 500000")
      done
    done
  done
done

# Total: 240 + 24 + 144 = 408

read ENV_FULL TEMP E QAGG SEED STEPS <<< "${JOB_LIST[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$ENV_FULL" ]; then
  echo "No job for index $SLURM_ARRAY_TASK_ID, exiting"
  exit 0
fi

ENV_BASE=$(echo $ENV_FULL | sed 's/_task[0-9]//')
TASK_NUM=$(echo $ENV_FULL | grep -o 'task[0-9]' | grep -o '[0-9]')

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="task${TASK_NUM}_t${TEMP_TAG}_eta${ETA_TAG}_q${QAGG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $ENV_FULL seed=$SEED steps=$STEPS node=$(hostname)"

$PYTHON scripts/train.py \
  env=$ENV_FULL \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=$QAGG \
  train.max_steps=$STEPS \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_${QAGG},algo_gfot,${ENV_BASE}]"

echo "[$(date)] Done: $EXP_NAME $ENV_FULL seed=$SEED"
