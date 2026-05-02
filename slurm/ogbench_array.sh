#!/bin/bash
# Generic Slurm array runner for OGBench sweeps.

ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
export PYTHONUNBUFFERED=1
NVIDIA=$ENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

SWEEP_YAML=$1
echo "[$(date)] Job $SLURM_ARRAY_TASK_ID  sweep=$SWEEP_YAML  node=$(hostname)"
$PYTHON scripts/sweep_runner.py run --sweep $SWEEP_YAML --idx $SLURM_ARRAY_TASK_ID
echo "[$(date)] Done idx=$SLURM_ARRAY_TASK_ID"
