#!/bin/bash
#SBATCH -p dgx                              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                                # Specify number of GPU per task, tasks per node
#SBATCH -t 10:00:00                         # Specify maximum time limit (hour: minute: second)
#SBATCH -A proj0183                         # Specify project name
#SBATCH -J finetune                         # Specify job name

# Activate your environment
source /tarafs/data/project/proj0183-ATS/finetune/miniconda3/bin/activate
conda activate finetune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 pattern-proj/seallmv2-causal-qlora-trainer.py