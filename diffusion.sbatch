#!/bin/sh
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1
 
CUDA_LAUNCH_BLOCKING=1 srun python src/train.py