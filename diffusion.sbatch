#!/bin/sh
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1
 
srun python src/main.py