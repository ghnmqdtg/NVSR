#!/bin/sh

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

python main.py