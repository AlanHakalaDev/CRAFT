#!/bin/bash
# Name of the job
#SBATCH --job-name=SwinTransformerTraining
# File output location
#SBATCH --output=/scratch/nmw264/output.txt
#SBATCH --error=/scratch/nmw264/error.err
# Time
#SBATCH --time=30:00
# Work dir
#SBATCH --chdir=/home/nmw264/dev/projects/CRAFT/ai/src/models/
# GPUs for task
#SBATCH --constraint=a100
#SBATCH --gpus=1
# CPUs for task
#SBATCH --cpus-per-task=1
# Mem allocation
#SBATCH --mem=13000


module load cuda/12.5.0
module load cudnn/8.9.3
module load anaconda3/2024.02


srun /home/nmw264/.conda/envs/tf_final/bin/python /home/nmw264/dev/projects/CRAFT/ai/src/models/swinTransformer.py /scratch/nmw264/swinTransformer/
