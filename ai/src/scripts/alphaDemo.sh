#!/bin/bash
# Name of the job
#SBATCH --job-name=alphaDemo
# File output location
#SBATCH --output=/scratch/nmw264/alphaDemoTest.txt
#SBATCH --error=/scratch/nmw264/alphaDemoTest.err
# Time
#SBATCH --time=2:30
# Work dir
#SBATCH --chdir=/home/nmw264/dev/projects/CRAFT/ai/src/models/
# GPUs for task
#SBATCH --constraint=a100
#SBATCH --gpus=1
# CPUs for task
#SBATCH --cpus-per-task=2
# Mem allocation
#SBATCH --mem=4000


module load cuda/12.6.0
module load cudnn/8.9.3
module load anaconda3/2024.02


srun /home/nmw264/.conda/envs/tf_final/bin/python /home/nmw264/dev/projects/CRAFT/ai/src/models/alphaDemo.py /scratch/nmw264/alphaDemo1/
