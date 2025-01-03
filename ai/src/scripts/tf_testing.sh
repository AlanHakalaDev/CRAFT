#!/bin/bash
# Name of the job
#SBATCH --job-name=tf_testing
# File output location
#SBATCH --output=/scratch/nmw264/tf_testing.txt
#SBATCH --error=/scratch/nmw264/tf_testing.err
# Time
#SBATCH --time=00:30
# Work dir
#SBATCH --chdir=/home/nmw264/dev/projects/CRAFT_AI/Capstone_DVD/Train_7_category_lists
# GPUs for task
#SBATCH --constraint=a100
#SBATCH --gpus=1
# CPUs for task
#SBATCH --cpus-per-task=1
# Mem allocation
#SBATCH --mem=1000


module load cuda/12.5.0
module load cudnn/8.9.3
module load anaconda3/2024.02


srun /home/nmw264/.conda/envs/tf_final/bin/python -c "import tensorflow as tf; print(\"Num Gpus Avail: \", len(tf.config.list_physical_devices('GPU')))"
