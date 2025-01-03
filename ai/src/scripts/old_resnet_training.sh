#!/bin/bash
# Name of the job
#SBATCH --job-name=original_resnet_training
# File output location
#SBATCH --output=/scratch/nmw264/original_resnet_training.txt
#SBATCH --error=/scratch/nmw264/original_resnet_training.err
# Time
#SBATCH --time=15:00
# Work dir
#SBATCH --chdir=/home/nmw264/dev/projects/CRAFT_AI/Capstone_DVD/Train_7_category_lists
# GPUs for task
#SBATCH --constraint=a100
#SBATCH --gpus=1
# CPUs for task
#SBATCH --cpus-per-task=1
# Mem allocation
#SBATCH --mem=13000


module load cuda/11.2
module load cudnn/8.9.3
module load anaconda3/2024.02


srun /home/nmw264/.conda/envs/old_tf/bin/python /home/nmw264/dev/projects/CRAFT_AI/Capstone_DVD/Train_7_category_lists/ResNet152V2_model_processor_lists.py --set 1 --run 1
