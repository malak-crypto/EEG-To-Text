#!/bin/bash
#SBATCH --job-name=stats                     # Job name
#SBATCH --partition=standard-gpu             # GPU partition (adjust if needed)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --gres=gpu:1                         # Request 1 V100 GPU
#SBATCH --mem=32G                            # Memory allocation
#SBATCH --time=2:00:00                       # Max runtime (HH:MM:SS)
#SBATCH --output=stats_%j.out                # Standard output log
#SBATCH --error=stats_%j.err                 # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es

module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"
conda activate /home/w314/w314139/.conda/envs/EEGToText

# Run the training script
python3 data_stats.py
