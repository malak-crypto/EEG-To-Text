#!/bin/bash

#SBATCH --job-name=gpt                     # Job name
#SBATCH --partition=standard-gpu            # GPU partition
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=4                   # CPU cores per task
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --mem=32G                           # Memory allocation
#SBATCH --time=2:00:00                      # Max runtime (HH:MM:SS)
#SBATCH --output=gpt_%j.out                 # Standard output log
#SBATCH --error=gpt_%j.err                  # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL          # Email notifications
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es

# --- Embed your OpenAI API key here ---
export OPENAI_API_KEY="your_api_key_here"
# ---------------------------------------
export OPENAI_MODEL="gpt-4.1"
# Load Anaconda and activate environment
module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"
conda activate /home/w314/w314139/.conda/envs/EEGToText

# Verify the key is set (printed to output)
env | grep OPENAI_API_KEY

# Run the reconstruction
python3 reconstruct.py
