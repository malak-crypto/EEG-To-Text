#!/bin/bash

#SBATCH --job-name=T5Translator_training
#SBATCH --partition=standard-gpu                    # GPU partition (adjust if different)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=4                  # CPU cores per task (4 for data loading)
#SBATCH --gres=gpu:v100:1                  # Request 1 V100 GPU
#SBATCH --mem=32G                          # Memory allocation (adjust as needed)
#SBATCH --time=5:00:00                    # Maximum runtime (48 hours, adjust as needed)
#SBATCH --output=logs/T5Translator_%j.out  # Standard output log
#SBATCH --error=logs/T5Translator_%j.err   # Error log
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es  # Your email address

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust according to CeSViMa's module system)
module purge
module load anaconda3  # or whatever anaconda module is available

# Activate your conda environment
# Replace 'EEGToText' with your actual environment name
source activate EEGToText

# Verify GPU availability
echo "Available GPUs:"
nvidia-smi

# Set CUDA devices (SLURM will assign the GPU automatically)
export CUDA_VISIBLE_DEVICES=0

# Print environment information
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Change to your project directory (adjust path as needed)
# cd /path/to/your/project

# Run your training script
echo "Starting training at: $(date)"
python3 train_decoding.py --model_name T5Translator \
    --task_name task1_task2_taskNRv2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit status: $?"

# Optional: Send completion notification
echo "Training job $SLURM_JOB_ID completed" | mail -s "Job $SLURM_JOB_ID finished" malak.hassanein@alumnos.upm.es
