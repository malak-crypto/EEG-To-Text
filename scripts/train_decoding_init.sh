#!/bin/bash
#SBATCH --job-name=T5Translator_training     # Job name
#SBATCH --partition=standard-gpu             # GPU partition (adjust if needed)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --gres=gpu:1                         # Request 1 V100 GPU
#SBATCH --mem=32G                            # Memory allocation
#SBATCH --time=5:00:00                       # Max runtime (HH:MM:SS)
#SBATCH --output=logs/T5Translator_%j.out    # Standard output log
#SBATCH --error=logs/T5Translator_%j.err     # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es

apps/2021
module load Anaconda3/2024.02-1

mkdir -p logs

source activate EEGToText



# Log training start time
echo "Starting training at: $(date)"

# Run the training script
python3 train_decoding.py \
    --model_name T5Translator \
    --task_name task1_task2_taskNRv2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    --lr1 0.00002 \
    --lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding\
    -cuda cuda:0

# Log job completion
echo "==============================="
echo "Job completed at: $(date)"
echo "Exit status: $?"

# Optional: Send email notification upon job completion
echo "Training job $SLURM_JOB_ID completed" | mail -s "Job $SLURM_JOB_ID finished" malak.hassanein@alumnos.upm.es
