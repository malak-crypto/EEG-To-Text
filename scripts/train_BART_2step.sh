#!/bin/bash
#SBATCH --job-name=BART_training     # Job name
#SBATCH --partition=standard-gpu             # GPU partition (adjust if needed)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --gres=gpu:1                         # Request 1 V100 GPU
#SBATCH --mem=32G                            # Memory allocation
#SBATCH --time=5:00:00                       # Max runtime (HH:MM:SS)
#SBATCH --output=BART_%j.out    # Standard output log
#SBATCH --error=BART_%j.err     # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es

module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"
conda activate /home/w314/w314139/.conda/envs/EEGToText

# Log training start time
echo "Starting training at: $(date)"

# Run the training script
python3 train_decoding.py \
    -m BrainTranslator \
    -t task1_task2_taskNRv2 \
    -2step \
    -pre \
    -load1 \
    -ne1 20 \
    -ne2 30 \
    -train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding \
    -cuda cuda:0

# Log job completion
echo "==============================="
echo "Job completed at: $(date)"
echo "Exit status: $?"

# Optional: Send email notification upon job completion
echo "Training job $SLURM_JOB_ID completed" | mail -s "Job $SLURM_JOB_ID finished" malak.hassanein@alumnos.upm.es
