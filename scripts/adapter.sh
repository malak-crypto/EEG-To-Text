#!/bin/bash
#SBATCH --job-name=AdapterPreFT       # Job name
#SBATCH --partition=standard-gpu      # GPU partition
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4             
#SBATCH --gres=gpu:1                  
#SBATCH --mem=32G                     
#SBATCH --time=2:00:00                
#SBATCH --output=adapter_ft_%j.out    
#SBATCH --error=adapter_ft_%j.err     
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=malak.hassanein@alumnos.upm.es

module load Anaconda3/2024.02-1

source activate /home/w314/w314139/.conda/envs/EEGToText

echo "Adapter fine-tuning start: $(date)"

python3 adapter_preencoder_finetune.py \
  -checkpoint checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_True_2steptraining_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
  -config config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_True_2steptraining_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
  -model BrainTranslator \
  -batch_size 8 \
  -epochs 2 \
  -lr 1e-4 \
  -cuda cuda:0 \
  -output checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_True_2steptraining_b32_20_30_2e-05_2e-05_unique_sent_EEG_adapt.pt

echo "Adapter fine-tuning end: $(date), status: $?"
