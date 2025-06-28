#!/bin/bash
#SBATCH --job-name=env_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=env_test_%j.out
#SBATCH --error=env_test_%j.err

module purge
module load miniconda

echo "Loaded modules"

# If 'source activate' doesn't work, use 'conda activate' after sourcing base
source ~/.bashrc
conda activate EEGToText

echo "Conda environment activated"

which python
python --version
