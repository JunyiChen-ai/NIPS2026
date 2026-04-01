#!/bin/bash
#SBATCH --job-name=reproduce
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jehc223/NIPS2026/reproduce/logs/reproduce_%j.out
#SBATCH --error=/data/jehc223/NIPS2026/reproduce/logs/reproduce_%j.err

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

cd /data/jehc223/NIPS2026/reproduce

# Step 1: Create val splits (if not already done)
python create_val_split.py

# Step 2: Run all methods
python run_all.py
