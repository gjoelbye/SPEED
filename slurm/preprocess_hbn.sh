#!/bin/bash

#SBATCH --partition=cyclopes
#SBATCH --job-name=hbn_test
#SBATCH --output=/scratch/linsk/hbn_test_preprocessed/logs/slurm-%J.out
#SBATCH --cpus-per-task=8 
#SBATCH --mem=20gb
#SBATCH --mail-user=linsk@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=24:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
python scripts/preprocess_hbn.py --config configs/hbn_test_titans.yaml

echo "Done: $(date +%F-%R:%S)"