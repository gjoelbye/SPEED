#!/bin/bash

#SBATCH --partition=cyclopes
#SBATCH --job-name=tuh_30s
#SBATCH --output=/scratch/linsk/tuh_test_preprocessed/logs/slurm-%J.out
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64gb
#SBATCH --mail-user=linsk@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=144:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
python scripts/preprocess_hbn.py --config configs/tuh_30s.yaml

echo "Done: $(date +%F-%R:%S)"