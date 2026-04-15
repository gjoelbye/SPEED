#!/bin/bash

#SBATCH --partition=cyclopes
#SBATCH --nodelist=comp-cpu01
#SBATCH --job-name=hbn_60s
#SBATCH --output=/scratch/agjma/HBN_SPEED/logs/slurm-%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=144:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
python scripts/preprocess.py --config configs/pretrain/hbn.yaml

echo "Done: $(date +%F-%R:%S)"
