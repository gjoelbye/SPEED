#!/bin/bash

#SBATCH --partition=cyclopes
#SBATCH --job-name=downstream
#SBATCH --output=/home/agjma/SPEED/logs/slurm-%A-%a-%x.out
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=48:00:00

CONFIGS=(
    configs/downstream_eegmat.yaml
    configs/downstream_mumtaz2016.yaml
    configs/downstream_bcic_iv_2a.yaml
    configs/downstream_isruc.yaml
    configs/downstream_hmc.yaml
    configs/downstream_eegmmidb.yaml
    configs/downstream_chbmit.yaml
    configs/downstream_shu_mi.yaml
    configs/downstream_mobi.yaml
    configs/downstream_siena.yaml
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"
echo ""

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate EEGModel

mkdir -p /home/agjma/SPEED/logs

python scripts/preprocess_downstream.py --config "${CONFIG}"

echo "Done: $(date +%F-%R:%S)"
