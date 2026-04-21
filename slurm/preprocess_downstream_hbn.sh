#!/bin/bash

# =============================================================================
# HBN downstream preprocessing — 6-task SLURM array for cyclopes
# =============================================================================
# One array index per HBN downstream config. Each task runs
# scripts/preprocess_downstream.py to completion, producing per-task HDF5
# batches under out_path defined in the config.
#
# BEFORE SUBMITTING: generate the per-task file lists that the configs'
# dataset_path entries point at. Run once:
#
#   python scripts/build_hbn_file_lists.py \
#       --dataset_root /dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original \
#       --out_dir /scratch/agjma/HBN_SPEED/lists
#
# RESUMING: each run writes .speed_cache.json in its out_path and skips files
# already processed. If a task hits the 48 h walltime, just `sbatch` again
# (same command) and it picks up from where it stopped.
#
# SUBMITTING:
#   sbatch slurm/preprocess_downstream_hbn.sh              # all 6 tasks
#   sbatch --array=0,1,2 slurm/preprocess_downstream_hbn.sh  # only CCD configs
#   sbatch --array=3 slurm/preprocess_downstream_hbn.sh      # only rest_ec_eo
# =============================================================================

#SBATCH --partition=cyclopes
#SBATCH --job-name=hbn_downstream
#SBATCH --output=/home/agjma/SPEED/logs/slurm-%A-%a-%x.out
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=48:00:00

CONFIGS=(
    configs/downstream/hbn_ccd_rt.yaml
    configs/downstream/hbn_ccd_correct.yaml
    configs/downstream/hbn_cbcl.yaml
    configs/downstream/hbn_rest_ec_eo.yaml
    configs/downstream/hbn_surroundsupp.yaml
    configs/downstream/hbn_symbolsearch.yaml
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
