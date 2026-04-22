#!/bin/bash
# =============================================================================
# DTU HPC (LSF) — hbn_cbcl downstream preprocessing, 5-shard array
# =============================================================================
# Submits one array job with 5 parallel shards on the 'hpc' queue. Each shard
# processes ~1/5 of the HBN file list and writes batches to
# /work3/agjma/HBN_SPEED_downstream/<task>/shard_XX/.
#
# Submit: bsub < jobs/spectrum/hbn_cbcl.sh
# Monitor: bjobs -u $USER ; bpeek <JOBID>
# =============================================================================

#BSUB -q hpc
#BSUB -J "hbn_cbcl[1-5]"
#BSUB -oo $HOME/SPEED/logs/spectrum/hbn_cbcl_%J.%I.out
#BSUB -eo $HOME/SPEED/logs/spectrum/hbn_cbcl_%J.%I.err
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

set -euo pipefail

REPO_DIR="${HOME}/SPEED"
VENV_DIR="/work3/agjma/venvs/SPEED"
CONFIG="configs/downstream/hbn_cbcl.yaml"
NAME="hbn_cbcl"
N_SHARDS=5
SHARD=$((LSB_JOBINDEX - 1))

# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/spectrum/_shard_common.sh"
