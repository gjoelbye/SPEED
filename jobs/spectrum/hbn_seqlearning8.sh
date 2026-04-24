#!/bin/bash
# =============================================================================
# DTU HPC (LSF) — hbn_seqlearning8 downstream preprocessing, 5-shard array
# =============================================================================
# 5-shard array on the 'hpc' queue. Output →
# /work3/agjma/HBN_SPEED_downstream/seqlearning8/shard_XX/.
#
# Submit: bsub < jobs/spectrum/hbn_seqlearning8.sh
# =============================================================================

#BSUB -q hpc
#BSUB -J "hbn_seqlearning8[1-5]"
#BSUB -oo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_seqlearning8_%J.%I.out
#BSUB -eo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_seqlearning8_%J.%I.err
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

set -euo pipefail

REPO_DIR="${HOME}/SPEED"
VENV_DIR="/work3/agjma/venvs/SPEED"
CONFIG="configs/downstream/hbn_seqlearning8.yaml"
NAME="hbn_seqlearning8"
N_SHARDS=5
SHARD=$((LSB_JOBINDEX - 1))

# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/spectrum/_shard_common.sh"
