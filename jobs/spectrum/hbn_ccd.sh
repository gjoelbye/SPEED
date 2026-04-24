#!/bin/bash
# =============================================================================
# DTU HPC (LSF) — hbn_ccd (unified CCD multi-target) downstream preprocessing
# =============================================================================
# Replaces the legacy hbn_ccd_rt.sh + hbn_ccd_rt_4s.sh + hbn_ccd_correct.sh
# trio — one pipeline run writes all CCD targets (RT, correct, target_side,
# button_side, non_target) + identity + biometrics to the targets/ group.
#
# 5-shard array on the 'hpc' queue. Output → /work3/agjma/HBN_SPEED_downstream/ccd/shard_XX/.
#
# Submit: bsub < jobs/spectrum/hbn_ccd.sh
# Monitor: bjobs -u $USER ; bpeek <JOBID>
# =============================================================================

#BSUB -q hpc
#BSUB -J "hbn_ccd[1-5]"
#BSUB -oo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_ccd_%J.%I.out
#BSUB -eo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_ccd_%J.%I.err
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

set -euo pipefail

REPO_DIR="${HOME}/SPEED"
VENV_DIR="/work3/agjma/venvs/SPEED"
CONFIG="configs/downstream/hbn_ccd.yaml"
NAME="hbn_ccd"
N_SHARDS=5
SHARD=$((LSB_JOBINDEX - 1))

# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/spectrum/_shard_common.sh"
