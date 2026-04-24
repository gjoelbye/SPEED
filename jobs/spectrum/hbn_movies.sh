#!/bin/bash
# =============================================================================
# DTU HPC (LSF) — hbn_movies downstream preprocessing, 5-shard array
# =============================================================================
# Pools DespicableMe + FunwithFractals + ThePresent + DiaryOfAWimpyKid into a
# single bucket with 30 s overlapping windows (stride 15 s) — movie name is
# primary 4-class label. Subject-wise splitting is essential (every window
# carries targets/subject_id to guarantee that).
#
# 5-shard array on the 'hpc' queue. Output →
# /work3/agjma/HBN_SPEED_downstream/movies/shard_XX/.
#
# Movie windows are 6000 samples each (200 Hz × 30 s) so the memory
# footprint per shard is ~10× the event-window configs. If an OOM kill
# lands, halve the event_tlen in the yaml or reduce n_jobs in
# _shard_common.sh.
#
# Submit: bsub < jobs/spectrum/hbn_movies.sh
# =============================================================================

#BSUB -q hpc
#BSUB -J "hbn_movies[1-5]"
#BSUB -oo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_movies_%J.%I.out
#BSUB -eo /zhome/33/6/147533/SPEED/logs/spectrum/hbn_movies_%J.%I.err
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"

set -euo pipefail

REPO_DIR="${HOME}/SPEED"
VENV_DIR="/work3/agjma/venvs/SPEED"
CONFIG="configs/downstream/hbn_movies.yaml"
NAME="hbn_movies"
N_SHARDS=5
SHARD=$((LSB_JOBINDEX - 1))

# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/spectrum/_shard_common.sh"
