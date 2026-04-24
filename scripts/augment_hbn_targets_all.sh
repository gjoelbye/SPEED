#!/bin/bash
# =============================================================================
# Runner for scripts/augment_hbn_targets.py — walks every existing HBN bucket
# (merged on cyclopes, sharded on DTU HPC) and appends the targets/ group
# in place. Idempotent: re-runs skip buckets whose targets/ is already
# populated. Pass --force to overwrite.
#
# Usage:
#   ./scripts/augment_hbn_targets_all.sh                 # run all
#   ./scripts/augment_hbn_targets_all.sh --dry-run       # preview
#   ./scripts/augment_hbn_targets_all.sh --force         # overwrite
#   ./scripts/augment_hbn_targets_all.sh --only symbolsearch,ccd_rt
#
# Host-specific path roots — edit to match your environment:
#   MERGED_ROOT     cyclopes post-merge buckets (single run of combined_*.hdf5)
#   SHARDED_ROOT    DTU HPC per-shard outputs (shard_NN/combined_*.hdf5)
#   RAW_ROOT        HBN raw tree (events.tsv + participants.tsv live here)
# =============================================================================
set -euo pipefail

MERGED_ROOT="${MERGED_ROOT:-/scratch/agjma/HBN_SPEED_downstream_merged}"
SHARDED_ROOT="${SHARDED_ROOT:-/work3/agjma/HBN_SPEED_downstream}"
RAW_ROOT="${RAW_ROOT:-/dtu-compute/EEG_at_scale/HBN_EEG/hbn_eeg_original}"
STEM_INDEX="${STEM_INDEX:-/tmp/hbn_stem_index.pkl}"
N_JOBS="${N_JOBS:-8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
AUGMENT="${SCRIPT_DIR}/augment_hbn_targets.py"

# (bucket_name task event_tmin) — canonical per-bucket args.
# Tasks correspond to speed/downstream_targets.py dispatch keys.
declare -a BUCKETS=(
    "ccd          ccd          -0.5"   # new unified config
    "ccd_rt       ccd_rt        0.5"   # legacy 2 s post-stim
    "ccd_rt_4s    ccd_rt       -0.5"   # legacy 4 s superset window
    "ccd_correct  ccd_correct  -0.5"
    "symbolsearch symbolsearch -1.0"
    "surroundsupp surroundsupp  0.0"
    "rest_ec_eo   rest_ec_eo    0.0"
    "cbcl         cbcl          0.0"
    "seqlearning6 seqlearning6 -0.1"
    "seqlearning8 seqlearning8 -0.1"
    "movies       movies        0.0"
)

EXTRA_ARGS=()
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --only) ;;   # consumed below
        --only=*) ONLY="${arg#--only=}" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done
# Also accept "--only foo,bar" (separate token).
while [ "${1:-}" != "" ]; do
    if [ "$1" == "--only" ] && [ -n "${2:-}" ]; then
        ONLY="$2"
        shift 2
    else
        shift || break
    fi
done

run_one() {
    local bucket_name="$1"
    local task="$2"
    local event_tmin="$3"
    local path="$4"
    if [ ! -d "$path" ]; then
        echo "  SKIP $bucket_name  ($path does not exist)"
        return 0
    fi
    echo "=== $bucket_name  (task=$task tmin=$event_tmin)  $path ==="
    python "$AUGMENT" \
        --bucket "$path" \
        --task "$task" \
        --event-tmin "$event_tmin" \
        --raw-root "$RAW_ROOT" \
        --stem-index "$STEM_INDEX" \
        --n-jobs "$N_JOBS" \
        "${EXTRA_ARGS[@]}"
}

for row in "${BUCKETS[@]}"; do
    read -r bucket_name task event_tmin <<< "$row"
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in
            *",$bucket_name,"*) ;;
            *) continue ;;
        esac
    fi
    # Try merged-root first; then sharded-root (shard_* subdirs of the same name).
    run_one "$bucket_name" "$task" "$event_tmin" "${MERGED_ROOT}/${bucket_name}"
    run_one "$bucket_name" "$task" "$event_tmin" "${SHARDED_ROOT}/${bucket_name}"
done

echo
echo "All buckets processed. Summary:"
echo "  merged root:   $MERGED_ROOT"
echo "  sharded root:  $SHARDED_ROOT"
echo "  raw root:      $RAW_ROOT"
