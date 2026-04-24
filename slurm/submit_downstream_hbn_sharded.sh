#!/bin/bash
# =============================================================================
# HBN downstream — sequential-datasets dispatcher
# =============================================================================
# Runs the HBN downstream configs ONE DATASET AT A TIME, but each dataset is
# parallelised across N_SHARDS SLURM array tasks. Datasets are chained with
# --dependency=afterany so dataset K only starts once every shard of K-1 has
# finished (or failed — individual shard failures don't block the next
# dataset; the user can requeue failed shards separately).
#
# Usage:
#   ./slurm/submit_downstream_hbn_sharded.sh [N_SHARDS] [CONFIG ...]
#
# Examples:
#   # All 6 configs, 8 shards each (default, matches 2×48-core cyclopes)
#   ./slurm/submit_downstream_hbn_sharded.sh
#
#   # 16 shards, only two configs (processed in the order given)
#   ./slurm/submit_downstream_hbn_sharded.sh 16 \
#       configs/downstream/hbn_ccd_rt.yaml \
#       configs/downstream/hbn_cbcl.yaml
#
#   # Single dataset fast: 24 shards of symbolsearch
#   ./slurm/submit_downstream_hbn_sharded.sh 24 \
#       configs/downstream/hbn_symbolsearch.yaml
# =============================================================================
set -euo pipefail

# Default 6 shards × --cpus-per-task=16 = 96 cores = exact cyclopes capacity.
# 8-shard × 80G mem would need 640G, doesn't fit the 510G cluster; 6 × 80 = 480G does.
# All 6 shards run concurrently.
N_SHARDS=${1:-6}
shift || true

if [ "$#" -eq 0 ]; then
    # Unified CCD config replaces the old ccd_rt + ccd_correct + ccd_rt_4s
    # trio (one preprocessing run, many per-window targets). Seq-learning
    # and movies are new in the multi-target buildout.
    CONFIGS=(
        configs/downstream/hbn_ccd.yaml
        configs/downstream/hbn_cbcl.yaml
        configs/downstream/hbn_rest_ec_eo.yaml
        configs/downstream/hbn_surroundsupp.yaml
        configs/downstream/hbn_symbolsearch.yaml
        configs/downstream/hbn_seqlearning6.yaml
        configs/downstream/hbn_seqlearning8.yaml
        configs/downstream/hbn_movies.yaml
    )
else
    CONFIGS=("$@")
fi

INNER=slurm/preprocess_downstream_hbn_shard.sh

prev_jid=""
for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || { echo "Missing config: $cfg" >&2; exit 1; }
    dep=()
    [ -n "$prev_jid" ] && dep=(--dependency=afterany:"$prev_jid")
    # Pass CONFIG and N_SHARDS via env prefix + --export=ALL. This is more
    # portable than `--export=ALL,CONFIG=X,N_SHARDS=Y`, which some Slurm
    # builds parse inconsistently.
    jid=$(CONFIG="$cfg" N_SHARDS="$N_SHARDS" \
          sbatch --parsable \
                 --export=ALL \
                 --array=0-$((N_SHARDS-1)) \
                 "${dep[@]}" \
                 "$INNER")
    tag=$(basename "$cfg" .yaml)
    echo "submitted $tag  jobid=$jid  shards=$N_SHARDS  ${dep[*]:-dep=none}"
    prev_jid=$jid
done
