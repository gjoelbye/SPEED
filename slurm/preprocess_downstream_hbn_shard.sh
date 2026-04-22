#!/bin/bash
# =============================================================================
# HBN downstream preprocessing — single shard of a single dataset
# =============================================================================
# One SLURM array task = one shard of one HBN downstream config.
# Meant to be submitted by slurm/submit_downstream_hbn_sharded.sh, which
# injects CONFIG and N_SHARDS via --export. The shard index comes from
# SLURM_ARRAY_TASK_ID.
#
# The shard (a) slices the base config's file list into N_SHARDS stripes
# (interleaved: line k goes to shard k % N_SHARDS), (b) materialises a
# shard-local derived yaml that overrides dataset_path / out_path / log_path
# to per-shard subdirectories, then (c) runs preprocess_downstream.py.
# =============================================================================

#SBATCH --partition=cyclopes
#SBATCH --job-name=hbn_ds_shard
#SBATCH --output=/home/agjma/SPEED/logs/slurm-%A-%a-%x.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --export=ALL
#SBATCH --time=48:00:00

set -euo pipefail
: "${CONFIG:?CONFIG env var required (base yaml path)}"
: "${N_SHARDS:?N_SHARDS env var required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required — submit via sbatch --array=...}"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate EEGModel

SHARD=$SLURM_ARRAY_TASK_ID
SHARD_TAG=$(printf '%02d' "$SHARD")

# Pull dataset_path + out_path out of the base yaml.
read -r list_file base_out <<<"$(python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['dataset_path'], c['out_path'])
")"

SHARD_DIR="${base_out}/shard_${SHARD_TAG}"
SHARD_LIST="${SHARD_DIR}/input.txt"
SHARD_YAML="${SHARD_DIR}/config.yaml"
SHARD_LOG="${SHARD_DIR}/log.txt"

mkdir -p "$SHARD_DIR" /home/agjma/SPEED/logs

# Interleaved split: each line's shard is ((NR-1) % N_SHARDS) under 1-based NR,
# or equivalently NR % N_SHARDS when s ranges 0..N-1 and we match lines with
# (NR % N) == s. Verified partitioning + zero overlap on symbolsearch.txt.
awk -v n="$N_SHARDS" -v s="$SHARD" 'NR % n == s' "$list_file" > "$SHARD_LIST"

# Materialise the shard-local yaml — preprocess_downstream.py's three
# positional args (dataset_path / out_path / log_path) cannot be overridden
# on the CLI, so we edit the config in place. Also injects n_jobs, which
# preprocess_downstream.py reads to size its ProcessPoolExecutor.
python - <<PY
import yaml, os
with open("$CONFIG") as f:
    c = yaml.safe_load(f)
c["dataset_path"] = "$SHARD_LIST"
c["out_path"]     = "$SHARD_DIR"
c["log_path"]     = "$SHARD_LOG"
c["n_jobs"]       = 12  # hardcoded: 4 cores of slack vs 16 allocated for main/BLAS/IO jitter; caps peak mem at 12×5GB ≈ 60GB in the 80GB cgroup
with open("$SHARD_YAML", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY

echo "Node:   $(hostname)"
echo "Start:  $(date +%F-%R:%S)"
echo "Config: $CONFIG"
echo "Shard:  $SHARD / $N_SHARDS   files: $(wc -l < "$SHARD_LIST")"
echo "Out:    $SHARD_DIR"
echo

python scripts/preprocess_downstream.py --config "$SHARD_YAML"

echo
echo "Done:   $(date +%F-%R:%S)"
