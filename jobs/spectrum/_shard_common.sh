# =============================================================================
# DTU HPC (LSF) — shared shard body for HBN downstream preprocessing
# =============================================================================
# Sourced by jobs/spectrum/hbn_<task>.sh. Expects these env vars set by the
# caller before the source line:
#   REPO_DIR   absolute path to the SPEED repo (usually $HOME/SPEED)
#   VENV_DIR   absolute path to the SPEED venv (usually /work3/agjma/venvs/SPEED)
#   CONFIG     path to a downstream yaml config, relative to REPO_DIR
#              (e.g. configs/downstream/hbn_ccd_rt.yaml)
#   NAME       short tag for the dataset, used only in log lines
#   N_SHARDS   number of LSF array tasks for this submission (default 5)
#   SHARD      0-based shard index (derived from $LSB_JOBINDEX - 1)
#
# Responsibilities (mirrors slurm/preprocess_downstream_hbn_shard.sh):
#   1. Load python 3.12 module and activate the SPEED venv.
#   2. Read dataset_path + out_path from the base yaml and re-root them from
#      /scratch/agjma/ (cyclopes) to /work3/agjma/ (DTU HPC). Configs stay
#      cluster-agnostic; path translation happens here.
#   3. Interleaved-split the file list into N_SHARDS stripes; write this
#      shard's stripe to $SHARD_DIR/input.txt.
#   4. Materialise a shard-local yaml overriding dataset_path / out_path /
#      log_path / n_jobs.
#   5. Run scripts/preprocess_downstream.py on the shard yaml.
# =============================================================================

module purge
module load python3/3.12.11
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_DIR}"

SHARD_TAG=$(printf '%02d' "${SHARD}")

# Read list + base output dir from the base yaml, re-rooting cyclopes paths.
read -r list_file base_out <<<"$(python - <<PY
import yaml
c = yaml.safe_load(open("${CONFIG}"))
dtu = lambda p: p.replace("/scratch/agjma/", "/work3/agjma/")
print(dtu(c["dataset_path"]), dtu(c["out_path"]))
PY
)"

SHARD_DIR="${base_out}/shard_${SHARD_TAG}"
SHARD_LIST="${SHARD_DIR}/input.txt"
SHARD_YAML="${SHARD_DIR}/config.yaml"
SHARD_LOG="${SHARD_DIR}/log.txt"

mkdir -p "${SHARD_DIR}" $HOME/SPEED/logs/spectrum

# Interleaved split: line k goes to shard (k % N_SHARDS). Matches the SLURM
# inner script; verified partitioning + zero overlap on cyclopes.
awk -v n="${N_SHARDS}" -v s="${SHARD}" 'NR % n == s' "${list_file}" > "${SHARD_LIST}"

# Materialise the shard-local yaml. preprocess_downstream.py's positional
# args (dataset_path / out_path / log_path) can't be overridden on the CLI,
# so we edit the config in place. n_jobs=12 caps peak mem at ~12×5GB = 60GB
# in the 80GB cgroup, leaving 4 cores of slack for main/BLAS/IO jitter.
python - <<PY
import yaml
with open("${CONFIG}") as f:
    c = yaml.safe_load(f)
c["dataset_path"] = "${SHARD_LIST}"
c["out_path"]     = "${SHARD_DIR}"
c["log_path"]     = "${SHARD_LOG}"
c["n_jobs"]       = 12
with open("${SHARD_YAML}", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY

echo "============================================"
echo "Dataset: ${NAME}"
echo "Node:    $(hostname)"
echo "Job:     ${LSB_JOBID:-unknown} shard ${SHARD}/${N_SHARDS} (LSB_JOBINDEX=${LSB_JOBINDEX:-?})"
echo "Config:  ${CONFIG}"
echo "Shard:   ${SHARD_YAML}"
echo "Files:   $(wc -l < "${SHARD_LIST}")"
echo "Out:     ${SHARD_DIR}"
echo "Start:   $(date +%F-%T)"
echo "============================================"

python scripts/preprocess_downstream.py --config "${SHARD_YAML}"

echo "============================================"
echo "Done:    $(date +%F-%T)"
echo "============================================"
