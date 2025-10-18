#!/bin/bash -l
#SBATCH --job-name=hbn_single
#SBATCH --account=project_465000940
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0                   # use all memory on the node
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --output=/users/madsenan/slurm-%J.out
#SBATCH --hint=nomultithread

echo "Nodes: $SLURM_NODELIST"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

module load cray-python/3.11.7
source /users/madsenan/speed/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Hardcoded config
CONFIG_FILE="/users/madsenan/SPEED/configs/tuh_lumi.yaml"
OUT_FILE="/users/madsenan/SPEED/slurm_tuh.out"

echo "Running $CONFIG_FILE using $SLURM_CPUS_PER_TASK cores and all memory"
srun python scripts/preprocess_hbn.py --config "$CONFIG_FILE" > "$OUT_FILE" 2>&1

echo "Done: $(date +%F-%R:%S)"
