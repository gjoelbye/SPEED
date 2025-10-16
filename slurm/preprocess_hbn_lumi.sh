#!/bin/bash -l
#SBATCH --job-name=hbn_batch
#SBATCH --account=project_465000940
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agjma@dtu.dk
#SBATCH --output=/users/madsenan/slurm-%J.out

echo "Nodes: $SLURM_NODELIST"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

module load cray-python/3.11.7
source /users/madsenan/speed/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Build configs ds005505.yaml ... ds005516.yaml
cfg_list=""
for i in $(seq 5505 5516); do
  printf -v cfg "configs/lumi/ds%06d.yaml" "$i"
  cfg_list+="$cfg "
done
export CFG_LIST="$cfg_list"

# One srun that launches 12 tasks. Slurm packs 4 per node.
srun --ntasks=12 \
     --ntasks-per-node=4 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --hint=nomultithread \
     --cpu-bind=cores \
     bash -lc '
       # Split CFG_LIST into an array
       read -r -a configs <<< "$CFG_LIST"
       cfg="${configs[$SLURM_PROCID]}"
       tag="$(basename "$cfg" .yaml)"
       outfile="${tag}.out"
       echo "Task $SLURM_PROCID on $(hostname) running $cfg"
       python scripts/preprocess_hbn.py --config "$cfg" > "$outfile" 2>&1
     '

echo "Done: $(date +%F-%R:%S)"
