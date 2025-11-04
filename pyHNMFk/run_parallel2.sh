#!/bin/bash
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 64
#SBATCH --time 00:10:00
#SBATCH --job-name run_diff_separation
#SBATCH --mail-user ves@unm.edu
#SBATCH --mail-type ALL
#SBATCH --array=0-1                # 2 batches: 0-63 and 64-103
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

mkdir -p logs results

module load parallel
module load miniconda3
pip install .

TOTAL_OBS=104
BATCH_SIZE=64

# Calculate start and end indices for this batch
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE - 1))

# Don't go past the last observation
if [ $END -ge $TOTAL_OBS ]; then
    END=$((TOTAL_OBS - 1))
fi

echo "Batch ${SLURM_ARRAY_TASK_ID}: Processing observations ${START} to ${END}"

# Run this batch in parallel
seq $START $END | parallel --jobs $SLURM_NTASKS \
    "python run_task2.py {}"