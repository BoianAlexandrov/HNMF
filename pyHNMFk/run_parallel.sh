#!/bin/bash
#SBATCH --partition debug
#SBATCH --nodes 21
#SBATCH --ntasks-per-node 2
#SBATCH --time 00:05:00
#SBATCH --job-name run_diff_separation
#SBATCH --mail-user ves@unm.edu
#SBATCH --mail-type ALL

mkdir -p logs results

module load parallel
module load miniconda3
pip install .
# source activate numpy
TOTAL_OBS=42

# seq 0 $((TOTAL_OBS - 1)) | parallel --jobs $SLURM_NTASKS \
#     --sshloginfile $SLURM_JOB_NODELIST \
#     "srun --nodes=1 --ntasks=1 --exclusive python run_task.py {}"

seq 0 $((TOTAL_OBS - 1)) | parallel --jobs $SLURM_NTASKS \
    "python run_task.py {}"
