#!/bin/bash
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 64
#SBATCH --time 00:05:00
#SBATCH --job-name run_diff_separation2
#SBATCH --mail-user ves@unm.edu
#SBATCH --mail-type ALL

mkdir -p logs results

module load parallel
module load miniconda3
pip install .
# source activate numpy
# TOTAL_OBS=64

# seq 0 $((TOTAL_OBS - 1)) | parallel --jobs $SLURM_NTASKS \
#     --sshloginfile $SLURM_JOB_NODELIST \
#     "srun --nodes=1 --ntasks=1 --exclusive python run_task.py {}"

# seq 0 $((TOTAL_OBS - 1)) | parallel --jobs $SLURM_NTASKS \
#     "python run_task.py {}"

seq 0 63 | parallel --jobs $SLURM_NTASKS \
    "python run_task.py {}"
