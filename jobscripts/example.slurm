#!/bin/bash
#SBATCH --job-name example
#SBATCH --out job.slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --mem 512
#SBATCH --time 30:00:00
#SBATCH --array=1-35

cmd="echo matlab array=${SLURM_ARRAY_TASK_ID}"

parallel --max-procs=${SLURM_CPUS_PER_TASK} "$cmd,subjob={1}" ::: {1}
