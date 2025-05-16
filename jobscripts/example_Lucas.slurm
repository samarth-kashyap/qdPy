#!/bin/bash
#SBATCH --job-name=array
#SBATCH --output=array_%A_%a.out
# SBATCH --error=array_%A_%a.err
#SBATCH --array=1-35
#SBATCH --time=00:10:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem=4G
# Print the task id.
# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
OFFSET=150
l="$(($SLURM_ARRAY_TASK_ID+$OFFSET))"
# echo $l
python qdpt.py --n0 0 --l0 $l 
# Add lines here to run your computations.
