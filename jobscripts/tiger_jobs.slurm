B#!/bin/bash
#SBATCH --job-name=qdPy         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=35               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sbdas@princeton.edu

module load anaconda3
conda activate helio

while read l
do
    srun -N 1 -n 1 python qdpt.py --n0 0 --l0 $l > ./jobscripts/$l.out 2> ./jobscripts/$l.err &
done < ./jobscripts/ells.dat
wait
