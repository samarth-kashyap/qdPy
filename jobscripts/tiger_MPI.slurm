#!/bin/bash
#SBATCH --job-name=MCMC_qdPy_200_201      # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=30              # number of tasks per node
# SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=350G
# SBATCH --mem-per-cpu=18G        # memory per cpu-core (4G is default)
#SBATCH --time=32:00:00          # total run time limit (HH:MM:SS)
# SBATCH --mail-type=begin        # send email when job begins
# SBATCH --mail-type=end          # send email when job ends
# SBATCH --mail-user=sbdas@princeton.edu

module purge
module load anaconda3
# module load openmpi/gcc
conda activate helio

n0=0
lmin=200
lmax=240
maxiter=500

python qdpt_sampler.py --n0 ${n0} --lmin ${lmin} --lmax ${lmax} --maxiter ${maxiter}
mpiexec -n 40 python qdpt_sampler.py --n0 ${n0} --lmin ${lmin} --lmax ${lmax}  --usempi --maxiter ${maxiter} > slurm.out 2> slurm.err
