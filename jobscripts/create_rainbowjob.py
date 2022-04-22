import os
import subprocess
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
snrnmais_dir = dirnames[2]

_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]

nlist = np.loadtxt(f'{scratch_dir}/data_files/nlist.txt').astype('int')
llist = np.loadtxt(f'{scratch_dir}/data_files/llist.txt').astype('int')

with open("ipjobs_rainbow.sh", "w") as f:
    for i in range(len(nlist)):
        n0, l0 = nlist[i], llist[i]
        f.write(f"{pythonpath} {package_dir}/qdpt.py --n0 {nlist[i]}" +
                f" --l0 {llist[i]} --smax 7 &>logs/{n0}.{l0}.log\n")


jobname = f"sgk.qdPy"

gnup_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=112:mem=700gb
#PBS -l walltime=06:00:00
#PBS -q clx
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
source activate jaxpyro
/homes/hanasoge/parallel/bin/parallel --jobs 80 < $PBS_O_WORKDIR/ipjobs_rainbow.sh
echo \"Finished at \"`date`
"""

slurm_str = f"""#!/bin/bash                                                                  
#SBATCH --job-name={jobname}                                                              
#SBATCH --output=out-{jobname}.log                                                            
#SBATCH --error=err-{jobname}.log
#SBATCH --nodes=1                                                                             
#SBATCH --ntasks-per-node=40                                                                  
#SBATCH --mem=180G                                                                            
#SBATCH --time=15:00:00                                                                       
echo \"Starting at \"`date`                                                                   
module purge                                                                                  
module load anaconda3                                                                         
conda activate jax-gpu                                                                        
echo \"Starting at \"`date`                                                                   
parallel --jobs 3 < {package_dir}/jobscripts/ipjobs_rainbow.sh                                
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_rainbow.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/gnup_rainbow.slurm", "w") as f:
    f.write(slurm_str)
