import os
for n in range(1):
    fname = f"/home/sbdas/Research/Helioseismology/qdPy/jobscripts/gnup_cs_{n:02d}.slurm"
    with open(fname, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=qdPy.n{n:02d}\n")
        f.write(f"#SBATCH --output=qdPy-out.n{n:02d}.log\n")
        f.write(f"#SBATCH --error=qdPy-serr.n{n:02d}.log\n")
        f.write("#SBATCH --nodes=2\n")
        f.write("#SBATCH --ntasks=80\n")
        f.write("#SBATCH --time=00:10:00\n")
        f.write("#SBATCH --mem-per-cpu=3G\n")
        f.write("echo \"Starting at \"`date`\n")
        f.write(f"parallel --jobs 80 < /home/sbdas/Research/Helioseismology/qdPy/jobscripts/ipjobs_{n:02d}.sh\n")
        f.write("echo \"Finished at \"`date`\n")
    os.system(f"sbatch {fname}")
