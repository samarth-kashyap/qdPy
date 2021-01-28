import os
for n in range(25):
    fname = f"gnup_qdpy_{n:02d}.sh"
    with open(fname, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#PBS -N Q-DPT.n{n:02d}\n")
        f.write(f"#PBS -o gdnsout.n{n:02d}.log\n")
        f.write(f"#PBS -e gdnserr.n{n:02d}.log\n")
        f.write("#PBS -l select=1:ncpus=2:mem=4gb\n")
        f.write("#PBS -l walltime=02:30:00\n")
        if n%2==0:
            f.write("#PBS -q large\n")
        else:
            f.write("#PBS -q small\n")
        f.write("echo \"Starting at \"`date`\n")
        f.write("cd /home/g.samarth/qdPy/\n")
        f.write("export PATH=$PATH:/home/apps/GnuParallel/bin\n")
        f.write("export TERM=xterm\n")
        f.write("echo $PBS_JOBID\n")
        f.write(f"parallel --jobs 2 < /home/g.samarth/qdPy/jobscripts/ipjobs_{n:02d}.sh\n")
        f.write("echo \"Finished at \"`date`\n")
    os.system(f"qsub {fname}")
