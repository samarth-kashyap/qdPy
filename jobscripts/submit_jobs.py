import os
for n in range(6):
    """
    fname = f"gnup_ps_{n:02d}.sh"
    with open(fname, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#PBS -N ps.n{n:02d}.data\n")
        f.write(f"#PBS -o psout.n{n:02d}.log\n")
        f.write(f"#PBS -e pserr.n{n:02d}.log\n")
        f.write("#PBS -l select=1:ncpus=4:mem=8gb\n")
        f.write("#PBS -l walltime=00:55:00\n")
        f.write("#PBS -q small\n")
        f.write("echo \"Starting at \"`date`\n")
        f.write("cd /home/g.samarth/Woodard2013/\n")
        f.write("export PATH=$PATH:/home/apps/GnuParallel/bin\n")
        f.write("export TERM=xterm\n")
        f.write("cd $PBS_WORKDIR\n")
        f.write("echo $PBS_JOBID\n")
        f.write(f"parallel --jobs 4 < /home/g.samarth/Woodard2013/job_scripts/ipjobs_ps_{n:02d}.sh\n")
        f.write("echo \"Finished at \"`date`\n")
    os.system(f"qsub {fname}")
    """

    fname = f"gnup_cs_{n:02d}.sh"
    with open(fname, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#PBS -N pQ-DPT.n{n:02d}\n")
        f.write(f"#PBS -o gdnsout.n{n:02d}.log\n")
        f.write(f"#PBS -e gdnserr.n{n:02d}.log\n")
        f.write("#PBS -l select=1:ncpus=12:mem=24gb\n")
        f.write("#PBS -l walltime=04:30:00\n")
        if n%2==0:
            f.write("#PBS -q large\n")
        else:
            f.write("#PBS -q small\n")
        f.write("echo \"Starting at \"`date`\n")
        f.write("cd /home/g.samarth/qdPy/\n")
        f.write("export PATH=$PATH:/home/apps/GnuParallel/bin\n")
        f.write("export TERM=xterm\n")
        f.write("echo $PBS_JOBID\n")
        f.write(f"parallel --jobs 12 < /home/g.samarth/qdPy/jobscripts/ipjobs_{n:02d}.sh\n")
        f.write("echo \"Finished at \"`date`\n")
    os.system(f"qsub {fname}")
