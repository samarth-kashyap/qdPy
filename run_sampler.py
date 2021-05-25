import os
os.system("mpirun -n 5 python qdpt_sampler.py --n 0 --lmin 200 --lmax 201 --usempi --maxiter 3")
