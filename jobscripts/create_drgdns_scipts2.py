import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", help='radial order',
                    type=int)
args = parser.parse_args()

pythonpath = "/home/sbdas/.conda/envs/helio/bin/python"
execpath = "/home/sbdas/Research/Helioseismology/qdPy/qdpt.py"

hmi_data = np.loadtxt("/scratch/gpfs/sbdas/Helioseismology/get-solar-eigs/efs_Antia/data/HMI/hmi.6328.36")

n = args.n
mask_n = hmi_data[:, 1] == n
l_arr = hmi_data[mask_n, 0]
len_ell = len(l_arr)

filename = f"/home/sbdas/Research/Helioseismology/qdPy/jobscripts/ipjobs_{n:02d}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        ell = int(l_arr[i])
        exec_cmd = f"{pythonpath} {execpath} --n0 {n} --l0 {ell}\n"
        f.write(exec_cmd)
