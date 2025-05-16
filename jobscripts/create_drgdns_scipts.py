import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", help='radial order',
                    type=int)
args = parser.parse_args()

pythonpath = "/home/g.samarth/anaconda3/bin/python"
execpath = "/home/g.samarth/qdPy/qdpt.py"

n = args.n
l_arr = np.array([])
l0_arr = np.load(f"/scratch/g.samarth/csfit/lall_radleak.npy")
n0_arr = np.load(f"/scratch/g.samarth/csfit/nall_radleak.npy")
l0_arr = l0_arr[n0_arr==n]
l_arr = np.append(l_arr, l0_arr)
print(l0_arr)
try:
    l1_arr = np.load(f"/scratch/g.samarth/csfit/l{n:02d}_used.npy")
    l2_arr = np.load(f"/scratch/g.samarth/csfit/l{n:02d}_unused.npy")
    l_arr = np.append(l_arr, l1_arr)
    l_arr = np.append(l_arr, l2_arr)
    print(l1_arr)
    print(l2_arr)
except FileNotFoundError:
    pass

l_arr = np.unique(l_arr)
l_arr = np.arange(l_arr.min(), l_arr.max()+1)
len_ell = len(l_arr)
filename = f"/home/g.samarth/qdPy/jobscripts/ipjobs_{n:02d}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        ell = int(l_arr[i])
        exec_cmd = f"{pythonpath} {execpath} --n0 {n} --l0 {ell}\n"
        f.write(exec_cmd)
