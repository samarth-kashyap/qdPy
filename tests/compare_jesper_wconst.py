import numpy as np
import argparse
import sys

sys.path.append('/home/g.samarth/qdPy')
from ritzlavely import ritzLavelyPoly as RLP
datadir = '/scratch/g.samarth/qdPy/new_freqs'


def get_larr(n):
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
    return l_arr


# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="radial order", type=int)
args = parser.parse_args()
# }}} argparse

data_jesper = np.loadtxt('1d.split')
nl_jesper = data_jesper[:, :2].astype('int').tolist()

l_arr = get_larr(args.n)
l_arr = l_arr.astype('int')

# lmax = 130
# l_arr = l_arr[l_arr<lmax]

for ell in l_arr:
    rl = RLP(ell, 5)
    rl.get_Pjl()
    try:
        dpt_freqs = np.load(f'{datadir}/dpt_{args.n:02d}_{ell:03d}.npy')
        acoeff_dpt = rl.get_coeffs(dpt_freqs)
        jidx = nl_jesper.index([ell, args.n])
        acoeffs_jesper = data_jesper[jidx, 3:]*1.0e-3
        er1 = acoeff_dpt[1] - acoeffs_jesper[0]
        er2 = acoeff_dpt[2]
        er3 = acoeff_dpt[3] - acoeffs_jesper[1]
        er4 = acoeff_dpt[4]
        er5 = acoeff_dpt[5] - acoeffs_jesper[2]
        per1 = (acoeff_dpt[1] - acoeffs_jesper[0])/acoeffs_jesper[0]*100
        # per3 = (acoeff_dpt[3] - acoeffs_jesper[1])/acoeffs_jesper[1]*100
        # per5 = (acoeff_dpt[5] - acoeffs_jesper[2])/acoeffs_jesper[2]*100
        # print(f"{ell:5d} {er1:12.2e} {er2:12.2e} {er3:12.2e} {er4:12.2e} {er5:12.2e}")
        print(f"{ell:5d} {per1:6.2f}% ")#{per3:12.2e} {per5:12.2e}")
    except FileNotFoundError or IndexError:
        pass

