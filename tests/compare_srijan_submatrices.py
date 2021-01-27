import numpy as np


def scaling_factor(sm1, sm2):
    r12 = sm1 / sm2
    masknan = ~np.isnan(r12)
    r12vec = r12[masknan]
    if len(r12vec) > 0:
        sf = r12vec[0]
        return abs(sf), abs(r12vec - sf).sum()
    else:
        return 1.0, abs(sm1 - sm2).sum()

lmin, lmax = 195, 205
n = 0

sgk_datadir = '/scratch/g.samarth/qdPy/submatrices'
sri_datadir = '/scratch/g.samarth/Solar_Eigen_function/DR_Simulation_submatrices'

for ell1 in range(lmin, lmax+1):
    for ell2 in range(ell1, lmax+1):
        if (ell2 - ell1) > 5:
            continue
        submat_sgk = np.load(f'{sgk_datadir}/{ell1}.{ell2}.npy')
        mode_found = True
        try:
            submat_sri = np.load(f'{sri_datadir}/{n}_{ell1}_{n}_{ell2}.npy')
        except FileNotFoundError:
            try:
                submat_sri = np.load(f'{sri_datadir}/{n}_{ell2}_{n}_{ell1}.npy')
                submat_sri = submat_sri.T.conj()
            except FileNotFoundError:
                mode_found = False
                submat_sri = None

        if mode_found:
            sf, diffsum = scaling_factor(submat_sgk, submat_sri)
            print(f' n = {n}, ell1 = {ell1}, ell2 = {ell2} ' +
                f' scaleFactor - 1.0 = {(sf - 1):10.8e}, sum(diffmat) = {diffsum}')
        else:
            print(f' n = {n}, ell1 = {ell1}, ell2 = {ell2} ' +
                f' sum(diffmat) = NaN -- Mode not found')
