import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', help='Radial order', type=int)
parser.add_argument('--l', help='Spherical harmonic degree', type=int)
args = parser.parse_args()

if __name__ == "__main__":
    cwd_sgk = "/scratch/g.samarth/Solar_Eigen_function/new_freqs"
    cwd_sri = "/scratch/g.samarth/qdPy/new_freqs"
    enn, ell = args.n, args.l
    qdpt_sgk = np.load(f"{cwd_sgk}/qdpt_{enn:02d}_{ell:03d}.npy")
    qdpt_sri = np.load(f"{cwd_sri}/qdpt_{enn:02d}_{ell:03d}.npy")
    dpt_sgk = np.load(f"{cwd_sgk}/dpt_{enn:02d}_{ell:03d}.npy")
    dpt_sri = np.load(f"{cwd_sri}/dpt_{enn:02d}_{ell:03d}.npy")

    diffsum_qdpt = abs(qdpt_sgk - qdpt_sri).sum()
    diffsum_dpt = abs(dpt_sgk - dpt_sri).sum()
    print(f'sum(diff) QDPT = {diffsum_qdpt}')
    print(f'sum(diff) DPT = {diffsum_dpt}')

    plt.figure()
    plt.plot(qdpt_sgk - qdpt_sri, 'r', label='QDPT diff')
    plt.plot(dpt_sgk - dpt_sri, 'b', label='DPT diff')
    plt.plot(qdpt_sgk - dpt_sgk, '--k', label='QDPT - DPT (sgk)')
    plt.plot(qdpt_sri - dpt_sri, '--g', label='QDPT - DPT (sri)')
    plt.legend()
    plt.show()

