import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', help='Radial order', type=int)
parser.add_argument('--l', help='Spherical harmonic degree', type=int)
args = parser.parse_args()

if __name__ == "__main__":
    cwd_sgk = "/scratch/g.samarth/Solar_Eigen_function/new_freqs"
    cwd_sbd = "/scratch/g.samarth/qdPy/new_freqs"
    enn, ell = args.n, args.l
    qdpt_sgk = np.load(f"{cwd_sgk}/qdpt_{enn:02d}_{ell:03d}.npy")
    qdpt_sbd = np.load(f"{cwd_sbd}/qdpt_{enn:02d}_{ell:03d}.npy")
    dpt_sgk = np.load(f"{cwd_sgk}/dpt_{enn:02d}_{ell:03d}.npy")
    dpt_sbd = np.load(f"{cwd_sbd}/dpt_{enn:02d}_{ell:03d}.npy")

    diffsum_qdpt = abs(qdpt_sgk - qdpt_sbd).sum()
    diffsum_dpt = abs(dpt_sgk - dpt_sbd).sum()
    print(f'sum(diff) QDPT = {diffsum_qdpt}')
    print(f'sum(diff) DPT = {diffsum_dpt}')

    plt.figure(figsize=(5, 9))
    plt.subplot(311)
    plt.plot(qdpt_sgk - qdpt_sbd, 'r', label='QDPT diff')
    plt.legend()

    plt.subplot(312)
    plt.plot(dpt_sgk - dpt_sbd, 'b', label='DPT diff')
    plt.legend()

    plt.subplot(313)
    plt.plot(qdpt_sgk - dpt_sgk, '--k', label='QDPT - DPT (sgk)')
    plt.plot(qdpt_sbd - dpt_sbd, '--g', label='QDPT - DPT (sbd)')
    plt.legend()
    plt.show()

