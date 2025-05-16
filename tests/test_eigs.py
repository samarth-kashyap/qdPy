import numpy as np
import matplotlib.pyplot as plt
import argparse


# {{{ def create_argparser():
def create_argparser():
    """Creates argument parser for arguments passed during
    execution of script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="radial order", type=int)
    parser.add_argument("--l", help="angular degree", type=int)
    args = parser.parse_args()
    return args
# }}} create_argparser()



rA = np.loadtxt("/scratch/g.samarth/get-solar-eigs/efs_Antia/snrnmais_files/data_files/r.dat")
rJ = np.loadtxt("/scratch/g.samarth/get-solar-eigs/efs_Jesper/snrnmais_files/data_files/r.dat")
nl_antia = np.loadtxt("/scratch/g.samarth/get-solar-eigs/efs_Antia/snrnmais_files/data_files/nl.dat").astype("int").tolist()
nl_jesper = np.loadtxt("/scratch/g.samarth/get-solar-eigs/efs_Jesper/snrnmais_files/data_files/nl.dat").astype("int").tolist()
ARGS = create_argparser()
n, l = ARGS.n, ARGS.l
RMIN = 0.96
RMAX = 1.1

def compare_eigs_UV(n, l): 
    idx_antia = nl_antia.index([n, l]) 
    idx_jesper = nl_jesper.index([n, l]) 
    U_antia = np.loadtxt(f"/scratch/g.samarth/get-solar-eigs/efs_Antia/snrnmais_files/eig_files/U{idx_antia}.dat") 
    V_antia = np.loadtxt(f"/scratch/g.samarth/get-solar-eigs/efs_Antia/snrnmais_files/eig_files/V{idx_antia}.dat") 
    U_jesper = np.loadtxt(f"/scratch/g.samarth/get-solar-eigs/efs_Jesper/snrnmais_files/eig_files/U{idx_jesper}.dat") 
    V_jesper = np.loadtxt(f"/scratch/g.samarth/get-solar-eigs/efs_Jesper/snrnmais_files/eig_files/V{idx_jesper}.dat") 
    return (U_antia, V_antia), (U_jesper, V_jesper)

def plot_eigs(n, l):
    UVA, UVJ = compare_eigs_UV(n, l)
    rAmin_idx = np.argmin(abs(rA - RMIN))
    rAmax_idx = np.argmin(abs(rA - RMAX))
    rJmin_idx = np.argmin(abs(rJ - RMIN))
    rJmax_idx = np.argmin(abs(rJ - RMAX))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    axs.flatten()[0].plot(rA[rAmin_idx:rAmax_idx],
                          UVA[0][rAmin_idx:rAmax_idx],
                          'k', label='Antia', linewidth=0.8)
    axs.flatten()[0].plot(rJ[rJmin_idx:rJmax_idx],
                          UVJ[0][rJmin_idx:rJmax_idx],
                          '--r', label='JCD', linewidth=0.8)
    axs.flatten()[2].plot(rA[rAmin_idx:rAmax_idx],
                          UVA[1][rAmin_idx:rAmax_idx],
                          'k', label='Antia', linewidth=0.8)
    axs.flatten()[2].plot(rJ[rJmin_idx:rJmax_idx],
                          UVJ[1][rJmin_idx:rJmax_idx],
                          '--r', label='JCD', linewidth=0.8)

    argmaxA = np.argmax(UVA[0])
    argmaxJ = np.argmax(UVJ[0])
    width = 2e-3

    rmin_common = min(rA[argmaxA] - width, rJ[argmaxJ] - width)
    rmax_common = max(rA[argmaxA] + width, rJ[argmaxJ] + width)
    rAmin_idx = np.argmin(abs(rA - rmin_common))
    rAmax_idx = np.argmin(abs(rA - rmax_common))
    rJmin_idx = np.argmin(abs(rJ - rmin_common))
    rJmax_idx = np.argmin(abs(rJ - rmax_common))
    axs.flatten()[1].plot(rA[rAmin_idx:rAmax_idx],
                          UVA[0][rAmin_idx:rAmax_idx],
                          'k', label='Antia', linewidth=0.8)
    axs.flatten()[1].plot(rJ[rJmin_idx:rJmax_idx],
                          UVJ[0][rJmin_idx:rJmax_idx],
                          '--r', label='JCD', linewidth=0.8)
    axs.flatten()[3].plot(rA[rAmin_idx:rAmax_idx],
                          UVA[1][rAmin_idx:rAmax_idx],
                          'k', label='Antia', linewidth=0.8)
    axs.flatten()[3].plot(rJ[rJmin_idx:rJmax_idx],
                          UVJ[1][rJmin_idx:rJmax_idx],
                          '--r', label='JCD', linewidth=0.8)
    axs.flatten()[0].legend()
    axs.flatten()[1].legend()

    axs.flatten()[0].set_xlabel("$r/R_\odot$")
    axs.flatten()[1].set_xlabel("$r/R_\odot$")
    axs.flatten()[2].set_xlabel("$r/R_\odot$")
    axs.flatten()[3].set_xlabel("$r/R_\odot$")
    axs.flatten()[0].set_ylabel("$U_{n\ell}$")
    axs.flatten()[1].set_ylabel("$U_{n\ell}$")
    axs.flatten()[2].set_ylabel("$V_{n\ell}$")
    axs.flatten()[3].set_ylabel("$V_{n\ell}$")
    fig.suptitle(f"$n$ = {n}, $\ell$ = {l}")
    return fig

if __name__ == "__main__":
    fig = plot_eigs(n, l)
    fig.savefig(f"/scratch/g.samarth/qdPy/plots/eig_compare_{n:02d}_{l:03d}.pdf")

