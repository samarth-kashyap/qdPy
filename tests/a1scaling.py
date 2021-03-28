import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import simps

parser = argparse.ArgumentParser()
parser.add_argument('--n', help='Radial order', type=int)
args = parser.parse_args()

n = args.n

eig_dir = "/scratch/g.samarth/Solar_Eigen_function/eig_files"
data_dir = "/scratch/g.samarth/Solar_Eigen_function/data_files"
r = np.loadtxt(f"{data_dir}/r.dat")
rho = np.loadtxt(f"{data_dir}/rho.dat")
nl = np.loadtxt(f"{data_dir}/nl.dat").astype('int').tolist()
mask95 = r > 0.65
mask1 = r <= 1.0
r95 = r[mask95]


def lambda2(n, ell):
    idx = nl.index([n, ell])
    V = np.loadtxt(f"{eig_dir}/V{idx}.dat")
    integrand = V*V*rho*r
    l2 = simps(integrand, x=r)
    return -l2

def lambda1(n, ell):
    idx = nl.index([n, ell])
    U = np.loadtxt(f"{eig_dir}/U{idx}.dat")
    V = np.loadtxt(f"{eig_dir}/V{idx}.dat")
    integrand = (U*U - 2*U*V - V*V)*rho*r
    l1 = simps(integrand, x=r)
    return -l1


def plot_knls(n):
    ellmin, ellmax = 150, 300
    ells = np.arange(ellmin, ellmax)
    knls = np.zeros(len(ells), dtype=np.float32)
    norm = np.zeros(len(ells), dtype=np.float32)
    for il, ell in enumerate(ells):
        L = np.sqrt(ell*(ell+1))
        idx = nl.index([n, ell])
        U = np.loadtxt(f"{eig_dir}/U{idx}.dat")
        V = np.loadtxt(f"{eig_dir}/V{idx}.dat")
        norm[il] = simps((r*r*rho*(U*U + L*L*V*V))[mask1], x=r[mask1])
        knls[il] = simps((r*r*rho*(U*U + L*L*V*V - 2*U*V - V*V))[mask1], x=r[mask1])

    fig = plt.figure()
    plt.plot(ells, norm, '+k', label='norm')
    plt.plot(ells, knls, 'xb', label='knl')
    return fig, norm, knls

if __name__ == "__main__":
    ell_list = np.array([50, 100, 150, 200, 250, 300])
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for il, ell in enumerate(ell_list):
        L = np.sqrt(ell*(ell+1))
        idx = nl.index([n, ell])
        U = np.loadtxt(f"{eig_dir}/U{idx}.dat")[mask95]
        V = np.loadtxt(f"{eig_dir}/V{idx}.dat")[mask95]
        axs.flatten()[0].semilogy(r95, abs(U/L/V), label='$\ell$ = ' + f'{ell}')
        axs.flatten()[1].semilogy(r95, U, 'k', label='U for ' + '$\ell$ = ' + f'{ell}')
        axs.flatten()[1].semilogy(r95, L*V, '--k', label='LV for ' + '$\ell$ = ' + f'{ell}')
    axs.flatten()[0].set_xlabel("$r/R_\odot$")
    axs.flatten()[1].set_xlabel("$r/R_\odot$")
    axs.flatten()[0].set_ylabel("$U/LV$")
    axs.flatten()[1].set_ylabel("U, LV")
    fig.suptitle(f"n = {n}")
    axs.flatten()[0].legend()
    axs.flatten()[1].legend()
    plt.show(fig)

   
