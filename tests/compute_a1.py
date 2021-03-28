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

if __name__ == "__main__":
    tl3 = np.array([])
    tl2 = np.array([])
    tl1 = np.array([])
    a1 = np.array([])
    ells = np.arange(100, 300)
    for ell in ells:
        ld1 = lambda1(n, ell)
        ld2 = lambda2(n, ell)
        # print(f"ell = {ell}; ld1/ell3 = {ld1/ell/ell/ell:15.7e}; " +
              # f"ld2/ell2 = {ld2/ell/ell:15.7e}; ld2/ell = {ld2/ell:15.7e}")
        a1_ = np.sqrt(3.0/4.0/np.pi) * 430./8. * (ld1/ell**3 + ld2/ell**2 + ld2/ell)
        print(f"ell = {ell}; a1 = {a1_:10.3e}")
        a1 = np.append(a1, a1_)
        tl3 = np.append(tl3, ld1/ell/ell/ell)
        tl2 = np.append(tl2, ld2/ell/ell)
        tl1 = np.append(tl1, ld2/ell)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(7, 4))
    axs.flatten()[0].semilogy(ells, abs(tl3), "--k", label="$\lambda_1\ell^{-3}$")
    axs.flatten()[0].semilogy(ells, abs(tl2), "b", label="$\lambda_2\ell^{-2}$")
    axs.flatten()[0].semilogy(ells, abs(tl1), "-.r", label="$\lambda_2\ell^{-1}$")
    axs.flatten()[0].set_xlabel('$\ell$')
    axs.flatten()[0].legend()

    axs.flatten()[1].plot(ells, -tl3, "--k", label="$\lambda_1\ell^{-3}$")
    axs.flatten()[1].plot(ells, -tl1, "-.r", label="$\lambda_2\ell^{-1}$")
    axs.flatten()[1].set_xlabel('$\ell$')
    axs.flatten()[1].legend()
    plt.show()
    
    
