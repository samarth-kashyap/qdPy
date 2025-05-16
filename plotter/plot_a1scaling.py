import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

data_dir = "/home/g.samarth/qdPy/postprocess"

dptvals = np.loadtxt(f"{data_dir}/dpt.430nhz.6.splits")
enns = dptvals[:, 1]
mask0 = enns==0
ells = dptvals[:, 0][mask0]
a1 = dptvals[:, 3][mask0]
r = np.loadtxt("/scratch/g.samarth/qdPy/r.dat")
rho = np.loadtxt("/scratch/g.samarth/qdPy/rho.dat")
nl = np.loadtxt("/scratch/g.samarth/qdPy/nl.dat").astype("int").tolist()
N = np.zeros_like(ells, dtype=np.float32)

for il, ell in enumerate(ells):
    idx = nl.index([0, ell])
    L = np.sqrt(ell*(ell+1))
    U = np.loadtxt(f"/scratch/g.samarth/Solar_Eigen_function/eig_files/U{idx}.dat")
    V = np.loadtxt(f"/scratch/g.samarth/Solar_Eigen_function/eig_files/V{idx}.dat")
    N[il] = simps((U*U + L*L*V*V)*rho*r*r, x=r)

a1 /= N
a1th = 430*(1 - 1/ells - 1/2/ells/ells)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs.flatten()[0].plot(ells, a1, '+k', label='Eigenvalue problem')
axs.flatten()[0].plot(ells, a1th, 'r', label='scaling relation', alpha=0.5)
axs.flatten()[0].set_xlabel("$\ell$")
axs.flatten()[0].set_ylabel("$a_1$ in nHz")
axs.flatten()[0].legend()

axs.flatten()[1].plot(ells, a1/a1th, '+k')
axs.flatten()[1].set_xlabel("$\ell$")
axs.flatten()[1].set_ylabel("$a_1$ ratio")
fig.tight_layout()
fig.savefig("/scratch/g.samarth/qdPy/plots/a1scaling.pdf")

