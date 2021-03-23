import matplotlib.pyplot as plt
import numpy as np

datadir = "/home/g.samarth/qdPy/postprocess"
qdpt = np.loadtxt(f'{datadir}/qdpt_const_full.splits')
dpt = np.loadtxt(f'{datadir}/dpt_const_full.splits')

qdpt_half = np.loadtxt(f'{datadir}/qdpt_const_half.splits')
dpt_half = np.loadtxt(f'{datadir}/dpt_const_half.splits')

enn = qdpt[:, 0]
ell = qdpt[:, 1]

mask_n0 = enn == 0
mask_n1 = enn == 1

qdpt_splits = qdpt[:, 3]
dpt_splits = dpt[:, 3]

qdpt_half_splits = qdpt_half[:, 3]
dpt_half_splits = dpt_half[:, 3]

rat = (qdpt_half_splits - dpt_half_splits)/(qdpt_splits - dpt_splits)
rat_qdpt = qdpt_half_splits / qdpt_splits
rat_dpt = dpt_half_splits / dpt_splits

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
fig.suptitle('a1 ratio = ($\\frac{a1^{QDPT}_{half} - a1^{DPT}_{half}}{a1^{QDPT}_{full} - a1^{DPT}_{full}}$)')
axs[0].set_title('n=0')
axs[0].set_xlabel('$\ell$')
axs[0].set_ylabel('a1 ratio')
axs[0].plot(ell[mask_n0], rat[mask_n0], '+k')

axs[1].set_title('n=1')
axs[1].set_xlabel('$\ell$')
axs[1].set_ylabel('a1 ratio')
axs[1].plot(ell[mask_n1], rat[mask_n1], '+k')
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig('a1_err_ratio.pdf')


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
fig.suptitle('a1 ratio = ($\\frac{a1_{half}}{a1_{full}}$)')
axs[0].set_title('n=0')
axs[0].set_xlabel('$\ell$')
axs[0].set_ylabel('a1 ratio')
axs[0].plot(ell[mask_n0], rat_qdpt[mask_n0], '+k', label='QDPT')
axs[0].plot(ell[mask_n0], rat_dpt[mask_n0], '+b', label='DPT')
axs[0].legend()

axs[1].set_title('n=1')
axs[1].set_xlabel('$\ell$')
axs[1].set_ylabel('a1 ratio')
axs[1].plot(ell[mask_n1], rat_qdpt[mask_n1], '+k', label='QDPT')
axs[1].plot(ell[mask_n1], rat_dpt[mask_n1], '+b', label='DPT')
axs[1].legend()
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig('a1_ratio.pdf')
