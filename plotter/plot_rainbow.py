import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
snrnmais_dir = dirnames[2]

nlist = np.loadtxt(f'{scratch_dir}/data_files/nlist.txt').astype('int')
llist = np.loadtxt(f'{scratch_dir}/data_files/llist.txt').astype('int')
omegalist = np.loadtxt(f'{scratch_dir}/data_files/omegalist.txt')
hmidata = np.loadtxt(f'{scratch_dir}/data_files/hmi.6328.36')
num_modes = len(nlist)
smax = 7
fwindow = 150

font = {'family' : 'normal',
        'weight' : 'normal',
        'size' : 18}  


def get_distance(a1, a2, sig):
    diff_ratio = abs(a1 - a2)/sig
    mask1 = diff_ratio > 1.0/np.sqrt(5)
    return mask1.sum()


ac_dpt = []
ac_qdpt = []
ac_sig = []
nu_vals = []

nvals = []
lvals = []
omegavals = []
colors_even = []
colors_odd = []

for i in range(num_modes):
    n0, l0 = nlist[i], llist[i]
    omega = omegalist[i]
    mode_idx = np.where((hmidata[:, 0]==l0) * (hmidata[:, 1]==n0))[0][0]
    _acsig = hmidata[mode_idx, 48:84]
    try:
        _acd = np.load(f"{scratch_dir}/output_files/mdi/dpt-ac-{n0:02d}.{l0:03d}.{smax}.{fwindow}.npy")
        _acq = np.load(f"{scratch_dir}/output_files/mdi/qdpt-ac-{n0:02d}.{l0:03d}.{smax}.{fwindow}.npy")
    except FileNotFoundError:
        continue
    ac_dpt.append(_acd)
    ac_qdpt.append(_acq)
    ac_sig.append(_acsig)
    nvals.append(n0)
    lvals.append(l0)
    omegavals.append(omega)

    colors_even.append(get_distance(_acd[0::2][1:], _acq[0::2][1:], _acsig[1::2]))
    colors_odd.append(get_distance(_acd[1::2], _acq[1::2], _acsig[0::2]))

   
colors_even = np.array(colors_even)
colors_odd = np.array(colors_odd)
max_even = max(colors_even)
max_odd = max(colors_odd)

fig1, ax1 = plt.subplots(figsize=(10, 5))
im = ax1.scatter(lvals, omegavals, s=5, #size,
                 c=colors_even, linewidth=0.5, #edgecolor='k',
                 # cmap='binary', vmin=vmin, vmax=vmax, alpha = 1.0)
                 cmap='jet', vmin=0, vmax=max_even)
ax1.set_title('Even a-coefficients')
plt.colorbar(im, ax=ax1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
im = ax2.scatter(lvals, omegavals, s=5, #size,
                 c=colors_odd, linewidth=0.5, #edgecolor='k',
                 # cmap='binary', vmin=vmin, vmax=vmax, alpha = 1.0)
                 cmap='jet', vmin=0, vmax=max_even)
ax2.set_title('Odd a-coefficients')
plt.colorbar(im, ax=ax2)

