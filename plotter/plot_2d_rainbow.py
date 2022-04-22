import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order", type=int, default=0)
ARGS = parser.parse_args()

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
    diff_ratio = (a1 - a2)/sig
    return diff_ratio


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
    if n0 == ARGS.n0:
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
max_even = abs(colors_even).max()
max_odd = abs(colors_odd).max()

cmapval = "seismic"
fig1, ax1 = plt.subplots(figsize=(6, 4))
im = ax1.imshow(colors_even, cmap=cmapval, vmin=-max_even, vmax=max_even,
                aspect=colors_even.shape[1]/colors_even.shape[0], 
                extent=[2, 36, lvals[-1], lvals[0]])
ax1.set_title('Even a-coefficients')
plt.colorbar(im, ax=ax1)


fig1, ax1 = plt.subplots(figsize=(6, 4))
im = ax1.imshow(colors_odd, cmap=cmapval, vmin=-max_even/20., vmax=max_even/20.,
                aspect=colors_odd.shape[1]/colors_odd.shape[0], 
                extent=[1, 35, lvals[-1], lvals[0]])
ax1.set_title('Odd a-coefficients')
plt.colorbar(im, ax=ax1)

