import os
import sys
from matplotlib.lines import Line2D
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# reading the current and scratch directory from .config
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
snrnmais_dir = dirnames[2]

# reading the mode-set from the 360d HMI data file
hmidata = np.loadtxt(f'{snrnmais_dir}/data_files/hmi.6328.36')

# other miscellaneous params
smax = 11
fwindow = 150
daynum = 7768
instr = 'hmi'

# the multiplets
nl_arr_hmi = hmidata[:, :2].astype('int')

# excluding modes which are ell > 300 - smax
mask_ell = nl_arr_hmi[:, 0] < (300 - smax)
nl_arr_hmi = nl_arr_hmi[mask_ell]

# arranging nl_arr as radial order (with ell in each ascending)
nl_arr = nl_arr_hmi[nl_arr_hmi[:,1] == 0]

for n in range(1, np.max(nl_arr_hmi[:,1])+1):
    mask = nl_arr_hmi[:,1] == n
    nl_arr = np.append(nl_arr, nl_arr_hmi[mask], axis=0)

num_modes = nl_arr.shape[0]
nlist, llist = nl_arr[:, 1], nl_arr[:, 0]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size' : 18}  

# to calculate how many a-coefficients are significantly different
def get_distance(a1, a2, sig):
    diff_ratio = abs(a1 - a2)/sig
    mask1 = diff_ratio > 1.0 # 1.0/np.sqrt(5)
    return mask1.sum()

# arrays for plotting
ac_dpt = []
ac_qdpt = []
ac_sig = []
nu_vals = []

nvals = []
lvals = []
omegavals = []
colors_even = []
colors_odd = []

# looping over all multiplets
for i in range(num_modes):
    # extracting mode parameters
    n0, l0 = nlist[i], llist[i]
    omega_idx = np.where((hmidata[:,0] == l0) * (hmidata[:,1] == n0))
    omega = hmidata[omega_idx, 2]
    mode_idx = np.where((hmidata[:, 0]==l0) * (hmidata[:, 1]==n0))[0][0]
    _acsig = hmidata[mode_idx, 48:84]
    
    # checking if the files exist
    try:
        dpt_file = f"{scratch_dir}/output_files/hmi_allmodes/"+\
                   f"dpt-ac-{n0:02d}.{l0:03d}.{smax}.{fwindow}.{instr}.{daynum}.npy"
        qdpt_file = f"{scratch_dir}/output_files/hmi_allmodes/"+\
                    f"qdpt-ac-{n0:02d}.{l0:03d}.{smax}.{fwindow}.{instr}.{daynum}.npy"
        
        _acd = np.load(dpt_file)
        _acq = np.load(qdpt_file)
                       
    except FileNotFoundError:
        continue
        
    # appending in the arrays to be used for plotting
    ac_dpt.append(_acd)
    ac_qdpt.append(_acq)
    ac_sig.append(_acsig)
    nvals.append(n0)
    lvals.append(l0)
    omegavals.append(omega)
    
    # the color arrays for counting the number of a-coefficients that are sig-diff
    colors_even.append(get_distance(_acd[0::2][1:], _acq[0::2][1:], _acsig[1::2]))
    colors_odd.append(get_distance(_acd[1::2], _acq[1::2], _acsig[0::2]))

# converting the lists to array-like
colors_even = np.asarray(colors_even)
colors_odd = np.asarray(colors_odd)
lvals = np.asarray(lvals)
omegavals = np.asarray(omegavals)/1e3  # converting to mHz
max_even = max(colors_even)
max_odd = max(colors_odd)

# choosing every second data point for better plotting
colors_even = colors_even[::2]
colors_odd = colors_odd[::2]
lvals = lvals[::2]
omegavals = omegavals[::2]

# filtering out the ones with zero a-coeffs which are sig-diff
colors_even_zero = np.where(np.abs(colors_even) < 0.5)[0]
colors_even_one = np.where((np.abs(colors_even) > 0.5) * (np.abs(colors_even) < 1.5))[0]
colors_even_two = np.where(np.abs(colors_even) > 1.5)[0]

fig1, ax1 = plt.subplots(figsize=(10, 5))
im1 = ax1.scatter(lvals[colors_even_zero], omegavals[colors_even_zero], s=5, marker='x',
                  c='k', linewidth=0.5)

im2 = ax1.scatter(lvals[colors_even_one], omegavals[colors_even_one], s=5, marker='^',
                  c='b', linewidth=0.5)

im3 = ax1.scatter(lvals[colors_even_two], omegavals[colors_even_two], s=5, marker='o',
                  c='r', linewidth=0.5)

legend_elements = [Line2D([0], [0], marker='x', color='k', lw = 0,
                        label='Number of significantly different even $a$-coefficients =  0',
                          markerfacecolor='k', markersize=12),
                   Line2D([0], [0], marker='^', color='w', lw = 0,
                        label='Number of significantly different even $a$-coefficients =  1',
                          markerfacecolor='b', markersize=12),
                   Line2D([0], [0], marker='o', color='w', lw = 0,
                        label='Number of significantly different even $a$-coefficients =  2',
                          markerfacecolor='r', markersize=12)]

ax1.set_xlim([-5, 290])
ax1.legend(handles=legend_elements)
ax1.set_xlabel('$\ell$', fontsize=16)
ax1.set_ylabel('Frequency in mHz', fontsize=16)
plt.tight_layout()
plt.savefig('even_acoeffs.pdf')
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 5))
im = ax2.scatter(lvals, omegavals, s=5,
                 c=colors_odd, linewidth=0.5,
                 cmap='jet', vmin=0, vmax=max_even)
ax2.set_title('Odd a-coefficients')
plt.colorbar(im, ax=ax2)

plt.savefig('odd_acoeffs.pdf')
plt.close()
