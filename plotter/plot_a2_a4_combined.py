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

# other miscellaneous params                                                                 
smax = 11
fwindow = 150
daynum = 7768
instr = 'hmi'

jmax = 19

# reading the mode-set from the 360d HMI data file                                           
hmidata = np.loadtxt(f'{snrnmais_dir}/data_files/hmi.6328.36')
nl_arr = hmidata[:, :2].astype('int')

# excluding modes which are ell > 300 - smax                                                  
mask_ell = nl_arr[:, 0] < (300 - smax)
nl_arr = nl_arr[mask_ell]

num_modes = nl_arr.shape[0]
nlist, llist = nl_arr[:, 1], nl_arr[:, 0]

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
colors_even = np.zeros((1,36))
colors_odd = np.zeros((1,36))

# looping over all multiplets                                                                 
for i in range(num_modes):
    # extracting mode parameters                                                              
    n0, l0 = nlist[i], llist[i]
    if(n0 == ARGS.n0):
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
        
        colors_even_thismult = np.zeros((1,36))
        colors_odd_thismult = np.zeros((1,36))
        
        # the color arrays for counting the number of a-coefficients that are sig-diff        
        colors_even_thismult[0,1::2] = get_distance(_acd[0::2][1:], _acq[0::2][1:],
                                                  _acsig[1::2])
        colors_odd_thismult[0,::2] = get_distance(_acd[1::2], _acq[1::2],
                                                _acsig[0::2])
        
        colors_even = np.append(colors_even, colors_even_thismult, axis=0)
        colors_odd = np.append(colors_odd, colors_odd_thismult, axis=0)
   
# rejecting the first mult which is all zero and keeping until jmax
colors_even = colors_even[1:, :jmax]
colors_odd = colors_odd[1:, :jmax]

colors_even = np.array(colors_even)
colors_odd = np.array(colors_odd)

plt.figure()

plt.plot(lvals, colors_even[:, 1], label='$s = 2$')
plt.plot(lvals, colors_even[:, 3], label='$s = 4$')
plt.fill_between(lvals, np.ones_like(lvals), -np.ones_like(lvals),
                 hatch = '/', facecolor='w')
plt.xlabel('$\ell$', fontsize=16)
plt.ylabel('$\\frac{a_s^{\mathrm{QDPT}} - a_s^{\mathrm{DPT}}}{\sigma(a_s)}$', fontsize=16)
plt.xlim([np.min(lvals), np.max(lvals)])
plt.grid(True, alpha=0.5)
plt.legend()
plt.savefig('a2_a4_combined.pdf')
