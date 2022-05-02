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

# multiplet frequencies in muHz from globalvars
nl_arr_modelS = np.loadtxt(f'{snrnmais_dir}/data_files/nl.dat') 
muhz_modelS = np.loadtxt(f'{snrnmais_dir}/data_files/muhz.dat')

# functions defined in Vorontsov 2011
def a2_V11(domega_dell):
    return (-1.2e-5 * domega_dell)

def a4_V11(domega_dell):
    return (1.47e-6 * domega_dell)

# function to compute the centered omega derivative
def get_domega_dell(n_arr, l_arr):
    domega_dell_arr = np.zeros(len(n_arr))
    
    for mult_ind in range(len(n_arr)):
        n0, ell0 = n_arr[mult_ind], l_arr[mult_ind]
        
        mult_ind_p1 = np.where((nl_arr_modelS[:,0] == n0) * (nl_arr_modelS[:,1] == ell0+1))
        mult_ind_m1 = np.where((nl_arr_modelS[:,0] == n0) * (nl_arr_modelS[:,1] == ell0-1))
        
        # frequencies of ell+1 and ell-1 in muHz
        omega_p1 = muhz_modelS[mult_ind_p1]
        omega_m1 = muhz_modelS[mult_ind_m1]
        
        # taking the centered difference
        domega_dell = (omega_p1 - omega_m1)/2.
        
        domega_dell_arr[mult_ind] = domega_dell
        
    return domega_dell_arr

ac_dpt = []
ac_qdpt = []
ac_sig = []
nu_vals = []

nvals = []
lvals = []
omegavals = []
a2_hmi = []
a4_hmi = []
a2_hmi_sigma = []
a4_hmi_sigma = []

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
        
        # obtaining the measured a2 and a4 in nHz from HMI data file
        a2_hmi.append(hmidata[mode_idx, 13])
        a4_hmi.append(hmidata[mode_idx, 15])
        a2_hmi_sigma.append(hmidata[mode_idx, 49])
        a4_hmi_sigma.append(hmidata[mode_idx, 51])


# converting to array-like
a2_hmi = np.asarray(a2_hmi)
a4_hmi = np.asarray(a4_hmi)
a2_hmi_sigma = np.asarray(a2_hmi_sigma)
a4_hmi_sigma = np.asarray(a4_hmi_sigma)
ac_dpt = np.asarray(ac_dpt)
ac_qdpt = np.asarray(ac_qdpt)
nvals = np.asarray(nvals)
lvals = np.asarray(lvals)
           
# getting the domega_dell in nhz
domega_dell_arr = get_domega_dell(nvals, lvals) * 1e3

a2_cen = a2_V11(domega_dell_arr)
a4_cen = a4_V11(domega_dell_arr)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

ax[0].plot(lvals, a2_cen, '--k', label='Centrifugal effect as in V11')
ax[0].plot(lvals, a2_hmi, 'ok', label='HMI measurement')
ax[0].plot(lvals, ac_qdpt[:, 2], 'k', label='Mode coupling')
ax[0].plot(lvals, ac_qdpt[:, 2] + a2_cen, 'r', label='Mode coupling + centrifugal effect')
ax[0].set_ylabel('$a_2$ in nHz', fontsize=16)
ax[0].set_xlabel('$\ell$', fontsize=16)
ax[0].set_ylim([-0.3, 0.3])
ax[0].set_xlim([np.min(lvals)-2, np.max(lvals)+2])
ax[0].grid(True, alpha=0.5)
# ax[0].text(280, 0.25, '(a)')
ax[0].legend()

ax[1].plot(lvals, a4_cen, '--k', label='Centrifugal effect as in V11')
ax[1].plot(lvals, a4_hmi, 'ok', label='HMI measurement')
ax[1].plot(lvals, ac_qdpt[:, 4], 'k', label='Mode coupling')
ax[1].plot(lvals, ac_qdpt[:, 4] + a4_cen, 'r', label='Mode coupling + centrifugal effect')
ax[1].set_ylabel('$a_4$ in nHz', fontsize=16)
ax[1].set_xlabel('$\ell$', fontsize=16)
ax[1].set_ylim([-0.3, 0.3])
ax[1].set_xlim([np.min(lvals)-2, np.max(lvals)+2])
ax[1].grid(True, alpha=0.5)
# ax[1].text(280, 0.25, '(b)')
ax[1].legend()

plt.tight_layout()
plt.savefig('compare_evenacoeff_V11.pdf')
