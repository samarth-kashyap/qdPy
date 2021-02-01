import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

import sys
sys.path.append('/home/g.samarth/Solar_Eigen_function/')
import functions as fn


datadir = "/scratch/g.samarth/qdPy"
OM = np.loadtxt(f'{datadir}/OM.dat') #importing normalising frequency value from file (in Hz (cgs))
omega_list = np.loadtxt(f'{datadir}/muhz.dat') * 1e-6 / OM #normlaised frequency list

font = {'family' : 'normal',
'weight' : 'normal',
'size' : 18}  

# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--nmax", help="max radial order", type=int)
args = parser.parse_args()
# }}} argparse

nmax = int(args.nmax)

def collect_multiplets(n):
    gdns_arr = np.array([])
    n_list = np.array([])
    l_list = np.array([])
    l_arr = np.array([])

    l0_arr = np.load(f"/scratch/g.samarth/csfit/lall_radleak.npy")
    n0_arr = np.load(f"/scratch/g.samarth/csfit/nall_radleak.npy")
    l0_arr = l0_arr[n0_arr==n]
    l_arr = np.append(l_arr, l0_arr)
    # print(l0_arr)
    try:
        l1_arr = np.load(f"/scratch/g.samarth/csfit/l{n:02d}_used.npy")
        l2_arr = np.load(f"/scratch/g.samarth/csfit/l{n:02d}_unused.npy")
        l_arr = np.append(l_arr, l1_arr)
        l_arr = np.append(l_arr, l2_arr)
        # print(l1_arr)
        # print(l2_arr)
    except FileNotFoundError:
        pass
    l_arr = np.unique(l_arr)
    l_arr = l_arr.astype('int64')
    
    for ell in l_arr:
        try:
            # gdns = np.loadtxt(f"{datadir}/Cross_coupling_goodness/" +
            gdns = np.loadtxt(f"{datadir}/qdpt_error/" +
                            f"offsets_{n:02d}_{ell:03d}.dat")
            # print(f" n = {n}, ell = {ell}, rop = {gdns}")
        except OSError:
            continue
        if np.isinf(gdns) or np.isnan(gdns):
            gdns = 0.0
        gdns_arr = np.append(gdns_arr, gdns)
        l_list = np.append(l_list, ell)
        n_list = np.append(n_list, n)
    nl_list = np.zeros((len(l_list), 2), dtype=int)
    nl_list[:, 0] = n_list
    nl_list[:, 1] = l_list

    assert len(gdns_arr) == nl_list.shape[0], "Size mismatch: gdns = {len(gdns_arr)}; nl = {nl_list.shape[0]}"
    return gdns_arr, nl_list

# qdpt_contrib_rel = np.ma.masked_greater(qdpt_contrib_rel, 90) 
# qdpt_contrib_rel = np.ma.masked_invalid(qdpt_contrib_rel) 
vmax = 0.0
vmin = 100000000000

print(f"Computing vmin, vmax...")
for n0 in range(nmax+1):
    print(f"-- n = {n0}")
    qdpt_contrib_rel, nl_list = collect_multiplets(n0)
    qdpt_contrib_rel = np.ma.masked_greater(qdpt_contrib_rel,90) 
    colors = qdpt_contrib_rel
    size = qdpt_contrib_rel/0.01 * 500
    #Setting the minimum limit to 1.0 => Correspond to 5% or less change
    size[size<10.0] = 10.0    
    vmin = min(np.amin(colors), vmin)  #To make the smallest dots look gray
    vmax = max(np.amax(colors), vmax)

lmax = 0
# vmin *= 10
print(f"Plotting ...")
fig, ax = plt.subplots(1, figsize=(10, 5), dpi=200, facecolor='w', edgecolor='k')
for n0 in range(nmax+1):
    print(f"-- Appending n = {n0}")
    qdpt_contrib_rel, nl_list = collect_multiplets(n0)
    qdpt_contrib_rel = np.ma.masked_greater(qdpt_contrib_rel,90) 
    colors = np.log10(qdpt_contrib_rel)
    colors_norm = (colors - np.log10(vmin))/(np.log10(vmax) - np.log10(vmin))
    size = qdpt_contrib_rel/0.01 * 10

    lmax = max(nl_list[:, 1].max(), lmax)
    #Setting the minimum limit to 1.0 => Correspond to 5% or less change
    size[size<10.0] = 10.0    
    nl_list = nl_list[nl_list[:,0]==n0]
    nl_list = nl_list.astype('int64')
    l = nl_list[:,1]    #isolating the \ell's

    omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list]) #important to have nl_list as integer type

    im = ax.scatter(l,omega_nl*OM*1e6/1e3, s=5, #size,
                    c=colors_norm, linewidth=0.5, #edgecolor='k',
                    # cmap='binary', vmin=vmin, vmax=vmax, alpha = 1.0)
                    cmap='jet', vmin=0, vmax=1) #np.log10(vmin), vmax=np.log10(vmax), alpha = 1.0)

# plt.colorbar(fig)

ax.set_xlim([-10, lmax+10])
ax.set_ylim([1, 5])

# making colorbar
# axins = inset_axes(ax,
#                    width="3%", # width = 5% of parent_bbox width
#                    height="100%", # height : 50%
#                    loc='lower right',
#                    bbox_to_anchor=(0.06, 0., 1, 1),
#                    bbox_transform=ax.transAxes,
#                    borderpad=0,
#                    )
cbar = fig.colorbar(im, ticks=np.linspace(0, 1, 5)) #cax=axins
cbar_vals = 10**(np.linspace(np.log10(vmin), np.log10(vmax), 5))
cbar_vals_str = np.array([]) 
for val in cbar_vals:
    cbar_vals_str = np.append(cbar_vals_str, f"{val:.1e}")
# cbar.ax.set_yticklabels([f'{-maxx:.0e}',f'{0:.0e}',f'{maxx:.0e}'])
cbar.ax.set_yticklabels(cbar_vals_str)
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title('%', fontsize=14)

print(f"vmax = {vmax}")


ax.set_title('$\\frac{L_2^{QDPT}-L_2^{DPT}}{L_2^{DPT}} \\times 100$%',\
                fontsize=20, pad = 14)
ax.set_xlabel('$\ell$', fontsize=18)
ax.set_ylabel('Unperturbed frequency $\\nu_0$ in mHz',fontsize=18)
ax.tick_params("both", labelsize=14)
fig.subplots_adjust(left=0.080, right=1.05, bottom=0.15, top=0.85)
fig.savefig(f'{datadir}/qdpt_error/DR_all.pdf')
plt.close(fig)

