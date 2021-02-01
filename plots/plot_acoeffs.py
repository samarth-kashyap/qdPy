import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/g.samarth/qdPy')
import ritzlavely as RL

jmax = 36
imax = 6
datadir = '/scratch/g.samarth/qdPy'
data = np.loadtxt('/home/g.samarth/Woodard2013/WoodardPy/HMI/hmi.6328.36')
fig, axs = plt.subplots(ncols=2, nrows=int(imax//2), figsize=(7, int(7*imax//4)))

def get_acoeffs_list(n, imax):
    ac_n_qdpt = []
    ac_n_dpt = []
    l_arr = np.array([])
    lmax = 290
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
    mask_ell = l_arr < lmax
    l_arr = l_arr[mask_ell]
    print(f"l_arr = {l_arr}")
    if len(l_arr) == 0:
        return l_arr, l_arr, l_arr

    for ell in l_arr:
        # print(f"Computing a-coeffs for n = {n}, ell = {ell}")
        count = 0
        try:
            fqdpt = np.load(f"{datadir}/new_freqs/qdpt_{n:02d}_{ell:03d}.npy")
            fdpt = np.load(f"{datadir}/new_freqs/dpt_{n:02d}_{ell:03d}.npy")
        except FileNotFoundError:
            count += 1
            pass

        rlp = RL.ritzLavelyPoly(ell, jmax)
        rlp.get_Pjl()

        # in nHz
        ac_ell_qdpt = rlp.get_coeffs(fqdpt)*1000.
        ac_ell_dpt = rlp.get_coeffs(fdpt)*1000.

        nu = fqdpt[ell]

        ac_n_qdpt.append(ac_ell_qdpt[1:imax+1])
        ac_n_dpt.append(ac_ell_dpt[1:imax+1])
    print(f"Not found count = {count}")
    # return np.array(ac_n_qdpt), np.array(ac_n_dpt), nu/l_arr
    return np.array(ac_n_qdpt), np.array(ac_n_dpt), l_arr


def plot_acoeff_error(ac_qdpt, ac_dpt, larr):
    if len(ac_qdpt) == 0:
        return fig
    maskn = data[:, 1] == n
    elldata = data[:, 0][maskn]
    splitdata = data[maskn, 12:12+imax]
    for i in range(imax):
        axs.flatten()[i].plot(larr, ac_qdpt[:, i], 'r', markersize=0.7,
                              label='QDPT')
        axs.flatten()[i].plot(larr, ac_dpt[:, i], 'b', markersize=0.7,
                              label='DPT')
        axs.flatten()[i].plot(elldata, splitdata[:, i], '+k', markersize=0.8,
                              label='HMI Pipeline')
        axs.flatten()[i].set_title(f'a{i+1} in nHz')
        axs.flatten()[i].set_xlabel('$\\ell$', fontsize=18)
        # axs.flatten()[i].set_xlim([2, 100])
    return fig


for n in range(7):
    ac_qdpt, ac_dpt, larr = get_acoeffs_list(n, 6)
    fig = plot_acoeff_error(ac_qdpt, ac_dpt, larr)
handles, labels = axs.flatten()[0].get_legend_handles_labels()
fig.legend(handles[:3], labels[:3], loc='upper center')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.show()
