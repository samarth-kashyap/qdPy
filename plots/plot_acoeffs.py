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
fig2, axs2 = plt.subplots(ncols=2, nrows=int(imax), figsize=(7, int(3*imax)))

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

    dirnamenew = "new_freqs_w135"

    for ell in l_arr:
        # print(f"Computing a-coeffs for n = {n}, ell = {ell}")
        count = 0
        try:
            fqdpt = np.load(f"{datadir}/{dirnamenew}/qdpt_{n:02d}_{ell:03d}.npy")
            fdpt = np.load(f"{datadir}/{dirnamenew}/dpt_{n:02d}_{ell:03d}.npy")
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


def plot_acoeff_percenterror(ac_qdpt, ac_dpt, larr):
    if len(ac_qdpt) == 0:
        return fig2
    maskn = data[:, 1] == n
    elldata = data[:, 0][maskn]
    splitdata = data[maskn, 12:12+imax]
    sigdata = data[maskn, 48:48+imax]
    for i in range(imax):
        axs2[i, 0].plot(larr, ac_qdpt[:, i], 'r', markersize=0.7,
                       label='QDPT')
        axs2[i, 0].plot(larr, ac_dpt[:, i], 'b', markersize=0.7,
                       label='DPT')
        ac_dpt[abs(ac_dpt[:, i])<1e-15, i] = 1e-15
        axs2[i, 0].set_title(f'a{i+1} in nHz')
        diff = (ac_qdpt[:, i] - ac_dpt[:, i])
        perr = (diff)*100/ac_dpt[:, i]
        perr_data = sigdata[:, i]/splitdata[:, i]*100
        diff_data = sigdata[:, i]
        # if i%2 == 0:
        #     axs2[i, 1].plot(larr, perr, 'k', markersize=0.8, label='error qdpt vs dpt')
        #     axs2[i, 1].plot(elldata, perr_data, '+k', markersize=0.8, label='uncertainty in HMI data')
        #     axs2[i, 1].set_title(f'% Error in a{i+1}')
        # else:
        axs2[i, 1].plot(larr, diff, 'k', markersize=0.8, label='error qdpt vs dpt')
        # axs2[i, 1].plot(elldata, diff_data, '+k', markersize=0.8, label='uncertainty in HMI data')
        axs2[i, 1].fill_between(elldata, -diff_data, diff_data, alpha=0.5, label='uncertainty in HMI data')
        axs2[i, 1].set_title(f'Error in a{i+1} in nHz')
        axs2[i, 0].set_xlabel('$\\ell$', fontsize=18)
        axs2[i, 1].set_xlabel('$\\ell$', fontsize=18)
        # axs2.flatten()[i].set_xlim([2, 100])
    return fig2



for n in range(2):
    ac_qdpt, ac_dpt, larr = get_acoeffs_list(n, imax)
    # fig = plot_acoeff_error(ac_qdpt, ac_dpt, larr)
    fig2 = plot_acoeff_percenterror(ac_qdpt, ac_dpt, larr)
# handles, labels = axs.flatten()[0].get_legend_handles_labels()
handles2, labels2 = axs2.flatten()[0].get_legend_handles_labels()
# fig.legend(handles[:3], labels[:3], loc='upper center')
# fig.tight_layout()
# fig.subplots_adjust(top=0.85)
fig2.legend(handles2[:2], labels2[:2], loc='upper center')
fig2.tight_layout()
fig2.subplots_adjust(top=0.95)
#fig.show()
# fig.savefig('/scratch/g.samarth/plots/acoeffs.pdf')
fig2.savefig('/scratch/g.samarth/qdPy/plots/acoeffs_error.pdf')
