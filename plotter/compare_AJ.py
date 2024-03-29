import numpy as np
import sys
import matplotlib.pyplot as plt
from qdPy import globalvars
import qdPy.ritzlavely as RL

GVAR = globalvars.globalVars()   # some arbitraty choice of n0, l0

jmax = 36
imax = 6
outdir = GVAR.outdir
data = np.loadtxt(f'{GVAR.datadir}/hmi.6328.36')
fig, axs = plt.subplots(ncols=2, nrows=int(imax//2), figsize=(7, int(7*imax//4)))
fig2, axs2 = plt.subplots(ncols=2, nrows=int(imax), figsize=(7, int(3*imax)))

def get_acoeffs_list(n, imax, dpt_or_qdpt='dpt'):
    ac_n_qdpt_antia = []
    ac_n_qdpt_jesper = []
    ac_n_dpt_antia = []
    ac_n_dpt_jesper = []
    l_arr = np.array([])
    lmax = 290
    try:
        # extracting the available modes from the hmi mode parameter file
        mask_n = data[:,1] == n
        l_arr = data[:,0][mask_n]
    except FileNotFoundError:
        pass
    l_arr = np.unique(l_arr)
    l_arr = l_arr.astype('int64')
    mask_ell = l_arr < lmax
    l_arr = l_arr[mask_ell]
    print(f"l_arr = {l_arr}")
    if len(l_arr) == 0:
        return l_arr, l_arr, l_arr

    dirname_antia = "w135_antia"
    dirname_jesper = "w135_jesper"

    larr = np.array([])

    for ell in l_arr:
        # print(f"Computing a-coeffs for n = {n}, ell = {ell}")
        file_found = True
        count = 0
        try:
            if dpt_or_qdpt == 'dpt':
                fdpt_antia = np.load(f"{outdir}/{dirname_antia}/dpt_opt_{n:02d}_{ell:03d}.npy")
                fdpt_jesper = np.load(f"{outdir}/{dirname_jesper}/dpt_opt_{n:02d}_{ell:03d}.npy")
            elif dpt_or_qdpt == 'qdpt':
                fqdpt_antia = np.load(f"{outdir}/{dirname_antia}/qdpt_opt_{n:02d}_{ell:03d}.npy")
                fqdpt_jesper = np.load(f"{outdir}/{dirname_jesper}/qdpt_opt_{n:02d}_{ell:03d}.npy")
        except FileNotFoundError:
            count += 1
            file_found = False
            pass

        if file_found:
            rlp = RL.ritzLavelyPoly(ell, jmax)
            rlp.get_Pjl()
            print(f"Computing a-coeffs for n = {n}, ell = {ell}")

            # in nHz
            if dpt_or_qdpt == 'dpt':
                ac_ell_dpt_antia = rlp.get_coeffs(fdpt_antia)*1000.
                ac_ell_dpt_jesper = rlp.get_coeffs(fdpt_jesper)*1000.
                ac_n_dpt_antia.append(ac_ell_dpt_antia[1:imax+1])
                ac_n_dpt_jesper.append(ac_ell_dpt_jesper[1:imax+1])
            elif dpt_or_qdpt == 'qdpt':
                ac_ell_qdpt_antia = rlp.get_coeffs(fqdpt_antia)*1000.
                ac_ell_qdpt_jesper = rlp.get_coeffs(fqdpt_jesper)*1000.
                ac_n_qdpt_antia.append(ac_ell_qdpt_antia[1:imax+1])
                ac_n_qdpt_jesper.append(ac_ell_qdpt_jesper[1:imax+1])

            larr = np.append(larr, np.array([ell]))

    print(f"Not found count = {count}")
    if dpt_or_qdpt == 'dpt':
        splits_both = (np.array(ac_n_dpt_antia), np.array(ac_n_dpt_jesper))
    elif dpt_or_qdpt == 'qdpt':
        splits_both = (np.array(ac_n_qdpt_antia), np.array(ac_n_qdpt_jesper))
    # return np.array(ac_n_qdpt), np.array(ac_n_dpt), nu/l_arr
    return splits_both, larr


def plot_acoeff_error(ac_qdpt, ac_dpt, larr):
    if len(ac_qdpt) == 0:
        return fig
    maskn = data[:, 1] == n
    elldata = data[:, 0][maskn]
    splitdata = data[maskn, 12:12+imax]
    for i in range(imax):
        axs.flatten()[i].plot(larr, ac_qdpt[:, i], 'or', markersize=0.7,
                              label='QDPT')
        axs.flatten()[i].plot(larr, ac_dpt[:, i], 'ob', markersize=0.7,
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
        axs2[i, 0].plot(larr, ac_qdpt[:, i], 'or', markersize=0.7,
                       label='Antia')
        axs2[i, 0].plot(larr, ac_dpt[:, i], 'ob', markersize=0.7,
                       label='JCD')
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
    splits, larr = get_acoeffs_list(n, imax, dpt_or_qdpt='qdpt')
    sA, sJ = splits
    # fig = plot_acoeff_error(ac_qdpt, ac_dpt, larr)
    fig2 = plot_acoeff_percenterror(sA, sJ, larr)
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
fig2.savefig(f'{GVAR.outdir}/plots/acoeffs_comparison_AJ.png')
