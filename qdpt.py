import numpy as np
import matplotlib.pyplot as plt
import qdclasses as qdcls
import ritzlavely as RL
import globalvars
import logging
import argparse

# {{{ Logging module
# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')
# }}} logging


# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order", type=int)
parser.add_argument("--l0", help="angular degree", type=int)
parser.add_argument("--read", help="read submatrix from file",
                    action="store_true")
args = parser.parse_args()
# }}} argparse


# {{{ Reading global variables
rmin, rmax = 0.0, 1.0
# (Since we are using lmax = 300, 0.45*300 \approx 150)
SMAX = 5      # maximum s for constructing supermatrix
FWINDOW = 150   # microHz
gvar = globalvars.globalVars(rmin, rmax, SMAX, FWINDOW, args)
# }}} global variables


def get_cenmode_freqs_qdpt(eigvals, eigvecs):
    cenmode_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(cenmode_idx)
    sm = qdcls.subMatrix(cenblock_idx, cenblock_idx, super_matrix)
    block_start = sm.startx
    block_end = sm.endx

    eigvals_sorted = np.zeros_like(eigvals)
    eigvecs_sorted = np.zeros_like(eigvecs)

    for i in range(len(eigvals)):
        eidx = abs(eigvecs[:, i]).argmax()
        eigvecs_sorted[eidx] = eigvecs[:, i]
        eigvals_sorted[eidx] = eigvals[i]

    return eigvals_sorted[block_start:block_end], eigvecs_sorted
    # return eigvals_sorted, eigvecs_sorted


def get_cenmode_freqs_dpt(eigvals):
    cenmode_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(cenmode_idx)
    sm = qdcls.subMatrix(cenblock_idx, cenblock_idx, super_matrix)
    block_start = sm.startx
    block_end = sm.endx
    return eigvals[block_start:block_end]
    # return eigvals


def get_RL_coeffs(rlpObj, omega_nlm):
    # delta_omega_nlm = omega_nlm - analysis_modes.omega0*gvar.OM*1e6
    delta_omega_nlm = omega_nlm# - analysis_modes.omega0*gvar.OM*1e6
    rlpObj.get_Pjl()
    acoeffs = rlpObj.get_coeffs(delta_omega_nlm)
    acoeffs[0] = analysis_modes.omega0*gvar.OM*1e6
    return acoeffs


def store_offset():
    domega_QDPT = np.linalg.norm(fqdpt - analysis_modes.omega0*gvar.OM*1e6)
    domega_DPT = np.linalg.norm(fdpt - analysis_modes.omega0*gvar.OM*1e6)
    rel_offset_percent = np.abs((domega_QDPT-domega_DPT)/domega_DPT) * 100.0
    np.savetxt(f"{gvar.datadir}/qdpt_error_full/offsets_{args.n0:02d}_{args.l0:03d}.dat",
            np.array([rel_offset_percent]))
    return 0



analysis_modes = qdcls.qdptMode(gvar)
super_matrix = analysis_modes.create_supermatrix()
np.save('/scratch/g.samarth/qdPy/supermat_00_200.npy', super_matrix.supmat)
eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
eigvals_qdpt_unsorted, eigvecs_qdpt = super_matrix.get_eigvals(type='QDPT', sorted=False)

eigvals_cenmode_dpt = get_cenmode_freqs_dpt(eigvals_dpt_unsorted)
eigvals_cenmode_qdpt, eigvecs_qdpt = get_cenmode_freqs_qdpt(eigvals_qdpt_unsorted,
                                              eigvecs_qdpt)

fdpt = (analysis_modes.omega0 + eigvals_cenmode_dpt/2/analysis_modes.omega0)
fqdpt = (analysis_modes.omega0 + eigvals_cenmode_qdpt/2/analysis_modes.omega0)
# fdpt = np.sqrt(analysis_modes.omega0**2 + eigvals_cenmode_dpt)
# fqdpt = np.sqrt(analysis_modes.omega0**2 + eigvals_cenmode_qdpt)

# converting to muHz
fdpt *= gvar.OM * 1e6
fqdpt *= gvar.OM * 1e6

# dirname = "new_freqs_half"
dirnamenew = "new_freqs_jesper"

np.save(f'{gvar.datadir}/{dirnamenew}/qdpt_{args.n0:02d}_{args.l0:03d}.npy', fqdpt)
np.save(f'{gvar.datadir}/{dirnamenew}/dpt_{args.n0:02d}_{args.l0:03d}.npy', fdpt)
# store_offset()

ritz_degree = min(args.l0//2+1, 36)
rlp = RL.ritzLavelyPoly(args.l0, ritz_degree)

# acoeffs_qdpt = get_RL_coeffs(rlp, fqdpt[::-1])*1e3
# acoeffs_dpt = get_RL_coeffs(rlp, fdpt[::-1])*1e3
acoeffs_qdpt = get_RL_coeffs(rlp, fqdpt)*1e3
acoeffs_dpt = get_RL_coeffs(rlp, fdpt)*1e3

acoeffs_qdpt[0] /= 1e3
acoeffs_dpt[0] /= 1e3

acoeffs_qdpt = np.pad(acoeffs_qdpt, (0, 36-ritz_degree), mode='constant')
acoeffs_dpt = np.pad(acoeffs_dpt, (0, 36-ritz_degree), mode='constant')

print(f"QDPT a-coeffs = {acoeffs_qdpt}")
print(f" DPT a-coeffs = {acoeffs_dpt}")
np.save(f'{gvar.datadir}/{dirnamenew}/qdpt_acoeffs_{args.n0:02d}_{args.l0:03d}.npy',
        acoeffs_qdpt)
np.save(f'{gvar.datadir}/{dirnamenew}/dpt_acoeffs_{args.n0:02d}_{args.l0:03d}.npy',
        acoeffs_dpt)

# plt.plot(fqdpt, 'r')
# plt.plot(fdpt, 'b')
