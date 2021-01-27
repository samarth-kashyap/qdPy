import numpy as np
import matplotlib.pyplot as plt
import qdclasses as qdcls
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



analysis_modes = qdcls.qdptMode(gvar)
super_matrix = analysis_modes.create_supermatrix()
# eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT')
# eigvals_qdpt_unsorted = super_matrix.get_eigvals(type='QDPT')
eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
eigvals_qdpt_unsorted, eigvecs_qdpt = super_matrix.get_eigvals(type='QDPT', sorted=False)

eigvals_cenmode_dpt = get_cenmode_freqs_dpt(eigvals_dpt_unsorted)
eigvals_cenmode_qdpt, eigvecs_qdpt = get_cenmode_freqs_qdpt(eigvals_qdpt_unsorted,
                                              eigvecs_qdpt)

# ??
fqdpt = np.sqrt(analysis_modes.omega0**2 + eigvals_cenmode_qdpt)
fdpt = (analysis_modes.omega0 + eigvals_cenmode_dpt/2/analysis_modes.omega0)
# fqdpt = np.sqrt(analysis_modes.omega0**2 + eigvals_qdpt_unsorted)
# fdpt = (analysis_modes.omega0 + eigvals_dpt_unsorted/2/analysis_modes.omega0)

# converting to muHz
fdpt *= gvar.OM * 1e6
fqdpt *= gvar.OM * 1e6

np.save(f'{gvar.datadir}/new_freqs/qdpt_{args.n0:02d}_{args.l0:03d}.npy', fqdpt)
np.save(f'{gvar.datadir}/new_freqs/dpt_{args.n0:02d}_{args.l0:03d}.npy', fdpt)

plt.plot(fqdpt, 'r')
plt.plot(fdpt, 'b')
