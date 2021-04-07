"""Computes the eigenfrequencies using QDPT and DPT"""
import argparse
import logging
import numpy as np
import qdclasses as qdcls
import ritzlavely as RL
import globalvars
import functions as FN


# {{{ def create_argparser():
def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0", help="radial order", type=int)
    parser.add_argument("--l0", help="angular degree", type=int)
    ARGS = parser.parse_args()
    return ARGS
# }}} create_argparser()


LOGGER = FN.create_logger(__name__, 'logs/qdpt.log', logging.DEBUG)
ARGS = create_argparser()


# {{{ Reading global variables
# setting rmax as 1.2 because the entire r array needs to be used
# in order to reproduce
# (1) the correct normalization
# (2) a1 = \omega_0 ( 1 - 1/ell ) scaling
RMIN, RMAX = 0.0, 1.2

# (Since we are using lmax = 300, 0.45*300 \approx 150)
SMAX = 5      # maximum s for constructing supermatrix
FWINDOW = 150   # microHz
GVAR = globalvars.globalVars(RMIN, RMAX, SMAX, FWINDOW, ARGS)
# }}} global variables


# {{{ def get_cenmode_freqs_qdpt(eigvals, eigvecs):
def get_cenmode_freqs_qdpt(eigvals, eigvecs):
    """Obtain the frequencies corresponding to the central ell - QDPT.

    Inputs:
    -------
    eigvals - np.ndarray(ndim=1, dtype=np.float)
        list of all the eigenvalues
    eigvecs - np.ndarray(ndim=2, dtype=np.float)
        orthogonal matrix with the columns corresponding to eigenvectors.

    Outputs:
    --------
    eigvals_sorted - np.ndarray(ndim=1, dtype=np.float)
        eigenvalues of the central ell sorted in ascending order of value
    eigvecs_sorted - np.ndarray(ndim=2, dtype=np.float)
        eigenvectors corresponding to the sorted eigenvalues.

    """
    cenmode_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(cenmode_idx)
    submat = qdcls.subMatrix(cenblock_idx, cenblock_idx, super_matrix)
    block_start = submat.startx
    block_end = submat.endx

    eigvals_sorted = np.zeros_like(eigvals)
    eigvecs_sorted = np.zeros_like(eigvecs)

    for i in range(len(eigvals)):
        eidx = abs(eigvecs[:, i]).argmax()
        eigvecs_sorted[eidx] = eigvecs[:, i]
        eigvals_sorted[eidx] = eigvals[i]

    return eigvals_sorted[block_start:block_end], eigvecs_sorted
# }}} get_cenmode_freqs_qdpt(eigvals, eigvecs)


# {{{ def get_cenmode_freqs_dpt(eigvals):
def get_cenmode_freqs_dpt(eigvals):
    """Obtain the frequencies corresponding to the central ell - DPT.

    Inputs:
    -------
    eigvals - np.ndarray(ndim=1, dtype=np.float)
        list of all the eigenvalues
    eigvecs - np.ndarray(ndim=2, dtype=np.float)
        orthogonal matrix with the columns corresponding to eigenvectors.

    Outputs:
    --------
    eigvals_sorted - np.ndarray(ndim=1, dtype=np.float)
        eigenvalues of the central ell sorted in ascending order of value
    eigvecs_sorted - np.ndarray(ndim=2, dtype=np.float)
        eigenvectors corresponding to the sorted eigenvalues.

    """
    cenmode_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(cenmode_idx)
    submat = qdcls.subMatrix(cenblock_idx, cenblock_idx, super_matrix)
    block_start = submat.startx
    block_end = submat.endx
    return eigvals[block_start:block_end]
# }}} get_cenmode_freqs_dpt(eigvals)


# {{{ def get_RL_coeffs(rlpObj, delta_omega_nlm):
def get_RL_coeffs(delta_omega_nlm):
    """Obtain the Ritzwoller-Lavely coefficients for a given
    array of frequencies.

    Inputs:
    -------
    delta_omega_nlm - np.ndarray(ndim=1, dtype=np.float)
        array containing the change in eigenfrequency due to
        rotation perturbation.

    Outputs:
    --------
    acoeffs - np.ndarray(ndim=1, dtype=np.float)
        the RL coefficients.
    """
    ritz_degree = min(ARGS.l0//2+1, 36)
    rlp = RL.ritzLavelyPoly(ARGS.l0, ritz_degree)
    rlp.get_Pjl()
    acoeffs = rlp.get_coeffs(delta_omega_nlm)
    acoeffs[0] = analysis_modes.omega0*GVAR.OM*1e6
    acoeffs = np.pad(acoeffs, (0, 36-ritz_degree), mode='constant')
    return acoeffs
# }}} get_RL_coeffs(rlpObj, delta_omega_nlm)


def store_offset():
    """Function to obtain the %change in L2 norm of QDPT - DPT. 
    Used to obtain the plot in Kashyap & Bharati Das et. al. (2021).
    """
    domega_QDPT = np.linalg.norm(fqdpt - analysis_modes.omega0*GVAR.OM*1e6)
    domega_DPT = np.linalg.norm(fdpt - analysis_modes.omega0*GVAR.OM*1e6)
    rel_offset_percent = np.abs((domega_QDPT-domega_DPT)/domega_DPT) * 100.0
    np.savetxt(f"{GVAR.datadir}/qdpt_error_full/offsets_{ARGS.n0:02d}_{ARGS.l0:03d}.dat",
            np.array([rel_offset_percent]))
    return 0


if __name__ == "__main__":
    analysis_modes = qdcls.qdptMode(GVAR)
    super_matrix = analysis_modes.create_supermatrix()
    eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
    eigvals_qdpt_unsorted, eigvecs_qdpt = super_matrix.get_eigvals(type='QDPT', sorted=False)

    eigvals_cenmode_dpt = get_cenmode_freqs_dpt(eigvals_dpt_unsorted)
    eigvals_cenmode_qdpt, eigvecs_qdpt = get_cenmode_freqs_qdpt(eigvals_qdpt_unsorted,
                                                eigvecs_qdpt)

    fdpt = (analysis_modes.omega0 + eigvals_cenmode_dpt/2/analysis_modes.omega0)
    fqdpt = (analysis_modes.omega0 + eigvals_cenmode_qdpt/2/analysis_modes.omega0)

    # converting to muHz
    fdpt *= GVAR.OM * 1e6
    fqdpt *= GVAR.OM * 1e6

    # dirname = "new_freqs_half"
    # dirnamenew = "new_freqs_430"
    dirnamenew = "new_freqs_w135_half"

    np.save(f'{GVAR.datadir}/{dirnamenew}/qdpt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fqdpt)
    np.save(f'{GVAR.datadir}/{dirnamenew}/dpt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fdpt)

    # converting to nHz before computing splitting coefficients
    fdpt *= 1e3
    fqdpt *= 1e3

    acoeffs_qdpt = get_RL_coeffs(fqdpt)
    acoeffs_dpt = get_RL_coeffs(fdpt)

    LOGGER.info("QDPT a-coeffs = {}".format(acoeffs_qdpt))
    LOGGER.info(" DPT a-coeffs = {}".format(acoeffs_dpt))
    np.save(f"{GVAR.datadir}/{dirnamenew}/" +
            f"qdpt_acoeffs_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_qdpt)
    np.save(f"{GVAR.datadir}/{dirnamenew}/" +
            f"dpt_acoeffs_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_dpt)
