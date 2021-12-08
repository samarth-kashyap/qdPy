B"""Computes the eigenfrequencies using QDPT and DPT"""
import logging
import time
import numpy as np
import qdPy.qdclasses as qdcls
import qdPy.ritzlavely as RL
from qdPy import globalvars
import qdPy.functions as FN
import qdPy.w_Bsplines as w_Bsp
import os


LOGGER = FN.create_logger_stream(__name__, 'logs/qdpt.log', logging.WARNING)
ARGS = FN.create_argparser()
DIRNAME_NEW = "w135_antia"

T1 = time.time()

GVAR = globalvars.globalVars(args=ARGS)
# }}} global variables

# creates new dir if it does not exist
if(not os.path.isdir(f"{GVAR.outdir}/{DIRNAME_NEW}")):
    os.mkdir(f"{GVAR.outdir}/{DIRNAME_NEW}")

# {{{ def get_l0_freqs_qdpt(eigvals, eigvecs):
def get_l0_freqs_qdpt(eigvals, eigvecs):
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
    l0_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(l0_idx)
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
# }}} get_l0_freqs_qdpt(eigvals, eigvecs)


# {{{ def get_l0_freqs_dpt(eigvals):
def get_l0_freqs_dpt(eigvals):
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
    l0_idx = analysis_modes.idx
    neighbors_idx = analysis_modes.nl_neighbors_idx.tolist()
    cenblock_idx = neighbors_idx.index(l0_idx)
    submat = qdcls.subMatrix(cenblock_idx, cenblock_idx, super_matrix)
    block_start = submat.startx
    block_end = submat.endx
    return eigvals[block_start:block_end]
# }}} get_l0_freqs_dpt(eigvals)


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


# {{{ def store_offset():
def store_offset():
    """Function to obtain the %change in L2 norm of QDPT - DPT. 
    Used to obtain the plot in Kashyap & Bharati Das et. al. (2021).
    """
    omega0 = analysis_modes.omega0*GVAR.OM*1e6
    domega_qdpt = np.linalg.norm(fqdpt - omega0)
    domega_dpt = np.linalg.norm(fdpt - omega0)
    rel_offset_percent = np.abs((domega_qdpt -domega_dpt)/domega_dpt) * 100.0
    np.savetxt(f"{GVAR.datadir}/qdpt_error_full/" +
               f"offsets_{ARGS.n0:02d}_{ARGS.l0:03d}.dat",
               np.array([rel_offset_percent]))
    return 0
# }}} store_offset()


# {{{ def solve_eigprob():
def solve_eigprob():
    eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
    eigvals_qdpt_unsorted, eigvecs_qdpt = super_matrix.get_eigvals(type='QDPT', sorted=False)

    eigvals_l0_dpt = get_l0_freqs_dpt(eigvals_dpt_unsorted)
    eigvals_l0_qdpt, eigvecs_qdpt = get_l0_freqs_qdpt(eigvals_qdpt_unsorted,
                                                                eigvecs_qdpt)

    fdpt = (analysis_modes.omega0 + eigvals_l0_dpt/2/analysis_modes.omega0)
    fqdpt = (analysis_modes.omega0 + eigvals_l0_qdpt/2/analysis_modes.omega0)
    return fdpt, fqdpt
# }}} def solve_eigprob():



if __name__ == "__main__":
    spline_dict = w_Bsp.wsr_Bspline(GVAR)     # can access the coeffs through spline_dict.[c1,c3,c5]
    analysis_modes = qdcls.qdptMode(GVAR, spline_dict)
    super_matrix = analysis_modes.create_supermatrix()

    # saving the supermatrix to compare with pyro
    np.save(f'supmat_qdpt_{ARGS.l0}.npy', super_matrix.supmat)
    
    fdpt, fqdpt = solve_eigprob()

    # converting to muHz
    fdpt *= GVAR.OM * 1e6
    fqdpt *= GVAR.OM * 1e6

    np.save(f'{GVAR.outdir}/{DIRNAME_NEW}/qdpt_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fqdpt)
    np.save(f'{GVAR.outdir}/{DIRNAME_NEW}/dpt_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fdpt)

    # converting to nHz before computing splitting coefficients
    fdpt *= 1e3
    fqdpt *= 1e3

    acoeffs_qdpt = get_RL_coeffs(fqdpt)
    acoeffs_dpt = get_RL_coeffs(fdpt)

    LOGGER.info("QDPT a-coeffs = {}".format(acoeffs_qdpt))
    LOGGER.info(" DPT a-coeffs = {}".format(acoeffs_dpt))
    np.save(f"{GVAR.outdir}/{DIRNAME_NEW}/" +
            f"qdpt_acoeffs_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_qdpt)
    np.save(f"{GVAR.outdir}/{DIRNAME_NEW}/" +
            f"dpt_acoeffs_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_dpt)

    T2 = time.time()
    LOGGER.info("Time taken = {:7.2f} seconds".format((T2-T1)))
    print("Time taken = {:7.2f} seconds".format((T2-T1)))

