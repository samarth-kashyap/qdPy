"""Computes the eigenfrequencies using QDPT and DPT"""
import logging
import time
import numpy as np
import qdPy.qdclasses as qdcls
import qdPy.ritzlavely as RL
from qdPy import globalvars
import qdPy.functions as FN
import qdPy.w_Bsplines as w_Bsp
from schwimmbad import MPIPool
from multiprocessing import Pool
from multiprocessing import cpu_count
import os
import emcee
import pickle as pkl


LOGGER = FN.create_logger_stream(__name__, 'logs/qdpt.log', logging.ERROR)
ARGS = FN.create_argparser()
DIRNAME_NEW = "w135_antia"

T1 = time.time()

GVAR = globalvars.globalVars(args=ARGS)
mcdict = {}
# }}} global variables

# creates new dir if it does not exist
if(not os.path.isdir(f"{GVAR.outdir}/{DIRNAME_NEW}")):
    os.mkdir(f"{GVAR.outdir}/{DIRNAME_NEW}")



def init_mcdict():
    spline_dict = w_Bsp.wsr_Bspline(GVAR, initialize=True)
    mcdict['spline'] = spline_dict
    return mcdict



# {{{ def start_mcmc():
def start_mcmc(ndim):
    """ Starts the MCMC sampler. """
    if ARGS.usempi:
        LOGGER.info(f"Process {mpirank:3d}: Starting MC sampler")
    else:
        LOGGER.info("Starting MC sampler")

    rth_percent = int(mcdict['spline'].rth*100)
    params_init = np.loadtxt(f"{GVAR.datadir}/" +
                             f"params_init_{rth_percent:03d}.txt")

    sampler = run_markov(params_init, maxiter=ARGS.maxiter,
                         usempi=ARGS.usempi)
                         # parallel=args.parallel, usempi=ARGS.usempi)

    targetdir = f"{GVAR.outdir}/{DIRNAME_NEW}"
    fn_spl = (f"{targetdir}/sampler.pkl")

    with open(fn_spl, "wb") as f:
        pkl.dump(sampler, f)
    return None
# }}} start_mcmc()


# {{{ def run_markov(self):
def run_markov(params_init, maxiter=10, parallel=False,
               backend=None, usempi=False):
    nwalkers = 2*len(params_init) + 1
    ndim = len(params_init)
    psig = abs(params_init/10)
    params_init = (params_init.reshape(1, ndim) +
                  abs(np.random.randn(nwalkers, ndim)*psig.reshape(1, ndim)))

    # prior_type = mcdict['prior_type']

    if parallel:
        print(f"Running Parallel")
        ncpu = cpu_count()
        print("{0} CPUs".format(ncpu))

        # NEED TO MAKE THIS DYNAMIC
        # 32 = ncpu? how does it work while using multiple nodes?
        with Pool(ncpu-1) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            pool=pool)
            sampler.run_mcmc(params_init, maxiter, progress=True)
            # , backend=backend)
    elif usempi:
        print(f"Process {mpirank:3d}: Running MPI")
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            pool=pool)
            sampler.run_mcmc(params_init, maxiter, progress=True)
    else:
        print(f"Running serial")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(params_init, maxiter, progress=True)
        # , backend=backend)
    return sampler
# }}} run_markov(self)


# {{{ def log_probability(params):
def log_probability(params):
    logpr = log_prior(params)
    # save up computation by checking if logpr is infinite
    # most expensive step is computation of log_likelihood
    # if not np.isfinite(logpr):
        # return -np.inf
    return logpr + log_likelihood(params)
    # return log_likelihood(params)
# }}} log_probability(params)


# {{{ def log_likelihood(params):
def log_likelihood(params):
    res_by_sigma_sum = compute_res(params)
    return -0.5 * res_by_sigma_sum
# }}} log_likelihood(params)


# {{{ def log_prior(params):
def log_prior(params):
    ndim = len(params)
    logpr = 0

    rth_percent = int(mcdict['spline'].rth*100)
    fsuffix = f"{rth_percent:03d}.txt"
    # loading params from the saved file of upex and loex profiles
    true_params = np.loadtxt(f"{GVAR.datadir}/params_init_{fsuffix}")
    params_upper = np.loadtxt(f"{GVAR.datadir}/params_init_upex_{fsuffix}")
    params_lower = np.loadtxt(f"{GVAR.datadir}/params_init_loex_{fsuffix}")

    for i in range(ndim):
        if params_lower[i] < params[i] < params_upper[i]:
            logpr += -np.log(abs(params_upper[i] - params_lower[i]))
        else:
            logpr += -np.inf
    return logpr
# }}} log_prior(self, params)


# {{{ def compute_res(params):
def compute_res(params):
    n = ARGS.n0
    ells = np.arange(ARGS.lmin, ARGS.lmax+1)
    res = 0.0
    for ell in ells:
        ARGS.l0 = ell
        ritz_degree = min(ARGS.l0//2+1, 36)
        GVAR = globalvars.globalVars(args=ARGS)
        spline_dict = w_Bsp.wsr_Bspline(GVAR)
        spline_dict.update_wsr_for_MCMC(params)
        analysis_modes = qdcls.qdptMode(GVAR, spline_dict)
        super_matrix = analysis_modes.create_supermatrix()

        fqdpt = solve_eigprob(analysis_modes)

        # converting to nHz
        fqdpt *= GVAR.OM * 1e9

        acoeffs_qdpt = get_RL_coeffs(analysis_modes, GVAR, fqdpt, ritz_degree)

        mask_nl = (GVAR.hmidata[:, 0] == ARGS.l0)*(GVAR.hmidata[:, 1] == ARGS.n0)
        splitdata = GVAR.hmidata[mask_nl, 12:12+ritz_degree]
        sigdata = GVAR.hmidata[mask_nl, 48:48+ritz_degree]

        res += np.sum(((acoeffs_qdpt[1:] - splitdata)**2)/(sigdata**2))
    print(f"==========RES = {res} =================")
    print(f"==========params = {params} =================")
    return res
# }}} compute_res(params)


# {{{ def get_l0_freqs_qdpt(eigvals, eigvecs):
def get_l0_freqs_qdpt(analysis_modes, eigvals, eigvecs):
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
    super_matrix = analysis_modes.super_matrix
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


# {{{ def get_RL_coeffs(rlpObj, delta_omega_nlm):
def get_RL_coeffs(analysis_modes, GVAR, delta_omega_nlm, ritz_degree):
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
    rlp = RL.ritzLavelyPoly(analysis_modes.l0, ritz_degree)
    rlp.get_Pjl()
    acoeffs = rlp.get_coeffs(delta_omega_nlm)
    acoeffs[0] = analysis_modes.omega0*GVAR.OM*1e6
    acoeffs = np.pad(acoeffs, (0, 36-ritz_degree), mode='constant')
    return acoeffs
# }}} get_RL_coeffs(rlpObj, delta_omega_nlm)


# {{{ def solve_eigprob(super_matrix):
def solve_eigprob(analysis_modes):
    super_matrix = analysis_modes.super_matrix
    eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
    eigvals_qdpt_unsorted, eigvecs_qdpt = super_matrix.get_eigvals(type='QDPT', sorted=False)

    eigvals_l0_qdpt, eigvecs_qdpt = get_l0_freqs_qdpt(analysis_modes,
                                                      eigvals_qdpt_unsorted,
                                                      eigvecs_qdpt)

    fqdpt = (analysis_modes.omega0 + eigvals_l0_qdpt/2/analysis_modes.omega0)
    return fqdpt
# }}} def solve_eigprob(super_matrix)



if __name__ == "__main__":
    mcdict = init_mcdict()
    start_mcmc(3)
    # spline_dict = w_Bsp.wsr_Bspline(GVAR)     # can access the coeffs through spline_dict.[c1,c3,c5]
    # analysis_modes = qdcls.qdptMode(GVAR, spline_dict)
    # super_matrix = analysis_modes.create_supermatrix()

    # fdpt, fqdpt = solve_eigprob()

    # # converting to muHz
    # fdpt *= GVAR.OM * 1e6
    # fqdpt *= GVAR.OM * 1e6

    # np.save(f'{GVAR.outdir}/{DIRNAME_NEW}/qdpt_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fqdpt)
    # np.save(f'{GVAR.outdir}/{DIRNAME_NEW}/dpt_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy', fdpt)

    # # converting to nHz before computing splitting coefficients
    # fdpt *= 1e3
    # fqdpt *= 1e3

    # acoeffs_qdpt = get_RL_coeffs(fqdpt)
    # acoeffs_dpt = get_RL_coeffs(fdpt)

    # LOGGER.info("QDPT a-coeffs = {}".format(acoeffs_qdpt))
    # LOGGER.info(" DPT a-coeffs = {}".format(acoeffs_dpt))
    # np.save(f"{GVAR.outdir}/{DIRNAME_NEW}/" +
    #         f"qdpt_acoeffs_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_qdpt)
    # np.save(f"{GVAR.outdir}/{DIRNAME_NEW}/" +
    #         f"dpt_acoeffs_opt_{ARGS.n0:02d}_{ARGS.l0:03d}.npy", acoeffs_dpt)

    # T2 = time.time()
    # LOGGER.info("Time taken = {:7.2f} seconds".format((T2-T1)))
    # print("Time taken = {:7.2f} seconds".format((T2-T1)))

