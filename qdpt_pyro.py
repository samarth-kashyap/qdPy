# miscellaneous imports                                                                                            
import jax
from jax import random
from jax import grad, jit
import jax.numpy as np # using jax.numpy instead                                                                   
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import scipy.sparse
import sys
import numpy as onp
from scipy.interpolate import BSpline as BSp

from qdPy import functions as FN
from qdPy import globalvars
from qdPy import w_Bsplines_pyro as w_Bsp
from qdPy import qdclasses_pyro as qdcls
########################################################################
# importing the global variables
ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(args=ARGS)

DTYPE='float32'
########################################################################
# jax function

# extracting the eigenfunctions
def get_eig(mode_idx):
    try:
        U = onp.loadtxt(f'{GVAR.eigdir}/' +
                       f'U{mode_idx}.dat')[GVAR.rmin_idx:GVAR.rmax_idx]
        V = onp.loadtxt(f'{GVAR.eigdir}/' +
                       f'V{mode_idx}.dat')[GVAR.rmin_idx:GVAR.rmax_idx]
    except FileNotFoundError:
        return None
    # converting to device array
    U_jax, V_jax = np.asarray(U, DTYPE), np.asarray(V, DTYPE) 
    return U_jax, V_jax

@jit
def Omega(ell, N):
    return np.sqrt(0.5 * (ell+N) * (ell-N+1))
    # the if statement was causing an error for jax
    # the jitted function takes 50microsec and 
    # the non-jitted function takes 125microsec
    '''
    if abs(N) > ell:
        return 0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1))
    '''



# computing the T_s_r kernel for QDPT rotation
def compute_Tsr(ix, iy, s_arr):
    Tsr = np.zeros((len(s_arr), len(GVAR.r)))
    if GVAR.args.use_precomputed:
        enn1 = sup.nl_neighbors[ix, 0]
        ell1 = sup.nl_neighbors[ix, 1]
        enn2 = sup.nl_neighbors[iy, 0]
        ell2 = sup.nl_neighbors[iy, 1]
        arg_str1 = f"{enn1}.{ell1}"
        arg_str2 = f"{enn2}.{ell2}"
        U1 = sup.eigU[arg_str1]
        U2 = sup.eigU[arg_str2]
        V1 = sup.eigV[arg_str1]
        V2 = sup.eigV[arg_str2]
    else:
        m1idx = sup.nl_neighbors_idx[ix]
        m2idx = sup.nl_neighbors_idx[iy]
        U1, V1 = get_eig(m1idx)
        U2, V2 = get_eig(m2idx)
    L1sq = ell1*(ell1+1)
    L2sq = ell2*(ell2+1)
    Om1 = Omega(ell1, 0)
    Om2 = Omega(ell2, 0)
    for i in range(len(s_arr)):
        s = s_arr[i]
        ls2fac = L1sq + L2sq - s*(s+1)
        eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
        wigval = w3j(ell1, s, ell2, -1, 0, 1)
        Tsr[i, :] = -(1 - minus1pow(ell1 + ell2 + s)) * \
                    Om1 * Om2 * wigval * eigfac / GVAR.r
    return Tsr

# SBD: I dont think we need to @jit this function but other functions 
# called from this.
# {{{ def compute_res(params):                                                                                     
def compute_res(params):
    n = GVAR.n0
    ells = np.arange(GVAR.args.lmin, GVAR.args.lmax+1)
    res = 0.0
    counter = 0

    spline_dict = w_Bsp.wsr_Bspline(GVAR)
    # return spline_dict
    spline_dict.update_wsr_for_MCMC(params)

    for ell in ells:
        GVAR.args.l0 = ell
        ritz_degree = min(GVAR.args.l0//2+1, 36)
        # GVAR = globalvars.globalVars(args=ARGS)
        analysis_modes = qdcls.qdptMode(GVAR, spline_dict)
        super_matrix = analysis_modes.create_supermatrix()
        supmat_qdpt = super_matrix.supmat
        fqdpt = solve_eigprob(supmat_qdpt, analysis_modes.omega0)
        # converting to nHz                                                                                        
        fqdpt *= GVAR.OM * 1e9

        acoeffs_qdpt = get_RL_coeffs(analysis_modes, GVAR, fqdpt, ritz_degree)

        mask_nl = (GVAR.hmidata[:, 0] == ARGS.l0)*(GVAR.hmidata[:, 1] == ARGS.n0)
        splitdata = GVAR.hmidata[mask_nl, 12:12+ritz_degree].flatten()
        sigdata = GVAR.hmidata[mask_nl, 48:48+ritz_degree].flatten()

        res += np.sum(((acoeffs_qdpt[1:] - splitdata)**2)/(sigdata**2))
        counter += len(splitdata)

    print(f"==========RES = {res} =================")
    print(f"==========params = {params} =================")
    print(f'Memory used = {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3} GB')
    T2 = time.time()
    print(f"Ttal time taken = {(T2-T1)/60:7.3f} minutes")
    return res/counter

    return None

# {{{ def solve_eigprob(super_matrix):                                                                             
@jit
def solve_eigprob(supmat_qdpt, analysis_modes_omega0):
    supmat_dpt = np.diag(np.diag(supmat_qdpt))
    eigvals_dpt_unsorted = qdcls.get_eigvals_DPT(supmat_dpt)
    eigvals_qdpt_unsorted, eigvecs_qdpt = qdcls.get_eigvals_QDPT(supmat_qdpt)

    #eigvals_l0_qdpt, eigvecs_qdpt = get_l0_freqs_qdpt(analysis_modes,
    #                                                  eigvals_qdpt_unsorted,
    #                                                  eigvecs_qdpt)

    #fqdpt = (analysis_modes_omega0 + eigvals_l0_qdpt/2/analysis_modes_omega0)
    #return fqdpt
    return np.mean(eigvals_qdpt_unsorted)
# }}} def solve_eigprob(super_matrix)  

########################################################################
# testing the jax functions
# get_eig
U,V = get_eig(1000)

# comptue_res (this is the pivotal function)
param_true = np.asarray(onp.loadtxt(f'{GVAR.datadir}/params_init_098.txt'))
spline_dict = compute_res(param_true)

sys.exit()
########################################################################

# Choose the "true" parameters.                                                                                    
m_true = -0.9594
b_true = 4.294
f_true = 0.534
# f_true = 1.0                                                                                                     

# initializing parameters                                                                                          
N = 50; J = 2
X = random.normal(random.PRNGKey(seed = 123), (N, J))
weight = np.array([m_true, 10*m_true])
# weight = np.zeros((1,1)) + m_true                                                                                
error = 0.1 * random.normal(random.PRNGKey(234), (N, )) # standard Normal                                          
y_obs = f_true * (X @ weight + b_true) + error*0.01

# setting up model                                                                                                 
def model(X, y=None):
    ndims = np.shape(X)[-1]
    ws = numpyro.sample('betas', dist.Normal(0.0,10.0*np.ones(ndims)))
    b = numpyro.sample('b', dist.Normal(0.0, 10.0))
    sigma = numpyro.sample('sigma', dist.Uniform(0.0, 10.0))
    f = numpyro.sample('f', dist.Normal(0.0, 2.5))
    mu = f * (X @ ws + b)
    return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

# setting up the sampler                                                                                           
nuts_kernel = NUTS(model)
num_warmup, num_samples = 500, 1500
mcmc = MCMC(nuts_kernel, num_warmup, num_samples, num_chains=1)

# sampling                                                                                                         
mcmc.run(random.PRNGKey(240), X, y = y_obs)

# printing the NUTS summary                                                                                        
print(mcmc.print_summary())
