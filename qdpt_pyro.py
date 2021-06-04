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

from qdPy import functions as FN
from qdPy import globalvars
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
        LOGGER.info('Mode file not found for mode index = {}'\
                    .format(mode_idx))
        return None
    # converting to device array
    U_jax, V_jax = np.asarray(U, DTYPE), np.asarray(V, DTYPE) 
    return U_jax, V_jax

def Omega(ell, N):
    if abs(N) > ell:
        return 0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1))

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

########################################################################
# testing the jax functions
U,V = get_eig(1000)

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
