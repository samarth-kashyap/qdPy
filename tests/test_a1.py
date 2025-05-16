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
args = parser.parse_args()
# }}} argparse


# {{{ Reading global variables
rmin, rmax = 0.0, 1.0
# (Since we are using lmax = 300, 0.45*300 \approx 150)
SMAX = 5      # maximum s for constructing supermatrix
FWINDOW = 150   # microHz
gvar = globalvars.globalVars(rmin, rmax, SMAX, FWINDOW, args)
# }}} global variables

ells_list = np.array([50, 100, 150, 200, 250, 300])
plt.figure()

for ell in ells_list:
    args.l0 = ell
    gvar = globalvars.globalVars(rmin, rmax, SMAX, FWINDOW, args)
    analysis_modes = qdcls.qdptMode(gvar)
    super_matrix = analysis_modes.create_supermatrix()
    eigvals_dpt_unsorted = super_matrix.get_eigvals(type='DPT', sorted=False)
    max_idx = 2*ell + 1
    fdpt = eigvals_dpt_unsorted[:max_idx].real/2/analysis_modes.omega0
    fdpt *= gvar.OM * 1e9
    m_arr = np.arange(-ell, ell+1)
    a1 = (fdpt*m_arr).sum()/(m_arr*m_arr).sum()
    a1th = 430*(1 - 1.0/ell - 1.0/2/ell/ell)
    plt.plot(ell, a1, '+k', label='QDPT')
    plt.plot(ell, a1th, '+b', label='theoretical value')
