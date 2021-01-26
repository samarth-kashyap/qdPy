import numpy as np
import matplotlib.pyplot as plt
import modeparams as modepar
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

#----------------------------------------------------------------------
#                       All qts in CGS
# M_sol = 1.989e33 g
# R_sol = 6.956e10 cm
# B_0 = 10e5 G
# OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
# rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
#----------------------------------------------------------------------


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
gvar = globalvars.globalVars(rmin, rmax, args)
# }}} global variables


# (Since we are using lmax = 300, 0.45*300 \approx 150)
SMAX = 5      # maximum s for constructing supermatrix
FWINDOW = 150 # muHz

analysis_modes = modepar.qdptMode(gvar, SMAX, FWINDOW)
super_matrix = analysis_modes.create_supermatrix()
eig_vals_dpt = super_matrix.get_eigvals(type='DPT')
eig_vals_qdpt = super_matrix.get_eigvals(type='QDPT')

f_dpt = (analysis_modes.omega0 + eig_vals_dpt[0]/2/analysis_modes.omega0) 
f_qdpt = np.sqrt(analysis_modes.omega0**2 + eig_vals_qdpt[0])

# converting to muHz
f_dpt *= gvar.OM * 1e6
f_qdpt *= gvar.OM * 1e6
