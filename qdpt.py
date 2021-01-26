import numpy as np
import matplotlib.pyplot as plt
import miscfuncs as fn
import globalvars
import logging
import argparse

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
args = parser.parse_args()
# }}} argparse


# {{{ Reading global variables
rmin, rmax = 0.0, 1.0
gvar = globalvars.globalVars(rmin, rmax, args)
nl_all_list = gvar.nl_all.astype('int').tolist()
omega_list = gvar.omega_list
# }}} global variables


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

#max s values for DR and magnetic field to speed up supermatrix computation
s_max_DR = 5    

#the frequency window around each mode
# f_window = 50  #l0 #in muHz
# Since we are using lmax = 300, 0.45*300 \approx 150
f_window = 150  #in muHz

omega_nl0 = omega_list[fn.nl_idx(n0, l0)]
omega_nl0 = omega_list[fn.nl_idx(100, 100)]
omega_ref0 = omega_nl0

#central mode is at index zero in nl_list
nl_list = fn.nearest_freq_modes(l0,s_max_DR,omega_nl0,f_window)
nl_list = nl_list.astype('int64')   #making the modes integers for future convenience

omega_nl = np.array([omega_list[fn.nl_idx(mode[0], mode[1])]
                     for mode in nl_list])

