import numpy as np
import os

#----------------------------------------------------------------------
#                       All qts in CGS
# M_sol = 1.989e33 g
# R_sol = 6.956e10 cm
# B_0 = 10e5 G
# OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
# rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
#----------------------------------------------------------------------
filenamepath = os.path.realpath(__file__)
filepath = '/'.join(filenamepath.split('/')[:-1])
configpath = filepath
with open(f"{configpath}/.config", "r") as f:
    dirnames = f.read().splitlines()

print(dirnames)

class globalVars():

    def __init__(self, rmin, rmax, smax, fwindow, args):
        self.datadir = "/scratch/g.samarth/qdPy"
        self.progdir = "/home/g.samarth/qdPy"
        self.eigdir = "/scratch/g.samarth/Solar_Eigen_function/eig_files"

        # Frequency unit conversion factor (in Hz (cgs))
        self.OM = np.loadtxt(f"{self.datadir}/OM.dat")
        self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{self.datadir}/r.dat")
        self.nl_all = np.loadtxt(f"{self.datadir}/nl.dat").astype('int')
        self.nl_all_list = np.loadtxt(f"{self.datadir}/nl.dat").astype('int').tolist()
        self.omega_list = np.loadtxt(f"{self.datadir}/muhz.dat") * 1e-6 / self.OM

        # getting indices for minimum and maximum r
        self.rmin_idx = self.get_idx(self.r, rmin)
        self.rmax_idx = self.get_idx(self.r, rmax) #+ 1
        self.rmin = rmin
        self.rmax = rmax
        self.smax = smax
        self.fwindow = fwindow

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        self.rho = self.mask_minmax(self.rho)

        self.n0 = args.n0
        self.l0 = args.l0

    def get_idx(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr):
        return arr[self.rmin_idx:self.rmax_idx]
