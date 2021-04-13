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

class globalVars():

    def __init__(self, rmin, rmax, smax, fwindow, args):
        # self.datadir = "/scratch/g.samarth/qdPy"
        # self.progdir = "/home/g.samarth/qdPy"
        # self.eigdir = "/scratch/g.samarth/Solar_Eigen_function/eig_files"

        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais = dirnames[2]
        self.datadir = f"{self.snrnmais}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.eigdir = f"{self.snrnmais}/eig_files"
        self.progdir = self.local_dir

        # self.datadir = dirnames[0]
        # self.progdir = dirnames[1]
        # self.eigdir = dirnames[2]

        self.args = args

        # Frequency unit conversion factor (in Hz (cgs))
        #all quantities in cgs
        self.M_sol = 1.989e33 #gn,l = 0,200
        self.R_sol = 6.956e10 #cm
        self.B_0 = 10e5 #G
        self.OM = np.sqrt(4*np.pi*self.R_sol*self.B_0**2/self.M_sol) 
        # should be 2.096367060263202423e-05 for above numbers

        # self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{self.datadir}/r.dat")
        self.nl_all = np.loadtxt(f"{self.datadir}/nl.dat").astype('int')
        self.nl_all_list = np.loadtxt(f"{self.datadir}/nl.dat").astype('int').tolist()
        self.omega_list = np.loadtxt(f"{self.datadir}/muhz.dat") * 1e-6 / self.OM

        # getting indices for minimum and maximum r
        if args.precompute:
            self.rmin = 0.0
            self.rmax = 0.95
        elif args.use_precomputed:
            self.rmin = 0.95
            self.rmax = rmax
        else:
            self.rmin = rmin
            self.rmax = rmax

        self.rmin_idx = self.get_idx(self.r, self.rmin)
        self.rmax_idx = self.get_idx(self.r, self.rmax) #+ 1
        print(f"rmin index = {self.rmin_idx}; rmax index = {self.rmax_idx}")

        self.smax = smax
        self.fwindow = fwindow

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        # self.rho = self.mask_minmax(self.rho)

        self.n0 = args.n0
        self.l0 = args.l0

    def get_idx(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr):
        return arr[self.rmin_idx:self.rmax_idx]
