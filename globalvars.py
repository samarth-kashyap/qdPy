import numpy as np

class globalVars():

    def __init__(self, rmin, rmax, args):
        self.datadir = "/scratch/g.samarth/qdPy"
        self.progdir = "/home/g.samarth/qdPy"
        # Frequency unit conversion factor (in Hz (cgs))
        self.OM = np.loadtxt(f"{self.datadir}/OM.dat")
        self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{self.datadir}/r.dat")
        self.nl_all = np.loadtxt(f"{self.datadir}/nl.dat")
        self.omega_list = np.loadtxt(f"{self.datadir}/muhz.dat") * 1e-6 / self.OM

        # getting indices for minimum and maximum r
        self.rmin_idx = self.get_idx(self.r, rmin)
        self.rmax_idx = self.get_idx(self.r, rmax)
        self.rmin = rmin
        self.rmax = rmax

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        self.rho = self.mask_minmax(self.rho)

        self.n0 = args.n0
        self.l0 = args.l0

    def get_idx(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr):
        return arr[self.rmin_idx:self.rmax_idx]
