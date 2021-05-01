from scipy.interpolate import BSpline as BSp
from scipy import interpolate
import numpy as np

WFNAME = 'w_s/w.dat'

class wsr_Bspline:
    def __init__(self, gvar, k=3, knot_num=56):
        self.knot_num = knot_num
        self.k = k
        self.r = gvar.r
        lenr = len(self.r)
        r_spacing = int(lenr//knot_num)
        r_filtered = self.r[::r_spacing]
        self.knot_locs = r_filtered[1:-1]

        # will contain the knots
        self.t = None
        # to store the spline coeffs
        self.c1, self.c2, self.c3 = None, None, None

        # getting the initial guess of the spline coefficients from dpt profile
        self.wsr_dpt = np.loadtxt(f'{gvar.datadir}/{WFNAME}')\
            [:, gvar.rmin_idx:gvar.rmax_idx] * (-1.0)

        self.get_spline_coeffs(self.wsr_dpt)

        self.wsr = np.zeros_like(self.wsr_dpt)

    def get_spline_coeffs(self, wsr):
        w1, w3, w5 = wsr[0,:], wsr[1,:], wsr[2,:]
        # getting spline attributes (knots, coefficients, degree)
        self.t, self.c1, __ = interpolate.splrep(self.r, w1, s=0, k=self.k, t=self.knot_locs)
        __, self.c3, __ = interpolate.splrep(self.r, w3, s=0, k=self.k, t=self.knot_locs)
        __, self.c5, __ = interpolate.splrep(self.r, w5, s=0, k=self.k, t=self.knot_locs)

    def get_wsr_from_Bspline(self):
        spline = BSp(self.t, self.c1, self.k, extrapolate=True)
        self.wsr[0,:] = spline(self.r)

        spline = BSp(self.t, self.c3, self.k, extrapolate=True)
        self.wsr[1,:] = spline(self.r)

        spline = BSp(self.t, self.c5, self.k, extrapolate=True)
        self.wsr[2,:] = spline(self.r)