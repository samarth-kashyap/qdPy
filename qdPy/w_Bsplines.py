from scipy.interpolate import BSpline as BSp
from scipy import interpolate
import numpy as np
import qdPy.functions as FN

WFNAME = 'w_s/w.dat'

class wsr_Bspline:
    def __init__(self, gvar, rth=0.8, k=3, knot_num=56,
                 knot_update_num=1):
        self.knot_num = knot_num
        self.k = k
        self.r = gvar.r

        # the threshold radius beyond which spline is to be fitted
        self.rth = rth
        # finding the index of radius below which the profile does not change 
        # this is essentially where the matching function goes to zero
        self.r_th_ind = np.argmin(np.abs(get_matching_function(r, rth) - 1e-3))

        # radius array for spline
        self.r_spline = r[self.r_th_ind:]
        self.datadir = gvar.datadir
        lenr = len(self.r_spline)
        r_spacing = int(lenr//knot_num)
        r_filtered = self.r_spline[::r_spacing]
        self.knot_locs = r_filtered[1:-1]

        # will contain the knots
        self.t = None
        # to store the spline coeffs
        self.c1, self.c3, self.c5 = None, None, None

        # getting the initial guess of the spline coefficients from dpt profile
        self.wsr_dpt = np.loadtxt(f'{gvar.datadir}/{WFNAME}')\
            [:, gvar.rmin_idx:gvar.rmax_idx] * (-1.0)

        self.get_spline_coeffs(self.wsr_dpt[self.r_th_ind:])       # getting coefficients for the r_spline part
        self.knot_mask = np.zeros_like(self.t, dtype=np.bool)
        self.knot_mask[-knot_update_num-self.k-1:-self.k-1] = True
        self.knot_update_num = knot_update_num

        self.store_params_init()
        self.wsr = np.zeros_like(self.wsr_dpt) 
        self.get_wsr_from_Bspline()

    def store_params_init(self):
        params_init = []
        params_init.append(self.c1[self.knot_mask])
        params_init.append(self.c3[self.knot_mask])
        params_init.append(self.c5[self.knot_mask])
        # ?? to flatten or not flatten
        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_{self.knot_update_num:02d}.txt",
                   params_init)

        # storing the params for the extreme cases
        w1_upex_matched = np.zeros_like(self.wsr_dpt)
        w1_loex_matched = np.zeros_like(self.wsr_dpt)

        # looping over all the s in wsr
        for i in range(len(self.wsr_dpt)):
            w1_upex_matched[i,:] = FN.create_nearsurface_profile(self.r, self.rth, self.w_dpt[i,:], which_ex='upex')
            w1_loex_matched[i,:] = FN.create_nearsurface_profile(self.r, self.rth, self.w_dpt[i,:], which_ex='loex')

        # generating and saving coefficients for the upper extreme profiles
        c1, c3, c5 = self.get_spline_coeffs(self.wsr_upex_matched[self.r_th_ind:]) 

        params_init = []
        params_init.append(c1)
        params_init.append(c3)
        params_init.append(c5)

        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_upex{self.knot_update_num:02d}.txt",
                   params_init)

        # generating and saving coefficients for the lower extreme profiles
        c1, c3, c5 = self.get_spline_coeffs(self.wsr_loex_matched[self.r_th_ind:]) 

        params_init = []
        params_init.append(c1)
        params_init.append(c3)
        params_init.append(c5)

        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_loex{self.knot_update_num:02d}.txt",
                   params_init)
        

    def get_spline_coeffs(self, wsr, return_coeffs=False):
        w1, w3, w5 = wsr[0,:], wsr[1,:], wsr[2,:]
        # getting spline attributes (knots, coefficients, degree)
        t, c1, __ = interpolate.splrep(self.r_spline, w1, s=0,
                                                 k=self.k, t=self.knot_locs)
        __, c3, __ = interpolate.splrep(self.r_spline, w3, s=0, k=self.k, t=self.knot_locs)
        __, c5, __ = interpolate.splrep(self.r_spline, w5, s=0, k=self.k, t=self.knot_locs)
        # len(self.t) =/= len(self.knot_locs) WHY??

        # gets used during initializing extremal profiles
        if return_coeffs: return c1, c3, c5

        # gets used during MCMC
        else: 
            self.t = t
            self.c1, self.c3, self.c5 = c1, c3, c5

    def get_wsr_from_Bspline(self):
        # the non-fitting part
        self.wsr[:,:self.rth] = self.wsr_dpt[:,:self.rth]


        # the fitting part
        spline = BSp(self.t, self.c1, self.k, extrapolate=True)
        self.wsr[0, self.rth:] = spline(self.r_spline)

        spline = BSp(self.t, self.c3, self.k, extrapolate=True)
        self.wsr[1, self.rth:] = spline(self.r_spline)

        spline = BSp(self.t, self.c5, self.k, extrapolate=True)
        self.wsr[2, self.rth:] = spline(self.r_spline)


    def update_wsr_for_MCMC(self, params):
        ndim = len(params)
        assert ndim == self.knot_update_num*3, "Parameter size mismatch"
        slice1, slice2 = ndim//3, 2*ndim//3
        self.c1[self.knot_mask] = params[:slice1]
        self.c3[self.knot_mask] = params[slice1:slice2]
        self.c5[self.knot_mask] = params[slice2:]
        self.get_wsr_from_Bspline()
