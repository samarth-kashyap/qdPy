from scipy.interpolate import BSpline as BSp
from scipy import interpolate
import numpy as np

WFNAME = 'w_s/w.dat'

class wsr_Bspline:
    def __init__(self, gvar, k=3, knot_num=56, initialize=False):
        self.knot_num = knot_num
        self.k = k
        self.r = gvar.r
        self.gvar = gvar

        # the threshold radius beyond which spline is to be fitted
        self.rth = gvar.rth
        # finding the index of radius below which the profile does not change 
        # this is essentially where the matching function goes to zero
        self.rth_ind = np.argmin(np.abs(self.get_matching_function() - 1e-3))

        # radius array for spline
        # r_spline contains only values where r > rth
        self.r_spline = self.r[self.rth_ind:]
        self.datadir = gvar.datadir
        lenr = len(self.r_spline)
        r_spacing = int(lenr//knot_num)
        rfull_filtered = self.r[::r_spacing]
        self.knot_locs_full = rfull_filtered[1:-1]

        # among all the knots that exist between 0 < r < 1
        # we select only the knots that lie between rth < r < 1
        # only the BSpline coefficients corresponding to these knots
        # change for different iterations/walkers of MCMC.
        # knot_update_num counts the number of such knots.
        mask_rth = self.knot_locs_full > self.rth
        self.knot_locs = self.knot_locs_full[mask_rth]
        self.knot_update_num = len(self.knot_locs)

        # will contain the knots
        self.t = None
        # to store the spline coeffs
        self.c1, self.c3, self.c5 = None, None, None

        # getting the initial guess of the spline coefficients from dpt profile
        self.wsr_dpt = np.loadtxt(f'{gvar.datadir}/{WFNAME}')\
            [:, gvar.rmin_idx:gvar.rmax_idx] * (-1.0)

        # getting coefficients for the r_spline part
        self.get_spline_coeffs(self.wsr_dpt[:, self.rth_ind:])       
        self.knot_mask = np.zeros_like(self.t, dtype=np.bool)
        self.knot_mask[-self.knot_update_num-self.k-1:-self.k-1] = True

        if initialize:
            self.store_params_init()
        self.wsr = np.zeros_like(self.wsr_dpt) 
        self.get_wsr_from_Bspline()

    def store_params_init(self):
        fsuffix = f"{int(self.rth*100):03d}.txt"
        params_init = []
        params_init.append(self.c1[self.knot_mask])
        params_init.append(self.c3[self.knot_mask])
        params_init.append(self.c5[self.knot_mask])
        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_{fsuffix}", params_init)

        # storing the params for the extreme cases
        self.wsr_upex_matched = np.zeros_like(self.wsr_dpt)
        self.wsr_loex_matched = np.zeros_like(self.wsr_dpt)

        # looping over all the s in wsr
        for i in range(len(self.wsr_dpt)):
            self.wsr_upex_matched[i, :] = self.create_nearsurface_profile(i,
                                                                          which_ex='upex')
            self.wsr_loex_matched[i, :] = self.create_nearsurface_profile(i,
                                                                          which_ex='loex')

        # generating and saving coefficients for the upper extreme profiles
        c1, c3, c5 = self.get_spline_coeffs(self.wsr_upex_matched[:, self.rth_ind:],
                                            return_coeffs=True) 

        params_init = []
        params_init.append(c1)
        params_init.append(c3)
        params_init.append(c5)

        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_upex_{fsuffix}", params_init)

        # generating and saving coefficients for the lower extreme profiles
        c1, c3, c5 = self.get_spline_coeffs(self.wsr_loex_matched[:, self.rth_ind:],
                                            return_coeffs=True) 

        params_init = []
        params_init.append(c1)
        params_init.append(c3)
        params_init.append(c5)

        params_init = np.array(params_init).flatten()
        np.savetxt(f"{self.datadir}/params_init_loex_{fsuffix}", params_init)
        

    def get_spline_coeffs(self, wsr, return_coeffs=False):
        w1, w3, w5 = wsr[0, :], wsr[1, :], wsr[2, :]
        # getting spline attributes (knots, coefficients, degree)
        t, c1, __ = interpolate.splrep(self.r_spline, w1, s=0, k=self.k, t=self.knot_locs)
        __, c3, __ = interpolate.splrep(self.r_spline, w3, s=0, k=self.k, t=self.knot_locs)
        __, c5, __ = interpolate.splrep(self.r_spline, w5, s=0, k=self.k, t=self.knot_locs)

        # gets used during initializing extremal profiles
        if return_coeffs: return c1, c3, c5

        # gets used during MCMC
        else: 
            self.t = t
            self.c1, self.c3, self.c5 = c1, c3, c5

    def get_wsr_from_Bspline(self):
        # the non-fitting part
        self.wsr[:, :self.rth_ind] = self.wsr_dpt[:, :self.rth_ind]

        # the fitting part
        spline = BSp(self.t, self.c1, self.k, extrapolate=True)
        self.wsr[0, self.rth_ind:] = spline(self.r_spline)

        spline = BSp(self.t, self.c3, self.k, extrapolate=True)
        self.wsr[1, self.rth_ind:] = spline(self.r_spline)

        spline = BSp(self.t, self.c5, self.k, extrapolate=True)
        self.wsr[2, self.rth_ind:] = spline(self.r_spline)


    def update_wsr_for_MCMC(self, params):
        ndim = len(params)
        assert ndim == self.knot_update_num*3, "Parameter size mismatch"
        slice1, slice2 = ndim//3, 2*ndim//3
        self.c1[self.knot_mask] = params[:slice1]
        self.c3[self.knot_mask] = params[slice1:slice2]
        self.c5[self.knot_mask] = params[slice2:]
        self.get_wsr_from_Bspline()


    # functions to create the extreme profiles near the surface to get the 
    # range of spline coefficients for MCMC simulations
    def get_matching_function(self):
        return (np.tanh((self.r - self.rth)/0.05) + 1)/2.0

    def create_nearsurface_profile(self, idx, which_ex='upex'):
        w_dpt = self.wsr_dpt[idx, :]
        w_new = np.zeros_like(w_dpt)
        matching_function = self.get_matching_function()

        if which_ex == 'upex':
            if idx == 0: scale_factor = 1.1
            else: scale_factor = self.gvar.fac_up
        else:
            if idx == 0: scale_factor = 0.9 
            else: scale_factor = self.gvar.fac_lo

        # nea surface enhanced or suppressed profile
        w_new = matching_function * scale_factor * w_dpt

        # adding the complementary part below the rth
        w_new += (1 - matching_function) * w_dpt

        return w_new


