"""Class to handle QDPT computation"""
import logging
import numpy as np
import py3nj
from scipy.integrate import simps
import scipy as sp
import qdPy.functions as FN
from tqdm import tqdm
import time


NAX = np.newaxis
LOGGER = FN.create_logger_stream(__name__, 'logs/qdpt.log', logging.ERROR)
WFNAME = 'w_s/w.dat'
# WFNAME = 'w_s/w_const.dat'
# WFNAME = 'w_s/w_const_430.dat'
# WFNAME = 'w_s/w_jesper.dat'  # to match with jesper's data
LOGGER.info(f"Using velocity profile - {WFNAME}")


def w3j(l1, l2, l3, m1, m2, m3):
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = int(2*m1)
    m2 = int(2*m2)
    m3 = int(2*m3)
    try:
        wigval = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    except ValueError:
        return 0.0
    return wigval


def w3j_vecm(l1, l2, l3, m1, m2, m3):
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    return wigvals


def Omega(ell, N):
    if abs(N) > ell:
        return 0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1))


def minus1pow(num):
    if num%2 == 1:
        return -1.0
    else:
        return 1.0


def minus1pow_vec(num):
    modval = num % 2
    retval = np.zeros_like(modval)
    retval[modval == 1] = -1.0
    retval[modval == 0] = 1.0
    return retval


def gamma(ell):
    return np.sqrt((2*ell + 1)/4/np.pi)


class qdptMode:
    def __init__(self, gvar, spline_dict):
        self.gvar = gvar
        self.spline_dict = spline_dict
        self.n0 = gvar.n0
        self.l0 = gvar.l0
        self.smax = gvar.smax
        self.freq_window = gvar.fwindow

        self.idx = self.nl_idx(self.n0, self.l0)
        self.omega0 = self.gvar.omega_list[self.idx]
        self.get_mode_neighbors_params()
        self.spline_dict = spline_dict

    def nl_idx(self, n0, l0):
        try:
            idx = self.gvar.nl_all_list.index([n0, l0])
        except ValueError:
            idx = None
            logger.error('Mode not found')
        return idx

    def nl_idx_vec(self, nl_list):
        nlnum = nl_list.shape[0]
        nlidx = np.zeros(nlnum, dtype=np.int)
        for i in range(nlnum):
            nlidx[i] = self.nl_idx(nl_list[i][0],
                                   nl_list[i][1])
        return nlidx

    def get_omega_neighbors(self, nl_idx):
        nlnum = len(nl_idx)
        omega_neighbors = np.zeros(nlnum)
        for i in range(nlnum):
            omega_neighbors[i] = self.gvar.omega_list[nl_idx[i]]
        return omega_neighbors

    def get_mode_neighbors_params(self):
        omega_list = self.gvar.omega_list
        omega0 = self.omega0
        nl_all = self.gvar.nl_all
        omega_diff = (omega_list - omega0) * self.gvar.OM * 1e6
        mask_omega = abs(omega_diff) <= self.freq_window
        mask_ell = abs(nl_all[:, 1] - self.l0) <= self.smax

        # only even l1-l2 is coupled for odd-s rotation perturbation
        mask_odd = ((nl_all[:, 1] - self.l0)%2) == 0
        mask_nb = mask_omega * mask_ell * mask_odd
        sort_idx = np.argsort(abs(omega_diff[mask_nb]))
        self.nl_neighbors = nl_all[mask_nb][sort_idx]
        self.nl_neighbors_idx = self.nl_idx_vec(self.nl_neighbors)
        self.omega_neighbors = self.get_omega_neighbors(self.nl_neighbors_idx)
        self.num_neighbors = len(self.nl_neighbors_idx)
        LOGGER.info("Found {} neighbors to mode (n, ell) = ({}, {})"\
                    .format(self.num_neighbors, self.n0, self.l0))

    def create_supermatrix(self):
        t1 = time.time()
        supmat = superMatrix(self.gvar, self.spline_dict, self.num_neighbors,
                             self.nl_neighbors, self.nl_neighbors_idx,
                             self.omega0, self.omega_neighbors)
        self.super_matrix = supmat
        t2 = time.time()
        LOGGER.info("Time taken to create supermatrix = {:7.2f} seconds".format((t2-t1)))
        LOGGER.debug("Max supermatrix = {}".format(abs(supmat.supmat).max()))
        return supmat

    def update_supermatrix(self):
        t1 = time.time()
        self.super_matrix.fill_supermatrix()
        self.super_matrix.fill_supermatrix_freqdiag()
        t2 = time.time()
        print("Time taken to update supermatrix = {:7.2f} seconds".format((t2-t1)))


class superMatrix():
    def __init__(self, gvar, spline_dict, dim, nl_neighbors, nl_neighbors_idx, omegaref,
                 omega_neighbors):
        self.gvar = gvar
        self.spline_dict = spline_dict
        self.omegaref = omegaref
        self.omega_neighbors = omega_neighbors
        self.dim_blocks = dim
        self.nl_neighbors = nl_neighbors
        self.nl_neighbors_idx = nl_neighbors_idx
        self.dimX_submat = 2*nl_neighbors[:, 1].reshape(1, dim) * np.ones((dim, 1)) + 1
        self.dimY_submat = 2*nl_neighbors[:, 1].reshape(dim, 1) * np.ones((1, dim)) + 1
        self.dim_super = int(self.dimX_submat[0, :].sum())
        self.supmat = np.zeros((self.dim_super, self.dim_super),
                                    dtype=np.complex128)
        self.supmat_dpt = np.zeros((self.dim_super, self.dim_super),
                                    dtype=np.complex128)

        # loading precomputed values before recomputing the submatrices.
        if gvar.args.use_precomputed:
            narr = self.nl_neighbors[:, 0]
            larr = self.nl_neighbors[:, 1]
            self.Cvec_pc = self.get_precomputed_Cvec(narr, larr)

        self.eigU, self.eigV = self.load_eigs()

        self.fill_supermatrix()
        self.supmat_dpt = np.diag(np.diag(self.supmat))
        self.fill_supermatrix_freqdiag()
        return None

    def load_eigs(self):
        eigU = {}
        eigV = {}
        for im, mode_idx in enumerate(self.nl_neighbors_idx):
            LOGGER.debug('Getting eigenfunctions for mode_idx = {}: nl = {}'\
                        .format(mode_idx, self.nl_neighbors[im, :]))
            try:
                U = np.loadtxt(f'{self.gvar.eigdir}/' +
                            f'U{mode_idx}.dat')[self.gvar.rmin_idx:self.gvar.rmax_idx]
                V = np.loadtxt(f'{self.gvar.eigdir}/' +
                            f'V{mode_idx}.dat')[self.gvar.rmin_idx:self.gvar.rmax_idx]
            except FileNotFoundError:
                LOGGER.info('Mode file not found for mode index = {}'\
                            .format(mode_idx))
            enn, ell = self.nl_neighbors[im, 0], self.nl_neighbors[im, 1]
            arg_str = f"{enn}.{ell}"
            eigU[arg_str] = U
            eigV[arg_str] = V
        return eigU, eigV


    def get_precomputed_Cvec(self, narr, larr):
        Cvec_pc = {}
        for i in range(self.dim_blocks):
            for ii in range(i, self.dim_blocks):
                n1, n2 = narr[i], narr[ii]
                l1, l2 = larr[i], larr[ii]
                arg_str = f"{n1}.{l1}-{n2}.{l2}"
                Cvec_pc[arg_str] = np.load(f"{self.gvar.outdir}/submatrices/" +
                                           f"pc.{n1}.{l1}-{n2}.{l2}.npy")
        return Cvec_pc

    def fill_supermatrix(self):
        LOGGER.info("Creating submatrices for: ")
        for i in tqdm(range(self.dim_blocks), desc='Submatrices'):
            for ii in range(i, self.dim_blocks):
                sm = subMatrix(i, ii, self, printinfo=True)
                submat = sm.get_submat()
                self.supmat[sm.startx:sm.endx,
                            sm.starty:sm.endy] = submat
                self.supmat[sm.starty:sm.endy,
                            sm.startx:sm.endx] = submat.T.conj()
                LOGGER.debug("- Max submat ({}, {}) = {}".format(sm.ell1,
                                                                 sm.ell2,
                                                                 abs(submat).max()))

    def fill_supermatrix_freqdiag(self):
        for i in range(self.dim_blocks):
            sm = subMatrix(i, i, self)
            om2diff = self.omega_neighbors[i]**2 - self.omegaref**2
            om2diff *= np.identity(sm.endx-sm.startx)
            self.supmat[sm.startx:sm.endx,
                        sm.starty:sm.endy] += om2diff


    def get_eigvals(self, type='DPT', sorted=False):
        eigvals_list = []
        if type == 'DPT':
            eigvals_all = np.diag(self.supmat_dpt)
            return eigvals_all.real
        elif type == 'QDPT':
            eigvals_all, eigvecs = sp.linalg.eigh(self.supmat)
            return eigvals_all.real, eigvecs

        if sorted:
            if type == 'DPT':
                return np.sort(eigvals_all).real
            elif type == 'QDPT':
                eigbasis_sort = np.zeros(len(eigvals_all), dtype=np.int)
                for i in range(len(eigvals_all)):
                    eigbasis_sort[i] = abs(eigvecs[i]).argmax()
                return eigvals_all[eigbasis_sort].real

        # for i in range(self.dim_blocks):
        #     sm = subMatrix(i, i, self)
        #     eigvals_list.append(eigvals_all[sm.startx:sm.endx])
        # return eigvals_list


class subMatrix():
    def __init__(self, ix, iy, sup, printinfo=False):
        LOGGER.info("--- (n1, l1) = ({}, {}) and (n2, l2) = ({}, {})"\
                    .format(sup.nl_neighbors[ix, 0], sup.nl_neighbors[ix, 1],
                            sup.nl_neighbors[iy, 0], sup.nl_neighbors[iy, 1]))
        self.ix, self.iy = int(ix), int(iy)
        self.sup = sup
        self.rmin_idx = self.sup.gvar.rmin_idx
        self.rmax_idx = self.sup.gvar.rmax_idx
        self.startx = int(sup.dimX_submat[0, :ix].sum())
        self.starty = int(sup.dimY_submat[:iy, 0].sum())
        self.endx = int(sup.dimX_submat[0, :int(ix+1)].sum()) # + 1)
        self.endy = int(sup.dimY_submat[:int(iy+1), 0].sum()) # + 1)

        self.n1 = sup.nl_neighbors[ix, 0]
        self.n2 = sup.nl_neighbors[iy, 0]
        self.ell1 = sup.nl_neighbors[ix, 1]
        self.ell2 = sup.nl_neighbors[iy, 1]


    def get_submat(self, s_arr=np.array([1, 3, 5])):
        Cvec = self.get_Cvec(s_arr)
        if self.sup.gvar.args.use_precomputed:
            arg_str = f"{self.n1}.{self.ell1}-{self.n2}.{self.ell2}"
            Cvec += self.sup.Cvec_pc[arg_str]
        Cmat = np.diag(Cvec)
        submatrix = np.zeros((int(self.sup.dimX_submat[0, self.ix]),
                              int(self.sup.dimY_submat[self.iy, 0])))
        dell = self.ell1 - self.ell2
        absdell = abs(dell)

        if dell > 0:
            submatrix[absdell:-absdell, :] = Cmat
        elif dell < 0:
            submatrix[:, absdell:-absdell] = Cmat
        elif dell == 0:
            submatrix = Cmat

        self.mat = submatrix
        # np.save(f"{self.sup.gvar.datadir}/submatrices/{self.ell1}.{self.ell2}.npy",
                # submatrix)
        return submatrix


    def get_Cvec(self, s_arr):
        ell = min(self.ell1, self.ell2)
        m = np.arange(-ell, ell+1)

        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = w3j_vecm(self.ell1, s_arr[i], self.ell2, -m, 0*m, m)

        Tsr = self.compute_Tsr(s_arr)
        # -1 factor from definition of toroidal field
        '''wsr = np.loadtxt(f'{self.sup.gvar.datadir}/{WFNAME}')\
            [:, self.rmin_idx:self.rmax_idx] * (-1.0)'''
        # self.sup.spline_dict.get_wsr_from_Bspline()
        wsr = self.sup.spline_dict.wsr
        # wsr[0, :] *= 0.0 # setting w1 = 0
        # wsr[1, :] *= 0.0 # setting w3 = 0
        # wsr[2, :] *= 0.0 # setting w5 = 0
        # wsr /= 2.0
        # integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]
        integrand = Tsr * wsr   # since U and V are scaled by sqrt(rho) * r
        LOGGER.debug(" -- Max wsr = {}; Max integrand = {}"\
                     .format(abs(wsr).max(), abs(integrand).max()))
        LOGGER.debug(" -- wsr shape = {}; Tsr shape = {}"\
                     .format(wsr.shape, Tsr.shape))
        integral = simps(integrand, axis=1, x=self.sup.gvar.r)
        prod_gammas = gamma(self.ell1) * gamma(self.ell2) * gamma(s_arr)
        omegaref = self.sup.omegaref
        Cvec = minus1pow_vec(m) * 8*np.pi * omegaref * (wigvals @ (prod_gammas * integral))
        if self.sup.gvar.args.precompute:
            np.save(f"{self.sup.gvar.outdir}/submatrices/" +
                    f"pc.{self.n1}.{self.ell1}-{self.n2}.{self.ell2}.npy", Cvec)
        return Cvec


    def compute_Tsr(self, s_arr):
        Tsr = np.zeros((len(s_arr), len(self.sup.gvar.r)))
        if self.sup.gvar.args.use_precomputed:
            enn1 = self.sup.nl_neighbors[self.ix, 0]
            ell1 = self.sup.nl_neighbors[self.ix, 1]
            enn2 = self.sup.nl_neighbors[self.iy, 0]
            ell2 = self.sup.nl_neighbors[self.iy, 1]
            arg_str1 = f"{enn1}.{ell1}"
            arg_str2 = f"{enn2}.{ell2}"
            U1 = self.sup.eigU[arg_str1]
            U2 = self.sup.eigU[arg_str2]
            V1 = self.sup.eigV[arg_str1]
            V2 = self.sup.eigV[arg_str2]
        else:
            m1idx = self.sup.nl_neighbors_idx[self.ix]
            m2idx = self.sup.nl_neighbors_idx[self.iy]
            U1, V1 = self.get_eig(m1idx)
            U2, V2 = self.get_eig(m2idx)
        L1sq = self.ell1*(self.ell1+1)
        L2sq = self.ell2*(self.ell2+1)
        Om1 = Omega(self.ell1, 0)
        Om2 = Omega(self.ell2, 0)
        for i in range(len(s_arr)):
            s = s_arr[i]
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            wigval = w3j(self.ell1, s, self.ell2, -1, 0, 1)
            Tsr[i, :] = -(1 - minus1pow(self.ell1 + self.ell2 + s)) * \
                Om1 * Om2 * wigval * eigfac / self.sup.gvar.r
            LOGGER.debug(" -- s = {}, eigmax = {}, wigval = {}, Tsrmax = {}"\
                         .format(s, abs(eigfac).max(), wigval, abs(Tsr[i, :]).max()))
        return Tsr


    def get_eig(self, mode_idx):
        LOGGER.debug('Getting eigenfunctions for mode_idx = {}: nl = {}'\
                     .format(mode_idx, self.sup.gvar.nl_all[mode_idx]))
        try:
            U = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                           f'U{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
            V = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                           f'V{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
        except FileNotFoundError:
            LOGGER.info('Mode file not found for mode index = {}'\
                        .format(mode_idx))
            return None
        return U, V

