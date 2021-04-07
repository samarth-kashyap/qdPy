import numpy as np
import wigner
import py3nj
from scipy.integrate import simps
# from scipy.integrate import trapz as simps 


NAX = np.newaxis


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


class qdptMode():
    def __init__(self, gvar):
        self.gvar = gvar
        self.n0 = gvar.n0
        self.l0 = gvar.l0
        self.smax = gvar.smax
        self.freq_window = gvar.fwindow

        self.idx = self.nl_idx(self.n0, self.l0)
        self.omega0 = self.gvar.omega_list[self.idx]
        self.get_mode_neighbors_params()

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
        mask_nb = mask_omega * mask_ell
        sort_idx = np.argsort(abs(omega_diff[mask_nb]))
        self.nl_neighbors = nl_all[mask_nb][sort_idx]
        self.nl_neighbors_idx = self.nl_idx_vec(self.nl_neighbors)
        self.omega_neighbors = self.get_omega_neighbors(self.nl_neighbors_idx)
        self.num_neighbors = len(self.nl_neighbors_idx)
        print(f"Found {self.num_neighbors} neighbors to mode" +
              f" (n, ell) = ({self.n0}, {self.l0})")

    def create_supermatrix(self):
        supmat = superMatrix(self.gvar, self.num_neighbors,
                             self.nl_neighbors, self.nl_neighbors_idx,
                             self.omega0, self.omega_neighbors)
        return supmat


class superMatrix():
    def __init__(self, gvar, dim, nl_neighbors, nl_neighbors_idx, omegaref,
                 omega_neighbors):
        self.gvar = gvar
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
        self.fill_supermatrix()
        self.supmat_dpt = np.diag(np.diag(self.supmat))
        self.fill_supermatrix_freqdiag()

    def fill_supermatrix(self):
        print(f"Creating submatrices for: ")
        for i in range(self.dim_blocks):
            for ii in range(i, self.dim_blocks):
                sm = subMatrix(i, ii, self, printinfo=True)
                submat = sm.get_submat()
                self.supmat[sm.startx:sm.endx,
                            sm.starty:sm.endy] = submat
                self.supmat[sm.starty:sm.endy,
                            sm.startx:sm.endx] = submat.T.conj()

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
            eigvals_all, eigvecs = np.linalg.eigh(self.supmat)
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
        printstr = (f"--- (n1, l1) = " +
                    f"({sup.nl_neighbors[ix, 0]}, {sup.nl_neighbors[ix, 1]})" +
                    f" and (n2, l2) = " +
                    f"({sup.nl_neighbors[iy, 0]}, {sup.nl_neighbors[iy, 1]})")
        if printinfo:
            print(printstr)
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
        np.save(f"{self.sup.gvar.datadir}/submatrices/{self.ell1}.{self.ell2}.npy",
                submatrix)
        return submatrix

    def get_Cvec(self, s_arr):
        ell = min(self.ell1, self.ell2)
        m = np.arange(-ell, ell+1)

        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = w3j_vecm(self.ell1, s_arr[i], self.ell2, -m, 0*m, m)

        Tsr = self.compute_Tsr(s_arr)
        fname = 'w.dat'
        # fname = 'w_const.dat'
        # fname = 'w_const_430.dat'
        print(f"Using velocity profile - {fname}")
        # fname = 'w_jesper.dat'  # to match with jesper's data
        # -1 factor from definition of toroidal field
        wsr = np.loadtxt(f'{self.sup.gvar.datadir}/{fname}')\
            [:, self.rmin_idx:self.rmax_idx] * (-1.0)
        # wsr[0, :] *= 0.0 # setting w1 = 0
        # wsr[1, :] *= 0.0 # setting w3 = 0
        # wsr[2, :] *= 0.0 # setting w5 = 0
        wsr /= 2.0
        integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]
        integral = simps(integrand, axis=1, x=self.sup.gvar.r)
        prod_gammas = gamma(self.ell1) * gamma(self.ell2) * gamma(s_arr)
        omegaref = self.sup.omegaref
        Cvec = minus1pow_vec(m) * 8*np.pi * omegaref * (wigvals @ (prod_gammas * integral))
        return Cvec


    def compute_Tsr(self, s_arr):
        Tsr = np.zeros((len(s_arr), len(self.sup.gvar.r)))
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
        return Tsr


    def get_eig(self, mode_idx):
        # print(f'Getting eigenfunctions for mode_idx = {mode_idx}:' +
              # f'nl = {self.sup.gvar.nl_all[mode_idx]}')
        try:
            U = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                           f'U{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
            V = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                           f'V{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
        except FileNotFoundError:
            print('Mode file not found for mode index = {mode_idx}')
            return None
        return U, V

