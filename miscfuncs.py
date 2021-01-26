def nearest_idx(arr, val):
    return abs(arr - val).argmin()


def mask_minmax(arr, amin, amax):
    minidx = nearest_idx(arr - amin)
    maxidx = nearest_idx(arr - amax)
    idx = (minidx, maxidx)
    return idx, arr[minidx:maxidx]


def nl_idx(n0, l0):
    try:
        idx = nl_all_list.index([n0, l0])
    except ValueError:
        idx = None
        logger.error('Mode not found')
    return None



class qdptMode():
    def __init__(self, gvar, n0, l0, smax, freq_window):
        self.gvar = gvar
        self.n0 = n0
        self.l0 = l0
        self.smax = smax
        self.freq_window = freq_window

        r = np.loadtxt(f'{datadir}/r.dat')
        minmax_idx, r = mask_minmax(r, r_start, r_end)
        rmin_idx, rmax_idx = idxs
        self.r = r
        self.rmin_idx = rmin_idx
        self.rmax_idx = rmax_idx
        self.omega0 = omega_list[nl_idx(self.n0, self.l0)]
        return self

    def mode_neighbors():
