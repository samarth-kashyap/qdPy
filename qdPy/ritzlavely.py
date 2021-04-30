"""Class to handle Ritzwoller-Lavely polynomials"""
import logging
import numpy as np
import scipy.special as special
import numpy.polynomial.legendre as npleg
import qdPy.functions as FN

NAX = np.newaxis
LOGGER = FN.create_logger(__name__, 'logs/qdpt.log', logging.INFO)

class ritzLavelyPoly():
    def __init__(self, ell, jmax):
        assert ell > 0, "Ritzwoller-Lavely polynomials don't exist for ell=0"
        assert jmax + 1 <= 2*ell, "Max degree (jmax) should be smaller than 2*ell"
        self.ell = ell
        self.jmax = jmax + 1
        self.m = np.arange(-ell, ell+1) * 1.0
        self.L = np.sqrt(ell*(ell+1))
        self.m_by_L = self.m/self.L
        self.Pjl = np.zeros((self.jmax, len(self.m)), dtype=np.float64)
        self.Pjl_exists = False

    def get_Pjl(self):
        if self.Pjl_exists:
            LOGGER.info('Ritzwoller-Lavely polynomials already computed')
            return self.Pjl
        else:
            self.Pjl[0, :] += self.ell
            self.Pjl[1, :] += self.m
            for j in range(2, self.jmax):
                coeffs = np.zeros(j+1)
                coeffs[-1] = 1.0
                P2j = self.L * npleg.legval(self.m_by_L, coeffs)
                cj = self.Pjl[:j, :] @ P2j / (self.Pjl[:j, :]**2).sum(axis=1)
                P1j = P2j - (cj[:, NAX] * self.Pjl[:j, :]).sum(axis=0)
                self.Pjl[j, :] += self.ell * P1j/P1j[-1]
            self.Pjl_exists = True
            # self.normalize_Pjl()
            return self.Pjl

    def normalize_Pjl(self):
        norms = np.zeros(self.Pjl.shape[0])
        for i in range(len(norms)):
            norms[i] = np.sqrt(self.Pjl[i, :] @ self.Pjl[i, :])
        self.Pjl = self.Pjl / norms[:, NAX]

    def get_coeffs(self, arrm):
        if not self.Pjl_exists:
            self.get_Pjl()
        assert len(arrm) == len(self.m), "Length of input array =/= 2*ell+1"
        aj = (self.Pjl @ arrm) / np.diag(self.Pjl @ self.Pjl.T)
        return aj

    def polyval(self, acoeffs):
        assert len(acoeffs) == self.jmax, (f"Number of coeffs ({len(acoeffs)} " +
                                           f"=/= jmax ({self.jmax})")
        return (acoeffs[:, NAX] * self.Pjl).sum(axis=0)
