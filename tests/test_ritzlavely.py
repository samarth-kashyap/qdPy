import numpy as np
import sys
sys.path.append('/home/g.samarth/Woodard2013/WoodardPy/')
sys.path.append('/home/g.samarth/qdPy/')
from helioPy import datafuncs as cdata
import ritzlavely as RL
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval 

def get_old_Pjl():
    L = rpl.L
    Pjl_old = np.zeros((rpl.jmax, len(marr)))
    for i in range(len(marr)):
        m = marr[i]
        for ii in range(jmax):
            coeffs = np.zeros(ii+1)
            coeffs[-1] = 1
            Pjl_old[ii, i] = legval(m/L, coeffs) * L
    return Pjl_old

def compare_oldnew(n):
    plt.cla()
    plt.plot(Pjl_old[n, :], 'b', alpha=0.6, label='Legpoly')
    plt.plot(rpl.Pjl[n, :], 'r', alpha=0.6, label='RitzLavely')
    plt.legend()
    plt.show()
    return None

data = np.loadtxt('/home/g.samarth/Woodard2013/WoodardPy/HMI/hmi.6328.36')
ell = 250
n = 0
marr = np.arange(-ell, ell+1)*1.0
jmax = 36
rpl = RL.ritzLavelyPoly(ell, jmax)
rpl.get_Pjl()


mode_idx = np.where((data[:, 0]==ell)*(data[:, 1]==n))[0][0]
nu = data[mode_idx, 2]
splits = np.append([0.0], data[mode_idx, 12:48])

freqs_old, __, __ = cdata.findfreq_vec(data, ell*np.ones_like(marr),
                                       n, marr)
freqs_old -= nu
freqs_new = rpl.polyval(splits*0.001)
error_percent = (freqs_new - freqs_old)*100./freqs_old

Pjl_old = get_old_Pjl()

print(f"Max error = {abs(error_percent).max():5.3f}%")

plt.figure(figsize=(5, 5))
plt.plot(freqs_old, 'b')
plt.plot(freqs_new, 'r')
plt.plot(freqs_new - freqs_old, 'gray')
plt.show()
