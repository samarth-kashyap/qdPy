import numpy as np
from ritzlavelyPy import rlclass as RLP

cenfreq = 1500
ell = 200
marr = np.arange(-ell, ell+1)
rot_rate = 0.43
eigval_supmat = 2*marr*rot_rate*cenfreq

freq1 = cenfreq + eigval_supmat/2/cenfreq
freq2 = np.sqrt(cenfreq**2 + eigval_supmat)

rl = RLP.ritzLavelyPoly(ell, 5)
rl.get_Pjl()
ac1 = rl.get_coeffs(freq1)
ac2 = rl.get_coeffs(freq2)

for i in range(6):
    print(f"a{i} = {ac1[i]:12.3e} {ac2[i]:12.3e}")
