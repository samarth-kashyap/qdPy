import numpy as np
import sys
sys.path.append('/home/g.samarth/qdPy/')
sys.path.append('/home/g.samarth/Solar_Eigen_function/')
import ritzlavely as RL
import functions as fn
import time

ell = 100
jmax = 36

t1 = time.time()
print("Computing old polynomial")
old_poly = np.zeros((jmax, 2*ell+1))
for j in range(jmax):
    temp_poly = fn.P_a(ell, j)
    old_poly[j, :] = temp_poly
t2 = time.time()
print(f"-- time taken = {(t2-t1):5.3e} seconds")

t1 = time.time()
print("Computing new polynomial")
rlp = RL.ritzLavelyPoly(ell, jmax)
new_poly = rlp.get_Pjl()
t2 = time.time()
print(f"-- time taken = {(t2-t1):5.3e} seconds")

print(f" sum(abs(diff)) = {abs(new_poly - old_poly).sum()}")
print(f" avg_error = {abs(new_poly - old_poly).sum()/new_poly.size}")
