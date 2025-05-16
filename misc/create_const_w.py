import numpy as np
import globalvars
import argparse

# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order", type=int)
parser.add_argument("--l0", help="angular degree", type=int)
parser.add_argument("--omega", help="constant rotation rate (nHz)",
                    type=np.float32)
args = parser.parse_args()
# }}} argparse


# {{{ Reading global variables
rmin, rmax = 0.0, 1.0
SMAX = 5        # maximum s for constructing supermatrix
FWINDOW = 150   # microHz
gvar = globalvars.globalVars(rmin, rmax, SMAX, FWINDOW, args)
# }}} global variables


OMEGA0_muhz = args.omega * 1e-3
w1 = np.sqrt(4*np.pi/3.) * OMEGA0_muhz * 1e-6 / gvar.OM
r = np.loadtxt("/scratch/g.samarth/qdPy/r.dat")
w135 = np.zeros((3, len(r)), dtype=np.float32)
w135[0, :] = r*w1
np.savetxt(f"/scratch/g.samarth/qdPy/w_const_{int(args.omega)}.dat", w135)
