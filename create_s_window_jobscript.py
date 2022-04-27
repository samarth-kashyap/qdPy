import os
import subprocess
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

#----------------- getting full pythonpath -----------------------#                       
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------#

n0, l0 = 0, 280

# frequency window in muHz
fwindow = 150

# j array
smax = 11
smax_arr = np.arange(1, smax+1, 2)

with open(f"{current_dir}/qdpyjobs.sh", "w") as f:
    for smax in smax_arr:
        f.write(f"{pythonpath} qdpt.py --n0 {n0} --l0 {l0} --smax {smax} "+
                f"--fwindow {fwindow}\n")
        
os.system(f"chmod u+x {current_dir}/qdpyjobs.sh")
