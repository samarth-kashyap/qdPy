import os
import subprocess
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
hmidata_path = '/scratch/gpfs/sbdas/Helioseismology/get-solar-eigs/efs_Jesper/snrnmais_files/data_files'

hmidata = np.loadtxt(f'{hmidata_path}/hmi.6328.36')

#----------------- getting full pythonpath -----------------------#                       
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------#

nl_arr = hmidata[:,:2].astype('int')

# frequency window in muHz
fwindow = 150

# j array
smax_wsr_val = 11
smax_val = 11

instrument = 'hmi'
daynum = 7768

with open(f"{current_dir}/qdpyjobs_allmodes.sh", "w") as f:
    for mult_ind in range(len(nl_arr)):
        l0, n0 = nl_arr[mult_ind]
        
        f.write(f"{pythonpath} qdpt.py --n0 {n0} --l0 {l0} --smax_wsr {smax_wsr_val} "+
                f"--fwindow {fwindow} --smax {smax_val} "+
                f"--instr {instrument} --daynum {daynum}\n")
        
os.system(f"chmod u+x {current_dir}/qdpyjobs_allmodes.sh")
