import os
import subprocess
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

#----------------- getting full pythonpath -----------------------#                       
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------#

instrument_arr = np.array(['hmi', 'hmi', 'mdi', 'mdi'])
daynum_arr = np.array([7768, 9856, 1216, 3232])

n0, l0 = 0, 280

# frequency window in muHz
fwindow = 150

# j array
smax_wsr = 19
smax_wsr_arr = np.arange(1, smax_wsr+1, 2)

with open(f"{current_dir}/qdpyjobs.sh", "w") as f:
    for instr_daynum_ind in range(4):
        instrument = instrument_arr[instr_daynum_ind]
        daynum = daynum_arr[instr_daynum_ind]
        for smax_wsr_val in smax_wsr_arr:
            f.write(f"{pythonpath} qdpt.py --n0 {n0} --l0 {l0} --smax_wsr {smax_wsr_val} "+
                    f"--fwindow {fwindow} --smax {smax_wsr} "+
                    f"--instr {instrument} --daynum {daynum}\n")
        
os.system(f"chmod u+x {current_dir}/qdpyjobs.sh")
