import os
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
snrnmais_dir = dirnames[2]

lmin, lmax = 10, 290

hmidata = np.loadtxt(f"{scratch_dir}/data_files/hmi.6328.36")
ell_list = hmidata[:, 0].astype('int')
enn_list = hmidata[:, 1].astype('int')
omg_list = hmidata[:, 2]

mask_lmin_lmax = (ell_list <= lmax) * (ell_list >= lmin)
ell_list = ell_list[mask_lmin_lmax]
enn_list = enn_list[mask_lmin_lmax]
omg_list = omg_list[mask_lmin_lmax]
np.savetxt(f"{scratch_dir}/data_files/nlist.txt", enn_list, fmt="%d")
np.savetxt(f"{scratch_dir}/data_files/llist.txt", ell_list, fmt="%d")
np.savetxt(f"{scratch_dir}/data_files/omegalist.txt", omg_list)


