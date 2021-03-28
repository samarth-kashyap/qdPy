import numpy as np

datadir = "/scratch/g.samarth/qdPy/new_freqs_w135"

data = np.loadtxt("/home/g.samarth/Woodard2013/WoodardPy/HMI/hmi.6328.36")
lendata = data.shape[0]
ells = data[:, 0].astype('int')
enns = data[:, 1].astype('int')

# max_splits = 6
max_splits = 36
sigs = data[:, 48:48+max_splits]

nmax = 7
mask_nmax = enns <= nmax
ells = ells[mask_nmax]
enns = enns[mask_nmax]
sigs = sigs[mask_nmax, :]
lendata = mask_nmax.sum()

split_idxs = np.arange(1, max_splits+1, 2)
#split_idxs = np.arange(1, max_splits+1, 2)

fqd = open("qdpt.w135.36.splits", "w")
fd = open("dpt.w135.36.splits", "w")

for i in range(lendata):
    ell, enn = ells[i], enns[i]
    fname_qdpt = f"qdpt_acoeffs_{enn:02d}_{ell:03d}.npy"
    fname_dpt = f"dpt_acoeffs_{enn:02d}_{ell:03d}.npy"
    mode_found = True
    try:
        ac_qdpt = np.load(f"{datadir}/{fname_qdpt}")
        ac_dpt = np.load(f"{datadir}/{fname_dpt}")
    except FileNotFoundError:
        print(f"file not found: {fname_dpt}")
        print(f"file not found: {fname_qdpt}")
        mode_found = False
        pass
    if mode_found:
        fqd.write("{:2d}  {:3d}  {:20.12e} ".format(ell, enn, ac_qdpt[0]))
        fd.write("{:2d}  {:3d}  {:20.12e} ".format(ell, enn, ac_dpt[0]))
        # fqd.write("{:2d}  {:3d}  {:20.12e}  {:20.12e}  {:20.12e}  {:20.12e}  {:14.7e}  {:14.7e}  {:14.7e}\n"\
        #        .format(ell, enn, ac_qdpt[0], ac_qdpt[1], ac_qdpt[3], ac_qdpt[5],
        #                sigs[i, 0], sigs[i, 2], sigs[i, 4]))
        # fd.write("{:2d}  {:3d}  {:20.12e}  {:20.12e}  {:20.12e}  {:20.12e}  {:14.7e}  {:14.7e}  {:14.7e}\n"\
        #         .format(ell, enn, ac_dpt[0], ac_dpt[1], ac_dpt[3], ac_dpt[5], 
        #                 sigs[i, 0], sigs[i, 2], sigs[i, 4]))
        for ii in split_idxs:
            fqd.write("{:20.12e} ".format(ac_qdpt[ii]))
            fd.write("{:20.12e} ".format(ac_dpt[ii]))
        
        for ii in split_idxs:
            fqd.write("{:20.12e} ".format(sigs[i, ii-1]))
            fd.write("{:20.12e} ".format(sigs[i, ii-1]))
        
        fqd.write("\n")
        fd.write("\n")
fqd.close()
fd.close()
