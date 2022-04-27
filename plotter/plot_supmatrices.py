import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

supmat_dir = '../supmat_dir'

for i in range(11, 0, -2):
    print(f'Loading the largest supermatrix for s = {i}')
    supmat = np.load(f'{supmat_dir}/supmat_qdpt.280.{i}.150.0.npy').real
    
    if(i == 11):
        supmat_largedim = supmat.shape[0]
        z = supmat
    else:
        z = np.zeros((supmat_largedim, supmat_largedim))
        supmat_dim = supmat.shape[0]
        z[:supmat_dim, :supmat_dim] = supmat
    
    #just something dummy to plot the whole matrix
    z[-1,-1] = 1
        
    sp = sparse.coo_matrix(z)
    rows, cols, data = sp.row, sp.col, sp.data
    
    fig, ax = plt.subplots(1, 1, figsize=(12,10))
    
    im = ax.scatter(rows, cols, c=data, s=1e-3, cmap='seismic',
                    norm = colors.SymLogNorm(linthresh=1e-1,
                                             linscale=1e-1,
                                             vmin = 561,
                                             vmax = -561))
                     # vmin=-8, vmax=3.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.5)
                     
    fig.colorbar(im, cax=cax)
    
    plt.savefig(f'supmat_{i}.png')
    
