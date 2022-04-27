import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# arrays for smax and fwindow                                                            
jmax = 11
smax_arr = np.arange(1, jmax+1, 2)
fwindow = 150  # in muHz

# loading the a-coefficients for (0, 280) only odd acoeffs
acoeffs_sig = np.array([0.0311663, 0.0547268, 0.0655665, 0.0728381, 0.0787113, 0.0837615,
                        0.0875416, 0.0887825, 0.0882349, 0.088512 , 0.0863005, 0.0827442,
                        0.0782114, 0.0743386, 0.070113 , 0.0640645, 0.0577018, 0.0487467])
# taking upto jmax
acoeffs_sig = acoeffs_sig[:jmax//2+1]


def get_acoeffs_refdiff(tag):
    # directory where the results are stored                                                 
    acoeffs_dir = f'qdpt280_{tag}'
        
    # array to store the relative difference w.r.t acoeffs sigma                             
    acoeffs_reldiff = np.zeros((len(smax_arr), len(smax_arr)))
                       
    # loading the largest fwindow and largest smax case to be used as reference               
    acoeffs_ref = np.load(f'{acoeffs_dir}/qdpt-ac-00.280.{smax_arr.max()}.{fwindow}.npy')
    acoeffs_ref = acoeffs_ref[1:jmax+1:2] # only picking out the odd a-coeffs                 
    
    for i, smax in enumerate(smax_arr):
        acoeffs_arr = np.load(f'{acoeffs_dir}/qdpt-ac-00.280.{smax}.{fwindow}.npy')
        acoeffs_arr = acoeffs_arr[1:jmax+1:2] # only picking out the odd a-coeffs          
        
        # only picking out the odd a-coefficients                                          
        d_acoeffs_arr = acoeffs_arr - acoeffs_ref
                
        acoeffs_reldiff[i] = np.abs(d_acoeffs_arr/acoeffs_sig)
        
    return acoeffs_reldiff

def plot_lines(tag):
    acoeffs_reldiff = get_acoeffs_refdiff(tag)
    
    plt.figure()
    for i, smax in enumerate(smax_arr):
        plt.semilogy(smax_arr[:i+1], acoeffs_reldiff[i, :i+1], label=f'{smax}') 

    plt.legend()
    plt.savefig(f'lineplot_{tag}.pdf')
    plt.close()

def plot_2D(tag):
    acoeffs_reldiff = get_acoeffs_refdiff(tag)
    
    fig, ax = plt.subplots(1,1)
    z = np.zeros_like(acoeffs_reldiff)
    for i, smax in enumerate(smax_arr):
        z[i, :i+1] = acoeffs_reldiff[i, :i+1]

    # excluding smax=19
    z = z[:jmax//2, :jmax//2]
    
    print(z.shape, acoeffs_reldiff.shape, smax_arr[:jmax//2])

    # defining custom colormap
    divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=0.2)

    im = ax.pcolormesh(smax_arr[:jmax//2], smax_arr[:jmax//2], np.log10(z),
                       norm=divnorm, cmap='seismic')
    ax.set_xticks(smax_arr[:jmax//2])
    ax.set_yticks(smax_arr[:jmax//2])
    fig.colorbar(im)
    plt.savefig(f'2D_plot_{tag}.pdf')
    plt.close()


def plot_2D_minima_maxima_MDI_HMI():
    instr = np.array(['mdi', 'hmi'])
    day_tags = np.array([[1216, 3232],
                         [9856, 7768]])

    dates_arr = np.array([['1996-05-01', '2001-11-07'],
                          ['2019-12-27', '2014-04-09']])
    
    # making same instrument for one row. 
    # First row is MDI and second row is HMI
    # First column is solar minima
    # Second column is solar maxima

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    for row in range(2):
        for col in range(2):
            if(col == 0):
                minmax = 'minima'
            else:
                minmax = 'maxima'
            
            tag = f'{instr[row]}_{minmax}_{day_tags[row, col]}'

            acoeffs_reldiff = get_acoeffs_refdiff(tag)
            
            z = np.zeros_like(acoeffs_reldiff)
            for i, smax in enumerate(smax_arr):
                z[i, :i+1] = acoeffs_reldiff[i, :i+1]
                
            # excluding smax=19                                                           
            z = z[:jmax//2, :jmax//2]
            
            # defining custom colormap                                                   
            divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=0.2)
            
            im = ax[row,col].pcolormesh(smax_arr[:jmax//2],
                                        smax_arr[:jmax//2], np.log10(z),
                                        norm=divnorm, cmap='seismic')
            ax[row,col].set_aspect('equal')

            ax[row,col].text(12, 2, dates_arr[row,col], fontsize=16)
            
            ax[row,col].set_xticks(smax_arr[:jmax//2])
            ax[row,col].set_yticks(smax_arr[:jmax//2])

            if(row==0 and col==0):
                ax[row,col].set_title('Solar minima', fontsize=18)
            if(row==0 and col==1):
                ax[row,col].set_title('Solar maxima', fontsize=18)
            if(row==0 and col==0):
                ax[row,col].set_ylabel('MDI', fontsize=18)
            if(row==1 and col==0):
                ax[row,col].set_ylabel('HMI', fontsize=18)
            
    cax = fig.add_axes([ax[1,1].get_position().x1 + 0.05, ax[1,1].get_position().y0,
                        0.02, ax[0,1].get_position().y1-ax[1,1].get_position().y0])
    fig.colorbar(im, ax=ax.ravel().tolist(), cax=cax)
            
    plt.subplots_adjust(left=0.05,
                        bottom=0.05,
                        right=0.93,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)

    plt.savefig(f'2D_plot_minmax.pdf')
    plt.close()
            
            
tag = 'new'
plot_lines(tag)
plot_2D(tag)
# plot_2D_minima_maxima_MDI_HMI()
