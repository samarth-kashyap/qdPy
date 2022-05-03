import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

# arrays for smax and fwindow                                                            
jmax = 19
smax_arr_all = np.arange(1, jmax+1)
fwindow = 150  # in muHz

# loading the 360d sigmas. Contains from [a_1, a_2, ....]
acoeffs_sig_7768 = np.load('ac_sigma.hmi.7768.npy')
acoeffs_sig_6328 = np.load('ac_sigma.hmi.6328.npy')

def get_acoeffs_refdiff(tag, daynum, odd_even):
    # directory where the results are stored                                                 
    acoeffs_dir = f'../qdpt280_{tag}'
                               
    
    if(daynum == 7768):
        acoeffs_dir = '/home/sbdas/Research/Helioseismology/qdPy/plotter/jmax_saturate_hmi_7768'
    
    # loading the largest fwindow and largest smax case to be used as reference               
    acoeffs_ref = np.load(f'{acoeffs_dir}/dpt-ac-00.280.{smax_arr_all.max()}.{fwindow}.'+
                          f'hmi.{daynum}.npy')
    
    # choosing the starting and ending indices according to odd or even acoeffs
    if(odd_even == 'odd'):
        smax_arr = smax_arr_all[::2]
        start_ind, end_ind = 1, jmax+1
        # extracting the acoeff_sig for either odd
        if(daynum == 7768):
            acoeffs_sig = acoeffs_sig_7768[:jmax+1:2]
        else:
            acoeffs_sig = acoeffs_sig_6328[:jmax+1:2]

    else:
        smax_arr = smax_arr_all[1::2]
        start_ind, end_ind = 2, jmax
        # extracting the acoeff_sig for either odd                                            
        if(daynum == 7768):
            acoeffs_sig = acoeffs_sig_7768[1:jmax:2]
        # or even
        else:
            acoeffs_sig = acoeffs_sig_6328[1:jmax:2]
    
    # this is always 1,3,..,19
    smax_arr_wsr = smax_arr_all[::2]
    
    # array to store the relative difference w.r.t acoeffs sigma                         
    acoeffs_reldiff = np.zeros((len(smax_arr_wsr), len(smax_arr)))
    
    acoeffs_ref = acoeffs_ref[start_ind:end_ind:2]
    
    for i, smax in enumerate(smax_arr_wsr):
        acoeffs_arr = np.load(f'{acoeffs_dir}/qdpt-ac-00.280.{smax}.{fwindow}.'+
                              f'hmi.{daynum}.npy') + 1e-12
        acoeffs_arr = acoeffs_arr[start_ind:end_ind:2]
        
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
    day_tags = np.array([9856, 7768])

    dates_arr = np.array(['2019-12-27', '2014-04-09'])
    
    odd_even_arr = np.array(['odd', 'even'])
    
    # making same instrument for one row. 
    # First row is MDI and second row is HMI
    # First column is solar minima
    # Second column is solar maxima

    fig, ax = plt.subplots(2, 2, figsize=(14,13))

    for row, odd_even in enumerate(odd_even_arr):
        if(row==0):
            smax_arr = smax_arr_all[::2]
        else:
            smax_arr = smax_arr_all[1::2]
            
        # the x-label indicative of smax_wsr
        smax_arr_wsr = smax_arr_all[::2]
        
        for col in range(2):
            if(col == 0):
                minmax = 'minima'
            else:
                minmax = 'maxima'
                
            tag = f'hmi_{minmax}_{day_tags[col]}'
            
            acoeffs_reldiff = get_acoeffs_refdiff(tag, day_tags[col], odd_even)
            
            z = np.zeros_like(acoeffs_reldiff)
            '''
            for i, smax in enumerate(smax_arr_wsr):
                z[i, :i+1] = acoeffs_reldiff[i, :i+1]
            '''
            z = acoeffs_reldiff
            # defining custom colormap                                                   
            divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1)

            print(smax_arr_wsr[:jmax//2+1], smax_arr[:jmax//2+1], z.shape)

            im = ax[row,col].pcolormesh(smax_arr_wsr, smax_arr, np.log10(z.T),
                                        norm=divnorm, cmap='seismic')
            ax[row,col].set_aspect('equal')
            
            ax[row,col].set_xticks(smax_arr_wsr[:jmax//2+1])
            ax[row,col].set_yticks(smax_arr[:jmax//2+1])
            
            if((row==0) * (col==0)):
                ax[row,col].set_title(f'Solar minima: {dates_arr[col]}', fontsize=18)
                ax[row,col].set_ylabel(f'{odd_even} $a$-coefficients', fontsize=18)
            if((row==0) * (col==1)):
                ax[row,col].set_title(f'Solar maxima: {dates_arr[col]}', fontsize=18)
            if((row==1) * (col==0)):
                ax[row,col].set_ylabel(f'{odd_even} $a$-coefficients', fontsize=18)
                
            ax[row,col].set_xlabel('$s_{\mathrm{max}}$ for non-zero $w_s(r)$',
                                   fontsize=18)
                
    cax = fig.add_axes([ax[1,1].get_position().x1 + 0.05, ax[1,1].get_position().y0,
                        0.02, ax[0,1].get_position().y1-ax[1,1].get_position().y0])
    fig.colorbar(im, ax=ax[row,col], cax=cax)
                
    plt.subplots_adjust(left=0.05,
                        bottom=0.05,
                        right=0.93,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)

    plt.savefig(f'2D_plot_minmax.pdf')
    plt.close()
            
'''            
tag = 'hmi_minima_9856'
plot_lines(tag)
plot_2D(tag)
'''
plot_2D_minima_maxima_MDI_HMI()
