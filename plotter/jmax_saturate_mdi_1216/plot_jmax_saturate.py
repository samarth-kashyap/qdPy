import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import qdPy.ritzlavely as RL

# {{{ def get_RL_coeffs(rlpObj, delta_omega_nlm):                                             
def get_RL_coeffs(delta_omega_nlm, omega_cnm):
    """Obtain the Ritzwoller-Lavely coefficients for a given                                  
    array of frequencies.                                                                     
                                                                                              
    Inputs:                                                                                   
    -------                                                                                   
    delta_omega_nlm - np.ndarray(ndim=1, dtype=np.float)                                      
        array containing the change in eigenfrequency due to                                  
        rotation perturbation.                                                                
                                                                                              
    Outputs:                                                                                  
    --------                                                                                  
    acoeffs - np.ndarray(ndim=1, dtype=np.float)                                              
        the RL coefficients.                                                                  
    """
    ritz_degree = min(2*l0, 36)
    rlp = RL.ritzLavelyPoly(l0, ritz_degree)
    rlp.get_Pjl()
    acoeffs = rlp.get_coeffs(delta_omega_nlm)
    acoeffs[0] = omega_cnm*OM*1e9
    acoeffs = np.pad(acoeffs, (0, 36-ritz_degree), mode='constant')
    return acoeffs
# }}} get_RL_coeffs(rlpObj, delta_omega_nlm)

def get_cenmult_freqs_qdpt(supmat, omega_cnm, cemult_nbs, jmax):
    """Returns the frequency splitings from qdpt treatment.
    """
    ell_nbs_cropped = cenmult_nbs[np.abs(cenmult_nbs - cenmult_nbs[0]) <= jmax]
    submat_cropped_dim = np.sum(2 * ell_nbs_cropped + 1)
    supmat_cropped = supmat[:submat_cropped_dim, :submat_cropped_dim]
    
    eigvals, eigvecs = sp.linalg.eigh(supmat_cropped)
    
    # sorting eigvals according to the eigvecs
    eigbasis_sort = np.zeros(len(eigvals), dtype=np.int)
    for i in range(len(eigvals)):
        eigbasis_sort[i] = abs(eigvecs[i]).argmax()

    eigvals = eigvals[eigbasis_sort].real
    
    return eigvals[:2*cenmult_nbs[0]+1]/2./omega_cnm

def plot_acoeff_saturation(ac_dpt, ac_qdpt, ac_sigma):    
    # extracting the odd a-coefficients
    ac_odd_dpt = np.zeros((ac_dpt.shape[0], ac_dpt.shape[1]//2))
    ac_odd_qdpt = np.zeros((ac_dpt.shape[0], ac_dpt.shape[1]//2))
    for i in range(len(ac_dpt)):
        ac_odd_dpt[i] = ac_dpt[i, ::2]
        ac_odd_qdpt[i] = ac_qdpt[i, ::2]
    
    ac_odd_diff_relsigma = (ac_odd_qdpt - ac_odd_dpt)/ac_sigma[::2]

    # extracting the even a-coefficients
    ac_even_dpt = np.zeros((ac_dpt.shape[0], ac_dpt.shape[1]//2))
    ac_even_qdpt = np.zeros((ac_dpt.shape[0], ac_dpt.shape[1]//2))
    for i in range(len(ac_dpt)):
        ac_even_dpt[i] = ac_dpt[i, 1::2]
        ac_even_qdpt[i] = ac_qdpt[i, 1::2]
    
    ac_even_diff_relsigma = (ac_even_qdpt - ac_even_dpt)/ac_sigma[1::2]

    # plotting the odd a-coefficients
    fig, ax = plt.subplots(1, 1)

    for i in range(len(ac_odd_dpt)):
        jmin, jmax = 2*i + 1, 2*len(ac_odd_dpt)
        j_axis = np.arange(jmin, jmax+1, 2)

        plt.plot(j_axis, ac_odd_diff_relsigma[i:, i], label='$a_{%i}$'%(jmin))

    plt.legend()
    plt.savefig('odd_acoeff_saturation.pdf')
    plt.close()

    # plotting the even a-coefficients
    fig, ax = plt.subplots(1, 1)
    
    for i in range(len(ac_even_dpt)-1):
        jmin, jmax = 2*(i+1), 2*len(ac_even_dpt)
        j_axis = np.arange(jmin, jmax+1, 2)

        plt.plot(j_axis, ac_even_diff_relsigma[i:, i], label='$a_{%i}$'%(jmin))
    
    plt.legend()
    plt.savefig('even_acoeff_saturation.pdf')
    plt.close()
    
    return ac_odd_dpt, ac_odd_qdpt, ac_even_dpt, ac_even_qdpt

def get_cenmult_freqs_dpt(supmat, omega_cnm, cenmult_nbs, jmax):
    """Returns the frequency splittings from dpt treatment.
    """
    ell_nbs_cropped = cenmult_nbs[np.abs(cenmult_nbs - cenmult_nbs[0]) <= jmax]
    submat_cropped_dim = np.sum(2 * ell_nbs_cropped + 1)
    supmat_cropped = supmat[:submat_cropped_dim, :submat_cropped_dim]
        
    eigvals_dpt = np.diag(supmat_cropped)
    
    return eigvals_dpt[:2*cenmult_nbs[0]+1]/2./omega_cnm 

if __name__=="__main__":
    l0 = 280
    supmat = np.load('supmat_qdpt_00.280.19.150.npy').real
    omega0 = np.load('cenmult_omega0.npy')
    cenmult_nbs = np.load('cenmult_nbs_00.280.19.150.npy')[:,1]
    ac_sigma = np.load('ac_sigma.mdi.1216.npy')

    M_sol = 1.989e33 #gn,l = 0,200                                                    
    R_sol = 6.956e10 #cm                                                              
    B_0 = 10e5 #G 
    OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

                      
    max_jmax = cenmult_nbs.shape[0]

    ac_dpt = np.zeros((max_jmax//2 + 1, 37))
    ac_qdpt = np.zeros((max_jmax//2 + 1, 37))
                          
    for jmax in range(1, max_jmax+1, 2):
        print(jmax)                  
        
        # in non-dimensional units
        fqdpt = get_cenmult_freqs_qdpt(supmat, omega0, cenmult_nbs, jmax)
        fdpt = get_cenmult_freqs_dpt(supmat, omega0, cenmult_nbs, jmax)
    
        # in nHz
        fdpt *= OM * 1e9
        fqdpt *= OM * 1e9
        
        # converting freqency splittings to a-coefficients
        ac_dpt[jmax//2] = get_RL_coeffs(fdpt, omega0)
        ac_qdpt[jmax//2] = get_RL_coeffs(fqdpt, omega0)

    # rejecting the j=0 component
    ac_dpt = ac_dpt[:, 1:]
    ac_qdpt = ac_qdpt[:, 1:]
        
    ac_odd_dpt, ac_odd_qdpt, ac_even_dpt, ac_even_qdpt =\
                            plot_acoeff_saturation(ac_dpt, ac_qdpt, ac_sigma)
