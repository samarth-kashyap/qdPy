import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', help='Radial order', type=int)
parser.add_argument('--l', help='Spherical harmonic degree', type=int)
args = parser.parse_args()

def print_coeffs(coefs):
#    coefs = coefs / abs(coefs).max()
    print(f" Difference = \sum_i (a_i m^i) ")
    for i in range(len(coefs)):
        contrib = coefs[i]*(args.l**i)
        print(f" a_{i} = {coefs[i]:10.3e};    " +
              f" a_{i}*{args.l:02d}^{i} = {contrib:10.3e}")
    return None


def plot_both(enn, ell):
    nl_str = f' (n = {enn}, $\\ell$ = {ell})'
    qdpt = np.load(f"{cwd}/qdpt_{enn:02d}_{ell:03d}.npy")
    dpt = np.load(f"{cwd}/dpt_{enn:02d}_{ell:03d}.npy")
    marr = np.arange(-ell, ell+1)
    fig, axs = plt.subplots(figsize=(5, 7), nrows=2, ncols=1)
    axs[0].plot(marr, qdpt.real, 'k', label='QDPT', alpha=0.8)
    axs[0].plot(marr, dpt.real, '--b', label='DPT', alpha=0.8)
    axs[0].set_xlabel('$m$')
    axs[0].set_title('$\\nu_{n\\ell m}$ in $\\mu$Hz' + nl_str)
    axs[0].legend()
    
    diffarr = (qdpt.real - dpt.real)
    fitpoly_coeffs = np.polyfit(marr, diffarr, 4)
    diff_fit = np.polyval(fitpoly_coeffs, marr)
    print_coeffs(fitpoly_coeffs[::-1])

    axs[1].plot(marr, diffarr, '+r', label='Computed')
    axs[1].plot(marr, diff_fit, 'k', label='Polynomial fit')
    axs[1].plot(marr, diffarr - diff_fit, 'b', label='Residue', alpha=0.5)
    axs[1].legend()
    axs[1].set_title('$\\nu_{n\\ell m}^{QDPT} - \\nu_{n\ell m}^{DPT}$' +
                     '  in $\\mu$Hz' + nl_str)
    # axs[1].set_title('$\\left( \\frac{\\nu_{nlm}^{QDPT} - \\nu_{nlm}^{DPT}}{\\nu_{nlm}^{DPT}} X 100\\right) $% in in $\\mu$Hz' +
    #                  nl_str)
    axs[0].set_xlabel('$m$')
    fig.tight_layout()
#    plt.show(fig)
    return (qdpt, dpt), (diffarr, diff_fit), fig


if __name__ == "__main__":
    cwd = "/scratch/g.samarth/Solar_Eigen_function/new_freqs_full"
    enn, ell = args.n, args.l
    freqs, diffs, fig = plot_both(enn, ell)
    qdpt, dpt = freqs
    diffarr, diff_fit = diffs
    figname = f"{cwd}/qdpt_dpt_{enn:02d}_{ell:03d}.pdf"
    fig.savefig(figname)

