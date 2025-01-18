'''
calculate the magnon spectrum of Heisenberg model that shown
in paper
Rinaldi M, Mrovec M, Bochkarev A, et al. 
Non-collinear magnetic atomic cluster expansion for iron[J]. 
npj Computational Materials, 2024, 10(1): 12.
, which reads

E_i(q) = sum_j [J_ij * (1 - cos(q * R_ij))]

, in which the i denotes the site index, j denotes
neighboring sites, J_ij is the exchange constant,
q is the wave vector, and R_ij is the distance
between site i and j.
'''
# built-in modules
import unittest
import numpy as np
import matplotlib.pyplot as plt

# home-made modules
from structure import neighborgen, qpathgen
from utils import init

# set the default font as Arial and font size as 16
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams.update({'font.size': 16})

def cal_ener_bilinear_heisenberg(exch_const, q, pos) -> float:
    '''
    calculate the energy of Heisenberg model at wave vector q
    
    Parameters
    ----------
    exch_const : list of float
        exchange constant between atoms for each shell
    q : np.array
        wave vector in shape of (3,)
    pos : np.array
        positions of atoms or displacement respect to the origin, in shape
        of (n, 3), n is the number of atoms
    '''
    if len(exch_const) != len(pos):
        raise ValueError('The number of exchange constants should be the \
same as the number of shells')
    
    q = np.array(q)
    return np.sum([j * np.sum(1 - np.cos(r @ q * 2 * np.pi)) \
                   for j, r in zip(exch_const, pos)])

def cal_magnon_spectrum(exch_const, q, R0j):
    '''
    calculate the magnon spectrum of Heisenberg model
    
    Parameters
    ----------
    exch_const : list
        exchange constant between atoms for each shell
    q : np.array
        wave vector path
    R0j : np.array
        positions of atoms or displacement respect to the origin

    Returns
    -------
    E : np.array
        magnon spectrum
    '''
    return np.array([cal_ener_bilinear_heisenberg(exch_const, qi, R0j) for qi in q])

class TestMagnonSpectrumCalculation(unittest.TestCase):

    @unittest.skip('The calculation of magnon spectrum is not a typical unittest')
    def test_magnon_spectrum(self):

        # Rinaldi M, Mrovec M, Bochkarev A, et al. 
        # Non-collinear magnetic atomic cluster expansion for iron[J]. 
        # npj Computational Materials, 2024, 10(1): 12.      
        J = np.array([21.8145, 13.3996, 3.53721, -0.137615, -0.0356779])

        # Wang H, Ma P W, Woo C H. 
        # Exchange interaction function for spin-lattice coupling in bcc iron[J]. 
        # Physical Review B—Condensed Matter and Materials Physics, 2010, 82(14): 144304.
        # see TABLE I.
        # GGA
        #J = np.array([1.218, 1.080, -0.042, -0.185, -0.117, 0.061, -0.013, 0.017]) * 13.6
        # LDA
        #J = np.array([1.235, 0.799, -0.009, -0.128, -0.093, 0.044, 0.001, 0.018]) * 13.6
        # Ref. 50 (LDA)
        #J = np.array([1.24, 0.646, 0.007, -0.108, -0.071, 0.035, 0.002, 0.014]) * 13.6
        # Ref. 51 (LDA, von Barth and Hedin)
        #J = np.array([1.2, 0.646, -0.030, -0.100, -0.068, 0.042, -0.001, 0.014]) * 13.6
        # Ref. 52 (LDA, Vosko-Wilk-Nusair)
        #J = np.array([1.432, 0.815, -0.015, -0.126, -0.146, 0.062, 0.001, 0.015]) * 13.6
        # Tanaka T, Gohda Y. 
        # Prediction of the Curie temperature considering the dependence of the phonon free 
        # energy on magnetic states[J]. 
        # npj Computational Materials, 2020, 6(1): 184.
        #J = np.array([27.18, 2.62, 1.33, 0.21, -1.37])

        # generate structure according to given Jij
        _, _, pos = neighborgen('bcc', 2.86304, len(J), direct=True, group=True)
        _, q = qpathgen('bcc', 2.86304, 'GNPGHN', direct=True, n=20)
        # q /= 2 * np.pi
        # print(pos)
        E = cal_magnon_spectrum(J, q, pos) / 1e3 # meV to eV
        
        plt.plot(E, 'r-', linewidth = 2)
        plt.xticks(np.linspace(0, len(q), 6), ['$\Gamma$', 'N', 'P', '$\Gamma$', 'H', 'N'])
        plt.ylabel('$\omega_{\mathbf{q}}$ (eV/atom)')
        plt.tight_layout()
        plt.savefig('fig7b.png')

    @unittest.skip('The comparison of magnon spectrum is not a typical unittest')
    def test_magnon_comp(self):

        J_series = [
            np.array([21.8145, 13.3996, 3.53721, -0.137615, -0.0356779]),
            np.array([21.8145, 13.3996, -3.53721, -0.137615, -0.0356779]),
            np.array([1.218, 1.080, -0.042, -0.185, -0.117, 0.061, -0.013, 0.017]) * 13.6,
            np.array([1.235, 0.799, -0.009, -0.128, -0.093, 0.044, 0.001, 0.018]) * 13.6,
            np.array([1.24, 0.646, 0.007, -0.108, -0.071, 0.035, 0.002, 0.014]) * 13.6,
            np.array([1.2, 0.646, -0.030, -0.100, -0.068, 0.042, -0.001, 0.014]) * 13.6,
            np.array([1.432, 0.815, -0.015, -0.126, -0.146, 0.062, 0.001, 0.015]) * 13.6
        ]
        legends = ['Rinaldi et al.', 
                   'Rinaldi et al. (corrected)', 
                   'Wang et al. (GGA)', 'Wang et al. (LDA)', 'Ref. 50', 'Ref. 51', 'Ref. 52']

        for J, legend in zip(J_series, legends):
            _, _, pos = neighborgen('bcc', 2.86304, len(J), direct=True, group=True)
            _, q = qpathgen('bcc', 2.86304, 'GNPGHN', direct=True, n=20)
            E = cal_magnon_spectrum(J, q, pos) / 1e3

            plt.plot(E, '-', linewidth = 2, label = legend)
        plt.xticks(np.linspace(0, len(q), 6), ['$\Gamma$', 'N', 'P', '$\Gamma$', 'H', 'N'])
        plt.ylabel('$\omega_{\mathbf{q}}$ (eV/atom)')
        plt.tight_layout()
        plt.legend(fontsize = 12, loc = 'lower right')
        plt.savefig('fig7b_comp.png')

if __name__ == '__main__':
    
    test = init()
    unittest.main(exit=test)
