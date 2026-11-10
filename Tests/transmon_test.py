# Author: Thomas Ersevim, 2026
###################################################################################################

'''A Transmon (defined using LC basis) test suite'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import pytest
import copy
import matplotlib.pyplot as plt
import scqubits #https://scqubits.readthedocs.io/en/latest/installation.html#arm64-compatibility

from Circuit_Objs.qchard_transmon import *
import Notebooks.plotting_settings

std_transmon = Transmon(**std_transmon_params, **std_transmon_sim_params)
std_transmon.nlev_lc = 50

def test_eigenvals():
    error_threshold = 0.02
    scTransmon = scqubits.Transmon(EC=1, EJ=50, ng=0, ncut=15)
    zeroed_scTransmon_vals = (scTransmon.eigenvals() - scTransmon.eigenvals()[0])[1:]
    qchard_transmon = Transmon(E_C=1, E_J=50, n_g=0, nlev=6, nlev_lc=25, units='GHz')
    zeroed_qchardTransmon_vals = qchard_transmon.transition_energies()[1:]
    remainder = (1 - (zeroed_qchardTransmon_vals/zeroed_scTransmon_vals))
    max_off = max(abs(remainder))
    assert max_off < error_threshold
    
@pytest.mark.parametrize('tested_transmon', [])
def test_convergence(tested_transmon:Transmon, plotting=False):
    '''
    A test for the convergence of the gridium spectra so that as a function of the nlev_lc, the
    change is negligible (i.e. the asymptote gives within less than 1 percent of the chosen
    nlev_lc cutoff value.
    
    :param tested_gridium: The gridium used for the convergence test
    '''

    copied_transmon = copy.deepcopy(tested_transmon)
    error_threshold = 1e-4
    nlev = copied_transmon.nlev

    if not plotting: 
        default_lc_transitions = copied_transmon.transition_energies(nlev=nlev)[1:]
        copied_transmon.nlev_lc = copied_transmon.nlev_lc*2
        twice_lc_transitions = copied_transmon.transition_energies(nlev=nlev)[1:]
        pct_error = (default_lc_transitions-twice_lc_transitions)/np.max(np.abs(twice_lc_transitions))
        worst_error = np.max(np.abs(pct_error)) 
        assert worst_error < error_threshold # For testing, the assertion is that the precent error, scaled to the largest transition (to get around zeros) should be < error_threshold

    if plotting: # To visually ensure that the convergence is good and stable.
        plotting_points = 50
        nlevs_lc = np.linspace(3,copied_transmon.nlev_lc*2, plotting_points, dtype=int)
        convergence_array = np.empty((plotting_points, nlev))
        for i, nlev_lc in enumerate(nlevs_lc):
            copied_transmon.nlev_lc = nlev_lc
            convergence_array[i,:] = copied_transmon.transition_energies(nlev=nlev)
        for i in range(nlev):
            plt.plot(nlevs_lc, convergence_array[:,i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel('H.O. Basis Cutoff Dimentionality')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_transmon.units))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

if __name__=='__main__':
    test_convergence(std_transmon, plotting=True)