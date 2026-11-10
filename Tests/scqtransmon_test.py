# Author: Thomas Ersevim, 2026
###################################################################################################

'''A test suite for a SCQubits based transmon wrapped in the necessary class'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import copy
import pytest
import scqubits #https://scqubits.readthedocs.io/en/latest/installation.html#arm64-compatibility
import matplotlib.pyplot as plt

from Circuit_Objs.qchard_transmon import *
import Notebooks.plotting_settings

std_transmon = SCQTransmon(**std_transmon_params, **std_transmon_sim_params)

@pytest.mark.parametrize('tested_transmon', [std_transmon])
def test_convergence(tested_transmon:SCQTransmon, plotting=False):
    '''
    A test for the convergence of the transmon spectra so that as a function of the nlev_lc, the
    change is negligible (i.e. the asymptote gives within less than 1 percent of the chosen
    nlev_lc cutoff value.
    '''

    transmon = copy.deepcopy(tested_transmon)
    error_threshold = 1e-4 # As a precentage
    cutoff_scaling = 2

    nlev = transmon.nlev

    if not plotting:
        default_cutoff_energies = transmon.transition_energies(nlev=nlev)[1:]
        transmon.nlev_cutoff = transmon.nlev_cutoff*cutoff_scaling
        higher_cutoff_energies = transmon.transition_energies(nlev=nlev)[1:]
        pct_error = (default_cutoff_energies-higher_cutoff_energies)/np.max(np.abs(higher_cutoff_energies))
        worst_error = np.max(np.abs(pct_error)) 
        assert worst_error < error_threshold # For testing, the assertion is that the precent error, scaled to the largest transition (to get around zeros) should be < error_threshold

    if plotting: # To visually ensure that the convergence is good and stable.
        plotting_points = 10
        n_cutoffs = np.linspace(transmon.nlev,transmon.nlev_cutoff*2, plotting_points, dtype=int) # Using start point as number of energy levels
        convergence_array = np.empty((plotting_points, nlev))
        for i, nlev_lc in enumerate(n_cutoffs):
            transmon.nlev_cutoff = nlev_lc
            convergence_array[i,:] = transmon.transition_energies(nlev=nlev)
        for i in range(nlev):
            plt.plot(n_cutoffs, convergence_array[:,i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel('H.O. Basis Cutoff Dimentionality')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_transmon.units))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

if __name__=='__main__':
    test_convergence(std_transmon, plotting=True)