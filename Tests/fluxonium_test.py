# Author: Thomas Ersevim, 2026
###################################################################################################

'''An Fluxonium test suite'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import pytest
import copy
import matplotlib.pyplot as plt

from Circuit_Objs.qchard_fluxonium import *
import Notebooks.plotting_settings

heavy_fluxonium = Fluxonium(**heavy_fluxonium_params, **std_fluxonium_sim_params)
light_fluxonium = Fluxonium(**heavy_fluxonium_params, **std_fluxonium_sim_params)

def sweep_param_spectrum(sweep_array:np.ndarray, param_name:str, tested_fluxonium:Fluxonium) -> np.ndarray:
    ''' Returns an array of energies based on a changing parameter "parameter name" using sweep_array as the data array.'''
    copied_gridium = copy.deepcopy(tested_fluxonium)
    spectrum_array = np.empty((*np.shape(sweep_array), tested_fluxonium.nlev))
    for i, value in enumerate(sweep_array):
        print(i)
        setattr(copied_gridium, param_name, value)
        spectrum_array[i] = copied_gridium.transition_energies()
    return spectrum_array

## Tests ##
# All tests currently agree well with Dat's simulations (when given a sufficently high lc cutoff)

@pytest.mark.parametrize('tested_fluxonium', [heavy_fluxonium, light_fluxonium])
def test_phi_ext_spectrum(tested_fluxonium:Fluxonium, plotting=False):
    '''
    Tests phi external spectrum is pi periodic and/or plots the spectrum.
    '''

    error_threshold = 1e-4
    checking_points = 7

    if not plotting: 
        fluxes = np.linspace(0, np.pi, checking_points+1)[1:]
        right_spectrum = sweep_param_spectrum((np.pi + fluxes), 'phi_ext', tested_fluxonium)
        left_spectrum = sweep_param_spectrum((np.pi - fluxes), 'phi_ext', tested_fluxonium)
        max_pct_errors = list()
        for lev in range(tested_fluxonium.nlev-1): # Removing all zero transition
            pct_error = (right_spectrum[:,lev+1]-left_spectrum[:,lev+1])/np.max(np.abs(right_spectrum[:,lev+1]))
            max_pct_error = np.max(np.abs(pct_error))
            max_pct_errors.append(max_pct_error)
        worst_error = np.max(np.abs(max_pct_errors)) 
        print('List of max percent errors: {}\n'.format(max_pct_errors))
        print('Worst percent error: {}'.format(worst_error))
        assert worst_error < error_threshold # Assertion is that phi_ext values + pi should be identical (within threshold)

    if plotting: # To visually ensure that the convergence is good and stable.
        fluxes = np.linspace(0, 2*np.pi, 41)
        right_spectrum = sweep_param_spectrum(fluxes, 'phi_ext', tested_fluxonium)
        right_spectrum = right_spectrum[:, 1:]
        for i in range(right_spectrum.shape[1]):
            plt.plot(fluxes/(2*np.pi), right_spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel(r'$\phi_{ext}/2\pi$')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_fluxonium.units))
        plt.legend()
        plt.show()
        return

@pytest.mark.parametrize('tested_fluxonium', [heavy_fluxonium, light_fluxonium])
def test_convergence(tested_fluxonium:Fluxonium, plotting=False):
    '''
    A test for the convergence of the gridium spectra so that as a function of the nlev_lc, the
    change is negligible (i.e. the asymptote gives within less than 1 percent of the chosen
    nlev_lc cutoff value.
    
    :param tested_gridium: The gridium used for the convergence test
    '''

    copied_fluxonium = copy.deepcopy(tested_fluxonium)
    error_threshold = 1e-4
    nlev = copied_fluxonium.nlev

    if not plotting:
        default_lc_transitions = copied_fluxonium.transition_energies(nlev=nlev)[1:]
        copied_fluxonium.nlev_lc = copied_fluxonium.nlev_lc*2
        twice_lc_transitions = copied_fluxonium.transition_energies(nlev=nlev)[1:]
        pct_error = (default_lc_transitions-twice_lc_transitions)/np.max(np.abs(twice_lc_transitions))
        worst_error = np.max(np.abs(pct_error)) 
        assert worst_error < error_threshold # For testing, the assertion is that the precent error, scaled to the largest transition (to get around zeros) should be < error_threshold

    if plotting: # To visually ensure that the convergence is good and stable.
        plotting_points = 20
        nlevs_lc = np.linspace(10,copied_fluxonium.nlev_lc*2, plotting_points, dtype=int)
        convergence_array = np.empty((plotting_points, nlev))
        for i, nlev_lc in enumerate(nlevs_lc):
            copied_fluxonium.nlev_lc = nlev_lc
            convergence_array[i,:] = copied_fluxonium.transition_energies(nlev=nlev)
        for i in range(nlev):
            plt.plot(nlevs_lc, convergence_array[:,i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel('H.O. Basis Cutoff Dimentionality')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_fluxonium.units))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

# TODO: Implement eigenvector plots -- will help with understanding.

if __name__=='__main__':
    # test_phi_ext_spectrum(Fluxonium(**heavy_fluxonium_params, **std_fluxonium_sim_params), plotting=True)
    # test_convergence(tested_fluxonium=light_fluxonium, plotting=True)