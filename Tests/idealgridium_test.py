# Author: Thomas Ersevim, 2026
###################################################################################################

'''An IdealGridium test suite'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import pytest
import copy
import matplotlib.pyplot as plt

from Circuit_Objs.qchard_idealgridium import *
import Notebooks.plotting_settings

soft_IdealGridium = IdealGridium(**soft_IdealGridium_params,
                                 **std_IdealGridium_sim_params)
hard_IdealGridium = IdealGridium(**hard_IdealGridium_params,
                                 **std_IdealGridium_sim_params)
# Could also give these .name atributes which would print during tests

def sweep_param_spectrum(sweep_array:np.ndarray, param_name:str, gridium:IdealGridium) -> np.ndarray:
    ''' Returns an array of energies based on a changing parameter "parameter name" using sweep_array as the data array.'''
    copied_gridium = copy.deepcopy(gridium)
    spectrum_array = np.empty((*np.shape(sweep_array), gridium.nlev))
    for i, value in enumerate(sweep_array):
        print(i)
        setattr(copied_gridium, param_name, value)
        spectrum_array[i] = copied_gridium.transition_energies()
    return spectrum_array

## Tests ##
# All tests currently agree well with Dat's simulations (when given a sufficently high lc cutoff)

@pytest.mark.parametrize('tested_gridium', [soft_IdealGridium, hard_IdealGridium])
def test_phi_ext_spectrum(tested_gridium, plotting=False):
    '''Tests phi external spectrum is pi periodic and/or plots the spectrum'''

    error_threshold = 1e-4
    tested_gridium = copy.deepcopy(tested_gridium)

    if not plotting: 
        fluxes = np.linspace(0, np.pi, 7)
        spectrum = sweep_param_spectrum(fluxes, 'phi_ext', tested_gridium)
        spectrum_plus_pi = sweep_param_spectrum(fluxes+np.pi, 'phi_ext', tested_gridium)
        max_pct_errors = list()
        for lev in range(tested_gridium.nlev-1): # Removing all zero transition
            pct_error = (spectrum[:,lev+1]-spectrum_plus_pi[:,lev+1])/np.max(np.abs(spectrum[:,lev+1]))
            max_pct_error = np.max(np.abs(pct_error))
            max_pct_errors.append(max_pct_error)
        worst_error = np.max(np.abs(max_pct_errors)) 
        print(max_pct_errors)
        print(worst_error)
        assert worst_error < error_threshold # Assertion is that phi_ext values + pi should be identical (within threshold)

    if plotting: # To visually ensure that the convergence is good and stable.
        fluxes = np.linspace(0, 1, 81) # In units of 2*pi
        tested_gridium.nlev = 6
        tested_gridium.nlev_lc = 400
        spectrum = sweep_param_spectrum(fluxes*(2*np.pi), 'phi_ext', tested_gridium)
        spectrum = spectrum[:, 1:]
        for i in range(spectrum.shape[1]):
            plt.plot(fluxes, spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel(r'$\phi_{ext}/2\pi$')
        plt.ylabel(r'Transition Energy ({})'.format(tested_gridium.units))
        # plt.legend()
        plt.tight_layout()
        plt.show()
        return

@pytest.mark.parametrize('tested_gridium', [soft_IdealGridium, hard_IdealGridium]) # TODO: could turn this into an actual test. perhaps like that degeneracies are degeneracies or something.
def test_ng_spectrum(tested_gridium, plotting=False):
    '''
    Tests ng dispersion for agreement with Dat's previous spectrum results
    NOTE: This test is most sensitive to the lc cutoff. For hard gridium needs about 1000 or more.
    '''

    ngs = np.linspace(0, 1, 41) #Is symmetric about 0 with periodicity of 1, so only doing this range
    spectrum = sweep_param_spectrum(ngs, 'ng', tested_gridium)
    zeros = tested_gridium.transition_energies()
    zeroed_spectrum = (spectrum-zeros)[:,1:]

    if plotting:
        for i in range(zeroed_spectrum.shape[1]):
            plt.plot(ngs, zeroed_spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel(r'$n_g$')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_gridium.units))
        plt.legend()
        plt.tight_layout()
        plt.show()

@pytest.mark.parametrize('tested_gridium', [soft_IdealGridium, hard_IdealGridium])
def test_convergence(tested_gridium, plotting=False):
    '''
    A test for the convergence of the gridium spectra so that as a function of the nlev_lc, the
    change is negligible (i.e. the asymptote gives within less than 1 percent of the chosen
    nlev_lc cutoff value.
    
    :param tested_gridium: The gridium used for the convergence test
    '''

    copied_gridium = copy.deepcopy(tested_gridium)
    error_threshold = 1e-4
    nlev = copied_gridium.nlev

    if not plotting:
        default_lc_transitions = copied_gridium.transition_energies(nlev=nlev)[1:]
        copied_gridium.nlev_lc = copied_gridium.nlev_lc*2
        twice_lc_transitions = copied_gridium.transition_energies(nlev=nlev)[1:]
        pct_error = (default_lc_transitions-twice_lc_transitions)/np.max(np.abs(twice_lc_transitions))
        worst_error = np.max(np.abs(pct_error)) 
        assert worst_error < error_threshold # For testing, the assertion is that the precent error, scaled to the largest transition (to get around zeros) should be < error_threshold

    if plotting: # To visually ensure that the convergence is good and stable.
        plotting_points = 20
        nlevs_lc = np.linspace(10,copied_gridium.nlev_lc*2, plotting_points, dtype=int)
        convergence_array = np.empty((plotting_points, nlev))
        for i, nlev_lc in enumerate(nlevs_lc):
            copied_gridium.nlev_lc = nlev_lc
            convergence_array[i,:] = copied_gridium.transition_energies(nlev=nlev)
        for i in range(nlev):
            plt.plot(nlevs_lc, convergence_array[:,i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel('H.O. Basis Cutoff Dimentionality')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_gridium.units))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

@pytest.mark.parametrize('tested_gridium, variable', [(hard_IdealGridium, 'n'), (hard_IdealGridium, 'phi')])
def test_matrix_element(tested_gridium:IdealGridium, variable, plotting=False):
    '''
    A test to see the matrix elements of the gridium.
    
    Allowed variables: 'n', 'phi'

    NOTE: This test seems to be the one that is most sensitive to a lower lc cutoff. In the case of the hard gridium, the higher energy matrix elements are off until lc ~ 1000
    '''

    copied_gridium = copy.deepcopy(tested_gridium)
    error_threshold = 0.02 # Defines a minimum 0-1 matrix element for the hard_IdealGridium test
    nlev = copied_gridium.nlev
    # copied_gridium.nlev_lc = 800

    var_matrix = np.zeros((nlev,nlev))
    if variable=='n':
        for level1 in range(nlev):
            for level2 in range(nlev):
                var_matrix[level1, level2] = np.abs(copied_gridium.n_ij(level1=level1, level2=level2))
    if variable=='phi':
        for level1 in range(nlev):
            for level2 in range(nlev):
                var_matrix[level1, level2] = np.abs(copied_gridium.phi_ij(level1=level1, level2=level2))
    np.fill_diagonal(var_matrix, 0) # Diagonal matrix elements are meaningless in this context.
    
    if plotting:
        plt.imshow(var_matrix, cmap='Blues', vmin=0, vmax=2)
        plt.gca().invert_yaxis() # Remove if desirous of conventional matrix labeling
        # plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        for i in range(var_matrix.shape[0]):
            for j in range(var_matrix.shape[1]):
                plt.text(j, i, f'{var_matrix[i, j]:.2f}', ha='center', va='center', color='white')
        plt.colorbar(shrink=0.8)
        if variable=='n':
            plt.title(r'$| \langle j  | \hat n | k \rangle |$')
        if variable=='phi':
            plt.title(r'$| \langle j  | \hat \phi | k \rangle |$')
        plt.xticks()
        plt.xlabel('eigenstates')
        plt.ylabel('eigenstates')
        plt.show()
    else:
        # Checking to ensure that in the hard Gridium case, the 0-1 transitions are small for both phi and n, and that it is a symmetric matrix
        assert var_matrix[0,1] < error_threshold
        assert np.allclose(var_matrix, var_matrix.T)

# TODO: Implement eigenvector plots -- will help with understanding.

if __name__=='__main__':
    # test_ng_spectrum(soft_IdealGridium, plotting=True)
    # test_matrix_element(hard_IdealGridium, variable='phi', plotting=True)
    test_phi_ext_spectrum(hard_IdealGridium, plotting=True)
    # test_convergence(plotting=True, tested_gridium=soft_IdealGridium)
