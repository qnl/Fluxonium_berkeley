# Author: Thomas Ersevim, 2026
###################################################################################################

'''A test suite for a SCQubits based transmon wrapped in the necessary class'''

import sys
sys.path.append('/Users/thomasersevim/QNL/2q_gridium')

import numpy as np
import copy
import yaml
import pytest
import scqubits #https://scqubits.readthedocs.io/en/latest/installation.html#arm64-compatibility
from itertools import product
import matplotlib.pyplot as plt

from Circuit_Objs.qchard_idealgridium import *
from Circuit_Objs.qchard_transmon import *
from Circuit_Objs.qchard_coupobj import *
from Circuit_Objs.qchard_fluxonium import *
from Simulations.Cphase.workflow_funcs import *
import Notebooks.plotting_settings

__all__ = ['test_matrix_element']

#@pytest.mark.parametrize('tested_gridium, variable', [(hard_IdealGridium, 'n'), (hard_IdealGridium, 'phi')])
def test_matrix_element(system:CoupledObjects, system_cfg:SystemConfig, qubit:str, operator:str, all_transitions=False, plotting=False):
    '''
    A test to see the matrix elements of the gridium.
    Allowed variables: 'n', 'phi'
    NOTE: This test seems to be the one that is most sensitive to a lower lc cutoff. In the case of the hard gridium, the higher energy matrix elements are off until lc ~ 1000
    '''
    copied_system = copy.deepcopy(system)
    error_threshold = 0.02 # Defines a minimum 0-1 matrix element for the hard_IdealGridium test

    qubitA = copied_system._objects[0]
    qubitB = copied_system._objects[1] 
    if qubit=='A':
        qubit = qubitA
    elif qubit=='B':
        qubit = qubitB
    if operator=='flux':
        operator = qubit.phi()
    elif operator=='charge':
        operator = qubit.n()

    if all_transitions:
        total_length = qubitA.nlev*qubitB.nlev
        inner_prod_matrix = np.zeros((total_length, total_length))
        tuples = list(product(range(qubitA.nlev), range(qubitB.nlev)))
        i = 0
        for tuple_row in tuples:
            j = 0
            for tuple_column in tuples:
                inner_prod_matrix[i, j] = np.abs(copied_system.matr_el(qubit, operator, tuple_row, tuple_column, interaction='on'))
                j += 1
            i += 1
        np.fill_diagonal(inner_prod_matrix, 0)

        if plotting:
            plt.imshow(inner_prod_matrix, cmap='Blues') #, vmin=0, vmax=0.5)
            plt.gca().invert_yaxis()
            # i = 0
            # for tuple_row in tuples:
            #     j = 0
            #     for tuple_column in tuples:
            #         plt.text(j, i, f'{inner_prod_matrix[j, i]:.2f}', ha='center', va='center', color='white')
            plt.colorbar(shrink=0.8)
            if operator=='n':
                plt.title(r'$| \langle j  | \hat n | k \rangle |$')
            if operator=='phi':
                plt.title(r'$| \langle j  | \hat \phi | k \rangle |$')
            label_tuples = [str(tuple) for tuple in tuples]
            plt.xticks(ticks=range(total_length), labels=label_tuples, size='small', minor=False)
            plt.yticks(ticks=range(total_length), labels=label_tuples, size='small', minor=False)
            plt.xlabel('eigenstates')
            plt.ylabel('eigenstates')
            plt.tight_layout()
            plt.show()
    if not all_transitions:
        all_relevant_states = (
            system_cfg.transitions_to_drive
        + system_cfg.detuned_transitions
        + system_cfg.comparitive_transitions
        + system_cfg.detuned_comparitive_transitions
        + system_cfg.coupled_resonant_transitions)
        
        all_relevant_states = tuple(dict.fromkeys(all_relevant_states)) # Deduplicates while preserving order

        total_length = len(all_relevant_states)
        inner_prod_matrix = np.zeros((total_length, total_length))
        for i, row_state in enumerate(all_relevant_states):
            for j, column_state in enumerate(all_relevant_states):
                inner_prod_matrix[i, j] = np.abs(copied_system.matr_el(qubit, operator, row_state, column_state, interaction='on'))
        np.fill_diagonal(inner_prod_matrix, 0)

        if plotting:
            plt.imshow(inner_prod_matrix, cmap='Blues') #, vmin=0, vmax=0.5
            plt.gca().invert_yaxis()
            i = 0
            plt.colorbar(shrink=0.8)
            if operator=='n':
                plt.title(r'$| \langle j  | \hat n | k \rangle |$')
            if operator=='phi':
                plt.title(r'$| \langle j  | \hat \phi | k \rangle |$')
            label_tuples = all_relevant_states
            plt.xticks(ticks=range(total_length), labels=label_tuples, size='small', minor=False)
            plt.yticks(ticks=range(total_length), labels=label_tuples, size='small', minor=False)
            plt.xlabel('eigenstates')
            plt.ylabel('eigenstates')
            plt.show()
    if not plotting:
        pass

if __name__=='__main__':
    pulse_path = 'Simulations/CPhase/yamls/pulses/fluxonium_idealgridium_hard.yaml'
    syscfg_path = 'Simulations/CPhase/yamls/syscfgs/fluxonium_idealgridium_hard.yaml'

    with open(pulse_path, 'r') as f:
        data = yaml.safe_load(f)
        pulse_cfg = PulseConfig(**data)

    with open(syscfg_path, 'r') as f:
        data = yaml.safe_load(f)
        system_cfg = SystemConfig(**data)
        del data

    fluxonium = Fluxonium(**light_fluxonium_params, **std_fluxonium_sim_params)
    gridium = IdealGridium(**hard_IdealGridium_params, **std_IdealGridium_sim_params)
    gridium.nlev = 6
    fluxonium.nlev = 4

    qubitA = fluxonium
    qubitB = gridium
    n_shown_states = 3

    qubitA, qubitB = load_qubits(qubitA, qubitB)
    qubitA, qubitB = scale_qubitA_transition(qubitA, qubitB, system_cfg)
    system = couple_qubits(qubitA, qubitB, system_cfg, pulse_cfg, mute=False)
    test_matrix_element(system, system_cfg, 'B', 'n', all_transitions=False, plotting=True) # The thing you are driving on should be the same as the qubit measured here with the same coupling!