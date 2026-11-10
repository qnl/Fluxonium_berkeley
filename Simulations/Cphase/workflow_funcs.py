'''
A set of helper functions for a CPhase simulation. To be used in conjunction with the main.ipynb
in the same directory

Author: Thomas Ersevim, 2026
'''

import sys
sys.path.append('Users/thomasersevim/anaconda3')
sys.path.append('/Users/thomasersevim/QNL/2q_gridium/')

import os
import numpy as np
import scipy
import yaml
import dill
import copy
from scipy.optimize import minimize
from dataclasses import dataclass
from matplotlib import pyplot as plt
from IPython.display import display, Latex

from Circuit_Objs import qchard_evolgates as gates;
from Circuit_Objs.qchard_coupobj import CoupledObjects
from Circuit_Objs.qchard_idealgridium import *
from Circuit_Objs.qchard_fluxonium import *
from Circuit_Objs.qchard_abstractobj import AbstractQubit

__all__ = ['PulseConfig', 'SystemConfig', 'load_qubits', 'couple_qubits', 'solve',
           'visualize_state_propagation', 'solve_coupled_qubits', 'minimize_infidelity',
           'converge_on_pi', 'scale_qubitA_transition']

@dataclass
class PulseConfig:
    T_gate: int
    pulse_shape: str
    T_rise: int
    pulse_sigma: float
    DRAG: bool
    DRAG_coeff: float
    drive_amplitude_factor: float
    targeted_drive: str
    drive_detuning: float

@dataclass
class SystemConfig:
    qubitA: str
    qubitB: str
    driven_qubit: str
    J_C: float
    interaction: str
    coupling_type:str
    computational_space: list
    transitions_to_drive: list
    detuned_transitions: list
    comparitive_transitions: list
    detuned_comparitive_transitions: list
    coupled_resonant_transitions: list

def load_qubits(qubitA, qubitB, dir:str='/Users/thomasersevim/QNL/2q_gridium/etc/qubits/'):
    # Attempt to load the qubits (pre-diagonalized) from a file if it exists

    qubitA_path = dir+qubitA._save_str()
    if os.path.exists(qubitA_path):
        with open(qubitA_path, 'rb') as f:
            qubitA = dill.load(f)
    
    qubitB_path = dir+qubitB._save_str()
    if os.path.exists(qubitB_path):
        with open(qubitB_path, 'rb') as f:
            qubitB = dill.load(f)

    return qubitA, qubitB

def scale_qubitA_transition(qubitA:AbstractQubit, qubitB: AbstractQubit, system_cfg:SystemConfig, mute=False) -> tuple[AbstractQubit, AbstractQubit]:
    # Scales all relevant parameters of qubitA to match the transitions_to_drive splitting of qubitB

    scaling_margin_factor = 1.001 # Should be > 1
    scaling_warning_factor = 5 # Should be > 1

    qubitA = copy.deepcopy(qubitA)
    qubitB = copy.deepcopy(qubitB)

    # A bit of a clever trick to allow use of system_cfg tags to get at transitions
    system = CoupledObjects(qubitA, qubitB, [qubitA, qubitB, system_cfg.J_C, system_cfg.coupling_type])
    qA_trans = (system.level(system_cfg.detuned_transitions[0], interaction='off')
              - system.level(system_cfg.transitions_to_drive[0], interaction='off'))
    qB_trans = (system.level(system_cfg.coupled_resonant_transitions[1], interaction='off')
              - system.level(system_cfg.coupled_resonant_transitions[0], interaction='off'))

    if np.issubdtype(qA_trans.dtype, np.complexfloating) or np.issubdtype(qB_trans.dtype, np.complexfloating):
        raise Exception('Qubit transitions are yeilding a complex frequency. Check convergences/cutoffs')

    scaling = qB_trans/qA_trans

    if not mute:
        print('{} uncoupled, prescaled transition is {:.3f} GHz'.format(qubitA.name, qA_trans))
        print('{} uncoupled intended resonant transition is {:.3f} GHz'.format(qubitB.name, qB_trans))
    if scaling < scaling_margin_factor and scaling > scaling_margin_factor**(-1): # Prevents unintended resolving needed for hard qubit diagonalization
        return qubitA, qubitB
    if scaling > scaling_warning_factor or scaling < scaling_warning_factor**(-1):
        print("WARNING: the {}'s parameters have been rescaled by a factor of {:.3f} and may no longer be physically realizable".format(qubitA.name, scaling))

    qubitA._scale_E_params(scaling)
    system = CoupledObjects(qubitA, qubitB, [qubitA, qubitB, system_cfg.J_C, system_cfg.coupling_type])
    qA_trans = (system.level(system_cfg.detuned_transitions[0], interaction='off')
              - system.level(system_cfg.transitions_to_drive[0], interaction='off'))

    if np.issubdtype(qA_trans.dtype, np.complexfloating) or np.issubdtype(qB_trans.dtype, np.complexfloating):
        raise Exception('Qubit transitions are yeilding a complex frequency. Check convergences/cutoffs')

    if not mute:
        print('{} uncoupled, postscaled transition is {:.3f} GHz'.format(qubitA.name, qA_trans))

    return qubitA, qubitB

def couple_qubits(
        qubitA:AbstractQubit,
        qubitB:AbstractQubit,
        system_cfg:SystemConfig,
        pulse_cfg:PulseConfig, # TODO: Remove. Unnecessary
        mute=False):
    # Couple qubits and define driven qubit(s). Report back relevant information

    transitions_to_drive = system_cfg.transitions_to_drive
    comparitive_transitions = system_cfg.comparitive_transitions
    detuned_transitions =  system_cfg.detuned_transitions
    detuned_comparitive_transitions = system_cfg.detuned_comparitive_transitions
    coupling_type = system_cfg.coupling_type
    J_C = system_cfg.J_C

    system = CoupledObjects(qubitA, qubitB, [qubitA, qubitB, J_C, coupling_type])
    # Calculate the drive frequency.
    driven_transition_detuning = np.real(
        system.freq(detuned_transitions[0], detuned_transitions[1]) -
        system.freq(transitions_to_drive[0], transitions_to_drive[1]))
    comparitive_transition_detuning = np.real(
        system.freq(detuned_comparitive_transitions[0], detuned_comparitive_transitions[1]) -
        system.freq(comparitive_transitions[0], comparitive_transitions[1]))
    relative_detuning = comparitive_transition_detuning - driven_transition_detuning 

    if not mute:
        print('\nIntended detuning between {}-{} and {}-{} (2nd minus 1st): {:.3f} MHz'.format(
            transitions_to_drive[0], transitions_to_drive[1],
            detuned_transitions[0], detuned_transitions[1],
            1000*driven_transition_detuning))
        print('Comparative detuning between {}-{} and {}-{} (2nd minus 1st): {:.3f} MHz'.format(
            comparitive_transitions[0], comparitive_transitions[1],
            detuned_comparitive_transitions[0], detuned_comparitive_transitions[1],
            1000*comparitive_transition_detuning))
        print('Relative detuning is {:.3f} MHz (comparitive - intended)'.format(1000*relative_detuning))
        print('Transition to drive: {} - {} with frequency {:.4f} GHz'.format(
            transitions_to_drive[0], transitions_to_drive[1],
            np.real(system.freq(transitions_to_drive[0], transitions_to_drive[1]))))
        print('Comparative transition: {} - {} with frequency {:4f} GHz'.format(
            comparitive_transitions[0], comparitive_transitions[1],
            np.real(system.freq(comparitive_transitions[0], comparitive_transitions[1]))))
    return system

def solve(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False):
    # Solve according to parts

    comp_space = system_cfg.computational_space
    interaction = system_cfg.interaction

    driven_qubit = system_cfg.driven_qubit
    transitions_to_drive = system_cfg.transitions_to_drive
    detuned_transitions = system_cfg.detuned_transitions
    
    T_gate = pulse_cfg.T_gate
    shape = pulse_cfg.pulse_shape
    T_rise = pulse_cfg.T_rise
    sigma = pulse_cfg.pulse_sigma
    drag = pulse_cfg.DRAG
    drag_coeff = pulse_cfg.DRAG_coeff
    drive_amplitude_factor = pulse_cfg.drive_amplitude_factor
    targeted_drive = pulse_cfg.targeted_drive
    drive_detuning = pulse_cfg.drive_detuning
    
    qubitA = system._objects[0]
    qubitB = system._objects[1]

    # Drive line coupling    
    A_n_matrix_el = np.abs(system.n_ij(qubitA, transitions_to_drive[0], transitions_to_drive[1], interaction='on'))
    B_n_matrix_el = np.abs(system.n_ij(qubitB, transitions_to_drive[0], transitions_to_drive[1], interaction='on'))

    # Changes where the charge drive occurs and scales the drive to the size of the matrix element
    if not system.n(0).check_herm():
        raise Exception('system.n(0) is not hermitian!')
    if not system.n(1).check_herm():
        raise Exception('system.n(1) is not hermitian!')

    # I see. The matrix element is so tiny, that upon cleanup etc the H_drive ceases to be hermitian, and this is an issue.

    if driven_qubit=='A':
        H_drive = (system.n(0)/A_n_matrix_el)*drive_amplitude_factor
    elif driven_qubit=='B':
        H_drive = (system.n(1)/B_n_matrix_el)*drive_amplitude_factor
    elif driven_qubit=='AB':
        H_drive = (system.n(0) + system.n(1))/(A_n_matrix_el + B_n_matrix_el)*drive_amplitude_factor
    H_drive_dummy = 0 * system.n(0)

    if targeted_drive=='default':
        omega_d = system.freq(transitions_to_drive[0], transitions_to_drive[1]) + drive_detuning
    elif targeted_drive=='split':
        omega_d = ((system.freq(transitions_to_drive[0], transitions_to_drive[1], interaction='on')
                  + system.freq(detuned_transitions[0], detuned_transitions[1], interaction='on'))/2
                  + pulse_cfg.drive_detuning) 
    elif targeted_drive=='alt':
        omega_d = system.freq(detuned_transitions[0], detuned_transitions[1]) + drive_detuning
    else:
        raise Exception('Not a valid targeted drive point')

    omega_d = np.abs(omega_d)

    if not mute:
        print('\nDrive frequency: {:.4f} GHz'.format(omega_d))
        print('Drive amplitude scale factor: {:.4f}'.format(drive_amplitude_factor))

    t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    if solve_method == 'sesolve':
        # This calculates the evolution operator that works for computational levels only.
        U_t = gates.evolution_compspace_microwave(
            system.H(),
            H_drive,
            comp_space=comp_space,
            t_points=t_points,T_gate=T_gate,
            shape=shape,
            sigma=sigma,
            omega_d=omega_d,
            interaction=interaction)
    elif solve_method == 'propagator':
        # This calculates the evolution operator for the whole system  
        U_t = gates.evolution_operator_microwave(system.H(),
            H_drive,
            comp_space=comp_space,
            t_points=t_points,
            T_gate=T_gate,
            shape=shape,
            sigma=sigma,
            DRAG=drag,
            DRAG_coefficient=drag_coeff,
            omega_d=omega_d,
            interaction=interaction,
            T_rise=T_rise)

    U_real = gates.change_operator_proj_subspace(system, U_t, subspace=comp_space, interaction=interaction)
    fidelity = gates.fidelity_cz_gate(system, U_t, comp_space=comp_space, interaction=interaction, single_gates='z')
    # single_qubit_gates = gates.operator_single_qub_z(system, U_real[-1])
    U_f = U_t[-1]
    U_me = {}
    for state in comp_space:
        vec = system.eigvec(state, interaction=interaction)
        U_me[state] = U_f.matrix_element(vec.dag(), vec)
    phase_accum = (np.angle(U_me[comp_space[0]]) + np.angle(U_me[comp_space[3]])
                 - np.angle(U_me[comp_space[1]]) - np.angle(U_me[comp_space[2]]))
    phase_accum = phase_accum / np.pi

    if not mute:
        #Note: this is only for unitary evolution. We shall investigate dephasing errors later.
        print('\nMax fidelity during the simulations: ', np.max(fidelity), 'at', np.argmax(fidelity)/2, 'ns')
        print('** Final values **')
        print('Fidelity: ', fidelity[-1])
        print('\nDiagonal elements of the evolution operator ' +
            '(amplitudes and phases with respect to E*t in units of pi)')
        for state in comp_space:
            print(state, np.abs(U_me[state]),
                (np.angle(U_me[state] * np.exp(2j * np.pi * system.level(state) * T_gate))) / np.pi)
        display(Latex(r'$(\phi_{{{}}} + \phi_{{{}}} - \phi_{{{}}} - \phi_{{{}}})/\pi=$ {}'.format(
            comp_space[0], comp_space[3], comp_space[1], comp_space[2], phase_accum)))

        initial_state = system.eigvec(transitions_to_drive[0])
        final_state = system.eigvec(transitions_to_drive[1])
        P_driven_transition = gates.prob_transition(U_t, initial_state, final_state)
        t_2nd_excited = scipy.integrate.trapezoid(P_driven_transition, t_points)
        print('Time spent in the 2nd state for {} - {}: {:.1f} ns'.format(
            transitions_to_drive[0], transitions_to_drive[1], t_2nd_excited))
        # Like an integrated time so should be proportional to phase accumulation!
    return t_points, U_t, phase_accum, fidelity

def minimize_infidelity(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False, x0=[0,0]):
    def infidelity(x):
        pulse_cfg.drive_detuning, pulse_cfg.DRAG = x
        _, _, _, fidelity = solve(system, pulse_cfg, system_cfg, solve_method, mute=True)
        last_infidelity = 1-fidelity[-1]
        print(last_infidelity)
        return last_infidelity
    
    xopt = minimize(infidelity, x0, method='Nelder-Mead')
    return xopt, infidelity(x=xopt.x)

def converge_on_pi(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False):
    def cphase_pi_error(x):
        _, _, phase_accum, _ = solve(system, pulse_cfg, system_cfg, solve_method, mute=True)
        print(phase_accum)
        return (np.pi - phase_accum)%(2*np.pi)
    
    xopt = minimize(cphase_pi_error, pulse_cfg.T_gate, method='Nelder-Mead')
    return xopt, cphase_pi_error(x=xopt.x)

def visualize_state_propagation(
        system:CoupledObjects,
        system_cfg:SystemConfig,
        t_points,
        U_t,
        phase_accum,
        fidelity,
        n_shown_states):
    # Return graphs with all the relevant details

    qubitA:AbstractQubit = system._objects[0]
    qubitB:AbstractQubit = system._objects[1]
    comp_space = system_cfg.computational_space

    # Printing relevant parameters
    plt_00 = {}
    plt_01 = {}
    plt_10 = {}
    plt_11 = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    fig.suptitle('{} {} coupled to {}'.format(qubitA.name, system_cfg.coupling_type, qubitB.name))

    ax000 = axes[0, 0]
    ax001 = axes[0, 1]
    ax010 = axes[1, 0]
    ax011 = axes[1, 1]

    all_states_str = []
    for i in range(qubitA.nlev):
        for j in range(qubitB.nlev):
            all_states_str.append(f"{i}{j}")

    # All plots
    max_final_amp = {}
    for state in all_states_str: # a-priori we don't know what positions to show, so let's search through all of them, and choose to display the ones with the highest final values (that aren't the identity case)
        plt_00[state] = gates.prob_transition(U_t, system.eigvec(comp_space[0]), system.eigvec(state))
        max_final_amp[state] = np.max(plt_00[state]) # plt_00[state][-1] # NOTE: Maybe we want to actually show the states with highest intermediate values? (i.e. np.max(plt_00[state])
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax000.scatter(t_points, plt_00[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[0], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_01[state] = gates.prob_transition(U_t, system.eigvec(comp_space[1]), system.eigvec(state))
        max_final_amp[state] = plt_01[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax001.scatter(t_points, plt_01[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[1], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_10[state] = gates.prob_transition(U_t, system.eigvec(comp_space[2]), system.eigvec(state))
        max_final_amp[state] = plt_10[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax010.scatter(t_points, plt_10[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[2], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_11[state] = gates.prob_transition(U_t, system.eigvec(comp_space[3]), system.eigvec(state))
        max_final_amp[state] = plt_11[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax011.scatter(t_points, plt_11[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[3], key))

    # General information
    textfontsize = 18
    fig.text(0.5, 0.16, r'At $t = {}$ ns: '.format(int(t_points[-1])),
            fontsize=textfontsize, ha='center')
    fig.text(0.5, 0.13,
            r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[0],
                comp_space[0],
                plt_00[comp_space[0]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[1],
                comp_space[1],
                plt_01[comp_space[1]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[2],
                comp_space[2],
                plt_10[comp_space[2]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$'.format(
                comp_space[3],
                comp_space[3],
                plt_11[comp_space[3]][-1]),
            fontsize=textfontsize, ha='center')

    # In plot text for final values
    ax000.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[0],
                comp_space[0], 
                plt_00[comp_space[0]][-1]),
            ha='right', va='top', transform=ax000.transAxes,
            fontsize=textfontsize)
    ax001.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[1],
                comp_space[1],
                plt_01[comp_space[1]][-1]),
            ha='right', va='top', transform=ax001.transAxes,
            fontsize=textfontsize)
    ax010.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[2],
                comp_space[2],
                plt_10[comp_space[2]][-1]),
            ha='right', va='top', transform=ax010.transAxes,
            fontsize=textfontsize)
    ax011.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[3],
                comp_space[3],
                plt_11[comp_space[3]][-1]),
            ha='right', va='top', transform=ax011.transAxes,
            fontsize=textfontsize)

    # Below plots text for phase and fidelity
    fig.text(0.5, 0.1,
            r'CZ gate phase accumulation: '
            + r'$\phi_{{{}}} + \phi_{{{}}} - \phi_{{{}}} - \phi_{{{}}} = $'.format(
                comp_space[0],
                comp_space[3],
                comp_space[2],
                comp_space[1])
            + r'${:.3f} \pi $'.format(phase_accum),
            fontsize=textfontsize, ha='center');
    fig.text(0.5, 0.05,
            r'Fidelity: '
            + r'$F = {:.6f}$'.format(fidelity[-1]),
            fontsize=textfontsize, ha='center')

    for axarr in axes:
        for ax in axarr:
            ax.legend(loc='lower left')
            ax.set_xlim([np.min(t_points), np.max(t_points)])
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(r'$P_{i\rightarrow f}$')

    ax000.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[0]))
    ax001.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[1]))
    ax010.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[2]))
    ax011.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[3]))

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    
    # TODO: Add meta data to figure
    return fig

def visualize_lost_trace():
    # Graphs trace over time to show losses (one plot is better here)
    return

def solve_coupled_qubits(qubitA, qubitB, pulse_path:str, syscfg_path:str, n_shown_states=3):
    #combined function that concatenates all the previous functions. Typical entry point

    with open(pulse_path, 'r') as f:
        data = yaml.safe_load(f)
        pulse_cfg = PulseConfig(**data)

    with open(syscfg_path, 'r') as f:
        data = yaml.safe_load(f)
        system_cfg = SystemConfig(**data)
        del data
 
    qubitA, qubitB = load_qubits(qubitA, qubitB) # TODO: is this order correct?
    qubitA, qubitB = scale_qubitA_transition(qubitA, qubitB, system_cfg)
    system = couple_qubits(qubitA, qubitB, system_cfg, pulse_cfg, mute=False)
    t_points, U_t, phase_accum, fidelity = solve(system, pulse_cfg, system_cfg, solve_method='propagator', mute=False)
    fig = visualize_state_propagation(system, system_cfg, t_points, U_t, phase_accum, fidelity, n_shown_states=n_shown_states)
    # visualize_lost_trace()
    return fig