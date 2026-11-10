"""One-mode IdealGridium bias-offset diagnostic for frozen X gates.

This is deliberately not a fabrication-asymmetry model.  It changes only the
existing ``phi_ext`` parameter of the one-mode IdealGridium Hamiltonian and
compares the already validated X90 and X180 pulse candidates.  State labels
and eigenvector phases are tracked relative to the protected-point
``phi_ext = pi`` eigensystem.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import replace
from pathlib import Path

import numpy as np

from Circuit_Objs.qchard_statetracking import track_eigensystems
from Simulations.Rabi_3Photon import calibrate_simultaneous_x180 as calibration
from Simulations.Rabi_3Photon import compare_x180_protocols as comparison
from Simulations.Rabi_3Photon import workflow_funcs as workflow


CONFIG_DIRECTORY = Path(__file__).parent / 'yamls' / 'experiments'
DEFAULT_CONFIGS = {
    'X90': CONFIG_DIRECTORY / 'idealgridium_soft_candidate_x90.yaml',
    'X180': CONFIG_DIRECTORY / 'idealgridium_soft_candidate_x180.yaml',
}
DEFAULT_OFFSETS = (-0.01, -0.005, -0.0025, -0.001, 0.0,
                   0.001, 0.0025, 0.005, 0.01)
TRANSITION_NAMES = ('outer_05', 'middle_54', 'outer_41')
DOUBLET_PAIRS = ((0, 1), (2, 3), (4, 5))
TRACKING_OPTIONS = {
    'minimum_overlap': 0.5,
    'ambiguity_margin': 0.05,
    'degeneracy_atol': 1e-4,
    'degeneracy_rtol': 1e-9,
}


def _candidate_vector(experiment):
    candidate = experiment['validated_candidate']
    return np.asarray([
        candidate[name] for name in calibration.PARAMETER_NAMES], dtype=float)


def _qubit(experiment, phi_ext, nlev, nlev_lc):
    model = dict(experiment['model'])
    model.update(phi_ext=float(phi_ext), nlev=nlev, nlev_lc=nlev_lc)
    return comparison._build_qubit(model)


def _track(reference_qubit, qubit):
    return track_eigensystems(
        reference_qubit.levels(), reference_qubit.eigvecs(),
        qubit.levels(), qubit.eigvecs(), **TRACKING_OPTIONS)


def _tracked_transition(transition, tracking):
    return tuple(int(tracking.permutation[index]) for index in transition)


def _pulse_configs(
        experiment, parameters, reference_qubit, qubit, tracking, *,
        follow_transitions=False, phase_corrections=(0.0, 0.0),
        detuning_corrections=(0.0, 0.0, 0.0)):
    """Build a physically frozen pulse, optionally following bare carriers.

    The validated workflow normalizes each tone's flux operator by its target
    matrix element.  To freeze the physical operator scale as ``phi_ext`` is
    varied, the numerical amplitude is compensated by the ratio of the new
    and reference matrix elements.  Thus the effective coefficient of
    ``qubit.phi()`` is unchanged.  ``follow_transitions`` changes only each
    absolute carrier from its reference transition to the corresponding
    tracked transition at the offset.
    """
    configs = calibration._pulse_configs(experiment, parameters)
    reference_phi = reference_qubit.phi()
    phi = qubit.phi()
    updated = []
    for index, (name, config) in enumerate(zip(TRANSITION_NAMES, configs)):
        reference_transition = tuple(experiment['transitions'][name])
        transition = _tracked_transition(reference_transition, tracking)
        reference_matrix_element = abs(
            reference_phi[reference_transition[0], reference_transition[1]])
        matrix_element = abs(phi[transition[0], transition[1]])
        if np.isclose(reference_matrix_element, 0.0):
            raise ValueError(
                'Reference global-phase-control normalization is undefined for {}.'
                .format(reference_transition))

        base_frequency = abs(reference_qubit.freq(*reference_transition))
        if follow_transitions:
            base_frequency = abs(qubit.freq(*transition))
        phase_correction = 0.0 if index == 0 else phase_corrections[index - 1]
        updated.append(replace(
            config,
            targeted_drive=transition,
            drive_frequency=base_frequency,
            drive_amplitude_factor=(
                config.drive_amplitude_factor
                * matrix_element / reference_matrix_element),
            drive_detuning=(
                config.drive_detuning + detuning_corrections[index]),
            carrier_phase=config.carrier_phase + phase_correction))
    return updated


def _time_points(duration, samples_per_ns):
    intervals = int(np.ceil(duration * samples_per_ns))
    return np.linspace(0.0, duration, intervals + 1)


def _tracking_summary(tracking, relevant_states=range(6)):
    relevant = set(relevant_states)
    records = []
    for ambiguity in tracking.ambiguities:
        states = ambiguity.get('reference_states')
        if states is None:
            states = (ambiguity.get('reference_state'),)
        if relevant.intersection(states):
            records.append(ambiguity)
    return records


def evaluate_offset(
        experiment, parameters, reference_qubit, qubit, tracking, *,
        follow_transitions=False, phase_corrections=(0.0, 0.0),
        detuning_corrections=(0.0, 0.0, 0.0), samples_per_ns=8,
        solver_options=calibration.VERIFICATION_SOLVER_OPTIONS):
    configs = _pulse_configs(
        experiment, parameters, reference_qubit, qubit, tracking,
        follow_transitions=follow_transitions,
        phase_corrections=phase_corrections,
        detuning_corrections=detuning_corrections)
    t_points = _time_points(parameters[5], samples_per_ns)
    with contextlib.redirect_stderr(io.StringIO()):
        _, propagators = workflow.solve_pulse_schedule(
            qubit, configs, mute=True, solver_options=solver_options,
            t_points=t_points)
    interaction = comparison._interaction_picture(
        qubit, t_points, propagators)
    tracked_propagators = np.asarray([
        tracking.transform_operator(propagator)
        for propagator in interaction
    ])
    metrics = workflow.evaluate_logical_gate(
        tracked_propagators, experiment['target'], t_points=t_points)
    states = {}
    for initial in (0, 1):
        diagnostic = metrics['state_diagnostics'][initial]
        states[initial] = {
            'final_P0': float(diagnostic['logical_populations'][-1, 0]),
            'final_P1': float(diagnostic['logical_populations'][-1, 1]),
            'final_leakage': diagnostic['final_leakage'],
            'peak_leakage': diagnostic['peak_leakage'],
        }
    return {
        'logical_gate_fidelity': metrics['logical_gate_fidelity'],
        'final_leakage': metrics['final_leakage'],
        'peak_leakage': metrics['peak_leakage'],
        'duration': float(parameters[5]),
        'states': states,
        'phase_corrections': tuple(phase_corrections),
        'detuning_corrections': tuple(detuning_corrections),
    }


def _spectrum_summary(experiment, reference_qubit, qubit, tracking):
    transitions = {}
    for name in TRANSITION_NAMES:
        reference_transition = tuple(experiment['transitions'][name])
        transition = _tracked_transition(reference_transition, tracking)
        transitions[name] = abs(qubit.freq(*transition))
    levels = qubit.levels()
    doublets = {}
    for initial, final in DOUBLET_PAIRS:
        tracked_initial, tracked_final = _tracked_transition(
            (initial, final), tracking)
        doublets['{}{}'.format(initial, final)] = abs(
            levels[tracked_final] - levels[tracked_initial])
    return {
        'transition_frequencies': transitions,
        'doublet_splittings': doublets,
        'matched_overlaps': tracking.matched_overlaps[:6].copy(),
        'ambiguities': _tracking_summary(tracking),
    }


def run_diagnostic(
        offsets=DEFAULT_OFFSETS, *, nlev=16, nlev_lc=460,
        samples_per_ns=8):
    experiments = {
        target: comparison._load_config(path)
        for target, path in DEFAULT_CONFIGS.items()
    }
    reference_qubits = {
        target: _qubit(experiment, np.pi, nlev, nlev_lc)
        for target, experiment in experiments.items()
    }
    results = {}
    for offset in offsets:
        results[offset] = {}
        for target, experiment in experiments.items():
            reference_qubit = reference_qubits[target]
            qubit = _qubit(experiment, np.pi + offset, nlev, nlev_lc)
            tracking = _track(reference_qubit, qubit)
            parameters = _candidate_vector(experiment)
            results[offset][target] = {
                'spectrum': _spectrum_summary(
                    experiment, reference_qubit, qubit, tracking),
                'frozen': evaluate_offset(
                    experiment, parameters, reference_qubit, qubit, tracking,
                    follow_transitions=False,
                    samples_per_ns=samples_per_ns),
                'carrier_followed': evaluate_offset(
                    experiment, parameters, reference_qubit, qubit, tracking,
                    follow_transitions=True,
                    samples_per_ns=samples_per_ns),
            }
    return results


def _ambiguity_label(ambiguities):
    if not ambiguities:
        return 'none'
    labels = []
    for ambiguity in ambiguities:
        states = ambiguity.get(
            'reference_states', (ambiguity.get('reference_state'),))
        status = 'resolved' if ambiguity.get('resolved', False) else 'flagged'
        labels.append('{}:{}:{}'.format(
            ambiguity['kind'], '-'.join(str(state) for state in states), status))
    return ','.join(labels)


def _print_results(results):
    print('One-mode IdealGridium bias-offset diagnostic (not fabrication asymmetry)')
    print('delta_phi | target | mode | F_gate | final_leak | peak_leak '
          '| duration_ns | d01_GHz | d23_GHz | d45_GHz | f05_GHz '
          '| f54_GHz | f41_GHz | min_overlap_0_5 | tracking')
    for offset, target_results in results.items():
        for target, target_result in target_results.items():
            spectrum = target_result['spectrum']
            doublets = spectrum['doublet_splittings']
            frequencies = spectrum['transition_frequencies']
            minimum_overlap = float(np.min(spectrum['matched_overlaps']))
            ambiguity = _ambiguity_label(spectrum['ambiguities'])
            for mode in ('frozen', 'carrier_followed'):
                result = target_result[mode]
                print('{:+.6f} | {} | {} | {:.9f} | {:.9f} | {:.9f} '
                      '| {:.7f} | {:.9f} | {:.9f} | {:.9f} | {:.9f} '
                      '| {:.9f} | {:.9f} | {:.6f} | {}'.format(
                          offset, target, mode,
                          result['logical_gate_fidelity'],
                          result['final_leakage'], result['peak_leakage'],
                          result['duration'], doublets['01'], doublets['23'],
                          doublets['45'], frequencies['outer_05'],
                          frequencies['middle_54'], frequencies['outer_41'],
                          minimum_overlap, ambiguity))

    print('\nstate-resolved results')
    print('delta_phi | target | mode | input | P0 | P1 | final_leak | peak_leak')
    for offset, target_results in results.items():
        for target, target_result in target_results.items():
            for mode in ('frozen', 'carrier_followed'):
                for initial, state in target_result[mode]['states'].items():
                    print('{:+.6f} | {} | {} | {} | {:.9f} | {:.9f} '
                          '| {:.9f} | {:.9f}'.format(
                              offset, target, mode, initial,
                              state['final_P0'], state['final_P1'],
                              state['final_leakage'], state['peak_leakage']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nlev', type=int, default=16)
    parser.add_argument('--nlev-lc', type=int, default=460)
    parser.add_argument('--samples-per-ns', type=float, default=8)
    parser.add_argument('--offsets', type=float, nargs='+', default=DEFAULT_OFFSETS)
    arguments = parser.parse_args()
    results = run_diagnostic(
        arguments.offsets, nlev=arguments.nlev, nlev_lc=arguments.nlev_lc,
        samples_per_ns=arguments.samples_per_ns)
    _print_results(results)


if __name__ == '__main__':
    main()
