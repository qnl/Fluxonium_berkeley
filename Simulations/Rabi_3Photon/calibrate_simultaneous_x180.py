"""Modest Pareto search for the simultaneous 0-5-4-1 X180 protocol."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.stats import qmc, spearmanr

from Simulations.Rabi_3Photon import compare_x180_protocols as comparison
from Simulations.Rabi_3Photon import workflow_funcs as workflow


DEFAULT_CONFIG = comparison.DEFAULT_CONFIG
TRANSITION_NAMES = ('outer_05', 'middle_54', 'outer_41')
PARAMETER_NAMES = (
    'amplitude_05', 'amplitude_54', 'amplitude_41',
    'phase_54', 'phase_41', 'duration',
    'detuning_05', 'detuning_54', 'detuning_41')
VERIFICATION_SOLVER_OPTIONS = {
    'nsteps': 100000,
    'progress_bar': False,
    'normalize_output': False,
    'method': 'vern9',
    'rtol': 1e-10,
    'atol': 1e-12,
}


def _bounds(search_config):
    amplitude = search_config['amplitude_bounds']
    phase = search_config['relative_phase_bounds']
    duration = search_config['duration_bounds']
    detuning = search_config['detuning_bounds']
    upper = np.array([
        amplitude[1], amplitude[1], amplitude[1],
        phase[1], phase[1], duration[1],
        detuning[1], detuning[1], detuning[1]], dtype=float)
    lower = np.array([
        amplitude[0], amplitude[0], amplitude[0],
        phase[0], phase[0], duration[0],
        detuning[0], detuning[0], detuning[0]], dtype=float)
    return lower, upper


def _baseline_vector(experiment):
    if 'search_start' in experiment:
        return np.array([
            experiment['search_start'][name]
            for name in PARAMETER_NAMES], dtype=float)
    pulse = experiment['pulse']
    return np.array([
        pulse['amplitude'], pulse['amplitude'], pulse['amplitude'],
        0.0, 0.0, pulse['duration'],
        pulse['detuning'], pulse['detuning'], pulse['detuning']],
        dtype=float)


def _pulse_configs(experiment, parameters):
    pulse = experiment['pulse']
    transitions = experiment['transitions']
    amplitudes = parameters[:3]
    phases = (0.0, parameters[3], parameters[4])
    duration = parameters[5]
    detunings = parameters[6:]
    return [
        workflow.PulseConfig(
            T_gate=duration,
            T_start=0.0,
            pulse_shape=pulse['shape'],
            pulse_sigma=pulse['sigma'],
            targeted_drive=transitions[name],
            drive_amplitude_factor=amplitude,
            drive_detuning=detuning,
            carrier_phase=phase,
            drive_type='flux')
        for name, amplitude, phase, detuning in zip(
            TRANSITION_NAMES, amplitudes, phases, detunings)
    ]


def evaluate_candidate(
        qubit, experiment, parameters, candidate_id, solver_options=None):
    with contextlib.redirect_stderr(io.StringIO()):
        t_points, propagators = workflow.solve_pulse_schedule(
            qubit, _pulse_configs(experiment, parameters), mute=True,
            solver_options=solver_options)
    interaction_propagators = comparison._interaction_picture(
        qubit, t_points, propagators)
    metrics = workflow.evaluate_logical_gate(
        interaction_propagators, experiment['target'], t_points=t_points)

    states = {}
    for initial in (0, 1):
        diagnostics = metrics['state_diagnostics'][initial]
        states[initial] = {
            'final_P0': float(diagnostics['logical_populations'][-1, 0]),
            'final_P1': float(diagnostics['logical_populations'][-1, 1]),
            'final_leakage': diagnostics['final_leakage'],
            'peak_leakage': diagnostics['peak_leakage'],
        }
    return {
        'id': candidate_id,
        'parameters': np.asarray(parameters, dtype=float),
        'logical_gate_fidelity': metrics['logical_gate_fidelity'],
        'logical_process_fidelity': metrics['logical_process_fidelity'],
        'final_leakage': metrics['final_leakage'],
        'peak_leakage': metrics['peak_leakage'],
        'states': states,
    }


def _select_anchors(results, search_config):
    best_fidelity = max(results, key=lambda result:
                        result['logical_gate_fidelity'])
    balanced = [
        result for result in results
        if result['final_leakage']
        <= search_config['balanced_final_leakage_limit']]
    best_balanced = max(
        balanced, key=lambda result: result['logical_gate_fidelity'])
    fidelity_qualified = [
        result for result in results
        if result['logical_gate_fidelity']
        >= search_config['baseline_fidelity']]
    lowest_leakage = min(
        fidelity_qualified, key=lambda result: result['final_leakage'])

    anchors = []
    for candidate in (best_fidelity, best_balanced, lowest_leakage):
        if candidate['id'] not in {anchor['id'] for anchor in anchors}:
            anchors.append(candidate)
    return anchors


def _local_samples(anchor, search_config, count, seed, lower, upper):
    widths = search_config['local_half_widths']
    half_width = np.array([
        widths['amplitude'], widths['amplitude'], widths['amplitude'],
        widths['relative_phase'], widths['relative_phase'],
        widths['duration'], widths['detuning'], widths['detuning'],
        widths['detuning']], dtype=float)
    local_lower = np.maximum(lower, anchor['parameters'] - half_width)
    local_upper = np.minimum(upper, anchor['parameters'] + half_width)
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=seed)
    return qmc.scale(sampler.random(count), local_lower, local_upper)


def pareto_front(results):
    """Return candidates not dominated in fidelity and final leakage."""
    front = []
    for candidate in results:
        dominated = any(
            (other['logical_gate_fidelity']
             >= candidate['logical_gate_fidelity']
             and other['final_leakage'] <= candidate['final_leakage'])
            and (other['logical_gate_fidelity']
                 > candidate['logical_gate_fidelity']
                 or other['final_leakage'] < candidate['final_leakage'])
            for other in results)
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda result: result['final_leakage'])


def _parameter_correlations(results):
    parameters = np.asarray([result['parameters'] for result in results])
    outputs = {
        'fidelity': np.asarray([
            result['logical_gate_fidelity'] for result in results]),
        'final_leakage': np.asarray([
            result['final_leakage'] for result in results]),
    }
    features = {
        name: parameters[:, index]
        for index, name in enumerate(PARAMETER_NAMES)
        if not name.startswith('phase_')
    }
    for index, name in ((3, 'phase_54'), (4, 'phase_41')):
        features[name] = np.column_stack((
            np.sin(parameters[:, index]), np.cos(parameters[:, index])))

    correlations = {}
    for feature_name, values in features.items():
        correlations[feature_name] = {}
        for output_name, output in outputs.items():
            if values.ndim == 1:
                rho = spearmanr(values, output).statistic
            else:
                rho = max(
                    abs(spearmanr(values[:, column], output).statistic)
                    for column in range(values.shape[1]))
            correlations[feature_name][output_name] = float(abs(rho))
    return correlations


def run_search(config_path=DEFAULT_CONFIG):
    experiment = comparison._load_config(config_path)
    search_config = experiment['search']
    qubit = comparison._build_qubit(experiment['model'])
    lower, upper = _bounds(search_config)

    sampler = qmc.LatinHypercube(
        d=len(PARAMETER_NAMES), seed=search_config['seed'])
    global_samples = qmc.scale(
        sampler.random(search_config['global_samples']), lower, upper)
    samples = np.vstack((_baseline_vector(experiment), global_samples))
    results = [
        evaluate_candidate(qubit, experiment, parameters, 'global-{}'.format(index))
        for index, parameters in enumerate(samples)
    ]

    anchors = _select_anchors(results, search_config)
    for anchor_index, anchor in enumerate(anchors):
        local_samples = _local_samples(
            anchor, search_config,
            search_config['local_samples_per_anchor'],
            search_config['seed'] + anchor_index + 1, lower, upper)
        results.extend(
            evaluate_candidate(
                qubit, experiment, parameters,
                'local-{}-{}'.format(anchor_index, sample_index))
            for sample_index, parameters in enumerate(local_samples)
        )
    coarse_front = pareto_front(results)
    verified_results = [
        evaluate_candidate(
            qubit, experiment, candidate['parameters'], candidate['id'],
            solver_options=VERIFICATION_SOLVER_OPTIONS)
        for candidate in coarse_front
    ]
    return (
        experiment, results, pareto_front(verified_results),
        _parameter_correlations(results))


def _format_parameters(candidate):
    return ', '.join(
        '{}={:.9g}'.format(name, value)
        for name, value in zip(PARAMETER_NAMES, candidate['parameters']))


def _print_candidate(label, candidate):
    print('\n{}'.format(label))
    print('id={} F_gate={:.9f} final_leak={:.9f} peak_leak={:.9f}'.format(
        candidate['id'], candidate['logical_gate_fidelity'],
        candidate['final_leakage'], candidate['peak_leakage']))
    print(_format_parameters(candidate))
    for initial, state in candidate['states'].items():
        print('input={} P0={:.9f} P1={:.9f} final_leak={:.9f} '
              'peak_leak={:.9f}'.format(
                  initial, state['final_P0'], state['final_P1'],
                  state['final_leakage'], state['peak_leakage']))


def _print_results(experiment, results, front, correlations):
    search_config = experiment['search']
    best_fidelity = max(front, key=lambda result:
                        result['logical_gate_fidelity'])
    lowest_leakage = min(front, key=lambda result: result['final_leakage'])
    balanced = max(
        (result for result in front
         if result['final_leakage']
         <= search_config['balanced_final_leakage_limit']),
        key=lambda result: result['logical_gate_fidelity'])
    qualified_low_leakage = min(
        (result for result in front
         if result['logical_gate_fidelity']
         >= search_config['baseline_fidelity']),
        key=lambda result: result['final_leakage'])

    print('evaluated_candidates={} pareto_candidates={}'.format(
        len(results), len(front)))
    _print_candidate('highest fidelity', best_fidelity)
    _print_candidate('lowest final leakage (unconstrained)', lowest_leakage)
    _print_candidate('highest fidelity with final leakage <= 0.1', balanced)
    _print_candidate(
        'lowest leakage with fidelity >= fixed baseline',
        qualified_low_leakage)

    print('\nPareto front: F_gate | final_leak | peak_leak | id')
    for candidate in front:
        print('{:.9f} | {:.9f} | {:.9f} | {}'.format(
            candidate['logical_gate_fidelity'],
            candidate['final_leakage'], candidate['peak_leakage'],
            candidate['id']))
        print('  {}'.format(_format_parameters(candidate)))

    print('\nAbsolute Spearman correlations: parameter | fidelity | final_leakage')
    ordered = sorted(
        correlations.items(),
        key=lambda item: max(item[1].values()), reverse=True)
    for name, values in ordered:
        print('{} | {:.3f} | {:.3f}'.format(
            name, values['fidelity'], values['final_leakage']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    experiment, results, front, correlations = run_search(arguments.config)
    _print_results(experiment, results, front, correlations)


if __name__ == '__main__':
    main()
