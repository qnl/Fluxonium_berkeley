"""Scientific validation of the frozen low-leakage simultaneous X180 pulse."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from Circuit_Objs.qchard_statetracking import track_eigensystems
from Simulations.Rabi_3Photon import calibrate_simultaneous_x180 as calibration
from Simulations.Rabi_3Photon import compare_x180_protocols as comparison
from Simulations.Rabi_3Photon import workflow_funcs as workflow


DEFAULT_CONFIG = comparison.DEFAULT_CONFIG
TIGHT_OPTIONS = calibration.VERIFICATION_SOLVER_OPTIONS
TIGHTER_OPTIONS = {
    **TIGHT_OPTIONS,
    'rtol': 1e-12,
    'atol': 1e-14,
}


def _candidate_vector(experiment):
    candidate = experiment['validated_candidate']
    return np.array([
        candidate['amplitude_05'], candidate['amplitude_54'],
        candidate['amplitude_41'], candidate['phase_54'],
        candidate['phase_41'], candidate['duration'],
        candidate['detuning_05'], candidate['detuning_54'],
        candidate['detuning_41']], dtype=float)


def _qubit(experiment, nlev, nlev_lc):
    model = dict(experiment['model'])
    model.update(nlev=nlev, nlev_lc=nlev_lc)
    return comparison._build_qubit(model)


def _time_points(duration, samples_per_ns):
    intervals = int(np.ceil(duration * samples_per_ns))
    return np.linspace(0, duration, intervals + 1)


def _align_to_reference_basis(propagators, reference_qubit, qubit):
    tracking = track_eigensystems(
        reference_qubit.levels(nlev=qubit.nlev),
        reference_qubit.eigvecs(nlev=qubit.nlev),
        qubit.levels(), qubit.eigvecs(),
        minimum_overlap=0.5,
        ambiguity_margin=0.05,
        degeneracy_atol=1e-4,
        degeneracy_rtol=1e-9)
    if tracking.has_unresolved_ambiguity:
        raise ValueError(
            'Unresolved eigensystem tracking ambiguity: {}'.format(
                tracking.ambiguities))
    return np.asarray([
        tracking.transform_operator(propagator)
        for propagator in propagators
    ])


def evaluate_case(
        qubit, experiment, parameters, samples_per_ns,
        solver_options=TIGHT_OPTIONS, logical_basis_reference=None):
    t_points = _time_points(parameters[5], samples_per_ns)
    with contextlib.redirect_stderr(io.StringIO()):
        _, propagators = workflow.solve_pulse_schedule(
            qubit, calibration._pulse_configs(experiment, parameters),
            mute=True, solver_options=solver_options, t_points=t_points)
    interaction_propagators = comparison._interaction_picture(
        qubit, t_points, propagators)
    if logical_basis_reference is not None:
        interaction_propagators = _align_to_reference_basis(
            interaction_propagators, logical_basis_reference, qubit)
    metrics = workflow.evaluate_logical_gate(
        interaction_propagators, experiment['target'], t_points=t_points)
    final = interaction_propagators[-1]
    unitarity_residual = float(np.max(np.abs(
        (final.dag() * final).full() - np.eye(qubit.nlev))))
    return {
        'qubit': qubit,
        't_points': t_points,
        'propagators': interaction_propagators,
        'metrics': metrics,
        'unitarity_residual': unitarity_residual,
    }


def _metric_row(result):
    metrics = result['metrics']
    return (
        metrics['logical_gate_fidelity'], metrics['final_leakage'],
        metrics['peak_leakage'], result['unitarity_residual'])


def _coherence_checks(result, target_name):
    propagator = result['propagators'][-1].full()
    dimension = propagator.shape[0]
    target = workflow.logical_gate_target(target_name).full()
    logical_inputs = {
        '0': np.array([1, 0], dtype=complex),
        '1': np.array([0, 1], dtype=complex),
        'plus': np.array([1, 1], dtype=complex) / np.sqrt(2),
        'plus_i': np.array([1, 1j], dtype=complex) / np.sqrt(2),
    }
    checks = {}
    for name, logical_input in logical_inputs.items():
        initial = np.zeros(dimension, dtype=complex)
        initial[:2] = logical_input
        expected = target @ logical_input
        actual = propagator @ initial
        logical_actual = actual[:2]
        leakage = float(np.sum(np.abs(actual[2:]) ** 2))
        survival = float(np.sum(np.abs(logical_actual) ** 2))
        overlap = np.vdot(expected, logical_actual)
        unconditional_fidelity = float(abs(overlap) ** 2)
        conditional_fidelity = (
            0.0 if np.isclose(survival, 0)
            else unconditional_fidelity / survival)
        aligned = logical_actual * np.exp(-1j * np.angle(overlap))
        checks[name] = {
            'unconditional_fidelity': unconditional_fidelity,
            'conditional_fidelity': conditional_fidelity,
            'leakage': leakage,
            'aligned_logical_output': aligned,
            'expected_logical_output': expected,
        }
    return checks


def _level_populations(result):
    final = result['propagators'][-1].full()
    return {
        initial: np.abs(final[:, initial]) ** 2
        for initial in (0, 1)
    }


def _robustness_cases(parameters, validation_config):
    perturbations = validation_config['robustness']
    cases = []
    for index, name in enumerate(calibration.PARAMETER_NAMES):
        if name.startswith('amplitude_'):
            step = abs(parameters[index]) * perturbations['amplitude_fraction']
        elif name.startswith('phase_'):
            step = perturbations['phase_radians']
        elif name.startswith('detuning_'):
            step = perturbations['detuning_ghz']
        elif name == 'duration':
            step = parameters[index] * perturbations['duration_fraction']
        else:
            raise ValueError('No robustness step defined for {}.'.format(name))
        for direction in (-1, 1):
            varied = parameters.copy()
            varied[index] += direction * step
            cases.append((name, direction, step, varied))
    return cases


def run_validation(config_path=DEFAULT_CONFIG):
    experiment = comparison._load_config(config_path)
    validation = experiment['validation']
    parameters = _candidate_vector(experiment)
    max_levels = max(validation['retained_levels'])
    logical_basis_reference = _qubit(experiment, max_levels, 230)

    solver_qubit = _qubit(experiment, 8, 230)
    solver_cases = {}
    for name, options, rate in (
            ('repository_default_2_per_ns', None, 2),
            ('tight_2_per_ns', TIGHT_OPTIONS, 2),
            ('tighter_2_per_ns', TIGHTER_OPTIONS, 2),
            ('tight_4_per_ns', TIGHT_OPTIONS, 4),
            ('tight_8_per_ns', TIGHT_OPTIONS, 8)):
        solver_cases[name] = evaluate_case(
            solver_qubit, experiment, parameters, rate,
            solver_options=options,
            logical_basis_reference=logical_basis_reference)

    retained_level_cases = {}
    for nlev in validation['retained_levels']:
        retained_level_cases[nlev] = evaluate_case(
            _qubit(experiment, nlev, 230), experiment, parameters, 8,
            logical_basis_reference=logical_basis_reference)

    lc_cutoff_cases = {}
    for cutoff in validation['lc_cutoffs']:
        lc_cutoff_cases[cutoff] = evaluate_case(
            _qubit(experiment, max_levels, cutoff),
            experiment, parameters, 8,
            logical_basis_reference=logical_basis_reference)

    reference = lc_cutoff_cases[max(validation['lc_cutoffs'])]
    robustness = []
    for name, direction, step, varied in _robustness_cases(
            parameters, validation):
        result = evaluate_case(
            reference['qubit'], experiment, varied, 4,
            logical_basis_reference=logical_basis_reference)
        robustness.append({
            'parameter': name,
            'direction': direction,
            'step': step,
            'result': result,
        })

    return {
        'experiment': experiment,
        'parameters': parameters,
        'solver_cases': solver_cases,
        'retained_level_cases': retained_level_cases,
        'lc_cutoff_cases': lc_cutoff_cases,
        'reference': reference,
        'coherence': _coherence_checks(reference, experiment['target']),
        'level_populations': _level_populations(reference),
        'robustness': robustness,
    }


def _print_metric_cases(title, cases):
    print('\n{}'.format(title))
    print('case | F_gate | final_leak | peak_leak | unitarity_residual')
    for name, result in cases.items():
        print('{} | {:.9f} | {:.9f} | {:.9f} | {:.3e}'.format(
            name, *_metric_row(result)))


def _format_complex(value):
    return '{:+.9f}{:+.9f}j'.format(value.real, value.imag)


def _print_results(validation):
    _print_metric_cases('solver and sampling convergence',
                        validation['solver_cases'])
    _print_metric_cases('retained-level convergence',
                        validation['retained_level_cases'])
    _print_metric_cases('LC-cutoff convergence at nlev=16',
                        validation['lc_cutoff_cases'])

    print('\ncoherence validation at nlev=16, nlev_lc=460')
    print('input | unconditional_F | conditional_F | leakage '
          '| aligned_output | expected_output')
    for name, check in validation['coherence'].items():
        aligned = ','.join(
            _format_complex(value)
            for value in check['aligned_logical_output'])
        expected = ','.join(
            _format_complex(value)
            for value in check['expected_logical_output'])
        print('{} | {:.9f} | {:.9f} | {:.9f} | [{}] | [{}]'.format(
            name, check['unconditional_fidelity'],
            check['conditional_fidelity'], check['leakage'],
            aligned, expected))

    print('\nfinal populations by retained level at nlev=16, nlev_lc=460')
    print('level | input_0 | input_1')
    populations = validation['level_populations']
    for level in range(len(populations[0])):
        print('{} | {:.12f} | {:.12f}'.format(
            level, populations[0][level], populations[1][level]))

    baseline = validation['reference']
    baseline_metrics = baseline['metrics']
    print('\nlocal one-at-a-time robustness at nlev=16, nlev_lc=460')
    print('parameter | direction | step | F_gate | delta_F '
          '| final_leak | delta_final_leak | peak_leak')
    for case in validation['robustness']:
        metrics = case['result']['metrics']
        print('{} | {:+d} | {:.9g} | {:.9f} | {:+.9f} '
              '| {:.9f} | {:+.9f} | {:.9f}'.format(
                  case['parameter'], case['direction'], case['step'],
                  metrics['logical_gate_fidelity'],
                  metrics['logical_gate_fidelity']
                  - baseline_metrics['logical_gate_fidelity'],
                  metrics['final_leakage'],
                  metrics['final_leakage']
                  - baseline_metrics['final_leakage'],
                  metrics['peak_leakage']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    _print_results(run_validation(arguments.config))


if __name__ == '__main__':
    main()
