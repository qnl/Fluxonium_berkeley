"""Run the first fixed-parameter comparison for the 0-5-4-1 pathway."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from Circuit_Objs.qchard_idealgridium import IdealGridium
from Simulations.Rabi_3Photon import workflow_funcs as workflow


DEFAULT_CONFIG = (
    Path(__file__).parent / 'yamls' / 'experiments'
    / 'idealgridium_soft_candidate_x180.yaml')


def _load_config(path):
    with Path(path).open('r') as stream:
        return yaml.safe_load(stream)


def _build_qubit(model_config):
    parameters = dict(model_config)
    if parameters.get('phi_ext') == 'pi':
        parameters['phi_ext'] = np.pi
    return IdealGridium(**parameters)


def _pulse_event(pulse_config, transition, event):
    return workflow.PulseConfig(
        T_gate=pulse_config['duration'],
        T_start=event['start'],
        pulse_shape=pulse_config['shape'],
        pulse_sigma=pulse_config['sigma'],
        targeted_drive=transition,
        drive_amplitude_factor=pulse_config['amplitude'],
        drive_detuning=pulse_config['detuning'],
        carrier_phase=pulse_config['phase'],
        drive_type='flux')


def _interaction_picture(qubit, t_points, propagators):
    hamiltonian = qubit.H()
    return np.asarray([
        (1j * 2 * np.pi * hamiltonian * time).expm() * propagator
        for time, propagator in zip(t_points, propagators)
    ])


def _population_summary(propagators, t_points, initial, level):
    population = np.asarray([
        abs(propagator[level, initial]) ** 2
        for propagator in propagators], dtype=float)
    peak_index = int(np.argmax(population))
    return {
        'final': float(population[-1]),
        'peak': float(population[peak_index]),
        'peak_time': float(t_points[peak_index]),
    }


def evaluate_protocol(qubit, experiment, protocol_name):
    events = experiment['protocols'][protocol_name]
    pulse_config = experiment['pulse']
    transitions = experiment['transitions']
    pulse_events = [
        _pulse_event(
            pulse_config, transitions[event['transition']], event)
        for event in events
    ]
    t_points, propagators = workflow.solve_pulse_schedule(
        qubit, pulse_events, mute=True)
    interaction_propagators = _interaction_picture(
        qubit, t_points, propagators)
    metrics = workflow.evaluate_logical_gate(
        interaction_propagators, experiment['target'], t_points=t_points)

    state_results = {}
    for initial in (0, 1):
        diagnostics = metrics['state_diagnostics'][initial]
        state_results[initial] = {
            'final_P0': float(diagnostics['logical_populations'][-1, 0]),
            'final_P1': float(diagnostics['logical_populations'][-1, 1]),
            'final_leakage': diagnostics['final_leakage'],
            'peak_leakage': diagnostics['peak_leakage'],
            'peak_leakage_time': diagnostics['peak_leakage_time'],
            'level_4': _population_summary(
                interaction_propagators, t_points, initial, 4),
            'level_5': _population_summary(
                interaction_propagators, t_points, initial, 5),
        }

    return {
        'duration': float(t_points[-1]),
        'logical_gate_fidelity': metrics['logical_gate_fidelity'],
        'logical_process_fidelity': metrics['logical_process_fidelity'],
        'final_leakage': metrics['final_leakage'],
        'peak_leakage': metrics['peak_leakage'],
        'peak_leakage_time': metrics['peak_leakage_time'],
        'states': state_results,
    }


def run_comparison(config_path=DEFAULT_CONFIG):
    experiment = _load_config(config_path)
    qubit = _build_qubit(experiment['model'])
    results = {
        name: evaluate_protocol(qubit, experiment, name)
        for name in experiment['protocols']
    }
    return experiment, qubit, results


def _print_results(experiment, qubit, results):
    print('IdealGridium candidate-path comparison')
    print('target={} frame={}'.format(
        experiment['target'], experiment['evaluation_frame']))
    for name, transition in experiment['transitions'].items():
        print('{} {}: frequency={:.9f} GHz, |phi_ij|={:.9f}'.format(
            name, transition, abs(qubit.freq(*transition)),
            abs(qubit.phi_ij(*transition))))

    print('\nprotocol | duration_ns | F_gate | final_leak | peak_leak | peak_ns')
    for name, result in results.items():
        print('{} | {:.1f} | {:.9f} | {:.9f} | {:.9f} | {:.1f}'.format(
            name, result['duration'], result['logical_gate_fidelity'],
            result['final_leakage'], result['peak_leakage'],
            result['peak_leakage_time']))

    print('\nprotocol | input | P0_final | P1_final | leak_final | leak_peak '
          '| P4_final | P4_peak | P5_final | P5_peak')
    for name, result in results.items():
        for initial, state in result['states'].items():
            print('{} | {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} '
                  '| {:.9f} | {:.9f} | {:.9f} | {:.9f}'.format(
                      name, initial, state['final_P0'], state['final_P1'],
                      state['final_leakage'], state['peak_leakage'],
                      state['level_4']['final'], state['level_4']['peak'],
                      state['level_5']['final'], state['level_5']['peak']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    experiment, qubit, results = run_comparison(arguments.config)
    _print_results(experiment, qubit, results)


if __name__ == '__main__':
    main()
