"""Export time-resolved fidelity for the frozen IdealGridium gates.

This module reruns only the frozen X90 and X180 candidates.  It preserves
their amplitudes, phases, detunings, duration, model cutoffs, and validated
solver settings.  The existing phase-sensitive logical-gate evaluator is
applied independently to every saved interaction-picture propagator.

The 8-samples/ns trajectory reproduces the validated reference convention.
A second 32-samples/ns trajectory checks whether the sampled fidelity maximum
is stable against a denser output-time grid; it is not an optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from Simulations.Rabi_3Photon import compare_x180_protocols as comparison
from Simulations.Rabi_3Photon import validate_simultaneous_x180 as validation
from Simulations.Rabi_3Photon import workflow_funcs as workflow


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).parent / 'yamls' / 'experiments'
DEFAULT_CONFIGS = (
    EXPERIMENT_DIR / 'idealgridium_soft_candidate_x90.yaml',
    EXPERIMENT_DIR / 'idealgridium_soft_candidate_x180.yaml',
)
DEFAULT_OUTPUT_DIR = ROOT / 'Figures' / 'Rabi_3Photon'
REFERENCE_SAMPLES_PER_NS = 8
RESOLUTION_CHECK_SAMPLES_PER_NS = 32
REFERENCE_NLEV = 16
REFERENCE_LC_CUTOFF = 460
LOGICAL_BASIS_LC_CUTOFF = 230


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty CSV artifact.')
    if fieldnames is None:
        fieldnames = list(rows[0])
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _trajectory_rows(result, target, samples_per_ns):
    rows = []
    for time, propagator in zip(result['t_points'], result['propagators']):
        metrics = workflow.evaluate_logical_gate(
            propagator, target, t_points=[time],
        )
        rows.append({
            'time_ns': float(time),
            'logical_gate_fidelity': metrics['logical_gate_fidelity'],
            'logical_process_fidelity': (
                metrics['logical_process_fidelity']),
            'average_leakage': metrics['final_leakage'],
            'input_0_leakage': (
                metrics['state_diagnostics'][0]['final_leakage']),
            'input_1_leakage': (
                metrics['state_diagnostics'][1]['final_leakage']),
            'samples_per_ns': samples_per_ns,
        })
    peak_index = int(np.argmax([
        row['logical_gate_fidelity'] for row in rows
    ]))
    for index, row in enumerate(rows):
        row['is_peak_gate_fidelity_sample'] = index == peak_index
        row['is_nominal_final_sample'] = index == len(rows) - 1
    return rows, peak_index


def _run_trajectory(experiment, parameters, logical_basis_reference,
                    samples_per_ns):
    qubit = validation._qubit(
        experiment, REFERENCE_NLEV, REFERENCE_LC_CUTOFF,
    )
    result = validation.evaluate_case(
        qubit,
        experiment,
        parameters,
        samples_per_ns,
        solver_options=validation.TIGHT_OPTIONS,
        logical_basis_reference=logical_basis_reference,
    )
    rows, peak_index = _trajectory_rows(
        result, experiment['target'], samples_per_ns,
    )
    return result, rows, peak_index


def _summary_row(config_path, experiment, parameters, reference, dense):
    reference_result, reference_rows, reference_peak_index = reference
    dense_result, dense_rows, dense_peak_index = dense
    reference_peak = reference_rows[reference_peak_index]
    dense_peak = dense_rows[dense_peak_index]
    reference_final = reference_rows[-1]
    dense_final = dense_rows[-1]
    duration = float(parameters[5])
    return {
        'target_gate': experiment['target'],
        'source_config': _relative_path(config_path),
        'nominal_gate_time_ns': duration,
        'final_logical_gate_fidelity': (
            reference_final['logical_gate_fidelity']),
        'final_logical_process_fidelity': (
            reference_final['logical_process_fidelity']),
        'final_average_leakage': reference_final['average_leakage'],
        'peak_time_ns': dense_peak['time_ns'],
        'peak_logical_gate_fidelity': (
            dense_peak['logical_gate_fidelity']),
        'peak_logical_process_fidelity': (
            dense_peak['logical_process_fidelity']),
        'average_leakage_at_peak': dense_peak['average_leakage'],
        'input_0_leakage_at_peak': dense_peak['input_0_leakage'],
        'input_1_leakage_at_peak': dense_peak['input_1_leakage'],
        'peak_minus_final_gate_fidelity': (
            dense_peak['logical_gate_fidelity']
            - reference_final['logical_gate_fidelity']),
        'reference_grid_peak_time_ns': reference_peak['time_ns'],
        'reference_grid_peak_gate_fidelity': (
            reference_peak['logical_gate_fidelity']),
        'dense_grid_peak_time_ns': dense_peak['time_ns'],
        'dense_grid_peak_gate_fidelity': (
            dense_peak['logical_gate_fidelity']),
        'peak_time_shift_dense_minus_reference_ns': (
            dense_peak['time_ns'] - reference_peak['time_ns']),
        'peak_fidelity_shift_dense_minus_reference': (
            dense_peak['logical_gate_fidelity']
            - reference_peak['logical_gate_fidelity']),
        'dense_grid_final_gate_fidelity': (
            dense_final['logical_gate_fidelity']),
        'dense_minus_reference_final_gate_fidelity': (
            dense_final['logical_gate_fidelity']
            - reference_final['logical_gate_fidelity']),
        'peak_is_strictly_interior': (
            0 < dense_peak_index < len(dense_rows) - 1),
        'time_from_peak_to_nominal_end_ns': duration - dense_peak['time_ns'],
        'reference_samples_per_ns': REFERENCE_SAMPLES_PER_NS,
        'resolution_check_samples_per_ns': (
            RESOLUTION_CHECK_SAMPLES_PER_NS),
        'reference_sample_count': len(reference_result['t_points']),
        'resolution_check_sample_count': len(dense_result['t_points']),
        'nlev': REFERENCE_NLEV,
        'nlev_lc': REFERENCE_LC_CUTOFF,
        'logical_basis_reference_nlev_lc': LOGICAL_BASIS_LC_CUTOFF,
        'solver_settings': json.dumps(
            validation.TIGHT_OPTIONS, sort_keys=True,
        ),
        'evaluation_frame': experiment['evaluation_frame'],
        'pulse_shape': experiment['pulse']['shape'],
        'amplitude_05': parameters[0],
        'amplitude_54': parameters[1],
        'amplitude_41': parameters[2],
        'phase_54': parameters[3],
        'phase_41': parameters[4],
        'detuning_05_ghz': parameters[6],
        'detuning_54_ghz': parameters[7],
        'detuning_41_ghz': parameters[8],
    }


def _plot_trajectory(path, experiment, rows, summary):
    times = np.asarray([row['time_ns'] for row in rows])
    gate_fidelity = np.asarray([
        row['logical_gate_fidelity'] for row in rows
    ])
    process_fidelity = np.asarray([
        row['logical_process_fidelity'] for row in rows
    ])
    leakage = np.asarray([row['average_leakage'] for row in rows])

    figure, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': (2.2, 1)},
    )
    axes[0].plot(times, gate_fidelity, label=r'$F_{gate}(t)$')
    axes[0].plot(times, process_fidelity, label=r'$F_{process}(t)$')
    axes[0].scatter(
        [summary['peak_time_ns']],
        [summary['peak_logical_gate_fidelity']],
        color='tab:red', zorder=3, label='Peak sampled gate fidelity',
    )
    axes[0].axvline(
        summary['nominal_gate_time_ns'], color='black', linestyle='--',
        alpha=0.7, label='Nominal final time',
    )
    axes[0].set_ylabel('Logical fidelity')
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='best')

    axes[1].plot(times, leakage, color='tab:orange')
    axes[1].scatter(
        [summary['peak_time_ns']],
        [summary['average_leakage_at_peak']],
        color='tab:red', zorder=3,
    )
    axes[1].axvline(
        summary['nominal_gate_time_ns'], color='black', linestyle='--',
        alpha=0.7,
    )
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Average leakage')
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        '{} frozen IdealGridium pulse: fidelity versus time'.format(
            experiment['target']),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def generate_gate_artifacts(config_path, output_dir=DEFAULT_OUTPUT_DIR):
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = comparison._load_config(config_path)
    parameters = validation._candidate_vector(experiment)
    logical_basis_reference = validation._qubit(
        experiment, REFERENCE_NLEV, LOGICAL_BASIS_LC_CUTOFF,
    )

    reference = _run_trajectory(
        experiment, parameters, logical_basis_reference,
        REFERENCE_SAMPLES_PER_NS,
    )
    dense = _run_trajectory(
        experiment, parameters, logical_basis_reference,
        RESOLUTION_CHECK_SAMPLES_PER_NS,
    )
    summary = _summary_row(
        config_path, experiment, parameters, reference, dense,
    )

    slug = experiment['target'].lower()
    csv_path = output_dir / '{}_fidelity_vs_time.csv'.format(slug)
    figure_path = output_dir / '{}_fidelity_vs_time.png'.format(slug)
    _write_csv(csv_path, dense[1])
    _plot_trajectory(figure_path, experiment, dense[1], summary)
    return {
        'summary': summary,
        'csv_path': csv_path,
        'figure_path': figure_path,
    }


def generate_artifacts(config_paths=DEFAULT_CONFIGS,
                       output_dir=DEFAULT_OUTPUT_DIR):
    artifacts = []
    for config_path in config_paths:
        print('Evaluating frozen trajectory: {}'.format(config_path), flush=True)
        artifact = generate_gate_artifacts(config_path, output_dir)
        artifacts.append(artifact)
        summary = artifact['summary']
        print(
            '{}: final F_gate={:.12f}; peak F_gate={:.12f} at {:.9f} ns'
            .format(
                summary['target_gate'],
                summary['final_logical_gate_fidelity'],
                summary['peak_logical_gate_fidelity'],
                summary['peak_time_ns'],
            ),
            flush=True,
        )

    summary_path = Path(output_dir) / 'gate_fidelity_peak_summary.csv'
    _write_csv(summary_path, [artifact['summary'] for artifact in artifacts])
    return artifacts, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', dest='configs')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    configs = DEFAULT_CONFIGS if arguments.configs is None else arguments.configs
    artifacts, summary_path = generate_artifacts(configs, arguments.output_dir)
    print('Summary: {}'.format(summary_path))
    for artifact in artifacts:
        print(artifact['csv_path'])
        print(artifact['figure_path'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
