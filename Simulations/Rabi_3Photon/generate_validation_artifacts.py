"""Generate durable figures and tables for frozen IdealGridium gate runs.

This script does not calibrate or optimize pulses. It reruns the existing
bounded validation for each supplied frozen experiment YAML and exports the
resulting reference trajectory and convergence data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from Simulations.Rabi_3Photon import calibrate_simultaneous_x180 as calibration
from Simulations.Rabi_3Photon import validate_simultaneous_x180 as validation
from Simulations.Rabi_3Photon import workflow_funcs as workflow


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).parent / 'yamls' / 'experiments'
DEFAULT_CONFIGS = (
    EXPERIMENT_DIR / 'idealgridium_soft_candidate_x90.yaml',
    EXPERIMENT_DIR / 'idealgridium_soft_candidate_x180.yaml',
)
DEFAULT_OUTPUT_DIR = ROOT / 'Figures' / 'Rabi_3Photon'


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty CSV artifact.')
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _convergence_rows(result):
    rows = []
    groups = (
        ('solver_sampling', result['solver_cases']),
        ('retained_levels', result['retained_level_cases']),
        ('lc_cutoff', result['lc_cutoff_cases']),
    )
    for category, cases in groups:
        for case, case_result in cases.items():
            metrics = case_result['metrics']
            rows.append({
                'category': category,
                'case': str(case),
                'logical_gate_fidelity': metrics['logical_gate_fidelity'],
                'logical_process_fidelity': metrics['logical_process_fidelity'],
                'final_leakage': metrics['final_leakage'],
                'peak_leakage': metrics['peak_leakage'],
                'unitarity_residual': case_result['unitarity_residual'],
                'nlev': case_result['qubit'].nlev,
                'nlev_lc': case_result['qubit'].nlev_lc,
                'sample_count': len(case_result['t_points']),
            })
    return rows


def _summary_row(result, config_path, pulse_configs, traces):
    experiment = result['experiment']
    reference = result['reference']
    metrics = reference['metrics']
    parameters = result['parameters']
    row = {
        'target_gate': experiment['target'],
        'model': 'IdealGridium',
        'control_operator': 'IdealGridium.phi() abstract global phase',
        'legacy_drive_type': 'flux',
        'gate_duration_ns': parameters[5],
        'logical_gate_fidelity': metrics['logical_gate_fidelity'],
        'logical_process_fidelity': metrics['logical_process_fidelity'],
        'final_leakage': metrics['final_leakage'],
        'peak_leakage': metrics['peak_leakage'],
        'peak_leakage_time_ns': metrics['peak_leakage_time'],
        'nlev': reference['qubit'].nlev,
        'nlev_lc': reference['qubit'].nlev_lc,
        'sample_count': len(reference['t_points']),
        'samples_per_ns': 8,
        'solver_settings': json.dumps(
            validation.TIGHT_OPTIONS, sort_keys=True),
        'evaluation_frame': experiment['evaluation_frame'],
        'pulse_shape': experiment['pulse']['shape'],
        'pulse_sigma': experiment['pulse']['sigma'],
        'source_config': _relative_path(config_path),
    }
    for index, (name, config, trace) in enumerate(zip(
            calibration.TRANSITION_NAMES, pulse_configs, traces), start=1):
        transition = '{}-{}'.format(*config.targeted_drive)
        row.update({
            'tone_{}_name'.format(index): name,
            'tone_{}_transition'.format(index): transition,
            'tone_{}_frequency_ghz'.format(index): (
                trace['carrier_frequency_ghz']),
            'tone_{}_amplitude'.format(index): config.drive_amplitude_factor,
            'tone_{}_phase_rad'.format(index): config.carrier_phase,
            'tone_{}_detuning_ghz'.format(index): config.drive_detuning,
        })
    return row


def _timeseries_rows(result, populations, traces):
    reference = result['reference']
    metrics = reference['metrics']
    rows = []
    for time_index, time in enumerate(reference['t_points']):
        row = {'time_ns': time}
        for initial in (0, 1):
            for level, value in enumerate(populations[initial][time_index]):
                row['input_{}_P{}'.format(initial, level)] = value
            row['input_{}_leakage'.format(initial)] = (
                metrics['state_diagnostics'][initial]['leakage'][time_index])
        row['average_leakage'] = metrics['leakage_by_time'][time_index]
        for tone_index, trace in enumerate(traces, start=1):
            row['tone_{}_coefficient'.format(tone_index)] = (
                trace['values'][time_index])
        rows.append(row)
    return rows


def _plot_state_populations(path, result, populations, shown_levels=8):
    metrics = result['reference']['metrics']
    times = result['reference']['t_points']
    level_count = min(shown_levels, populations[0].shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, initial in zip(axes, (0, 1)):
        for level in range(level_count):
            axis.plot(times, populations[initial][:, level],
                      label=r'$|{}\rangle$'.format(level))
        axis.set_title(
            r'Input $|{}\rangle$: final leak={:.3e}, peak={:.3e}'.format(
                initial,
                metrics['state_diagnostics'][initial]['final_leakage'],
                metrics['state_diagnostics'][initial]['peak_leakage']))
        axis.set_xlabel('Time (ns)')
        axis.set_ylabel('Energy-eigenstate population')
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[-1].legend(ncol=2, fontsize=8, loc='best')
    fig.suptitle('{} state propagation — 0-5-4-1 three-tone protocol'.format(
        result['experiment']['target']))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_intermediate_populations(path, result, populations,
                                   extra_level_count=4):
    """Plot the most active nonlogical levels on their natural scale."""
    times = result['reference']['t_points']
    level_count = populations[0].shape[1]
    required_levels = {level for level in (4, 5) if level < level_count}
    candidate_levels = [
        level for level in range(2, level_count)
        if level not in required_levels
    ]
    peak_by_level = {
        level: max(
            np.max(populations[initial][:, level]) for initial in (0, 1))
        for level in candidate_levels
    }
    ranked_extras = sorted(
        candidate_levels, key=peak_by_level.get, reverse=True)
    shown_levels = sorted(
        required_levels.union(ranked_extras[:extra_level_count]))

    maximum_population = max(
        np.max(populations[initial][:, shown_levels]) for initial in (0, 1))
    upper_limit = 1.08 * maximum_population if maximum_population > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, initial in zip(axes, (0, 1)):
        for level in shown_levels:
            axis.plot(
                times, populations[initial][:, level],
                label=r'$|{}\rangle$'.format(level))
        axis.set_title(r'Input $|{}\rangle$'.format(initial))
        axis.set_xlabel('Time (ns)')
        axis.set_ylabel('Nonlogical energy-eigenstate population')
        axis.set_ylim(-0.02 * upper_limit, upper_limit)
        axis.grid(alpha=0.25)
    axes[-1].legend(ncol=2, fontsize=8, loc='best')
    fig.suptitle(
        '{} intermediate populations — expanded scale'.format(
            result['experiment']['target']))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_leakage(path, result):
    reference = result['reference']
    metrics = reference['metrics']
    fig, axis = plt.subplots(figsize=(8, 5))
    for initial in (0, 1):
        axis.plot(
            reference['t_points'],
            metrics['state_diagnostics'][initial]['leakage'],
            label=r'Input $|{}\rangle$'.format(initial))
    axis.plot(reference['t_points'], metrics['leakage_by_time'],
              color='black', linewidth=2, label='Logical-input average')
    axis.set_xlabel('Time (ns)')
    axis.set_ylabel('Leakage')
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    axis.set_title(
        '{} leakage: final={:.6g}, peak={:.6g}'.format(
            result['experiment']['target'], metrics['final_leakage'],
            metrics['peak_leakage']))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_drive_traces(path, result, traces):
    times = result['reference']['t_points']
    fig, axis = plt.subplots(figsize=(10, 5))
    for trace in traces:
        transition = '{}↔{}'.format(*trace['transition'])
        label = '{} ({:.6f} GHz)'.format(
            transition, trace['carrier_frequency_ghz'])
        axis.plot(times, trace['values'], label=label)
    axis.set_xlabel('Time (ns)')
    axis.set_ylabel('Actual scalar drive coefficient')
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    axis.set_title(
        '{} three-tone coefficients for the abstract global phase operator'
        .format(result['experiment']['target']))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_convergence(path, result):
    groups = (
        ('Solver / sampling', result['solver_cases']),
        ('Retained levels', result['retained_level_cases']),
        ('LC cutoff', result['lc_cutoff_cases']),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for axis, (title, cases) in zip(axes, groups):
        labels = [str(label) for label in cases]
        infidelity = [
            1.0 - case['metrics']['logical_gate_fidelity']
            for case in cases.values()
        ]
        final_leakage = [
            case['metrics']['final_leakage'] for case in cases.values()
        ]
        peak_leakage = [
            case['metrics']['peak_leakage'] for case in cases.values()
        ]
        positions = np.arange(len(labels))
        axis.plot(
            positions, infidelity, 'o-', color='tab:blue',
            label=r'Gate infidelity ($1-F_{gate}$)')
        axis.set_ylabel(r'Gate infidelity ($1-F_{gate}$)', color='tab:blue')
        axis.tick_params(axis='y', labelcolor='tab:blue')
        axis.ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2), useOffset=False)
        axis.set_xticks(positions, labels, rotation=20, ha='right')
        axis.set_title(title)
        axis.grid(alpha=0.25)
        leakage_axis = axis.twinx()
        leakage_axis.plot(positions, final_leakage, 's--', color='tab:orange',
                           label='Final leakage')
        leakage_axis.plot(positions, peak_leakage, '^:', color='tab:red',
                           label='Peak leakage')
        leakage_axis.set_ylabel('Leakage')
        lines = axis.lines + leakage_axis.lines
        axis.legend(lines, [line.get_label() for line in lines], loc='best')
    fig.suptitle('{} numerical convergence'.format(
        result['experiment']['target']))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_validation_artifacts(result, config_path, output_dir=DEFAULT_OUTPUT_DIR):
    """Write figures and machine-readable tables for one validation result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = result['experiment']['target'].lower()
    reference = result['reference']
    populations = workflow.state_population_trajectories(
        reference['propagators'])
    pulse_configs = calibration._pulse_configs(
        result['experiment'], result['parameters'])
    traces = workflow.drive_coefficient_traces(
        reference['qubit'], pulse_configs, reference['t_points'])

    paths = {
        'state_populations': output_dir / '{}_state_populations.png'.format(slug),
        'intermediate_populations': (
            output_dir / '{}_intermediate_populations.png'.format(slug)),
        'leakage': output_dir / '{}_leakage.png'.format(slug),
        'drive_traces': output_dir / '{}_drive_traces.png'.format(slug),
        'convergence_figure': output_dir / '{}_convergence.png'.format(slug),
        'summary': output_dir / '{}_summary.csv'.format(slug),
        'timeseries': output_dir / '{}_timeseries.csv'.format(slug),
        'convergence_table': output_dir / '{}_convergence.csv'.format(slug),
    }
    _plot_state_populations(paths['state_populations'], result, populations)
    _plot_intermediate_populations(
        paths['intermediate_populations'], result, populations)
    _plot_leakage(paths['leakage'], result)
    _plot_drive_traces(paths['drive_traces'], result, traces)
    _plot_convergence(paths['convergence_figure'], result)

    summary = _summary_row(result, config_path, pulse_configs, traces)
    _write_csv(paths['summary'], [summary])
    _write_csv(paths['timeseries'], _timeseries_rows(result, populations, traces))
    _write_csv(paths['convergence_table'], _convergence_rows(result))
    return {
        'summary': summary,
        'paths': paths,
        'metrics': reference['metrics'],
    }


def generate_artifacts(config_paths=DEFAULT_CONFIGS,
                       output_dir=DEFAULT_OUTPUT_DIR):
    """Run the frozen validations and export individual and combined data."""
    generated = []
    for config_path in config_paths:
        print('Validating frozen candidate: {}'.format(config_path), flush=True)
        result = validation.run_validation(config_path)
        artifact = write_validation_artifacts(result, config_path, output_dir)
        generated.append(artifact)
        metrics = artifact['metrics']
        print(
            '{}: F_gate={:.9f} F_process={:.9f} final_leak={:.9f} '
            'peak_leak={:.9f}'.format(
                result['experiment']['target'],
                metrics['logical_gate_fidelity'],
                metrics['logical_process_fidelity'],
                metrics['final_leakage'], metrics['peak_leakage']),
            flush=True)

    combined_path = Path(output_dir) / 'validated_gate_runs.csv'
    fieldnames = []
    for artifact in generated:
        for field in artifact['summary']:
            if field not in fieldnames:
                fieldnames.append(field)
    _write_csv(
        combined_path, [artifact['summary'] for artifact in generated],
        fieldnames=fieldnames)
    return generated, combined_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', dest='configs')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    configs = DEFAULT_CONFIGS if arguments.configs is None else arguments.configs
    generated, combined_path = generate_artifacts(configs, arguments.output_dir)
    print('Combined summary: {}'.format(combined_path))
    for artifact in generated:
        for path in artifact['paths'].values():
            print(path)


if __name__ == '__main__':
    main()
