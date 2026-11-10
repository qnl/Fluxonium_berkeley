"""Audit and smoke-test the four-mode abstract ``grid_phi`` control path.

This is a bounded research diagnostic, not a gate optimizer.  It uses the
canonical netlist-derived four-mode model and keeps the existing per-target
matrix-element normalization used by the IdealGridium workflow.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode
from Simulations.Rabi_3Photon import workflow_funcs as workflow


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'Figures' / 'Rabi_3Photon'
BENCHMARK = dict(
    EJ=5.0, EC=0.5, EL=1.0, ELK=1.0, EJS=4.0, ECS=8.0,
    eC=5.5, eP=10.0, eps_J=0.10, eps_LK=0.05,
    ng=0.0, phi_ext=0.0, theta_ext=np.pi,
)
REFERENCE_CUTOFFS = dict(n1max=4, N2=31, L2=11.0, N3=31, L3=14.0,
                         N4=4, nkeep=180)
DIAGNOSTIC_CUTOFFS = dict(REFERENCE_CUTOFFS, nkeep=120)


def _model(cutoffs, nlev=12):
    return Gridium4Mode(**BENCHMARK, **cutoffs, nlev=nlev)


def _best_three_step_path(matrix, nstates):
    weights = np.abs(matrix)
    candidates = []
    for middle1, middle2 in itertools.product(range(2, nstates), repeat=2):
        if len({0, 1, middle1, middle2}) < 4:
            continue
        edges = (weights[0, middle1], weights[middle1, middle2],
                 weights[middle2, 1])
        score = float(np.prod(edges))
        if score > 0:
            candidates.append((score, (0, middle1, middle2, 1), edges))
    if not candidates:
        raise RuntimeError('No three-step grid_phi pathway connects |0> to |1>.')
    return max(candidates)


def audit(cutoffs=None, nlev=12, output_dir=DEFAULT_OUTPUT):
    """Compute spectrum, matrix elements, pathways, and cutoff differences."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = _model(DIAGNOSTIC_CUTOFFS if cutoffs is None else cutoffs, nlev)
    energies = np.asarray(diagnostic.levels(), dtype=float)
    matrix = np.abs(diagnostic.grid_phi().full())
    path_score, path, path_edges = _best_three_step_path(matrix, len(energies))

    rows = []
    for i, energy in enumerate(energies):
        rows.append({'level': i, 'energy_GHz': energy,
                     'transition_from_ground_GHz': energy - energies[0]})
    with (output_dir / 'four_mode_grid_phi_spectrum_reduced_diagnostic.csv').open(
            'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / 'four_mode_grid_phi_matrix_reduced_diagnostic.csv').open(
            'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['level'] + list(range(len(energies))))
        for i, row in enumerate(matrix):
            writer.writerow([i] + list(row))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, origin='lower', cmap='magma')
    ax.set(xlabel='final eigenstate |j>', ylabel='initial eigenstate |i>',
           title='Four-mode |<i|grid_phi|j>| (radians)')
    fig.colorbar(image, ax=ax, label='absolute matrix element')
    for i in range(len(energies)):
        for j in range(len(energies)):
            if matrix[i, j] > 0.03:
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max() * .45 else 'black',
                        fontsize=7)
    fig.tight_layout()
    fig.savefig(
        output_dir / 'four_mode_grid_phi_matrix_heatmap_reduced_diagnostic.png',
        dpi=180)
    plt.close(fig)

    return {
        'qubit': diagnostic, 'energies_GHz': energies, 'matrix': matrix,
        'path': path, 'path_score': path_score, 'path_edges': path_edges,
    }


def smoke(audit_result, output_dir=DEFAULT_OUTPUT):
    """Run a modest simultaneous three-tone propagation for the audited path."""
    output_dir = Path(output_dir)
    q = audit_result['qubit']
    path = audit_result['path']
    configs = [workflow.PulseConfig(
        T_gate=8.0, pulse_shape='cos', targeted_drive=(path[i], path[i + 1]),
        drive_amplitude_factor=0.03, carrier_phase=0.0,
        pulse_sigma=0.25, drive_detuning=0.0, drive_type='flux')
        for i in range(3)]
    t_points, propagators = workflow.solve_pulse_schedule(
        q, configs, mute=True,
        solver_options={'nsteps': 100000, 'atol': 1e-10, 'rtol': 1e-9})
    metrics = workflow.evaluate_logical_gate(propagators, 'X180', t_points)
    populations = workflow.state_population_trajectories(propagators, (0, 1))
    traces = workflow.drive_coefficient_traces(q, configs, t_points)
    unitarity = max(float(np.linalg.norm(
        (u.dag() * u - qt.qeye(q.nlev)).full(), ord=np.inf)) for u in propagators)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for initial in (0, 1):
        axes[0].plot(t_points, populations[initial][:, :min(q.nlev, 10)], alpha=.55)
        axes[1].plot(t_points, metrics['state_diagnostics'][initial]['leakage'],
                     label=f'input |{initial}>')
    axes[0].set(ylabel='population', title='Four-mode smoke populations')
    axes[1].set(ylabel='leakage', title='logical-subspace leakage')
    axes[1].legend(); axes[1].set_xlabel('time (ns)')
    fig.tight_layout(); fig.savefig(
        output_dir
        / 'four_mode_three_tone_smoke_population_leakage_reduced_diagnostic.png',
        dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for trace in traces:
        ax.plot(t_points, trace['values'], label=str(trace['transition']))
    ax.set(xlabel='time (ns)', ylabel='drive coefficient', title='Three-tone abstract drive traces')
    ax.legend(); fig.tight_layout()
    fig.savefig(
        output_dir / 'four_mode_three_tone_smoke_drive_traces_reduced_diagnostic.png',
        dpi=180)
    plt.close(fig)
    return {'metrics': metrics, 'unitarity_residual': unitarity,
            'path': path, 't_points': t_points, 'traces': traces}


if __name__ == '__main__':
    result = audit()
    smoke_result = smoke(result)
    print('levels_GHz=', np.array2string(result['energies_GHz'], precision=9))
    print('path=', result['path'], 'edges=', result['path_edges'])
    print('unitarity_residual=', smoke_result['unitarity_residual'])
    print('peak_leakage=', smoke_result['metrics']['peak_leakage'])
