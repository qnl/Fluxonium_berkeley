"""Audit the inherited symmetric-study shell against an asymmetric reference.

This diagnostic tests the 2026-09-18 shell-informed construction: retain
Stage-1 ranks 0..159 and replace the ordinary ranks 160..169 by the known
high-shell ranks (165, 169, 171, 172, 173, 174, 175, 176, 178, 179).
The asymmetric N=51 study rejects this inherited basis.  The script remains
as provenance and as the source of the corrected noncontiguous state tracker;
it is intentionally not a new basis selector.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg as la
import scipy.sparse as sps

from Circuit_Objs.qchard_gridium_netlist import (
    _assemble_nonlinear_sector, _coeffs, _stage1_eigensolve,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'Figures' / 'Rabi_3Photon'
BENCHMARK = dict(
    EJ=5.0, EC=0.5, EL=1.0, ELK=1.0, EJS=4.0, ECS=8.0,
    eC=5.5, eP=10.0, eps_J=0.10, eps_LK=0.05,
    ng=0.0, phi_ext=0.0, theta_ext=np.pi,
)
REDUCED_SPATIAL = dict(n1max=4, N2=31, L2=11.0, N3=31, L3=14.0,
                       N4=4)
ASYMMETRIC_SPATIAL_51 = dict(
    n1max=4, N2=51, L2=11.0, N3=51, L3=14.0, N4=8,
)
HAND_INDICES = np.array(
    list(range(160)) + [165, 169, 171, 172, 173, 174, 175, 176, 178, 179],
    dtype=int)


def _act(sector_operator, oscillator_operator, tensor):
    return np.einsum('ij,mn,jnk->imk', sector_operator,
                     oscillator_operator, tensor, optimize=True)


def _three_step_paths(matrix, nstates, count=10):
    weights = np.abs(matrix)
    paths = []
    for middle1, middle2 in itertools.product(range(2, nstates), repeat=2):
        if len({0, 1, middle1, middle2}) < 4:
            continue
        edges = (weights[0, middle1], weights[middle1, middle2],
                 weights[middle2, 1])
        score = float(np.prod(edges))
        if score > 0:
            paths.append({
                'path': [0, middle1, middle2, 1],
                'score_product': score,
                'edge_magnitudes': [float(x) for x in edges],
            })
    return sorted(paths, key=lambda row: row['score_product'], reverse=True)[:count]


def _build_stage1(spatial_cutoffs=REDUCED_SPATIAL, stage1_k=180):
    ECm, K, EJ1, EJ2 = _coeffs(
        BENCHMARK['EJ'], BENCHMARK['EC'], BENCHMARK['EL'],
        BENCHMARK['ELK'], BENCHMARK['EJS'], BENCHMARK['ECS'],
        BENCHMARK['eps_J'], BENCHMARK['eps_LK'],
        BENCHMARK['eC'], BENCHMARK['eP'])
    H1, o = _assemble_nonlinear_sector(
        ECm[:3, :3], K[:2, :2], EJ1, EJ2, BENCHMARK['EJS'],
        BENCHMARK['ng'], BENCHMARK['phi_ext'], BENCHMARK['theta_ext'],
        spatial_cutoffs['n1max'], spatial_cutoffs['N2'], spatial_cutoffs['L2'],
        spatial_cutoffs['N3'], spatial_cutoffs['L3'])
    w1, V, diagnostics = _stage1_eigensolve(
        H1, k=stage1_k, sigma=-(EJ1 + EJ2 + BENCHMARK['EJS'] + 10.0),
        which='LM', tol=1e-8, permc_spec='MMD_AT_PLUS_A',
        collect_diagnostics=True)

    residuals = np.linalg.norm(H1 @ V - V * w1, axis=0)
    orthogonality_residual = np.linalg.norm(
        V.conj().T @ V - np.eye(stage1_k), ord=np.inf)
    diagnostics.update({
        'matrix_dimension': int(H1.shape[0]),
        'matrix_nnz': int(H1.nnz),
        'k': int(stage1_k),
        'sigma': float(-(EJ1 + EJ2 + BENCHMARK['EJS'] + 10.0)),
        'which': 'LM',
        'tol': 1e-8,
        'ncv': None,
        'v0': None,
        'max_eigenpair_residual': float(np.max(residuals)),
        'eigenpair_residuals': residuals.tolist(),
        'orthogonality_residual_inf': float(orthogonality_residual),
    })

    kron3 = o['kron3']
    def proj(operator):
        return V.conj().T @ (operator @ V)

    operators = {
        'n1': proj(kron3(sps.diags(o['nvec']), o['I2'], o['I3'])),
        'n2': proj(kron3(o['I1'], o['n2'], o['I3'])),
        'n3': proj(kron3(o['I1'], o['I2'], o['n3'])),
        'x2': proj(kron3(o['I1'], sps.diags(o['x2']), o['I3'])),
        'x3': proj(kron3(o['I1'], o['I2'], sps.diags(o['x3']))),
    }
    return ECm, K, w1, V, operators, diagnostics


def _stage2_templates(ECm, K, w1, operators, retained_indices, n4):
    retained_indices = np.asarray(retained_indices, dtype=int)
    nkeep = len(retained_indices)
    selected_w1 = w1[retained_indices]
    selected_operators = {
        name: operator[np.ix_(retained_indices, retained_indices)]
        for name, operator in operators.items()
    }
    A4, B4 = 4 * ECm[3, 3], 0.5 * K[2, 2]
    f4 = (A4 / (4 * B4)) ** 0.25
    g4 = 1.0 / (2 * f4)
    a = np.diag(np.sqrt(np.arange(1, n4)), 1)
    x4 = f4 * (a + a.T)
    p4 = 1j * g4 * (a.T - a)
    w4 = 2 * np.sqrt(A4 * B4) * (np.arange(n4) + 0.5)
    n1 = selected_operators['n1']
    n2 = selected_operators['n2']
    n3 = selected_operators['n3']
    x2 = selected_operators['x2']
    x3 = selected_operators['x3']
    h = (np.kron(np.diag(selected_w1), np.eye(n4))
         + np.kron(np.eye(nkeep), np.diag(w4))
         + 8 * ECm[0, 3] * np.kron(n1, p4)
         + 8 * ECm[1, 3] * np.kron(n2, p4)
         + 8 * ECm[2, 3] * np.kron(n3, p4)
         + K[0, 2] * np.kron(x2, x4)
         + K[1, 2] * np.kron(x3, x4))
    h = 0.5 * (h + h.conj().T)
    values, vectors = la.eigh(h, subset_by_index=(0, 10), driver='evr')
    tensor = vectors.reshape(nkeep, n4, 11)
    grid_phi_sector = 0.5 * x2 + x3
    applied = _act(grid_phi_sector, np.eye(n4), tensor)
    grid_phi = np.einsum('imA,imB->AB', tensor.conj(), applied, optimize=True)
    return values, vectors, tensor, grid_phi


def _track(reference_vectors, vectors, retained_indices, n4, nstates=11):
    """Assign candidate states after embedding their actual Stage-1 ranks."""
    retained_indices = np.asarray(retained_indices, dtype=int)
    full_stage1_size = reference_vectors.shape[0] // n4
    if reference_vectors.shape != (full_stage1_size * n4, nstates):
        raise ValueError('reference_vectors has an incompatible shape.')
    if vectors.shape != (len(retained_indices) * n4, nstates):
        raise ValueError('vectors has an incompatible retained-basis shape.')
    if (len(np.unique(retained_indices)) != len(retained_indices)
            or np.any(retained_indices < 0)
            or np.any(retained_indices >= full_stage1_size)):
        raise ValueError('retained_indices must be unique valid Stage-1 ranks.')

    reference = reference_vectors.reshape(full_stage1_size, n4, nstates)
    candidate = vectors.reshape(len(retained_indices), n4, nstates)
    embedded = np.zeros_like(reference)
    embedded[retained_indices] = candidate
    overlap = embedded.reshape(-1, nstates).conj().T @ reference.reshape(-1, nstates)
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    assignment = np.zeros(nstates, dtype=int)
    assignment[columns] = rows
    matched = np.abs(overlap[assignment, np.arange(nstates)])
    return assignment, matched


def _apply_state_assignment(values, vectors, tensor, matrix, assignment):
    """Put every state-indexed quantity into reference-state order."""
    assignment = np.asarray(assignment, dtype=int)
    if sorted(assignment.tolist()) != list(range(len(assignment))):
        raise ValueError('assignment must be a permutation of state indices.')
    tracked_values = values[assignment]
    tracked_vectors = vectors[:, assignment]
    tracked_tensor = tensor[..., assignment]
    tracked_matrix = matrix[np.ix_(assignment, assignment)]
    transitions = tracked_values - tracked_values[0]
    return (tracked_values, tracked_vectors, tracked_tensor, tracked_matrix,
            transitions)


def _case(label, indices, ECm, K, w1, operators, reference_vectors, n4):
    values, vectors, tensor, matrix = _stage2_templates(
        ECm, K, w1, operators, indices, n4)
    assignment, overlaps = _track(
        reference_vectors, vectors, indices, n4)
    (ordered_values, ordered_vectors, ordered_tensor, ordered_matrix,
     transitions) = _apply_state_assignment(
        values, vectors, tensor, matrix, assignment)
    return {
        'label': label,
        'retained_indices': indices.tolist(),
        'energies_GHz': ordered_values.tolist(),
        'transitions_GHz': transitions[1:].tolist(),
        'transition_errors_MHz_vs_k180': (
            1e3 * (transitions[1:] - REFERENCE_TRANSITIONS)).tolist(),
        'state_assignment_candidate_row_for_reference_state': assignment.tolist(),
        'state_overlaps': overlaps.tolist(),
        'minimum_state_overlap': float(np.min(overlaps)),
        'grid_phi_abs': np.abs(ordered_matrix).tolist(),
        'grid_phi_pathways': _three_step_paths(ordered_matrix, 11),
        'logical_to_excited': {
            str(initial): sorted([
                {'state': int(level), 'matrix_element': float(abs(ordered_matrix[initial, level]))}
                for level in range(2, 11)
            ], key=lambda row: row['matrix_element'], reverse=True)
        for initial in (0, 1)},
    }


REFERENCE_TRANSITIONS = np.zeros(10)


def _write_stage1_checkpoint(checkpoint_dir, spatial_cutoffs, w1, vectors,
                             operators, diagnostics):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_dir / 'stage1_eigensystem.npz',
        eigenvalues_GHz=w1,
        eigenvectors=vectors,
        eigenpair_residuals=np.asarray(diagnostics['eigenpair_residuals']),
    )
    np.savez(checkpoint_dir / 'projected_operators.npz', **operators)
    metadata = {
        'provenance': 'fresh asymmetric Gridium4Mode Stage-1 solve',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'physical_parameters': BENCHMARK,
        'spatial_cutoffs': spatial_cutoffs,
        'solver': diagnostics,
        'files': ['stage1_eigensystem.npz', 'projected_operators.npz'],
    }
    (checkpoint_dir / 'checkpoint.json').write_text(json.dumps(metadata, indent=2))


def run(output_dir=OUTPUT_DIR, spatial_cutoffs=None,
        artifact_stem='four_mode_retained_basis_grid_phi_audit_reduced',
        checkpoint_dir=None):
    global REFERENCE_TRANSITIONS
    spatial_cutoffs = dict(
        REDUCED_SPATIAL if spatial_cutoffs is None else spatial_cutoffs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ECm, K, w1, stage1_vectors, operators, diagnostics = _build_stage1(
        spatial_cutoffs)
    if checkpoint_dir is not None:
        _write_stage1_checkpoint(
            checkpoint_dir, spatial_cutoffs, w1, stage1_vectors, operators,
            diagnostics)
    full_indices = np.arange(180, dtype=int)
    reference_values, reference_vectors, _, _ = _stage2_templates(
        ECm, K, w1, operators, full_indices, spatial_cutoffs['N4'])
    REFERENCE_TRANSITIONS = reference_values[1:] - reference_values[0]
    cases = {}
    for label, indices in (
            ('energy_k160', np.arange(160, dtype=int)),
            ('energy_k170', np.arange(170, dtype=int)),
            ('energy_k180', np.arange(180, dtype=int)),
            ('inherited_symmetric_shell_k170', HAND_INDICES)):
        cases[label] = _case(label, indices, ECm, K, w1, operators,
                              reference_vectors, spatial_cutoffs['N4'])
    result = {
        'provenance': {
            'description': 'fresh solve using the physical parameters below',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'historical_checkpoint_reused': False,
        },
        'benchmark': BENCHMARK,
        'spatial_cutoffs': spatial_cutoffs,
        'inherited_symmetric_shell_indices': HAND_INDICES.tolist(),
        'stage1_diagnostics': diagnostics,
        'reference_k180': {
            'transitions_GHz': REFERENCE_TRANSITIONS.tolist(),
            'grid_phi_abs': cases['energy_k180']['grid_phi_abs'],
        },
        'cases': cases,
        'elapsed_seconds': time.perf_counter() - started,
    }
    path = output_dir / f'{artifact_stem}.json'
    path.write_text(json.dumps(result, indent=2))
    _write_summary_csv(result, output_dir / f'{artifact_stem}_summary.csv')
    _plot_pathways(result, output_dir / f'{artifact_stem}_pathways.png')
    print(json.dumps({
        'output': str(path), 'elapsed_seconds': result['elapsed_seconds'],
        'min_overlaps': {name: case['minimum_state_overlap']
                         for name, case in cases.items()},
        'top_paths': {name: case['grid_phi_pathways'][:3]
                      for name, case in cases.items()},
    }, indent=2))
    return result


def _write_summary_csv(result, path):
    rows = []
    for name, case in result['cases'].items():
        for level, error in enumerate(case['transition_errors_MHz_vs_k180'], 1):
            rows.append({'case': name, 'transition': level,
                         'transition_error_MHz': error,
                         'minimum_state_overlap': case['minimum_state_overlap']})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot_pathways(result, path):
    import matplotlib.pyplot as plt

    labels = list(result['cases'])
    top = [result['cases'][label]['grid_phi_pathways'][0]['score_product']
           for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, top); ax.set_ylabel('top three-edge product')
    ax.set_title('Retained-basis grid_phi pathway ranking')
    ax.tick_params(axis='x', rotation=25); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == '__main__':
    run()
