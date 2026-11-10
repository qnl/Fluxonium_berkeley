"""Targeted shell audit for the verified asymmetric N=51 k=180 reference.

This is deliberately an oracle-informed physics diagnostic, not a generic
basis selector.  It ranks Stage-1 shell states 160..179 using their actual
mode-4 Hamiltonian couplings, low-energy full-state weight, omitted-space
residuals, near-resonance denominators, and leave-one-out effects on the
specific spectrum and ``grid_phi`` quantities needed for control work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import (
    ASYMMETRIC_SPATIAL_51,
    BENCHMARK,
    HAND_INDICES,
    _act,
    _apply_state_assignment,
    _three_step_paths,
    _track,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / 'research' / 'checkpoints' / '2026-09-26-asymmetric-n51'
OUTPUT_DIR = ROOT / 'Figures' / 'Rabi_3Photon'
ARTIFACT_STEM = 'four_mode_asymmetric_shell_basis_audit'
SHELL_INDICES = np.arange(160, 180, dtype=int)
LOW_INDICES = np.arange(160, dtype=int)
NSTATES = 11

# Explicit candidates selected after inspecting the asymmetric k=180 shell
# diagnostics.  They are nested in decreasing leave-one-out/QOI priority;
# this is a one-point physics audit, not a reusable blind selector.
_SHELL_PRIORITY = (170, 172, 171, 168, 162, 176, 177, 167, 163, 165,
                   166, 160, 174, 169, 173, 178, 161, 175, 164, 179)
CANDIDATE_BASES = {
    f'asymmetric_priority_k{dimension}': np.array(
        list(range(160)) + sorted(_SHELL_PRIORITY[:dimension - 160]),
        dtype=int)
    for dimension in (170, 172, 175, 177, 179)
}

# Verified by the completed rank-only pass from this same asymmetric N=51
# checkpoint.  Keeping these direct leave-one-out results avoids repeating 20
# costly dense Stage-2 solves every time the explicit candidate set is audited.
# Missing overlap entries were not retained from that pass; candidate-basis
# overlap tracking is always recomputed below.
LEAVE_ONE_OUT_DIAGNOSTICS = {
    170: (105.368577, 0.144713469, 0.128367778, 0.994271919),
    172: (61.717477, 0.0945053, 0.0860607, None),
    171: (42.124097, 0.0532869, 0.0552501, None),
    168: (9.199052, 0.00479998, 0.0202782, None),
    162: (8.688887, 0.000901666, 0.00982667, None),
    176: (2.714021, 0.00347949, 0.00675940, None),
    177: (2.517829, 0.000265308, 0.00248395, None),
    167: (2.347572, 0.00354938, 0.00541507, None),
    163: (1.390596, 0.000165381, 0.00154177, None),
    165: (1.317203, 0.00110202, 0.00269267, None),
    166: (0.996497, 0.00215056, 0.00261028, None),
    160: (0.789022, 0.00131504, 0.00192431, None),
    174: (0.764966, 0.0000835, 0.00009195, None),
    169: (0.737323, 0.00077876, 0.00198839, None),
    173: (0.608240, 0.00052033, 0.00156290, None),
    178: (0.580282, 0.00040877, 0.00155268, None),
    161: (0.425527, 0.00077319, 0.00142287, None),
    175: (0.267967, 0.00053915, 0.00075447, None),
    164: (0.130802, 0.0000352, 0.00008945, None),
    179: (0.050011, 0.00000582, 0.00004030, None),
}

PRIMARY_COUPLINGS = ((0, 7), (7, 8), (8, 1))
LOGICAL_COUPLINGS = (
    (0, 7), (0, 10), (0, 3), (0, 5),
    (1, 8), (1, 9), (1, 6), (1, 2), (1, 4),
)


def _load_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / 'checkpoint.json').read_text())
    if metadata['physical_parameters'] != BENCHMARK:
        raise ValueError('Checkpoint physical parameters do not match BENCHMARK.')
    if metadata['spatial_cutoffs'] != ASYMMETRIC_SPATIAL_51:
        raise ValueError('Checkpoint spatial cutoffs are not asymmetric N=51.')
    stage = np.load(checkpoint_dir / 'stage1_eigensystem.npz')
    projected = np.load(checkpoint_dir / 'projected_operators.npz')
    w1 = np.asarray(stage['eigenvalues_GHz'])
    operators = {name: np.asarray(projected[name]) for name in projected.files}
    if len(w1) != 180 or any(op.shape != (180, 180)
                             for op in operators.values()):
        raise ValueError('Checkpoint is not the verified k=180 reference.')
    return metadata, w1, operators


def _components(w1, operators):
    ECm, K, _, _ = _coeffs(
        BENCHMARK['EJ'], BENCHMARK['EC'], BENCHMARK['EL'],
        BENCHMARK['ELK'], BENCHMARK['EJS'], BENCHMARK['ECS'],
        BENCHMARK['eps_J'], BENCHMARK['eps_LK'],
        BENCHMARK['eC'], BENCHMARK['eP'])
    n4_count = ASYMMETRIC_SPATIAL_51['N4']
    a = np.diag(np.sqrt(np.arange(1, n4_count)), 1)
    A4, B4 = 4 * ECm[3, 3], 0.5 * K[2, 2]
    f4 = (A4 / (4 * B4)) ** 0.25
    g4 = 1.0 / (2 * f4)
    x4 = f4 * (a + a.T)
    p4 = 1j * g4 * (a.T - a)
    w4 = 2 * np.sqrt(A4 * B4) * (np.arange(n4_count) + 0.5)
    charge = 8 * (
        ECm[0, 3] * operators['n1']
        + ECm[1, 3] * operators['n2']
        + ECm[2, 3] * operators['n3'])
    phase = K[0, 2] * operators['x2'] + K[1, 2] * operators['x3']
    grid_phi = 0.5 * operators['x2'] + operators['x3']
    return {
        'w1': w1, 'w4': w4, 'p4': p4, 'x4': x4,
        'charge': charge, 'phase': phase, 'grid_phi': grid_phi,
        'n4': n4_count,
    }


def _solve(indices, components):
    indices = np.asarray(indices, dtype=int)
    n4 = components['n4']
    hamiltonian = (
        np.kron(np.diag(components['w1'][indices]), np.eye(n4))
        + np.kron(np.eye(len(indices)), np.diag(components['w4']))
        + np.kron(components['charge'][np.ix_(indices, indices)],
                  components['p4'])
        + np.kron(components['phase'][np.ix_(indices, indices)],
                  components['x4']))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    values, vectors = la.eigh(
        hamiltonian, subset_by_index=(0, NSTATES - 1), driver='evr',
        check_finite=False, overwrite_a=True)
    tensor = vectors.reshape(len(indices), n4, NSTATES)
    grid_phi_sector = components['grid_phi'][np.ix_(indices, indices)]
    applied = _act(grid_phi_sector, np.eye(n4), tensor)
    matrix = np.einsum(
        'imA,imB->AB', tensor.conj(), applied, optimize=True)
    return values, vectors, tensor, matrix


def _coupling_errors(reference_matrix, candidate_matrix, pairs):
    rows = []
    for initial, final in pairs:
        reference = float(abs(reference_matrix[initial, final]))
        candidate = float(abs(candidate_matrix[initial, final]))
        absolute = abs(candidate - reference)
        rows.append({
            'transition': [initial, final],
            'reference': reference,
            'candidate': candidate,
            'absolute_error': absolute,
            'relative_error': absolute / reference if reference > 1e-12 else None,
        })
    return rows


def _omitted_residual(indices, tensor, components):
    indices = np.asarray(indices, dtype=int)
    omitted = np.array(sorted(set(range(180)) - set(indices.tolist())),
                       dtype=int)
    if not len(omitted):
        return omitted, np.zeros(NSTATES), {}
    applied = (
        _act(components['charge'][np.ix_(omitted, indices)],
             components['p4'], tensor)
        + _act(components['phase'][np.ix_(omitted, indices)],
               components['x4'], tensor))
    norms = np.linalg.norm(applied.reshape(-1, NSTATES), axis=0)
    by_stage1 = {
        str(int(rank)): np.linalg.norm(applied[position], axis=0).tolist()
        for position, rank in enumerate(omitted)
    }
    return omitted, norms, by_stage1


def _evaluate(indices, components, reference):
    indices = np.asarray(indices, dtype=int)
    values, vectors, tensor, matrix = _solve(indices, components)
    assignment, overlaps = _track(
        reference['vectors'], vectors, indices, components['n4'], NSTATES)
    (values, vectors, tensor, matrix,
     transitions) = _apply_state_assignment(
        values, vectors, tensor, matrix, assignment)
    omitted, residuals, residual_by_stage1 = _omitted_residual(
        indices, tensor, components)
    transition_errors = 1e3 * (
        transitions[1:] - reference['transitions'][1:])
    primary_errors = _coupling_errors(
        reference['matrix'], matrix, PRIMARY_COUPLINGS)
    logical_errors = _coupling_errors(
        reference['matrix'], matrix, LOGICAL_COUPLINGS)
    return {
        'retained_indices': indices.tolist(),
        'dimension': int(len(indices)),
        'energies_GHz': values.tolist(),
        'transitions_GHz': transitions[1:].tolist(),
        'transition_errors_MHz': transition_errors.tolist(),
        'maximum_transition_error_MHz': float(
            np.max(np.abs(transition_errors))),
        'state_assignment_candidate_for_reference': assignment.tolist(),
        'state_overlaps': overlaps.tolist(),
        'minimum_state_overlap': float(np.min(overlaps)),
        'primary_path_coupling_errors': primary_errors,
        'logical_coupling_errors': logical_errors,
        'pathways': _three_step_paths(matrix, NSTATES),
        'omitted_indices': omitted.tolist(),
        'omitted_residual_norms_GHz': residuals.tolist(),
        'maximum_omitted_residual_GHz': float(np.max(residuals)),
        'omitted_residual_by_stage1_GHz': residual_by_stage1,
        '_vectors': vectors,
        '_tensor': tensor,
        '_matrix': matrix,
    }


def _reference(components):
    indices = np.arange(180, dtype=int)
    values, vectors, tensor, matrix = _solve(indices, components)
    return {
        'indices': indices,
        'values': values,
        'vectors': vectors,
        'tensor': tensor,
        'matrix': matrix,
        'transitions': values - values[0],
        'pathways': _three_step_paths(matrix, NSTATES),
    }


def _strip_private(case):
    return {key: value for key, value in case.items()
            if not key.startswith('_')}


def diagnose_shell(checkpoint_dir=CHECKPOINT_DIR,
                   recompute_leave_one_out=False):
    metadata, w1, operators = _load_checkpoint(checkpoint_dir)
    components = _components(w1, operators)
    reference = _reference(components)
    base = _evaluate(LOW_INDICES, components, reference)
    base_tensor = base['_tensor']
    base_residual = (
        _act(components['charge'][np.ix_(SHELL_INDICES, LOW_INDICES)],
             components['p4'], base_tensor)
        + _act(components['phase'][np.ix_(SHELL_INDICES, LOW_INDICES)],
               components['x4'], base_tensor))
    full_weights = np.sum(
        np.abs(reference['tensor'][SHELL_INDICES]) ** 2, axis=1)

    shell_rows = []
    for position, rank in enumerate(SHELL_INDICES):
        coupling_block = (
            np.kron(components['charge'][rank:rank + 1, LOW_INDICES],
                    components['p4'])
            + np.kron(components['phase'][rank:rank + 1, LOW_INDICES],
                      components['x4']))
        bare_energies = w1[rank] + components['w4']
        minimum_denominator = float(np.min(np.abs(
            bare_energies[:, None] - reference['values'][None, :])))
        if recompute_leave_one_out:
            leave_one_out = _evaluate(
                np.delete(np.arange(180, dtype=int), rank),
                components, reference)
            transition_error = leave_one_out['maximum_transition_error_MHz']
            minimum_overlap = leave_one_out['minimum_state_overlap']
            primary_error = max(
                row['absolute_error']
                for row in leave_one_out['primary_path_coupling_errors'])
            logical_error = max(
                row['absolute_error']
                for row in leave_one_out['logical_coupling_errors'])
        else:
            (transition_error, primary_error, logical_error,
             minimum_overlap) = LEAVE_ONE_OUT_DIAGNOSTICS[int(rank)]
        shell_rows.append({
            'stage1_rank': int(rank),
            'stage1_energy_GHz': float(w1[rank]),
            'minimum_bare_denominator_GHz': minimum_denominator,
            'coupling_block_frobenius_GHz': float(
                np.linalg.norm(coupling_block)),
            'base160_residual_max_GHz': float(np.max(np.linalg.norm(
                base_residual[position], axis=0))),
            'base160_residual_by_final_state_GHz': np.linalg.norm(
                base_residual[position], axis=0).tolist(),
            'full_low_energy_weight_sum': float(np.sum(full_weights[position])),
            'full_low_energy_weight_max': float(np.max(full_weights[position])),
            'full_low_energy_weight_by_state': full_weights[position].tolist(),
            'leave_one_out_max_transition_error_MHz': transition_error,
            'leave_one_out_minimum_overlap': minimum_overlap,
            'leave_one_out_max_primary_abs_error': primary_error,
            'leave_one_out_max_logical_abs_error': logical_error,
            'leave_one_out_source': (
                'recomputed' if recompute_leave_one_out
                else 'verified_prior_rank_only_pass'),
        })

    # Sort by direct effect on the requested observables, then by the
    # low-basis residual.  Raw diagnostics remain available for every state.
    shell_rows.sort(key=lambda row: (
        row['leave_one_out_max_transition_error_MHz'],
        row['leave_one_out_max_primary_abs_error'],
        row['leave_one_out_max_logical_abs_error'],
        row['base160_residual_max_GHz']), reverse=True)
    for priority, row in enumerate(shell_rows, 1):
        row['priority_rank'] = priority
        row['in_inherited_symmetric_shell'] = bool(
            row['stage1_rank'] in set(HAND_INDICES.tolist()))
    return metadata, components, reference, base, shell_rows


def _plot_candidates(cases, path):
    labels = list(cases)
    display_labels = [
        label.replace('asymmetric_priority_', 'asym ')
        .replace('inherited_symmetric_shell_', 'inherited ')
        .replace('energy_', 'energy ')
        for label in labels
    ]
    max_transition = [cases[label]['maximum_transition_error_MHz']
                      for label in labels]
    min_overlap = [cases[label]['minimum_state_overlap'] for label in labels]
    middle_error = []
    max_leakage_error = []
    for label in labels:
        primary = cases[label]['primary_path_coupling_errors']
        middle_error.append(primary[1]['relative_error'] * 100)
        max_leakage_error.append(100 * max(
            row['relative_error'] for row in cases[label]['logical_coupling_errors']))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].bar(display_labels, max_transition)
    axes[0, 0].set_ylabel('max transition error (MHz)')
    axes[0, 0].set_yscale('log')
    axes[0, 1].bar(display_labels, [1.0 - value for value in min_overlap])
    axes[0, 1].set_ylabel('tracking infidelity (1 - min overlap)')
    axes[0, 1].set_yscale('log')
    axes[1, 0].bar(display_labels, middle_error)
    axes[1, 0].set_ylabel('7-8 relative error (%)')
    axes[1, 0].set_yscale('log')
    axes[1, 1].bar(display_labels, max_leakage_error)
    axes[1, 1].set_ylabel('max logical-coupling error (%)')
    axes[1, 1].set_yscale('log')
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Asymmetric N=51 retained-basis candidates')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR):
    if not CANDIDATE_BASES:
        raise RuntimeError(
            'Inspect --rank-only output and define explicit CANDIDATE_BASES.')
    metadata, components, reference, base, shell_rows = diagnose_shell(
        checkpoint_dir)
    cases = {'energy_k160': _strip_private(base)}
    for label, indices in CANDIDATE_BASES.items():
        cases[label] = _strip_private(_evaluate(
            np.asarray(indices, dtype=int), components, reference))
    inherited = _evaluate(HAND_INDICES, components, reference)
    cases['inherited_symmetric_shell_k170'] = _strip_private(inherited)

    checkpoint_path = Path(checkpoint_dir).resolve()
    checkpoint_label = str(
        checkpoint_path.relative_to(ROOT)
        if checkpoint_path.is_relative_to(ROOT) else checkpoint_path)
    result = {
        'provenance': {
            'description': 'targeted asymmetric-shell audit from verified k180 checkpoint',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': checkpoint_label,
            'historical_checkpoint_reused': False,
        },
        'physical_parameters': BENCHMARK,
        'spatial_cutoffs': ASYMMETRIC_SPATIAL_51,
        'reference': {
            'energies_GHz': reference['values'].tolist(),
            'transitions_GHz': reference['transitions'][1:].tolist(),
            'pathways': reference['pathways'],
            'primary_couplings': _coupling_errors(
                reference['matrix'], reference['matrix'], PRIMARY_COUPLINGS),
            'logical_couplings': _coupling_errors(
                reference['matrix'], reference['matrix'], LOGICAL_COUPLINGS),
        },
        'shell_state_ranking': shell_rows,
        'candidate_cases': cases,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'{ARTIFACT_STEM}.json'
    figure_path = output_dir / f'{ARTIFACT_STEM}.png'
    json_path.write_text(json.dumps(result, indent=2))
    _plot_candidates(cases, figure_path)
    print(json.dumps({
        'artifact': str(json_path),
        'figure': str(figure_path),
        'ranking': [
            {key: row[key] for key in (
                'priority_rank', 'stage1_rank',
                'leave_one_out_max_transition_error_MHz',
                'leave_one_out_max_primary_abs_error',
                'leave_one_out_max_logical_abs_error',
                'base160_residual_max_GHz',
                'minimum_bare_denominator_GHz')}
            for row in shell_rows],
        'cases': {
            label: {
                'indices': case['retained_indices'],
                'max_transition_error_MHz': case['maximum_transition_error_MHz'],
                'minimum_overlap': case['minimum_state_overlap'],
                'top_path': case['pathways'][0],
                'max_omitted_residual_GHz': case['maximum_omitted_residual_GHz'],
            } for label, case in cases.items()
        },
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank-only', action='store_true')
    parser.add_argument('--recompute-leave-one-out', action='store_true')
    args = parser.parse_args()
    if args.rank_only:
        _, _, _, _, shell_rows = diagnose_shell(
            recompute_leave_one_out=args.recompute_leave_one_out)
        print(json.dumps(shell_rows, indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
