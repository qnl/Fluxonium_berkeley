"""Report numerical diagnostics for the symmetric three-mode S23 model.

This is a convergence/diagnostic runner, not a fit and not a physics model.
It uses Table S1 regime a at the paper's (phi_ext, theta_ext) = (0, pi)
operating point and prints machine-readable JSON.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from Circuit_Objs.qchard_gridium3 import (
    SymmetricThreeModeGridium,
    table_s1_regime_a,
)
from Circuit_Objs.qchard_statetracking import track_eigensystems


CUTOFFS = ((24, 24, 20), (26, 26, 22), (28, 28, 24))
NLEVELS = 12


def _max_sparse_abs(matrix):
    sparse = matrix.data.as_scipy()
    return 0.0 if sparse.nnz == 0 else float(np.max(np.abs(sparse.data)))


def _csr_memory_bytes(sparse):
    return int(sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes)


def _model(cutoffs, phi_ext=0.0):
    return SymmetricThreeModeGridium(
        **table_s1_regime_a,
        phi_ext=phi_ext,
        theta_ext=np.pi,
        trunc_sigma=cutoffs[0],
        trunc_delta=cutoffs[1],
        trunc_s=cutoffs[2],
        nlev=NLEVELS,
        eigensolver_tol=1e-10,
    )


def _cutoff_report(cutoffs):
    start = perf_counter()
    model = _model(cutoffs)
    hamiltonian = model.hamiltonian_primitive()
    build_seconds = perf_counter() - start
    sparse = hamiltonian.data.as_scipy()

    start = perf_counter()
    energies, eigenvectors = model.levels(eigvecs=True)
    solve_seconds = perf_counter() - start
    transitions = energies - energies[0]

    residuals = []
    for energy, vector in zip(energies, eigenvectors):
        residuals.append(float((hamiltonian * vector - energy * vector).norm()))

    return model, {
        'cutoffs_sigma_delta_s': list(cutoffs),
        'tensor_dimension': model.hilbert_dimension,
        'hamiltonian_nnz': int(sparse.nnz),
        'hamiltonian_csr_bytes': _csr_memory_bytes(sparse),
        'hamiltonian_hermiticity_max_abs': _max_sparse_abs(
            hamiltonian - hamiltonian.dag()),
        'build_seconds': build_seconds,
        'solve_seconds': solve_seconds,
        'eigen_residual_max': max(residuals),
        'energies_GHz': energies.tolist(),
        'transitions_from_ground_GHz': transitions.tolist(),
    }


def run_validation():
    reports = []
    largest_model = None
    for cutoffs in CUTOFFS:
        largest_model, report = _cutoff_report(cutoffs)
        if reports:
            previous = np.asarray(reports[-1]['energies_GHz'])
            current = np.asarray(report['energies_GHz'])
            report['energy_change_from_previous_GHz'] = (
                current - previous).tolist()
            report['max_abs_energy_change_from_previous_GHz'] = float(
                np.max(np.abs(current - previous)))
            previous_transition = np.asarray(
                reports[-1]['transitions_from_ground_GHz'])
            current_transition = np.asarray(
                report['transitions_from_ground_GHz'])
            report['max_abs_transition_change_from_previous_GHz'] = float(
                np.max(np.abs(current_transition - previous_transition)))
        reports.append(report)

    matrix_elements = {}
    for name in ('phi_2', 'n_1'):
        projected = largest_model.operator(
            name, basis='energy', nlev=NLEVELS).full()
        matrix_elements[name] = {
            'abs_0_to_j': np.abs(projected[0]).tolist(),
            'hermiticity_max_abs': float(
                np.max(np.abs(projected - projected.conj().T))),
        }

    shifted_model = _model(CUTOFFS[-1], phi_ext=1e-3)
    reference_energies, reference_vectors = largest_model.levels(eigvecs=True)
    shifted_energies, shifted_vectors = shifted_model.levels(eigvecs=True)
    tracking = track_eigensystems(
        reference_energies, reference_vectors,
        shifted_energies, shifted_vectors)

    return {
        'model': 'SymmetricThreeModeGridium, paper Eq. S23',
        'parameter_source': 'Table S1 regime a',
        'parameters_h_GHz': dict(table_s1_regime_a),
        'biases': {'phi_ext': 0.0, 'theta_ext': float(np.pi)},
        'phase_scales_largest_case': list(largest_model.phase_scales),
        'phase_scale_role': 'numerical basis parameters, not physical inputs',
        'cutoff_reports': reports,
        'matrix_elements_largest_case': matrix_elements,
        'same_basis_tracking_phi_ext_plus_0p001': {
            'permutation': tracking.permutation.tolist(),
            'matched_overlaps': tracking.matched_overlaps.tolist(),
            'ambiguities': tracking.ambiguities,
            'has_unresolved_ambiguity': tracking.has_unresolved_ambiguity,
        },
        'paper_comparison_limit': (
            'The paper plots spectra and matrix-element trends for Table S1 '
            'but does not tabulate reference eigenenergies or matrix elements.'),
    }


if __name__ == '__main__':
    print(json.dumps(run_validation(), indent=2))
