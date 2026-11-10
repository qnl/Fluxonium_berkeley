"""Stage B basis diagnostics for the symmetric three-mode S23 model.

Numerical oscillator centers and scales in this module change only the finite
basis used to represent the physical S23 coordinates.  They do not add terms
to, or otherwise alter, the Hamiltonian.

The profiles are intentionally separate so that expensive diagnostics can be
run selectively.  For example::

    python -m Simulations.MultimodeGridium.validate_s23_basis centers
    python -m Simulations.MultimodeGridium.validate_s23_basis sigma_cutoff
    python -m Simulations.MultimodeGridium.validate_s23_basis scales
    python -m Simulations.MultimodeGridium.validate_s23_basis second_bias
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from Circuit_Objs.qchard_gridium3 import (
    SymmetricThreeModeGridium,
    table_s1_regime_a,
)


NLEVELS = 12
DEFAULT_SCALES = SymmetricThreeModeGridium(
    **table_s1_regime_a, nlev=NLEVELS).phase_scales


def _csr_memory_bytes(sparse):
    return int(sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes)


def evaluate_case(
        *, cutoffs, phase_centers, scale_factors=(1.0, 1.0, 1.0),
        phi_ext=0.0, theta_ext=np.pi):
    """Evaluate one explicitly specified finite-basis representation."""
    phase_scales = tuple(
        scale * factor
        for scale, factor in zip(DEFAULT_SCALES, scale_factors))
    model = SymmetricThreeModeGridium(
        **table_s1_regime_a,
        phi_ext=phi_ext,
        theta_ext=theta_ext,
        trunc_sigma=cutoffs[0],
        trunc_delta=cutoffs[1],
        trunc_s=cutoffs[2],
        nlev=NLEVELS,
        phase_centers=phase_centers,
        phase_scales=phase_scales,
        eigensolver_tol=1e-10)

    start = perf_counter()
    hamiltonian = model.hamiltonian_primitive()
    build_seconds = perf_counter() - start
    start = perf_counter()
    energies, eigenvectors = model.levels(eigvecs=True)
    solve_seconds = perf_counter() - start

    sparse = hamiltonian.data.as_scipy()
    hermiticity = hamiltonian - hamiltonian.dag()
    hermiticity_sparse = hermiticity.data.as_scipy()
    hermiticity_residual = (
        0.0 if hermiticity_sparse.nnz == 0
        else float(np.max(np.abs(hermiticity_sparse.data))))
    eigenpair_residual = max(
        float((hamiltonian * vector - energy * vector).norm())
        for energy, vector in zip(energies, eigenvectors))

    matrix_elements = {}
    for name in ('phi_2', 'n_1'):
        projected = model.operator(name, basis='energy', nlev=NLEVELS)
        matrix_elements[name] = np.abs(projected.full()[0]).tolist()

    return {
        'cutoffs_sigma_delta_s': list(cutoffs),
        'phase_centers_sigma_delta_s': list(phase_centers),
        'phase_scales_sigma_delta_s': list(phase_scales),
        'scale_factors_sigma_delta_s': list(scale_factors),
        'phi_ext': float(phi_ext),
        'theta_ext': float(theta_ext),
        'tensor_dimension': model.hilbert_dimension,
        'hamiltonian_nnz': int(sparse.nnz),
        'hamiltonian_csr_bytes': _csr_memory_bytes(sparse),
        'build_seconds': build_seconds,
        'solve_seconds': solve_seconds,
        'hermiticity_max_abs': hermiticity_residual,
        'eigenpair_residual_max': eigenpair_residual,
        'energies_GHz': energies.tolist(),
        'transitions_from_ground_GHz': (energies - energies[0]).tolist(),
        'abs_ground_matrix_elements': matrix_elements,
    }


def profile_cases(profile):
    centered = (0.0, np.pi / 2, 0.0)
    if profile == 'centers':
        return [
            {'label': 'zero', 'cutoffs': (24, 24, 20),
             'phase_centers': (0.0, 0.0, 0.0)},
            {'label': 'delta_inductive_center', 'cutoffs': (24, 24, 20),
             'phase_centers': centered},
            {'label': 'delta_classical_local_minimum',
             'cutoffs': (24, 24, 20),
             'phase_centers': (0.0, 0.264356265, 0.0)},
        ]
    if profile == 'sigma_cutoff':
        return [
            {'label': 'N_sigma_{}'.format(value),
             'cutoffs': (value, 24, 24), 'phase_centers': centered}
            for value in (24, 28, 32, 36, 40)]
    if profile == 'delta_cutoff':
        return [
            {'label': 'N_delta_{}'.format(value),
             'cutoffs': (28, value, 24), 'phase_centers': centered}
            for value in (24, 28, 32)]
    if profile == 's_cutoff':
        return [
            {'label': 'N_s_{}'.format(value),
             'cutoffs': (28, 28, value), 'phase_centers': centered}
            for value in (16, 20, 24, 28)]
    if profile == 'scales':
        cases = []
        for mode_index, mode_name in enumerate(('sigma', 'delta', 's')):
            for factor in (0.75, 1.0, 1.25):
                factors = [1.0, 1.0, 1.0]
                factors[mode_index] = factor
                cases.append({
                    'label': '{}_scale_{:g}'.format(mode_name, factor),
                    'cutoffs': (24, 24, 20),
                    'phase_centers': centered,
                    'scale_factors': tuple(factors),
                })
        return cases
    if profile == 'second_bias':
        return [
            {'label': 'phi_pi_theta_pi_24_24_20',
             'cutoffs': (24, 24, 20),
             'phase_centers': (np.pi, np.pi / 2, np.pi),
             'phi_ext': np.pi},
            {'label': 'phi_pi_theta_pi_28_28_24',
             'cutoffs': (28, 28, 24),
             'phase_centers': (np.pi, np.pi / 2, np.pi),
             'phi_ext': np.pi},
        ]
    raise ValueError('Unknown profile {!r}.'.format(profile))


def run_profile(profile):
    results = []
    for configuration in profile_cases(profile):
        configuration = dict(configuration)
        label = configuration.pop('label')
        result = evaluate_case(**configuration)
        result['label'] = label
        results.append(result)
    return {
        'model': 'SymmetricThreeModeGridium, paper Eq. S23',
        'parameter_source': 'Table S1 regime a',
        'profile': profile,
        'numerical_basis_warning': (
            'Centers and scales are finite-basis parameters only. Agreement '
            'at one setting is not physical validation.'),
        'results': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'profile', choices=(
            'centers', 'sigma_cutoff', 'delta_cutoff', 's_cutoff',
            'scales', 'second_bias'))
    arguments = parser.parse_args()
    print(json.dumps(run_profile(arguments.profile), indent=2))
