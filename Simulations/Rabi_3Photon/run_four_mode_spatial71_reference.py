"""Portable, fail-closed runner for frozen asymmetric N=71 references.

Each named profile prepares exactly one expensive Stage-1 solve.  Its full
parent reference, nested retained-basis audits, and the N=51 -> N=71 spatial
comparison all derive from that same solve.  The larger parent is a reference
endpoint for the configured comparison, not ground truth or an oracle.

Execution is deliberately guarded by ``--execute``.  Importing the module or
using ``--dry-run`` never assembles the Hamiltonian and never calls ARPACK.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import qutip as qt
import scipy
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sps

from Circuit_Objs.qchard_gridium_netlist import (
    _assemble_nonlinear_sector,
    _coeffs,
    _stage1_eigensolve,
)
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import (
    _apply_state_assignment,
    _stage2_templates,
    _three_step_paths,
    _track,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / 'research' / 'checkpoints'
DEFAULT_N51_CHECKPOINT = (
    DEFAULT_OUTPUT_ROOT / '2026-09-26-asymmetric-n51'
)
DEFAULT_PROFILE_NAME = 'n71-k180'
NSTATES = 11

PHYSICAL_PARAMETERS = {
    'EJ': 5.0,
    'EC': 0.5,
    'EL': 1.0,
    'ELK': 1.0,
    'EJS': 4.0,
    'ECS': 8.0,
    'eC': 5.5,
    'eP': 10.0,
    'eps_J': 0.10,
    'eps_LK': 0.05,
    'ng': 0.0,
    'phi_ext': 0.0,
    'theta_ext': float(np.pi),
}
SPATIAL_CUTOFFS = {
    'n1max': 4,
    'N2': 71,
    'L2': 11.0,
    'N3': 71,
    'L3': 14.0,
    'N4': 8,
}
N51_SPATIAL_CUTOFFS = {
    'n1max': 4,
    'N2': 51,
    'L2': 11.0,
    'N3': 51,
    'L3': 14.0,
    'N4': 8,
}
N51_SOLVER_SETTINGS = {
    'k': 180,
    'sigma': -24.0,
    'which': 'LM',
    'tol': 1e-8,
    'ncv': None,
    'v0': None,
    'permc_spec': 'MMD_AT_PLUS_A',
}
PROFILE_DEFINITIONS = {
    'n71-k180': {
        'configuration': {
            'physical_parameters': PHYSICAL_PARAMETERS,
            'spatial_cutoffs': SPATIAL_CUTOFFS,
            'solver_settings': deepcopy(N51_SOLVER_SETTINGS),
        },
        'retained_cases': (
            ('asymmetric_priority_k170', 170),
            ('asymmetric_priority_k172', 172),
            ('asymmetric_priority_k175', 175),
        ),
        'retained_index_mode': 'asymmetric_shell_priority',
        'final_directory_name': 'four-mode-asymmetric-n71-k180',
    },
    'n71-k220': {
        'configuration': {
            'physical_parameters': PHYSICAL_PARAMETERS,
            'spatial_cutoffs': SPATIAL_CUTOFFS,
            'solver_settings': {
                'k': 220,
                'sigma': -24.0,
                'which': 'LM',
                'tol': 1e-8,
                'ncv': None,
                'v0': None,
                'permc_spec': 'MMD_AT_PLUS_A',
            },
        },
        'retained_cases': (
            ('nested_prefix_k180', 180),
            ('nested_prefix_k200', 200),
            ('nested_prefix_k220', 220),
        ),
        'retained_index_mode': 'prefix',
        'final_directory_name': 'four-mode-asymmetric-n71-k220',
    },
}
# Legacy aliases preserve the original import-level API and the default k=180
# profile.  New code resolves settings through ``profile_definition``.
FROZEN_CONFIGURATION = PROFILE_DEFINITIONS[
    DEFAULT_PROFILE_NAME
]['configuration']
SOLVER_SETTINGS = FROZEN_CONFIGURATION['solver_settings']
FINAL_DIRECTORY_NAME = PROFILE_DEFINITIONS[
    DEFAULT_PROFILE_NAME
]['final_directory_name']
EXPECTED_STAGE1_DIMENSION = (
    (2 * SPATIAL_CUTOFFS['n1max'] + 1)
    * SPATIAL_CUTOFFS['N2']
    * SPATIAL_CUTOFFS['N3']
)
EXPECTED_OPERATOR_NAMES = ('n1', 'n2', 'n3', 'x2', 'x3')
N51_ORACLE_SHA256 = {
    'checkpoint.json': (
        '03058680bebe8569be4c9d071682a5b00105765454e45c2f14bff68398c04e51'
    ),
    'stage1_eigensystem.npz': (
        '3f836490946a400877837eaf4df74305e4bad9dd9e57b69a4070abcde1938ea8'
    ),
    'projected_operators.npz': (
        '7236dc7baf9fd25fbe7b75965aea38ad7484fae311d4162df803fd7d34112ce1'
    ),
}
PROVENANCE_SOURCE_FILES = (
    Path('Simulations/Rabi_3Photon/run_four_mode_spatial71_reference.py'),
    Path('Simulations/Rabi_3Photon/four_mode_retained_basis_audit.py'),
    Path('Circuit_Objs/qchard_gridium_netlist.py'),
)
MAX_EIGENPAIR_RESIDUAL = 1e-7
MAX_ORTHOGONALITY_RESIDUAL = 1e-8
# Warning-only thresholds for state-specific cross-grid claims.  They do not
# alter the Hungarian assignment or any eigensystem.  Raw overlaps and
# subspace singular values remain the authoritative diagnostics.
MIN_ASSIGNED_OVERLAP_FOR_CLAIM = 0.90
MAX_COMPETITOR_RATIO_FOR_CLAIM = 0.50
MIN_OVERLAP_MARGIN_FOR_CLAIM = 0.25
REQUIRED_THREAD_ENVIRONMENT = {
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
}
# Frozen from ``four_mode_asymmetric_shell_audit._SHELL_PRIORITY``.  Keeping
# the exact tuple here makes the external job self-contained and prevents a
# later audit edit from silently changing this eigenproblem's candidate sets.
ASYMMETRIC_SHELL_PRIORITY = (
    170, 172, 171, 168, 162, 176, 177, 167, 163, 165,
    166, 160, 174, 169, 173, 178, 161, 175, 164, 179,
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=_json_default,
    )


def configuration_fingerprint(configuration=FROZEN_CONFIGURATION):
    return hashlib.sha256(_canonical_json(configuration).encode()).hexdigest()


CONFIGURATION_SHA256_BY_PROFILE = {
    name: configuration_fingerprint(definition['configuration'])
    for name, definition in PROFILE_DEFINITIONS.items()
}
CONFIGURATION_SHA256 = CONFIGURATION_SHA256_BY_PROFILE[DEFAULT_PROFILE_NAME]


def profile_definition(profile_name=DEFAULT_PROFILE_NAME):
    try:
        return PROFILE_DEFINITIONS[profile_name]
    except KeyError as error:
        choices = ', '.join(sorted(PROFILE_DEFINITIONS))
        raise ValueError(
            f'Unknown frozen profile {profile_name!r}; choose from {choices}.'
        ) from error


def validate_frozen_configuration(configuration,
                                  profile_name=DEFAULT_PROFILE_NAME):
    """Reject any physical, spatial, or solver mismatch before a solve."""
    expected = profile_definition(profile_name)['configuration']
    if _canonical_json(configuration) != _canonical_json(expected):
        raise ValueError(
            'Configuration does not exactly match the frozen asymmetric N=71 '
            f'{profile_name} eigenproblem.'
        )
    if (configuration_fingerprint(configuration)
            != CONFIGURATION_SHA256_BY_PROFILE[profile_name]):
        raise ValueError('Frozen-configuration fingerprint mismatch.')


def _profile_for_checkpoint(configuration, fingerprint, declared_profile=None):
    candidates = (
        (declared_profile,) if declared_profile is not None
        else tuple(PROFILE_DEFINITIONS)
    )
    for profile_name in candidates:
        if profile_name not in PROFILE_DEFINITIONS:
            continue
        expected = profile_definition(profile_name)['configuration']
        if (_canonical_json(configuration) == _canonical_json(expected)
                and fingerprint
                == CONFIGURATION_SHA256_BY_PROFILE[profile_name]):
            return profile_name
    raise ValueError('Checkpoint configuration fingerprint mismatch.')


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp-{uuid.uuid4().hex}')
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_write_json(path, value):
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, allow_nan=False, default=_json_default)
        + '\n',
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(arguments):
    result = subprocess.run(
        ['git', *arguments], cwd=ROOT, check=False, capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return result.stdout.rstrip()


def collect_runtime_provenance():
    status = _git_output(['status', '--porcelain=v1'])
    return {
        'generated_utc': _utc_now(),
        'git_revision': _git_output(['rev-parse', 'HEAD']),
        'git_branch': _git_output(['branch', '--show-current']),
        'git_dirty': None if status is None else bool(status),
        'git_status_porcelain': None if status is None else status.splitlines(),
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'qutip_version': qt.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'argv': [Path(sys.argv[0]).name, *sys.argv[1:]],
        'source_file_sha256': {
            str(path): _sha256_file(ROOT / path)
            for path in PROVENANCE_SOURCE_FILES
        },
        'thread_environment': {
            name: os.environ.get(name)
            for name in REQUIRED_THREAD_ENVIRONMENT
        },
    }


def validate_thread_environment(environment=None):
    environment = os.environ if environment is None else environment
    mismatches = {
        name: environment.get(name)
        for name, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            'Execute only with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, '
            f'and MKL_NUM_THREADS=1; mismatches: {mismatches}'
        )


def dry_run_manifest(output_root=DEFAULT_OUTPUT_ROOT,
                     n51_checkpoint=DEFAULT_N51_CHECKPOINT,
                     profile_name=DEFAULT_PROFILE_NAME):
    """Return the exact job description without assembling a matrix."""
    profile = profile_definition(profile_name)
    configuration = profile['configuration']
    validate_frozen_configuration(configuration, profile_name)
    spatial = configuration['spatial_cutoffs']
    parent_k = configuration['solver_settings']['k']
    return {
        'mode': 'dry-run-no-matrix-assembly',
        'profile': profile_name,
        'configuration': deepcopy(configuration),
        'configuration_sha256': CONFIGURATION_SHA256_BY_PROFILE[
            profile_name
        ],
        'expected_stage1_dimension': _stage1_dimension(spatial),
        'parent_k': parent_k,
        'expected_stage2_parent_dimension': parent_k * spatial['N4'],
        'expected_low_energy_states': NSTATES,
        'candidate_stage1_indices': {
            name: retained_indices(dimension, profile_name).tolist()
            for name, dimension in profile['retained_cases']
        },
        'retained_stage1_dimensions': [
            dimension for _, dimension in profile['retained_cases']
        ],
        'reference_semantics': (
            f'k={parent_k} is the larger reference endpoint for nested '
            'same-parent comparisons; it is not ground truth or an oracle'
        ),
        'n51_checkpoint': str(Path(n51_checkpoint)),
        'final_output_directory': str(
            Path(output_root) / profile['final_directory_name']
        ),
        'required_thread_environment': REQUIRED_THREAD_ENVIRONMENT,
        'state_assignment_claim_warning_thresholds': {
            'minimum_assigned_overlap': MIN_ASSIGNED_OVERLAP_FOR_CLAIM,
            'maximum_competitor_ratio': MAX_COMPETITOR_RATIO_FOR_CLAIM,
            'minimum_overlap_margin': MIN_OVERLAP_MARGIN_FOR_CLAIM,
        },
        'completion_requires': [
            'checkpoint.json with status=complete',
            'matching configuration_sha256',
            'COMPLETE marker written last',
        ],
        'expected_checkpoint_files': [
            'run_state.json',
            'stage1_eigensystem.npz',
            'projected_operators.npz',
            'analysis.json',
            'checkpoint.json',
            'COMPLETE',
        ],
    }


def _stage1_dimension(spatial_cutoffs):
    return (
        (2 * spatial_cutoffs['n1max'] + 1)
        * spatial_cutoffs['N2']
        * spatial_cutoffs['N3']
    )


def retained_indices(dimension, profile_name=DEFAULT_PROFILE_NAME):
    profile = profile_definition(profile_name)
    allowed = tuple(value for _, value in profile['retained_cases'])
    if dimension not in allowed:
        choices = ', '.join(f'k{value}' for value in allowed)
        raise ValueError(
            f'Only the frozen {profile_name} candidates {choices} exist.'
        )
    if profile['retained_index_mode'] == 'prefix':
        indices = np.arange(dimension, dtype=int)
    elif profile['retained_index_mode'] == 'asymmetric_shell_priority':
        extras = sorted(ASYMMETRIC_SHELL_PRIORITY[:dimension - 160])
        indices = np.asarray(list(range(160)) + extras, dtype=int)
    else:
        raise ValueError('Frozen retained-index mode is invalid.')
    if len(indices) != dimension or len(np.unique(indices)) != dimension:
        raise AssertionError('Asymmetric retained-basis definition is invalid.')
    parent_k = profile['configuration']['solver_settings']['k']
    if np.any(indices < 0) or np.any(indices >= parent_k):
        raise AssertionError('Retained indices exceed the frozen parent solve.')
    return indices


def _assemble_stage1(profile_name=DEFAULT_PROFILE_NAME):
    configuration = profile_definition(profile_name)['configuration']
    parameters = configuration['physical_parameters']
    spatial = configuration['spatial_cutoffs']
    expected_dimension = _stage1_dimension(spatial)
    ECm, K, EJ1, EJ2 = _coeffs(
        parameters['EJ'], parameters['EC'], parameters['EL'],
        parameters['ELK'], parameters['EJS'], parameters['ECS'],
        parameters['eps_J'], parameters['eps_LK'], parameters['eC'],
        parameters['eP'],
    )
    matrix, raw_operators = _assemble_nonlinear_sector(
        ECm[:3, :3], K[:2, :2], EJ1, EJ2, parameters['EJS'],
        parameters['ng'], parameters['phi_ext'], parameters['theta_ext'],
        spatial['n1max'], spatial['N2'], spatial['L2'], spatial['N3'],
        spatial['L3'],
    )
    if matrix.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f'Unexpected Stage-1 shape {matrix.shape}; expected '
            f'{expected_dimension} squared.'
        )
    return ECm, K, matrix, raw_operators


def _validate_eigensystem(matrix, values, vectors,
                          profile_name=DEFAULT_PROFILE_NAME):
    configuration = profile_definition(profile_name)['configuration']
    k = configuration['solver_settings']['k']
    expected_dimension = _stage1_dimension(
        configuration['spatial_cutoffs']
    )
    if values.shape != (k,):
        raise ValueError(f'Expected {k} eigenvalues, received {values.shape}.')
    if vectors.shape != (expected_dimension, k):
        raise ValueError(f'Unexpected eigenvector shape {vectors.shape}.')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError('Stage-1 eigensystem contains non-finite entries.')
    if np.any(np.diff(values) < 0):
        raise ValueError('Stage-1 eigenvalues are not sorted.')

    residuals = np.linalg.norm(matrix @ vectors - vectors * values, axis=0)
    orthogonality = np.linalg.norm(
        vectors.conj().T @ vectors - np.eye(k), ord=np.inf,
    )
    if not np.all(np.isfinite(residuals)) or not np.isfinite(orthogonality):
        raise ValueError('Residual diagnostics are non-finite.')
    if float(np.max(residuals)) > MAX_EIGENPAIR_RESIDUAL:
        raise ValueError(
            f'Max eigenpair residual {np.max(residuals):.3e} exceeds '
            f'{MAX_EIGENPAIR_RESIDUAL:.3e}.'
        )
    if float(orthogonality) > MAX_ORTHOGONALITY_RESIDUAL:
        raise ValueError(
            f'Orthogonality residual {orthogonality:.3e} exceeds '
            f'{MAX_ORTHOGONALITY_RESIDUAL:.3e}.'
        )
    return residuals, float(orthogonality)


def _project_operators(raw, vectors, profile_name=DEFAULT_PROFILE_NAME):
    kron3 = raw['kron3']

    def projected(operator):
        return vectors.conj().T @ (operator @ vectors)

    operators = {
        'n1': projected(kron3(
            sps.diags(raw['nvec']), raw['I2'], raw['I3'])),
        'n2': projected(kron3(raw['I1'], raw['n2'], raw['I3'])),
        'n3': projected(kron3(raw['I1'], raw['I2'], raw['n3'])),
        'x2': projected(kron3(
            raw['I1'], sps.diags(raw['x2']), raw['I3'])),
        'x3': projected(kron3(
            raw['I1'], raw['I2'], sps.diags(raw['x3']))),
    }
    parent_k = profile_definition(
        profile_name
    )['configuration']['solver_settings']['k']
    expected = (parent_k, parent_k)
    for name, operator in operators.items():
        if operator.shape != expected or not np.all(np.isfinite(operator)):
            raise ValueError(f'Invalid projected operator {name}.')
    return operators


def _solve_stage1_once(profile_name=DEFAULT_PROFILE_NAME):
    configuration = profile_definition(profile_name)['configuration']
    validate_frozen_configuration(configuration, profile_name)
    started = time.perf_counter()
    ECm, K, matrix, raw = _assemble_stage1(profile_name)
    assembly_seconds = time.perf_counter() - started
    settings = configuration['solver_settings']
    values, vectors, diagnostics = _stage1_eigensolve(
        matrix,
        k=settings['k'],
        sigma=settings['sigma'],
        which=settings['which'],
        tol=settings['tol'],
        ncv=settings['ncv'],
        v0=settings['v0'],
        permc_spec=settings['permc_spec'],
        collect_diagnostics=True,
    )
    residuals, orthogonality = _validate_eigensystem(
        matrix, values, vectors, profile_name,
    )
    projection_started = time.perf_counter()
    operators = _project_operators(raw, vectors, profile_name)
    projection_seconds = time.perf_counter() - projection_started
    diagnostics.update({
        'matrix_dimension': int(matrix.shape[0]),
        'matrix_nnz': int(matrix.nnz),
        'k': settings['k'],
        'sigma': settings['sigma'],
        'which': settings['which'],
        'tol': settings['tol'],
        'ncv': settings['ncv'],
        'v0': settings['v0'],
        'permc_spec': settings['permc_spec'],
        'assembly_seconds': assembly_seconds,
        'projection_seconds': projection_seconds,
        'eigenpair_residuals': residuals.tolist(),
        'max_eigenpair_residual': float(np.max(residuals)),
        'median_eigenpair_residual': float(np.median(residuals)),
        'orthogonality_residual_inf': orthogonality,
        'validation_thresholds': {
            'max_eigenpair_residual': MAX_EIGENPAIR_RESIDUAL,
            'orthogonality_residual_inf': MAX_ORTHOGONALITY_RESIDUAL,
        },
    })
    return ECm, K, values, vectors, operators, residuals, diagnostics


def _coupling_rows(reference_matrix, candidate_matrix, pairs):
    rows = []
    for initial, final in pairs:
        reference = float(abs(reference_matrix[initial, final]))
        candidate = float(abs(candidate_matrix[initial, final]))
        difference = abs(candidate - reference)
        rows.append({
            'transition': [int(initial), int(final)],
            'reference': reference,
            'candidate': candidate,
            'absolute_change': difference,
            'relative_change': (
                difference / reference if reference > 1e-12 else None
            ),
        })
    return rows


def _dominant_logical_pairs(matrix, count_per_logical=5):
    pairs = []
    for logical in (0, 1):
        ranked = sorted(
            (
                (float(abs(matrix[logical, state])), logical, state)
                for state in range(2, NSTATES)
            ),
            reverse=True,
        )[:count_per_logical]
        pairs.extend((logical, state) for _, _, state in ranked)
    return pairs


def _path_pairs(path):
    return list(zip(path[:-1], path[1:]))


def _error_pair(candidate, reference):
    absolute = abs(float(candidate) - float(reference))
    return {
        'absolute': absolute,
        'relative': absolute / abs(float(reference))
        if abs(float(reference)) > 1e-12 else None,
    }


def _thomas_observables(transitions, matrix, state_overlaps, reference,
                        parent_k):
    """Return the explicitly requested same-parent convergence observables."""
    frequencies = {
        'f07': float(transitions[7]),
        'f08': float(transitions[8]),
    }
    reference_frequencies = {
        'f07': float(reference['transitions'][7]),
        'f08': float(reference['transitions'][8]),
    }
    overlaps = {
        'state7_overlap_vs_parent': float(state_overlaps[7]),
        'state8_overlap_vs_parent': float(state_overlaps[8]),
    }
    reference_overlaps = {
        'state7_overlap_vs_parent': 1.0,
        'state8_overlap_vs_parent': 1.0,
    }
    grid_phi = {
        'abs_grid_phi_07': float(abs(matrix[0, 7])),
        'abs_grid_phi_78': float(abs(matrix[7, 8])),
        'abs_grid_phi_81': float(abs(matrix[8, 1])),
    }
    reference_grid_phi = {
        'abs_grid_phi_07': float(abs(reference['matrix'][0, 7])),
        'abs_grid_phi_78': float(abs(reference['matrix'][7, 8])),
        'abs_grid_phi_81': float(abs(reference['matrix'][8, 1])),
    }
    ranked_pathways = [
        {**row, 'rank': rank}
        for rank, row in enumerate(
            _three_step_paths(matrix, NSTATES, count=None), start=1,
        )
    ]
    target_path = [0, 7, 8, 1]
    target_rank = next(
        (row['rank'] for row in ranked_pathways
         if row['path'] == target_path),
        None,
    )
    return {
        'reference_endpoint': f'k{parent_k}',
        'reference_semantics': (
            'larger reference endpoint; not ground truth or an oracle'
        ),
        'convergence_scope': (
            f'agreement toward k{parent_k} does not prove absolute '
            'convergence'
        ),
        'transition_frequencies_GHz': frequencies,
        'state_tracking': overlaps,
        'grid_phi_matrix_elements_abs': grid_phi,
        f'errors_vs_k{parent_k}': {
            'transition_frequencies_GHz': {
                name: _error_pair(value, reference_frequencies[name])
                for name, value in frequencies.items()
            },
            'state_tracking': {
                name: _error_pair(value, reference_overlaps[name])
                for name, value in overlaps.items()
            },
            'grid_phi_matrix_elements_abs': {
                name: _error_pair(value, reference_grid_phi[name])
                for name, value in grid_phi.items()
            },
        },
        'leading_pathways': ranked_pathways[:5],
        'all_ranked_pathways': ranked_pathways,
        'path_0_7_8_1_rank': target_rank,
        'path_0_7_8_1_is_rank_1': target_rank == 1,
    }


def _full_stage2(ECm, K, values, operators, n4):
    indices = np.arange(len(values), dtype=int)
    energies, vectors, tensor, matrix = _stage2_templates(
        ECm, K, values, operators, indices, n4,
    )
    pathways = _three_step_paths(matrix, NSTATES)
    return {
        'indices': indices,
        'energies': energies,
        'vectors': vectors,
        'tensor': tensor,
        'matrix': matrix,
        'transitions': energies - energies[0],
        'pathways': pathways,
    }


def _case_analysis(label, indices, values, matrix, assignment, overlaps,
                   reference, parent_k):
    transitions = values - values[0]
    transition_errors = 1e3 * (
        transitions[1:] - reference['transitions'][1:]
    )
    primary_path = reference['pathways'][0]['path']
    primary_rows = _coupling_rows(
        reference['matrix'], matrix, _path_pairs(primary_path),
    )
    logical_rows = _coupling_rows(
        reference['matrix'], matrix,
        _dominant_logical_pairs(reference['matrix']),
    )
    return {
        'label': label,
        'retained_indices': indices.tolist(),
        'energies_GHz': values.tolist(),
        'transitions_GHz': transitions[1:].tolist(),
        f'transition_errors_MHz_vs_k{parent_k}': transition_errors.tolist(),
        'maximum_transition_error_MHz': float(
            np.max(np.abs(transition_errors))
        ),
        'state_assignment_candidate_row_for_reference_state': (
            assignment.tolist()
        ),
        'state_overlaps': overlaps.tolist(),
        'minimum_state_overlap': float(np.min(overlaps)),
        'reference_primary_path': primary_path,
        'primary_path_grid_phi_errors': primary_rows,
        'maximum_primary_path_grid_phi_absolute_error': max(
            row['absolute_change'] for row in primary_rows
        ),
        'maximum_primary_path_grid_phi_relative_error': max(
            row['relative_change'] for row in primary_rows
            if row['relative_change'] is not None
        ),
        'dominant_logical_leakage_grid_phi_errors': logical_rows,
        'maximum_logical_leakage_grid_phi_absolute_error': max(
            row['absolute_change'] for row in logical_rows
        ),
        'maximum_logical_leakage_grid_phi_relative_error': max(
            row['relative_change'] for row in logical_rows
            if row['relative_change'] is not None
        ),
        'grid_phi_abs': np.abs(matrix).tolist(),
        'pathways': _three_step_paths(matrix, NSTATES),
        'thomas_observables': _thomas_observables(
            transitions, matrix, overlaps, reference, parent_k,
        ),
    }


def _retained_case(label, indices, ECm, K, stage1_values, operators,
                   reference, n4, parent_k):
    values, vectors, tensor, matrix = _stage2_templates(
        ECm, K, stage1_values, operators, indices, n4,
    )
    assignment, overlaps = _track(
        reference['vectors'], vectors, indices, n4, NSTATES,
    )
    values, vectors, tensor, matrix, _ = _apply_state_assignment(
        values, vectors, tensor, matrix, assignment,
    )
    return _case_analysis(
        label, indices, values, matrix, assignment, overlaps,
        reference, parent_k,
    )


def _reference_endpoint_case(label, reference, parent_k):
    assignment = np.arange(NSTATES, dtype=int)
    overlaps = np.ones(NSTATES)
    return _case_analysis(
        label, reference['indices'], reference['energies'],
        reference['matrix'], assignment, overlaps, reference, parent_k,
    )


def _validate_n51_metadata(metadata):
    if metadata.get('physical_parameters') != PHYSICAL_PARAMETERS:
        raise ValueError('N=51 checkpoint physical parameters do not match.')
    if metadata.get('spatial_cutoffs') != N51_SPATIAL_CUTOFFS:
        raise ValueError('N=51 checkpoint spatial settings do not match.')
    solver = metadata.get('solver', {})
    # N=51 is an independently pinned k=180 reference.  Its authenticated
    # solver metadata must not be compared with the selected N=71 parent k.
    for name, expected in N51_SOLVER_SETTINGS.items():
        observed_name = 'permc_spec' if name == 'permc_spec' else name
        if solver.get(observed_name) != expected:
            raise ValueError(
                f'N=51 checkpoint solver mismatch for {name}: '
                f'{solver.get(observed_name)!r} != {expected!r}.'
            )
    expected_dimension = (
        (2 * N51_SPATIAL_CUTOFFS['n1max'] + 1)
        * N51_SPATIAL_CUTOFFS['N2']
        * N51_SPATIAL_CUTOFFS['N3']
    )
    if solver.get('matrix_dimension') != expected_dimension:
        raise ValueError('N=51 checkpoint matrix dimension does not match.')
    if solver.get('path') != 'explicit_shift_invert':
        raise ValueError('N=51 checkpoint did not use explicit shift-invert.')
    if (solver.get('orthogonality_residual_inf') is None
            or solver['orthogonality_residual_inf']
            > MAX_ORTHOGONALITY_RESIDUAL):
        raise ValueError('N=51 checkpoint orthogonality validation failed.')


def load_verified_n51_checkpoint(checkpoint_dir=DEFAULT_N51_CHECKPOINT):
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / 'checkpoint.json'
    stage_path = checkpoint_dir / 'stage1_eigensystem.npz'
    operators_path = checkpoint_dir / 'projected_operators.npz'
    paths = {
        'checkpoint.json': metadata_path,
        'stage1_eigensystem.npz': stage_path,
        'projected_operators.npz': operators_path,
    }
    observed_hashes = {name: _sha256_file(path)
                       for name, path in paths.items()}
    if observed_hashes != N51_ORACLE_SHA256:
        raise ValueError(
            'N=51 checkpoint bytes do not match the verified asymmetric '
            'reference.'
        )
    metadata = json.loads(metadata_path.read_text())
    _validate_n51_metadata(metadata)
    with np.load(stage_path) as stage:
        values = np.asarray(stage['eigenvalues_GHz'])
        vectors = np.asarray(stage['eigenvectors'])
        residuals = np.asarray(stage['eigenpair_residuals'])
    with np.load(operators_path) as saved:
        operators = {name: np.asarray(saved[name])
                     for name in EXPECTED_OPERATOR_NAMES}
    expected_dimension = (
        (2 * N51_SPATIAL_CUTOFFS['n1max'] + 1)
        * N51_SPATIAL_CUTOFFS['N2']
        * N51_SPATIAL_CUTOFFS['N3']
    )
    n51_parent_k = N51_SOLVER_SETTINGS['k']
    if (values.shape != (n51_parent_k,)
            or vectors.shape != (expected_dimension, n51_parent_k)):
        raise ValueError('N=51 checkpoint has incompatible Stage-1 shapes.')
    if residuals.shape != (n51_parent_k,):
        raise ValueError('N=51 checkpoint residual vector has wrong shape.')
    if any(operators[name].shape != (n51_parent_k, n51_parent_k)
           for name in EXPECTED_OPERATOR_NAMES):
        raise ValueError('N=51 projected operators have incompatible shapes.')
    if not all(np.all(np.isfinite(array)) for array in (
            values, vectors, residuals, *operators.values())):
        raise ValueError('N=51 checkpoint contains non-finite values.')
    if float(np.max(residuals)) > MAX_EIGENPAIR_RESIDUAL:
        raise ValueError('N=51 checkpoint eigenpair residual validation failed.')
    return {
        'metadata': metadata,
        'values': values,
        'vectors': vectors,
        'residuals': residuals,
        'operators': operators,
        'file_sha256': {
            **observed_hashes,
        },
    }


def _linear_interpolation_matrix(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError('Interpolation grids must be one-dimensional.')
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError('Interpolation grids must be strictly increasing.')
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError('Target grid lies outside the source domain.')
    matrix = np.zeros((len(target), len(source)))
    upper = np.searchsorted(source, target, side='right')
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    fraction = (target - source[lower]) / (source[upper] - source[lower])
    rows = np.arange(len(target))
    matrix[rows, lower] = 1.0 - fraction
    matrix[rows, upper] += fraction
    return matrix


def interpolate_raw_states(raw_states, source_spatial, target_spatial):
    """Map normalized grid coefficients while preserving quadrature scale.

    Stage-1 vectors flatten ``(theta3, theta2, n1)`` in C order.  Dividing by
    the source quadrature weight, interpolating the wavefunction density, and
    applying the target weight gives vectors in the target Euclidean basis.
    """
    source_spatial = dict(source_spatial)
    target_spatial = dict(target_spatial)
    if source_spatial['n1max'] != target_spatial['n1max']:
        raise ValueError('Cross-grid tracking requires the same n1max.')
    if source_spatial['N4'] != target_spatial['N4']:
        raise ValueError('Cross-grid tracking requires the same N4.')
    if (source_spatial['L2'] != target_spatial['L2']
            or source_spatial['L3'] != target_spatial['L3']):
        raise ValueError('Cross-grid tracking requires identical domains.')

    n1_count = 2 * source_spatial['n1max'] + 1
    nstates = raw_states.shape[-1]
    expected_source = (
        source_spatial['N3'] * source_spatial['N2'] * n1_count
    )
    if raw_states.shape != (
            expected_source, source_spatial['N4'], nstates):
        raise ValueError('Raw-state array has an incompatible source shape.')

    x2_source = np.linspace(
        -source_spatial['L2'], source_spatial['L2'],
        source_spatial['N2'],
    )
    x3_source = np.linspace(
        -source_spatial['L3'], source_spatial['L3'],
        source_spatial['N3'],
    )
    x2_target = np.linspace(
        -target_spatial['L2'], target_spatial['L2'],
        target_spatial['N2'],
    )
    x3_target = np.linspace(
        -target_spatial['L3'], target_spatial['L3'],
        target_spatial['N3'],
    )
    interpolation2 = _linear_interpolation_matrix(x2_source, x2_target)
    interpolation3 = _linear_interpolation_matrix(x3_source, x3_target)
    dx_source = (x2_source[1] - x2_source[0]) * (
        x3_source[1] - x3_source[0]
    )
    dx_target = (x2_target[1] - x2_target[0]) * (
        x3_target[1] - x3_target[0]
    )

    shaped = raw_states.reshape(
        source_spatial['N3'], source_spatial['N2'], n1_count,
        source_spatial['N4'], nstates,
    ) / np.sqrt(dx_source)
    along3 = np.tensordot(interpolation3, shaped, axes=(1, 0))
    interpolated = np.einsum(
        'Bb,Ab...->AB...', interpolation2, along3, optimize=True,
    ) * np.sqrt(dx_target)
    flattened = interpolated.reshape(
        target_spatial['N3'] * target_spatial['N2'] * n1_count,
        target_spatial['N4'], nstates,
    )
    norms = np.linalg.norm(flattened.reshape(-1, nstates), axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError('Interpolated state norm is invalid.')
    return flattened / norms[None, None, :]


def reconstruct_raw_final_states(stage1_vectors, stage2_tensor):
    if stage1_vectors.shape[1] != stage2_tensor.shape[0]:
        raise ValueError('Stage-1 and Stage-2 dimensions are incompatible.')
    return np.einsum(
        'da,ams->dms', stage1_vectors, stage2_tensor, optimize=True,
    )


def track_cross_grid_states(reference_raw, candidate_raw):
    """Assign candidate states to reference states using physical overlaps."""
    if reference_raw.shape != candidate_raw.shape:
        raise ValueError('Cross-grid raw-state arrays must have equal shapes.')
    nstates = reference_raw.shape[-1]
    reference = reference_raw.reshape(-1, nstates)
    candidate = candidate_raw.reshape(-1, nstates)
    reference = reference / np.linalg.norm(reference, axis=0)[None, :]
    candidate = candidate / np.linalg.norm(candidate, axis=0)[None, :]
    overlap = candidate.conj().T @ reference
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    assignment = np.empty(nstates, dtype=int)
    assignment[columns] = rows
    matched = np.abs(overlap[assignment, np.arange(nstates)])
    return assignment, matched, overlap


def assignment_confidence_diagnostics(
        overlap_matrix, assignment,
        minimum_assigned_overlap=MIN_ASSIGNED_OVERLAP_FOR_CLAIM,
        maximum_competitor_ratio=MAX_COMPETITOR_RATIO_FOR_CLAIM,
        minimum_overlap_margin=MIN_OVERLAP_MARGIN_FOR_CLAIM):
    """Describe uniqueness of every fixed Hungarian assignment.

    Thresholds only emit warnings and gate state-specific claims.  They never
    change the assignment or relabel an eigenvector.
    """
    overlap = np.abs(np.asarray(overlap_matrix))
    assignment = np.asarray(assignment, dtype=int)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError('Overlap matrix must be square.')
    nstates = overlap.shape[0]
    if assignment.shape != (nstates,) or sorted(assignment.tolist()) != list(
            range(nstates)):
        raise ValueError('Assignment must be a permutation of candidate rows.')
    candidate_to_reference = np.empty(nstates, dtype=int)
    candidate_to_reference[assignment] = np.arange(nstates)

    rows = []
    for reference_state in range(nstates):
        assigned_candidate = int(assignment[reference_state])
        assigned_overlap = float(overlap[
            assigned_candidate, reference_state
        ])
        if nstates == 1:
            competing_candidate = None
            competing_reference = None
            competing_overlap = 0.0
        else:
            competitors = overlap[:, reference_state].copy()
            competitors[assigned_candidate] = -np.inf
            competing_candidate = int(np.argmax(competitors))
            competing_reference = int(
                candidate_to_reference[competing_candidate]
            )
            competing_overlap = float(competitors[competing_candidate])
        margin = assigned_overlap - competing_overlap
        ratio = (
            competing_overlap / assigned_overlap
            if assigned_overlap > 1e-15 else None
        )
        low_overlap_warning = assigned_overlap < minimum_assigned_overlap
        ratio_warning = ratio is None or ratio >= maximum_competitor_ratio
        margin_warning = margin <= minimum_overlap_margin
        # Cluster formation uses the relative competition test.  A small
        # absolute margin caused only by globally poor overlap is deferred as a
        # physical mismatch, rather than mislabeled as an internal rotation.
        competition_warning = ratio_warning
        rows.append({
            'reference_state': reference_state,
            'assigned_candidate_state': assigned_candidate,
            'assigned_overlap': assigned_overlap,
            'best_competing_candidate_state': competing_candidate,
            'best_competing_reference_state': competing_reference,
            'best_competing_overlap': competing_overlap,
            'overlap_margin': margin,
            'competitor_to_assigned_ratio': ratio,
            'low_assigned_overlap_warning': bool(low_overlap_warning),
            'competitor_ratio_warning': bool(ratio_warning),
            'overlap_margin_warning': bool(margin_warning),
            'competition_warning': bool(competition_warning),
            'state_specific_claim_safe': bool(
                not low_overlap_warning
                and not ratio_warning
                and not margin_warning
            ),
        })
    return rows


def subspace_overlap_diagnostics(overlap_matrix, reference_states,
                                 candidate_states):
    """Return principal-overlap diagnostics for two equal-size subspaces."""
    overlap = np.asarray(overlap_matrix)
    reference_states = np.asarray(reference_states, dtype=int)
    candidate_states = np.asarray(candidate_states, dtype=int)
    if (reference_states.ndim != 1 or candidate_states.ndim != 1
            or len(reference_states) != len(candidate_states)
            or not len(reference_states)):
        raise ValueError('Subspaces must have equal positive dimensions.')
    block = overlap[np.ix_(candidate_states, reference_states)]
    singular_values = np.linalg.svd(block, compute_uv=False)
    clipped = np.clip(singular_values, 0.0, 1.0)
    principal_angles = np.degrees(np.arccos(clipped))
    return {
        'dimension': int(len(reference_states)),
        'reference_states': reference_states.tolist(),
        'candidate_states': candidate_states.tolist(),
        'overlap_block_real': np.real(block).tolist(),
        'overlap_block_imag': np.imag(block).tolist(),
        'overlap_block_abs': np.abs(block).tolist(),
        'singular_values': singular_values.tolist(),
        'minimum_singular_value': float(np.min(singular_values)),
        'principal_angles_degrees': principal_angles.tolist(),
        'maximum_principal_angle_degrees': float(np.max(principal_angles)),
    }


def ambiguous_subspace_clusters(overlap_matrix, assignment,
                                confidence_rows):
    """Build transparent competition-connected state clusters."""
    assignment = np.asarray(assignment, dtype=int)
    graph = {state: set() for state in range(len(assignment))}
    for row in confidence_rows:
        competitor = row['best_competing_reference_state']
        if (row['competition_warning'] and competitor is not None
                and competitor != row['reference_state']):
            state = row['reference_state']
            graph[state].add(competitor)
            graph[competitor].add(state)

    clusters = []
    visited = set()
    for state in range(len(assignment)):
        if state in visited or not graph[state]:
            continue
        pending = [state]
        component = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(graph[current] - component)
        visited.update(component)
        labels = sorted(component)
        candidates = assignment[labels]
        diagnostic = subspace_overlap_diagnostics(
            overlap_matrix, labels, candidates,
        )
        diagnostic['cluster_id'] = len(clusters)
        diagnostic['formation_rule'] = (
            'connected reference states with competition_warning=true'
        )
        clusters.append(diagnostic)
    return clusters


def _grid_phi_subspace_metrics(matrix, labels):
    matrix = np.asarray(matrix)
    labels = np.asarray(labels, dtype=int)
    complement = np.asarray([
        state for state in range(matrix.shape[0]) if state not in set(labels)
    ], dtype=int)
    internal = matrix[np.ix_(labels, labels)]
    to_complement = matrix[np.ix_(labels, complement)]
    logical = {}
    for state in (0, 1):
        if state not in labels:
            logical[str(state)] = float(np.linalg.norm(matrix[state, labels]))
    return {
        'logical_to_subspace_l2_norm': logical,
        'internal_block_frobenius_norm': float(np.linalg.norm(internal)),
        'internal_block_singular_values': np.linalg.svd(
            internal, compute_uv=False,
        ).tolist(),
        'subspace_to_complement_frobenius_norm': float(
            np.linalg.norm(to_complement)
        ),
    }


def grid_phi_subspace_diagnostics(reference_matrix, candidate_matrix,
                                  clusters):
    """Compare basis-invariant grid_phi strengths for ambiguous clusters."""
    rows = []
    for cluster in clusters:
        labels = cluster['reference_states']
        reference = _grid_phi_subspace_metrics(reference_matrix, labels)
        candidate = _grid_phi_subspace_metrics(candidate_matrix, labels)
        logical_changes = {
            logical: abs(
                candidate['logical_to_subspace_l2_norm'][logical]
                - value
            )
            for logical, value in reference[
                'logical_to_subspace_l2_norm'
            ].items()
        }
        rows.append({
            'cluster_id': cluster['cluster_id'],
            'states': labels,
            'reference': reference,
            'candidate': candidate,
            'absolute_changes': {
                'logical_to_subspace_l2_norm': logical_changes,
                'internal_block_frobenius_norm': abs(
                    candidate['internal_block_frobenius_norm']
                    - reference['internal_block_frobenius_norm']
                ),
                'subspace_to_complement_frobenius_norm': abs(
                    candidate['subspace_to_complement_frobenius_norm']
                    - reference['subspace_to_complement_frobenius_norm']
                ),
            },
        })
    return rows


def _path_claim_status(path, confidence_rows, clusters):
    confidence = {row['reference_state']: row for row in confidence_rows}
    path_states = set(path)
    ambiguous_clusters = [
        cluster['cluster_id'] for cluster in clusters
        if path_states.intersection(cluster['reference_states'])
    ]
    unsafe_states = sorted(
        state for state in path_states
        if not confidence[state]['state_specific_claim_safe']
    )
    if ambiguous_clusters:
        status = 'ambiguous'
    elif unsafe_states:
        status = 'deferred'
    else:
        status = 'resolved'
    return {
        'claim_status': status,
        'claim_unsafe_states': unsafe_states,
        'ambiguous_cluster_ids': ambiguous_clusters,
    }


def annotate_pathway_claims(pathways, confidence_rows, clusters):
    annotated = []
    for pathway in pathways:
        row = dict(pathway)
        row.update(_path_claim_status(
            row['path'], confidence_rows, clusters,
        ))
        annotated.append(row)
    return annotated


def _cross_grid_analysis(n51, n71_vectors, n51_reference, n71_reference,
                         n71_spatial=SPATIAL_CUTOFFS):
    raw51 = reconstruct_raw_final_states(
        n51['vectors'], n51_reference['tensor'],
    )
    mapped51 = interpolate_raw_states(
        raw51, N51_SPATIAL_CUTOFFS, n71_spatial,
    )
    raw71 = reconstruct_raw_final_states(
        n71_vectors, n71_reference['tensor'],
    )
    assignment, overlaps, overlap_matrix = track_cross_grid_states(
        mapped51, raw71,
    )
    confidence = assignment_confidence_diagnostics(
        overlap_matrix, assignment,
    )
    clusters = ambiguous_subspace_clusters(
        overlap_matrix, assignment, confidence,
    )
    values71 = n71_reference['energies'][assignment]
    matrix71 = n71_reference['matrix'][np.ix_(assignment, assignment)]
    transitions71 = values71 - values71[0]
    transition_shifts = 1e3 * (
        transitions71[1:] - n51_reference['transitions'][1:]
    )

    paths51 = n51_reference['pathways']
    raw_paths71 = _three_step_paths(matrix71, NSTATES)
    paths71 = annotate_pathway_claims(raw_paths71, confidence, clusters)
    leading51 = paths51[0]['path']
    leading_claim = _path_claim_status(
        leading51, confidence, clusters,
    )
    raw_leading_path_match = raw_paths71[0]['path'] == leading51
    if leading_claim['claim_status'] == 'resolved':
        leading_conclusion = (
            'survives' if raw_leading_path_match else 'does_not_survive'
        )
    else:
        leading_conclusion = 'deferred'
    primary_rows = _coupling_rows(
        n51_reference['matrix'], matrix71, _path_pairs(leading51),
    )
    logical_rows = _coupling_rows(
        n51_reference['matrix'], matrix71,
        _dominant_logical_pairs(n51_reference['matrix']),
    )
    absolute_matrix_change = np.abs(
        np.abs(matrix71) - np.abs(n51_reference['matrix'])
    )
    relative_matrix_change = []
    reference_abs = np.abs(n51_reference['matrix'])
    for row in range(NSTATES):
        relative_matrix_change.append([
            (float(absolute_matrix_change[row, column]
                   / reference_abs[row, column])
             if reference_abs[row, column] > 1e-12 else None)
            for column in range(NSTATES)
        ])
    return {
        'tracking_method': (
            'N=51 full Stage-2 states reconstructed on the physical '
            '(theta3,theta2,n1,N4) basis, quadrature-aware bilinear '
            'interpolation to N=71, then Hungarian maximum-overlap assignment'
        ),
        'n71_energy_row_for_n51_state': assignment.tolist(),
        'state_reordering_detected': bool(
            np.any(assignment != np.arange(NSTATES))
        ),
        'matched_state_overlaps': overlaps.tolist(),
        'minimum_matched_state_overlap': float(np.min(overlaps)),
        'absolute_overlap_matrix': np.abs(overlap_matrix).tolist(),
        'complex_overlap_matrix': {
            'real': np.real(overlap_matrix).tolist(),
            'imag': np.imag(overlap_matrix).tolist(),
        },
        'assignment_confidence_thresholds': {
            'minimum_assigned_overlap_for_claim': (
                MIN_ASSIGNED_OVERLAP_FOR_CLAIM
            ),
            'maximum_competitor_to_assigned_ratio_for_claim': (
                MAX_COMPETITOR_RATIO_FOR_CLAIM
            ),
            'minimum_overlap_margin_for_claim': (
                MIN_OVERLAP_MARGIN_FOR_CLAIM
            ),
            'semantics': (
                'warning and state-specific claim gating only; thresholds do '
                'not alter assignment or eigenstate labels'
            ),
        },
        'individual_assignment_confidence': confidence,
        'ambiguous_subspace_clusters': clusters,
        'n51_transitions_GHz': n51_reference['transitions'][1:].tolist(),
        'tracked_n71_transitions_GHz': transitions71[1:].tolist(),
        'transition_shifts_MHz': transition_shifts.tolist(),
        'maximum_absolute_transition_shift_MHz': float(
            np.max(np.abs(transition_shifts))
        ),
        'n51_leading_path': leading51,
        'n71_ranked_paths_in_n51_tracked_labels': paths71,
        'n51_leading_path_raw_assigned_basis_rank_match': bool(
            raw_leading_path_match
        ),
        'n51_leading_path_claim_status': leading_claim,
        'n51_leading_path_state_specific_conclusion': leading_conclusion,
        'n51_leading_path_grid_phi_changes': primary_rows,
        'dominant_logical_leakage_grid_phi_changes': logical_rows,
        'ambiguous_subspace_grid_phi_diagnostics': (
            grid_phi_subspace_diagnostics(
                n51_reference['matrix'], matrix71, clusters,
            )
        ),
        'n51_grid_phi_abs': reference_abs.tolist(),
        'tracked_n71_grid_phi_abs': np.abs(matrix71).tolist(),
        'grid_phi_absolute_change': absolute_matrix_change.tolist(),
        'grid_phi_relative_change': relative_matrix_change,
    }


def analyze_same_solve(ECm, K, n71_values, n71_vectors, n71_operators,
                       n51_checkpoint=DEFAULT_N51_CHECKPOINT,
                       profile_name=DEFAULT_PROFILE_NAME):
    profile = profile_definition(profile_name)
    configuration = profile['configuration']
    spatial_cutoffs = configuration['spatial_cutoffs']
    parent_k = configuration['solver_settings']['k']
    n4 = spatial_cutoffs['N4']
    reference71 = _full_stage2(
        ECm, K, n71_values, n71_operators, n4,
    )
    candidates = {}
    for label, dimension in profile['retained_cases']:
        indices = retained_indices(dimension, profile_name)
        if dimension == parent_k:
            candidates[label] = _reference_endpoint_case(
                label, reference71, parent_k,
            )
        else:
            candidates[label] = _retained_case(
                label, indices, ECm, K, n71_values, n71_operators,
                reference71, n4, parent_k,
            )

    n51 = load_verified_n51_checkpoint(n51_checkpoint)
    reference51 = _full_stage2(
        ECm, K, n51['values'], n51['operators'],
        N51_SPATIAL_CUTOFFS['N4'],
    )
    spatial = _cross_grid_analysis(
        n51, n71_vectors, reference51, reference71, spatial_cutoffs,
    )
    result = {
        'profile': profile_name,
        'configuration_sha256': CONFIGURATION_SHA256_BY_PROFILE[
            profile_name
        ],
        'control_operator': 'grid_phi = theta3 + theta2/2',
        'reference_endpoint_semantics': (
            f'k{parent_k} is the larger reference endpoint for this '
            'same-parent comparison; it is not ground truth or an oracle'
        ),
        'stage1_parent_reuse': (
            f'parent k{parent_k} and retained '
            + ', '.join(f'k{dimension}' for _, dimension in profile[
                'retained_cases'
            ])
            + f' Stage-2 analyses all use the same single N=71 Stage-1 '
            f'k{parent_k} solve'
        ),
        f'n71_reference_k{parent_k}': {
            'energies_GHz': reference71['energies'].tolist(),
            'transitions_GHz': reference71['transitions'][1:].tolist(),
            'grid_phi_abs': np.abs(reference71['matrix']).tolist(),
            'pathways': reference71['pathways'],
            'thomas_observables': _thomas_observables(
                reference71['transitions'], reference71['matrix'],
                np.ones(NSTATES), reference71, parent_k,
            ),
            'logical_to_excited': {
                str(logical): sorted([
                    {
                        'state': state,
                        'matrix_element': float(
                            abs(reference71['matrix'][logical, state])
                        ),
                    }
                    for state in range(2, NSTATES)
                ], key=lambda row: row['matrix_element'], reverse=True)
                for logical in (0, 1)
            },
        },
        'retained_basis_cases': candidates,
        'n51_checkpoint_provenance': {
            'directory_name': Path(n51_checkpoint).name,
            'file_sha256': n51['file_sha256'],
            'stage1_solver_diagnostics': n51['metadata']['solver'],
        },
        'n51_to_n71_spatial_comparison': spatial,
        'spatial_convergence_classification': None,
        'classification_status': (
            'deferred for scientific review; this runner records data only'
        ),
    }
    if profile_name == DEFAULT_PROFILE_NAME:
        result['stage1_oracle_reuse'] = (
            'k180, k170, k172, and k175 Stage-2 analyses all use the same '
            'single N=71 Stage-1 k180 solve'
        )
    return result


def _completion_payload(checkpoint_sha256, configuration_sha256,
                        profile_name=None):
    payload = {
        'status': 'complete',
        'configuration_sha256': configuration_sha256,
        'checkpoint_sha256': checkpoint_sha256,
        'completed_utc': _utc_now(),
    }
    if profile_name is not None:
        payload['profile'] = profile_name
    return payload


def _validate_checkpoint_payload(checkpoint_dir, metadata,
                                 profile_name=None):
    checkpoint_dir = Path(checkpoint_dir)
    if metadata.get('status') != 'complete':
        raise ValueError('Checkpoint status is not complete.')
    declared_profile = metadata.get('profile', profile_name)
    resolved_profile = _profile_for_checkpoint(
        metadata.get('configuration'),
        metadata.get('configuration_sha256'),
        declared_profile,
    )
    if profile_name is not None and resolved_profile != profile_name:
        raise ValueError('Checkpoint profile does not match the request.')
    validate_frozen_configuration(
        metadata.get('configuration'), resolved_profile,
    )
    required_files = {
        'stage1_eigensystem.npz',
        'projected_operators.npz',
        'analysis.json',
    }
    file_hashes = metadata.get('file_sha256', {})
    if set(file_hashes) != required_files:
        raise ValueError('Checkpoint file-hash manifest is incomplete.')
    for name, expected_hash in file_hashes.items():
        path = checkpoint_dir / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f'Checkpoint file hash mismatch: {name}.')
    return resolved_profile


def publish_checkpoint_directory(incomplete_dir, final_dir):
    """Atomically expose validated files, then write COMPLETE last."""
    incomplete_dir = Path(incomplete_dir)
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {final_dir}.')
    if not incomplete_dir.is_dir():
        raise FileNotFoundError(incomplete_dir)
    checkpoint_path = incomplete_dir / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    profile_name = _validate_checkpoint_payload(incomplete_dir, checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    os.replace(incomplete_dir, final_dir)
    _atomic_write_json(
        final_dir / 'COMPLETE', _completion_payload(
            checkpoint_sha256, checkpoint['configuration_sha256'],
            profile_name if checkpoint.get('profile') is not None else None,
        ),
    )
    return final_dir


def validate_complete_checkpoint_directory(checkpoint_dir):
    """Fail closed unless metadata and the last-written marker both match."""
    checkpoint_dir = Path(checkpoint_dir)
    marker_path = checkpoint_dir / 'COMPLETE'
    metadata_path = checkpoint_dir / 'checkpoint.json'
    if not marker_path.is_file() or not metadata_path.is_file():
        raise ValueError('Checkpoint is incomplete: metadata/COMPLETE missing.')
    marker = json.loads(marker_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    profile_name = _profile_for_checkpoint(
        metadata.get('configuration'),
        metadata.get('configuration_sha256'),
        metadata.get('profile'),
    )
    expected_fingerprint = CONFIGURATION_SHA256_BY_PROFILE[profile_name]
    for document, label in ((marker, 'COMPLETE'), (metadata, 'checkpoint')):
        if document.get('status') != 'complete':
            raise ValueError(f'{label} status is not complete.')
        if document.get('configuration_sha256') != expected_fingerprint:
            raise ValueError(f'{label} configuration fingerprint mismatch.')
        if (document.get('profile') is not None
                and document.get('profile') != profile_name):
            raise ValueError(f'{label} profile mismatch.')
    if marker.get('checkpoint_sha256') != _sha256_file(metadata_path):
        raise ValueError('COMPLETE does not authenticate checkpoint.json.')
    _validate_checkpoint_payload(checkpoint_dir, metadata, profile_name)
    return metadata


def execute(output_root=DEFAULT_OUTPUT_ROOT,
            n51_checkpoint=DEFAULT_N51_CHECKPOINT,
            profile_name=DEFAULT_PROFILE_NAME):
    """Run exactly one frozen N=71 Stage-1 solve and publish if valid."""
    profile = profile_definition(profile_name)
    configuration = profile['configuration']
    configuration_sha256 = CONFIGURATION_SHA256_BY_PROFILE[profile_name]
    validate_thread_environment()
    validate_frozen_configuration(configuration, profile_name)
    output_root = Path(output_root).resolve()
    n51_checkpoint = Path(n51_checkpoint).resolve()
    # Authenticate the pinned N=51 comparison before allocating the N=71
    # matrix.  Its k=180 solver metadata is independent of the N=71 parent k.
    load_verified_n51_checkpoint(n51_checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / profile['final_directory_name']
    if final_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {final_dir}.')
    incomplete_dir = output_root / (
        f'.{profile["final_directory_name"]}.incomplete-{uuid.uuid4().hex}'
    )
    incomplete_dir.mkdir()
    started = time.perf_counter()
    run_state = {
        'status': 'incomplete',
        'started_utc': _utc_now(),
        'profile': profile_name,
        'configuration_sha256': configuration_sha256,
        'configuration': deepcopy(configuration),
        'provenance': collect_runtime_provenance(),
    }
    _atomic_write_json(incomplete_dir / 'run_state.json', run_state)

    try:
        (ECm, K, values, vectors, operators, residuals,
         diagnostics) = _solve_stage1_once(profile_name)
        np.savez(
            incomplete_dir / 'stage1_eigensystem.npz',
            eigenvalues_GHz=values,
            eigenvectors=vectors,
            eigenpair_residuals=residuals,
        )
        np.savez(
            incomplete_dir / 'projected_operators.npz', **operators,
        )
        analysis = analyze_same_solve(
            ECm, K, values, vectors, operators, n51_checkpoint,
            profile_name,
        )
        _atomic_write_json(incomplete_dir / 'analysis.json', analysis)

        hashed_names = (
            'stage1_eigensystem.npz',
            'projected_operators.npz',
            'analysis.json',
        )
        file_hashes = {
            name: _sha256_file(incomplete_dir / name)
            for name in hashed_names
        }
        total_runtime = time.perf_counter() - started
        checkpoint = {
            'status': 'complete',
            'validated_utc': _utc_now(),
            'profile': profile_name,
            'configuration': deepcopy(configuration),
            'configuration_sha256': configuration_sha256,
            'provenance': run_state['provenance'],
            'generator': (
                'Simulations/Rabi_3Photon/'
                'run_four_mode_spatial71_reference.py'
            ),
            'stage1_solver_diagnostics': diagnostics,
            'total_runtime_seconds': total_runtime,
            'file_sha256': file_hashes,
            'validation': {
                'finite_eigensystem': True,
                'expected_dimensions': True,
                'frozen_configuration_match': True,
                'maximum_eigenpair_residual': diagnostics[
                    'max_eigenpair_residual'
                ],
                'median_eigenpair_residual': diagnostics[
                    'median_eigenpair_residual'
                ],
                'orthogonality_residual_inf': diagnostics[
                    'orthogonality_residual_inf'
                ],
            },
        }
        _atomic_write_json(incomplete_dir / 'checkpoint.json', checkpoint)
        run_state.update({
            'status': 'complete',
            'validated_utc': checkpoint['validated_utc'],
            'total_runtime_seconds': total_runtime,
            'completed_utc': _utc_now(),
        })
        _atomic_write_json(incomplete_dir / 'run_state.json', run_state)
        final_dir = publish_checkpoint_directory(incomplete_dir, final_dir)
        validate_complete_checkpoint_directory(final_dir)
        return final_dir
    except BaseException as error:
        failure_dir = final_dir if final_dir.exists() else incomplete_dir
        if failure_dir.exists():
            run_state.update({
                'status': 'failed',
                'failed_utc': _utc_now(),
                'error_type': type(error).__name__,
                'error': str(error),
            })
            _atomic_write_json(failure_dir / 'run_state.json', run_state)
            _atomic_write_text(
                failure_dir / 'failure.txt', traceback.format_exc(),
            )
            if (failure_dir / 'COMPLETE').exists():
                (failure_dir / 'COMPLETE').unlink()
        raise


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--dry-run', action='store_true',
        help='Print frozen metadata only; do not assemble or solve.',
    )
    mode.add_argument(
        '--execute', action='store_true',
        help='Run the selected frozen N=71 parent calculation.',
    )
    parser.add_argument(
        '--profile', choices=tuple(PROFILE_DEFINITIONS),
        default=DEFAULT_PROFILE_NAME,
        help='Frozen named N=71 profile (default: n71-k180).',
    )
    parser.add_argument(
        '--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
        help='Parent directory for the fixed final checkpoint name.',
    )
    parser.add_argument(
        '--n51-checkpoint', type=Path, default=DEFAULT_N51_CHECKPOINT,
        help='Verified asymmetric N=51 checkpoint used for comparison.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse_arguments(argv)
    if arguments.dry_run:
        print(json.dumps(
            dry_run_manifest(
                arguments.output_root, arguments.n51_checkpoint,
                arguments.profile,
            ),
            indent=2,
            allow_nan=False,
        ))
        return 0
    final_dir = execute(
        arguments.output_root, arguments.n51_checkpoint, arguments.profile,
    )
    print(json.dumps({
        'status': 'complete',
        'checkpoint': str(final_dir),
        'profile': arguments.profile,
        'configuration_sha256': CONFIGURATION_SHA256_BY_PROFILE[
            arguments.profile
        ],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
