"""Validation tests for the handwritten four-mode netlist-frame gridium.

The default suite contains fast, independent checks of the circuit coefficients,
hierarchical diagonalization, and external-flux derivatives. Production spectrum
convergence is enabled with ``GRID_NETLIST_RUN_HAND_CONVERGENCE=1``. Production
drive-operator convergence is enabled with ``GRID_NETLIST_RUN_DRIVE_VALIDATION=1``.

Expensive results are cached under ``GRID_NETLIST_OUTPUT_DIR`` (default:
``/private/tmp/four_mode_gridium_netlist_validation``). Set ``GRID_NETLIST_FORCE=1``
to discard a matching cached result.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Circuit_Objs.qchard_gridium_netlist as gridium_netlist
from Circuit_Objs.qchard_gridium_netlist import (
    Gridium4Mode,
    _assemble_nonlinear_sector,
    _coeffs,
    _stage1_eigensolve,
)


def _flag(name):
    return os.environ.get(name, '').lower() in {'1', 'true', 'yes'}


RUN_HAND = _flag('GRID_NETLIST_RUN_HAND_CONVERGENCE')
RUN_DRIVE = _flag('GRID_NETLIST_RUN_DRIVE_VALIDATION')
FORCE = _flag('GRID_NETLIST_FORCE')

SELF_DOUBLET_TOL_MHZ = float(os.environ.get('GRID_NETLIST_DOUBLET_TOL_MHZ', '0.5'))
SELF_TRANSITION_TOL_MHZ = float(os.environ.get('GRID_NETLIST_TRANSITION_TOL_MHZ', '5.0'))
DRIVE_SELF_RTOL = float(os.environ.get('GRID_NETLIST_DRIVE_SELF_RTOL', '0.03'))
DRIVE_SELF_ATOL_MHZ = float(os.environ.get('GRID_NETLIST_DRIVE_SELF_ATOL_MHZ', '2.0'))

REGIME_A = dict(EJ=5.0, EC=0.5, EL=1.0, ELK=1.0, EJS=4.0, ECS=8.0)
FOUR_MODE_CAPS = dict(eC=5.5, eP=10.0)
TRANSFORMATION = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
])

WORST_POINT = dict(theta_ext=0.0, phi_ext=0.0, ng=0.0)
PROTECTION_POINT = dict(theta_ext=np.pi, phi_ext=0.0, ng=0.0)
FINAL_HAND_CUTOFFS = dict(
    n1max=5, N2=111, L2=11.0, N3=101, L3=14.0, N4=8, nkeep=320,
)
FINAL_NKEEP_LADDER = (320, 360)
EXTENDED_EVALS_COUNT = 11

#determines where the results are cached 
def _output_dir():
    default = '/private/tmp/four_mode_gridium_netlist_validation'
    return Path(os.environ.get('GRID_NETLIST_OUTPUT_DIR', default))

#converts numpy and python objects into json data type
def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _cached(kind, inputs, compute):
    payload = _plain(inputs)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    path = _output_dir() / ('%s_%s.json' % (kind, digest))
    if path.exists() and not FORCE:
        return json.loads(path.read_text())

    result = _plain(compute())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'inputs': payload, 'result': result}, indent=2) + '\n')
    return {'inputs': payload, 'result': result}

#computs the absolute changes in transition frequencies 
def _transition_delta_mhz(first, second):
    return 1e3 * np.abs(np.asarray(second, dtype=float) - np.asarray(first, dtype=float))

#checks whether transition frequcnies remains stable when cutoff is increased 
#convergence requirement: <= 5MHz
def _assert_transition_convergence(label, baseline, refined, protected_doublet=True):
    previous = np.asarray(baseline['transitions_GHz'])
    current = np.asarray(refined['transitions_GHz'])
    delta_mhz = _transition_delta_mhz(previous, current)
    print('%s transition convergence:' % label)
    print('  baseline=%s GHz' % np.array2string(previous, precision=9))
    print('  refined =%s GHz' % np.array2string(current, precision=9))
    print('  delta   =%s MHz' % np.array2string(delta_mhz, precision=3))
    if protected_doublet:
        assert delta_mhz[0] <= SELF_DOUBLET_TOL_MHZ, (
            '%s doublet splitting changed by %.3f MHz.' % (label, delta_mhz[0])
        )
        upper_delta = delta_mhz[1:]
    else:
        upper_delta = delta_mhz
    assert np.max(upper_delta) <= SELF_TRANSITION_TOL_MHZ, (
        '%s transitions changed by up to %.3f MHz.' % (label, np.max(upper_delta))
    )

#independently derives the hamiltonian coefficicents matrices directly from branch connectivity 
#used as reference 
def _branch_reference_coefficients(eps_J, eps_LK):

    def outer(vector):
        vector = np.asarray(vector, dtype=float)
        return np.outer(vector, vector)

    Cnode = (
        outer([1, 0, 0, 0]) / REGIME_A['EC']
        + outer([0, 1, 0, 0]) / REGIME_A['EC']
        + outer([0, 0, 1, 0]) / REGIME_A['ECS']
        + outer([1, -1, 0, 0]) / FOUR_MODE_CAPS['eC']
        + outer([0, 0, 0, 1]) / FOUR_MODE_CAPS['eP']
    )
    ECmat = np.linalg.inv(TRANSFORMATION.T @ Cnode @ TRANSFORMATION)

    branch_data = (
        (REGIME_A['ELK'] * (1 - eps_LK), np.array([1, 0, 0, -1.0])),
        (REGIME_A['ELK'] * (1 + eps_LK), np.array([0, 1, 0, -1.0])),
        (REGIME_A['EL'], np.array([0, 0, 1, -1.0])),
    )
    K = np.zeros((3, 3))
    for energy, node_branch in branch_data:
        theta_branch = TRANSFORMATION.T @ node_branch
        assert abs(theta_branch[0]) < 1e-14
        K += energy * outer(theta_branch[1:])

    return (
        ECmat,
        K,
        REGIME_A['EJ'] * (1 - eps_J),
        REGIME_A['EJ'] * (1 + eps_J),
    )

#three cases: 
#symmetric:             e_J = 0; e_LK = 0
#positive asymmetry:    e_J = +0.10; e_LK = +0.05
#reversed asymmetry:    e_J = -0.10; e_LK = -0.05
@pytest.mark.parametrize('eps_J,eps_LK', [(0.0, 0.0), (0.10, 0.05), (-0.10, -0.05)])


def test_four_mode_coefficients_match_independent_branch_construction(eps_J, eps_LK):
    expected = _branch_reference_coefficients(eps_J, eps_LK)
    actual = _coeffs(**REGIME_A, eps_J=eps_J, eps_LK=eps_LK, **FOUR_MODE_CAPS)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)

#verifies that changing the asymmetry parameters modifies the intended KITE branches 
def test_four_mode_asymmetry_coefficient_derivatives_have_correct_branch_signs():
    epsilon = 1e-4
    minus_j = _coeffs(**REGIME_A, eps_J=-epsilon, eps_LK=0.0, **FOUR_MODE_CAPS)
    plus_j = _coeffs(**REGIME_A, eps_J=epsilon, eps_LK=0.0, **FOUR_MODE_CAPS)
    assert np.isclose((plus_j[2] - minus_j[2]) / (2 * epsilon), -REGIME_A['EJ'])
    assert np.isclose((plus_j[3] - minus_j[3]) / (2 * epsilon), REGIME_A['EJ'])

    minus_l = _coeffs(**REGIME_A, eps_J=0.0, eps_LK=-epsilon, **FOUR_MODE_CAPS)
    plus_l = _coeffs(**REGIME_A, eps_J=0.0, eps_LK=epsilon, **FOUR_MODE_CAPS)
    v1 = np.array([0.0, 0.0, -1.0])
    v2 = np.array([1.0, 0.0, -1.0])
    expected_dK = REGIME_A['ELK'] * (-np.outer(v1, v1) + np.outer(v2, v2))
    np.testing.assert_allclose(
        (plus_l[1] - minus_l[1]) / (2 * epsilon), expected_dK,
        rtol=1e-11, atol=1e-11,
    )

#independently constructs the forth-mode phase operator, charge operator and bare harmonic-oscillator energies 
def _oscillator_operators(ECmat, K, N4):


    #A4, B4 = 4 * ECmat[3, 3],
    A4, B4 = 4 * ECmat[3, 3], 0.5 * K[2, 2]

    #zero-point amplitudes 
    f4 = (A4 / (4 * B4)) ** 0.25 
    g4 = 1.0 / (2 * f4)

    a = np.diag(np.sqrt(np.arange(1, N4)), 1) #annihilation operator 
    theta4 = f4 * (a + a.T) #phase operator 
    n4 = 1j * g4 * (a.T - a) #charge operator 
    energies4 = 2 * np.sqrt(A4 * B4) * (np.arange(N4) + 0.5) 
    return theta4, n4, energies4

#constructs the complete four-mode hamiltonian 
def _direct_four_mode_hamiltonian(model):

    ECmat, K, EJ1, EJ2 = _coeffs(
        model.EJ, model.EC, model.EL, model.ELK, model.EJS, model.ECS,
        model.eps_J, model.eps_LK, eC=model.eC, eP=model.eP,
    )

    #build nonlinear sector: H_123
    H1, ops = _assemble_nonlinear_sector(
        ECmat[:3, :3], K[:2, :2], EJ1, EJ2, model.EJS, model.ng,
        model.phi_ext, model.theta_ext,
        model.n1max, model.N2, model.L2, model.N3, model.L3,
    )

    theta4, n4, energies4 = _oscillator_operators(ECmat, K, model.N4)

    kron3 = ops['kron3']

    sector_ops = (
        (8 * ECmat[0, 3], kron3(sps.diags(ops['nvec'] + model.ng), ops['I2'], ops['I3']), n4),
        (8 * ECmat[1, 3], kron3(ops['I1'], ops['n2'], ops['I3']), n4),
        (8 * ECmat[2, 3], kron3(ops['I1'], ops['I2'], ops['n3']), n4),
        (K[0, 2], kron3(ops['I1'], sps.diags(ops['x2']), ops['I3']), theta4),
        (K[1, 2], kron3(ops['I1'], ops['I2'], sps.diags(ops['x3'])), theta4),
    )

    H = np.kron(H1.toarray(), np.eye(model.N4)) #uncoupled tensor hamiltonian 
    H += np.kron(np.eye(H1.shape[0]), np.diag(energies4))

    for coefficient, sector_operator, node_operator in sector_ops: #add every coupling 
        H += coefficient * np.kron(sector_operator.toarray(), node_operator)
    return 0.5 * (H + H.conj().T)


@pytest.mark.parametrize('case', [
    dict(eps_J=0.0, eps_LK=0.0, ng=0.0, phi_ext=0.0, theta_ext=np.pi), #symmetric protection point 
    dict(eps_J=0.07, eps_LK=0.04, ng=0.2, phi_ext=0.3, theta_ext=np.pi + 0.1), #generic point 
])

#compare hierarchial vs direct solver in low-energy spectrum 
def test_four_mode_complete_hierarchy_matches_direct_tensor_diagonalization(case):

    tiny = dict(n1max=1, 
                            N2=5, 
                            L2=4.0, 
                            N3=5, 
                            L3=4.0, 
                            N4=3
                            ) #D_full = 225

    sector_dimension = (2 * tiny['n1max'] + 1) * tiny['N2'] * tiny['N3'] #complete sector 

    model = Gridium4Mode(
        **REGIME_A, **FOUR_MODE_CAPS, **case, **tiny,
        nkeep=sector_dimension, nlev=6,
    )

    hierarchical = model.levels()
    direct = np.linalg.eigvalsh(_direct_four_mode_hamiltonian(model))[:model.nlev]

    np.testing.assert_allclose(hierarchical, direct, rtol=1e-10, atol=1e-9)


def _tiny_complete_model(**overrides):
    settings = dict(
        **REGIME_A, **FOUR_MODE_CAPS,
        eps_J=0.07, eps_LK=0.04, ng=0.17,
        phi_ext=0.31, theta_ext=2.63,
        n1max=1, N2=5, L2=4.0, N3=5, L3=4.0, N4=3, nlev=6,
    )
    settings.update(overrides)
    settings['nkeep'] = (2 * settings['n1max'] + 1) * settings['N2'] * settings['N3']
    return Gridium4Mode(**settings)


def _primitive_flux_derivative(model, parameter, delta=1e-5):
    fluxes = dict(phi_ext=model.phi_ext, theta_ext=model.theta_ext)
    fluxes[parameter] = getattr(model, parameter) + delta
    plus = _direct_four_mode_hamiltonian(_tiny_complete_model(**fluxes))
    fluxes[parameter] = getattr(model, parameter) - delta
    minus = _direct_four_mode_hamiltonian(_tiny_complete_model(**fluxes))
    return (plus - minus) / (2 * delta)


@pytest.mark.parametrize('parameter', ['phi_ext', 'theta_ext'])
def test_four_mode_fluxes_are_two_pi_periodic(parameter):
    reference = _tiny_complete_model()
    shifted = _tiny_complete_model(
        **{parameter: getattr(reference, parameter) + 2 * np.pi}
    )

    # Flux enters the implemented branch cosines, so the primitive-basis
    # Hamiltonian itself (not only its spectrum) is exactly 2*pi periodic.
    reference_hamiltonian = _direct_four_mode_hamiltonian(reference)
    shifted_hamiltonian = _direct_four_mode_hamiltonian(shifted)
    np.testing.assert_allclose(
        shifted_hamiltonian,
        reference_hamiltonian,
        rtol=0.0,
        atol=5e-13,
    )
    np.testing.assert_allclose(
        shifted.levels(), reference.levels(), rtol=0.0, atol=1e-10,
    )

    reference_energies, reference_states = np.linalg.eigh(reference_hamiltonian)
    shifted_energies, shifted_states = np.linalg.eigh(shifted_hamiltonian)
    nlev = reference.nlev
    reference_states = reference_states[:, :nlev]
    shifted_states = shifted_states[:, :nlev]

    # The anchor is intentionally nondegenerate, including separation from the
    # first state outside the compared subspace.
    assert np.min(np.diff(reference_energies[:nlev + 1])) > 1e-3
    assert np.min(np.diff(shifted_energies[:nlev + 1])) > 1e-3

    overlap = reference_states.conj().T @ shifted_states
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    matching = np.empty(nlev, dtype=int)
    matching[rows] = columns
    matched_overlaps = np.array([
        overlap[index, matching[index]] for index in range(nlev)
    ])
    assert np.min(np.abs(matched_overlaps)) > 1.0 - 1e-10
    np.testing.assert_allclose(
        shifted_energies[matching], reference_energies[:nlev],
        rtol=0.0, atol=1e-10,
    )

    aligned_shifted_states = shifted_states[:, matching] * np.exp(
        -1j * np.angle(matched_overlaps)
    )[np.newaxis, :]
    np.testing.assert_allclose(
        reference_states.conj().T @ aligned_shifted_states,
        np.eye(nlev), rtol=0.0, atol=5e-10,
    )

    for derivative_parameter, operator_name in (
        ('phi_ext', 'd_phi'), ('theta_ext', 'd_theta'),
    ):
        reference_derivative = _primitive_flux_derivative(
            reference, derivative_parameter,
        )
        shifted_derivative = _primitive_flux_derivative(
            shifted, derivative_parameter,
        )

        # These derivative operators use the same primitive basis and gauge, so
        # periodicity requires elementwise equality, including signs.
        np.testing.assert_allclose(
            shifted_derivative, reference_derivative, rtol=0.0, atol=1e-9,
        )

        reference_projected = (
            reference_states.conj().T @ reference_derivative @ reference_states
        )
        shifted_projected = (
            aligned_shifted_states.conj().T
            @ shifted_derivative
            @ aligned_shifted_states
        )
        np.testing.assert_allclose(
            shifted_projected, reference_projected, rtol=0.0, atol=1e-9,
        )

        # Tie the state-aligned primitive check to the public projected
        # operators. At this nondegenerate anchor their remaining ambiguity is
        # only an independent phase for each eigenvector.
        reference_public = getattr(reference, operator_name)().full()
        shifted_public = getattr(shifted, operator_name)().full()
        np.testing.assert_allclose(
            np.real(np.diag(shifted_public)),
            np.real(np.diag(reference_public)),
            rtol=0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.abs(shifted_public), np.abs(reference_public),
            rtol=0.0, atol=1e-10,
        )


def test_offset_charge_periodicity_is_a_gauge_identity_with_cutoff_error():
    common = dict(
        **REGIME_A, **FOUR_MODE_CAPS,
        eps_J=0.07, eps_LK=0.04, phi_ext=0.31, theta_ext=2.63,
        N2=3, L2=4.0, N3=3, L3=4.0, N4=2, nlev=4,
    )
    ng = 0.17

    # Formally, ng -> ng + 1 is undone by translating compact charge by one.
    # A finite charge basis lacks one boundary state, but the overlapping
    # interior blocks of the complete primitive Hamiltonians remain identical.
    n1max = 2
    sector_dimension = (2 * n1max + 1) * common['N2'] * common['N3']
    reference = Gridium4Mode(
        **common, ng=ng, n1max=n1max, nkeep=sector_dimension,
    )
    shifted = Gridium4Mode(
        **common, ng=ng + 1.0, n1max=n1max, nkeep=sector_dimension,
    )
    reference_hamiltonian = _direct_four_mode_hamiltonian(reference)
    shifted_hamiltonian = _direct_four_mode_hamiltonian(shifted)

    charge_dimension = 2 * n1max + 1
    outer_dimension = common['N2'] * common['N3']
    shifted_sector_indices = np.concatenate([
        outer * charge_dimension + np.arange(charge_dimension - 1)
        for outer in range(outer_dimension)
    ])
    reference_sector_indices = np.concatenate([
        outer * charge_dimension + np.arange(1, charge_dimension)
        for outer in range(outer_dimension)
    ])

    def include_fourth_mode(sector_indices):
        return np.concatenate([
            index * common['N4'] + np.arange(common['N4'])
            for index in sector_indices
        ])

    shifted_indices = include_fourth_mode(shifted_sector_indices)
    reference_indices = include_fourth_mode(reference_sector_indices)
    np.testing.assert_allclose(
        shifted_hamiltonian[np.ix_(shifted_indices, shifted_indices)],
        reference_hamiltonian[np.ix_(reference_indices, reference_indices)],
        rtol=0.0,
        atol=5e-13,
    )

    # The full finite matrices retain different boundary states, so periodicity
    # is approximate. This small ladder checks decreasing finite-boundary error
    # at this bounded anchor. The terminal bounds below are empirical regression
    # thresholds, not production-convergence criteria.
    absolute_errors = []
    transition_errors = []
    for cutoff in (1, 2, 3, 4):
        dimension = (2 * cutoff + 1) * common['N2'] * common['N3']
        levels = []
        for offset_charge in (ng, ng + 1.0):
            model = Gridium4Mode(
                **common, ng=offset_charge, n1max=cutoff, nkeep=dimension,
            )
            levels.append(model.levels())
        absolute_errors.append(np.max(np.abs(levels[1] - levels[0])))
        transition_errors.append(np.max(np.abs(
            (levels[1] - levels[1][0]) - (levels[0] - levels[0][0])
        )))

    assert np.all(np.diff(absolute_errors) < 0.0)
    assert np.all(np.diff(transition_errors) < 0.0)
    assert absolute_errors[-1] < 1e-2
    assert transition_errors[-1] < 1e-2


def _small_stage1_problem():
    settings = dict(
        **REGIME_A, **FOUR_MODE_CAPS,
        eps_J=0.07, eps_LK=0.04, ng=0.17,
        phi_ext=0.31, theta_ext=2.63,
        n1max=1, N2=7, L2=4.0, N3=7, L3=4.0,
    )
    ECmat, K, EJ1, EJ2 = _coeffs(
        settings['EJ'], settings['EC'], settings['EL'], settings['ELK'],
        settings['EJS'], settings['ECS'], settings['eps_J'],
        settings['eps_LK'], eC=settings['eC'], eP=settings['eP'],
    )
    matrix, _ = _assemble_nonlinear_sector(
        ECmat[:3, :3], K[:2, :2], EJ1, EJ2, settings['EJS'],
        settings['ng'], settings['phi_ext'], settings['theta_ext'],
        settings['n1max'], settings['N2'], settings['L2'],
        settings['N3'], settings['L3'],
    )
    sigma = -(EJ1 + EJ2 + settings['EJS'] + 10.0)
    return settings, matrix, sigma


def test_stage1_hamiltonian_is_hermitian():
    _, matrix, _ = _small_stage1_problem()
    antihermitian = matrix - matrix.conj().T
    residual = (
        0.0 if antihermitian.nnz == 0
        else np.max(np.abs(antihermitian.data))
    )
    assert residual < 1e-13


def test_stage1_explicit_shift_invert_matches_scipy_default_path():
    _, matrix, sigma = _small_stage1_problem()
    v0 = np.linspace(1.0, 2.0, matrix.shape[0])
    common = dict(
        k=14, sigma=sigma, which='LM', tol=1e-10, ncv=32, v0=v0,
        collect_diagnostics=True,
    )
    default_values, default_vectors, default_diagnostics = _stage1_eigensolve(
        matrix, **common,
    )
    explicit_values, explicit_vectors, explicit_diagnostics = _stage1_eigensolve(
        matrix, permc_spec='MMD_AT_PLUS_A', **common,
    )

    np.testing.assert_allclose(
        explicit_values, default_values, rtol=1e-12, atol=1e-11,
    )
    for values, vectors in (
        (default_values, default_vectors),
        (explicit_values, explicit_vectors),
    ):
        residuals = np.linalg.norm(
            matrix @ vectors - vectors * values[np.newaxis, :], axis=0,
        )
        assert np.max(residuals) < 1e-8

    assert default_diagnostics['path'] == 'scipy_default_shift_invert'
    assert explicit_diagnostics['path'] == 'explicit_shift_invert'
    assert explicit_diagnostics['permc_spec'] == 'MMD_AT_PLUS_A'
    assert explicit_diagnostics['factorization_seconds'] >= 0.0
    assert explicit_diagnostics['eigsh_seconds'] >= 0.0
    assert explicit_diagnostics['opinv_solve_count'] > 0
    assert explicit_diagnostics['LU_nnz'] > 0
    assert explicit_diagnostics['fill_ratio'] > 0.0


def test_stage1_legacy_defaults_preserve_original_eigsh_call(monkeypatch):
    settings, matrix, expected_sigma = _small_stage1_problem()
    captured = []

    def fake_eigsh(actual_matrix, **kwargs):
        captured.append((actual_matrix, kwargs))
        values, vectors = np.linalg.eigh(actual_matrix.toarray())
        selected = np.argsort(np.abs(values - kwargs['sigma']))[:kwargs['k']]
        return values[selected], vectors[:, selected]

    def unexpected_splu(*args, **kwargs):
        raise AssertionError('The legacy/default path must not call splu.')

    monkeypatch.setattr(spsl, 'eigsh', fake_eigsh)
    monkeypatch.setattr(spsl, 'splu', unexpected_splu)

    model = Gridium4Mode(
        **settings, N4=3, nkeep=14, nlev=6,
    )
    levels = model.levels()

    assert len(captured) == 1
    actual_matrix, kwargs = captured[0]
    assert actual_matrix.shape == matrix.shape
    assert kwargs['sigma'] == expected_sigma
    assert kwargs['which'] == 'LM'
    assert kwargs['tol'] == 1e-8
    assert kwargs['ncv'] is None
    assert kwargs['v0'] is None
    assert 'OPinv' not in kwargs
    assert np.all(np.isfinite(levels))
    assert model.stage1_solver_diagnostics is None


def test_spectrum_sweep_preserves_but_does_not_sweep_stage1_options(monkeypatch):
    settings, matrix, sigma = _small_stage1_problem()
    supplied_v0 = np.linspace(1.0, 2.0, matrix.shape[0])
    model = Gridium4Mode(
        **settings, N4=3, nkeep=14, nlev=6,
        stage1_sigma=sigma, stage1_which='LM', stage1_tol=2e-9,
        stage1_ncv=30, stage1_v0=supplied_v0,
        stage1_permc_spec='MMD_AT_PLUS_A',
        stage1_collect_diagnostics=True,
    )
    supplied_v0[:] = -1.0
    assert not np.array_equal(model.stage1_v0, supplied_v0)
    model.stage1_solver_diagnostics = {'transient': True}

    constructor_kwargs = model._constructor_kwargs()
    reconstructed = Gridium4Mode(**constructor_kwargs)
    assert reconstructed.stage1_sigma == sigma
    assert reconstructed.stage1_which == 'LM'
    assert reconstructed.stage1_tol == 2e-9
    assert reconstructed.stage1_ncv == 30
    assert reconstructed.stage1_permc_spec == 'MMD_AT_PLUS_A'
    assert reconstructed.stage1_collect_diagnostics is True
    np.testing.assert_array_equal(reconstructed.stage1_v0, model.stage1_v0)
    assert reconstructed.stage1_v0 is not model.stage1_v0
    assert reconstructed.stage1_solver_diagnostics is None
    assert 'stage1_solver_diagnostics' not in constructor_kwargs

    def inspect_tasks(tasks, num_cpus):
        tasks = list(tasks)
        assert num_cpus == 1
        for kwargs, parameter, _, evals_count in tasks:
            assert parameter == 'phi_ext'
            assert kwargs['stage1_sigma'] == sigma
            assert kwargs['stage1_tol'] == 2e-9
            assert kwargs['stage1_ncv'] == 30
            assert kwargs['stage1_permc_spec'] == 'MMD_AT_PLUS_A'
            np.testing.assert_array_equal(kwargs['stage1_v0'], model.stage1_v0)
            assert 'stage1_solver_diagnostics' not in kwargs
        return [np.zeros(task[3]) for task in tasks]

    monkeypatch.setattr(gridium_netlist, '_ordered_process_map', inspect_tasks)
    result = model.get_spectrum_vs_paramvals(
        'phi_ext', [0.1, 0.2], evals_count=4, num_cpus=1,
    )
    assert result.energy_table.shape == (2, 4)

    for parameter in gridium_netlist._STAGE1_NUMERICAL_PARAMETERS:
        with pytest.raises(ValueError, match='Unknown or unsupported sweep parameter'):
            model.get_spectrum_vs_paramvals(parameter, [0.0], num_cpus=1)


def test_explicit_shift_invert_preserves_downstream_model_and_operators():
    settings, matrix, sigma = _small_stage1_problem()
    v0 = np.linspace(1.0, 2.0, matrix.shape[0])
    numerical = dict(
        N4=3, nkeep=14, nlev=6, stage1_sigma=sigma,
        stage1_which='LM', stage1_tol=1e-10, stage1_ncv=32,
        stage1_v0=v0, stage1_collect_diagnostics=True,
    )
    default = Gridium4Mode(**settings, **numerical)
    explicit = Gridium4Mode(
        **settings, **numerical, stage1_permc_spec='MMD_AT_PLUS_A',
    )

    default_operators = {
        name: getattr(default, name)().full()
        for name in ('d_phi', 'd_theta', 'n1', 'phase2', 'grid_n', 'grid_phi')
    }
    explicit_operators = {
        name: getattr(explicit, name)().full()
        for name in default_operators
    }

    np.testing.assert_allclose(
        explicit.levels(), default.levels(), rtol=1e-11, atol=1e-10,
    )
    # This generic asymmetric point is nondegenerate, so the two eigensolvers
    # may differ only by independent eigenvector phases, removed by abs().
    for name in default_operators:
        np.testing.assert_allclose(
            np.abs(explicit_operators[name]), np.abs(default_operators[name]),
            rtol=1e-9, atol=1e-10,
        )

    physical_parameters = (
        'EJ', 'EC', 'EL', 'ELK', 'EJS', 'ECS', 'eC', 'eP',
        'eps_J', 'eps_LK', 'ng', 'phi_ext', 'theta_ext',
        'n1max', 'N2', 'L2', 'N3', 'L3', 'N4', 'nkeep', 'nlev',
    )
    for name in physical_parameters:
        assert getattr(explicit, name) == getattr(default, name)
    assert default.stage1_solver_diagnostics['path'] == 'scipy_default_shift_invert'
    assert explicit.stage1_solver_diagnostics['path'] == 'explicit_shift_invert'


@pytest.mark.parametrize('parameter,operator_name', [
    ('theta_ext', 'd_theta'),
    ('phi_ext', 'd_phi'),
])

# Validates the two external-flux response/sensitivity operators.
def test_four_mode_flux_response_matches_brute_force_finite_difference(
    parameter, operator_name,
):
    delta = 1e-5
    model = _tiny_complete_model()
    _, states = np.linalg.eigh(_direct_four_mode_hamiltonian(model))
    states = states[:, :model.nlev]

    shifted = getattr(model, parameter)
    minus = _tiny_complete_model(**{parameter: shifted - delta})
    plus = _tiny_complete_model(**{parameter: shifted + delta})


    derivative_full = (
        _direct_four_mode_hamiltonian(plus) - _direct_four_mode_hamiltonian(minus)
    ) / (2 * delta)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        expected = states.conj().T @ derivative_full @ states
    expected = 0.5 * (expected + expected.conj().T)

    actual = getattr(model, operator_name)().full()
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(actual), np.abs(expected), rtol=2e-6, atol=2e-7)

    energy_derivative = (plus.levels() - minus.levels()) / (2 * delta)

    np.testing.assert_allclose(
        np.real(np.diag(actual)), energy_derivative, rtol=2e-6, atol=2e-7,
    )

#constructs four physical operators in the complete primitive four-mode basis: 
# compact charge n1
# phase theta2
# effective grid charge n_grid 
# effective grid phase phi_grid 

def _direct_coordinate_operators(model):

    ECmat, K, EJ1, EJ2 = _coeffs(model.EJ, 
                                 model.EC, 
                                 model.EL, 
                                 model.ELK, 
                                 model.EJS, 
                                 model.ECS,
                                 model.eps_J, 
                                 model.eps_LK, 
                                 eC=model.eC, 
                                 eP=model.eP,
    )

    #sector operators 
    _, ops = _assemble_nonlinear_sector(ECmat[:3, :3], 
                                        K[:2, :2], 
                                        EJ1, 
                                        EJ2, 
                                        model.EJS, 
                                        model.ng,
                                        model.phi_ext, 
                                        model.theta_ext,
                                        model.n1max, 
                                        model.N2, 
                                        model.L2, 
                                        model.N3, 
                                        model.L3,
    )

    #H_123 = H1 * H2 * H3
    kron3 = ops['kron3']

    n1_sector = kron3(
        sps.diags(ops['nvec']), 
        ops['I2'], 
        ops['I3']
    )

    n3_sector = kron3(
        ops['I1'], 
        ops['I2'], 
        ops['n3']
    )

    phase2_sector = kron3(
        ops['I1'], sps.diags(ops['x2']), 
        ops['I3'],
    )
    phase3_sector = kron3(
        ops['I1'], ops['I2'], 
        sps.diags(ops['x3']),
    )

    _, n4, _ = _oscillator_operators(ECmat, K, model.N4)

    identity4 = np.eye(model.N4)
    identity_sector = np.eye(n1_sector.shape[0])

    return {
        'n1': np.kron(n1_sector.toarray(), identity4),
        'phase2': np.kron(phase2_sector.toarray(), identity4),
        'grid_n': (
            np.kron((-n1_sector + n3_sector).toarray(), identity4)
            + np.kron(identity_sector, n4)
        ),
        'grid_phi': np.kron(
            (0.5 * phase2_sector + phase3_sector).toarray(), identity4,
        ),
    }

#verifies that the effective's grid phase is translated correctly into netlist coordinates
def test_effective_grid_pair_follows_paper_canonical_coordinate_transform():
    
    paper_from_netlist = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.5, 1.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 1.0, 1.0, -1.0],
    ])

    charge_from_netlist = np.linalg.inv(paper_from_netlist).T #transforming conjugate charges

    np.testing.assert_allclose(
        paper_from_netlist[1], 
        [0.0, 0.5, 1.0, 0.0], 
        rtol=0.0, 
        atol=1e-14,
    )

    np.testing.assert_allclose(
        charge_from_netlist[1], 
        [-1.0, 0.0, 1.0, 1.0], 
        rtol=0.0, 
        atol=1e-14,
    )

    assert np.isclose(
        paper_from_netlist[1] @ charge_from_netlist[1], 1.0,
    )


@pytest.mark.parametrize('operator_name', ['n1', 'phase2', 'grid_n', 'grid_phi'])

def test_four_mode_charge_phase_operator_matches_direct_tensor_projection(operator_name):

    model = _tiny_complete_model()

    _, states = np.linalg.eigh(_direct_four_mode_hamiltonian(model))
    states = states[:, :model.nlev]
    full_operator = _direct_coordinate_operators(model)[operator_name]

    
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        expected = states.conj().T @ full_operator @ states
    expected = 0.5 * (expected + expected.conj().T)

    actual = getattr(model, operator_name)().full()
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(actual), np.abs(expected), rtol=2e-8, atol=2e-9)


def _hand_snapshot(cutoffs, point, eps_J=0.0, eps_LK=0.0, evals_count=6):
    inputs = dict(
        parameters=REGIME_A, capacitors=FOUR_MODE_CAPS, cutoffs=cutoffs,
        point=point, eps_J=eps_J, eps_LK=eps_LK,
    )
    if evals_count != 6:
        inputs['evals_count'] = evals_count

    def compute():
        start = time.time()
        model = Gridium4Mode(
            **REGIME_A, **FOUR_MODE_CAPS, **cutoffs, **point,
            eps_J=eps_J, eps_LK=eps_LK, nlev=evals_count,
        )
        energies = model.levels()
        return dict(
            transitions_GHz=energies[1:] - energies[0],
            finite=bool(np.all(np.isfinite(energies))),
            seconds=time.time() - start,
        )

    return _cached('hand', inputs, compute)['result']


def _final_hierarchy_snapshot():
    inputs = dict(
        parameters=REGIME_A, capacitors=FOUR_MODE_CAPS,
        cutoffs=FINAL_HAND_CUTOFFS, point=WORST_POINT,
        nkeep=FINAL_NKEEP_LADDER, N4=(FINAL_HAND_CUTOFFS['N4'],),
        evals_count=EXTENDED_EVALS_COUNT,
    )

    def compute():
        start = time.time()
        ECmat, K, EJ1, EJ2 = _coeffs(
            **REGIME_A, eps_J=0.0, eps_LK=0.0, **FOUR_MODE_CAPS,
        )
        H1, ops = _assemble_nonlinear_sector(
            ECmat[:3, :3], K[:2, :2], EJ1, EJ2, REGIME_A['EJS'], 0.0,
            WORST_POINT['phi_ext'], WORST_POINT['theta_ext'],
            FINAL_HAND_CUTOFFS['n1max'], FINAL_HAND_CUTOFFS['N2'],
            FINAL_HAND_CUTOFFS['L2'], FINAL_HAND_CUTOFFS['N3'], FINAL_HAND_CUTOFFS['L3'],
        )
        sigma = -(EJ1 + EJ2 + REGIME_A['EJS'] + 10.0)
        max_nkeep = max(FINAL_NKEEP_LADDER)
        w1, vectors = spsl.eigsh(
            H1.tocsc(), k=max_nkeep, sigma=sigma, which='LM', tol=1e-8,
        )
        order = np.argsort(w1)
        w1, vectors = np.real(w1[order]), vectors[:, order]

        kron3 = ops['kron3']
        full_sector_ops = (
            kron3(sps.diags(ops['nvec']), ops['I2'], ops['I3']),
            kron3(ops['I1'], ops['n2'], ops['I3']),
            kron3(ops['I1'], ops['I2'], ops['n3']),
            kron3(ops['I1'], sps.diags(ops['x2']), ops['I3']),
            kron3(ops['I1'], ops['I2'], sps.diags(ops['x3'])),
        )
        projected = []
        for operator in full_sector_ops:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                matrix = vectors.conj().T @ (operator @ vectors)
            if not np.all(np.isfinite(matrix)):
                raise FloatingPointError('Projected hierarchy operator contains NaN/Inf.')
            projected.append(matrix)

        spectra = {}
        N4 = FINAL_HAND_CUTOFFS['N4']
        theta4, n4, energies4 = _oscillator_operators(ECmat, K, N4)
        for nkeep in FINAL_NKEEP_LADDER:
            Hc = np.kron(np.diag(w1[:nkeep]), np.eye(N4))
            Hc += np.kron(np.eye(nkeep), np.diag(energies4))
            coefficients = (
                (8 * ECmat[0, 3], n4),
                (8 * ECmat[1, 3], n4),
                (8 * ECmat[2, 3], n4),
                (K[0, 2], theta4),
                (K[1, 2], theta4),
            )
            for sector_operator, (coefficient, node_operator) in zip(projected, coefficients):
                Hc += coefficient * np.kron(
                    sector_operator[:nkeep, :nkeep], node_operator,
                )
            Hc = 0.5 * (Hc + Hc.conj().T)
            energies = np.linalg.eigvalsh(Hc)[:EXTENDED_EVALS_COUNT]
            spectra['nkeep=%d,N4=%d' % (nkeep, N4)] = energies[1:] - energies[0]
        return dict(spectra=spectra, finite=True, seconds=time.time() - start)

    return _cached('final_hierarchy_ladder', inputs, compute)['result']


@pytest.mark.skipif(
    not RUN_HAND,
    reason='Set GRID_NETLIST_RUN_HAND_CONVERGENCE=1 for production spectrum convergence.',
)
def test_final_handwritten_nkeep_320_to_360():
    result = _final_hierarchy_snapshot()
    assert result['finite']
    N4 = FINAL_HAND_CUTOFFS['N4']
    previous, current = FINAL_NKEEP_LADDER
    baseline = dict(
        transitions_GHz=result['spectra']['nkeep=%d,N4=%d' % (previous, N4)],
    )
    refined = dict(
        transitions_GHz=result['spectra']['nkeep=%d,N4=%d' % (current, N4)],
    )
    _assert_transition_convergence(
        'handwritten theta=0 nkeep %d -> %d (first six states)'
        % (previous, current),
        {'transitions_GHz': baseline['transitions_GHz'][:5]},
        {'transitions_GHz': refined['transitions_GHz'][:5]},
        protected_doublet=False,
    )
    print('all ten transition changes: %s MHz' % np.array2string(
        _transition_delta_mhz(baseline['transitions_GHz'], refined['transitions_GHz']),
        precision=3,
    ))


@pytest.mark.skipif(
    not RUN_HAND,
    reason='Set GRID_NETLIST_RUN_HAND_CONVERGENCE=1 for production spectrum convergence.',
)
def test_final_handwritten_theta2_resolution_111_to_131():
    cutoff_111 = dict(FINAL_HAND_CUTOFFS, N2=111, nkeep=320)
    cutoff_131 = dict(FINAL_HAND_CUTOFFS, N2=131, nkeep=320)
    baseline = _hand_snapshot(cutoff_111, WORST_POINT)
    refined = _hand_snapshot(cutoff_131, WORST_POINT)
    assert baseline['finite'] and refined['finite']
    _assert_transition_convergence(
        'handwritten theta=0 N2 111 -> 131', baseline, refined,
        protected_doublet=False,
    )


def _doublet_block_singular_values(operator):
    operator = np.asarray(operator)
    if operator.shape[0] < 6 or operator.shape[1] < 6:
        raise ValueError('At least six levels are required for three doublets.')
    result = {}
    for first in range(3):
        for second in range(first, 3):
            block = operator[2 * first:2 * first + 2, 2 * second:2 * second + 2]
            result['D%d-D%d' % (first, second)] = np.linalg.svd(
                block, compute_uv=False,
            )
    return result


def _serialize_flux_response_invariants(operators):
    return {
        name: {
            block: values.tolist()
            for block, values in _doublet_block_singular_values(operator).items()
        }
        for name, operator in operators.items()
    }


def _assert_flux_response_invariants_converged(label, baseline, refined):
    baseline_blocks = baseline['drive_block_singular_values_GHz_per_rad']
    refined_blocks = refined['drive_block_singular_values_GHz_per_rad']
    for operator_name in ('d_theta', 'd_phi'):
        print('%s: %s doublet-block convergence (GHz/radian):' % (label, operator_name))
        for block_name in baseline_blocks[operator_name]:
            previous = np.asarray(baseline_blocks[operator_name][block_name])
            current = np.asarray(refined_blocks[operator_name][block_name])
            delta_mhz = 1e3 * np.abs(current - previous)
            relative = np.abs(current - previous) / np.maximum(np.abs(current), 1e-12)
            print('  %-5s baseline=%s  refined=%s  delta=%s MHz  rel=%s' % (
                block_name,
                np.array2string(previous, precision=7),
                np.array2string(current, precision=7),
                np.array2string(delta_mhz, precision=3),
                np.array2string(relative, precision=4),
            ))
            np.testing.assert_allclose(
                previous, current,
                rtol=DRIVE_SELF_RTOL,
                atol=DRIVE_SELF_ATOL_MHZ / 1e3,
                err_msg='%s %s %s is not converged' % (label, operator_name, block_name),
            )


def _assert_hand_flux_response_snapshot_converged(label, baseline, refined):
    _assert_transition_convergence(label, baseline, refined)
    _assert_flux_response_invariants_converged(label, baseline, refined)


def _hand_flux_response_snapshot(cutoffs, point, eps_J=0.0, eps_LK=0.0):
    inputs = dict(
        parameters=REGIME_A, capacitors=FOUR_MODE_CAPS, cutoffs=cutoffs,
        point=point, eps_J=eps_J, eps_LK=eps_LK, evals_count=6,
    )

    def compute():
        start = time.time()
        model = Gridium4Mode(
            **REGIME_A, **FOUR_MODE_CAPS, **cutoffs, **point,
            eps_J=eps_J, eps_LK=eps_LK, nlev=6,
        )
        operators = {
            'd_theta': model.d_theta().full(),
            'd_phi': model.d_phi().full(),
        }
        energies = model.levels()
        return dict(
            transitions_GHz=energies[1:] - energies[0],
            drive_block_singular_values_GHz_per_rad=(
                _serialize_flux_response_invariants(operators)
            ),
            finite=bool(
                np.all(np.isfinite(energies))
                and all(np.all(np.isfinite(operator)) for operator in operators.values())
            ),
            seconds=time.time() - start,
        )

    return _cached('hand_drive', inputs, compute)['result']


@pytest.mark.skipif(
    not RUN_DRIVE,
    reason=('Set legacy GRID_NETLIST_RUN_DRIVE_VALIDATION=1 for production '
            'flux-response convergence.'),
)
def test_handwritten_flux_response_nkeep_320_to_360_convergence():
    baseline = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, nkeep=320), PROTECTION_POINT,
    )
    refined = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, nkeep=360), PROTECTION_POINT,
    )
    assert baseline['finite'] and refined['finite']
    _assert_hand_flux_response_snapshot_converged(
        'symmetric handwritten nkeep 320 -> 360', baseline, refined,
    )


@pytest.mark.skipif(
    not RUN_DRIVE,
    reason=('Set legacy GRID_NETLIST_RUN_DRIVE_VALIDATION=1 for production '
            'flux-response convergence.'),
)
def test_handwritten_flux_response_theta2_resolution_111_to_131_convergence():
    baseline = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, N2=111, nkeep=320), PROTECTION_POINT,
    )
    refined = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, N2=131, nkeep=320), PROTECTION_POINT,
    )
    assert baseline['finite'] and refined['finite']
    _assert_hand_flux_response_snapshot_converged(
        'symmetric handwritten N2 111 -> 131 at nkeep=320', baseline, refined,
    )


@pytest.mark.skipif(
    not RUN_DRIVE,
    reason=('Set legacy GRID_NETLIST_RUN_DRIVE_VALIDATION=1 for production '
            'flux-response convergence.'),
)
def test_asymmetric_handwritten_spectrum_and_flux_response_convergence():
    eps_J, eps_LK = 0.10, 0.05
    baseline = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, N2=111, nkeep=320),
        PROTECTION_POINT, eps_J=eps_J, eps_LK=eps_LK,
    )
    hierarchy_refined = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, N2=111, nkeep=360),
        PROTECTION_POINT, eps_J=eps_J, eps_LK=eps_LK,
    )
    grid_refined = _hand_flux_response_snapshot(
        dict(FINAL_HAND_CUTOFFS, N2=131, nkeep=320),
        PROTECTION_POINT, eps_J=eps_J, eps_LK=eps_LK,
    )
    assert baseline['finite'] and hierarchy_refined['finite'] and grid_refined['finite']
    _assert_hand_flux_response_snapshot_converged(
        'asymmetric handwritten nkeep 320 -> 360', baseline, hierarchy_refined,
    )
    _assert_hand_flux_response_snapshot_converged(
        'asymmetric handwritten N2 111 -> 131 at nkeep=320', baseline, grid_refined,
    )
