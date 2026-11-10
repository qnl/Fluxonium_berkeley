"""Tests for the source-grounded symmetric three-mode S23 model."""

import numpy as np
import qutip as qt

from Circuit_Objs.qchard_gridium3 import (
    SymmetricThreeModeGridium,
    table_s1_regime_a,
)
from Circuit_Objs.qchard_statetracking import track_eigensystems


def _gridium(**overrides):
    parameters = dict(table_s1_regime_a)
    parameters.update({
        'phi_ext': 0.0,
        'theta_ext': np.pi,
        'trunc_sigma': 6,
        'trunc_delta': 6,
        'trunc_s': 5,
        'nlev': 6,
    })
    parameters.update(overrides)
    return SymmetricThreeModeGridium(**parameters)


def _max_abs(operator):
    if hasattr(operator.data, 'as_scipy'):
        matrix = operator.data.as_scipy()
        return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))
    return float(np.max(np.abs(operator.full())))


def _parity(dimension):
    return qt.Qobj(np.diag((-1.0) ** np.arange(dimension))).to('csr')


def _cosine(operator):
    return 0.5 * ((1j * operator).expm() + (-1j * operator).expm())


def test_hamiltonian_is_literal_s23_operator_expression():
    gridium = SymmetricThreeModeGridium(
        E_C=0.7, E_J=3.1, E_L=0.8, E_LK=1.3,
        E_JS=2.4, E_CS=4.2,
        phi_ext=0.37, theta_ext=-0.41,
        trunc_sigma=3, trunc_delta=4, trunc_s=5,
        nlev=4, phase_scales=(0.8, 0.9, 1.1),
        phase_centers=(0.2, -0.3, 0.4))
    identity = qt.qeye(gridium.hilbert_dimension)
    identity.dims = gridium.phi_sigma().dims
    shifted_s = gridium.phi_s() - gridium.phi_ext * identity
    shifted_delta = (
        gridium.phi_delta() - 0.5 * gridium.theta_ext * identity)
    expected = (
        2 * gridium.E_C * (
            gridium.n_sigma() ** 2 + gridium.n_delta() ** 2)
        + 4 * gridium.E_CS * gridium.n_s() ** 2
        - 2 * gridium.E_J
        * _cosine(gridium.phi_sigma())
        * _cosine(gridium.phi_delta())
        - gridium.E_JS * _cosine(shifted_s)
        + gridium.ELprime
        * (gridium.phi_sigma() - gridium.phi_s()) ** 2
        + gridium.E_LK * shifted_delta ** 2)

    assert _max_abs(gridium.hamiltonian_primitive() - expected) < 1e-11


def test_primitive_operators_and_hamiltonian_are_hermitian_and_sparse():
    gridium = _gridium()
    names = (
        'phi_sigma', 'phi_delta', 'phi_s',
        'n_sigma', 'n_delta', 'n_s',
        'phi_1', 'phi_2', 'phi_3',
        'n_1', 'n_2', 'n_3',
        'dH_dphi_ext', 'dH_dtheta_ext',
    )

    for name in names:
        operator = gridium.operator(name)
        assert operator.data.__class__.__name__ == 'CSR'
        assert _max_abs(operator - operator.dag()) < 1e-11

    hamiltonian = gridium.hamiltonian_primitive()
    assert hamiltonian.data.__class__.__name__ == 'CSR'
    assert _max_abs(hamiltonian - hamiltonian.dag()) < 1e-11


def test_elprime_and_bias_derivatives_match_s23():
    gridium = _gridium()
    expected_elprime = (
        2 * gridium.E_LK * gridium.E_L
        / (2 * gridium.E_LK + gridium.E_L))
    assert gridium.ELprime == expected_elprime

    step = 1e-6
    center_phi = gridium.phi_ext
    gridium.phi_ext = center_phi + step
    h_plus = gridium.hamiltonian_primitive()
    gridium.phi_ext = center_phi - step
    h_minus = gridium.hamiltonian_primitive()
    gridium.phi_ext = center_phi
    finite_difference = (h_plus - h_minus) / (2 * step)
    assert _max_abs(finite_difference - gridium.dH_dphi_ext()) < 2e-7

    center_theta = gridium.theta_ext
    gridium.theta_ext = center_theta + step
    h_plus = gridium.hamiltonian_primitive()
    gridium.theta_ext = center_theta - step
    h_minus = gridium.hamiltonian_primitive()
    gridium.theta_ext = center_theta
    finite_difference = (h_plus - h_minus) / (2 * step)
    assert _max_abs(finite_difference - gridium.dH_dtheta_ext()) < 2e-7


def test_bias_points_have_the_s23_symmetries_and_theta_shift():
    # At both phi_ext=0 and pi, simultaneous sigma/S parity is exact.
    for phi_ext in (0.0, np.pi):
        gridium = _gridium(phi_ext=phi_ext, theta_ext=np.pi)
        parity_sigma_s = qt.tensor(
            _parity(gridium.trunc_sigma),
            qt.qeye(gridium.trunc_delta),
            _parity(gridium.trunc_s)).to('csr')
        commutator = (
            gridium.hamiltonian_primitive() * parity_sigma_s
            - parity_sigma_s * gridium.hamiltonian_primitive())
        assert _max_abs(commutator) < 1e-10

    # At theta_ext=0, delta parity is exact.
    gridium = _gridium(phi_ext=0.0, theta_ext=0.0)
    parity_delta = qt.tensor(
        qt.qeye(gridium.trunc_sigma),
        _parity(gridium.trunc_delta),
        qt.qeye(gridium.trunc_s)).to('csr')
    commutator = (
        gridium.hamiltonian_primitive() * parity_delta
        - parity_delta * gridium.hamiltonian_primitive())
    assert _max_abs(commutator) < 1e-10

    # S23's theta_ext=pi point is a displaced quadratic potential, not a
    # parity symmetry about zero. Check the exact endpoint displacement.
    h_zero = gridium.hamiltonian_primitive()
    phi_delta = gridium.phi_delta()
    identity = qt.qeye(gridium.hilbert_dimension)
    identity.dims = phi_delta.dims
    gridium.theta_ext = np.pi
    expected_shift = (
        -np.pi * gridium.E_LK * phi_delta
        + 0.25 * np.pi ** 2 * gridium.E_LK * identity)
    assert _max_abs(
        gridium.hamiltonian_primitive() - h_zero - expected_shift) < 1e-10


def test_tensor_dimensions_order_and_cross_mode_commutation():
    gridium = _gridium(trunc_sigma=4, trunc_delta=5, trunc_s=6, nlev=4)
    expected_dims = [[4, 5, 6], [4, 5, 6]]
    for name in ('phi_sigma', 'phi_delta', 'phi_s'):
        assert gridium.operator(name).dims == expected_dims
        assert gridium.operator(name).shape == (120, 120)

    annihilation = qt.destroy(4)
    local_phi = (
        gridium.phase_scales[0] / np.sqrt(2)
        * (annihilation + annihilation.dag()))
    expected_sigma = qt.tensor(local_phi, qt.qeye(5), qt.qeye(6))
    assert _max_abs(gridium.phi_sigma() - expected_sigma) < 1e-14

    cross_commutator = (
        gridium.phi_sigma() * gridium.n_delta()
        - gridium.n_delta() * gridium.phi_sigma())
    assert _max_abs(cross_commutator) < 1e-14


def test_numerical_phase_centers_shift_phi_but_not_n():
    zero_centered = _gridium()
    centers = (0.2, np.pi / 2, -0.4)
    centered = _gridium(phase_centers=centers)
    identity = qt.qeye(centered.hilbert_dimension)
    identity.dims = centered.phi_sigma().dims

    for phi_name, center in zip(
            ('phi_sigma', 'phi_delta', 'phi_s'), centers):
        np.testing.assert_allclose(
            (centered.operator(phi_name)
             - zero_centered.operator(phi_name)).full(),
            (center * identity).full(), atol=1e-14)
    for n_name in ('n_sigma', 'n_delta', 'n_s'):
        np.testing.assert_allclose(
            centered.operator(n_name).full(),
            zero_centered.operator(n_name).full(), atol=1e-14)

    h_zero = centered.hamiltonian_primitive()
    centered.phase_centers = (0.0, 0.0, 0.0)
    assert centered.phase_centers == (0.0, 0.0, 0.0)
    assert _max_abs(centered.hamiltonian_primitive() - h_zero) > 1e-3


def test_numerical_centers_do_not_follow_physical_biases_implicitly():
    centers = (0.1, np.pi / 2, -0.2)
    gridium = _gridium(phase_centers=centers)
    gridium.theta_ext = 0.3
    gridium.phi_ext = -0.5
    assert gridium.phase_centers == centers


def test_common_two_pi_cell_shift_preserves_s23_matrix():
    reference = _gridium(
        phase_centers=(0.0, np.pi / 2, 0.0),
        trunc_sigma=4, trunc_delta=5, trunc_s=6, nlev=4)
    translated = _gridium(
        phase_centers=(2 * np.pi, np.pi / 2, 2 * np.pi),
        trunc_sigma=4, trunc_delta=5, trunc_s=6, nlev=4)

    assert _max_abs(
        reference.hamiltonian_primitive()
        - translated.hamiltonian_primitive()) < 1e-11
    assert _max_abs(reference.phi_2() - translated.phi_2()) < 1e-14


def test_derived_coordinate_identities():
    gridium = _gridium()
    identities = (
        gridium.phi_1() - gridium.phi_s(),
        gridium.phi_2() - (gridium.phi_s() - gridium.phi_sigma()),
        gridium.phi_3() - gridium.phi_delta(),
        gridium.n_1() - (gridium.n_s() + gridium.n_sigma()),
        gridium.n_2() + gridium.n_sigma(),
        gridium.n_3() - gridium.n_delta(),
    )
    assert all(_max_abs(residual) < 1e-14 for residual in identities)


def test_sparse_low_energy_api_and_matrix_elements():
    gridium = _gridium()
    energies, vectors = gridium.levels(nlev=6, eigvecs=True)

    assert energies.shape == (6,)
    assert len(vectors) == 6
    assert all(vector.dims == [[6, 6, 5], [1]] for vector in vectors)
    assert np.all(np.diff(energies) >= 0)
    assert gridium.level(0) == energies[0]
    assert gridium.freq(0, 1) == energies[1] - energies[0]
    np.testing.assert_allclose(
        gridium.transition_energies(nlev=6), energies - energies[0])

    for name in ('phi_2', 'n_1'):
        projected = gridium.operator(name, basis='energy', nlev=6)
        assert projected.shape == (6, 6)
        np.testing.assert_allclose(projected.full(), projected.full().conj().T)
        assert gridium.matrix_element(name, 0, 1) == projected[0, 1]


def test_progressively_larger_independent_cutoffs_improve_ground_energy():
    cutoffs = ((6, 6, 5), (7, 7, 6), (8, 8, 7))
    ground_energies = []
    for trunc_sigma, trunc_delta, trunc_s in cutoffs:
        gridium = _gridium(
            trunc_sigma=trunc_sigma,
            trunc_delta=trunc_delta,
            trunc_s=trunc_s,
            nlev=3)
        ground_energies.append(gridium.levels(nlev=3)[0])

    first_change, second_change = np.abs(np.diff(ground_energies))
    assert second_change < first_change


def test_same_basis_eigensystem_is_compatible_with_state_tracking():
    reference = _gridium(trunc_sigma=7, trunc_delta=7, trunc_s=6, nlev=5)
    shifted = _gridium(
        phi_ext=1e-3,
        trunc_sigma=7, trunc_delta=7, trunc_s=6, nlev=5)
    reference_energies, reference_vectors = reference.levels(eigvecs=True)
    shifted_energies, shifted_vectors = shifted.levels(eigvecs=True)

    tracking = track_eigensystems(
        reference_energies, reference_vectors,
        shifted_energies, shifted_vectors)

    np.testing.assert_array_equal(tracking.permutation, np.arange(5))
    assert np.min(tracking.matched_overlaps) > 0.999
    assert not tracking.has_unresolved_ambiguity
