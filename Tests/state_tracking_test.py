"""Narrow tests for overlap-based eigenstate and gauge tracking."""

import numpy as np

from Circuit_Objs.qchard_statetracking import track_eigensystems


def _basis(dimension):
    return [np.eye(dimension, dtype=complex)[:, index]
            for index in range(dimension)]


def test_tracks_simple_sign_flips():
    reference = _basis(3)
    new = [reference[0], -reference[1], reference[2]]

    result = track_eigensystems([0, 1, 2], reference, [0, 1, 2], new)

    np.testing.assert_allclose(result.aligned_vectors, np.eye(3))
    np.testing.assert_array_equal(result.permutation, [0, 1, 2])
    assert not result.has_unresolved_ambiguity


def test_tracks_reordered_states():
    reference = _basis(3)
    new = [reference[2], reference[0], reference[1]]

    result = track_eigensystems(
        [0, 1, 2], reference, [2, 0, 1], new)

    np.testing.assert_array_equal(result.permutation, [1, 2, 0])
    np.testing.assert_allclose(result.aligned_vectors, np.eye(3))


def test_tracks_arbitrary_complex_phases():
    reference = _basis(3)
    phases = np.exp(1j * np.array([0.3, -1.2, 2.4]))
    new = [phase * vector for phase, vector in zip(phases, reference)]

    result = track_eigensystems([0, 1, 2], reference, [0, 1, 2], new)

    np.testing.assert_allclose(result.aligned_vectors, np.eye(3), atol=1e-14)


def test_aligns_rotated_near_degenerate_pair_as_subspace():
    reference = _basis(3)
    rotation = np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2)
    new_matrix = np.eye(3, dtype=complex)
    new_matrix[:2, :2] = rotation
    new = [new_matrix[:, index] for index in range(3)]

    result = track_eigensystems(
        [0, 1e-7, 1], reference, [0, 2e-7, 1], new,
        degeneracy_atol=1e-6)

    np.testing.assert_allclose(result.aligned_vectors, np.eye(3), atol=1e-14)
    assert result.has_ambiguity
    assert not result.has_unresolved_ambiguity
    assert result.degenerate_groups[0]['reference_states'] == (0, 1)


def test_reports_ambiguous_nondegenerate_matches():
    reference = _basis(2)
    new = [
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([-1, 1], dtype=complex) / np.sqrt(2),
    ]

    result = track_eigensystems(
        [0, 1], reference, [0, 1], new,
        ambiguity_margin=0.05, degeneracy_atol=1e-9)

    assert result.has_unresolved_ambiguity
    assert any(item['kind'] == 'competing_match'
               for item in result.ambiguities)
