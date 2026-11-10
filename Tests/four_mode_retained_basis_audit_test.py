"""Focused state-tracking tests for the four-mode retained-basis audit."""

import numpy as np

from Simulations.Rabi_3Photon import four_mode_retained_basis_audit as audit


def test_track_embeds_noncontiguous_retained_stage1_indices():
    # Full reference states occupy Stage-1 rows 0, 2, and 3.  The candidate
    # stores those rows compactly rather than at rows 0, 1, and 2.
    reference = np.zeros((4, 3), dtype=complex)
    reference[0, 0] = 1
    reference[2, 1] = 1
    reference[3, 2] = 1
    candidate = np.zeros((3, 3), dtype=complex)
    candidate[0, 0] = 1
    candidate[1, 1] = 1
    candidate[2, 2] = 1

    assignment, overlaps = audit._track(
        reference, candidate, retained_indices=np.array([0, 2, 3]),
        n4=1, nstates=3)

    np.testing.assert_array_equal(assignment, [0, 1, 2])
    np.testing.assert_allclose(overlaps, 1.0)


def test_track_assigns_reordered_candidate_states_by_overlap():
    reference = np.eye(3, dtype=complex)
    candidate = reference[:, [1, 0, 2]]

    assignment, overlaps = audit._track(
        reference, candidate, retained_indices=np.arange(3),
        n4=1, nstates=3)

    np.testing.assert_array_equal(assignment, [1, 0, 2])
    np.testing.assert_allclose(overlaps, 1.0)


def test_state_assignment_reorders_energies_vectors_and_operator_together():
    values = np.array([20.0, 10.0, 30.0])
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
    tensor = vectors.reshape(2, 1, 3)
    matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=complex)

    (ordered_values, ordered_vectors, ordered_tensor, ordered_matrix,
     transitions) = audit._apply_state_assignment(
        values, vectors, tensor, matrix, np.array([1, 0, 2]))

    np.testing.assert_array_equal(ordered_values, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(ordered_vectors, vectors[:, [1, 0, 2]])
    np.testing.assert_array_equal(ordered_tensor, tensor[..., [1, 0, 2]])
    np.testing.assert_array_equal(
        ordered_matrix, matrix[np.ix_([1, 0, 2], [1, 0, 2])])
    np.testing.assert_array_equal(transitions, [0.0, 10.0, 20.0])
