"""Overlap-based eigenstate matching and logical-basis gauge tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.optimize import linear_sum_assignment

__all__ = ['StateTrackingResult', 'track_eigensystems']


def _vector_array(vectors):
    columns = []
    for vector in vectors:
        if isinstance(vector, qt.Qobj):
            array = vector.full().reshape(-1)
        else:
            array = np.asarray(vector, dtype=complex).reshape(-1)
        norm = np.linalg.norm(array)
        if np.isclose(norm, 0):
            raise ValueError('Eigenvectors must have nonzero norm.')
        columns.append(array / norm)
    if not columns:
        raise ValueError('At least one eigenvector is required.')
    dimension = max(len(column) for column in columns)
    padded = np.zeros((dimension, len(columns)), dtype=complex)
    for index, column in enumerate(columns):
        padded[:len(column), index] = column
    return padded


def _common_overlap(reference_vectors, new_vectors):
    common_dimension = min(reference_vectors.shape[0], new_vectors.shape[0])
    return (
        reference_vectors[:common_dimension].conj().T
        @ new_vectors[:common_dimension])


def _degenerate_groups(energies, atol, rtol):
    groups = []
    start = 0
    for index in range(len(energies) - 1):
        scale = max(abs(energies[index]), abs(energies[index + 1]), 1.0)
        tolerance = atol + rtol * scale
        if abs(energies[index + 1] - energies[index]) > tolerance:
            if index + 1 - start > 1:
                groups.append(tuple(range(start, index + 1)))
            start = index + 1
    if len(energies) - start > 1:
        groups.append(tuple(range(start, len(energies))))
    return groups


@dataclass
class StateTrackingResult:
    """State assignment, gauge transform, and explicit ambiguity records."""

    permutation: np.ndarray
    basis_transform: np.ndarray
    aligned_vectors: np.ndarray
    overlap_matrix: np.ndarray
    matched_overlaps: np.ndarray
    ambiguities: list
    degenerate_groups: list

    @property
    def has_ambiguity(self):
        return bool(self.ambiguities)

    @property
    def has_unresolved_ambiguity(self):
        return any(not item.get('resolved', False) for item in self.ambiguities)

    def transform_operator(self, operator):
        """Express an operator from the new basis in the tracked basis."""
        is_qobj = isinstance(operator, qt.Qobj)
        array = operator.full() if is_qobj else np.asarray(operator)
        transformed = (
            self.basis_transform.conj().T
            @ array @ self.basis_transform)
        return qt.Qobj(transformed) if is_qobj else transformed


def track_eigensystems(
        reference_energies, reference_eigenvectors,
        new_energies, new_eigenvectors, *,
        minimum_overlap=0.5, ambiguity_margin=0.05,
        degeneracy_atol=1e-6, degeneracy_rtol=1e-9):
    """Match and align a new eigensystem to a reference eigensystem.

    States are globally assigned by maximizing total squared overlap. Isolated
    states receive a phase correction. A group that is near-degenerate in
    both eigensystems is aligned as a subspace with a unitary Procrustes
    rotation, which is stable under arbitrary rotations inside the group.

    Ambiguities are returned rather than raised. ``minimum_overlap`` and
    ``ambiguity_margin`` apply to squared overlaps. Energy tolerances are in
    the units supplied by the caller.
    """
    reference_energies = np.asarray(reference_energies, dtype=float)
    new_energies = np.asarray(new_energies, dtype=float)
    count = len(reference_energies)
    if count == 0 or len(new_energies) != count:
        raise ValueError('The eigensystems must contain the same state count.')
    if len(reference_eigenvectors) != count or len(new_eigenvectors) != count:
        raise ValueError('Each energy must have one eigenvector.')

    reference_vectors = _vector_array(reference_eigenvectors)
    new_vectors = _vector_array(new_eigenvectors)
    overlap = _common_overlap(reference_vectors, new_vectors)
    overlap_probability = np.abs(overlap) ** 2
    reference_indices, new_indices = linear_sum_assignment(
        -overlap_probability)
    permutation = np.empty(count, dtype=int)
    permutation[reference_indices] = new_indices

    permutation_matrix = np.zeros((count, count), dtype=complex)
    permutation_matrix[permutation, np.arange(count)] = 1.0
    matched_vectors = new_vectors @ permutation_matrix
    matched_overlap = _common_overlap(reference_vectors, matched_vectors)

    ambiguities = []
    matched_scores = overlap_probability[
        np.arange(count), permutation]
    reference_groups = _degenerate_groups(
        reference_energies, degeneracy_atol, degeneracy_rtol)
    group_members = {index for group in reference_groups for index in group}

    for index, assigned in enumerate(permutation):
        alternatives = np.delete(overlap_probability[index], assigned)
        competitor = float(np.max(alternatives)) if alternatives.size else 0.0
        if matched_scores[index] < minimum_overlap:
            ambiguities.append({
                'kind': 'low_overlap',
                'reference_state': index,
                'new_state': int(assigned),
                'overlap': float(matched_scores[index]),
                'resolved': False,
            })
        if (matched_scores[index] - competitor < ambiguity_margin
                and index not in group_members):
            ambiguities.append({
                'kind': 'competing_match',
                'reference_state': index,
                'new_state': int(assigned),
                'overlap': float(matched_scores[index]),
                'competitor_overlap': competitor,
                'resolved': False,
            })

    gauge = np.eye(count, dtype=complex)
    grouped = set()
    degenerate_reports = []
    for group in reference_groups:
        group_array = np.asarray(group)
        matched_energies = new_energies[permutation[group_array]]
        new_span = float(np.max(matched_energies) - np.min(matched_energies))
        scale = max(float(np.max(np.abs(matched_energies))), 1.0)
        tolerance = degeneracy_atol + degeneracy_rtol * scale
        if new_span > tolerance:
            ambiguities.append({
                'kind': 'split_degenerate_group',
                'reference_states': group,
                'new_states': tuple(permutation[group_array]),
                'new_energy_span': new_span,
                'resolved': False,
            })
            continue

        block = matched_overlap[np.ix_(group_array, group_array)]
        left, singular_values, right_dag = np.linalg.svd(block)
        rotation = right_dag.conj().T @ left.conj().T
        gauge[np.ix_(group_array, group_array)] = rotation
        grouped.update(group)
        minimum_subspace_overlap = float(np.min(singular_values) ** 2)
        report = {
            'kind': 'near_degenerate_subspace',
            'reference_states': group,
            'new_states': tuple(permutation[group_array]),
            'minimum_subspace_overlap': minimum_subspace_overlap,
            'resolved': minimum_subspace_overlap >= minimum_overlap,
        }
        degenerate_reports.append(report)
        ambiguities.append(report)

    for index in range(count):
        if index in grouped:
            continue
        phase = matched_overlap[index, index]
        if not np.isclose(abs(phase), 0):
            gauge[index, index] = np.conj(phase) / abs(phase)

    basis_transform = permutation_matrix @ gauge
    aligned_vectors = new_vectors @ basis_transform
    aligned_overlap = _common_overlap(reference_vectors, aligned_vectors)
    return StateTrackingResult(
        permutation=permutation,
        basis_transform=basis_transform,
        aligned_vectors=aligned_vectors,
        overlap_matrix=overlap,
        matched_overlaps=np.abs(np.diag(aligned_overlap)) ** 2,
        ambiguities=ambiguities,
        degenerate_groups=degenerate_reports)
