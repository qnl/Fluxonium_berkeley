"""Oracle-only N=71 retained-subspace post-processing; never solves Stage 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from Simulations.Rabi_3Photon.four_mode_asymmetric_shell_audit import (
    LOGICAL_COUPLINGS,
    LOW_INDICES,
    PRIMARY_COUPLINGS,
    SHELL_INDICES,
    _act,
    _components,
    _evaluate,
    _reference,
    _strip_private,
)
from Simulations.Rabi_3Photon.run_four_mode_spatial71_reference import (
    N51_SPATIAL_CUTOFFS,
    SPATIAL_CUTOFFS,
    _linear_interpolation_matrix,
    load_verified_n51_checkpoint,
    validate_complete_checkpoint_directory,
)


ROOT = Path(__file__).resolve().parents[2]
N51_DIR = ROOT / "research/checkpoints/2026-09-26-asymmetric-n51"
N71_DIR = ROOT / "research/checkpoints/four-mode-asymmetric-n71-k180"
OUTPUT = ROOT / "research/sandbox/n71_retained_subspace_postprocess.json"
DIMENSIONS = (170, 172, 175, 177, 179)
REQUESTED_LOGICAL = (
    (0, 3), (0, 5), (0, 10),
    (1, 2), (1, 4), (1, 6), (1, 8), (1, 9),
)


def load_n71():
    validate_complete_checkpoint_directory(N71_DIR)
    with np.load(N71_DIR / "stage1_eigensystem.npz") as saved:
        values = np.asarray(saved["eigenvalues_GHz"])
        vectors = np.asarray(saved["eigenvectors"])
    with np.load(N71_DIR / "projected_operators.npz") as saved:
        operators = {name: np.asarray(saved[name]) for name in saved.files}
    return values, vectors, operators


def coupling_rows(reference, candidate, pairs):
    rows = []
    for initial, final in pairs:
        ref = float(abs(reference[initial, final]))
        value = float(abs(candidate[initial, final]))
        error = abs(value - ref)
        rows.append({
            "transition": [initial, final],
            "reference": ref,
            "candidate": value,
            "absolute_error": error,
            "relative_error": error / ref if ref > 1e-12 else None,
        })
    return rows


def tracking_confidence(case, reference):
    indices = np.asarray(case["retained_indices"], dtype=int)
    candidate = case["_vectors"].reshape(len(indices) * 8, 11)
    embedded = np.zeros((180 * 8, 11), dtype=complex)
    embedded.reshape(180, 8, 11)[indices] = candidate.reshape(
        len(indices), 8, 11)
    overlap = np.abs(embedded.conj().T @ reference["vectors"])
    diagonal = np.diag(overlap)
    competitors = overlap.copy()
    np.fill_diagonal(competitors, -np.inf)
    best = np.max(competitors, axis=0)
    margins = diagonal - best
    ratios = best / diagonal
    return {
        "minimum_assigned_overlap": float(np.min(diagonal)),
        "minimum_overlap_margin": float(np.min(margins)),
        "maximum_competitor_ratio": float(np.max(ratios)),
        "ambiguous": bool(
            np.any(diagonal < 0.9)
            or np.any(margins <= 0.25)
            or np.any(ratios >= 0.5)
        ),
    }


def summarize_case(case, reference):
    primary = coupling_rows(
        reference["matrix"], case["_matrix"], PRIMARY_COUPLINGS)
    logical = coupling_rows(
        reference["matrix"], case["_matrix"], REQUESTED_LOGICAL)
    top_path = case["pathways"][0]
    return {
        **_strip_private(case),
        "primary_path_coupling_errors": primary,
        "audited_logical_leakage_coupling_errors": logical,
        "maximum_primary_absolute_error": max(
            row["absolute_error"] for row in primary),
        "maximum_primary_relative_error": max(
            row["relative_error"] for row in primary),
        "maximum_audited_absolute_error": max(
            row["absolute_error"] for row in logical),
        "maximum_audited_relative_error": max(
            row["relative_error"] for row in logical),
        "dominant_path_is_0_7_8_1": top_path["path"] == [0, 7, 8, 1],
        "tracking_confidence": tracking_confidence(case, reference),
    }


def shell_diagnostics(label, components, reference):
    print(f"{label}: evaluating core-160", flush=True)
    base = _evaluate(LOW_INDICES, components, reference)
    base_tensor = base["_tensor"]
    base_residual = (
        _act(
            components["charge"][np.ix_(SHELL_INDICES, LOW_INDICES)],
            components["p4"], base_tensor)
        + _act(
            components["phase"][np.ix_(SHELL_INDICES, LOW_INDICES)],
            components["x4"], base_tensor)
    )
    weights = np.sum(
        np.abs(reference["tensor"][SHELL_INDICES]) ** 2, axis=1)
    rows = []
    for position, rank in enumerate(SHELL_INDICES):
        print(f"{label}: leave-one-out {rank}", flush=True)
        case = _evaluate(
            np.delete(np.arange(180, dtype=int), rank), components, reference)
        primary = coupling_rows(
            reference["matrix"], case["_matrix"], PRIMARY_COUPLINGS)
        logical = coupling_rows(
            reference["matrix"], case["_matrix"], REQUESTED_LOGICAL)
        coupling_block = (
            np.kron(
                components["charge"][rank:rank + 1, LOW_INDICES],
                components["p4"])
            + np.kron(
                components["phase"][rank:rank + 1, LOW_INDICES],
                components["x4"])
        )
        bare = components["w1"][rank] + components["w4"]
        denominator = float(np.min(np.abs(
            bare[:, None] - reference["values"][None, :])))
        rows.append({
            "stage1_rank": int(rank),
            "stage1_energy_GHz": float(components["w1"][rank]),
            "adjacent_lower_spacing_GHz": float(
                components["w1"][rank] - components["w1"][rank - 1]),
            "minimum_bare_denominator_GHz": denominator,
            "coupling_block_frobenius_GHz": float(
                np.linalg.norm(coupling_block)),
            "base160_residual_max_GHz": float(np.max(np.linalg.norm(
                base_residual[position], axis=0))),
            "base160_residual_by_final_state_GHz": np.linalg.norm(
                base_residual[position], axis=0).tolist(),
            "full_low_energy_weight_sum": float(np.sum(weights[position])),
            "full_low_energy_weight_max": float(np.max(weights[position])),
            "full_low_energy_weight_by_state": weights[position].tolist(),
            "leave_one_out_max_transition_error_MHz": case[
                "maximum_transition_error_MHz"],
            "leave_one_out_transition_errors_MHz": case[
                "transition_errors_MHz"],
            "leave_one_out_minimum_overlap": case["minimum_state_overlap"],
            "leave_one_out_state_overlaps": case["state_overlaps"],
            "leave_one_out_primary_errors": primary,
            "leave_one_out_logical_errors": logical,
            "leave_one_out_max_primary_abs_error": max(
                row["absolute_error"] for row in primary),
            "leave_one_out_max_logical_abs_error": max(
                row["absolute_error"] for row in logical),
            "leave_one_out_max_omitted_residual_GHz": case[
                "maximum_omitted_residual_GHz"],
            "leave_one_out_dominant_path": case["pathways"][0],
        })
    rows.sort(key=lambda row: (
        row["leave_one_out_max_transition_error_MHz"],
        row["leave_one_out_max_primary_abs_error"],
        row["leave_one_out_max_logical_abs_error"],
        row["base160_residual_max_GHz"],
    ), reverse=True)
    for priority, row in enumerate(rows, 1):
        row["priority_rank"] = priority
    return summarize_case(base, reference), rows


def evaluate_priority(label, priority, components, reference):
    cases = {}
    for dimension in DIMENSIONS:
        extras = sorted(priority[:dimension - 160])
        indices = np.asarray(list(range(160)) + extras, dtype=int)
        print(f"{label}: candidate k{dimension} {extras}", flush=True)
        cases[f"k{dimension}"] = summarize_case(
            _evaluate(indices, components, reference), reference)
    return cases


def interpolate_stage1_shell(vectors51, vectors71):
    source = N51_SPATIAL_CUTOFFS
    target = SPATIAL_CUTOFFS
    n1_count = 2 * source["n1max"] + 1
    x2s = np.linspace(-source["L2"], source["L2"], source["N2"])
    x3s = np.linspace(-source["L3"], source["L3"], source["N3"])
    x2t = np.linspace(-target["L2"], target["L2"], target["N2"])
    x3t = np.linspace(-target["L3"], target["L3"], target["N3"])
    interp2 = _linear_interpolation_matrix(x2s, x2t)
    interp3 = _linear_interpolation_matrix(x3s, x3t)
    dxs = (x2s[1] - x2s[0]) * (x3s[1] - x3s[0])
    dxt = (x2t[1] - x2t[0]) * (x3t[1] - x3t[0])
    shell = vectors51[:, SHELL_INDICES].reshape(
        source["N3"], source["N2"], n1_count, 20) / np.sqrt(dxs)
    along3 = np.tensordot(interp3, shell, axes=(1, 0))
    interpolated = np.einsum(
        "Bb,Ab...->AB...", interp2, along3, optimize=True) * np.sqrt(dxt)
    interpolated = interpolated.reshape(-1, 20)
    interpolated /= np.linalg.norm(interpolated, axis=0)[None, :]
    overlap = np.abs(vectors71.conj().T @ interpolated)
    rows, columns = linear_sum_assignment(-overlap)
    assigned = np.empty(20, dtype=int)
    assigned[columns] = rows
    result = []
    for position, rank51 in enumerate(SHELL_INDICES):
        rank71 = int(assigned[position])
        competitors = overlap[:, position].copy()
        competitors[rank71] = -np.inf
        competitor = int(np.argmax(competitors))
        result.append({
            "n51_rank": int(rank51),
            "assigned_n71_rank": rank71,
            "assigned_overlap": float(overlap[rank71, position]),
            "strongest_competing_n71_rank": competitor,
            "strongest_competing_overlap": float(overlap[competitor, position]),
            "raw_best_n71_rank": int(np.argmax(overlap[:, position])),
            "raw_best_overlap": float(np.max(overlap[:, position])),
        })
    return result


def common_priority(rows51, rows71):
    rank51 = {row["stage1_rank"]: row["priority_rank"] for row in rows51}
    rank71 = {row["stage1_rank"]: row["priority_rank"] for row in rows71}
    priority = sorted(
        range(160, 180),
        key=lambda rank: (
            min(rank51[rank], rank71[rank]),
            max(rank51[rank], rank71[rank]),
            rank51[rank] + rank71[rank],
            rank,
        ),
    )
    return priority


def main():
    n51 = load_verified_n51_checkpoint(N51_DIR)
    values71, vectors71, operators71 = load_n71()
    components51 = _components(n51["values"], n51["operators"])
    components71 = _components(values71, operators71)
    print("reconstructing full Stage-2 references", flush=True)
    reference51 = _reference(components51)
    reference71 = _reference(components71)

    base51, rows51 = shell_diagnostics("N51", components51, reference51)
    base71, rows71 = shell_diagnostics("N71", components71, reference71)
    priority51 = [row["stage1_rank"] for row in rows51]
    priority71 = [row["stage1_rank"] for row in rows71]
    shared_priority = common_priority(rows51, rows71)

    candidates = {
        "n71_priority_on_n71": evaluate_priority(
            "N71-priority/N71", priority71, components71, reference71),
        "n71_priority_on_n51": evaluate_priority(
            "N71-priority/N51", priority71, components51, reference51),
        "common_priority_on_n71": evaluate_priority(
            "common/N71", shared_priority, components71, reference71),
        "common_priority_on_n51": evaluate_priority(
            "common/N51", shared_priority, components51, reference51),
    }

    result = {
        "method": {
            "shell_ranking": (
                "descending lexicographic leave-one-out maximum transition "
                "error, primary-path absolute error, audited logical/leakage "
                "absolute error, then core-160 omitted residual"),
            "common_priority": (
                "ascending best priority rank on either grid, then worst "
                "priority rank, rank sum, and Stage-1 rank"),
            "oracle_informed": True,
            "stage1_eigensolve_run": False,
        },
        "n51_core160": base51,
        "n71_core160": base71,
        "n51_shell_ranking": rows51,
        "n71_shell_ranking": rows71,
        "n51_priority": priority51,
        "n71_priority": priority71,
        "common_priority": shared_priority,
        "stage1_physical_character_mapping": interpolate_stage1_shell(
            n51["vectors"], vectors71),
        "candidates": candidates,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
