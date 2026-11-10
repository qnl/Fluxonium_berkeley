"""Checkpoint-only diagnostics for asymmetric Gridium states 0, 1, 7, 8.

This script authenticates the saved N=51 and N=71 parents, reconstructs only
the 1440-dimensional Stage-2 problem from their saved Stage-1 eigensystems and
projected operators, and never assembles or solves the large spatial
Hamiltonian.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from Simulations.Rabi_3Photon.four_mode_asymmetric_shell_audit import (
    _components,
    _reference,
)
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import (
    _three_step_paths,
)
from Simulations.Rabi_3Photon.run_four_mode_spatial71_reference import (
    N51_ORACLE_SHA256,
    load_verified_n51_checkpoint,
    validate_complete_checkpoint_directory,
)


ROOT = Path(__file__).resolve().parents[2]
N51_DIR = ROOT / "research/checkpoints/2026-09-26-asymmetric-n51"
N71_DIR = ROOT / "research/checkpoints/four-mode-asymmetric-n71-k180"
OUTPUT = ROOT / "research/sandbox/state_0178_checkpoint_diagnostics.json"
SELECTED_STATES = (0, 1, 7, 8)
PATH_EDGES = ((0, 7), (7, 8), (8, 1))


def _as_float(value):
    return float(np.real_if_close(value))


def _relative_change(candidate, reference):
    absolute = float(candidate - reference)
    return {
        "n51": float(reference),
        "n71": float(candidate),
        "signed_change": absolute,
        "absolute_change": abs(absolute),
        "relative_change": (
            absolute / abs(float(reference))
            if abs(float(reference)) > 1e-14 else None
        ),
    }


def _expect_product(sector_operator, mode_operator, state_tensor):
    applied = np.einsum(
        "ab,mn,bn->am",
        sector_operator,
        mode_operator,
        state_tensor,
        optimize=True,
    )
    return _as_float(np.vdot(state_tensor, applied))


def _moment(values, probabilities):
    mean = float(np.dot(values, probabilities))
    second = float(np.dot(values * values, probabilities))
    variance = max(0.0, second - mean * mean)
    return {
        "mean": mean,
        "second_moment": second,
        "standard_deviation": float(np.sqrt(variance)),
        "rms": float(np.sqrt(max(0.0, second))),
    }


def _top_rows(values, count=6):
    order = np.argsort(values)[::-1][:count]
    return [
        {"index": int(index), "probability": float(values[index])}
        for index in order
    ]


def _raw_state_metrics(stage1_vectors, state_tensor, spatial):
    n1_values = np.arange(-spatial["n1max"], spatial["n1max"] + 1)
    theta2 = np.linspace(-spatial["L2"], spatial["L2"], spatial["N2"])
    theta3 = np.linspace(-spatial["L3"], spatial["L3"], spatial["N3"])
    raw = stage1_vectors @ state_tensor
    shaped = raw.reshape(
        spatial["N3"], spatial["N2"], len(n1_values), spatial["N4"]
    )
    probability = np.abs(shaped) ** 2
    norm = float(np.sum(probability))
    probability /= norm

    marginal_n1 = probability.sum(axis=(0, 1, 3))
    marginal_theta2 = probability.sum(axis=(0, 2, 3))
    marginal_theta3 = probability.sum(axis=(1, 2, 3))
    marginal_2d = probability.sum(axis=(2, 3))
    marginal_mode4 = probability.sum(axis=(0, 1, 2))

    theta2_metrics = _moment(theta2, marginal_theta2)
    theta3_metrics = _moment(theta3, marginal_theta3)
    theta3_grid, theta2_grid = np.meshgrid(theta3, theta2, indexing="ij")
    grid_phi = theta3_grid + 0.5 * theta2_grid
    grid_phi_mean = float(np.sum(marginal_2d * grid_phi))
    grid_phi_second = float(np.sum(marginal_2d * grid_phi**2))
    covariance = float(np.sum(
        marginal_2d
        * (theta2_grid - theta2_metrics["mean"])
        * (theta3_grid - theta3_metrics["mean"])
    ))

    dx2 = float(theta2[1] - theta2[0])
    dx3 = float(theta3[1] - theta3[0])
    participation_area = float(
        dx2 * dx3 / np.sum(marginal_2d * marginal_2d)
    )
    return {
        "reconstructed_norm_before_normalization": norm,
        "compact_charge_n1": {
            **_moment(n1_values, marginal_n1),
            "probability_distribution": {
                str(int(value)): float(probability_value)
                for value, probability_value in zip(n1_values, marginal_n1)
            },
            "probability_abs_n1_ge_3": float(np.sum(
                marginal_n1[np.abs(n1_values) >= 3]
            )),
            "probability_at_charge_cutoff_abs_n1_eq_4": float(np.sum(
                marginal_n1[np.abs(n1_values) == spatial["n1max"]]
            )),
        },
        "theta2": {
            **theta2_metrics,
            "probability_abs_theta2_ge_0p8L2": float(np.sum(
                marginal_theta2[np.abs(theta2) >= 0.8 * spatial["L2"]]
            )),
            "probability_in_outer_two_grid_points": float(
                np.sum(marginal_theta2[:2]) + np.sum(marginal_theta2[-2:])
            ),
        },
        "theta3": {
            **theta3_metrics,
            "probability_abs_theta3_ge_0p8L3": float(np.sum(
                marginal_theta3[np.abs(theta3) >= 0.8 * spatial["L3"]]
            )),
            "probability_in_outer_two_grid_points": float(
                np.sum(marginal_theta3[:2]) + np.sum(marginal_theta3[-2:])
            ),
        },
        "theta2_theta3_covariance": covariance,
        "grid_phi_coordinate": {
            "mean": grid_phi_mean,
            "second_moment": grid_phi_second,
            "standard_deviation": float(np.sqrt(max(
                0.0, grid_phi_second - grid_phi_mean**2
            ))),
        },
        "theta2_theta3_participation_area": participation_area,
        "mode4_probability_from_raw_state": marginal_mode4.tolist(),
    }


def _state_metrics(state, reference, components, stage1_vectors, spatial):
    tensor = reference["tensor"][:, :, state]
    probability = np.abs(tensor) ** 2
    stage1_probability = probability.sum(axis=1)
    mode4_probability = probability.sum(axis=0)
    stage1_ranks = np.arange(len(stage1_probability))
    mode4_levels = np.arange(len(mode4_probability))

    rho4 = tensor.conj().T @ tensor
    schmidt_weights = np.linalg.eigvalsh(rho4)
    schmidt_weights = np.clip(np.real(schmidt_weights), 0.0, None)
    schmidt_weights /= np.sum(schmidt_weights)
    nonzero = schmidt_weights[schmidt_weights > 1e-15]
    purity = float(np.sum(schmidt_weights**2))

    energy_components = {
        "stage1_sector_GHz": float(np.dot(
            components["w1"], stage1_probability
        )),
        "bare_mode4_GHz": float(np.dot(
            components["w4"], mode4_probability
        )),
        "charge_mode4_coupling_GHz": _expect_product(
            components["charge"], components["p4"], tensor
        ),
        "phase_mode4_coupling_GHz": _expect_product(
            components["phase"], components["x4"], tensor
        ),
    }
    energy_components["sum_GHz"] = float(sum(energy_components.values()))

    values = reference["values"]
    nearby = sorted(
        (
            {
                "state": int(other),
                "absolute_spacing_GHz": float(abs(values[other] - values[state])),
                "signed_spacing_GHz": float(values[other] - values[state]),
            }
            for other in range(len(values)) if other != state
        ),
        key=lambda row: row["absolute_spacing_GHz"],
    )
    coupling_rows = sorted(
        (
            {
                "state": int(other),
                "abs_grid_phi": float(abs(reference["matrix"][state, other])),
                "transition_frequency_GHz": float(abs(
                    values[other] - values[state]
                )),
            }
            for other in range(len(values)) if other != state
        ),
        key=lambda row: row["abs_grid_phi"],
        reverse=True,
    )
    raw = _raw_state_metrics(stage1_vectors, tensor, spatial)
    return {
        "absolute_energy_GHz": float(values[state]),
        "transition_from_ground_GHz": float(values[state] - values[0]),
        "nearest_spectral_neighbors": nearby[:4],
        "minimum_spacing_within_low11_GHz": nearby[0]["absolute_spacing_GHz"],
        "dominant_grid_phi_couplings": coupling_rows[:6],
        "grid_phi_diagonal": _as_float(reference["matrix"][state, state]),
        "mode4": {
            "bare_frequency_GHz": float(components["w4"][1] - components["w4"][0]),
            "mean_HO_occupation": float(np.dot(
                mode4_levels, mode4_probability
            )),
            "HO_occupation_second_moment": float(np.dot(
                mode4_levels**2, mode4_probability
            )),
            "probability_distribution": mode4_probability.tolist(),
            "probability_n4_ge_1": float(np.sum(mode4_probability[1:])),
            "probability_n4_ge_2": float(np.sum(mode4_probability[2:])),
            "probability_at_n4_cutoff_7": float(mode4_probability[-1]),
        },
        "stage1_parent_participation": {
            "mean_rank": float(np.dot(stage1_ranks, stage1_probability)),
            "rank_standard_deviation": float(np.sqrt(max(
                0.0,
                np.dot(stage1_ranks**2, stage1_probability)
                - np.dot(stage1_ranks, stage1_probability) ** 2,
            ))),
            "effective_rank_count": float(
                1.0 / np.sum(stage1_probability**2)
            ),
            "probability_ranks_160_to_179": float(np.sum(
                stage1_probability[160:]
            )),
            "probability_ranks_170_to_179": float(np.sum(
                stage1_probability[170:]
            )),
            "top_stage1_ranks": [
                {
                    **row,
                    "stage1_energy_GHz": float(
                        components["w1"][row["index"]]
                    ),
                }
                for row in _top_rows(stage1_probability)
            ],
        },
        "stage1_mode4_entanglement": {
            "reduced_mode4_purity": purity,
            "Schmidt_number": float(1.0 / purity),
            "von_Neumann_entropy_bits": float(-np.sum(
                nonzero * np.log2(nonzero)
            )),
            "Schmidt_weights_descending": sorted(
                schmidt_weights.tolist(), reverse=True
            ),
        },
        "energy_decomposition": energy_components,
        "raw_basis": raw,
    }


def _load_n71():
    metadata = validate_complete_checkpoint_directory(N71_DIR)
    with np.load(N71_DIR / "stage1_eigensystem.npz") as saved:
        values = np.asarray(saved["eigenvalues_GHz"])
        vectors = np.asarray(saved["eigenvectors"])
    with np.load(N71_DIR / "projected_operators.npz") as saved:
        operators = {name: np.asarray(saved[name]) for name in saved.files}
    return metadata, values, vectors, operators


def _build_grid_result(label, values, vectors, operators, spatial):
    started = time.perf_counter()
    components = _components(values, operators)
    reference = _reference(components)
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "spatial_cutoffs": spatial,
        "stage2_reconstruction_seconds": elapsed,
        "stage2_dimension": int(len(values) * spatial["N4"]),
        "mode4_bare_frequency_GHz": float(
            components["w4"][1] - components["w4"][0]
        ),
        "states": {
            str(state): _state_metrics(
                state, reference, components, vectors, spatial
            )
            for state in SELECTED_STATES
        },
        "path_edges": {
            f"{initial}-{final}": float(abs(
                reference["matrix"][initial, final]
            ))
            for initial, final in PATH_EDGES
        },
        "top_five_pathways": _three_step_paths(
            reference["matrix"], len(reference["values"]), count=5
        ),
        "_reference": reference,
    }


def _reproduction_checks(n51_result, n71_result, saved_analysis):
    cross = saved_analysis["n51_to_n71_spatial_comparison"]
    saved_n71 = saved_analysis["n71_reference_k180"]
    reference51 = n51_result["_reference"]
    reference71 = n71_result["_reference"]
    return {
        "n51_max_transition_difference_GHz": float(np.max(np.abs(
            reference51["transitions"][1:]
            - np.asarray(cross["n51_transitions_GHz"])
        ))),
        "n71_max_transition_difference_GHz": float(np.max(np.abs(
            reference71["transitions"][1:]
            - np.asarray(saved_n71["transitions_GHz"])
        ))),
        "n51_max_abs_grid_phi_difference": float(np.max(np.abs(
            np.abs(reference51["matrix"])
            - np.asarray(cross["n51_grid_phi_abs"])
        ))),
        "n71_max_abs_grid_phi_difference": float(np.max(np.abs(
            np.abs(reference71["matrix"])
            - np.asarray(saved_n71["grid_phi_abs"])
        ))),
    }


def _comparison(n51_result, n71_result, saved_analysis):
    cross = saved_analysis["n51_to_n71_spatial_comparison"]
    assignment = cross["n71_energy_row_for_n51_state"]
    scalar_paths = {
        "transition_from_ground_GHz": ("transition_from_ground_GHz",),
        "minimum_spacing_within_low11_GHz": (
            "minimum_spacing_within_low11_GHz",
        ),
        "mean_mode4_occupation": ("mode4", "mean_HO_occupation"),
        "mode4_probability_ge_1": ("mode4", "probability_n4_ge_1"),
        "stage1_mode4_entropy_bits": (
            "stage1_mode4_entanglement", "von_Neumann_entropy_bits",
        ),
        "stage1_high_rank_probability_160_179": (
            "stage1_parent_participation", "probability_ranks_160_to_179",
        ),
        "n1_rms": ("raw_basis", "compact_charge_n1", "rms"),
        "n1_cutoff_probability": (
            "raw_basis", "compact_charge_n1",
            "probability_at_charge_cutoff_abs_n1_eq_4",
        ),
        "theta2_std": ("raw_basis", "theta2", "standard_deviation"),
        "theta3_std": ("raw_basis", "theta3", "standard_deviation"),
        "grid_phi_std": (
            "raw_basis", "grid_phi_coordinate", "standard_deviation",
        ),
        "theta2_theta3_participation_area": (
            "raw_basis", "theta2_theta3_participation_area",
        ),
    }

    def get(row, path):
        for name in path:
            row = row[name]
        return row

    state_rows = {}
    for state in SELECTED_STATES:
        state71 = int(assignment[state])
        row51 = n51_result["states"][str(state)]
        row71 = n71_result["states"][str(state71)]
        state_rows[str(state)] = {
            "tracked_n71_state": state71,
            "cross_grid_overlap": float(cross["matched_state_overlaps"][state]),
            "diagnostic_changes": {
                name: _relative_change(get(row71, path), get(row51, path))
                for name, path in scalar_paths.items()
            },
        }
    edge_rows = {}
    matrix51 = n51_result["_reference"]["matrix"]
    matrix71 = n71_result["_reference"]["matrix"]
    for initial, final in PATH_EDGES:
        initial71, final71 = assignment[initial], assignment[final]
        edge_rows[f"{initial}-{final}"] = _relative_change(
            abs(matrix71[initial71, final71]),
            abs(matrix51[initial, final]),
        )
    splitting51 = (
        n51_result["_reference"]["values"][8]
        - n51_result["_reference"]["values"][7]
    )
    splitting71 = (
        n71_result["_reference"]["values"][assignment[8]]
        - n71_result["_reference"]["values"][assignment[7]]
    )
    pathways51 = n51_result["top_five_pathways"]
    pathways71 = saved_analysis[
        "n51_to_n71_spatial_comparison"
    ]["n71_ranked_paths_in_n51_tracked_labels"]
    return {
        "tracking_assignment_n71_row_for_n51_state": assignment,
        "state_reordering_detected": cross["state_reordering_detected"],
        "states": state_rows,
        "path_edge_changes": edge_rows,
        "state7_state8_splitting_GHz": _relative_change(
            splitting71, splitting51
        ),
        "leading_path_score": _relative_change(
            pathways71[0]["score_product"],
            pathways51[0]["score_product"],
        ),
        "leading_to_second_path_score_ratio": {
            "n51": float(
                pathways51[0]["score_product"]
                / pathways51[1]["score_product"]
            ),
            "n71": float(
                pathways71[0]["score_product"]
                / pathways71[1]["score_product"]
            ),
        },
        "n51_leading_path": cross["n51_leading_path"],
        "n71_leading_path_in_n51_labels": cross[
            "n71_ranked_paths_in_n51_tracked_labels"
        ][0],
    }


def _retained_basis_sensitivity(saved_analysis):
    rows = {}
    for label, case in saved_analysis["retained_basis_cases"].items():
        rows[label] = {
            "state7_overlap_vs_k180": case["state_overlaps"][7],
            "state8_overlap_vs_k180": case["state_overlaps"][8],
            "f07_error_MHz_vs_k180": case[
                "transition_errors_MHz_vs_k180"
            ][6],
            "f08_error_MHz_vs_k180": case[
                "transition_errors_MHz_vs_k180"
            ][7],
            "path_edge_errors": case["primary_path_grid_phi_errors"],
            "leading_path": case["pathways"][0],
        }
    return rows


def _public_grid_result(result):
    return {key: value for key, value in result.items() if key != "_reference"}


def main():
    n51 = load_verified_n51_checkpoint(N51_DIR)
    n71_metadata, values71, vectors71, operators71 = _load_n71()
    saved_analysis = json.loads((N71_DIR / "analysis.json").read_text())

    n51_result = _build_grid_result(
        "N51",
        n51["values"],
        n51["vectors"],
        n51["operators"],
        n51["metadata"]["spatial_cutoffs"],
    )
    n71_result = _build_grid_result(
        "N71",
        values71,
        vectors71,
        operators71,
        n71_metadata["configuration"]["spatial_cutoffs"],
    )
    output = {
        "scope": (
            "checkpoint-only Stage-2 reconstruction; no Stage-1 spatial "
            "Hamiltonian assembly or eigensolve"
        ),
        "authenticated_inputs": {
            "n51_directory": str(N51_DIR.relative_to(ROOT)),
            "n51_file_sha256": n51["file_sha256"],
            "n51_hash_manifest_matches_pin": (
                n51["file_sha256"] == N51_ORACLE_SHA256
            ),
            "n71_directory": str(N71_DIR.relative_to(ROOT)),
            "n71_configuration_sha256": n71_metadata[
                "configuration_sha256"
            ],
            "n71_file_sha256": n71_metadata["file_sha256"],
        },
        "diagnostic_definitions": {
            "raw_basis_order": "(theta3, theta2, n1, mode4_HO)",
            "compact_charge": (
                "exact marginal in the saved n1=-4..4 raw basis after "
                "Stage-2 reconstruction"
            ),
            "theta_moments": (
                "probability moments on the saved uniform theta2/theta3 grids"
            ),
            "mode4_occupation": (
                "occupation of the saved eight-state bare harmonic-oscillator "
                "basis"
            ),
            "stage1_parent_participation": (
                "probability over the 180 saved nonlinear-sector eigenstates"
            ),
            "entanglement": (
                "Schmidt diagnostics between the saved Stage-1 sector and "
                "mode-4 oscillator"
            ),
            "path_score": "product of three absolute grid_phi edges",
        },
        "unsupported_from_saved_data": [
            "a source-grounded parity/symmetry operator for the asymmetric model",
            "sensitivity to omitted or altered parasitic capacitances",
            "derivatives with respect to absent model parameters",
            "absolute convergence beyond the saved k=180 parents",
        ],
        "reproduction_checks": _reproduction_checks(
            n51_result, n71_result, saved_analysis
        ),
        "grids": {
            "N51": _public_grid_result(n51_result),
            "N71": _public_grid_result(n71_result),
        },
        "n51_to_n71": _comparison(
            n51_result, n71_result, saved_analysis
        ),
        "saved_n71_retained_basis_sensitivity": (
            _retained_basis_sensitivity(saved_analysis)
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
