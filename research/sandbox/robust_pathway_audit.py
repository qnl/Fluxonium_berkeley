"""Checkpoint-only robustness audit of asymmetric Gridium three-step paths.

The calculation authenticates the saved N=51 and N=71 Stage-1 checkpoints,
reconstructs their already-defined 1440-dimensional Stage-2 Hamiltonians, and
uses saved retained-basis audits.  It never assembles or diagonalizes a large
spatial Stage-1 Hamiltonian.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from Simulations.Rabi_3Photon.four_mode_asymmetric_shell_audit import (
    _components,
    _reference,
)
from Simulations.Rabi_3Photon.run_four_mode_spatial71_reference import (
    N51_ORACLE_SHA256,
    load_verified_n51_checkpoint,
)
from research.sandbox.state_0178_checkpoint_diagnostics import (
    N51_DIR,
    N71_DIR,
    ROOT,
    _load_n71,
    _state_metrics,
)


OUTPUT_JSON = ROOT / "research/sandbox/robust_pathway_audit.json"
OUTPUT_CSV = ROOT / "research/sandbox/robust_pathway_paths.csv"
NSTATES = 11
ACTIVE_GRID_PHI_THRESHOLD = 1e-8
SERIOUS_SCORE_FRACTION = 0.01
CROWDING_WINDOWS_MHZ = (10.0, 50.0)


def _relative_change(candidate, reference):
    if abs(reference) <= 1e-14:
        return None
    return float((candidate - reference) / abs(reference))


def _build_grid(values, vectors, operators, spatial):
    components = _components(values, operators)
    reference = _reference(components)
    states = {
        str(state): _state_metrics(
            state, reference, components, vectors, spatial
        )
        for state in range(NSTATES)
    }
    return reference, states


def _active_transitions(values, matrix):
    transitions = []
    for initial, final in itertools.combinations(range(NSTATES), 2):
        coupling = float(abs(matrix[initial, final]))
        if coupling < ACTIVE_GRID_PHI_THRESHOLD:
            continue
        transitions.append({
            "pair": [initial, final],
            "frequency_GHz": float(abs(values[final] - values[initial])),
            "abs_grid_phi": coupling,
        })
    return transitions


def _carrier_crowding(pair, frequency, transitions, path_pairs):
    pair_key = tuple(sorted(pair))
    competitors = []
    for transition in transitions:
        candidate_pair = tuple(transition["pair"])
        if candidate_pair == pair_key:
            continue
        detuning_mhz = 1e3 * abs(transition["frequency_GHz"] - frequency)
        competitors.append({
            **transition,
            "detuning_MHz": float(detuning_mhz),
            "is_another_edge_of_this_path": candidate_pair in path_pairs,
        })
    competitors.sort(key=lambda row: row["detuning_MHz"])
    nearest = competitors[0]
    return {
        "pair": list(pair),
        "frequency_GHz": float(frequency),
        "nearest_active_competitor": nearest,
        "active_competitor_count_within_10_MHz": sum(
            row["detuning_MHz"] <= 10.0 for row in competitors
        ),
        "active_competitor_count_within_50_MHz": sum(
            row["detuning_MHz"] <= 50.0 for row in competitors
        ),
    }


def _character_summary(states, intermediate_states):
    rows = [states[str(state)] for state in intermediate_states]
    dominant_weights = [
        row["stage1_parent_participation"]["top_stage1_ranks"][0][
            "probability"
        ]
        for row in rows
    ]
    return {
        "mean_mode4_occupation_by_state": {
            str(state): row["mode4"]["mean_HO_occupation"]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_mean_mode4_occupation": max(
            row["mode4"]["mean_HO_occupation"] for row in rows
        ),
        "mode4_excited_probability_by_state": {
            str(state): row["mode4"]["probability_n4_ge_1"]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_mode4_excited_probability": max(
            row["mode4"]["probability_n4_ge_1"] for row in rows
        ),
        "stage1_mode4_entropy_bits_by_state": {
            str(state): row["stage1_mode4_entanglement"][
                "von_Neumann_entropy_bits"
            ]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_stage1_mode4_entropy_bits": max(
            row["stage1_mode4_entanglement"]["von_Neumann_entropy_bits"]
            for row in rows
        ),
        "mean_stage1_rank_by_state": {
            str(state): row["stage1_parent_participation"]["mean_rank"]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_mean_stage1_rank": max(
            row["stage1_parent_participation"]["mean_rank"]
            for row in rows
        ),
        "dominant_stage1_rank_weight_by_state": {
            str(state): weight
            for state, weight in zip(intermediate_states, dominant_weights)
        },
        "minimum_dominant_stage1_rank_weight": min(dominant_weights),
        "theta3_standard_deviation_by_state": {
            str(state): row["raw_basis"]["theta3"]["standard_deviation"]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_theta3_standard_deviation": max(
            row["raw_basis"]["theta3"]["standard_deviation"]
            for row in rows
        ),
        "grid_phi_standard_deviation_by_state": {
            str(state): row["raw_basis"]["grid_phi_coordinate"][
                "standard_deviation"
            ]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_grid_phi_standard_deviation": max(
            row["raw_basis"]["grid_phi_coordinate"]["standard_deviation"]
            for row in rows
        ),
        "compact_charge_cutoff_probability_by_state": {
            str(state): row["raw_basis"]["compact_charge_n1"][
                "probability_at_charge_cutoff_abs_n1_eq_4"
            ]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_compact_charge_cutoff_probability": max(
            row["raw_basis"]["compact_charge_n1"][
                "probability_at_charge_cutoff_abs_n1_eq_4"
            ]
            for row in rows
        ),
        "theta3_outer_20_percent_probability_by_state": {
            str(state): row["raw_basis"]["theta3"][
                "probability_abs_theta3_ge_0p8L3"
            ]
            for state, row in zip(intermediate_states, rows)
        },
        "maximum_theta3_outer_20_percent_probability": max(
            row["raw_basis"]["theta3"][
                "probability_abs_theta3_ge_0p8L3"
            ]
            for row in rows
        ),
    }


def _retained_sensitivity(path, reference_values, reference_matrix, cases):
    pairs = list(zip(path[:-1], path[1:]))
    reference_edges = [
        float(abs(reference_matrix[initial, final]))
        for initial, final in pairs
    ]
    reference_score = float(np.prod(reference_edges))
    reference_carriers = [
        float(abs(reference_values[final] - reference_values[initial]))
        for initial, final in pairs
    ]
    rows = {}
    for label, case in cases.items():
        values = np.asarray(case["energies_GHz"])
        matrix = np.asarray(case["grid_phi_abs"])
        edges = [float(matrix[initial, final]) for initial, final in pairs]
        score = float(np.prod(edges))
        carriers = [
            float(abs(values[final] - values[initial]))
            for initial, final in pairs
        ]
        edge_relative_changes = [
            _relative_change(candidate, reference)
            for candidate, reference in zip(edges, reference_edges)
        ]
        rows[label] = {
            "intermediate_state_overlaps_vs_k180": {
                str(state): float(case["state_overlaps"][state])
                for state in path[1:3]
            },
            "minimum_intermediate_state_overlap_vs_k180": float(min(
                case["state_overlaps"][state] for state in path[1:3]
            )),
            "carrier_frequency_errors_MHz_vs_k180": [
                float(1e3 * (candidate - reference))
                for candidate, reference in zip(carriers, reference_carriers)
            ],
            "maximum_absolute_carrier_frequency_error_MHz_vs_k180": float(
                max(1e3 * abs(candidate - reference)
                    for candidate, reference in zip(carriers, reference_carriers))
            ),
            "edge_magnitudes": edges,
            "edge_relative_changes_vs_k180": edge_relative_changes,
            "maximum_absolute_edge_relative_change_vs_k180": float(max(
                abs(value) for value in edge_relative_changes
                if value is not None
            )),
            "score_product": score,
            "score_relative_change_vs_k180": _relative_change(
                score, reference_score
            ),
        }
    return rows


def _path_row(path, ref51, ref71, states71, cross, retained_cases,
              active_transitions):
    matrix51 = np.abs(ref51["matrix"])
    matrix71 = np.abs(ref71["matrix"])
    values51 = ref51["values"]
    values71 = ref71["values"]
    pairs = list(zip(path[:-1], path[1:]))
    path_pair_keys = {tuple(sorted(pair)) for pair in pairs}
    edges51 = [float(matrix51[initial, final]) for initial, final in pairs]
    edges71 = [float(matrix71[initial, final]) for initial, final in pairs]
    carriers51 = [
        float(abs(values51[final] - values51[initial]))
        for initial, final in pairs
    ]
    carriers71 = [
        float(abs(values71[final] - values71[initial]))
        for initial, final in pairs
    ]
    crowding = [
        _carrier_crowding(pair, frequency, active_transitions, path_pair_keys)
        for pair, frequency in zip(pairs, carriers71)
    ]
    score51 = float(np.prod(edges51))
    score71 = float(np.prod(edges71))
    edge_changes = [
        _relative_change(candidate, reference)
        for candidate, reference in zip(edges71, edges51)
    ]
    overlaps = {
        str(state): float(cross["matched_state_overlaps"][state])
        for state in path
    }
    middle_frequency = carriers71[1]
    f01 = float(values71[1] - values71[0])
    retained = _retained_sensitivity(
        path, values71, matrix71, retained_cases
    )
    return {
        "path": path,
        "operator_active": all(edge >= ACTIVE_GRID_PHI_THRESHOLD
                               for edge in edges71),
        "score_product_N51": score51,
        "score_product_N71": score71,
        "score_relative_change_N51_to_N71": _relative_change(
            score71, score51
        ),
        "edge_magnitudes_N51": edges51,
        "edge_magnitudes_N71": edges71,
        "edge_relative_changes_N51_to_N71": edge_changes,
        "maximum_absolute_edge_relative_change_N51_to_N71": float(max(
            abs(value) for value in edge_changes if value is not None
        )),
        "intermediate_energies_from_ground_GHz_N51": [
            float(values51[state] - values51[0]) for state in path[1:3]
        ],
        "intermediate_energies_from_ground_GHz_N71": [
            float(values71[state] - values71[0]) for state in path[1:3]
        ],
        "maximum_intermediate_energy_GHz_N71": float(max(
            values71[state] - values71[0] for state in path[1:3]
        )),
        "carrier_frequencies_GHz_N51": carriers51,
        "carrier_frequencies_GHz_N71": carriers71,
        "carrier_frequency_shifts_MHz_N51_to_N71": [
            float(1e3 * (candidate - reference))
            for candidate, reference in zip(carriers71, carriers51)
        ],
        "maximum_absolute_carrier_shift_MHz_N51_to_N71": float(max(
            1e3 * abs(candidate - reference)
            for candidate, reference in zip(carriers71, carriers51)
        )),
        "state_overlaps_N51_to_N71": overlaps,
        "minimum_intermediate_state_overlap_N51_to_N71": float(min(
            cross["matched_state_overlaps"][state] for state in path[1:3]
        )),
        "N71_state_character": _character_summary(states71, path[1:3]),
        "N71_carrier_crowding": crowding,
        "minimum_active_transition_separation_MHz_N71": float(min(
            row["nearest_active_competitor"]["detuning_MHz"]
            for row in crowding
        )),
        "direct_logical_contamination": {
            "middle_tone_frequency_GHz_N71": middle_frequency,
            "logical_f01_GHz_N71": f01,
            "middle_tone_absolute_detuning_from_f01_MHz_N71": float(
                1e3 * abs(middle_frequency - f01)
            ),
            "direct_logical_abs_grid_phi_N71": float(matrix71[0, 1]),
            "direct_logical_to_intended_middle_edge_ratio_N71": float(
                matrix71[0, 1] / edges71[1]
            ),
        },
        "saved_N71_retained_basis_sensitivity": retained,
    }


def _dominates(left, right, objectives):
    left_values = [direction * left[key] for key, direction in objectives]
    right_values = [direction * right[key] for key, direction in objectives]
    return (
        all(a >= b for a, b in zip(left_values, right_values))
        and any(a > b for a, b in zip(left_values, right_values))
    )


def _pareto(rows, objectives):
    return [
        row for row in rows
        if not any(
            other is not row and _dominates(other, row, objectives)
            for other in rows
        )
    ]


def _frontier_view(row):
    character = row["N71_state_character"]
    contamination = row["direct_logical_contamination"]
    return {
        "path": row["path"],
        "score_product_N71": row["score_product_N71"],
        "maximum_intermediate_energy_GHz_N71": row[
            "maximum_intermediate_energy_GHz_N71"
        ],
        "absolute_score_change_fraction_N51_to_N71": abs(
            row["score_relative_change_N51_to_N71"]
        ),
        "minimum_intermediate_state_overlap_N51_to_N71": row[
            "minimum_intermediate_state_overlap_N51_to_N71"
        ],
        "maximum_stage1_mode4_entropy_bits": character[
            "maximum_stage1_mode4_entropy_bits"
        ],
        "minimum_active_transition_separation_MHz_N71": row[
            "minimum_active_transition_separation_MHz_N71"
        ],
        "middle_tone_detuning_from_f01_MHz_N71": contamination[
            "middle_tone_absolute_detuning_from_f01_MHz_N71"
        ],
    }


def _selection_structure(matrix):
    active_edges = [
        [initial, final]
        for initial, final in itertools.combinations(range(NSTATES), 2)
        if abs(matrix[initial, final]) >= ACTIVE_GRID_PHI_THRESHOLD
    ]
    colors = {}
    adjacency = {state: [] for state in range(NSTATES)}
    for initial, final in active_edges:
        adjacency[initial].append(final)
        adjacency[final].append(initial)
    for seed in range(NSTATES):
        if seed in colors:
            continue
        colors[seed] = 0
        queue = [seed]
        while queue:
            state = queue.pop(0)
            for neighbor in adjacency[state]:
                proposed = 1 - colors[state]
                if neighbor in colors and colors[neighbor] != proposed:
                    return {
                        "active_threshold": ACTIVE_GRID_PHI_THRESHOLD,
                        "active_edges": active_edges,
                        "active_graph_is_bipartite": False,
                        "physical_parity_label_assigned": False,
                    }
                if neighbor not in colors:
                    colors[neighbor] = proposed
                    queue.append(neighbor)
    groups = [
        sorted(state for state, color in colors.items() if color == value)
        for value in (0, 1)
    ]
    within = [
        float(abs(matrix[initial, final]))
        for group in groups
        for initial, final in itertools.combinations(group, 2)
    ]
    return {
        "active_threshold": ACTIVE_GRID_PHI_THRESHOLD,
        "active_edges": active_edges,
        "active_graph_is_bipartite": True,
        "numerical_bipartition": groups,
        "maximum_within_group_abs_grid_phi": max(within),
        "physical_parity_label_assigned": False,
        "interpretation": (
            "This is an empirical selection-rule-like pattern of the stored "
            "grid_phi matrix only. No saved symmetry/parity operator supports "
            "assigning a physical parity quantum number."
        ),
    }


def _flatten_for_pareto(row):
    character = row["N71_state_character"]
    contamination = row["direct_logical_contamination"]
    return {
        **row,
        "pareto_score": row["score_product_N71"],
        "pareto_energy": row["maximum_intermediate_energy_GHz_N71"],
        "pareto_spatial_instability": abs(
            row["score_relative_change_N51_to_N71"]
        ),
        "pareto_mixing": character["maximum_stage1_mode4_entropy_bits"],
        "pareto_crowding": row[
            "minimum_active_transition_separation_MHz_N71"
        ],
        "pareto_contamination": contamination[
            "middle_tone_absolute_detuning_from_f01_MHz_N71"
        ],
    }


def main():
    n51 = load_verified_n51_checkpoint(N51_DIR)
    n71_metadata, values71, vectors71, operators71 = _load_n71()
    saved_analysis = json.loads((N71_DIR / "analysis.json").read_text())

    ref51, states51 = _build_grid(
        n51["values"], n51["vectors"], n51["operators"],
        n51["metadata"]["spatial_cutoffs"],
    )
    ref71, states71 = _build_grid(
        values71, vectors71, operators71,
        n71_metadata["configuration"]["spatial_cutoffs"],
    )
    cross = saved_analysis["n51_to_n71_spatial_comparison"]
    if cross["n71_energy_row_for_n51_state"] != list(range(NSTATES)):
        raise RuntimeError("This audit expects the saved identity state tracking.")
    active_transitions = _active_transitions(
        ref71["values"], ref71["matrix"]
    )

    rows = []
    for middle1, middle2 in itertools.permutations(range(2, NSTATES), 2):
        rows.append(_path_row(
            [0, middle1, middle2, 1], ref51, ref71, states71,
            cross, saved_analysis["retained_basis_cases"],
            active_transitions,
        ))
    rows.sort(key=lambda row: row["score_product_N71"], reverse=True)
    active_rows = [row for row in rows if row["operator_active"]]
    best_score = active_rows[0]["score_product_N71"]
    serious_rows = [
        row for row in active_rows
        if row["score_product_N71"] >= SERIOUS_SCORE_FRACTION * best_score
    ]

    objectives = [
        ("pareto_score", +1),
        ("pareto_energy", -1),
        ("pareto_spatial_instability", -1),
        ("pareto_mixing", -1),
        ("pareto_crowding", +1),
        ("pareto_contamination", +1),
    ]
    flattened = [_flatten_for_pareto(row) for row in serious_rows]
    frontier = _pareto(flattened, objectives)
    selection51 = _selection_structure(ref51["matrix"])
    selection71 = _selection_structure(ref71["matrix"])

    saved_n71 = saved_analysis["n71_reference_k180"]
    output = {
        "scope": (
            "authenticated checkpoint-only Stage-2 reconstruction and "
            "postprocessing; no Stage-1 spatial solve or gate optimization"
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
        "reproduction_checks": {
            "N51_max_abs_grid_phi_difference": float(np.max(np.abs(
                np.abs(ref51["matrix"])
                - np.asarray(cross["n51_grid_phi_abs"])
            ))),
            "N71_max_abs_grid_phi_difference": float(np.max(np.abs(
                np.abs(ref71["matrix"])
                - np.asarray(saved_n71["grid_phi_abs"])
            ))),
            "N51_max_transition_difference_GHz": float(np.max(np.abs(
                ref51["transitions"][1:]
                - np.asarray(cross["n51_transitions_GHz"])
            ))),
            "N71_max_transition_difference_GHz": float(np.max(np.abs(
                ref71["transitions"][1:]
                - np.asarray(saved_n71["transitions_GHz"])
            ))),
        },
        "definitions": {
            "enumerated_paths": (
                "all 9*8=72 ordered paths with distinct m1,m2 in states 2..10"
            ),
            "operator_active_path": (
                "all three N71 |grid_phi| edges are at least 1e-8"
            ),
            "serious_candidate": (
                "operator-active and N71 coupling product at least 1% of the "
                "best path; this transparent reporting cutoff is not a "
                "physical sufficiency criterion"
            ),
            "active_transition_for_crowding": (
                "an unordered low-11 transition with |grid_phi| at least 1e-8"
            ),
            "crowding": (
                "for each carrier, nearest other active low-11 transition; "
                "another intended path edge is retained as a competitor and "
                "flagged"
            ),
            "pareto_objectives": {
                "maximize": [
                    "N71 path coupling product",
                    "minimum active-transition separation",
                    "middle-tone detuning from f01",
                ],
                "minimize": [
                    "maximum intermediate energy",
                    "absolute N51-to-N71 path-score fractional change",
                    "maximum intermediate Stage1-mode4 entropy",
                ],
                "note": (
                    "No weighted aggregate score is formed. Other reported "
                    "stability and character metrics remain diagnostic rather "
                    "than hidden Pareto weights."
                ),
            },
        },
        "counts": {
            "enumerated_ordered_paths": len(rows),
            "operator_active_paths": len(active_rows),
            "serious_candidates": len(serious_rows),
            "pareto_frontier_members": len(frontier),
        },
        "selection_rule_structure": {
            "N51": selection51,
            "N71": selection71,
            "same_numerical_bipartition_N51_to_N71": (
                selection51.get("numerical_bipartition")
                == selection71.get("numerical_bipartition")
            ),
            "same_active_edge_set_N51_to_N71": (
                selection51["active_edges"] == selection71["active_edges"]
            ),
        },
        "state_metrics": {"N51": states51, "N71": states71},
        "active_low11_transitions_N71": active_transitions,
        "top_coupling_ranked_operator_active_paths": active_rows,
        "serious_candidates": serious_rows,
        "pareto_frontier_among_serious_candidates": [
            _frontier_view(row) for row in frontier
        ],
        "all_enumerated_paths": rows,
        "limitations": [
            "Only states 0..10 are available in the saved Stage-2 reference.",
            "Crowding excludes transitions involving states above 10.",
            "Static matrix elements and detunings do not predict driven pulse fidelity.",
            "The saved checkpoints contain no physical parity/symmetry operator.",
            "No parasitic-capacitance or other omitted-physics perturbation exists in the saved data.",
            "N71 retained-basis evidence stops at the k180 parent with k170/k172/k175 reductions; k220 is pending.",
            "N51-to-N71 is a spatial-grid comparison, not a physical-model robustness test.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    with OUTPUT_CSV.open("w", newline="") as handle:
        fields = [
            "rank", "path", "operator_active", "serious_candidate",
            "score_N71", "score_change_percent_N51_to_N71",
            "intermediate_energies_GHz_N71", "min_intermediate_overlap",
            "max_entropy_bits", "max_mean_mode4_occupation",
            "min_crowding_MHz", "middle_detuning_from_f01_MHz",
            "k175_min_intermediate_overlap", "k175_max_carrier_error_MHz",
            "k175_max_edge_relative_change_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(active_rows, 1):
            character = row["N71_state_character"]
            retained = row["saved_N71_retained_basis_sensitivity"][
                "asymmetric_priority_k175"
            ]
            writer.writerow({
                "rank": rank,
                "path": "-".join(map(str, row["path"])),
                "operator_active": row["operator_active"],
                "serious_candidate": row in serious_rows,
                "score_N71": row["score_product_N71"],
                "score_change_percent_N51_to_N71": 100 * row[
                    "score_relative_change_N51_to_N71"
                ],
                "intermediate_energies_GHz_N71": ";".join(
                    map(str, row["intermediate_energies_from_ground_GHz_N71"])
                ),
                "min_intermediate_overlap": row[
                    "minimum_intermediate_state_overlap_N51_to_N71"
                ],
                "max_entropy_bits": character[
                    "maximum_stage1_mode4_entropy_bits"
                ],
                "max_mean_mode4_occupation": character[
                    "maximum_mean_mode4_occupation"
                ],
                "min_crowding_MHz": row[
                    "minimum_active_transition_separation_MHz_N71"
                ],
                "middle_detuning_from_f01_MHz": row[
                    "direct_logical_contamination"
                ]["middle_tone_absolute_detuning_from_f01_MHz_N71"],
                "k175_min_intermediate_overlap": retained[
                    "minimum_intermediate_state_overlap_vs_k180"
                ],
                "k175_max_carrier_error_MHz": retained[
                    "maximum_absolute_carrier_frequency_error_MHz_vs_k180"
                ],
                "k175_max_edge_relative_change_percent": 100 * retained[
                    "maximum_absolute_edge_relative_change_vs_k180"
                ],
            })
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
