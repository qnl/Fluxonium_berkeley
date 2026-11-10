#!/usr/bin/env python3
"""Execute the frozen k260 Stage-2 rank-addback experiment.

This script authenticates and reads the saved k260 parent. It never assembles
or diagonalizes a Stage-1 Hamiltonian.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CP = ROOT / "research/checkpoints/four-mode-asymmetric-n71-k260"
PREREG = HERE / "preregistration.json"
EXPECTED_PREREG_SHA256 = "cdde5cd5c520f80719cfc417c90270fc40508df44276809725a03825ed088c7b"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/sandbox/k260_preregistration"))
sys.path.insert(0, str(ROOT / "research/sandbox/k260_shell_attribution"))

from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import _three_step_paths
from k260_profile import install_profile
from attribute import solve


STATES = (0, 1, 2, 3, 7, 8)
SUBSPACE = (0, 1, 7, 8)
FREQUENCIES = {
    "f01": (0, 1), "f07": (0, 7), "f08": (0, 8), "f78": (7, 8),
    "f81": (1, 8), "f72": (2, 7), "f21": (1, 2), "f03": (0, 3),
    "f38": (3, 8),
}
EDGES = {
    "g07": (0, 7), "g78": (7, 8), "g81": (1, 8), "g72": (2, 7),
    "g21": (1, 2), "g03": (0, 3), "g38": (3, 8),
}
PATHS = {
    "0-7-8-1": (0, 7, 8, 1),
    "0-7-2-1": (0, 7, 2, 1),
    "0-3-8-1": (0, 3, 8, 1),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def tracked_snapshot():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {name: sha256(ROOT / name) for name in names if name and (ROOT / name).is_file()}


def checkpoint_snapshot():
    roots = [
        CP,
        ROOT / "research/checkpoints/four-mode-asymmetric-n71-k220",
        ROOT / "research/checkpoints/four-mode-asymmetric-n71-k180",
        ROOT / "research/checkpoints/2026-09-26-asymmetric-n51",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for checkpoint in roots for path in checkpoint.rglob("*") if path.is_file()
    }


def selected_indices(ranks):
    return np.asarray([rank * 8 + mode for rank in ranks for mode in range(8)], dtype=int)


def operator_metrics(matrix):
    blocks = {
        "full_0_1_7_8": (SUBSPACE, SUBSPACE),
        "logical_to_7_8": ((0, 1), (7, 8)),
        "internal_7_8": ((7, 8), (7, 8)),
    }
    result = {}
    for name, (rows, columns) in blocks.items():
        block = matrix[np.ix_(rows, columns)]
        result[name] = {
            "frobenius_norm": float(np.linalg.norm(block)),
            "singular_values": np.linalg.svd(block, compute_uv=False).tolist(),
        }
    return result


def classify_max(value, passed, borderline):
    value = abs(float(value))
    return "PASS" if value <= passed else "BORDERLINE" if value <= borderline else "FAIL"


def classify_min(value, passed, borderline):
    value = float(value)
    return "PASS" if value >= passed else "BORDERLINE" if value >= borderline else "FAIL"


def status_worst(statuses):
    order = {"PASS": 0, "BORDERLINE": 1, "FAIL": 2}
    return max(statuses, key=order.get)


def scalar_metrics(case):
    energies = case["e"]
    matrix = case["matrix"]
    ranked = _three_step_paths(matrix, 11, count=None)
    pathways = {tuple(row["path"]): {**row, "rank": rank} for rank, row in enumerate(ranked, 1)}
    frequencies = {
        name: float(abs(energies[right] - energies[left]) * 1000)
        for name, (left, right) in FREQUENCIES.items()
    }
    return {
        "state_energies_MHz": {f"E{state}": float(energies[state] * 1000) for state in STATES},
        "frequencies_MHz": frequencies,
        "selectivity_MHz": {
            "f78_minus_f01": frequencies["f78"] - frequencies["f01"],
            "f07_minus_f81": frequencies["f07"] - frequencies["f81"],
        },
        "edges": {name: float(abs(matrix[left, right])) for name, (left, right) in EDGES.items()},
        "scores": {name: float(pathways[path]["score_product"]) for name, path in PATHS.items()},
        "path_ranks": {name: int(pathways[path]["rank"]) for name, path in PATHS.items()},
        "operator_blocks": operator_metrics(matrix),
    }


def flattened_scalar_metrics(metrics):
    result = {}
    for name, value in metrics["state_energies_MHz"].items():
        result[("state_energy", name)] = value
    for name, value in metrics["frequencies_MHz"].items():
        result[("carrier", name)] = value
    for name, value in metrics["selectivity_MHz"].items():
        result[("selectivity", name)] = value
    return result


def relative_error(value, reference):
    if abs(reference) <= 1e-14:
        raise ValueError("Relative comparison against zero.")
    return (float(value) - float(reference)) / abs(float(reference))


def main():
    assert sha256(PREREG) == EXPECTED_PREREG_SHA256, "Frozen preregistration hash mismatch."
    prereg = json.loads(PREREG.read_text())
    assert prereg["status"] == "frozen-before-selected-space-diagonalization"
    before_tracked = tracked_snapshot()
    before_checkpoints = checkpoint_snapshot()

    install_profile(runner)
    metadata = runner.validate_complete_checkpoint_directory(CP)
    assert metadata["configuration_sha256"] == prereg["authenticated_inputs"]["configuration_sha256"]
    assert sha256(CP / "checkpoint.json") == prereg["authenticated_inputs"]["checkpoint_json_sha256"]
    assert sha256(ROOT / "research/sandbox/k260_preregistration/criteria.json") == prereg["authenticated_inputs"]["frozen_convergence_criteria_sha256"]
    assert sha256(ROOT / "research/sandbox/k260_shell_attribution/methodology.md") == prereg["authenticated_inputs"]["shell_methodology_sha256"]
    assert sha256(ROOT / "research/sandbox/k260_shell_attribution/summary.json") == prereg["authenticated_inputs"]["shell_summary_sha256"]

    saved = json.loads((CP / "analysis.json").read_text())["retained_basis_cases"]
    with np.load(CP / "stage1_eigensystem.npz", allow_pickle=False) as archive:
        stage1_values = np.asarray(archive["eigenvalues_GHz"])
        stage1_shape = list(np.asarray(archive["eigenvectors"]).shape)
    with np.load(CP / "projected_operators.npz", allow_pickle=False) as archive:
        operators = {name: np.asarray(archive[name]) for name in archive.files}
    assert stage1_values.shape == (260,) and stage1_shape == [45369, 260]
    assert set(operators) == {"n1", "n2", "n3", "x2", "x3"}
    assert all(value.shape == (260, 260) for value in operators.values())

    physical = metadata["configuration"]["physical_parameters"]
    ec_matrix, k_matrix, _, _ = _coeffs(
        physical["EJ"], physical["EC"], physical["EL"], physical["ELK"],
        physical["EJS"], physical["ECS"], physical["eps_J"], physical["eps_LK"],
        physical["eC"], physical["eP"],
    )
    A4, B4 = 4 * ec_matrix[3, 3], 0.5 * k_matrix[2, 2]
    f4 = (A4 / (4 * B4)) ** 0.25
    g4 = 1 / (2 * f4)
    annihilation = np.diag(np.sqrt(np.arange(1, 8)), 1)
    x4 = f4 * (annihilation + annihilation.T)
    p4 = 1j * g4 * (annihilation.T - annihilation)
    w4 = 2 * np.sqrt(A4 * B4) * (np.arange(8) + 0.5)
    hamiltonian = np.kron(np.diag(stage1_values), np.eye(8)).astype(complex)
    hamiltonian += np.kron(np.eye(260), np.diag(w4))
    for index in range(3):
        hamiltonian += 8 * ec_matrix[index, 3] * np.kron(operators[f"n{index + 1}"], p4)
    hamiltonian += k_matrix[0, 2] * np.kron(operators["x2"], x4)
    hamiltonian += k_matrix[1, 2] * np.kron(operators["x3"], x4)
    hermiticity_error = float(np.max(abs(hamiltonian - hamiltonian.conj().T)))
    assert hermiticity_error <= 1e-12
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2
    grid_phi = np.kron(0.5 * operators["x2"] + operators["x3"], np.eye(8))

    cases_spec = prereg["cases"]
    full_spec = next(item for item in cases_spec if item["id"] == "L_full_k260")
    full_ranks = list(range(220)) + full_spec["added_ranks"]
    full_indices = selected_indices(full_ranks)
    print("Solving authenticated finite Stage-2 endpoint", flush=True)
    cases = {"L_full_k260": solve(hamiltonian, full_indices)}
    reference = cases["L_full_k260"]["embedded"]

    for spec in cases_spec:
        case_id = spec["id"]
        if case_id == "L_full_k260":
            continue
        ranks = list(range(220)) + spec["added_ranks"]
        indices = selected_indices(ranks)
        print(f"Solving {case_id}: {len(ranks)} ranks, {len(indices)} dimensions", flush=True)
        cases[case_id] = solve(hamiltonian[np.ix_(indices, indices)], indices, reference)

    # Operator matrices, state metrics, endpoint and k240 overlaps.
    k240_ref = cases["H_full_Q1_k240"]["embedded"]
    for spec in cases_spec:
        case = cases[spec["id"]]
        case["matrix"] = case["embedded"].conj().T @ grid_phi @ case["embedded"]
        case["metrics"] = scalar_metrics(case)
        overlap_260 = case["embedded"].conj().T @ reference
        overlap_240 = case["embedded"].conj().T @ k240_ref
        case["overlap_260"] = overlap_260
        case["overlap_240"] = overlap_240
        case["subspace_260"] = runner.subspace_overlap_diagnostics(overlap_260, SUBSPACE, SUBSPACE)
        case["subspace_240"] = runner.subspace_overlap_diagnostics(overlap_240, SUBSPACE, SUBSPACE)

    # Reconstruction gate before interpretation.
    reconstruction_map = {
        "A_P": "nested_prefix_k220",
        "H_full_Q1_k240": "nested_prefix_k240",
        "L_full_k260": "nested_prefix_k260",
    }
    reconstruction = {}
    reconstruction_pass = True
    for case_id, saved_id in reconstruction_map.items():
        case = cases[case_id]
        expected = saved[saved_id]
        energy_error = float(np.max(abs(case["e"] - expected["energies_GHz"])))
        operator_error = float(np.max(abs(abs(case["matrix"]) - expected["grid_phi_abs"])))
        passed = energy_error <= 1e-10 and operator_error <= 1e-10 and case["residual"] <= 1e-9
        reconstruction_pass &= passed
        reconstruction[case_id] = {
            "energy_error_GHz": energy_error,
            "grid_phi_magnitude_error": operator_error,
            "eigenpair_residual_GHz": case["residual"],
            "status": "PASS" if passed else "FAIL",
        }
    (HERE / "reconstruction_tests.json").write_text(json.dumps({
        "status": "PASS" if reconstruction_pass else "FAIL",
        "tests": reconstruction,
        "hermiticity_error": hermiticity_error,
    }, indent=2) + "\n")
    if not reconstruction_pass:
        raise SystemExit("RECONSTRUCTION FAIL: scientific interpretation stopped.")
    print("Reconstruction PASS; computing preregistered comparisons", flush=True)

    baseline = cases["A_P"]
    target = cases["H_full_Q1_k240"]
    endpoint = cases["L_full_k260"]
    base_flat = flattened_scalar_metrics(baseline["metrics"])
    target_flat = flattened_scalar_metrics(target["metrics"])
    endpoint_flat = flattened_scalar_metrics(endpoint["metrics"])
    comparison_rows = []
    case_summaries = {}
    thresholds = prereg["frozen_thresholds"]

    for spec in cases_spec:
        case_id = spec["id"]
        case = cases[case_id]
        flat = flattened_scalar_metrics(case["metrics"])
        added_count = len(spec["added_ranks"])
        for key, value in flat.items():
            category, metric = key
            effect = value - base_flat[key]
            full_effect = target_flat[key] - base_flat[key]
            error_240 = value - target_flat[key]
            error_260 = value - endpoint_flat[key]
            recovery_ratio = effect / full_effect if abs(full_effect) >= 0.1 else None
            recovered_fraction = 1 - abs(effect - full_effect) / abs(full_effect) if abs(full_effect) >= 0.1 else None
            per_rank = ((abs(base_flat[key] - target_flat[key]) - abs(error_240)) / added_count
                        if added_count else None)
            comparison_rows.append({
                "case": case_id, "role": spec["role"], "added_rank_count": added_count,
                "category": category, "metric": metric, "value": value,
                "effect_vs_P": effect, "full_Q1_effect": full_effect,
                "signed_recovery_ratio": recovery_ratio, "recovered_fraction": recovered_fraction,
                "error_vs_k240": error_240, "error_vs_k260": error_260,
                "accuracy_gain_per_added_rank": per_rank,
            })

        # Frozen carrier errors against k240.
        carrier_classification = {}
        for name, value in case["metrics"]["frequencies_MHz"].items():
            error = value - target["metrics"]["frequencies_MHz"][name]
            limits = thresholds["fine_f78_delta_MHz"] if name == "f78" else thresholds["standard_carrier_delta_MHz"]
            carrier_classification[name] = {
                "error_MHz": error,
                "strict": classify_max(error, limits["strict_pass_max"], limits["strict_borderline_max"]),
                "practical": classify_max(error, limits["practical_pass_max"], limits["practical_borderline_max"]),
            }

        # Frozen operator classifications against k240.
        operator_components = []
        for name, value in case["metrics"]["edges"].items():
            reference_value = target["metrics"]["edges"][name]
            change = relative_error(value, reference_value)
            operator_components.append({"category": "edge", "metric": name, "relative_error": change,
                "status": classify_max(change, 0.03, 0.05)})
        for name, value in case["metrics"]["scores"].items():
            reference_value = target["metrics"]["scores"][name]
            change = relative_error(value, reference_value)
            operator_components.append({"category": "path_score", "metric": name, "relative_error": change,
                "status": classify_max(change, 0.05, 0.10)})
        for block_name, values in case["metrics"]["operator_blocks"].items():
            reference_values = target["metrics"]["operator_blocks"][block_name]
            pairs = [(f"{block_name}.frobenius_norm", values["frobenius_norm"], reference_values["frobenius_norm"])]
            pairs.extend((f"{block_name}.singular_value_{index}", value,
                          reference_values["singular_values"][index])
                         for index, value in enumerate(values["singular_values"]))
            for name, value, reference_value in pairs:
                change = relative_error(value, reference_value)
                operator_components.append({"category": "operator_block", "metric": name,
                    "relative_error": change, "status": classify_max(change, 0.03, 0.05)})
        operator_status = status_worst([row["status"] for row in operator_components])

        confidence = {row["reference_state"]: row for row in case["confidence"]}
        tracking_safe = all(confidence[state]["state_specific_claim_safe"] for state in STATES) if confidence else True
        overlaps_260 = {str(state): float(abs(case["overlap_260"][state, state])) for state in STATES}
        overlaps_240 = {str(state): float(abs(case["overlap_240"][state, state])) for state in STATES}
        overlap_status_240 = {
            str(state): classify_min(value, 0.999, 0.995) for state, value in
            ((state, overlaps_240[str(state)]) for state in STATES)
        }
        case_summaries[case_id] = {
            "role": spec["role"], "added_ranks": spec["added_ranks"],
            "rank_count": 220 + added_count, "dimension": (220 + added_count) * 8,
            "metrics": case["metrics"], "carrier_classification_vs_k240": carrier_classification,
            "worst_carrier_error_MHz": max(abs(row["error_MHz"]) for row in carrier_classification.values()),
            "primary_worst_carrier_error_MHz": max(abs(carrier_classification[name]["error_MHz"])
                                                    for name in ("f07", "f08", "f78")),
            "operator_components_vs_k240": operator_components,
            "operator_accuracy": operator_status,
            "tracking_safe": tracking_safe,
            "overlaps_vs_k260": overlaps_260, "overlaps_vs_k240": overlaps_240,
            "overlap_status_vs_k240": overlap_status_240,
            "subspace_vs_k260": case["subspace_260"], "subspace_vs_k240": case["subspace_240"],
            "eigenpair_residual_GHz": case["residual"],
        }

    # Recovery lookup and decision.
    recovery_lookup = {(row["case"], row["category"], row["metric"]): row for row in comparison_rows}
    dominant = case_summaries["F_dominant3"]
    primary_keys = [
        ("state_energy", "E7"), ("state_energy", "E8"),
        ("carrier", "f07"), ("carrier", "f08"), ("carrier", "f78"),
    ]
    primary_recovery = {
        metric: recovery_lookup[("F_dominant3", category, metric)]["recovered_fraction"]
        for category, metric in primary_keys
    }
    control_ids = ("M_control_prefix3", "N_control_spread3")
    dominant_state78_mean_error = np.mean([
        abs(recovery_lookup[("F_dominant3", "state_energy", name)]["error_vs_k240"])
        for name in ("E7", "E8")
    ])
    controls_beaten = True
    for control_id in control_ids:
        control_state_error = np.mean([
            abs(recovery_lookup[(control_id, "state_energy", name)]["error_vs_k240"])
            for name in ("E7", "E8")
        ])
        controls_beaten &= (
            dominant["primary_worst_carrier_error_MHz"] < case_summaries[control_id]["primary_worst_carrier_error_MHz"]
            and dominant_state78_mean_error < control_state_error
        )
    recovery_target = all(value is not None and value >= 0.80 for value in primary_recovery.values())
    carrier_target = (
        abs(dominant["carrier_classification_vs_k240"]["f07"]["error_MHz"]) <= 5
        and abs(dominant["carrier_classification_vs_k240"]["f08"]["error_MHz"]) <= 5
        and abs(dominant["carrier_classification_vs_k240"]["f78"]["error_MHz"]) <= 1
    )
    if (recovery_target and dominant["tracking_safe"] and dominant["operator_accuracy"] == "PASS"
            and carrier_target and controls_beaten):
        strategy = "PROMISING"
    elif (dominant["tracking_safe"] and dominant["operator_accuracy"] != "FAIL"
          and all(value is not None and value >= 0.50 for value in primary_recovery.values())):
        strategy = "INCONCLUSIVE"
    else:
        strategy = "NOT JUSTIFIED"

    # Exact nonadditivity/ablation for state energies and carriers.
    nonadditivity_rows = []
    nonadditivity = {}
    flat_cases = {case_id: flattened_scalar_metrics(case["metrics"]) for case_id, case in cases.items()}
    for key in base_flat:
        category, metric = key
        A = flat_cases["A_P"][key]
        B = flat_cases["B_223"][key]
        C = flat_cases["C_230"][key]
        D = flat_cases["D_225"][key]
        E = flat_cases["E_223_230"][key]
        F = flat_cases["F_dominant3"][key]
        G = flat_cases["G_complement17"][key]
        H = flat_cases["H_full_Q1_k240"][key]
        I = flat_cases["I_Q1_258"][key]
        J = flat_cases["J_Q1_259"][key]
        K = flat_cases["K_Q1_258_259"][key]
        values = {
            "effect_223": B - A,
            "effect_230": C - A,
            "effect_225": D - A,
            "effect_pair_223_230": E - A,
            "pair_nonadditivity": E - B - C + A,
            "effect_dominant3": F - A,
            "rank225_group_interaction": F - E - D + A,
            "effect_complement17": G - A,
            "effect_full_Q1": H - A,
            "dominant_complement_nonadditivity": H - F - G + A,
            "effect_258_on_Q1": I - H,
            "effect_259_on_Q1": J - H,
            "effect_258_259_on_Q1": K - H,
            "boundary_pair_nonadditivity": K - I - J + H,
        }
        nonadditivity[f"{category}:{metric}"] = values
        nonadditivity_rows.append({"category": category, "metric": metric, **values})

    after_tracked = tracked_snapshot()
    after_checkpoints = checkpoint_snapshot()
    assert after_tracked == before_tracked, "Tracked file changed during experiment."
    assert after_checkpoints == before_checkpoints, "Authenticated checkpoint changed during experiment."

    fieldnames = list(comparison_rows[0])
    with (HERE / "numerical_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(comparison_rows)
    with (HERE / "addback_ablation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nonadditivity_rows[0]))
        writer.writeheader(); writer.writerows(nonadditivity_rows)
    operator_rows = []
    for case_id, summary in case_summaries.items():
        for row in summary["operator_components_vs_k240"]:
            operator_rows.append({"case": case_id, **row})
        for state in STATES:
            operator_rows.append({"case": case_id, "category": "state_overlap_vs_k240",
                "metric": f"state{state}", "relative_error": summary["overlaps_vs_k240"][str(state)],
                "status": summary["overlap_status_vs_k240"][str(state)]})
        operator_rows.append({"case": case_id, "category": "subspace_overlap_vs_k240",
            "metric": "minimum_singular_value", "relative_error": summary["subspace_vs_k240"]["minimum_singular_value"],
            "status": classify_min(summary["subspace_vs_k240"]["minimum_singular_value"], 0.999, 0.995)})
    with (HERE / "operator_tracking_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operator_rows[0]))
        writer.writeheader(); writer.writerows(operator_rows)

    summary = {
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "authentication": {
            "status": "PASS", "configuration_sha256": metadata["configuration_sha256"],
            "checkpoint_json_sha256": sha256(CP / "checkpoint.json"),
            "payload_hashes_verified": True, "complete_verified": True,
            "stage1_eigensystem_shape": stage1_shape,
            "projected_operator_shapes": {name: list(value.shape) for name, value in operators.items()},
            "arbitrary_selected_spaces_sufficient": True,
        },
        "reconstruction": {"status": "PASS", "tests": reconstruction, "hermiticity_error": hermiticity_error},
        "cases": case_summaries,
        "dominant_three": {
            "recovery_fractions": primary_recovery,
            "recovery_target_met": recovery_target,
            "carrier_target_met": carrier_target,
            "controls_beaten": controls_beaten,
            "mean_state7_8_energy_error_vs_k240_MHz": float(dominant_state78_mean_error),
            "operator_accuracy": dominant["operator_accuracy"],
        },
        "nonadditivity": nonadditivity,
        "selected_basis_strategy": strategy,
        "formal_retained_basis_verdict": "FAIL (unchanged)",
        "gate_work": "HOLD", "spatial_convergence": "HOLD",
        "next_research_action": "to be selected from the frozen results in REPORT.md; no action executed",
        "preservation": {
            "tracked_files_unchanged": True, "authenticated_checkpoints_unchanged": True,
            "stage1_eigensolve_run": False,
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
            "commit_created": False, "push_performed": False,
        },
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n"
    )
    print(f"Completed. Selected-basis strategy: {strategy}", flush=True)


if __name__ == "__main__":
    main()
