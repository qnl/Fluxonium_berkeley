#!/usr/bin/env python3
"""Authenticate and analyze a future N71 k260 checkpoint.

No Stage-1 matrix is assembled.  Small Stage-2 problems are reconstructed from
the authenticated saved parent eigensystem and projected operators so that
subspace and basis-invariant operator diagnostics are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import (
    _apply_state_assignment,
    _stage2_templates,
    _three_step_paths,
)
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from k260_profile import (
    EXPECTED_CONFIGURATION_SHA256,
    PROFILE_NAME,
    install_profile,
)


CRITERIA_PATH = HERE / "criteria.json"
EXPECTED_CRITERIA_SHA256 = (
    "8f42b3c10453a048238dea23d28e4a590accacbd87ad5fe54439cca1af799be9"
)
REQUIRED_K = (180, 200, 220, 240, 260)
ENDPOINT = 260
ENDPOINT_COMPARISONS = (220, 240)
STATE_COUNT = 11
STATUS_ORDER = {"REFERENCE": -1, "PASS": 0, "BORDERLINE": 1, "FAIL": 2}
FREQUENCIES = {
    "f07": (0, 7), "f08": (0, 8), "f78": (7, 8), "f81": (8, 1),
    "f72": (7, 2), "f21": (2, 1), "f03": (0, 3), "f38": (3, 8),
}
EDGES = {
    "g07": (0, 7), "g78": (7, 8), "g81": (8, 1),
    "g72": (7, 2), "g21": (2, 1), "g03": (0, 3), "g38": (3, 8),
}
PATHS = {
    "0-7-8-1": (0, 7, 8, 1),
    "0-7-2-1": (0, 7, 2, 1),
    "0-3-8-1": (0, 3, 8, 1),
}
TRACKED_STATES = (2, 3, 7, 8)
SUBSPACE = (0, 1, 7, 8)
MANDATORY_PATH_STATES = tuple(sorted({state for path in PATHS.values() for state in path}))
HISTORICALLY_NONPASS_CARRIERS = tuple(FREQUENCIES)


class AnalysisError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worst(statuses):
    useful = [status for status in statuses if status != "REFERENCE"]
    return max(useful, key=STATUS_ORDER.get) if useful else "REFERENCE"


def classify_max(value, limits):
    magnitude = abs(float(value))
    if magnitude <= limits["pass_max"]:
        return "PASS"
    if magnitude <= limits["borderline_max"]:
        return "BORDERLINE"
    return "FAIL"


def classify_min(value, limits):
    value = float(value)
    if value >= limits["pass_min"]:
        return "PASS"
    if value >= limits["borderline_min"]:
        return "BORDERLINE"
    return "FAIL"


def relative_change(value, reference):
    if abs(reference) <= 1e-12:
        raise AnalysisError("Cannot classify relative change against zero.")
    return (float(value) - float(reference)) / abs(float(reference))


def trend(values):
    values = [float(value) for value in values]
    steps = [values[index + 1] - values[index]
             for index in range(len(values) - 1)]
    ratios = [
        abs(steps[index] / steps[index - 1])
        if abs(steps[index - 1]) > 1e-15 else None
        for index in range(1, len(steps))
    ]
    nonzero = [step for step in steps if abs(step) > 1e-15]
    monotonic = (
        "constant" if not nonzero
        else "increasing" if all(step > 0 for step in nonzero)
        else "decreasing" if all(step < 0 for step in nonzero)
        else "non-monotonic"
    )
    return {
        "values": {f"k{k}": value for k, value in zip(REQUIRED_K, values)},
        "adjacent_steps": {
            f"k{left}_to_k{right}": step
            for (left, right), step in zip(zip(REQUIRED_K[:-1], REQUIRED_K[1:]), steps)
        },
        "successive_absolute_step_ratios": {
            f"abs_{REQUIRED_K[index + 1]}_{REQUIRED_K[index + 2]}_over_abs_{REQUIRED_K[index]}_{REQUIRED_K[index + 1]}": ratio
            for index, ratio in enumerate(ratios)
        },
        "total_k180_to_k260": values[-1] - values[0],
        "total_k220_to_k260": values[-1] - values[2],
        "monotonicity": monotonic,
        "sign_reversal": monotonic == "non-monotonic",
    }


def load_criteria():
    if sha256(CRITERIA_PATH) != EXPECTED_CRITERIA_SHA256:
        raise AnalysisError("Frozen criteria hash mismatch.")
    criteria = json.loads(CRITERIA_PATH.read_text())
    if criteria.get("status") != "frozen-before-any-k260-result":
        raise AnalysisError("Criteria are not frozen before k260.")
    return criteria


def discover_analysis_cases(analysis):
    found = {}
    for case in analysis.get("retained_basis_cases", {}).values():
        indices = case.get("retained_indices") if isinstance(case, dict) else None
        if isinstance(indices, list) and len(indices) in REQUIRED_K:
            found[len(indices)] = case
    if set(found) != set(REQUIRED_K):
        raise AnalysisError(f"Required retained cases missing: {sorted(set(REQUIRED_K)-set(found))}")
    for k, case in found.items():
        if case["retained_indices"] != list(range(k)):
            raise AnalysisError(f"k{k} is not the preregistered prefix basis.")
    return found


def authenticate(checkpoint_dir, criteria):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    install_profile(runner)
    metadata = runner.validate_complete_checkpoint_directory(checkpoint_dir)
    if metadata.get("profile") != PROFILE_NAME:
        raise AnalysisError("Checkpoint profile is not n71-k260.")
    if metadata.get("configuration_sha256") != EXPECTED_CONFIGURATION_SHA256:
        raise AnalysisError("k260 fingerprint mismatch.")
    if metadata.get("configuration") != criteria["profile"]["configuration"]:
        raise AnalysisError("Checkpoint configuration differs from preregistration.")
    runner.load_verified_n51_checkpoint()
    runner.validate_complete_checkpoint_directory(
        ROOT / "research/checkpoints/four-mode-asymmetric-n71-k180"
    )
    runner.validate_complete_checkpoint_directory(
        ROOT / "research/checkpoints/four-mode-asymmetric-n71-k220"
    )
    if sha256(ROOT / "research/checkpoints/four-mode-asymmetric-n71-k220/checkpoint.json") != (
            criteria["authenticated_inputs"]["k220_checkpoint_sha256"]):
        raise AnalysisError("Authenticated k220 reference changed.")
    diagnostics = metadata["stage1_solver_diagnostics"]
    quality = {
        "maximum_eigenpair_residual": diagnostics["max_eigenpair_residual"],
        "maximum_eigenpair_residual_limit": criteria["parent_quality_limits"]["maximum_eigenpair_residual"],
        "orthogonality_residual_inf": diagnostics["orthogonality_residual_inf"],
        "orthogonality_residual_inf_limit": criteria["parent_quality_limits"]["maximum_orthogonality_residual_inf"],
    }
    quality["status"] = "PASS" if (
        quality["maximum_eigenpair_residual"] <= quality["maximum_eigenpair_residual_limit"]
        and quality["orthogonality_residual_inf"] <= quality["orthogonality_residual_inf_limit"]
    ) else "FAIL"
    return metadata, quality


def reconstruct_cases(checkpoint_dir, metadata, analysis_cases):
    checkpoint_dir = Path(checkpoint_dir)
    with np.load(checkpoint_dir / "stage1_eigensystem.npz", allow_pickle=False) as z:
        stage1_values = np.asarray(z["eigenvalues_GHz"])
        stage1_vectors = np.asarray(z["eigenvectors"])
    with np.load(checkpoint_dir / "projected_operators.npz", allow_pickle=False) as z:
        operators = {name: np.asarray(z[name]) for name in z.files}
    if stage1_values.shape != (260,) or stage1_vectors.shape != (45369, 260):
        raise AnalysisError("Unexpected k260 Stage-1 shapes.")
    p = metadata["configuration"]["physical_parameters"]
    ECm, K, _, _ = _coeffs(
        p["EJ"], p["EC"], p["EL"], p["ELK"], p["EJS"], p["ECS"],
        p["eps_J"], p["eps_LK"], p["eC"], p["eP"],
    )
    n4 = metadata["configuration"]["spatial_cutoffs"]["N4"]
    raw = {}
    for k in REQUIRED_K:
        values, vectors, tensor, matrix = _stage2_templates(
            ECm, K, stage1_values, operators, np.arange(k), n4
        )
        raw[k] = dict(values=values, vectors=vectors, tensor=tensor, matrix=matrix)
    reference = raw[ENDPOINT]
    cases = {}
    for k in REQUIRED_K:
        case = raw[k]
        if k == ENDPOINT:
            overlap = case["vectors"].conj().T @ reference["vectors"]
            assignment = np.arange(STATE_COUNT)
        else:
            embedded = np.zeros((ENDPOINT, n4, STATE_COUNT), dtype=complex)
            embedded[:k] = case["tensor"]
            overlap = embedded.reshape(-1, STATE_COUNT).conj().T @ reference["vectors"]
            rows, columns = linear_sum_assignment(-np.abs(overlap))
            assignment = np.empty(STATE_COUNT, dtype=int)
            assignment[columns] = rows
        values, vectors, tensor, matrix, _ = _apply_state_assignment(
            case["values"], case["vectors"], case["tensor"], case["matrix"], assignment
        )
        saved = analysis_cases[k]
        if not np.allclose(values, saved["energies_GHz"], rtol=0, atol=1e-10):
            raise AnalysisError(f"Reconstructed k{k} energies disagree with analysis.json.")
        if not np.allclose(np.abs(matrix), saved["grid_phi_abs"], rtol=0, atol=1e-10):
            raise AnalysisError(f"Reconstructed k{k} grid_phi disagrees with analysis.json.")
        confidence = runner.assignment_confidence_diagnostics(overlap, assignment)
        clusters = runner.ambiguous_subspace_clusters(overlap, assignment, confidence)
        subspace = runner.subspace_overlap_diagnostics(
            overlap, SUBSPACE, assignment[list(SUBSPACE)]
        )
        ranked = _three_step_paths(matrix, STATE_COUNT, count=None)
        cases[k] = dict(
            values=values, matrix=matrix, overlap=overlap, assignment=assignment,
            confidence=confidence, clusters=clusters, subspace=subspace,
            pathways={tuple(row["path"]): {**row, "rank": rank}
                      for rank, row in enumerate(ranked, 1)},
        )
    return cases


def operator_metrics(matrix):
    blocks = {
        "grid_phi_0_1_7_8": (SUBSPACE, SUBSPACE),
        "grid_phi_logical_to_7_8": ((0, 1), (7, 8)),
        "grid_phi_internal_7_8": ((7, 8), (7, 8)),
    }
    result = {}
    for name, (rows, columns) in blocks.items():
        block = matrix[np.ix_(rows, columns)]
        result[name] = {
            "frobenius_norm": float(np.linalg.norm(block)),
            "singular_values": np.linalg.svd(block, compute_uv=False).tolist(),
        }
    return result


def ordering_status(cases):
    endpoint = cases[ENDPOINT]["pathways"]
    target_order = sorted(PATHS, key=lambda name: endpoint[PATHS[name]]["score_product"], reverse=True)
    details = {}
    endpoint_primary_one = endpoint[PATHS["0-7-8-1"]]["rank"] == 1
    statuses = ["PASS" if endpoint_primary_one else "FAIL"]
    for k in REQUIRED_K[:-1]:
        rows = cases[k]["pathways"]
        order = sorted(PATHS, key=lambda name: rows[PATHS[name]]["score_product"], reverse=True)
        exact_ranks = all(rows[path]["rank"] == endpoint[path]["rank"] for path in PATHS.values())
        primary_one = rows[PATHS["0-7-8-1"]]["rank"] == 1
        if primary_one and exact_ranks and order == target_order:
            status = "PASS"
        elif primary_one and order[0] == target_order[0]:
            status = "BORDERLINE"
        else:
            status = "FAIL"
        statuses.append(status)
        details[f"k{k}"] = {
            "status": status,
            "target_order": order,
            "ranks": {name: rows[path]["rank"] for name, path in PATHS.items()},
        }
    details["k260"] = {
        "status": "PASS" if endpoint_primary_one else "FAIL",
        "target_order": target_order,
        "ranks": {name: endpoint[path]["rank"] for name, path in PATHS.items()},
    }
    return {"status": worst(statuses), "cases": details}


def analyze(checkpoint_dir):
    criteria = load_criteria()
    metadata, quality = authenticate(checkpoint_dir, criteria)
    analysis = json.loads((Path(checkpoint_dir) / "analysis.json").read_text())
    analysis_cases = discover_analysis_cases(analysis)
    cases = reconstruct_cases(checkpoint_dir, metadata, analysis_cases)
    thresholds = criteria["thresholds"]

    scalar_series = {"frequencies_GHz": {}, "grid_phi_edges": {}, "pathway_scores": {}}
    for name, (left, right) in FREQUENCIES.items():
        scalar_series["frequencies_GHz"][name] = [
            abs(cases[k]["values"][right] - cases[k]["values"][left]) for k in REQUIRED_K
        ]
    for name, (left, right) in EDGES.items():
        scalar_series["grid_phi_edges"][name] = [
            float(abs(cases[k]["matrix"][left, right])) for k in REQUIRED_K
        ]
    for name, path in PATHS.items():
        scalar_series["pathway_scores"][name] = [
            cases[k]["pathways"][path]["score_product"] for k in REQUIRED_K
        ]

    blocks = {k: operator_metrics(cases[k]["matrix"]) for k in REQUIRED_K}
    trends = {
        category: {name: trend(values) for name, values in rows.items()}
        for category, rows in scalar_series.items()
    }
    trends["operator_blocks"] = {}
    for block_name in blocks[ENDPOINT]:
        trends["operator_blocks"][block_name] = {
            "frobenius_norm": trend([blocks[k][block_name]["frobenius_norm"] for k in REQUIRED_K]),
            "singular_values": [
                trend([blocks[k][block_name]["singular_values"][index] for k in REQUIRED_K])
                for index in range(len(blocks[ENDPOINT][block_name]["singular_values"]))
            ],
        }

    comparisons = {}
    for candidate in ENDPOINT_COMPARISONS:
        components = []
        component_rows = []
        for name, values in scalar_series["frequencies_GHz"].items():
            value = values[REQUIRED_K.index(candidate)]
            reference = values[-1]
            delta_mhz = 1e3 * (value - reference)
            kind = "fine_carrier_delta_MHz" if name == "f78" else "standard_carrier_delta_MHz"
            strict = classify_max(delta_mhz, thresholds["strict"][kind])
            practical = classify_max(delta_mhz, thresholds["practical"][kind])
            components.append((strict, practical))
            component_rows.append(dict(metric=name, category="carrier_frequency", value=value,
                                       endpoint=reference, change=delta_mhz,
                                       change_units="MHz", strict=strict, practical=practical))
        for state in TRACKED_STATES:
            overlap = abs(cases[candidate]["overlap"][cases[candidate]["assignment"][state], state])
            strict = classify_min(overlap, thresholds["strict"]["state_or_subspace_overlap"])
            practical = classify_min(overlap, thresholds["practical"]["state_or_subspace_overlap"])
            components.append((strict, practical))
            component_rows.append(dict(metric=f"state{state}_overlap", category="state_overlap",
                                       value=float(overlap), endpoint=1.0, change=float(overlap-1),
                                       change_units="absolute", strict=strict, practical=practical))
        subspace_value = cases[candidate]["subspace"]["minimum_singular_value"]
        strict = classify_min(subspace_value, thresholds["strict"]["state_or_subspace_overlap"])
        practical = classify_min(subspace_value, thresholds["practical"]["state_or_subspace_overlap"])
        components.append((strict, practical))
        component_rows.append(dict(metric="subspace_0_1_7_8_minimum_singular_value",
                                   category="subspace_overlap", value=subspace_value,
                                   endpoint=1.0, change=subspace_value-1.0,
                                   change_units="absolute", strict=strict, practical=practical))
        for category in ("grid_phi_edges", "pathway_scores"):
            threshold_key = ("edge_or_operator_block_relative_change" if category == "grid_phi_edges"
                             else "path_score_relative_change")
            for name, values in scalar_series[category].items():
                value = values[REQUIRED_K.index(candidate)]
                reference = values[-1]
                change = relative_change(value, reference)
                strict = classify_max(change, thresholds["strict"][threshold_key])
                practical = classify_max(change, thresholds["practical"][threshold_key])
                components.append((strict, practical))
                component_rows.append(dict(metric=name, category=category, value=value,
                                           endpoint=reference, change=change,
                                           change_units="relative", strict=strict, practical=practical))
        for block_name, endpoint_block in blocks[ENDPOINT].items():
            candidate_block = blocks[candidate][block_name]
            scalar_pairs = [(f"{block_name}.frobenius_norm", candidate_block["frobenius_norm"], endpoint_block["frobenius_norm"])]
            scalar_pairs.extend(
                (f"{block_name}.singular_value_{index}", value, endpoint_block["singular_values"][index])
                for index, value in enumerate(candidate_block["singular_values"])
            )
            for name, value, reference in scalar_pairs:
                change = relative_change(value, reference)
                strict = classify_max(change, thresholds["strict"]["edge_or_operator_block_relative_change"])
                practical = classify_max(change, thresholds["practical"]["edge_or_operator_block_relative_change"])
                components.append((strict, practical))
                component_rows.append(dict(metric=name, category="operator_block", value=value,
                                           endpoint=reference, change=change,
                                           change_units="relative", strict=strict, practical=practical))
        comparisons[f"k{candidate}_vs_k260"] = {
            "strict": worst([row[0] for row in components]),
            "practical": worst([row[1] for row in components]),
            "components": component_rows,
        }

    ordering = ordering_status(cases)
    ambiguity = {
        f"k{k}": {
            "clusters": cases[k]["clusters"],
            "clusters_affecting_mandatory_paths": [
                cluster for cluster in cases[k]["clusters"]
                if set(cluster["reference_states"]) & set(MANDATORY_PATH_STATES)
            ],
            "unsafe_mandatory_states": [
                state for state in MANDATORY_PATH_STATES
                if not cases[k]["confidence"][state]["state_specific_claim_safe"]
            ],
        }
        for k in REQUIRED_K[:-1]
    }
    ambiguity_fail = any(row["clusters_affecting_mandatory_paths"]
                         or row["unsafe_mandatory_states"]
                         for row in ambiguity.values())
    trend_failures = []
    for name in HISTORICALLY_NONPASS_CARRIERS:
        row = trends["frequencies_GHz"][name]
        steps = list(row["adjacent_steps"].values())
        prior, final = steps[-2], steps[-1]
        if prior * final < 0:
            trend_failures.append({"metric": name, "reason": "final sign reversal"})
        elif abs(final) > abs(prior) + 1e-15:
            trend_failures.append({"metric": name, "reason": "final step grew"})

    strict_220 = comparisons["k220_vs_k260"]["strict"]
    strict_240 = comparisons["k240_vs_k260"]["strict"]
    practical = worst([row["practical"] for row in comparisons.values()])
    hard_fail = (
        practical == "FAIL" or strict_240 == "FAIL" or trend_failures
        or ordering["status"] == "FAIL" or quality["status"] == "FAIL"
        or ambiguity_fail
    )
    if hard_fail:
        verdict = "FAIL_C_nonmonotonic_or_unresolved"
    elif strict_220 == strict_240 == "PASS" and ordering["status"] == "PASS":
        verdict = "PASS_A_convincing_approach"
    else:
        verdict = "BORDERLINE_B_still_converging"

    return {
        "analysis_kind": "preregistered-n71-k260-retained-basis-audit",
        "criteria_sha256": EXPECTED_CRITERIA_SHA256,
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "endpoint_semantics": criteria["endpoint_semantics"],
        "scope": criteria["analysis_scope"],
        "checkpoint": str(Path(checkpoint_dir).resolve()),
        "checkpoint_sha256": sha256(Path(checkpoint_dir) / "checkpoint.json"),
        "complete_verified": True,
        "parent_solver_quality": quality,
        "values_and_trends": trends,
        "state_overlaps_vs_k260": {
            f"k{k}": {
                str(state): float(abs(cases[k]["overlap"][cases[k]["assignment"][state], state]))
                for state in TRACKED_STATES
            } for k in REQUIRED_K
        },
        "subspace_diagnostics": {f"k{k}": cases[k]["subspace"] for k in REQUIRED_K},
        "operator_blocks": {f"k{k}": blocks[k] for k in REQUIRED_K},
        "endpoint_comparisons": comparisons,
        "path_ordering": ordering,
        "tracking_ambiguity": ambiguity,
        "final_trend_failures": trend_failures,
        "overall_retained_basis_verdict": verdict,
        "scientific_guardrails": criteria["scientific_guardrails"],
    }


def schema_self_test():
    criteria = load_criteria()
    strict = criteria["thresholds"]["strict"]
    practical = criteria["thresholds"]["practical"]
    assert classify_max(1.0, strict["standard_carrier_delta_MHz"]) == "PASS"
    assert classify_max(1.000001, strict["standard_carrier_delta_MHz"]) == "BORDERLINE"
    assert classify_max(5.000001, strict["standard_carrier_delta_MHz"]) == "FAIL"
    assert classify_max(0.5, strict["fine_carrier_delta_MHz"]) == "PASS"
    assert classify_max(1.0, strict["fine_carrier_delta_MHz"]) == "BORDERLINE"
    assert classify_max(1.000001, strict["fine_carrier_delta_MHz"]) == "FAIL"
    assert classify_min(0.999, strict["state_or_subspace_overlap"]) == "PASS"
    assert classify_min(0.995, strict["state_or_subspace_overlap"]) == "BORDERLINE"
    assert classify_min(0.994999, strict["state_or_subspace_overlap"]) == "FAIL"
    assert classify_max(0.03, strict["edge_or_operator_block_relative_change"]) == "PASS"
    assert classify_max(0.05, strict["edge_or_operator_block_relative_change"]) == "BORDERLINE"
    assert classify_max(0.050001, strict["edge_or_operator_block_relative_change"]) == "FAIL"
    assert classify_max(5.0, practical["standard_carrier_delta_MHz"]) == "PASS"
    assert classify_max(10.0, practical["standard_carrier_delta_MHz"]) == "BORDERLINE"
    assert REQUIRED_K == (180, 200, 220, 240, 260)
    print("PASS: k260 analyzer schema and frozen threshold boundaries; no scientific mock data used.")


def write_outputs(output_dir, result):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "k260_analysis_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    rows = []
    for comparison, payload in result["endpoint_comparisons"].items():
        for component in payload["components"]:
            rows.append({"comparison": comparison, **component})
    with (output_dir / "k260_criterion_classification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Overall retained-basis verdict: {result['overall_retained_basis_verdict']}")
    print(f"JSON: {output_dir / 'k260_analysis_summary.json'}")
    print(f"CSV:  {output_dir / 'k260_criterion_classification.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "k260_return_analysis")
    parser.add_argument("--schema-self-test", action="store_true")
    args = parser.parse_args()
    if args.schema_self_test:
        if args.checkpoint is not None:
            parser.error("checkpoint is not accepted with --schema-self-test")
        schema_self_test()
        return 0
    if args.checkpoint is None:
        parser.error("checkpoint is required unless --schema-self-test is used")
    result = analyze(args.checkpoint)
    write_outputs(args.output_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
