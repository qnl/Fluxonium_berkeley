"""Post-checkpoint k340 convergence and shell diagnostics; no Stage-1 solve."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from profile import install_profile, EXPECTED_CONFIGURATION_SHA256


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _load(
    "epsj011_parameter_analysis_tools_for_k340",
    ROOT / "research/sandbox/parameter_point_validation/analyze_parameter_point.py",
)
selector = _load(
    "frozen_k300_selector_for_k340_analysis",
    ROOT / "research/sandbox/k300_preregistration/selector.py",
)

STATES = (0, 1, 2, 3, 7, 8)
PREFIXES = (260, 280, 300, 320, 340)
PAIRS = {
    "k260_to_k280": (260, 280),
    "k280_to_k300": (280, 300),
    "k300_to_k320": (300, 320),
    "k320_to_k340": (320, 340),
    "k300_to_k340": (300, 340),
}


def concentration(values):
    weights = np.sort(np.abs(np.asarray(values, float)))[::-1]
    total = float(weights.sum())
    if total == 0:
        return {"absolute_total_MHz": 0.0, "top3_fraction": 0.0,
                "count50": 0, "count80": 0, "count90": 0,
                "effective_count": 0.0}
    fractions = np.cumsum(weights) / total
    normalized = weights / total
    return {
        "absolute_total_MHz": total,
        "top3_fraction": float(weights[:3].sum() / total),
        "count50": int(np.searchsorted(fractions, 0.50) + 1),
        "count80": int(np.searchsorted(fractions, 0.80) + 1),
        "count90": int(np.searchsorted(fractions, 0.90) + 1),
        "effective_count": float(1.0 / np.sum(normalized ** 2)),
    }


def shell_diagnostic(h, base_case, full_case, rank_start, rank_stop, criteria):
    p_stop = rank_start * 8
    q_start, q_stop = rank_start * 8, rank_stop * 8
    hpp = h[:p_stop, :p_stop]
    hpq = h[:p_stop, q_start:q_stop]
    hqq = h[q_start:q_stop, q_start:q_stop]
    lam, u = eigh(hqq, driver="evr")
    state_rows = {}
    aggregate = np.zeros(rank_stop - rank_start)
    max_schur = 0.0
    max_identity = 0.0
    for state in STATES:
        full = full_case["embedded"][:, state]
        p = full[:p_stop]
        q = full[q_start:q_stop]
        energy = float(full_case["e"][state])
        pnorm = float(np.vdot(p, p).real)
        drive = hpq.conj().T @ p
        predicted_q = np.linalg.solve(energy * np.eye(len(hqq)) - hqq, drive)
        q_error = float(np.linalg.norm(predicted_q - q))
        exact = float(np.vdot(drive, q).real / pnorm) * 1000
        observed = float(full_case["e"][state] - base_case["e"][state]) * 1000
        relaxation = float(
            np.vdot(p, hpp @ p).real / pnorm - base_case["e"][state]
        ) * 1000
        schur_residual = float(np.linalg.norm(hpp @ p + hpq @ q - energy * p))
        identity_error = abs(observed - (exact + relaxation)) / 1000
        rank_exact = []
        rank_coupling = []
        rank_probability = []
        for offset in range(rank_stop - rank_start):
            sl = slice(8 * offset, 8 * (offset + 1))
            contribution = float(np.vdot(drive[sl], q[sl]).real / pnorm) * 1000
            rank_exact.append(contribution)
            rank_coupling.append(float(np.linalg.norm(drive[sl]) / np.sqrt(pnorm)))
            rank_probability.append(float(np.vdot(q[sl], q[sl]).real))
            aggregate[offset] += abs(contribution)
        modal_drive = u.conj().T @ drive
        denominators = energy - lam
        modal_exact = (np.abs(modal_drive) ** 2 / denominators / pnorm) * 1000
        eta = np.abs(modal_drive) / (np.sqrt(pnorm) * np.abs(denominators))
        near = (np.abs(denominators) <= criteria["shell_diagnostics"]["near_resonance_denominator_GHz"]) | (eta >= criteria["shell_diagnostics"]["near_resonance_eta"])
        row = {
            "observed_shift_MHz": observed,
            "exact_self_energy_MHz": exact,
            "P_space_relaxation_MHz": relaxation,
            "shell_probability": float(np.vdot(q, q).real),
            "q_vector_error": q_error,
            "schur_equation_residual_GHz": schur_residual,
            "identity_error_GHz": identity_error,
            "minimum_signed_denominator_GHz": float(np.min(denominators)),
            "minimum_abs_denominator_GHz": float(np.min(np.abs(denominators))),
            "maximum_eta": float(np.max(eta)),
            "near_resonant_absolute_weight_fraction": float(np.abs(modal_exact[near]).sum() / max(np.abs(modal_exact).sum(), 1e-300)),
            "rank_participation": concentration(rank_exact),
            "mode_participation": concentration(modal_exact),
            "ranks": [
                {"rank": rank_start + i,
                 "exact_self_energy_MHz": rank_exact[i],
                 "coupling_norm_GHz": rank_coupling[i],
                 "Q_probability": rank_probability[i]}
                for i in range(rank_stop - rank_start)
            ],
        }
        state_rows[str(state)] = row
        max_schur = max(max_schur, schur_residual)
        max_identity = max(max_identity, identity_error)
    order = np.argsort(-aggregate, kind="stable")
    aggregate_sorted = aggregate[order]
    return {
        "rank_range": [rank_start, rank_stop - 1],
        "P_dimension": p_stop,
        "Q_dimension": q_stop - q_start,
        "states": state_rows,
        "aggregate_rank_order": [rank_start + int(i) for i in order],
        "aggregate_rank_absolute_weight_MHz": aggregate_sorted.tolist(),
        "aggregate_rank_participation": concentration(aggregate),
        "max_schur_equation_residual_GHz": max_schur,
        "max_identity_error_GHz": max_identity,
    }


def comparison_trend(comparisons, criteria):
    first = comparisons["k300_to_k320"]
    final = comparisons["k320_to_k340"]
    rows = {}
    for key in first["components"]:
        a = first["components"][key]
        b = final["components"][key]
        def error(row):
            value = float(row["value"])
            return abs(1.0 - value) if key.startswith(("overlap:", "subspace:")) else abs(value)
        rows[key] = {
            "k300_to_k320_error": error(a),
            "k320_to_k340_error": error(b),
            "final_not_larger": error(b) <= error(a) + 1e-12,
        }
    reversals = {}
    for name in tools.FREQUENCIES:
        key = f"carrier:{name}"
        a = float(first["components"][key]["value"])
        b = float(final["components"][key]["value"])
        noise = criteria["cancellation"]["trend_noise_MHz"]
        reversals[name] = {
            "first_increment_MHz": a,
            "final_increment_MHz": b,
            "sign_reversal": bool(a * b < 0 and min(abs(a), abs(b)) > noise),
            "two_shell_cancellation": bool(abs(a + b) <= 0.25 * (abs(a) + abs(b)) and abs(a) + abs(b) > noise),
        }
    return {"components": rows, "carriers": reversals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    criteria = json.loads((HERE / "criteria.json").read_text())
    install_profile(runner)
    metadata = runner.validate_complete_checkpoint_directory(args.checkpoint)
    if metadata["configuration_sha256"] != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError("Configuration mismatch")
    seal_path = Path(str(args.selection) + ".seal.json")
    seal = json.loads(seal_path.read_text())
    if seal["selection_sha256"] != selector.sha(args.selection):
        raise ValueError("Selection seal mismatch")
    selection = json.loads(args.selection.read_text())
    analysis = json.loads((args.checkpoint / "analysis.json").read_text())
    link = analysis.get("selection_freeze", {})
    if link.get("selection_sha256") != seal["selection_sha256"] or not link.get("frozen_before_candidate_or_full_new_shell_dressed_validation"):
        raise ValueError("Checkpoint-selection linkage failed")
    with np.load(args.checkpoint / "stage1_eigensystem.npz", allow_pickle=False) as data:
        values = np.asarray(data["eigenvalues_GHz"])
    with np.load(args.checkpoint / "projected_operators.npz", allow_pickle=False) as data:
        operators = {name: np.asarray(data[name]) for name in data.files}
    if values.shape != (340,) or any(value.shape != (340, 340) for value in operators.values()):
        raise ValueError("Unexpected k340 payload dimensions")
    physical = metadata["configuration"]["physical_parameters"]
    ec, stiffness, _, _ = _coeffs(
        physical["EJ"], physical["EC"], physical["EL"], physical["ELK"],
        physical["EJS"], physical["ECS"], physical["eps_J"],
        physical["eps_LK"], physical["eC"], physical["eP"],
    )
    h, grid_phi = selector.build_hamiltonian(values, operators, ec, stiffness)
    tracking = analysis["nested_k300_to_k340_tracking"]
    endpoint = tools.reorder_endpoint(
        tools.selected_solve(h, range(340)),
        tracking["assignment_candidate_row_for_reference_state"],
        tracking["confidence"],
    )
    cases = {340: endpoint}
    for prefix in PREFIXES[:-1]:
        cases[prefix] = tools.selected_solve(h, range(prefix), endpoint["embedded"])
    metrics = {str(k): tools.metrics(case, grid_phi, criteria) for k, case in cases.items()}
    for row in metrics.values():
        row["f78_minus_f01_MHz"] = row["carriers_MHz"]["f78"] - row["carriers_MHz"]["f01"]
    comparisons = {}
    for label, (lower, upper) in PAIRS.items():
        comparisons[label] = tools.compare(
            cases[lower], cases[upper], metrics[str(lower)], metrics[str(upper)], criteria
        )
        comparisons[label]["f78_minus_f01_increment_MHz"] = metrics[str(upper)]["f78_minus_f01_MHz"] - metrics[str(lower)]["f78_minus_f01_MHz"]
    shell_diagnostics = {
        "new_S3": shell_diagnostic(h, cases[300], cases[320], 300, 320, criteria),
        "new_S4": shell_diagnostic(h, cases[320], cases[340], 320, 340, criteria),
    }
    selector_endpoints = {
        "S3": cases[320],
        "S4": tools.selected_solve(h, list(range(300)) + list(range(320, 340)), endpoint["embedded"]),
        "union": endpoint,
    }
    selector_metrics = {name: tools.metrics(case, grid_phi, criteria) for name, case in selector_endpoints.items()}
    selector_comparisons = {name: {} for name in selector_endpoints}
    selection_context = {"S3": "S1", "S4": "S2", "union": "union"}
    for name, reference in selector_endpoints.items():
        context = selection_context[name]
        for budget in criteria["selector"]["budgets_per_shell"]:
            selector_comparisons[name][str(budget)] = {}
            methods = ("physics", "prefix", "spread", "spread_reverse")
            for method in methods:
                ranks = selection["spaces"][f"{context}_{method}_{budget}"]
                case = tools.selected_solve(h, ranks, reference["embedded"])
                case_metrics = tools.metrics(case, grid_phi, criteria)
                selector_comparisons[name][str(budget)][method] = tools.compare(
                    case, reference, case_metrics, selector_metrics[name], criteria
                )
    hard_gates = {
        "checkpoint_authentication": True,
        "selection_seal": True,
        "checkpoint_selection_link": True,
        "parent_quality": metadata["validation"]["maximum_eigenpair_residual"] <= criteria["quality"]["stage1_residual_max"] and metadata["validation"]["orthogonality_residual_inf"] <= criteria["quality"]["stage1_orthogonality_inf_max"],
        "stage2_residuals": all(case["residual"] <= criteria["quality"]["stage2_residual_GHz_max"] for case in cases.values()),
        "tracking": all(row["state_specific_claim_safe"] for case in cases.values() for row in case["confidence"] if row["reference_state"] in STATES),
        "original_to_nested_k300_tracking": all(row["state_specific_claim_safe"] for row in analysis["original_k300_vs_nested_k300"]["tracking"]),
        "schur_quality": all(row["max_schur_equation_residual_GHz"] <= criteria["quality"]["schur_equation_GHz_max"] and row["max_identity_error_GHz"] <= criteria["quality"]["schur_identity_GHz_max"] for row in shell_diagnostics.values()),
    }
    payload = {
        "schema": 1,
        "configuration_sha256": metadata["configuration_sha256"],
        "selection_sha256": seal["selection_sha256"],
        "hard_gates": hard_gates,
        "prefix_metrics": metrics,
        "comparisons": comparisons,
        "trend": comparison_trend(comparisons, criteria),
        "shell_diagnostics": shell_diagnostics,
        "selector_secondary": {"comparisons": selector_comparisons},
        "original_k300_vs_nested_k300": analysis["original_k300_vs_nested_k300"],
        "endpoint_semantics": "k340 is finite and is not ground truth",
    }
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
