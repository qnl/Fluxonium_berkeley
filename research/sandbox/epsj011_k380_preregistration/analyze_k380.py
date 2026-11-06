"""Post-checkpoint k380 convergence and shell diagnostics; no Stage-1 solve."""
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
    "epsj011_parameter_analysis_tools_for_k380",
    HERE / "analysis_tools.py",
)
selector = _load(
    "frozen_k340_selector_for_k380_analysis",
    ROOT / "research/sandbox/k300_preregistration/selector.py",
)

STATES = (0, 1, 2, 3, 7, 8)
PREFIXES = (300, 320, 340, 360, 380)
PAIRS = {
    "k300_to_k320": (300, 320),
    "k320_to_k340": (320, 340),
    "k340_to_k360": (340, 360),
    "k360_to_k380": (360, 380),
    "k340_to_k380": (340, 380),
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
    absolute_mode_total = 0.0
    near_mode_total = 0.0
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
        absolute_mode_total += float(np.abs(modal_exact).sum())
        near_mode_total += float(np.abs(modal_exact[near]).sum())
        fixed_drive = u.conj().T @ hpq.conj().T @ base_case['embedded'][:p_stop, state]
        second_order = float(np.sum(abs(fixed_drive)**2 / (base_case['e'][state]-lam))) * 1000
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
            "fixed_vector_second_order_MHz": second_order,
            "second_order_minus_observed_MHz": second_order-observed,
            "dressed_shell_modes": [
                {'mode':i, 'energy_GHz':float(lam[i]), 'signed_denominator_GHz':float(denominators[i]),
                 'coupling_norm_GHz':float(abs(modal_drive[i])/np.sqrt(pnorm)),
                 'eta':float(eta[i]), 'exact_self_energy_MHz':float(modal_exact[i]),
                 'near_resonant':bool(near[i])} for i in range(len(lam))],
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
        "aggregate_near_resonant_fraction": near_mode_total / max(absolute_mode_total, 1e-300),
        "max_schur_equation_residual_GHz": max_schur,
        "max_identity_error_GHz": max_identity,
    }


def nonadditivity(h, base, full, start, stop, criteria):
    """Fixed ten-rank halves; no outcome-dependent choice of ranks."""
    middle = start + 10
    first = tools.selected_solve(h, list(range(start)) + list(range(start,middle)), full['embedded'])
    second = tools.selected_solve(h, list(range(start)) + list(range(middle,stop)), full['embedded'])
    no_cross_h = h[:stop*8,:stop*8].copy()
    no_cross_h[start*8:middle*8,middle*8:stop*8] = 0
    no_cross_h[middle*8:stop*8,start*8:middle*8] = 0
    no_cross = tools.selected_solve(no_cross_h, range(stop), full['embedded'][:stop*8])
    delta = (full['e']-base['e'])*1000
    additive = (first['e']+second['e']-2*base['e'])*1000
    cross = (full['e']-no_cross['e'])*1000
    residual_ok = all(c['residual'] <= criteria['quality']['stage2_residual_GHz_max'] for c in (first,second,no_cross))
    tracking_ok = all(r['state_specific_claim_safe'] for c in (first,second,no_cross) for r in c['confidence'] if r['reference_state'] in STATES)
    return {'partition':[list(range(start,middle)),list(range(middle,stop))],
            'quality_pass':residual_ok and tracking_ok,
            'states':{str(s):{'total_nonadditivity_MHz':float(delta[s]-additive[s]),
                             'explicit_cross_MHz':float(cross[s]),
                             'P_mediated_remainder_MHz':float(delta[s]-additive[s]-cross[s])} for s in STATES},
            'transitions':{name:{'total_nonadditivity_MHz':float((delta[j]-additive[j])-(delta[i]-additive[i])),
                                 'explicit_cross_MHz':float(cross[j]-cross[i])} for name,(i,j) in tools.FREQUENCIES.items()}}


def comparison_trend(comparisons, criteria):
    first = comparisons["k340_to_k360"]
    final = comparisons["k360_to_k380"]
    rows = {}
    for key in first["components"]:
        a = first["components"][key]
        b = final["components"][key]
        def error(row):
            value = float(row["value"])
            return abs(1.0 - value) if key.startswith(("overlap:", "subspace:")) else abs(value)
        rows[key] = {
            "k340_to_k360_error": error(a),
            "k360_to_k380_error": error(b),
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
    from authenticate_existing import authenticate_new
    authenticate_new(args.checkpoint, args.selection)
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
    if values.shape != (380,) or any(value.shape != (380, 380) for value in operators.values()):
        raise ValueError("Unexpected k380 payload dimensions")
    physical = metadata["configuration"]["physical_parameters"]
    ec, stiffness, _, _ = _coeffs(
        physical["EJ"], physical["EC"], physical["EL"], physical["ELK"],
        physical["EJS"], physical["ECS"], physical["eps_J"],
        physical["eps_LK"], physical["eC"], physical["eP"],
    )
    h, grid_phi = selector.build_hamiltonian(values, operators, ec, stiffness)
    tracking = analysis["nested_k340_to_k380_tracking"]
    endpoint = tools.reorder_endpoint(
        tools.selected_solve(h, range(380)),
        tracking["assignment_candidate_row_for_reference_state"],
        tracking["confidence"],
    )
    cases = {380: endpoint}
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
        "S1": shell_diagnostic(h, cases[340], cases[360], 340, 360, criteria),
        "S2": shell_diagnostic(h, cases[360], cases[380], 360, 380, criteria),
    }
    selector_endpoints = {
        "S1": cases[360],
        "S2": tools.selected_solve(h, list(range(340)) + list(range(360, 380)), endpoint["embedded"]),
        "union": endpoint,
    }
    selector_metrics = {name: tools.metrics(case, grid_phi, criteria) for name, case in selector_endpoints.items()}
    selector_comparisons = {name: {} for name in selector_endpoints}
    selection_context = {"S1": "S1", "S2": "S2", "union": "union"}
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
    nonadd = {name: nonadditivity(h, cases[a], cases[b], a, b, criteria) for name,a,b in [('S1',340,360),('S2',360,380)]}
    hard_gates = {
        "nonadditivity_quality": all(row['quality_pass'] for row in nonadd.values()),
        "checkpoint_authentication": True,
        "selection_seal": True,
        "checkpoint_selection_link": True,
        "parent_quality": metadata["validation"]["maximum_eigenpair_residual"] <= criteria["quality"]["stage1_residual_max"] and metadata["validation"]["orthogonality_residual_inf"] <= criteria["quality"]["stage1_orthogonality_inf_max"],
        "stage2_residuals": all(case["residual"] <= criteria["quality"]["stage2_residual_GHz_max"] for case in cases.values()),
        "tracking": all(row["state_specific_claim_safe"] for case in cases.values() for row in case["confidence"] if row["reference_state"] in STATES),
        "original_to_nested_k340_tracking": all(row["state_specific_claim_safe"] for row in analysis["original_k340_vs_nested_k340"]["tracking"]),
        "schur_quality": all(row["max_schur_equation_residual_GHz"] <= criteria["quality"]["schur_equation_GHz_max"] and row["max_identity_error_GHz"] <= criteria["quality"]["schur_identity_GHz_max"] for row in shell_diagnostics.values()),
        "q_reconstruction": all(s['q_vector_error'] <= criteria['quality']['q_vector_max'] for row in shell_diagnostics.values() for s in row['states'].values()),
        "hermiticity": bool(np.max(abs(h-h.conj().T)) <= criteria['quality']['hermiticity_max']),
        "path_ordering": all(comparisons[k]['path_ordering'] != 'FAIL' for k in criteria['primary_comparisons']),
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
        "original_k340_vs_nested_k340": analysis["original_k340_vs_nested_k340"],
        "endpoint_semantics": "k380 is finite and is not ground truth",
        "historical_evidence": json.loads((HERE / 'historical_evidence.json').read_text()),
        "nonadditivity": nonadd,
    }
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
