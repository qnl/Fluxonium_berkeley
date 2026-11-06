"""Comparative finite-Stage-2 Schur attribution; never runs Stage-1."""
from pathlib import Path
from itertools import combinations
from importlib.util import module_from_spec, spec_from_file_location
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW_CP = ROOT / "research/checkpoints/four-mode-asymmetric-n71-epsj011-k300"
OLD_CP = ROOT / "research/checkpoints/four-mode-asymmetric-n71-k300"
PARAM = ROOT / "research/sandbox/parameter_point_validation"
OLD_ATTR = ROOT / "research/sandbox/k300_shell_attribution"
METHOD_SHA256 = "1df2a8d19a7b11c4ea35195f63650eb31c0b1baff4f57dc03e4577adf3bcdbbc"
INPUT_HASHES_SHA256 = "86cea52b6585b692c0d79ca7a7b26d82a965520a9fdad9dd3b8f8b7a9471d84c"
CONFIGURATION_SHA256 = "8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646"
STATES = [0, 1, 2, 3, 7, 8]
FREQ = {"f01": (0, 1), "f07": (0, 7), "f08": (0, 8), "f78": (7, 8),
        "f81": (1, 8), "f72": (2, 7), "f21": (1, 2), "f03": (0, 3),
        "f38": (3, 8)}

spec = spec_from_file_location("validated_k300_attribution", OLD_ATTR / "attribute.py")
validated = module_from_spec(spec)
spec.loader.exec_module(validated)

sys.path.insert(0, str(ROOT))
from Circuit_Objs.qchard_gridium_netlist import _coeffs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tracked_snapshot():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {name: sha(ROOT / name) for name in names if name and (ROOT / name).is_file()}


def tree_snapshot(root):
    return {str(path.relative_to(ROOT)): sha(path) for path in root.rglob("*") if path.is_file()}


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs():
    if sha(HERE / "methodology.md") != METHOD_SHA256:
        raise RuntimeError("methodology hash changed")
    if sha(HERE / "input_hashes.json") != INPUT_HASHES_SHA256:
        raise RuntimeError("input-hash record changed")
    expected = json.loads((HERE / "input_hashes.json").read_text())
    paths = {
        "new_checkpoint_json": NEW_CP / "checkpoint.json",
        "new_complete": NEW_CP / "COMPLETE",
        "new_stage1_eigensystem": NEW_CP / "stage1_eigensystem.npz",
        "new_projected_operators": NEW_CP / "projected_operators.npz",
        "new_analysis": NEW_CP / "analysis.json",
        "new_authentication": PARAM / "authentication_post_adapter.json",
        "new_selection": PARAM / "selection.json",
        "new_selection_seal": PARAM / "selection.json.seal.json",
        "new_validation_metrics": PARAM / "parameter_point_metrics.json",
        "new_evaluation": PARAM / "evaluation.json",
        "original_checkpoint_json": OLD_CP / "checkpoint.json",
        "original_stage1_eigensystem": OLD_CP / "stage1_eigensystem.npz",
        "original_methodology": OLD_ATTR / "methodology.md",
        "original_summary": OLD_ATTR / "summary.json",
        "original_rank_attributions": OLD_ATTR / "rank_attributions.csv",
        "original_selector_diagnostics": OLD_ATTR / "selector_diagnostics.csv",
        "original_frequency_selectivity": OLD_ATTR / "frequency_selectivity.csv",
        "original_shell_mode_contributions": OLD_ATTR / "shell_mode_contributions.csv",
        "qchard_source": ROOT / "Circuit_Objs/qchard_gridium_netlist.py",
        "stage2_runner_source": ROOT / "Simulations/Rabi_3Photon/run_four_mode_spatial71_reference.py",
    }
    for name, path in paths.items():
        if sha(path) != expected[name]:
            raise RuntimeError(f"authenticated input changed: {name}")


def stage1_rank_tracking():
    with np.load(NEW_CP / "stage1_eigensystem.npz", allow_pickle=False) as data:
        new_window = np.asarray(data["eigenvectors"][:, 240:300]).copy()
    with np.load(OLD_CP / "stage1_eigensystem.npz", allow_pickle=False) as data:
        old_all = np.asarray(data["eigenvectors"])
        overlap = abs(new_window.conj().T @ old_all)
    rows, cols = linear_sum_assignment(-overlap)
    assignment = dict(zip(rows, cols))
    records = []
    for row in range(60):
        col = int(assignment[row])
        assigned = float(overlap[row, col])
        row_comp = float(np.max(np.delete(overlap[row], col)))
        col_comp = float(np.max(np.delete(overlap[:, col], row)))
        competitor = max(row_comp, col_comp)
        ratio = competitor / assigned
        margin = assigned - competitor
        records.append({
            "new_rank": row + 240,
            "old_rank": col,
            "assigned_overlap": assigned,
            "best_competing_overlap": competitor,
            "competitor_ratio": ratio,
            "overlap_margin": margin,
            "same_numeric_rank": col == row + 240,
            "safe": assigned >= 0.90 and ratio < 0.50 and margin > 0.25,
        })
    return records


def build_hamiltonian(meta):
    with np.load(NEW_CP / "stage1_eigensystem.npz", allow_pickle=False) as data:
        values = np.asarray(data["eigenvalues_GHz"]).copy()
    with np.load(NEW_CP / "projected_operators.npz", allow_pickle=False) as data:
        operators = {name: np.asarray(data[name]) for name in data.files}
    p = meta["configuration"]["physical_parameters"]
    ec, coupling, _, _ = _coeffs(p["EJ"], p["EC"], p["EL"], p["ELK"],
        p["EJS"], p["ECS"], p["eps_J"], p["eps_LK"], p["eC"], p["eP"])
    aa, bb = 4 * ec[3, 3], 0.5 * coupling[2, 2]
    scale_x = (aa / (4 * bb)) ** 0.25
    scale_p = 1 / (2 * scale_x)
    ladder = np.diag(np.sqrt(np.arange(1, 8)), 1)
    x4 = scale_x * (ladder + ladder.T)
    p4 = 1j * scale_p * (ladder.T - ladder)
    w4 = 2 * np.sqrt(aa * bb) * (np.arange(8) + 0.5)
    h = np.kron(np.diag(values), np.eye(8)) + np.kron(np.eye(300), np.diag(w4))
    h = h.astype(complex)
    for index in range(3):
        h += 8 * ec[index, 3] * np.kron(operators[f"n{index + 1}"], p4)
    h += coupling[0, 2] * np.kron(operators["x2"], x4)
    h += coupling[1, 2] * np.kron(operators["x3"], x4)
    hermiticity = float(np.max(abs(h - h.conj().T)))
    if hermiticity > 1e-12:
        raise RuntimeError(f"Hermiticity failure: {hermiticity}")
    h = (h + h.conj().T) / 2
    grid_phi = np.kron(0.5 * operators["x2"] + operators["x3"], np.eye(8))
    return h, grid_phi, hermiticity


def subset_addbacks(h, base, endpoint, influential):
    subsets = {}
    for size in (1, 2, 3):
        for subset in combinations(influential, size):
            shell = np.concatenate([np.arange(rank * 8, (rank + 1) * 8) for rank in subset])
            indices = np.r_[np.arange(2240), shell]
            subsets[subset] = validated.solve(h[np.ix_(indices, indices)], indices, endpoint["embedded"])
    output = {"influential_ranks": influential, "states": {}, "transitions": {}}
    for state in STATES:
        e0 = base["e"][state]
        effects = {",".join(map(str, key)): float((case["e"][state] - e0) * 1000)
                   for key, case in subsets.items()}
        pairs = {",".join(map(str, pair)): float((subsets[pair]["e"][state]
                 - subsets[(pair[0],)]["e"][state] - subsets[(pair[1],)]["e"][state] + e0) * 1000)
                 for pair in combinations(influential, 2)}
        a, b, c = influential
        triple = float((subsets[(a, b, c)]["e"][state] - subsets[(a, b)]["e"][state]
            - subsets[(a, c)]["e"][state] - subsets[(b, c)]["e"][state]
            + subsets[(a,)]["e"][state] + subsets[(b,)]["e"][state]
            + subsets[(c,)]["e"][state] - e0) * 1000)
        output["states"][str(state)] = {"effects_MHz": effects,
            "pair_interactions_MHz": pairs, "triple_interaction_MHz": triple}
    for name, pair in FREQ.items():
        f0 = validated.frequency(base, pair)
        effects = {",".join(map(str, key)): validated.frequency(case, pair) - f0
                   for key, case in subsets.items()}
        pairs = {",".join(map(str, ranks)): validated.frequency(subsets[ranks], pair)
                 - validated.frequency(subsets[(ranks[0],)], pair)
                 - validated.frequency(subsets[(ranks[1],)], pair) + f0
                 for ranks in combinations(influential, 2)}
        a, b, c = influential
        triple = (validated.frequency(subsets[(a, b, c)], pair)
            - validated.frequency(subsets[(a, b)], pair)
            - validated.frequency(subsets[(a, c)], pair)
            - validated.frequency(subsets[(b, c)], pair)
            + validated.frequency(subsets[(a,)], pair)
            + validated.frequency(subsets[(b,)], pair)
            + validated.frequency(subsets[(c,)], pair) - f0)
        output["transitions"][name] = {"effects_MHz": effects,
            "pair_interactions_MHz": pairs, "triple_interaction_MHz": triple}
    return output


def rank_maps(rows):
    by_shell = {}
    for row in rows:
        by_shell.setdefault(row["shell"], {}).setdefault(row["rank"], {})[str(row["state"])] = row
    return by_shell


def selector_alignment(selection, shell_rows):
    rows = []
    summary = {}
    for shell in ("S1", "S2"):
        data = selection["shells"][shell]
        exact = {rank: 0.0 for rank in data["order"]}
        signed = {rank: {str(state): 0.0 for state in STATES} for rank in data["order"]}
        for record in shell_rows[shell]:
            exact[record["rank"]] += abs(record["exact_self_energy_MHz"])
            signed[record["rank"]][str(record["state"])] = record["exact_self_energy_MHz"]
        total = sum(exact.values())
        exact_order = sorted(exact, key=lambda rank: (-exact[rank], rank))
        order_index = {rank: index + 1 for index, rank in enumerate(data["order"])}
        gain = {record["rank"]: record["gain"] for record in data["greedy_trace"]}
        feature = {record["rank"]: record for record in data["features"]}
        summary[shell] = {"physics_order": data["order"], "exact_aggregate_order": exact_order,
            "top5_overlap_count": len(set(data["order"][:5]) & set(exact_order[:5])), "budgets": {}}
        controls = {method: {rank: index + 1 for index, rank in
                    enumerate(selection["spaces"][f"{shell}_{method}_15"][260:])}
                    for method in ("prefix", "spread", "spread_reverse")}
        for budget in (3, 6, 10, 15):
            chosen = set(data["order"][:budget])
            summary[shell]["budgets"][str(budget)] = {
                "absolute_exact_rank_weight_capture": sum(exact[rank] for rank in chosen) / total,
                "selected_exact_top_ranks": sorted(chosen & set(exact_order[:budget])),
            }
        for rank in sorted(exact):
            groups = {tuple(group["states"]): group for group in feature[rank]["groups"]}
            row = {"shell": shell, "rank": rank, "physics_order": order_index[rank],
                "physics_gain": gain[rank], "exact_aggregate_abs_self_energy_MHz": exact[rank],
                "exact_aggregate_fraction": exact[rank] / total}
            for state in STATES:
                row[f"exact_state_{state}_MHz"] = signed[rank][str(state)]
            for pair, key in [((0, 1), "01"), ((2, 3), "23"), ((7, 8), "78")]:
                group = groups[pair]
                row.update({f"group_{key}_trace_GHz": group["trace_GHz"],
                    f"group_{key}_coupling_GHz": group["coupling_norm_GHz"],
                    f"group_{key}_min_den_GHz": group["minimum_denominator_GHz"],
                    f"group_{key}_max_eta": group["maximum_eta"]})
            for budget in (3, 6, 10, 15):
                row[f"physics_b{budget}"] = rank in set(data["order"][:budget])
            for method, order in controls.items():
                row[f"{method}_order"] = order.get(rank)
            rows.append(row)
    return summary, rows


def solve_rank_set(h, ranks, endpoint):
    shell = np.concatenate([np.arange(rank * 8, (rank + 1) * 8) for rank in sorted(ranks)])
    indices = np.r_[np.arange(2080), shell]
    return validated.solve(h[np.ix_(indices, indices)], indices, endpoint["embedded"])


def union_exception(h, cases, full, selection, validation_metrics, shell_rows):
    physics_ranks = [rank for rank in selection["spaces"]["union_physics_10"] if rank >= 260]
    prefix_ranks = [rank for rank in selection["spaces"]["union_global_prefix_10"] if rank >= 260]
    methods = {}
    for method, ranks in (("physics", physics_ranks), ("prefix", prefix_ranks)):
        case = solve_rank_set(h, ranks, full)
        raw = {str(state): float((case["e"][state] - full["e"][state]) * 1000) for state in STATES}
        excitation = {str(state): float(((case["e"][state] - case["e"][0])
            - (full["e"][state] - full["e"][0])) * 1000) for state in STATES}
        transition = {name: validated.frequency(case, pair) - validated.frequency(full, pair)
                      for name, pair in FREQ.items()}
        saved = validation_metrics["comparisons"]["union"]["10"][method]["components"]["carrier:f78"]["value"]
        if abs(transition["f78"] - saved) > 1e-7:
            raise RuntimeError(("union f78 reconstruction", method, transition["f78"], saved))
        methods[method] = {"ranks": ranks, "raw_state_errors_MHz": raw,
            "excitation_errors_MHz": excitation, "transition_errors_MHz": transition,
            "f78_cancellation_ratio": abs(transition["f78"])
                / (abs(excitation["7"]) + abs(excitation["8"])),
            "saved_f78_error_MHz": saved}
    attribution = rank_maps(shell_rows)
    for method, record in methods.items():
        selected = set(record["ranks"])
        record["isolated_schur_accounting"] = {}
        for shell, label in (("S1", "S1_isolated_260_279"), ("S2", "S2_isolated_280_299")):
            ranks = range(260, 280) if shell == "S1" else range(280, 300)
            record["isolated_schur_accounting"][shell] = {}
            for state in STATES:
                terms = {rank: attribution[label][rank][str(state)]["exact_self_energy_MHz"] for rank in ranks}
                record["isolated_schur_accounting"][shell][str(state)] = {
                    "selected_signed_MHz": sum(value for rank, value in terms.items() if rank in selected),
                    "omitted_signed_MHz": sum(value for rank, value in terms.items() if rank not in selected),
                    "selected_absolute_MHz": sum(abs(value) for rank, value in terms.items() if rank in selected),
                    "omitted_absolute_MHz": sum(abs(value) for rank, value in terms.items() if rank not in selected),
                }
    methods["common_ranks"] = sorted(set(physics_ranks) & set(prefix_ranks))
    methods["physics_only"] = sorted(set(physics_ranks) - set(prefix_ranks))
    methods["prefix_only"] = sorted(set(prefix_ranks) - set(physics_ranks))
    p7, p8 = methods["physics"]["excitation_errors_MHz"]["7"], methods["physics"]["excitation_errors_MHz"]["8"]
    q7, q8 = methods["prefix"]["excitation_errors_MHz"]["7"], methods["prefix"]["excitation_errors_MHz"]["8"]
    if max(abs(q7), abs(q8)) > 5 * max(abs(p7), abs(p8)) and methods["prefix"]["f78_cancellation_ratio"] <= 0.25:
        methods["classification"] = "B: cancellation of larger individual errors"
    elif abs(methods["prefix"]["transition_errors_MHz"]["f78"]) < abs(methods["physics"]["transition_errors_MHz"]["f78"]):
        methods["classification"] = "C: mixed behavior"
    else:
        methods["classification"] = "D: unresolved evidence"
    return methods


def cross_parameter_rows(current_rows, tracking, selection, old_summary):
    tracking_by_new = {row["new_rank"]: row for row in tracking}
    with (OLD_ATTR / "rank_attributions.csv").open() as stream:
        old_rows = list(csv.DictReader(stream))
    old_rank = {(row["shell"], int(row["rank"]), int(row["state"])): row for row in old_rows}
    with (OLD_ATTR / "selector_diagnostics.csv").open() as stream:
        old_selector = {int(row["rank"]): row for row in csv.DictReader(stream) if row["shell"] == "S2"}
    new_features = {row["rank"]: row for row in selection["shells"]["S2"]["features"]}
    output = []
    for row in current_rows:
        if row["shell"] != "P280_Q280_299":
            continue
        new_rank = row["rank"]
        track = tracking_by_new[new_rank]
        old_numeric = track["old_rank"]
        previous = old_rank[("P280_Q280_299", old_numeric, row["state"])]
        groups = {tuple(group["states"]): group for group in new_features[new_rank]["groups"]}
        old_feature = old_selector[old_numeric]
        output.append({
            "new_rank": new_rank, "matched_old_rank": old_numeric, "state": row["state"],
            "rank_tracking_overlap": track["assigned_overlap"], "rank_tracking_safe": track["safe"],
            "new_exact_self_energy_MHz": row["exact_self_energy_MHz"],
            "old_exact_self_energy_MHz": float(previous["exact_self_energy_MHz"]),
            "new_coupling_norm_GHz": row["coupling_norm_GHz"],
            "old_coupling_norm_GHz": float(previous["coupling_norm_GHz"]),
            "new_Q_probability": row["Q_probability"],
            "old_Q_probability": float(previous["Q_probability"]),
            "new_group78_coupling_GHz": groups[(7, 8)]["coupling_norm_GHz"],
            "old_group78_coupling_GHz": float(old_feature["group_78_coupling_GHz"]),
            "new_group78_min_den_GHz": groups[(7, 8)]["minimum_denominator_GHz"],
            "old_group78_min_den_GHz": float(old_feature["group_78_min_den_GHz"]),
        })
    return output


def main():
    verify_inputs()
    tracked_before = tracked_snapshot()
    checkpoints_before = {**tree_snapshot(NEW_CP), **tree_snapshot(OLD_CP)}

    authentication = json.loads((PARAM / "authentication_post_adapter.json").read_text())
    meta = json.loads((NEW_CP / "checkpoint.json").read_text())
    complete = json.loads((NEW_CP / "COMPLETE").read_text())
    saved_analysis = json.loads((NEW_CP / "analysis.json").read_text())
    selection = json.loads((PARAM / "selection.json").read_text())
    metrics = json.loads((PARAM / "parameter_point_metrics.json").read_text())
    evaluation = json.loads((PARAM / "evaluation.json").read_text())
    old_summary = json.loads((OLD_ATTR / "summary.json").read_text())
    if authentication["status"] != "PASS" or not authentication["payload_hashes_verified"]:
        raise RuntimeError("new-point authentication failed")
    if meta["status"] != "complete" or meta["configuration_sha256"] != CONFIGURATION_SHA256:
        raise RuntimeError("checkpoint metadata")
    if complete["checkpoint_sha256"] != sha(NEW_CP / "checkpoint.json"):
        raise RuntimeError("COMPLETE linkage")
    for name, digest in meta["file_sha256"].items():
        if sha(NEW_CP / name) != digest:
            raise RuntimeError(f"payload hash failure: {name}")
    freeze = saved_analysis["held_out_selection_freeze"]
    if (freeze["selection_sha256"] != sha(PARAM / "selection.json")
            or freeze["seal_sha256"] != sha(PARAM / "selection.json.seal.json")
            or not freeze["frozen_before_candidate_or_full_stage2"]):
        raise RuntimeError("selection freeze linkage")
    prefixes = sorted(int(key.rsplit("k", 1)[1]) for key in saved_analysis["retained_basis_cases"])
    if prefixes != [180, 200, 220, 240, 260, 280, 300]:
        raise RuntimeError("nested-prefix schema")

    print("Tracking Stage-1 rank identity across parameter points", flush=True)
    rank_tracking = stage1_rank_tracking()

    h, grid_phi, hermiticity = build_hamiltonian(meta)
    print("Reconstructing authenticated finite Stage-2 cases", flush=True)
    full = validated.solve(h, np.arange(2400))
    cases = {300: full}
    for size in (280, 260):
        cases[size] = validated.solve(h[:8 * size, :8 * size], np.arange(8 * size), full["embedded"])
    reconstruction = {}
    for size, case in cases.items():
        matrix = case["embedded"].conj().T @ grid_phi @ case["embedded"]
        saved = saved_analysis["retained_basis_cases"][f"nested_prefix_k{size}"]
        energy_error = float(max(abs(case["e"] - saved["energies_GHz"])))
        operator_error = float(np.max(abs(abs(matrix) - saved["grid_phi_abs"])))
        if energy_error > 1e-10 or operator_error > 1e-10:
            raise RuntimeError(("reconstruction", size, energy_error, operator_error))
        case["grid_phi"] = matrix
        reconstruction[str(size)] = {"energy_error_GHz": energy_error,
            "grid_phi_error": operator_error, "eigenpair_residual_GHz": case["residual"],
            "tracking": case["confidence"]}

    print("Computing frozen Schur attribution", flush=True)
    central, central_ranks, central_modes = validated.attribute(
        h, 2240, list(range(280, 300)), cases[280], full, "P280_Q280_299")
    s1_h = h[:2240, :2240]
    s1, s1_ranks, s1_modes = validated.attribute(
        s1_h, 2080, list(range(260, 280)), cases[260], cases[280], "S1_isolated_260_279")
    s2_indices = np.r_[np.arange(2080), np.arange(2240, 2400)]
    s2_h = h[np.ix_(s2_indices, s2_indices)]
    s2_full = validated.solve(s2_h, s2_indices, full["embedded"])
    s2, s2_ranks, s2_modes = validated.attribute(
        s2_h, 2080, list(range(280, 300)), cases[260], s2_full, "S2_isolated_280_299")
    rank_rows = central_ranks + s1_ranks + s2_ranks
    mode_rows = central_modes + s1_modes + s2_modes

    central_transitions = validated.transition_attribution(central, cases[280], full)
    s1_transitions = validated.transition_attribution(s1, cases[260], cases[280])
    s2_transitions = validated.transition_attribution(s2, cases[260], s2_full)

    print("Computing nonadditivity and exact rank addbacks", flush=True)
    q1 = validated.solve(h[:2320, :2320], np.arange(2320), full["embedded"])
    q2_indices = np.r_[np.arange(2240), np.arange(2320, 2400)]
    q2 = validated.solve(h[np.ix_(q2_indices, q2_indices)], q2_indices, full["embedded"])
    no_cross_h = h.copy()
    no_cross_h[2240:2320, 2320:] = 0
    no_cross_h[2320:, 2240:2320] = 0
    no_cross = validated.solve(no_cross_h, np.arange(2400), full["embedded"])
    nonadditivity = validated.energy_nonadditivity(cases[280], q1, q2, full, no_cross)
    nonadditivity["Q1Q2_coupling_frobenius_GHz"] = float(np.linalg.norm(h[2240:2320, 2320:]))

    aggregate = {rank: 0.0 for rank in range(280, 300)}
    for row in central_ranks:
        aggregate[row["rank"]] += abs(row["exact_self_energy_MHz"])
    influential = sorted(aggregate, key=lambda rank: (-aggregate[rank], rank))[:3]
    addbacks = subset_addbacks(h, cases[280], full, influential)
    addbacks["aggregate_absolute_self_energy_MHz"] = aggregate

    shell_rows = {"S1": s1_ranks, "S2": s2_ranks}
    selector_summary, selector_rows = selector_alignment(selection, shell_rows)
    exception = union_exception(h, cases, full, selection, metrics, rank_rows)
    cross_rows = cross_parameter_rows(central_ranks, rank_tracking, selection, old_summary)

    frequency_rows = []
    for size in (260, 280, 300):
        case = cases[size]
        row = {"epsilon_J": 0.11, "k": size}
        for name, pair in FREQ.items():
            row[name + "_MHz"] = validated.frequency(case, pair)
        row["f78_minus_f01_MHz"] = row["f78_MHz"] - row["f01_MHz"]
        row["f07_minus_f81_MHz"] = row["f07_MHz"] - row["f81_MHz"]
        row["g01"] = float(abs(case["grid_phi"][0, 1]))
        row["g78"] = float(abs(case["grid_phi"][7, 8]))
        frequency_rows.append(row)
    for old in old_summary["frequency_selectivity"]:
        frequency_rows.append({"epsilon_J": 0.10, **{("f07_minus_f81_MHz" if key == "outer_f07_minus_f81_MHz" else key): value
            for key, value in old.items()}})

    labels = [central[str(state)]["mechanism"] for state in (7, 8)]
    mechanism = "E" if "E" in labels else labels[0] if labels[0] == labels[1] else "D"
    old_mechanism = old_summary["central_final_shell"]["mechanism"]
    shell_tracking_safe = all(row["safe"] for row in rank_tracking if row["new_rank"] >= 280)
    old_top = old_summary["influential_rank_addbacks"]["influential_ranks"]
    mapped_new_for_old = {row["old_rank"]: row["new_rank"] for row in rank_tracking}
    tracked_old_top = [mapped_new_for_old.get(rank) for rank in old_top]
    if mechanism == old_mechanism and mechanism in ("A", "B", "C") and shell_tracking_safe \
            and len(set(influential[:5]) & set(tracked_old_top[:2])) == 2:
        cross_status = "SUPPORTED"
    elif mechanism != "E" and old_mechanism != "E" and shell_tracking_safe:
        cross_status = "PARTIAL"
    else:
        cross_status = "NOT SUPPORTED"

    selector_performance = {}
    for context in ("S1", "S2", "union"):
        selector_performance[context] = {}
        for budget in ("10", "15"):
            physics = metrics["comparisons"][context][budget]["physics"]
            selector_performance[context][budget] = {
                "carrier_errors_MHz": {name: physics["components"]["carrier:" + name]["value"]
                    for name in FREQ},
                "pathway_relative_errors": {name: physics["components"]["score:" + name]["value"]
                    for name in ("0781", "0721", "0381")},
                "path_ranks": physics["metrics"]["path_ranks"],
                "path_ordering": physics["path_ordering"],
                "tracking_safe": physics["tracking_safe"],
            }

    if tracked_snapshot() != tracked_before:
        raise RuntimeError("tracked file mutation")
    if {**tree_snapshot(NEW_CP), **tree_snapshot(OLD_CP)} != checkpoints_before:
        raise RuntimeError("checkpoint mutation")

    result = {
        "schema": 1,
        "methodology_sha256": METHOD_SHA256,
        "input_hashes_sha256": INPUT_HASHES_SHA256,
        "authentication": {"status": "PASS", "configuration_sha256": CONFIGURATION_SHA256,
            "checkpoint_sha256": sha(NEW_CP / "checkpoint.json"), "payload_hashes_verified": True,
            "selection_sha256": sha(PARAM / "selection.json"),
            "selection_seal_sha256": sha(PARAM / "selection.json.seal.json"),
            "selection_frozen_before_validation": True, "nested_prefixes": prefixes,
            "max_stage1_residual": authentication["max_eigenpair_residual"],
            "orthogonality_residual_inf": authentication["orthogonality_residual_inf"]},
        "schur_reconstruction": "PASS",
        "dimensions": {"P": 2240, "Q": 160, "full": 2400, "Q1": 80, "Q2": 80},
        "hermiticity_error": hermiticity,
        "reconstruction": reconstruction,
        "stage1_rank_tracking": rank_tracking,
        "central_final_shell": {"states": central, "transitions": central_transitions,
            "mechanism": mechanism},
        "isolated_selector_shells": {"S1": {"states": s1, "transitions": s1_transitions},
            "S2": {"states": s2, "transitions": s2_transitions}},
        "Q1_Q2_nonadditivity": nonadditivity,
        "influential_rank_addbacks": addbacks,
        "selector_alignment": selector_summary,
        "selector_performance": selector_performance,
        "union_budget10_f78_exception": exception,
        "cross_parameter": {"status": cross_status, "old_mechanism": old_mechanism,
            "new_mechanism": mechanism, "old_influential_ranks": old_top,
            "new_influential_ranks": influential, "tracked_old_top_to_new": tracked_old_top,
            "shell_rank_identity_safe": shell_tracking_safe},
        "frequency_selectivity": frequency_rows,
        "frozen_selector_validation": evaluation["parameter_point_selector_validation"]["verdict"],
        "multi_target_transfer": evaluation["multi_target_transfer"]["verdict"],
        "retained_basis_convergence": evaluation["retained_basis_convergence"]["verdict"],
        "tracked_files_unchanged": True,
        "authenticated_checkpoints_unchanged": True,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(HERE / "rank_attributions.csv", rank_rows)
    write_csv(HERE / "shell_mode_contributions.csv", mode_rows)
    write_csv(HERE / "cross_parameter_comparison.csv", cross_rows)
    write_csv(HERE / "selector_diagnostics.csv", selector_rows)
    write_csv(HERE / "frequency_selectivity.csv", frequency_rows)
    (HERE / "preservation.json").write_text(json.dumps({"tracked": tracked_before,
        "checkpoints": checkpoints_before}, indent=2) + "\n")
    print("Completed", mechanism, cross_status, influential, flush=True)


if __name__ == "__main__":
    main()
