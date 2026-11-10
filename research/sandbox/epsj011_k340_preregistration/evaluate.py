"""Frozen primary convergence, tail-hypothesis, and secondary-selector verdicts."""
from pathlib import Path
import argparse
import json
import math

HERE = Path(__file__).resolve().parent
PRIMARY = ("k300_to_k320", "k320_to_k340", "k300_to_k340")
CONTEXTS = ("S3", "S4", "union")
FAMILIES = {
    "logical_01": ("carrier:f01", "overlap:0", "overlap:1", "subspace:0_1", "centered:0", "centered:1", "excitation:1"),
    "comparator_23": ("carrier:f21", "carrier:f03", "overlap:2", "overlap:3", "subspace:2_3", "centered:2", "centered:3", "excitation:2", "excitation:3"),
    "primary_78": ("carrier:f78", "overlap:7", "overlap:8", "subspace:7_8", "centered:7", "centered:8", "excitation:7", "excitation:8"),
    "cross_carriers": ("carrier:f07", "carrier:f08", "carrier:f81", "carrier:f72", "carrier:f38"),
    "operators": ("prefix:edge:", "prefix:block:"),
    "pathways": ("prefix:score:",),
}


def _members(errors, patterns):
    return [value for key, value in errors.items() if any(key == pattern or (pattern.startswith("prefix:") and key.startswith(pattern[7:])) for pattern in patterns)]


def _family_rms(errors):
    result = {}
    for name, patterns in FAMILIES.items():
        values = _members(errors, patterns)
        if not values:
            raise ValueError(f"Missing selector family {name}")
        result[name] = math.sqrt(sum(value * value for value in values) / len(values))
    return result


def retained_verdict(data):
    failures = []
    misses = []
    if not all(data["hard_gates"].values()):
        failures.extend(name for name, value in data["hard_gates"].items() if not value)
    for comparison in PRIMARY:
        for name, row in data["comparisons"][comparison]["components"].items():
            if row["practical"] == "FAIL":
                failures.append(f"{comparison}/{name}: practical FAIL")
            if row["strict"] != "PASS":
                misses.append(f"{comparison}/{name}: {row['strict']}")
            if comparison == "k320_to_k340" and row["strict"] == "FAIL":
                failures.append(f"{comparison}/{name}: final strict FAIL")
    for name, row in data["trend"]["components"].items():
        if not row["final_not_larger"]:
            failures.append(f"{name}: final increment did not shrink")
    for name, row in data["trend"]["carriers"].items():
        if row["sign_reversal"] or row["two_shell_cancellation"]:
            failures.append(f"carrier:{name}: reversal/cancellation")
    verdict = "FAIL" if failures else "BORDERLINE" if misses else "PASS"
    return {"verdict": verdict, "failures": sorted(set(failures)), "strict_misses": sorted(set(misses))}


def tail_hypothesis(data, retained):
    if not all(data["hard_gates"].values()):
        return {"classification": "D_NONMONOTONIC_UNRESOLVED", "reason": "hard or tracking gate failed"}
    if any(row["sign_reversal"] or row["two_shell_cancellation"] for row in data["trend"]["carriers"].values()):
        return {"classification": "D_NONMONOTONIC_UNRESOLVED", "reason": "meaningful reversal or cancellation"}
    influential = {}
    pair_for_shell = {"new_S3": "k300_to_k320", "new_S4": "k320_to_k340"}
    for shell, comparison in pair_for_shell.items():
        practical = any(row["practical"] == "FAIL" for row in data["comparisons"][comparison]["components"].values())
        large_state = any(abs(row["observed_shift_MHz"]) > 10.0 for row in data["shell_diagnostics"][shell]["states"].values())
        influential[shell] = practical or large_state
    if any(influential.values()):
        return {"classification": "B_REPEATED", "influential": influential, "reason": "at least one post-k300 shell is influential"}
    aggregate = [data["shell_diagnostics"][name]["aggregate_rank_participation"] for name in ("new_S3", "new_S4")]
    first_env = max(data["comparisons"]["k300_to_k320"]["strict_units"].values())
    final_env = max(data["comparisons"]["k320_to_k340"]["strict_units"].values())
    distributed = all(row["top3_fraction"] <= 0.50 and row["count80"] >= 8 for row in aggregate)
    cumulative_miss = any(row["strict"] != "PASS" for row in data["comparisons"]["k300_to_k340"]["components"].values())
    if distributed and cumulative_miss and final_env >= 0.5 * first_env:
        return {"classification": "C_DISTRIBUTED_WEAK", "influential": influential, "reason": "weak distributed tail persists cumulatively"}
    if final_env <= first_env:
        return {"classification": "A_ISOLATED", "influential": influential, "reason": "no new influential shell and final envelope does not grow"}
    return {"classification": "D_NONMONOTONIC_UNRESOLVED", "influential": influential, "reason": "A/B/C conditions are not uniquely satisfied"}


def selector_verdict(data, criteria):
    comparisons = data["selector_secondary"]["comparisons"]
    failures, misses, ratios = [], [], {}
    comparison_rule = criteria["selector_comparison"]
    for context in CONTEXTS:
        ratios[context] = {}
        for budget in (10, 15):
            physics = comparisons[context][str(budget)]["physics"]
            if any(row["practical"] == "FAIL" for row in physics["components"].values()):
                failures.append(f"{context}/b{budget}: practical FAIL")
            if physics["path_ordering"] == "FAIL" or not physics["tracking_safe"]:
                failures.append(f"{context}/b{budget}: path/tracking")
            if budget == 15 and any(row["strict"] == "FAIL" for row in physics["components"].values()):
                failures.append(f"{context}/b15: strict FAIL")
            if budget == 15 and any(row["strict"] != "PASS" for row in physics["components"].values()):
                misses.append(f"{context}/b15: strict miss")
        b10 = comparisons[context]["10"]
        physics_family = _family_rms(b10["physics"]["strict_units"])
        controls = {name: _family_rms(row["strict_units"]) for name, row in b10.items() if name != "physics"}
        for family, value in physics_family.items():
            prefix = controls["prefix"][family]
            best = min(row[family] for row in controls.values())
            vs_prefix = value / prefix if prefix else (0.0 if value == 0 else math.inf)
            vs_best = value / best if best else (0.0 if value == 0 else math.inf)
            ratios[context][family] = {"vs_prefix": vs_prefix, "vs_best_control": vs_best}
            if vs_prefix >= comparison_rule["borderline_improvement_ratio"]:
                failures.append(f"{context}/{family}: no prefix improvement")
            if vs_best > comparison_rule["improvement_pass_ratio"]:
                misses.append(f"{context}/{family}: misses 20 percent improvement")
    return {"verdict": "FAIL" if failures else "BORDERLINE" if misses else "PASS", "failures": sorted(set(failures)), "pass_requirements_missed": sorted(set(misses)), "family_ratios": ratios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    criteria = json.loads((HERE / "criteria.json").read_text())
    data = json.loads(args.metrics_json.read_text())
    retained = retained_verdict(data)
    output = {
        "retained_basis_convergence": retained,
        "tail_hypothesis": tail_hypothesis(data, retained),
        "selector_secondary": selector_verdict(data, criteria),
        "claims": {
            "smaller_final_increment": "report measured trend only",
            "plateau_within_tested_range": retained["verdict"] == "PASS",
            "convergence_beyond_k340": False,
            "other_spatial_or_model_convergence": False,
            "k340_is_ground_truth": False,
        },
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
