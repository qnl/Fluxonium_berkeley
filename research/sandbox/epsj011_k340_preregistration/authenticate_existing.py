"""Read-only authentication of the epsilon_J=0.11 k300 evidence."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKPOINT = ROOT / "research/checkpoints/four-mode-asymmetric-n71-epsj011-k300"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate():
    criteria = json.loads((HERE / "criteria.json").read_text())
    expected = criteria["authenticated_inputs"]
    paths = {
        "epsj011_k300_checkpoint_sha256": CHECKPOINT / "checkpoint.json",
        "epsj011_k300_complete_sha256": CHECKPOINT / "COMPLETE",
        "epsj011_k300_stage1_sha256": CHECKPOINT / "stage1_eigensystem.npz",
        "epsj011_k300_operators_sha256": CHECKPOINT / "projected_operators.npz",
        "epsj011_k300_analysis_sha256": CHECKPOINT / "analysis.json",
        "epsj011_k300_run_state_sha256": CHECKPOINT / "run_state.json",
        "parameter_criteria_sha256": ROOT / "research/sandbox/parameter_point_validation/criteria.json",
        "parameter_evaluation_sha256": ROOT / "research/sandbox/parameter_point_validation/evaluation.json",
        "parameter_metrics_sha256": ROOT / "research/sandbox/parameter_point_validation/parameter_point_metrics.json",
        "parameter_selection_sha256": ROOT / "research/sandbox/parameter_point_validation/selection.json",
        "parameter_selection_seal_sha256": ROOT / "research/sandbox/parameter_point_validation/selection.json.seal.json",
        "shell_summary_sha256": ROOT / "research/sandbox/epsj011_shell_attribution/summary.json",
        "shell_methodology_sha256": ROOT / "research/sandbox/epsj011_shell_attribution/methodology.md",
        "shell_input_hashes_sha256": ROOT / "research/sandbox/epsj011_shell_attribution/input_hashes.json",
    }
    mismatches = {
        name: {"expected": expected[name], "observed": sha(path)}
        for name, path in paths.items()
        if not path.is_file() or sha(path) != expected[name]
    }
    source_mismatches = {
        rel: {"expected": digest, "observed": sha(ROOT / rel)}
        for rel, digest in expected["required_source_sha256"].items()
        if not (ROOT / rel).is_file() or sha(ROOT / rel) != digest
    }
    metadata = json.loads((CHECKPOINT / "checkpoint.json").read_text())
    complete = json.loads((CHECKPOINT / "COMPLETE").read_text())
    evaluation = json.loads(
        (ROOT / "research/sandbox/parameter_point_validation/evaluation.json").read_text()
    )
    shell = json.loads(
        (ROOT / "research/sandbox/epsj011_shell_attribution/summary.json").read_text()
    )
    semantic_checks = {
        "checkpoint_complete": metadata.get("status") == "complete",
        "complete_marker_links_checkpoint": complete.get("checkpoint_sha256") == sha(CHECKPOINT / "checkpoint.json"),
        "configuration_matches": metadata.get("configuration_sha256") == "8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646",
        "parent_k_is_300": metadata.get("configuration", {}).get("solver_settings", {}).get("k") == 300,
        "solver_path": metadata.get("stage1_solver_diagnostics", {}).get("path") == "explicit_shift_invert",
        "solver_permc": metadata.get("stage1_solver_diagnostics", {}).get("permc_spec") == "MMD_AT_PLUS_A",
        "parent_quality": metadata.get("validation", {}).get("maximum_eigenpair_residual", 1) <= 1e-7 and metadata.get("validation", {}).get("orthogonality_residual_inf", 1) <= 1e-8,
        "retained_convergence_fail": evaluation.get("retained_basis_convergence", {}).get("verdict") == "FAIL",
        "selector_borderline": evaluation.get("parameter_point_selector_validation", {}).get("verdict") == "BORDERLINE",
        "multi_target_partial": evaluation.get("multi_target_transfer", {}).get("verdict") == "PARTIAL",
        "shell_mechanism_supported": shell.get("cross_parameter", {}).get("status") == "SUPPORTED",
        "schur_reconstruction_pass": shell.get("schur_reconstruction") == "PASS",
    }
    return {
        "status": "PASS" if not mismatches and not source_mismatches and all(semantic_checks.values()) else "FAIL",
        "hash_mismatches": mismatches,
        "source_mismatches": source_mismatches,
        "semantic_checks": semantic_checks,
        "configuration_sha256": metadata.get("configuration_sha256"),
        "checkpoint_sha256": sha(CHECKPOINT / "checkpoint.json"),
        "max_stage1_residual": metadata["validation"]["maximum_eigenpair_residual"],
        "orthogonality_residual_inf": metadata["validation"]["orthogonality_residual_inf"],
        "total_runtime_seconds": metadata["total_runtime_seconds"],
    }


if __name__ == "__main__":
    print(json.dumps(authenticate(), indent=2, allow_nan=False))
