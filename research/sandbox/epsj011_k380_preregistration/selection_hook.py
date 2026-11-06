"""Freeze the unchanged selector on ranks 340-379 before validation."""
from pathlib import Path
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

from parent_analysis import analyze_parent

K340 = ROOT / "research/checkpoints/four-mode-asymmetric-n71-epsj011-k340"
SELECTION = HERE / "selection.json"
SELECTOR_PATH = ROOT / "research/sandbox/k300_preregistration/selector.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = _load("frozen_k340_selector_for_k380", SELECTOR_PATH)
old_profile = _load(
    "epsj011_k340_profile_for_k380",
    ROOT / "research/sandbox/epsj011_k340_preregistration/profile.py",
)


def _metrics(case):
    paths = runner._three_step_paths(case["matrix"], runner.NSTATES)
    wanted = {(0, 7, 8, 1): "0781", (0, 7, 2, 1): "0721", (0, 3, 8, 1): "0381"}
    ranked = {tuple(row["path"]): (i + 1, row["score_product"]) for i, row in enumerate(paths)}
    return {
        "energies_GHz": case["energies"].tolist(),
        "transitions_GHz": case["transitions"].tolist(),
        "grid_phi_abs": abs(case["matrix"]).tolist(),
        "pathways": {name: {"rank": ranked[path][0], "score": ranked[path][1]} for path, name in wanted.items()},
    }


def install_selection_hook():
    def select_then_analyze(ec, stiffness, values, vectors, operators,
                            n51_checkpoint=runner.DEFAULT_N51_CHECKPOINT,
                            profile_name=runner.DEFAULT_PROFILE_NAME):
        criteria = json.loads((HERE / "criteria.json").read_text())
        if selector.sha(SELECTOR_PATH) != criteria["selector"]["algorithm_sha256"]:
            raise ValueError("Frozen selector source changed")
        old_profile.install_profile(runner)
        old_meta = runner.validate_complete_checkpoint_directory(K340)
        if selector.sha(K340 / "checkpoint.json") != criteria["authenticated_inputs"]["epsj011_k340_checkpoint_sha256"]:
            raise ValueError("Authenticated epsilon_J=0.11 k340 checkpoint changed")
        with np.load(K340 / "stage1_eigensystem.npz", allow_pickle=False) as data:
            old_values = np.asarray(data["eigenvalues_GHz"])
            old_vectors = np.asarray(data["eigenvectors"])
        with np.load(K340 / "projected_operators.npz", allow_pickle=False) as data:
            old_ops = {name: np.asarray(data[name]) for name in data.files}
        physical = old_meta["configuration"]["physical_parameters"]
        old_ec, old_stiffness, _, _ = _coeffs(
            physical["EJ"], physical["EC"], physical["EL"], physical["ELK"],
            physical["EJS"], physical["ECS"], physical["eps_J"],
            physical["eps_LK"], physical["eC"], physical["eP"],
        )
        original = runner._full_stage2(old_ec, old_stiffness, old_values, old_ops, 8)
        old_analysis = json.loads((K340 / 'analysis.json').read_text())
        old_assignment = old_analysis['nested_k300_to_k340_tracking']['assignment_candidate_row_for_reference_state']
        (original['energies'], original['vectors'], original['tensor'],
         original['matrix'], original['transitions']) = runner._apply_state_assignment(
            original['energies'], original['vectors'], original['tensor'], original['matrix'], old_assignment)
        original_raw = runner.reconstruct_raw_final_states(old_vectors, original["tensor"])

        h, _ = selector.build_hamiltonian(values, operators, ec, stiffness)
        hp = h[:340 * 8, :340 * 8]
        base_e, base_v = eigh(hp, subset_by_index=(0, 10), driver="evr")
        base_tensor = base_v.reshape(340, 8, runner.NSTATES)
        base_raw = runner.reconstruct_raw_final_states(vectors[:, :340], base_tensor)
        assignment, matched, overlap = runner.track_cross_grid_states(original_raw, base_raw)
        confidence = runner.assignment_confidence_diagnostics(overlap, assignment)
        if not all(row["state_specific_claim_safe"] for row in confidence):
            raise ValueError("Unsafe original-k340 to nested-k340 tracking")
        base_e = base_e[assignment]
        base_v = base_v[:, assignment]
        phase = np.exp(1j * np.angle(overlap[assignment, np.arange(runner.NSTATES)]))
        base_v *= phase
        base_matrix = base_v.conj().T @ selector.build_hamiltonian(values[:340], {k: v[:340, :340] for k, v in operators.items()}, ec, stiffness)[1] @ base_v
        nested = {
            "energies": base_e,
            "transitions": base_e - base_e[0],
            "matrix": base_matrix,
        }
        original_nested = {
            "tracking": confidence,
            "minimum_raw_grid_overlap": float(min(matched)),
            "original": _metrics(original),
            "nested": _metrics(nested),
            "energy_differences_MHz": ((base_e - original["energies"]) * 1000).tolist(),
            "grid_phi_abs_max_difference": float(np.max(abs(abs(base_matrix) - abs(original["matrix"])))),
            "bitwise_identity_assumed": False,
        }
        provenance = {
            "criteria_sha256": selector.sha(HERE / "criteria.json"),
            "selector_sha256": selector.sha(SELECTOR_PATH),
            "selector_method_sha256": criteria["selector"]["method_sha256"],
            "reference_k340_checkpoint_sha256": selector.sha(K340 / "checkpoint.json"),
            "new_configuration_sha256": criteria["profile"]["configuration_sha256"],
            "original_to_nested_minimum_overlap": float(min(matched)),
            "selection_used_candidate_or_full_new_shell_dressed_results": False,
            "old_rank_lists_used": False,
        }
        selector.freeze(h, values, base_e, base_v, 340, SELECTION, provenance)
        record = {
            "selection_sha256": selector.sha(SELECTION),
            "selection_seal_sha256": selector.sha(str(SELECTION) + ".seal.json"),
            "frozen_before_candidate_or_full_new_shell_dressed_validation": True,
            "provenance": provenance,
        }
        return analyze_parent(
            runner, ec, stiffness, values, vectors, operators, profile_name,
            record, base_v, original_nested,
        )

    runner.analyze_same_solve = select_then_analyze
    return select_then_analyze
