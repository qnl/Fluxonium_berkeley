"""Same-parent k340 analysis, with no N51 cross-parameter mixing."""
import numpy as np


def analyze_parent(runner, ec, stiffness, values, vectors, operators,
                   profile_name, selection_record, base_vectors,
                   original_nested_record):
    profile = runner.profile_definition(profile_name)
    parent_k = profile["configuration"]["solver_settings"]["k"]
    n4 = profile["configuration"]["spatial_cutoffs"]["N4"]
    reference = runner._full_stage2(ec, stiffness, values, operators, n4)
    embedded_base = np.zeros((parent_k * n4, runner.NSTATES), complex)
    embedded_base[:base_vectors.shape[0]] = base_vectors
    overlap = reference["vectors"].conj().T @ embedded_base
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-abs(overlap))
    assignment = np.empty(runner.NSTATES, int)
    assignment[cols] = rows
    confidence = runner.assignment_confidence_diagnostics(overlap, assignment)
    if not all(row["state_specific_claim_safe"] for row in confidence):
        raise ValueError("Unsafe nested-k300 to k340 endpoint tracking")
    (
        reference["energies"], reference["vectors"], reference["tensor"],
        reference["matrix"], reference["transitions"],
    ) = runner._apply_state_assignment(
        reference["energies"], reference["vectors"], reference["tensor"],
        reference["matrix"], assignment,
    )
    reference["pathways"] = runner._three_step_paths(
        reference["matrix"], runner.NSTATES
    )
    cases = {}
    for label, dimension in profile["retained_cases"]:
        indices = runner.retained_indices(dimension, profile_name)
        if dimension == parent_k:
            cases[label] = runner._reference_endpoint_case(label, reference, parent_k)
        else:
            cases[label] = runner._retained_case(
                label, indices, ec, stiffness, values, operators,
                reference, n4, parent_k,
            )
    return {
        "profile": profile_name,
        "configuration_sha256": runner.CONFIGURATION_SHA256_BY_PROFILE[profile_name],
        "control_operator": "grid_phi = theta3 + theta2/2",
        "scope": "fixed N71, fixed four-mode Hamiltonian, epsilon_J=0.11",
        "reference_endpoint_semantics": "k340 is a finite endpoint, not ground truth",
        "stage1_parent_reuse": "all prefixes 260,280,300,320,340 use one k340 Stage-1 parent",
        "retained_basis_cases": cases,
        "selection_freeze": selection_record,
        "original_k300_vs_nested_k300": original_nested_record,
        "nested_k300_to_k340_tracking": {
            "assignment_candidate_row_for_reference_state": assignment.tolist(),
            "confidence": confidence,
            "minimum_overlap": float(min(abs(overlap[assignment, np.arange(runner.NSTATES)]))),
        },
        "spatial_convergence_classification": None,
        "scientific_restriction": "no spatial, N4, n1max, omitted-mode, or beyond-k340 convergence claim",
    }
