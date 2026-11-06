"""Same-parent new-point analysis without an invalid old-parameter N51 mix."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner


def analyze_parameter_parent(ECm, K, values, vectors, operators,
                             profile_name, selection_provenance,
                             base_vectors):
    profile = runner.profile_definition(profile_name)
    configuration = profile['configuration']
    parent_k = configuration['solver_settings']['k']
    n4 = configuration['spatial_cutoffs']['N4']
    reference = runner._full_stage2(ECm, K, values, operators, n4)
    # Carry the pre-selection cross-parameter labels from P260 into the full
    # new-point endpoint. This occurs only after the selection is sealed.
    embedded_base = np.zeros((parent_k*n4, runner.NSTATES), complex)
    embedded_base[:base_vectors.shape[0]] = base_vectors
    overlap = reference['vectors'].conj().T @ embedded_base
    from scipy.optimize import linear_sum_assignment
    rows,cols=linear_sum_assignment(-abs(overlap))
    assignment=np.empty(runner.NSTATES,int); assignment[cols]=rows
    confidence=runner.assignment_confidence_diagnostics(overlap,assignment)
    if not all(row['state_specific_claim_safe'] for row in confidence):
        raise ValueError('Unsafe P260-to-k300 endpoint state tracking')
    (reference['energies'],reference['vectors'],reference['tensor'],
     reference['matrix'],reference['transitions'])=runner._apply_state_assignment(
        reference['energies'],reference['vectors'],reference['tensor'],
        reference['matrix'],assignment)
    reference['pathways']=runner._three_step_paths(reference['matrix'],runner.NSTATES)
    candidates = {}
    for label, dimension in profile['retained_cases']:
        ix = runner.retained_indices(dimension, profile_name)
        if dimension == parent_k:
            candidates[label] = runner._reference_endpoint_case(label, reference, parent_k)
        else:
            candidates[label] = runner._retained_case(
                label, ix, ECm, K, values, operators, reference, n4, parent_k)
    return {
        'profile': profile_name,
        'configuration_sha256': runner.CONFIGURATION_SHA256_BY_PROFILE[profile_name],
        'control_operator': 'grid_phi = theta3 + theta2/2',
        'parameter_point_scope': 'eps_J=0.11 within the same asymmetric four-mode model',
        'reference_endpoint_semantics': 'k300 is a finite same-parent endpoint, not ground truth',
        'stage1_parent_reuse': 'all nested Stage-2 cases use the single new-point N71 k300 Stage-1 parent',
        'n71_reference_k300': {
            'energies_GHz': reference['energies'].tolist(),
            'transitions_GHz': reference['transitions'][1:].tolist(),
            'grid_phi_abs': np.abs(reference['matrix']).tolist(),
            'pathways': reference['pathways'],
            'thomas_observables': runner._thomas_observables(
                reference['transitions'], reference['matrix'],
                np.ones(runner.NSTATES), reference, parent_k)},
        'retained_basis_cases': candidates,
        'held_out_selection_freeze': selection_provenance,
        'parameter_point_endpoint_tracking': {
            'reference': 'cross-parameter-labeled new-point P260',
            'assignment_candidate_row_for_reference_state': assignment.tolist(),
            'confidence': confidence,
            'minimum_overlap': float(min(abs(overlap[assignment,np.arange(runner.NSTATES)])))},
        'n51_to_n71_spatial_comparison': None,
        'spatial_convergence_classification': None,
        'classification_status': 'deferred to sealed parameter-point validation analysis',
        'scientific_restriction': 'original-parameter N51 data were not combined with new-point couplings'}
