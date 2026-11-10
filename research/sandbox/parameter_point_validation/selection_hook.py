"""Freeze exact-selector ranks before any new-point candidate/full Stage 2."""
from pathlib import Path
import json, sys
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'research/sandbox/k260_preregistration'), str(ROOT)]
from Circuit_Objs.qchard_gridium_netlist import _coeffs
from frozen_selector import build_hamiltonian, freeze, sha, EXPECTED_SHA256
from k260_profile import install_profile as install_k260
from parameter_analysis import analyze_parameter_parent

REFERENCE = ROOT/'research/checkpoints/four-mode-asymmetric-n71-k260'
SELECTION = HERE/'selection.json'


def install_selection_hook(runner):
    def select_then_analyze(ec, stiffness, values, vectors, operators,
                            n51_checkpoint=runner.DEFAULT_N51_CHECKPOINT,
                            profile_name=runner.DEFAULT_PROFILE_NAME):
        criteria = json.loads((HERE/'criteria.json').read_text())
        install_k260(runner)
        meta = runner.validate_complete_checkpoint_directory(REFERENCE)
        if sha(REFERENCE/'checkpoint.json') != criteria['authenticated_inputs']['original_k260_checkpoint_sha256']:
            raise ValueError('Authenticated original k260 checkpoint changed')
        with np.load(REFERENCE/'stage1_eigensystem.npz', allow_pickle=False) as z:
            old_values=np.asarray(z['eigenvalues_GHz']); old_vectors=np.asarray(z['eigenvectors'])
        with np.load(REFERENCE/'projected_operators.npz', allow_pickle=False) as z:
            old_ops={name:np.asarray(z[name]) for name in z.files}
        p=meta['configuration']['physical_parameters']
        old_ec,old_k,_,_=_coeffs(p['EJ'],p['EC'],p['EL'],p['ELK'],p['EJS'],p['ECS'],
                                  p['eps_J'],p['eps_LK'],p['eC'],p['eP'])
        old=runner._full_stage2(old_ec,old_k,old_values,old_ops,8)
        old_raw=runner.reconstruct_raw_final_states(old_vectors,old['tensor'])

        # The only new-point dressed solve before selection is P=0..259.
        h,grid_phi=build_hamiltonian(values,operators,ec,stiffness)
        hp=h[:260*8,:260*8]
        base_e,base_v=eigh(hp,subset_by_index=(0,10),driver='evr')
        base_raw=runner.reconstruct_raw_final_states(vectors[:,:260],base_v.reshape(260,8,11))
        assignment,matched,overlap=runner.track_cross_grid_states(old_raw,base_raw)
        confidence=runner.assignment_confidence_diagnostics(overlap,assignment)
        if not all(row['state_specific_claim_safe'] for row in confidence):
            raise ValueError('Unsafe cross-parameter base-state tracking')
        base_e,base_v=base_e[assignment],base_v[:,assignment]
        phase=np.exp(1j*np.angle(overlap[assignment,np.arange(11)])); base_v*=phase
        provenance={
            'criteria_sha256':sha(HERE/'criteria.json'),
            'selector_sha256':EXPECTED_SHA256,
            'selector_method_sha256':criteria['frozen_selector']['method_sha256'],
            'reference_k260_checkpoint_sha256':sha(REFERENCE/'checkpoint.json'),
            'new_configuration_sha256':criteria['profile']['configuration_sha256'],
            'cross_parameter_tracking_minimum_overlap':float(min(matched)),
            'cross_parameter_tracking':confidence,
            'old_rank_lists_used':False,
            'selection_used_candidate_or_full_dressed_results':False}
        freeze(h,values,base_e,base_v,260,SELECTION,provenance)
        selection_sha=sha(SELECTION)
        freeze_record={'selection_sha256':selection_sha,
            'seal_sha256':sha(str(SELECTION)+'.seal.json'),
            'frozen_before_candidate_or_full_stage2':True,
            'base_tracking':confidence,'provenance':provenance}
        return analyze_parameter_parent(ec,stiffness,values,vectors,operators,
                                        profile_name,freeze_record,base_v)
    runner.analyze_same_solve=select_then_analyze
    return select_then_analyze
