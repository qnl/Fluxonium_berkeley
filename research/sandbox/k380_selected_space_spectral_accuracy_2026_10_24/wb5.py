"""Work Block 5 builders/evaluators. No eigensolver, launcher, or import-time work.

Research execution is deliberately unavailable until a new scoped guard integration
is implemented and reviewed. None of the Work Block 4 authorization is reusable.
"""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WB4 = ROOT / 'research/sandbox/k380_p340_s2_base_matched_validation_continuation_001'
DELIVERY = ROOT / 'research/sandbox/k380_s2_final_authorization_execution_2026_10_24'
INTEGRATION = ROOT / 'research/sandbox/k380_s2_scientific_launcher_integration'
PARENT = ROOT / 'research/checkpoints/four-mode-asymmetric-n71-epsj011-k380'
OLD_CRITERIA = ROOT / 'research/sandbox/epsj011_k380_preregistration/criteria.json'
V2_CRITERIA = ROOT / 'research/sandbox/epsj011_k380_schur_benchmark_v2_preregistration/criteria.json'
SCOPE = 'WB5:P340_Q360_380:selected_space_spectral_accuracy:v1'
METHODS = ('physics_frozen', 'bare_T300', 'denominator_Dplus', 'denominator_Vplus',
           'energy_prefix', 'spread', 'spread_reverse')
BUDGETS = (3, 6, 10, 15)
P = tuple(range(340))
Q = tuple(range(360, 380))
U = P + Q
TARGETS = tuple(range(11))
LEGACY_TARGETS = (0, 1, 2, 3, 7, 8)
ANCHORS = {
    'research/sandbox/k380_s2_independent_launch_review_2026_10_24/INDEPENDENT_REVIEW.md':
        '9bb7470f89737f6ddcfa5b0968ca7d99e002ce4d7859ca3ec2a50fc9d38be6e6',
    str((WB4/'OUTPUT_MANIFEST.json').relative_to(ROOT)):
        '418874afc97a8cfc017b452c9f7c5f65844944c1a55d8b74f0d97b0ee61cbe0f',
    str((DELIVERY/'DELIVERY_MANIFEST.json').relative_to(ROOT)):
        'f4c2d91c5eb3fa72f5f6ce03e0d3ad76fe87bf7895889cbd932ed828a3ab390a',
    str((INTEGRATION/'SOURCE_PINS.json').relative_to(ROOT)):
        'c1d45500037ba75c5f5660bceb41d4e70e08fb2cd18b93c00961782872bf80c9',
    str((INTEGRATION/'ARTIFACT_MANIFEST.json').relative_to(ROOT)):
        '4875d185d875118a76f40b01c9163ee62dc13ed7fdc49012a57faf86af424fe6',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_files(base, entries):
    for name, expected in entries.items():
        path = base/name
        require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), 'SOURCE_PATH')
        require(sha(path) == expected, 'HASH_MISMATCH: ' + name)
    return len(entries)


def authenticate_sources():
    """Streaming byte verification only. Never opens Stage-1 vector arrays."""
    counts = {'anchors': verify_files(ROOT, ANCHORS)}
    counts['source_pins'] = verify_files(ROOT, read(INTEGRATION/'SOURCE_PINS.json')['files'])
    counts['implementation_artifacts'] = verify_files(INTEGRATION, read(INTEGRATION/'ARTIFACT_MANIFEST.json'))
    counts['WB4_outputs'] = verify_files(WB4, read(WB4/'OUTPUT_MANIFEST.json'))
    counts['WB4_delivery'] = verify_files(DELIVERY, read(DELIVERY/'DELIVERY_MANIFEST.json'))
    result, receipt = read(WB4/'RUN_RESULT.json'), read(WB4/'EXECUTION_RECEIPT.json')
    require(result['status'] == receipt['result'] == 'COMPLETED', 'WB4_NOT_COMPLETE')
    require(result['partitions'] == {'P340': list(P), 'Q_S2': list(Q), 'U_S2': list(U)}, 'WB4_PARTITIONS')
    require(result['budgets'] == list(BUDGETS) and result['targets'] == 11, 'WB4_SCOPE')
    cp, complete = read(PARENT/'checkpoint.json'), read(PARENT/'COMPLETE')
    require(cp['status'] == complete['status'] == 'complete', 'PARENT_STATUS')
    require(complete['checkpoint_sha256'] == sha(PARENT/'checkpoint.json'), 'PARENT_LINK')
    require(digest(cp['configuration']) == cp['configuration_sha256'] ==
            '8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e', 'CONFIGURATION')
    counts['parent_payloads'] = verify_files(PARENT, cp['file_sha256'])
    return {'status': 'PASS', 'counts': counts, 'research_eigensolves': 0,
            'Stage1_vectors_accessed': False, 'configuration_sha256': cp['configuration_sha256']}


def external_functions():
    """Use the WB4 AST loader, which loads named bodies, never legacy launchers."""
    inserted = str(INTEGRATION) not in sys.path
    if inserted:
        sys.path.insert(0, str(INTEGRATION))
    try:
        spec = importlib.util.spec_from_file_location('wb5_inherited_functions', INTEGRATION/'historical_functions.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load()
    finally:
        if inserted:
            sys.path.remove(str(INTEGRATION))


def freeze_comparisons(rows, frozen):
    require(len(rows) == 28, 'COMPARISON_COUNT')
    require(frozen['base_ranks'] == list(P) and frozen['candidate_ranks'] == list(Q), 'FROZEN_SUPPORT')
    by_key, spaces = {}, {}
    for row in rows:
        method, budget = row['method'], row['budget']
        require(method in METHODS and type(budget) is int and budget in BUDGETS, 'METHOD_OR_BUDGET')
        key = (method, budget)
        require(key not in by_key, 'DUPLICATE_METHOD_BUDGET')
        order = row['selected_ranks']
        require(all(type(r) is int for r in order), 'RANK_TYPE')
        require(len(order) == budget and len(set(order)) == budget and set(order) <= set(Q), 'SELECTED_SUPPORT')
        require(order == frozen['eligible_control_orders'][method][:budget], 'FROZEN_ORDER_CHANGED')
        selected = tuple(sorted(order))
        by_key[key] = {'method': method, 'budget': budget, 'selection_order': order,
                       'selected_ranks': list(selected), 'additional_Stage1_states': budget,
                       'dimension': 8*(340+budget)}
        spaces.setdefault(selected, []).append(key)
    require(set(by_key) == {(m,b) for m in METHODS for b in BUDGETS}, 'MISSING_COMPARISON')
    jobs = []
    for index, selected in enumerate(sorted(spaces, key=lambda x: (len(x), x)), 1):
        job_id = f'cell_{index:02d}'
        aliases = [{'method': m, 'budget': b} for m,b in sorted(spaces[selected])]
        jobs.append({'job_id': job_id, 'budget': len(selected), 'selected_ranks': list(selected),
                     'retained_ranks': list(P+selected), 'dimension': 8*(340+len(selected)),
                     'eigenpairs': 11, 'driver': 'evr', 'subset_by_index': [0,10],
                     'aliases': aliases, 'maximum_attempts': 1})
        for key in spaces[selected]:
            by_key[key]['job_id'] = job_id
    return {'schema': 'wb5_frozen_comparisons_v1', 'scope': SCOPE,
            'study_classification': 'retrospective', 'research_execution_authorized': False,
            'P340': list(P), 'Q_S2': list(Q), 'U_S2': list(U), 'excluded_ranks': list(range(340,360)),
            'budgets': list(BUDGETS), 'methods': list(METHODS),
            'comparisons': [by_key[(m,b)] for b in BUDGETS for m in METHODS],
            'unique_jobs': jobs, 'maximum_new_large_eigensolves': len(jobs),
            'new_Stage1_eigensolves': 0, 'new_reference_eigensolves': 0}


def current_plan():
    frozen = ROOT/'research/sandbox/k380_p340_s2_base_matched_validation_2026_10_23/FROZEN_SELECTOR_RECORD.json'
    return freeze_comparisons(read(WB4/'SELECTOR_COMPARISON.json'), read(frozen))


def make_criteria():
    old, v2 = read(OLD_CRITERIA), read(V2_CRITERIA)
    constants_tree = ast.parse((OLD_CRITERIA.parent/'analysis_tools.py').read_text())
    frequencies = next(ast.literal_eval(node.value) for node in constants_tree.body
                       if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == 'FREQUENCIES' for target in node.targets))
    return {'schema': 'wb5_metrics_criteria_v1', 'scope': SCOPE,
            'targets': list(TARGETS), 'legacy_targets': list(LEGACY_TARGETS),
            'frequencies': frequencies, 'selectivity': 'f78-f01',
            'strict': old['strict'], 'practical': old['practical'],
            'tracking': old['tracking'], 'cancellation': old['cancellation'],
            'subspaces': old['subspaces'], 'validation': v2['validation'],
            'selectivity_strict_MHz': v2['new_extensions']['selectivity_strict_MHz'],
            'selectivity_practical_MHz': v2['new_extensions']['selectivity_practical_MHz'],
            'improvement_ratio_max': old['selector_comparison']['improvement_pass_ratio'],
            'no_harm_slack_strict_units': old['selector_comparison']['no_harm_slack_strict_units'],
            'zero_reference_floor': old['zero_reference']['floor'],
            'comparison_epsilon_MHz': old['cancellation']['trend_noise_MHz'],
            'criterion_status': 'PROPOSED_FROZEN_NOT_EXECUTION_AUTHORIZED',
            'extensions': ['All eleven states have raw, centered and excitation errors; original six retained separately',
                           'All eleven state assignments must pass existing WB4 confidence gates',
                           'Selection-specific relative assessment applies inherited 0.8 ratio and 0.1 strict-unit no-harm slack',
                           'Selectivity thresholds inherited explicitly from sealed v2 as a new-study proposal only',
                           'Cancellation flags prevent a favorable overall relative verdict; no aggregate can conceal omitted components']}


def gather(values, operators, ranks):
    """Gather before assembly. Generic for synthetic fixtures; caller validates scope."""
    import numpy as np
    require(all(type(r) is int for r in ranks) and len(ranks) > 0 and len(set(ranks)) == len(ranks), 'RANK_IDENTITIES')
    require(min(ranks) >= 0 and max(ranks) < len(values), 'PARENT_RANK_RANGE')
    require(np.asarray(values).dtype == np.float64, 'ENERGY_PRECISION')
    ix = np.asarray(ranks, dtype=int)
    selected = {}
    for name in ('n1','n2','n3','x2','x3'):
        a = operators[name]
        require(a.shape == (len(values),len(values)) and a.dtype == np.complex128, 'OPERATOR_PRECISION_SHAPE')
        selected[name] = a[np.ix_(ix,ix)]
    return np.asarray(values)[ix], selected


def build_job(values, operators, ec, stiffness, job, plan):
    """Sandbox-only selected-space builder. No solver or IO; future guarded worker only."""
    require(job in plan['unique_jobs'] and plan == current_plan(), 'UNREGISTERED_JOB')
    require(job['retained_ranks'] == list(P)+job['selected_ranks'], 'JOB_RANK_ORDER')
    require(not set(range(340,360)) & set(job['retained_ranks']), 'EXCLUDED_RANK')
    f = external_functions()
    start = time.monotonic()
    v, op = gather(values, operators, job['retained_ranks'])
    gather_seconds = time.monotonic()-start
    diagnostics = {}
    start = time.monotonic()
    h, phi = f['assemble'](v,op,ec,stiffness,diagnostics)
    require(h.shape == (job['dimension'],job['dimension']), 'JOB_DIMENSION')
    return h, phi, v, op, {'gather_seconds':gather_seconds,
                          'construction_seconds':time.monotonic()-start,'construction':diagnostics}


def embed(vectors, ranks, endpoint_ranks=U, n4=8):
    import numpy as np
    require(len(set(ranks)) == len(ranks) and set(ranks) <= set(endpoint_ranks), 'EMBED_SUPPORT')
    require(vectors.shape == (len(ranks)*n4,11), 'ELEVEN_VECTOR_SHAPE')
    positions = {r:i for i,r in enumerate(endpoint_ranks)}
    result = np.zeros((len(endpoint_ranks)*n4,11),complex)
    for local,rank in enumerate(ranks):
        result[n4*positions[rank]:n4*(positions[rank]+1)] = vectors[n4*local:n4*(local+1)]
    return result


def load_references():
    """Small saved dressed arrays only; no assembly, diagonalization or Stage-1 vectors."""
    import numpy as np
    authenticate_sources()
    out = {}
    for name,ranks in (('P340',P),('U_S2',U)):
        with np.load(WB4/(name+'.npz'),allow_pickle=False) as z:
            e,v,rr = z['energies_GHz'],z['vectors'],z['retained_ranks']
        require(e.shape==(11,) and v.shape==(8*len(ranks),11) and rr.tolist()==list(ranks), 'REFERENCE_SHAPE')
        require(np.isfinite(e).all() and np.isfinite(v).all(), 'REFERENCE_FINITE')
        require(np.linalg.norm(v.conj().T@v-np.eye(11),ord=np.inf)<=1e-8,'REFERENCE_ORTHOGONALITY')
        out[name]={'e':e,'v':v,'ranks':rr}
    return out


def accuracy_label(value, boundaries):
    value = abs(value)
    return 'PASS' if value <= boundaries[0] else 'BORDERLINE' if value <= boundaries[1] else 'FAIL'


def spectral_metrics(candidate, reference, criteria):
    """Candidate-minus-U errors in MHz; all eleven states, all mandatory carriers."""
    import numpy as np
    candidate,reference = np.asarray(candidate),np.asarray(reference)
    require(candidate.shape == reference.shape == (11,), 'ELEVEN_ENERGIES')
    require(np.isfinite(candidate).all() and np.isfinite(reference).all(), 'NONFINITE_ENERGIES')
    raw=(candidate-reference)*1000
    excitation=((candidate-candidate[0])-(reference-reference[0]))*1000
    centered=raw-np.mean(raw)
    carriers,cancellation={},{}
    for name,(i,j) in criteria['frequencies'].items():
        a,b=abs(candidate[j]-candidate[i])*1000,abs(reference[j]-reference[i])*1000
        error=float(a-b)
        xi,xj=float(excitation[i]),float(excitation[j])
        denominator=abs(xi)+abs(xj)
        rule=criteria['cancellation']
        ratio=abs(error)/denominator if denominator>rule['denominator_floor_MHz'] else None
        flagged=ratio is not None and ratio<=rule['transition_ratio_max'] and max(abs(xi),abs(xj))>rule['large_excitation_error_MHz']
        kind='f78_MHz' if name=='f78' else 'standard_MHz'
        carriers[name]={'candidate_MHz':float(a),'reference_MHz':float(b),'signed_error_MHz':error,
                        'absolute_error_MHz':abs(error),
                        'strict':accuracy_label(error,criteria['strict'][kind]),
                        'practical':accuracy_label(error,criteria['practical'][kind])}
        cancellation[name]={'excitation_error_i_MHz':xi,'excitation_error_j_MHz':xj,
                            'signed_transition_error_MHz':error,'denominator_MHz':denominator,
                            'ratio':ratio,'flagged':bool(flagged),
                            'severe':bool(flagged and max(abs(xi),abs(xj))>rule['severe_excitation_error_MHz'])}
    ca=carriers['f78']['candidate_MHz']-carriers['f01']['candidate_MHz']
    re=carriers['f78']['reference_MHz']-carriers['f01']['reference_MHz']
    sel=float(ca-re)
    return {'raw_state_error_MHz':raw.tolist(),'absolute_raw_state_error_MHz':abs(raw).tolist(),
            'centered_state_error_MHz':centered.tolist(),'absolute_centered_state_error_MHz':abs(centered).tolist(),
            'excitation_error_MHz':excitation.tolist(),'absolute_excitation_error_MHz':abs(excitation).tolist(),
            'max_absolute_excitation_error_MHz':float(max(abs(excitation))),
            'max_absolute_raw_state_error_MHz':float(max(abs(raw))),
            'legacy_six_max_absolute_excitation_error_MHz':float(max(abs(excitation[list(LEGACY_TARGETS)]))),
            'carriers':carriers,'cancellation':cancellation,
            'selectivity':{'candidate_MHz':ca,'reference_MHz':re,'signed_error_MHz':sel,
                           'absolute_error_MHz':abs(sel),
                           'strict':accuracy_label(sel,criteria['selectivity_strict_MHz']),
                           'practical':accuracy_label(sel,criteria['selectivity_practical_MHz'])}}


def comparison_axes(metrics, criteria):
    """No omission or cancellation across states; every error is absolute and scaled."""
    scale=criteria['strict']['standard_MHz'][0]
    axes={f'raw:{s}':metrics['absolute_raw_state_error_MHz'][s]/scale for s in TARGETS}
    axes.update({f'excitation:{s}':metrics['absolute_excitation_error_MHz'][s]/scale for s in TARGETS})
    axes.update({f'centered:{s}':metrics['absolute_centered_state_error_MHz'][s]/scale for s in TARGETS})
    axes.update({f'carrier:{k}':v['absolute_error_MHz']/criteria['strict']['f78_MHz' if k=='f78' else 'standard_MHz'][0]
                 for k,v in metrics['carriers'].items()})
    axes['selectivity:f78-f01']=metrics['selectivity']['absolute_error_MHz']/criteria['selectivity_strict_MHz'][0]
    return axes


def relative_verdict(physics,control,criteria,same_space=False):
    if same_space:
        require(physics==control,'IDENTICAL_SPACE_METRICS_DIFFER')
        return {'verdict':'IDENTICAL_SPACE','material_improvement':False,'no_harm':True}
    a,b=comparison_axes(physics,criteria),comparison_axes(control,criteria)
    require(set(a)==set(b),'COMPARISON_AXES')
    slack=criteria['no_harm_slack_strict_units']
    harms=[k for k in a if a[k]>b[k]+slack]
    # Separate envelopes prevent an improved splitting from hiding excitation damage.
    cenv=max(v for k,v in b.items() if k.startswith('carrier:'))
    penv=max(v for k,v in a.items() if k.startswith('carrier:'))
    checks=[]
    for pv,cv,noise in ((physics['max_absolute_excitation_error_MHz'],control['max_absolute_excitation_error_MHz'],criteria['comparison_epsilon_MHz']),
                        (penv,cenv,criteria['comparison_epsilon_MHz']/min(criteria['strict']['standard_MHz'][0],criteria['strict']['f78_MHz'][0]))):
        checks.append(cv>noise and pv<=criteria['improvement_ratio_max']*cv)
    cancellation=any(row['flagged'] for row in physics['cancellation'].values())
    improved=all(checks) and not harms and not cancellation
    return {'verdict':'IMPROVED' if improved else 'MIXED_OR_NOT_IMPROVED',
            'material_improvement':improved,'no_harm':not harms,'harm_components':harms,
            'cancellation_blocks_favorable_verdict':cancellation,
            'excitation_and_carrier_envelopes_both_improved':all(checks)}


def evaluate_cell(job, energies, vectors, h, phi, values, operators, ec, stiffness, references, criteria):
    """For a future supervised worker's supplied eigenpairs. This function never solves."""
    import numpy as np
    f=external_functions()
    limits=criteria['validation']
    lifted=embed(vectors,job['retained_ranks'])
    reference=references['U_S2']['v']
    mapping,phase,confidence=f['assignment'](lifted,reference,criteria['tracking'])
    f['safe_targets'](confidence,list(TARGETS))
    energies=np.asarray(energies)[mapping]
    vectors=np.asarray(vectors)[:,mapping]*phase
    lifted=lifted[:,mapping]*phase
    reverse,_,reverse_confidence=f['assignment'](reference,lifted,criteria['tracking'])
    require(reverse.tolist()==list(TARGETS),'REVERSE_TRACKING_IDENTITY')
    f['safe_targets'](reverse_confidence,list(TARGETS))
    base=embed(references['P340']['v'],P)
    base_map,_,base_conf=f['assignment'](lifted,base,criteria['tracking'])
    require(base_map.tolist()==list(TARGETS),'P340_LABEL_CONSISTENCY')
    f['safe_targets'](base_conf,list(TARGETS))
    th,tp=f['tensor_actions'](values,operators,ec,stiffness,vectors)
    quality=f['quality'](th,energies,vectors,limits)
    action=f['actions_match'](h,phi,vectors,th,tp,limits)
    overlap=lifted.conj().T@reference
    subspaces=f['subspaces'](overlap,criteria['subspaces'])
    small_phi=vectors.conj().T@phi@vectors
    gauge=np.exp(1j*np.arange(11)*0.37)
    phi_phase=(vectors*gauge).conj().T@phi@(vectors*gauge)
    require(np.max(abs(abs(small_phi)-abs(phi_phase)))<=limits['historical_phi_abs'],'PHASE_INVARIANCE')
    return {'spectral':spectral_metrics(energies,references['U_S2']['e'],criteria),
            'validation':{'quality':quality,'independent_action':action},
            'tracking':{'map':mapping.tolist(),'confidence':confidence,'reverse':reverse_confidence,
                        'P340_confidence':base_conf,'complex_subspaces':subspaces},
            'grid_phi_abs':abs(small_phi).tolist(),
            'reference':'U_S2 finite parent; no infinite-basis claim'}


def validate_result(result,job,criteria):
    """Strict complete cell schema, including costs and source identity."""
    def finite(value):
        if isinstance(value,float):require(math.isfinite(value),'RESULT_NONFINITE')
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(result)
    require(set(result)=={'schema','scope','status','job_id','selected_ranks','dimension','source_pins_sha256',
                         'eigenpair_archive_sha256','numerical','costs','resources'},'RESULT_KEYS')
    require(result['schema']=='wb5_cell_result_v1' and result['scope']==SCOPE and result['status']=='COMPLETED','RESULT_IDENTITY')
    require(result['job_id']==job['job_id'] and result['selected_ranks']==job['selected_ranks'] and result['dimension']==job['dimension'],'RESULT_JOB')
    require(all(isinstance(result[k],str) and len(result[k])==64 and all(x in '0123456789abcdef' for x in result[k])
                for k in ('source_pins_sha256','eigenpair_archive_sha256')),'RESULT_HASHES')
    data=result['numerical'];m=data['spectral']
    for name in ('raw_state_error_MHz','absolute_raw_state_error_MHz','centered_state_error_MHz',
                 'absolute_centered_state_error_MHz','excitation_error_MHz','absolute_excitation_error_MHz'):
        require(len(m[name])==11,'RESULT_ELEVEN_STATES')
    require(set(m['carriers'])==set(criteria['frequencies']) and set(m['cancellation'])==set(criteria['frequencies']),'RESULT_CARRIERS')
    for signed,absolute in (('raw_state_error_MHz','absolute_raw_state_error_MHz'),
                            ('centered_state_error_MHz','absolute_centered_state_error_MHz'),
                            ('excitation_error_MHz','absolute_excitation_error_MHz')):
        require(all(abs(a)==b for a,b in zip(m[signed],m[absolute])),'RESULT_ABSOLUTE_ERRORS')
    require(m['max_absolute_excitation_error_MHz']==max(m['absolute_excitation_error_MHz']) and
            m['max_absolute_raw_state_error_MHz']==max(m['absolute_raw_state_error_MHz']),'RESULT_MAXIMA')
    require({'selectivity','max_absolute_excitation_error_MHz','max_absolute_raw_state_error_MHz'}<=set(m),'RESULT_ENVELOPES')
    for carrier in list(m['carriers'].values())+[m['selectivity']]:
        require(set(carrier)=={'candidate_MHz','reference_MHz','signed_error_MHz','absolute_error_MHz','strict','practical'},'RESULT_CARRIER_FIELDS')
        require(carrier['absolute_error_MHz']==abs(carrier['signed_error_MHz']),'RESULT_CARRIER_ABSOLUTE_ERROR')
    require(all(set(row)=={'excitation_error_i_MHz','excitation_error_j_MHz','signed_transition_error_MHz',
                          'denominator_MHz','ratio','flagged','severe'} for row in m['cancellation'].values()),'RESULT_CANCELLATION_FIELDS')
    require(set(data['tracking']['complex_subspaces'])=={'_'.join(map(str,g)) for g in criteria['subspaces']},'RESULT_SUBSPACES')
    require(len(data['tracking']['confidence'])==11 and all(r['state_specific_claim_safe'] for r in data['tracking']['confidence']),'RESULT_TRACKING')
    require(len(data['validation']['quality']['residuals_GHz'])==11 and max(data['validation']['quality']['residuals_GHz'])<=criteria['validation']['residual_GHz'],'RESULT_RESIDUALS')
    require(data['validation']['quality']['orthogonality_inf']<=criteria['validation']['orthogonality_inf'],'RESULT_ORTHOGONALITY')
    require(len(data['grid_phi_abs'])==11 and all(len(row)==11 for row in data['grid_phi_abs']),'RESULT_PHI_SHAPE')
    require(all(len(data['tracking'][name])==11 and all(r['state_specific_claim_safe'] for r in data['tracking'][name])
                for name in ('reverse','P340_confidence')),'RESULT_REVERSE_BASE_TRACKING')
    require(set(data['validation']['independent_action'])=={'H_action_difference_GHz','Phi_action_difference','Phi_projection_max_difference'},'RESULT_INDEPENDENT_ACTION')
    for key,limit in (('H_action_difference_GHz','tensor_H_action_GHz'),('Phi_action_difference','tensor_phi_action')):
        vals=data['validation']['independent_action'][key]
        require(len(vals)==11 and max(vals)<=criteria['validation'][limit],'RESULT_ACTION_BOUND')
    require(data['validation']['independent_action']['Phi_projection_max_difference']<=criteria['validation']['historical_phi_abs'],'RESULT_PHI_PROJECTION_BOUND')
    required={'source_authentication_seconds','reusable_input_load_seconds','gather_seconds','construction_seconds',
              'diagonalization_seconds','tracking_validation_metrics_seconds','publication_seconds','total_cell_seconds',
              'candidate_screening_historical_seconds','base_target_preparation_historical_seconds',
              'shared_cost_accounting','speedup_claim'}
    require(set(result['costs'])==required,'RESULT_COSTS')
    for name,value in result['costs'].items():
        if name.endswith('_seconds') and 'historical' not in name:
            require(type(value) in (float,int) and value>=0,'RESULT_COST_VALUE')
    require(result['costs']['speedup_claim']=='NOT_ESTABLISHED','UNMEASURED_SPEEDUP')
    require(result['costs']['candidate_screening_historical_seconds'] is None and
            result['costs']['base_target_preparation_historical_seconds'] is None,'INVENTED_HISTORICAL_COST')
    require(set(result['resources'])=={'sampled_peak_group_RSS_bytes','cell_worker_peak_ru_maxrss_bytes',
                                    'process_peak_is_cumulative','rss_sampling_semantics','output_bytes'},'RESULT_RESOURCES')
    require(result['resources']['process_peak_is_cumulative'] is True,'CUMULATIVE_PEAK_SEMANTICS')
    require(result['resources']['sampled_peak_group_RSS_bytes']>=0 and result['resources']['output_bytes']>=0,'RESULT_RESOURCE_VALUE')
    # Missing mandatory metric names fail closed even if the outer shape is plausible.
    comparison_axes(m,criteria)
    return {'status':'PASS','schema':'wb5_cell_result_v1'}


def guard_compatibility():
    """Read-only contract inspection, never calls supervisor.run or launches a child."""
    contract=ast.parse((INTEGRATION/'contract.py').read_text())
    scope=next(ast.literal_eval(node.value) for node in contract.body if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='SCOPE' for t in node.targets))
    require(scope!=SCOPE,'UNEXPECTED_EXISTING_GUARD_SCOPE')
    return {'status':'BLOCKED_FOR_RESEARCH_EXECUTION','existing_scope':scope,'new_scope':SCOPE,
            'existing_maximum_large_solves':2,'planned_maximum_large_solves':19,
            'existing_authorization_reusable':False,'research_execution_enabled':False,
            'required_changes':['New exact-scope contract and nineteen-job attempt ledger',
                                'Worker/source authentication and hash-bound separate approval schemas',
                                'Bounded publication and complete result verification for nineteen cells',
                                'Synthetic watchdog/termination/cleanup tests of the new integrated call path',
                                'Independent review and explicit new user authorization before launch'],
            'proposed_unchanged_limits':{'rss_trigger_bytes':3*2**30,'intended_worker_budget_bytes':4*2**30,
                                         'wall_seconds_total':900,'output_bytes_total':2**30,
                                         'workers':1,'BLAS_threads':1,'automatic_retries':0},
            'memory_enforcement':'SUPERVISED_NOT_HARD'}


def execute_research(*args,**kwargs):
    raise RuntimeError('WB5_RESEARCH_EXECUTION_DISABLED: new scoped reviewed guard integration and authorization required')
