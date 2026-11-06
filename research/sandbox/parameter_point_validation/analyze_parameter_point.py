"""Sealed post-authentication selected-space analysis for eps_J=0.11."""
from pathlib import Path
import argparse, json, sys
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from Simulations.Rabi_3Photon.four_mode_retained_basis_audit import _three_step_paths
from frozen_selector import build_hamiltonian, indices, sha
from profile import install_profile, EXPECTED_CONFIGURATION_SHA256

STATES=(0,1,2,3,7,8)
FREQUENCIES={'f01':(0,1),'f07':(0,7),'f08':(0,8),'f78':(7,8),'f81':(1,8),
             'f72':(2,7),'f21':(1,2),'f03':(0,3),'f38':(3,8)}
EDGES={'g07':(0,7),'g78':(7,8),'g81':(1,8),'g72':(2,7),'g21':(1,2),
       'g03':(0,3),'g38':(3,8),'g01':(0,1)}
PATHS={'0781':(0,7,8,1),'0721':(0,7,2,1),'0381':(0,3,8,1)}


def selected_solve(h,ranks,reference=None):
    ix=indices(ranks); sub=h[np.ix_(ix,ix)]
    e,v=eigh(sub,subset_by_index=(0,10),driver='evr')
    embedded=np.zeros((len(h),11),complex); embedded[ix]=v; confidence=[]
    if reference is not None:
        overlap=embedded.conj().T@reference
        rows,cols=linear_sum_assignment(-abs(overlap))
        assignment=np.empty(11,int); assignment[cols]=rows
        confidence=runner.assignment_confidence_diagnostics(overlap,assignment)
        e,v,embedded=e[assignment],v[:,assignment],embedded[:,assignment]
        phase=np.exp(1j*np.angle(np.diag(embedded.conj().T@reference)))
        v,embedded=v*phase,embedded*phase
    residual=float(np.max(np.linalg.norm(sub@v-v*e,axis=0)))
    return {'e':e,'v':v,'embedded':embedded,'confidence':confidence,'residual':residual}


def reorder_endpoint(case,assignment,confidence):
    assignment=np.asarray(assignment,dtype=int)
    case['e']=case['e'][assignment]; case['v']=case['v'][:,assignment]
    case['embedded']=case['embedded'][:,assignment]; case['confidence']=confidence
    return case


def block_metrics(m,blocks):
    out={}
    for name,(rows,cols) in blocks.items():
        a=m[np.ix_(rows,cols)]
        out[name]={'frobenius':float(np.linalg.norm(a)),
                   'singular_values':np.linalg.svd(a,compute_uv=False).tolist()}
    return out


def metrics(case,grid_phi,criteria):
    e=case['e']; m=case['embedded'].conj().T@grid_phi@case['embedded']
    ranked=_three_step_paths(m,11,count=None)
    path_rows={tuple(row['path']):dict(row,rank=i+1) for i,row in enumerate(ranked)}
    return {'energies_GHz':e.tolist(),
      'carriers_MHz':{k:float(abs(e[j]-e[i])*1000) for k,(i,j) in FREQUENCIES.items()},
      'edges':{k:float(abs(m[i,j])) for k,(i,j) in EDGES.items()},
      'scores':{k:float(path_rows[p]['score_product']) for k,p in PATHS.items()},
      'path_ranks':{k:int(path_rows[p]['rank']) for k,p in PATHS.items()},
      'blocks':block_metrics(m,criteria['blocks']),'matrix_abs':abs(m).tolist()}


def max_status(value,limits):
    x=abs(float(value)); return 'PASS' if x<=limits[0] else 'BORDERLINE' if x<=limits[1] else 'FAIL'


def min_status(value,limits):
    x=float(value); return 'PASS' if x>=limits[0] else 'BORDERLINE' if x>=limits[1] else 'FAIL'


def relative(candidate,reference,floor):
    if abs(reference)<=floor:
        return 0.0 if abs(candidate-reference)<=floor else float('inf')
    return abs(candidate-reference)/abs(reference)


def compare(case,endpoint,cm,em,criteria):
    components,units={},{}
    for name in FREQUENCIES:
        d=cm['carriers_MHz'][name]-em['carriers_MHz'][name]
        key=f'carrier:{name}'; kind='f78_MHz' if name=='f78' else 'standard_MHz'
        components[key]={'value':d,'strict':max_status(d,criteria['strict'][kind]),
                         'practical':max_status(d,criteria['practical'][kind])}
        units[key]=abs(d)/criteria['strict'][kind][0]
    overlap=abs(case['embedded'].conj().T@endpoint['embedded'])
    for s in STATES:
        v=float(overlap[s,s]); key=f'overlap:{s}'
        components[key]={'value':v,'strict':min_status(v,criteria['strict']['overlap_min']),
                         'practical':min_status(v,criteria['practical']['overlap_min'])}
        units[key]=max(0,1-v)/(1-criteria['strict']['overlap_min'][0])
    for group in criteria['subspaces']:
        label='_'.join(map(str,group)); d=runner.subspace_overlap_diagnostics(overlap,group,group)
        v=float(d['minimum_singular_value']); key=f'subspace:{label}'
        components[key]={'value':v,'strict':min_status(v,criteria['strict']['overlap_min']),
                         'practical':min_status(v,criteria['practical']['overlap_min'])}
        units[key]=max(0,1-v)/(1-criteria['strict']['overlap_min'][0])

    # Gauge-independent recovery versus the finite shell endpoint.
    raw={s:float((case['e'][s]-endpoint['e'][s])*1000) for s in STATES}
    common=float(np.mean(list(raw.values())))
    centered={s:raw[s]-common for s in STATES}
    excitation={s:float(((case['e'][s]-case['e'][0])-(endpoint['e'][s]-endpoint['e'][0]))*1000)
                for s in STATES}
    for s in STATES:
        key=f'centered:{s}'; v=centered[s]
        components[key]={'value':v,'strict':max_status(v,criteria['strict']['standard_MHz']),
                         'practical':max_status(v,criteria['practical']['standard_MHz'])}
        units[key]=abs(v)/criteria['strict']['standard_MHz'][0]
        if s:
            key=f'excitation:{s}'; v=excitation[s]
            components[key]={'value':v,'strict':max_status(v,criteria['strict']['standard_MHz']),
                             'practical':max_status(v,criteria['practical']['standard_MHz'])}
            units[key]=abs(v)/criteria['strict']['standard_MHz'][0]

    floor=criteria['zero_reference']['floor']
    for name in EDGES:
        d=relative(cm['edges'][name],em['edges'][name],floor); key=f'edge:{name}'
        components[key]={'value':d,'strict':max_status(d,criteria['strict']['operator_relative']),
                         'practical':max_status(d,criteria['practical']['operator_relative'])}
        units[key]=d/criteria['strict']['operator_relative'][0]
    for name in PATHS:
        d=relative(cm['scores'][name],em['scores'][name],floor); key=f'score:{name}'
        components[key]={'value':d,'strict':max_status(d,criteria['strict']['score_relative']),
                         'practical':max_status(d,criteria['practical']['score_relative'])}
        units[key]=d/criteria['strict']['score_relative'][0]
    for name in criteria['blocks']:
        pairs=[('frobenius',cm['blocks'][name]['frobenius'],em['blocks'][name]['frobenius'])]
        pairs += [(f'sv{i}',a,b) for i,(a,b) in enumerate(zip(cm['blocks'][name]['singular_values'],em['blocks'][name]['singular_values']))]
        for suffix,a,b in pairs:
            d=relative(a,b,floor); key=f'block:{name}:{suffix}'
            components[key]={'value':d,'strict':max_status(d,criteria['strict']['operator_relative']),
                             'practical':max_status(d,criteria['practical']['operator_relative'])}
            units[key]=d/criteria['strict']['operator_relative'][0]

    cancellation={}
    c=criteria['cancellation']
    for name,(i,j) in FREQUENCIES.items():
        xi=0.0 if i==0 else excitation[i]; xj=excitation[j]
        denominator=abs(xi)+abs(xj)
        signed=float(np.sign(endpoint['e'][j]-endpoint['e'][i])*(
            (case['e'][j]-case['e'][i])-(endpoint['e'][j]-endpoint['e'][i]))*1000)
        ratio=abs(signed)/denominator if denominator>c['denominator_floor_MHz'] else None
        flagged=bool(ratio is not None and ratio<=c['transition_ratio_max'] and
                     max(abs(xi),abs(xj))>c['large_excitation_error_MHz'])
        severe=bool(flagged and max(abs(xi),abs(xj))>c['severe_excitation_error_MHz'])
        cancellation[name]={'excitation_error_i_MHz':xi,'excitation_error_j_MHz':xj,
            'transition_error_MHz':signed,'denominator_MHz':denominator,'ratio':ratio,
            'flagged':flagged,'severe':severe}
    safe=all(row['state_specific_claim_safe'] for row in case['confidence'] if row['reference_state'] in STATES)
    candidate_strong=max(PATHS,key=lambda x:cm['scores'][x]); endpoint_strong=max(PATHS,key=lambda x:em['scores'][x])
    if cm['path_ranks']['0781']!=1 or candidate_strong!=endpoint_strong: ordering='FAIL'
    elif cm['path_ranks']==em['path_ranks']: ordering='PASS'
    else: ordering='BORDERLINE'
    return {'components':components,'strict_units':units,'tracking_safe':safe,
      'path_ordering':ordering,'residual_GHz':case['residual'],'metrics':cm,
      'energy_recovery':{'raw_state_errors_MHz':raw,'common_offset_MHz':common,
                         'centered_state_errors_MHz':centered,'excitation_errors_MHz':excitation},
      'cancellation':cancellation}


def load_parent(checkpoint,criteria):
    install_profile(runner)
    metadata=runner.validate_complete_checkpoint_directory(checkpoint)
    if metadata['configuration_sha256']!=EXPECTED_CONFIGURATION_SHA256: raise ValueError('configuration mismatch')
    with np.load(checkpoint/'stage1_eigensystem.npz',allow_pickle=False) as z:
        values=np.asarray(z['eigenvalues_GHz']); shape=list(np.asarray(z['eigenvectors']).shape)
    with np.load(checkpoint/'projected_operators.npz',allow_pickle=False) as z:
        operators={k:np.asarray(z[k]) for k in z.files}
    if values.shape!=(300,) or shape!=[45369,300] or any(x.shape!=(300,300) for x in operators.values()):
        raise ValueError('payload dimensions')
    p=metadata['configuration']['physical_parameters']
    ec,k,_,_=_coeffs(p['EJ'],p['EC'],p['EL'],p['ELK'],p['EJS'],p['ECS'],p['eps_J'],p['eps_LK'],p['eC'],p['eP'])
    h,o=build_hamiltonian(values,operators,ec,k)
    return metadata,h,o,json.loads((checkpoint/'analysis.json').read_text())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('checkpoint',type=Path); ap.add_argument('selection',type=Path)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    criteria=json.loads((HERE/'criteria.json').read_text())
    seal=json.loads(Path(str(args.selection)+'.seal.json').read_text())
    if seal['selection_sha256']!=sha(args.selection): raise ValueError('selection seal')
    selection=json.loads(args.selection.read_text()); metadata,h,o,analysis=load_parent(args.checkpoint,criteria)
    link=analysis.get('held_out_selection_freeze',{})
    if link.get('selection_sha256')!=seal['selection_sha256'] or not link.get('frozen_before_candidate_or_full_stage2'):
        raise ValueError('checkpoint selection linkage')
    tracking=analysis['parameter_point_endpoint_tracking']
    union=reorder_endpoint(selected_solve(h,range(300)),
      tracking['assignment_candidate_row_for_reference_state'],tracking['confidence'])
    um=metrics(union,o,criteria)
    endpoints={'S1':selected_solve(h,range(280),union['embedded']),
               'S2':selected_solve(h,list(range(260))+list(range(280,300)),union['embedded']),
               'union':union}
    endpoint_metrics={k:metrics(v,o,criteria) for k,v in endpoints.items()}
    comparisons={c:{} for c in endpoints}
    for context in comparisons:
        for budget in (3,6,10,15):
            methods=['physics','prefix','spread','spread_reverse']
            spaces={m:selection['spaces'][f'{context}_{m}_{budget}'] for m in methods}
            if context=='union':
                spaces['balanced_prefix']=spaces['prefix']; spaces['prefix']=selection['spaces'][f'union_global_prefix_{budget}']
            comparisons[context][str(budget)]={}
            for name,ranks in spaces.items():
                case=selected_solve(h,ranks,endpoints[context]['embedded']); cm=metrics(case,o,criteria)
                comparisons[context][str(budget)][name]=compare(case,endpoints[context],cm,endpoint_metrics[context],criteria)
    prefix_cases={k:selected_solve(h,range(k),union['embedded']) for k in (260,280)}
    prefix_metrics={k:metrics(v,o,criteria) for k,v in prefix_cases.items()}
    cmp260=compare(prefix_cases[260],union,prefix_metrics[260],um,criteria)
    cmp280=compare(prefix_cases[280],union,prefix_metrics[280],um,criteria)
    prefix_components={key:{'k260_vs_k300':{'strict':row['strict'],'practical':row['practical']},
      'k280_vs_k300':{'strict':cmp280['components'][key]['strict'],'practical':cmp280['components'][key]['practical']}}
      for key,row in cmp260['components'].items()}
    for name in FREQUENCIES:
        a=prefix_metrics[280]['carriers_MHz'][name]-prefix_metrics[260]['carriers_MHz'][name]
        b=um['carriers_MHz'][name]-prefix_metrics[280]['carriers_MHz'][name]
        row=prefix_components[f'carrier:{name}']; noise=criteria['cancellation']['denominator_floor_MHz']
        row.update({'step_260_280':a,'step_280_300':b,'sign_reversal':bool(a*b<0 and min(abs(a),abs(b))>noise),
          'accidental_cancellation':bool(abs(a+b)<=.25*(abs(a)+abs(b)) and abs(a)+abs(b)>noise)})
    hard={'checkpoint_authentication':True,'selection_seal':True,'checkpoint_selection_link':True,
      'parent_quality':metadata['validation']['maximum_eigenpair_residual']<=criteria['quality']['stage1_residual_max'] and metadata['validation']['orthogonality_residual_inf']<=criteria['quality']['stage1_orthogonality_inf'],
      'stage2_residuals':all(x['residual']<=criteria['quality']['stage2_residual_GHz_max'] for x in endpoints.values()),
      'tracking':all(row['state_specific_claim_safe'] for c in endpoints.values() for row in c['confidence'] if row['reference_state'] in STATES),
      'cross_parameter_base_tracking':all(row['state_specific_claim_safe'] for row in link['base_tracking'])}
    payload={'schema':2,'configuration_sha256':metadata['configuration_sha256'],'hard_gates':hard,
      'comparisons':comparisons,'prefix_convergence':{'components':prefix_components},
      'endpoint_metrics':endpoint_metrics,'selection_sha256':seal['selection_sha256']}
    with args.output.open('x') as f: json.dump(payload,f,indent=2,allow_nan=False); f.write('\n')


if __name__=='__main__': main()
