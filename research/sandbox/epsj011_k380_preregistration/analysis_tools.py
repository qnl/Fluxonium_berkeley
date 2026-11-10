"""Value-preserving copy of validated comparison helpers; source hash pinned in history.
Only imports and explicit diagnostic reporting differ from the predecessor.
"""
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
from selection_hook import selector
build_hamiltonian, indices, sha = selector.build_hamiltonian, selector.indices, selector.sha
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
      'cancellation':cancellation,
      'principal_subspaces':{'_'.join(map(str,g)):runner.subspace_overlap_diagnostics(case['embedded'].conj().T@endpoint['embedded'],g,g) for g in criteria['subspaces']},
      'tracking_diagnostics':case['confidence']}

