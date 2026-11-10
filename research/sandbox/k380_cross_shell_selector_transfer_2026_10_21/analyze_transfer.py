"""Small, label-separated evaluation of sealed historical selector orders.

No scientific solver imports, array loading, Hamiltonian or response construction.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PRE=ROOT/'research/sandbox/epsj011_k380_preregistration'
HIST=ROOT/'research/sandbox/k300_preregistration'
CP=ROOT/'research/checkpoints/four-mode-asymmetric-n71-epsj011-k380'
FP=ROOT/'research/sandbox/k380_high_rank_state_fingerprints_2026_10_21'
WP2=ROOT/'research/sandbox/k380_coupling_vs_schur_2026_10_21'
REV=ROOT/'research/sandbox/epsj011_k380_diagnostic_review'
DEPEND=ROOT/'research/sandbox/epsj011_k380_selector_s2_dependency_review/SELECTOR_S2_DEPENDENCY_REVIEW.md'
BUDGETS=(3,6,10,15)
METHODS=('physics_frozen','bare_T300','bare_Tactual','denominator_Vplus','denominator_Dplus',
         'energy_prefix','spread','spread_reverse')
BARE={'bare_T300':'T300','bare_Tactual':'Tactual','denominator_Vplus':'Vplus','denominator_Dplus':'Dplus'}
SPREAD=(0,19,9,14,4,2,6,11,16,1,3,5,7,8,10,12,13,15,17,18)
SELECTION_SHA='6720cb7b2213962bf490ec6096ebd88e0b66e79ec7a2c25dee33116fc02eb8b7'
ALGORITHM_SHA='57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2'
METHOD_SHA='2055c7d54d81ada97379f4941ef08354dbb3cbf9378aa2b10dde1349ca2ff7cc'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def csvwrite(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in row.items()})


def status():return subprocess.check_output(['git','status','--short'],cwd=ROOT).decode()


def required_paths():
    paths=[PRE/p for p in ('criteria.json','preregistration_manifest.json','preregistration_seal.json',
          'selection.json','selection.json.seal.json','selection_hook.py','parent_analysis.py','analyze_k380.py',
          'analysis_plan.md','k380_metrics.json','execution_receipt.json')]
    paths += [HIST/p for p in ('selector.py','selector_method.md','preregistration_manifest.json','preregistration_seal.json')]
    paths += [CP/p for p in ('checkpoint.json','COMPLETE','analysis.json')]
    paths += [FP/p for p in ('RESEARCH_PLAN.md','REPORT.md','NUMERICAL_VALIDATION.md','SOURCE_PROVENANCE.md','OUTPUT_HASHES.json','SOURCE_AUTHENTICATION.json')]
    paths += [WP2/p for p in ('SHELL_COMPARISON.csv','RANKING_ANALYSIS.csv','PREDICTOR_DEFINITIONS.md','RESEARCH_PLAN.md',
                             'REPORT.md','SOURCE_PROVENANCE.md','SOURCE_AUTHENTICATION.json','FINAL_VALIDATION.json','OUTPUT_HASHES.json')]
    paths += [REV/p for p in ('REPORT.md','SOURCE_PROVENANCE.md','DIAGNOSTICS.json','OUTPUT_HASHES.json')]
    paths += [DEPEND]
    paths += [ROOT/p for p in ('EmilianoUPDATE.md','EmilianoUPDATE-unsaved-backup.md','Simulations/Cphase/main.ipynb',
                              'Simulations/Cphase/workflow_funcs.py','.git/index')]
    for dirname in ('epsj011_k380_schur_selector_benchmark','epsj011_k380_schur_benchmark_v2_preregistration'):
        paths += [ROOT/'research/sandbox'/dirname/p for p in ('preregistration_manifest.json','preregistration_seal.json')]
    return sorted(set(p for p in paths if p.is_file()))


def authenticate():
    assert sha(PRE/'selection.json')==SELECTION_SHA
    assert sha(HIST/'selector.py')==ALGORITHM_SHA
    assert sha(HIST/'selector_method.md')==METHOD_SHA
    manifest,seal,criteria=read(PRE/'preregistration_manifest.json'),read(PRE/'preregistration_seal.json'),read(PRE/'criteria.json')
    assert sha(PRE/'preregistration_manifest.json')=='95f33014769195605ef871f62fee229d0417f610fb6e96dd44df71e710707656'
    assert seal['manifest_sha256']==sha(PRE/'preregistration_manifest.json') and seal['criteria_sha256']==sha(PRE/'criteria.json')
    assert criteria['selector']['algorithm_sha256']==ALGORITHM_SHA and criteria['selector']['method_sha256']==METHOD_SHA
    assert criteria['selector']['budgets_per_shell']==list(BUDGETS) and criteria['selector']['base']==340
    assert criteria['selector']['operator_weight']==0 and not criteria['selector']['budget_tuning']
    checked={}
    for p in (HIST/'selector.py',HIST/'selector_method.md',PRE/'selection_hook.py',PRE/'parent_analysis.py',PRE/'analyze_k380.py',PRE/'analysis_plan.md',PRE/'criteria.json'):
        rel=str(p.relative_to(ROOT));assert sha(p)==manifest['files'][rel];checked[rel]=sha(p)
    oldmanifest,oldseal=read(HIST/'preregistration_manifest.json'),read(HIST/'preregistration_seal.json')
    assert oldseal['frozen_date_pacific']=='2026-10-08' and oldseal['manifest_sha256']==sha(HIST/'preregistration_manifest.json')
    for p in (HIST/'selector.py',HIST/'selector_method.md'):
        assert oldmanifest['files'][str(p.relative_to(ROOT))]==sha(p)
    meta,complete=read(CP/'checkpoint.json'),read(CP/'COMPLETE')
    assert sha(CP/'checkpoint.json')=='399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8'
    assert meta['status']==complete['status']=='complete' and complete['checkpoint_sha256']==sha(CP/'checkpoint.json')
    assert meta['configuration']==criteria['profile']['configuration']
    fp=hashlib.sha256(json.dumps(meta['configuration'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert fp==meta['configuration_sha256']==complete['configuration_sha256']=='8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e'
    assert sha(CP/'analysis.json')==meta['file_sha256']['analysis.json']
    assert sha(PRE/'k380_metrics.json')=='aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e'
    assert sha(PRE/'execution_receipt.json')=='c7ff154d805ded48367a28ac5137901ad9567d755372ed69bae64bc3c5c36d0b'
    receipt=read(PRE/'execution_receipt.json')
    assert receipt['status']=='SUCCESS' and receipt['configuration_sha256']==fp
    selection_seal=read(PRE/'selection.json.seal.json');analysis=read(CP/'analysis.json')
    assert selection_seal['selection_sha256']==SELECTION_SHA
    freeze=analysis['selection_freeze']
    assert freeze['selection_sha256']==SELECTION_SHA and freeze['selection_seal_sha256']==sha(PRE/'selection.json.seal.json')
    assert freeze['frozen_before_candidate_or_full_new_shell_dressed_validation']
    assert not freeze['provenance']['selection_used_candidate_or_full_new_shell_dressed_results']
    assert not freeze['provenance']['old_rank_lists_used']
    assert freeze['provenance']==selection_seal['provenance']
    # Inspect source call order statically; never import its solve-bearing module.
    hook=(PRE/'selection_hook.py').read_text()
    assert hook.index('selector.freeze(h,')<hook.index('return analyze_parent(')
    parent=(PRE/'parent_analysis.py').read_text()
    assert 'reference = runner._full_stage2' in parent
    tree=ast.parse((HIST/'selector.py').read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('GROUPS','BUDGETS')}
    assert constants==dict(GROUPS=((0,1),(2,3),(7,8)),BUDGETS=BUDGETS)
    consumed=[]
    for directory,names in (
        (FP,('RESEARCH_PLAN.md','REPORT.md','NUMERICAL_VALIDATION.md','SOURCE_PROVENANCE.md','SOURCE_AUTHENTICATION.json')),
        (WP2,('SHELL_COMPARISON.csv','RANKING_ANALYSIS.csv','PREDICTOR_DEFINITIONS.md','RESEARCH_PLAN.md','REPORT.md','SOURCE_PROVENANCE.md','SOURCE_AUTHENTICATION.json','FINAL_VALIDATION.json')),
        (REV,('REPORT.md','SOURCE_PROVENANCE.md','DIAGNOSTICS.json'))):
        pins=read(directory/'OUTPUT_HASHES.json')
        for name in names:assert sha(directory/name)==pins[name];consumed.append(str((directory/name).relative_to(ROOT)))
    previous=read(WP2/'SOURCE_AUTHENTICATION.json')
    assert previous['status']=='PASS' and read(WP2/'FINAL_VALIDATION.json')['preservation']=='PASS'
    dep_rel=str(DEPEND.relative_to(ROOT));assert sha(DEPEND)==previous['consumed_and_protected_hashes'][dep_rel]
    # Old payload authentication is reused; no checkpoint array is reread/rehashed.
    return dict(status='PASS',configuration_sha256=fp,algorithm_sha256=ALGORITHM_SHA,method_sha256=METHOD_SHA,
                selection_sha256=SELECTION_SHA,original_algorithm_frozen_pacific='2026-10-08',k380_preregistration_sealed=seal['sealed_date'],
                chronology='Pinned call order and linked records support selection before candidate validation; no independent timestamp.',
                required_source_pins=checked,prior_outputs_verified=consumed,array_payload_authentication='REUSED; no arrays loaded or rehashed',
                broad_prior_audits_rerun=False)


def bare_input(rows):
    """Whitelist predictor columns; never expose label/target columns downstream."""
    return [dict(shell=row['shell'],rank=int(row['rank']),**{key:float(row[key]) for key in BARE.values()}) for row in rows]


def make_orders(selection,bare):
    orders={}
    for shell,start in (('S1',340),('S2',360)):
        candidates=list(range(start,start+20));saved=selection['shells'][shell]
        assert sorted(saved['order'])==candidates
        assert [r['rank'] for r in saved['greedy_trace']]==saved['order']
        records=[r for r in bare if r['shell']==shell];assert sorted(r['rank'] for r in records)==candidates
        methods={'physics_frozen':list(saved['order']),'energy_prefix':candidates,
                 'spread':[start+i for i in SPREAD], 'spread_reverse':[start+19-i for i in SPREAD]}
        for method,key in BARE.items():methods[method]=[r['rank'] for r in sorted(records,key=lambda row:(-row[key],row['rank']))]
        for method in METHODS:
            assert sorted(methods[method])==candidates
            if method in ('physics_frozen','energy_prefix','spread','spread_reverse'):
                oldname={'physics_frozen':'physics','energy_prefix':'prefix'}.get(method,method)
                for b in BUDGETS:
                    assert selection['spaces'][f'{shell}_{oldname}_{b}']==list(range(340))+sorted(methods[method][:b])
        orders[shell]=methods
    return orders


def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);ranks=[0.]*len(values);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):ranks[order[k]]=(i+j+1)/2
        i=j
    return ranks


def pearson(a,b):
    ma=sum(a)/len(a);mb=sum(b)/len(b)
    numerator=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    denominator=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return numerator/denominator if denominator else None


def order_spearman(order,weights):
    # Position1 is best; use negative W so both ranks ascend in importance order.
    return pearson(list(range(1,len(order)+1)),average_ranks([-weights[r] for r in order]))


def metrics_for_selection(order,weights,b,compatible=True):
    if not compatible:return dict(attribution_total_MHz=None,selected_absolute_attribution_MHz=None,
        captured_fraction=None,top_k_recovered=None,top_k_recovery_fraction=None,top3_recovered=None,
        top3_recovery_fraction=None,dominant_included=None,dominant_position=None)
    truth=sorted(weights,key=lambda r:(-weights[r],r));chosen=order[:b];total=sum(weights.values())
    assert total>0 and len(chosen)==b and len(set(chosen))==b
    intersection=len(set(chosen)&set(truth[:b]));top3=len(set(chosen)&set(truth[:3]))
    return dict(attribution_total_MHz=total,selected_absolute_attribution_MHz=sum(weights[r] for r in chosen),
                captured_fraction=sum(weights[r] for r in chosen)/total,top_k_recovered=intersection,
                top_k_recovery_fraction=intersection/b,top3_recovered=top3,top3_recovery_fraction=top3/3,
                dominant_included=truth[0] in chosen,dominant_position=order.index(truth[0])+1)


def synthetic_tests():
    weights={10:4.,11:3.,12:2.,13:1.};order=[10,12,11,13]
    m=metrics_for_selection(order,weights,2)
    assert m['top_k_recovered']==1 and m['captured_fraction']==.6 and m['dominant_included']
    assert all(value is None for value in metrics_for_selection(order,weights,2,False).values())
    assert abs(order_spearman(order,weights)-(1-6*2/(4*(4*4-1))))<1e-14
    assert average_ranks([1,2,2,4])==[1.,2.5,2.5,4.]
    row=dict(shell='S1',rank='340',T300='1',Tactual='2',Vplus='3',Dplus='4',W_MHz='99',ExactC='55',BaseTrace='66')
    clean=bare_input([row]);row.update(W_MHz='-999',ExactC='garbage',BaseTrace='NaN')
    assert bare_input([row])==clean and 'W_MHz' not in clean[0] and 'BaseTrace' not in clean[0]
    assert sum(abs(x) for x in (-2.,2.))==4 and sum((-2.,2.))==0
    assert metrics_for_selection([11,10],{10:1.,11:1.},1)['top_k_recovered']==0
    return ['top-k intersection and absolute-weight capture','incompatible-base null metrics',
            'independent Spearman permutation identity','average ties','forbidden-label/trace-column invariance',
            'absolute attribution versus signed cancellation','lower-rank attribution tie rule']


def run(out):
    start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60 second cap')));signal.alarm(60)
    def monitor():
        while True:
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
            if peak>256*2**20:os._exit(70)
            time.sleep(.05)
    threading.Thread(target=monitor,daemon=True).start()
    baseline={str(p.relative_to(ROOT)):sha(p) for p in required_paths()};git_before=status()
    tests=synthetic_tests();auth=authenticate()
    out.mkdir(parents=True,exist_ok=True)
    selection=read(PRE/'selection.json')
    assert selection['base']==340 and selection['provenance']==read(PRE/'selection.json.seal.json')['provenance']
    feature_quality={}
    for shell in ('S1','S2'):
        saved=selection['shells'][shell]
        groups=[g for feature in saved['features'] for g in feature['groups']]
        assert all(g['minimum_denominator_GHz']>.010 and g['maximum_eta']<.10 for g in groups)
        for feature in saved['features']:
            assert [tuple(g['states']) for g in feature['groups']]==[(0,1),(2,3),(7,8)]
        assert len(saved['normalization'])==3
        for record in saved['normalization']:
            assert math.isclose(record['floor_GHz'],max(1e-14,1e-10*max(record['eigenvalues_GHz'])),rel_tol=1e-14)
        feature_quality[shell]=dict(normalization=saved['normalization'],
            floored_directions=sum(e<record['floor_GHz'] for record in saved['normalization'] for e in record['eigenvalues_GHz']),
            minimum_saved_isolated_denominator_GHz=min(g['minimum_denominator_GHz'] for g in groups),
            maximum_saved_eta=max(g['maximum_eta'] for g in groups),
            raw_matrix_recomputation=False)
    with (WP2/'SHELL_COMPARISON.csv').open(newline='') as f:bare=bare_input(csv.DictReader(f))
    orders=make_orders(selection,bare)
    frozen=dict(protocol_sha256=sha(HERE/'TRANSFER_PROTOCOL.md'),selection_sha256=SELECTION_SHA,
                label_source_parsed=False,orders=orders,budgets=list(BUDGETS),ranking_inputs=['sealed order/control rank lists','whitelisted bare columns only'])
    write(out/'RANKINGS_FROZEN.json',frozen)
    # Evaluation labels first parsed below; the ranking object is now fixed.
    frozen_hash=sha(out/'RANKINGS_FROZEN.json');labels=read(PRE/'k380_metrics.json')
    assert labels['selection_sha256']==SELECTION_SHA
    with (WP2/'RANKING_ANALYSIS.csv').open(newline='') as f:
        reused={(r['shell'],r['predictor']):float(r['spearman']) for r in csv.DictReader(f)
                if r['population']=='full' and r['predictor'] in BARE.values()}
    rows=[];correlations=[];evidence=[]
    for shell,lower in (('S1',340),('S2',360)):
        sh=labels['shell_diagnostics'][shell];assert sh['rank_range']==[lower,lower+19] and sh['P_dimension']==8*lower
        weights={r:0. for r in range(lower,lower+20)}
        for s in (0,1,2,3,7,8):
            entries=sh['states'][str(s)]['ranks'];assert [r['rank'] for r in entries]==list(weights)
            for record in entries:weights[record['rank']]+=abs(record['exact_self_energy_MHz'])
        assert abs(sum(weights.values())-sh['aggregate_rank_participation']['absolute_total_MHz'])<1e-10
        for method in METHODS:
            order=orders[shell][method];compatible=not(shell=='S2' and method=='physics_frozen')
            if not compatible:rho=None;correlation_source='INCOMPLETE: P340 order has no matching P340 labels'
            elif method in BARE:rho=reused[shell,BARE[method]];correlation_source='reused completed WP2 calculation'
            else:rho=order_spearman(order,weights);correlation_source='new evaluation of recorded order'
            correlations.append(dict(shell=shell,method=method,status='COMPLETE' if compatible else 'INCOMPLETE',spearman=rho,source=correlation_source))
            for b in BUDGETS:
                selected=order[:b]
                # Incomplete physics keeps its actual historical P340 support.
                ranking_base=340 if method=='physics_frozen' else None
                bare_prefix=(lower if method=='bare_Tactual' else 300) if method in BARE else None
                retained_base=340 if method=='physics_frozen' else lower
                row=dict(shell=shell,method=method,budget=b,
                    status='INCOMPLETE_BASE_MISMATCH' if not compatible else ('COMPATIBLE_S1' if shell=='S1' else 'CONTROL_ONLY_P360'),
                    evidence_class='retrospective_compatible_label_evaluation' if compatible and shell=='S1' else ('insufficient_data' if not compatible else 'retrospective_baseline_only'),
                    selection_parent_k=380,label_parent_k=380,ranking_dressed_base=ranking_base,bare_partner_prefix=bare_prefix,selected_support_base=retained_base,
                    label_base=lower,label_rank_start=lower,label_rank_end=lower+19,
                    selected_order=selected,selected_sorted=sorted(selected),full_retained_ranks=list(range(retained_base))+sorted(selected),
                    spearman=rho,correlation_source=correlation_source,
                    leakage_class='A+B+C; stored pre-evaluation matrix-greedy order' if method=='physics_frozen' else 'A+C; fixed retrospective bare/control rule',
                    **metrics_for_selection(order,weights,b,compatible))
                rows.append(row)
        evidence.append(dict(shell=shell,label_base=lower,total_absolute_attribution_MHz=sum(weights.values()),
            label_order=sorted(weights,key=lambda r:(-weights[r],r)),weights_MHz=weights))
    assert len(rows)==64 and sum(r['status']=='INCOMPLETE_BASE_MISMATCH' for r in rows)==4
    assert sha(out/'RANKINGS_FROZEN.json')==frozen_hash
    csvwrite(out/'TRANSFER_RESULTS.csv',rows)
    write(out/'TRANSFER_SUMMARY.json',dict(protocol_sha256=sha(HERE/'TRANSFER_PROTOCOL.md'),frozen_rankings_sha256=frozen_hash,
          rows=rows,correlations=correlations,label_evidence=evidence,historical_feature_quality=feature_quality,
          feasibility='PARTIAL',feature_leakage='LIMITATIONS',base_compatible_labels='PARTIAL',retrospective_transfer='INCOMPLETE'))
    changed=[rel for rel,h in baseline.items() if sha(ROOT/rel)!=h]
    assert not changed and status()==git_before
    auth.update(consumed_and_protected_hashes=baseline,preservation='PASS',git_status_before=git_before,git_status_after=status(),changed=changed)
    write(out/'SOURCE_AUTHENTICATION.json',auth)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    assert rss<256*2**20
    write(out/'NUMERICAL_VALIDATION.json',dict(status='PASS',synthetic_tests=tests,new_eigensolves=0,checkpoint_arrays_loaded=0,
          dressed_vectors_reconstructed=0,whitening_or_feature_recomputation=False,prior_correlations_recomputed=False,
          preservation='PASS',protected_files_checked=len(baseline),source_authentication='PASS',results_rows=64,
          incompatible_selector_rows=4,compatible_S1_rows=32,baseline_only_S2_rows=28,
          elapsed_seconds=time.monotonic()-start,peak_resident_bytes=rss))
    signal.alarm(0);print(json.dumps(dict(status='PASS',rows=len(rows),protected_files=len(baseline),new_eigensolves=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE);args=parser.parse_args();run(args.output)
