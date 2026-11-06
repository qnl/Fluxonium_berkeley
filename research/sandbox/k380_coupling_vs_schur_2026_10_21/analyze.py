"""Retrospective saved-operator/saved-response analysis; no eigenproblem paths."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FP=ROOT/'research/sandbox/k380_high_rank_state_fingerprints_2026_10_21'
REV=ROOT/'research/sandbox/epsj011_k380_diagnostic_review'
PRE=ROOT/'research/sandbox/epsj011_k380_preregistration'
CP=ROOT/'research/checkpoints/four-mode-asymmetric-n71-epsj011-k380'
TARGETS=(0,1,2,3,7,8)
PAIRS=((0,1),(2,3),(7,8))
BARE=('T300','Tactual','G0_11','G11_40','G40_100','G100_200','G200_300','Dgap','Dplus',
      'V300','Vplus','N1','N2','N3','X2','X3','Charge','Phase','RefInverse','Tref','Concentration')
BASE=('BaseC','BaseC01','BaseC23','BaseC78','BaseMinDen','BaseTrace','BaseTrace01','BaseTrace23','BaseTrace78')
EXACT=('ExactC','ExactC01','ExactC23','ExactC78','ExactRef')
MISSING=('DressedChannel','DressedInterference','MatchedP360Selector')
CATALOG={**dict.fromkeys(BARE,'bare'),**dict.fromkeys(BASE,'saved_base_target'),
         **dict.fromkeys(EXACT,'saved_exact_target'),**dict.fromkeys(MISSING,'unavailable')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def gitstatus():return subprocess.check_output(['git','status','--short'],cwd=ROOT).decode()


def source_files():
    paths=set()
    for directory in (FP,REV,CP):
        paths.update(p for p in directory.iterdir() if p.is_file())
    names=('criteria.json','preregistration_manifest.json','preregistration_seal.json','selection.json',
           'selection.json.seal.json','k380_metrics.json','evaluation.json','execution_receipt.json',
           'analyze_k380.py','REPORT.md')
    paths.update(PRE/p for p in names)
    paths.update(ROOT/p for p in ('AGENTS.md','README.md','Circuit_Objs/qchard_gridium_netlist.py',
        'Simulations/Rabi_3Photon/run_four_mode_spatial71_reference.py','research/sandbox/k300_preregistration/selector.py',
        'EmilianoUPDATE.md','EmilianoUPDATE-unsaved-backup.md','Simulations/Cphase/main.ipynb',
        'Simulations/Cphase/workflow_funcs.py','.git/index'))
    # Narrow sealed-artifact boundary, not the earlier broad 456-file inventory.
    for directory in (ROOT/'research/sandbox').iterdir():
        if directory.is_dir() and 'epsj011_k380' in directory.name:
            paths.update(p for p in directory.iterdir() if p.is_file() and
                         (p.suffix=='.md' or any(s in p.name for s in ('seal','manifest','authentication'))))
    return sorted(p for p in paths if p.is_file())


def authenticate():
    fp_manifest=read(FP/'OUTPUT_HASHES.json')
    for name,digest in fp_manifest.items():assert sha(FP/name)==digest,name
    previous=read(FP/'SOURCE_AUTHENTICATION.json')
    assert previous['status']=='PASS'
    meta,marker=read(CP/'checkpoint.json'),read(CP/'COMPLETE')
    assert sha(CP/'checkpoint.json')=='399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8'
    assert meta['status']==marker['status']=='complete' and marker['checkpoint_sha256']==sha(CP/'checkpoint.json')
    config=meta['configuration'];fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert fingerprint==meta['configuration_sha256']==marker['configuration_sha256']==previous['configuration_sha256']
    assert fingerprint=='8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e'
    assert config==read(PRE/'criteria.json')['profile']['configuration']
    assert config['spatial_cutoffs']==dict(n1max=4,N2=71,N3=71,N4=8,L2=11.,L3=14.)
    assert config['solver_settings']['k']==380
    for name,digest in meta['file_sha256'].items():assert sha(CP/name)==digest
    manifest=read(PRE/'preregistration_manifest.json')
    assert read(PRE/'preregistration_seal.json')['manifest_sha256']==sha(PRE/'preregistration_manifest.json')
    assert read(PRE/'selection.json.seal.json')['selection_sha256']==sha(PRE/'selection.json')
    relevant=['Circuit_Objs/qchard_gridium_netlist.py','Simulations/Rabi_3Photon/run_four_mode_spatial71_reference.py',
              'research/sandbox/k300_preregistration/selector.py','research/sandbox/epsj011_k380_preregistration/analyze_k380.py']
    checked={}
    for rel in relevant:
        expected=manifest['files'].get(rel) or previous['source_pins'].get(rel)
        assert expected and sha(ROOT/rel)==expected,rel
        checked[rel]=expected
    pins={PRE/'k380_metrics.json':'aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e',
          REV/'DIAGNOSTICS.json':'07b4682611a6fb5468e4a1789675decc88331dc256b4664aaaf10801b64d59a7',
          REV/'SOURCE_HASHES.json':'b15fbfe488906523f374b3393ca976165950c90e366f3739c52669a42ac1944b'}
    for p,h in pins.items():assert sha(p)==h
    assert meta['validation']['maximum_eigenpair_residual']<=1e-7
    assert meta['validation']['orthogonality_residual_inf']<=1e-8
    return dict(status='PASS',configuration_sha256=fingerprint,production_sources=checked,
                fingerprint_manifest_sha256=sha(FP/'OUTPUT_HASHES.json'),fingerprint_artifacts_checked=len(fp_manifest),
                prior_authentication_reused=True,broad_preservation_audit_rerun=False),meta


def ordered(rows,key):return sorted(rows,key=lambda row:(-row[key],row['rank']))


def corr(x,y,kind):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float((pearsonr(x,y) if kind=='pearson' else spearmanr(x,y)).statistic)


def evaluate(rows,key,shell,removed):
    dominant=354 if shell=='S1' else 362
    eligible=[r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    result=dict(shell=shell,population='without_dominant' if removed else 'full',predictor=key,
                source_class=CATALOG[key],status='COMPLETE' if len(eligible)==len(rows) else 'INCOMPLETE',
                eligible=len(eligible),population_size=len(rows),coverage=len(eligible)/len(rows),
                pearson=None,log10_pearson=None,log10_eligible=0,spearman=None,
                top3_recovered=None,top3_fraction=None,dominant_position=None,
                mean_absolute_rank_error=None,maximum_rank_error=None,predicted_top3=[],attribution_top3=[],largest_disagreements=[])
    if not eligible:
        if key=='MatchedP360Selector' and shell=='S1':result['status']='NOT_APPLICABLE'
        return result
    x=np.array([r[key] for r in eligible]);y=np.array([r['W_MHz'] for r in eligible])
    positive=(x>0)&(y>0)
    result.update(pearson=corr(x,y,'pearson'),spearman=corr(x,y,'spearman'),
                  log10_eligible=int(sum(positive)),log10_pearson=corr(np.log10(x[positive]),np.log10(y[positive]),'pearson'))
    pred=ordered(eligible,key);obs=ordered(eligible,'W_MHz')
    predpos={r['rank']:i+1 for i,r in enumerate(pred)};obspos={r['rank']:i+1 for i,r in enumerate(obs)}
    a=[r['rank'] for r in pred[:3]];b=[r['rank'] for r in obs[:3]]
    errors=[dict(rank=r['rank'],predictor_position=predpos[r['rank']],attribution_position=obspos[r['rank']],
                 absolute_difference=abs(predpos[r['rank']]-obspos[r['rank']])) for r in eligible]
    result.update(top3_recovered=len(set(a)&set(b)),top3_fraction=len(set(a)&set(b))/3,
                  dominant_position=predpos.get(dominant),predicted_top3=a,attribution_top3=b,
                  mean_absolute_rank_error=float(np.mean([r['absolute_difference'] for r in errors])),
                  maximum_rank_error=max(r['absolute_difference'] for r in errors),
                  largest_disagreements=sorted(errors,key=lambda r:(-r['absolute_difference'],r['rank']))[:3])
    return result


def synthetic_tests():
    # Independent tie/correlation and missing handling; no eigensolve even in tests.
    x=np.array([1.,2.,2.,4.]);y=np.array([4.,1.,3.,2.])
    xm=x-x.mean();ym=y-y.mean()
    manual=float(xm@ym/np.sqrt((xm@xm)*(ym@ym)))
    assert abs(corr(x,y,'pearson')-manual)<1e-14
    xr=rankdata(x);yr=rankdata(y)
    assert abs(corr(x,y,'spearman')-corr(xr,yr,'pearson'))<1e-14
    assert np.allclose(xr,[1,2.5,2.5,4])
    rows=[dict(rank=r,W_MHz=float(5-r),T300=1.) for r in range(4)]
    assert [r['rank'] for r in ordered(rows,'T300')]==[0,1,2,3]
    result=evaluate(rows,'T300','S1',False)
    assert result['spearman'] is None and result['top3_recovered']==3
    rows[0]['T300']=None;rows[1]['T300']=0.
    result=evaluate(rows,'T300','S1',False)
    assert result['eligible']==3 and result['log10_eligible']==2 and result['status']=='INCOMPLETE'
    assert corr([1,2],[3,4],'pearson') is None
    c=np.array([1+2j,3-4j]);coherent=abs(c.sum())**2;incoherent=float(np.sum(abs(c)**2))
    assert coherent!=incoherent
    assert np.isclose(abs((c*np.exp(.7j)).sum())**2,coherent)
    no=np.array([[0,-1j],[1j,0]]);th=np.array([[0,1],[1,0]])
    block=c[0]*no+c[1]*th
    assert np.allclose(block.conj().T,c[0].conjugate()*no+c[1].conjugate()*th)
    p=np.array([1.,2.,3.]);p/=p.sum()
    assert np.isclose(p.sum(),1) and 1/3<=sum(p*p)<=1
    assert np.isclose(corr(x,y,'pearson'),corr(1000*x,1000*y,'pearson'))
    assert np.isclose(corr(x,y,'spearman'),corr(1000*x,1000*y,'spearman'))
    return ['independent Pearson formula','average-tie Spearman','deterministic ties/top3',
            'missing/zero/constant handling','complex coherent/gauge invariance','Hermitian conjugate block',
            'concentration normalization','energy-unit correlation invariance']


def run(out):
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 second cap')));signal.alarm(180)
    def monitor():
        while True:
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
            if rss>512*2**20:os._exit(70)
            time.sleep(.05)
    threading.Thread(target=monitor,daemon=True).start()
    out.mkdir(parents=True,exist_ok=True)
    baseline={str(p.relative_to(ROOT)):sha(p) for p in source_files()};status=gitstatus()
    tests=synthetic_tests();auth,meta=authenticate()
    with np.load(CP/'stage1_eigensystem.npz',allow_pickle=False) as z:e=z['eigenvalues_GHz']
    assert e.shape==(380,) and np.all(np.diff(e)>0)
    # Never request the eigenvectors archive member.
    with np.load(CP/'projected_operators.npz',allow_pickle=False) as z:
        ops={k:z[k] for k in ('n1','n2','n3','x2','x3')}
    herm={k:float(np.max(abs(o-o.conj().T))) for k,o in ops.items()}
    assert all(o.shape==(380,380) for o in ops.values()) and max(herm.values())<1e-10
    fp=read(FP/'DIAGNOSTICS.json');coeff=fp['coefficients']
    ec=np.array(coeff['ECmat']);k=np.array(coeff['K']);f4=coeff['f4'];g4=coeff['g4'];omega=coeff['oscillator_spacing_GHz']
    # Verified numerical coefficient values are reused, not refitted or recomputed.
    ladder=np.diag(np.sqrt(np.arange(1,8)),1)
    n4=1j*g4*(ladder.T-ladder);th4=f4*(ladder+ladder.T)
    channels=[8*ec[i,3]*ops['n'+str(i+1)] for i in range(3)]+[k[0,2]*ops['x2'],k[1,2]*ops['x3']]
    a=sum(channels[:3]);b=sum(channels[3:])
    normn=np.linalg.norm(n4)**2;normx=np.linalg.norm(th4)**2;inner=np.vdot(n4,th4)
    t=abs(a)**2*normn+abs(b)**2*normx+2*np.real(a.conj()*b*inner)
    h01=a*n4[0,1]+b*th4[0,1];vac=abs(h01)**2
    assert abs(inner)<1e-12 and np.allclose((a*n4[1,0]+b*th4[1,0]),h01.conj().T,atol=1e-12)
    assert np.min(t)>-1e-12
    metrics=read(PRE/'k380_metrics.json');selection=read(PRE/'selection.json');review=read(REV/'DIAGNOSTICS.json')
    assert selection['base']==340
    features={}
    matrix_error=0.
    for shell in ('S1','S2'):
        saved=selection['shells'][shell]
        raw=np.asarray(saved['raw_importance']['real'])+1j*np.asarray(saved['raw_importance']['imag'])
        assert raw.shape==(20,3,2,2)
        assert np.max(abs(raw-raw.conj().swapaxes(-1,-2)))<1e-12
        for i,record in enumerate(saved['features']):
            features[record['rank']]=record['groups']
            assert [tuple(g['states']) for g in record['groups']]==list(PAIRS)
            for gi,group in enumerate(record['groups']):
                err=abs(float(np.trace(raw[i,gi]).real)-group['trace_GHz']);matrix_error=max(matrix_error,err)
                assert err<1e-12 and group['minimum_denominator_GHz']>0
                assert group['trace_GHz']<=group['coupling_norm_GHz']**2/group['minimum_denominator_GHz']+1e-12
    oldrows={r['rank']:r for r in fp['states']}
    rows=[];target_rows=[];fingerprint_errors=[]
    for shell,lower in (('S1',340),('S2',360)):
        sh=metrics['shell_diagnostics'][shell]
        assert sh['rank_range']==[lower,lower+19] and sh['P_dimension']==lower*8
        refs=metrics['prefix_metrics'][str(lower)]['energies_GHz']
        for r in range(lower,lower+20):
            delta=e[r]-e[:300];den=delta+omega
            reference=np.array([e[r]+omega/2-refs[s] for s in TARGETS])
            assert min(delta)>0 and min(reference)>0
            row=dict(shell=shell,parent_k=380,rank=r,energy_GHz=float(e[r]),exact_P=lower,selector_P=340,
                     bare_denominator_min_GHz=float(min(delta)),bare_denominator_max_GHz=float(max(delta)),
                     reference_denominator_min_GHz=float(min(reference)),reference_denominator_max_GHz=float(max(reference)))
            own=t[:lower,r];weights=t[:300,r];total=float(weights.sum());prob=weights/total
            row.update(T300=total,Tactual=float(own.sum()),Dgap=float(np.sum(weights/delta)),Dplus=float(np.sum(weights/den)),
                       V300=float(vac[:300,r].sum()),Vplus=float(np.sum(vac[:300,r]/den)),
                       RefInverse=float(np.mean(1/reference)),Tref=float(total*np.mean(1/reference)),
                       Concentration=float(np.sum(prob**2)),partner_top1_fraction=float(max(prob)),
                       partner_top3_fraction=float(np.sort(prob)[-3:].sum()),partner_effective_count=float(1/np.sum(prob**2)),
                       strongest_partner=int(np.argmax(weights)))
            for name,(lo,hi) in zip(('G0_11','G11_40','G40_100','G100_200','G200_300'),((0,11),(11,40),(40,100),(100,200),(200,300))):
                row[name]=float(t[lo:hi,r].sum())
            for i,name in enumerate(('N1','N2','N3','X2','X3')):
                row[name]=float(np.sum(abs(channels[i][:300,r])**2)*(normn if i<3 else normx))
            row['Charge']=float(np.sum(abs(a[:300,r])**2)*normn)
            row['Phase']=float(np.sum(abs(b[:300,r])**2)*normx)
            for label,coherent,incoherent in (
                ('charge',row['Charge'],row['N1']+row['N2']+row['N3']),
                ('phase',row['Phase'],row['X2']+row['X3']),
                ('charge_phase',row['T300'],row['Charge']+row['Phase']),
                ('vacuum',row['V300'],float(np.sum(abs(a[:300,r]*n4[0,1])**2+abs(b[:300,r]*th4[0,1])**2)))):
                row[label+'_cross_GHz2']=coherent-incoherent
                row[label+'_cross_fraction']=(coherent-incoherent)/incoherent if incoherent>0 else None
            base=features[r];cs=np.array([g['coupling_norm_GHz']**2 for g in base]);traces=np.array([g['trace_GHz'] for g in base]);md=np.array([g['minimum_denominator_GHz'] for g in base])
            row.update(BaseC=float(cs.sum()),BaseTrace=float(traces.sum()),BaseMinDen=float(sum(cs/md)))
            exact=[];w=[]
            for gi,pair in enumerate(PAIRS):
                tag=''.join(map(str,pair));row['BaseC'+tag]=float(cs[gi]);row['BaseTrace'+tag]=float(traces[gi])
                row['selector_min_den_'+tag+'_GHz']=float(md[gi])
            for si,s in enumerate(TARGETS):
                entries=sh['states'][str(s)]['ranks'];assert [j['rank'] for j in entries]==list(range(lower,lower+20))
                saved=entries[r-lower];sig=saved['exact_self_energy_MHz'];c=saved['coupling_norm_GHz']
                w.append(abs(sig));exact.append(c*c)
                target_rows.append(dict(shell=shell,rank=r,target=s,pair=''.join(map(str,PAIRS[si//2])),
                    signed_rank_scalar_MHz=sig,exact_coupling_GHz=c,Q_probability=saved['Q_probability'],
                    dressed_reference_energy_GHz=refs[s],bare_reference_separation_GHz=float(reference[si]),
                    shell_minimum_abs_dressed_denominator_GHz=sh['states'][str(s)]['minimum_abs_denominator_GHz']))
            row.update(W_MHz=float(sum(w)),ExactC=float(sum(exact)),ExactRef=float(sum(np.asarray(exact)/reference)))
            for gi,pair in enumerate(PAIRS):
                tag=''.join(map(str,pair));row['ExactC'+tag]=float(sum(exact[2*gi:2*gi+2]))
                row['W_pair'+tag+'_fraction']=sum(w[2*gi:2*gi+2])/row['W_MHz']
            row['shell_weight_fraction']=row['W_MHz']/sh['aggregate_rank_participation']['absolute_total_MHz']
            assert np.isclose(row['T300'],row['Charge']+row['Phase'],atol=1e-10)
            assert np.isclose(sum(row[g] for g in ('G0_11','G11_40','G40_100','G100_200','G200_300')),total,atol=1e-10)
            assert all(np.isfinite(row[p]) and row[p]>=0 for p in BARE+BASE+EXACT)
            for new,old in (('T300','p300_block_strength_GHz2'),('Vplus','p300_vacuum_proxy_GHz'),('W_MHz','schur_absolute_weight_MHz')):
                err=abs(row[new]-oldrows[r][old]);fingerprint_errors.append(err)
                assert np.isclose(row[new],oldrows[r][old],rtol=1e-12,atol=1e-11)
            rows.append(row)
        assert abs(sum(row['W_MHz'] for row in rows if row['shell']==shell)-sh['aggregate_rank_participation']['absolute_total_MHz'])<1e-10
    rankings=[]
    for shell,dominant in (('S1',354),('S2',362)):
        full=[r for r in rows if r['shell']==shell]
        assert ordered(full,'W_MHz')[0]['rank']==dominant
        for key in BARE+BASE+EXACT:
            for i,row in enumerate(ordered(full,key)):row[key+'_position']=i+1
        for i,row in enumerate(ordered(full,'W_MHz')):row['attribution_position']=i+1
        for removed in (False,True):
            population=[r for r in full if not removed or r['rank']!=dominant]
            for key in CATALOG:rankings.append(evaluate(population,key,shell,removed))
    assert len(rankings)==2*2*38 and len(rows)==40
    csvwrite(out/'SHELL_COMPARISON.csv',rows);csvwrite(out/'RANKING_ANALYSIS.csv',rankings);csvwrite(out/'TARGET_DIAGNOSTICS.csv',target_rows)
    history=next(r for r in review['rank_results'] if r['rank']==328)
    context=dict(source='epsj011_k380_diagnostic_review/DIAGNOSTICS.json',attribution_parent_k=340,
                 fingerprint_parent_k=380,rank=328,historical_record=history,fingerprint=oldrows[328],pooled=False)
    write(out/'CONTEXT_RANK328.json',context)
    definition_hashes={p:sha(HERE/p) for p in ('RESEARCH_PLAN.md','PREDICTOR_DEFINITIONS.md')}
    write(out/'RESULTS.json',dict(definition_hashes=definition_hashes,catalog=CATALOG,states=rows,ranking_analysis=rankings,
          targets=target_rows,omega_GHz=omega,unavailable=list(MISSING),historical_rank328=context))
    after={p:sha(ROOT/p) for p in baseline};changed=[p for p in baseline if baseline[p]!=after[p]]
    assert not changed and gitstatus()==status
    auth.update(consumed_and_protected_hashes=baseline,git_status_before=status,git_status_after=gitstatus(),preservation='PASS')
    write(out/'SOURCE_AUTHENTICATION.json',auth)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    assert peak<512*2**20
    validation=dict(status='PASS',tests=tests,hermiticity_errors=herm,selector_trace_max_error=matrix_error,
                    fingerprint_max_absolute_comparison_error=max(fingerprint_errors),attribution_shell_counts=[20,20],
                    available_predictors=35,unavailable_predictors=3,ranking_rows=len(rankings),target_rows=len(target_rows),
                    eigenvector_members_loaded=0,new_eigensolves=0,definition_hashes=definition_hashes,
                    preservation_files_checked=len(baseline),preservation='PASS',elapsed_seconds=time.monotonic()-start,peak_resident_bytes=peak)
    write(out/'RUN_VALIDATION.json',validation);signal.alarm(0)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE);args=parser.parse_args();run(args.output)
