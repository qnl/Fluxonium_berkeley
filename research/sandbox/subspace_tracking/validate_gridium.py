"""Authenticated saved-checkpoint validation. Only small Stage-2 reconstructions.

No large spatial Hamiltonian is assembled. Maximum eigensolve is 1440 x 1440;
all Stage-1 vectors are read, never recalculated. Outputs confined to this folder.
"""
from pathlib import Path
import ast,csv,hashlib,json,time
import numpy as np
from scipy import linalg as la
from tracking import (track_states,track_subspaces,ambiguity_clusters,embed_retained,
    compare_blocks,block_invariants,pathway_signature)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CP71=ROOT/'research/checkpoints/four-mode-asymmetric-n71-k180'
CP51=ROOT/'research/checkpoints/2026-09-26-asymmetric-n51'
HEAVY={'reference_frame','candidate_frame','reference_transform','candidate_transform'}

def serial(x):
    if isinstance(x,np.ndarray):
        return {'real':x.real.tolist(),'imag':x.imag.tolist()} if np.iscomplexobj(x) else x.tolist()
    if isinstance(x,(complex,np.complexfloating)): return dict(real=float(x.real),imag=float(x.imag))
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items() if k not in HEAVY}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    return x

def save(name,x): (HERE/name).write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n')

def table(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def authenticate():
    md=json.loads((CP71/'checkpoint.json').read_text()); marker=json.loads((CP71/'COMPLETE').read_text())
    assert md['status']==marker['status']=='complete'
    assert sha(CP71/'checkpoint.json')==marker['checkpoint_sha256']
    cfg=hashlib.sha256(json.dumps(md['configuration'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert cfg==md['configuration_sha256']==marker['configuration_sha256']=='2c996f23c3f4810f9bb7d72f9231f264cda516900b4a7067af2ab53dbd1b868a'
    for k,h in md['file_sha256'].items(): assert sha(CP71/k)==h
    analysis=json.loads((CP71/'analysis.json').read_text())
    for k,h in analysis['n51_checkpoint_provenance']['file_sha256'].items(): assert sha(CP51/k)==h
    source=ROOT/'Circuit_Objs/qchard_gridium_netlist.py'
    assert sha(source)==md['provenance']['source_file_sha256'][str(source.relative_to(ROOT))]
    # Read-only extraction of the existing pure NumPy coefficient helper avoids
    # importing unavailable QuTiP or any simulation entry point.
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_coeffs')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    legacy=ROOT/'research/sandbox/n71_retained_subspace_postprocess.json'
    save('provenance.json',dict(N71_checkpoint_sha256=marker['checkpoint_sha256'],
        N71_payload_sha256=md['file_sha256'],N51_file_sha256=analysis['n51_checkpoint_provenance']['file_sha256'],
        source_sha256=sha(source),retained_selection_file_sha256=sha(legacy),
        note='Saved basis selections are inputs, not new optimization. Reconstruct only Stage-2.'))
    return md,analysis,ns['_coeffs'],json.loads(legacy.read_text())

def components(cp,coeffs):
    md=json.loads((cp/'checkpoint.json').read_text()); cfg=md.get('configuration',md)
    p=cfg['physical_parameters']; sp=cfg['spatial_cutoffs']
    ec,k,_,_=coeffs(**{x:p[x] for x in ['EJ','EC','EL','ELK','EJS','ECS','eps_J','eps_LK','eC','eP']})
    with np.load(cp/'stage1_eigensystem.npz') as f: e=f['eigenvalues_GHz']; vectors=f['eigenvectors']
    with np.load(cp/'projected_operators.npz') as f: op={x:f[x] for x in f.files}
    n=sp['N4']; a=np.diag(np.sqrt(np.arange(1,n)),1); aa=4*ec[3,3]; bb=.5*k[2,2]; ff=(aa/(4*bb))**.25
    return dict(e=e,vectors=vectors,spatial=sp,n4=n,x4=ff*(a+a.T),p4=1j*(a.T-a)/(2*ff),
        w4=2*np.sqrt(aa*bb)*(np.arange(n)+.5),
        charge=8*sum(ec[j,3]*op['n'+str(j+1)] for j in range(3)),
        phase=k[0,2]*op['x2']+k[1,2]*op['x3'],phi=.5*op['x2']+op['x3'])

def apply_phi(c,v):
    return np.einsum('ab,bms->ams',c['phi'],v.reshape(180,c['n4'],-1),optimize=True).reshape(v.shape)

def solve(c,indices,label,nstates=48):
    idx=np.asarray(indices,int); n=c['n4']; sl=np.ix_(idx,idx)
    h=np.kron(np.diag(c['e'][idx]),np.eye(n))+np.kron(np.eye(len(idx)),np.diag(c['w4']))
    h=h+np.kron(c['charge'][sl],c['p4'])+np.kron(c['phase'][sl],c['x4']);h=(h+h.conj().T)/2
    start=time.perf_counter();e,v=la.eigh(h,subset_by_index=[0,nstates-1],driver='evr',check_finite=True)
    residual=float(np.max(la.norm(h@v-v*e,axis=0)))
    full=embed_retained(v,idx,180,n); phi=full.conj().T@apply_phi(c,full)
    result=dict(energy=e,vectors=full,phi=phi,indices=idx,seconds=time.perf_counter()-start,residual_GHz=residual)
    np.savez_compressed(HERE/(label+'.npz'),**result)
    print(label,'Stage2',len(h),'seconds',round(result['seconds'],2),'residual',residual,flush=True)
    return result

def interpolate(v,source,target):
    """Quadrature-aware physical tensor interpolation; never hides norm/Gram change."""
    for key in ['n1max','L2','L3']: assert source[key]==target[key]
    def weights(xs,xt):
        high=np.clip(np.searchsorted(xs,xt,side='right'),1,len(xs)-1); low=high-1
        f=(xt-xs[low])/(xs[high]-xs[low]); m=np.zeros((len(xt),len(xs)))
        m[np.arange(len(xt)),low]=1-f; m[np.arange(len(xt)),high]=f;return m
    xs=[np.linspace(-source['L'+k],source['L'+k],source['N'+k]) for k in ['3','2']]
    xt=[np.linspace(-target['L'+k],target['L'+k],target['N'+k]) for k in ['3','2']]
    m3,m2=[weights(s,t) for s,t in zip(xs,xt)]
    shaped=v.reshape(source['N3'],source['N2'],2*source['n1max']+1,-1)
    out=np.einsum('Aa,Bb,abcS->ABcS',m3,m2,shaped,optimize=True)
    factor=np.sqrt(np.prod([x[1]-x[0] for x in xt])/np.prod([x[1]-x[0] for x in xs]))
    return (out*factor).reshape(target['N3']*target['N2']*(2*target['n1max']+1),*v.shape[1:])

def coordinate(sp):
    t3=np.linspace(-sp['L3'],sp['L3'],sp['N3'])[:,None,None]
    t2=np.linspace(-sp['L2'],sp['L2'],sp['N2'])[None,:,None]
    return np.broadcast_to(t3+.5*t2,(sp['N3'],sp['N2'],2*sp['n1max']+1)).reshape(-1)

def subrow(label,group,r):
    return dict(case=label,group=group,reference_rank=r['reference_rank'],candidate_rank=r['candidate_rank'],
        minimum_singular_value=r['minimum_singular_value'],minimum_subspace_probability=r['minimum_singular_value']**2,
        maximum_angle_deg=float(max(r['principal_angles_degrees'])),mean_subspace_overlap=r['mean_subspace_overlap'],
        projector_frobenius=r['projector_frobenius'],projector_operator=r['projector_operator'],
        reference_gram_error=r['reference_gram']['gram_error'],candidate_gram_error=r['candidate_gram']['gram_error'])

def compare_case(label,u,v,er,ev,apply_operator,groups,shift_scales=None,reference_indices=None,candidate_indices=None):
    tracking=track_states(u,v,er,ev,shift_scales=shift_scales,
        reference_energy_indices=reference_indices,candidate_energy_indices=candidate_indices)
    greedy=track_states(u,v,er,ev,method='greedy')
    reports={}; subs=[]; ops=[]; vectors={}
    for tag,idx in groups.items():
        cand=tracking['assignment'][idx]
        if np.any(cand<0): continue
        r=track_subspaces(u[:,idx],v[:,cand]); reports[tag]=r; subs.append(subrow(label,tag,r))
        qu,qv=r['reference_frame'],r['candidate_frame']; rotation=r['rotation']
        a=qu.conj().T@apply_operator(qu);b=qv.conj().T@apply_operator(qv)
        comparison=compare_blocks(a,b,rotation,hermitian=True)
        ops.append(dict(case=label,group=tag,reference_frobenius=comparison['reference']['frobenius'],
            candidate_frobenius=comparison['candidate']['frobenius'],
            aligned_difference_frobenius=comparison['aligned_difference_frobenius'],
            aligned_difference_operator=comparison['aligned_difference_operator']))
        reports[tag]['operator_comparison']=comparison;vectors[tag]=(qu,qv,a,b,rotation)
    if 'logical01' in vectors and 'intermediate78' in vectors:
        ul,vl,_,_,rl=vectors['logical01'];uc,vc,ar,ac,rc=vectors['intermediate78']
        cr=uc.conj().T@apply_operator(ul);cc=vc.conj().T@apply_operator(vl)
        reports['pathway']=dict(reference=pathway_signature(cr,ar),candidate=pathway_signature(cc,ac),
            entry_comparison=compare_blocks(cr,cc,rc,rl))
    save(label+'_tracking.json',dict(tracking=tracking,greedy_total_probability=greedy['total_probability'],
        greedy_assignment=greedy['assignment'],subspaces=reports,
        automatic_clusters=ambiguity_clusters(tracking,u,v)))
    for row in tracking['rows']:
        row['case']=label
    return tracking,subs,ops

def main():
    md,analysis,coeffs,prior=authenticate();c71=components(CP71,coeffs);c51=components(CP51,coeffs)
    ref=solve(c71,range(180),'N71_k180'); ref51=solve(c51,range(180),'N51_k180',11)
    target=np.array(analysis['n71_reference_k180']['energies_GHz'])
    assert max(abs(ref['energy'][:11]-target))<1e-9
    assert np.max(abs(abs(ref['phi'][:11,:11])-analysis['n71_reference_k180']['grid_phi_abs']))<1e-8
    allrows=[]; subrows=[]; oprows=[]; checks=[]
    groups={'logical01':[0,1],'intermediate78':[7,8],'low11':list(range(11))}
    # All five previously selected N71-priority retained bases, plus original
    # checkpoint-authenticated k170/k172/k175 and the saved core160 stress case.
    cases={f'N71_priority_{k}':v for k,v in prior['candidates']['n71_priority_on_n71'].items()}
    cases.update({name:v for name,v in analysis['retained_basis_cases'].items()})
    cases['N71_core160']=prior['n71_core160']
    for label,case in cases.items():
        cand=solve(c71,case['retained_indices'],label)
        # Reference is candidate retained model; candidate is complete k180.
        er=cand['energy']-cand['energy'][0];ev=ref['energy']-ref['energy'][0]
        t,ss,oo=compare_case(label+'_to_k180',cand['vectors'],ref['vectors'],er,ev,
            lambda q:apply_phi(c71,q),groups)
        allrows+=t['rows'];subrows+=ss;oprows+=oo
        old=np.asarray(case['energies_GHz']); err=float(np.max(abs(cand['energy'][:11]-old)))
        assert err<1e-8,(label,err)
        checks.append(dict(case=label,low11_saved_energy_error_GHz=err,
            low11_minimum_probability=min(r['assigned_probability'] for r in t['rows'][:11]),
            high48_minimum_probability=min(r['assigned_probability'] for r in t['rows']),
            ambiguous_count=sum(r['ambiguous'] for r in t['rows']),
            ordering_changes=sum(bool(r['energy_rank_changed']) for r in t['rows'])))
    raw51=(c51['vectors']@ref51['vectors'].reshape(180,-1)).reshape(len(c51['vectors']),8,11)
    mapped51=interpolate(raw51,c51['spatial'],c71['spatial']).reshape(-1,11)
    raw71=(c71['vectors']@ref['vectors'][:,:11].reshape(180,-1)).reshape(-1,11)
    coord=np.repeat(coordinate(c71['spatial']),8)
    tr,ss,oo=compare_case('N51_to_N71_low11',mapped51,raw71,
        ref51['energy']-ref51['energy'][0],ref['energy'][:11]-ref['energy'][0],
        lambda q:coord[:,None]*q,groups)
    allrows+=tr['rows'];subrows+=ss;oprows+=oo
    saved=analysis['n51_to_n71_spatial_comparison']; savedamp=np.array(saved['absolute_overlap_matrix']).T
    overlap_error=float(np.max(abs(abs(tr['overlap'])-savedamp))); assert overlap_error<1e-8
    save('spatial_embedding_checks.json',dict(saved_overlap_amplitude_error=overlap_error,
        interpolation_column_norms=la.norm(mapped51,axis=0),
        normalized_gram_error=float(la.norm((mapped51/la.norm(mapped51,axis=0)).conj().T@(mapped51/la.norm(mapped51,axis=0))-np.eye(11),2)),
        operator_comparison='Common target-grid coordinate theta3+theta2/2, including interpolation and Gram whitening.'))
    del raw51,mapped51,raw71
    # Genuinely ambiguous saved high-energy Stage-1 shell; no eigensolve.
    shell=list(range(160,180)); mapped=interpolate(c51['vectors'][:,shell],c51['spatial'],c71['spatial'])
    v=c71['vectors'];coord1=coordinate(c71['spatial'])
    tr,ss,oo=compare_case('Stage1_shell_N51_to_N71',mapped,v,c51['e'][shell],c71['e'],
        lambda q:coord1[:,None]*q,{'shell160_179':list(range(20))},
        reference_indices=shell,candidate_indices=list(range(180)))
    # Save small cross/Gram matrices, not large raw physical arrays.
    np.savez_compressed(HERE/'shell_geometry.npz',overlap=tr['overlap'],
        reference_gram=(mapped/la.norm(mapped,axis=0)).conj().T@(mapped/la.norm(mapped,axis=0)),
        assignment=tr['assignment'],reference_energies=c51['e'][shell],candidate_energies=c71['e'])
    for row in tr['rows']: row['reference_stage1_rank']=shell[row['reference']]
    table('shell_assignments.csv',[{k:v for k,v in r.items() if k not in ['reasons','energy_watch_neighbors']} for r in tr['rows']])
    save('retained_validation_checks.json',checks);table('subspaces.csv',subrows+ss);table('operator_comparisons.csv',oprows+oo)
    table('state_assignments.csv',[{k:v for k,v in r.items() if k not in ['reasons','energy_watch_neighbors']} for r in allrows])
    save('run_summary.json',dict(reduced_eigensolves=11,maximum_eigensolve_dimension=1440,
        stage1_eigensolves=0,retained_cases=len(cases),all_saved_low11_checks_passed=True,
        spatial_overlap_reproduction_error=overlap_error))
    print('Gridium validation complete.',flush=True)

if __name__=='__main__':main()
