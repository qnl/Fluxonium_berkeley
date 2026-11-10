"""Bounded saved-data fingerprints. Never assembles H or calls an eigensolver."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import threading
import time

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CP = ROOT / 'research/checkpoints/four-mode-asymmetric-n71-epsj011-k380'
PRE = ROOT / 'research/sandbox/epsj011_k380_preregistration'
REV = ROOT / 'research/sandbox/epsj011_k380_diagnostic_review'
FOCAL = (328, 354, 362)
CONTROLS = tuple(range(300, 380, 10))
NEIGHBORS = tuple(sorted({r+d for r in FOCAL for d in (-2,-1,1,2)}))
SELECTED = tuple(sorted(set(FOCAL+CONTROLS+NEIGHBORS)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, sort_keys=True)+'\n')


def csvwrite(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def authenticate():
    pins = {
        CP/'checkpoint.json': '399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8',
        PRE/'preregistration_manifest.json': '95f33014769195605ef871f62fee229d0417f610fb6e96dd44df71e710707656',
        PRE/'k380_metrics.json': 'aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e',
        PRE/'evaluation.json': '43903489aba94442dc868dae2eafa319de82f3cf3e3441db6144d6568c4a91aa',
        PRE/'execution_receipt.json': 'c7ff154d805ded48367a28ac5137901ad9567d755372ed69bae64bc3c5c36d0b',
        REV/'DIAGNOSTICS.json': '07b4682611a6fb5468e4a1789675decc88331dc256b4664aaaf10801b64d59a7',
        REV/'SOURCE_HASHES.json': 'b15fbfe488906523f374b3393ca976165950c90e366f3739c52669a42ac1944b',
    }
    meta, marker = read(CP/'checkpoint.json'), read(CP/'COMPLETE')
    assert meta['status'] == marker['status'] == 'complete'
    assert marker['checkpoint_sha256'] == sha(CP/'checkpoint.json')
    expected = '8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e'
    config = meta['configuration']
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected
    assert meta['configuration_sha256'] == marker['configuration_sha256'] == expected
    assert config == read(PRE/'criteria.json')['profile']['configuration']
    assert config['spatial_cutoffs'] == dict(n1max=4,N2=71,L2=11.0,N3=71,L3=14.0,N4=8)
    assert config['solver_settings']['k'] == 380
    assert config['physical_parameters']['eps_J'] == .11
    manifest, seal = read(PRE/'preregistration_manifest.json'), read(PRE/'preregistration_seal.json')
    assert seal['status'] == 'SEALED' and seal['manifest_sha256'] == sha(PRE/'preregistration_manifest.json')
    pins.update({ROOT/p: v for p,v in manifest['files'].items()})
    pins.update({ROOT/p: v for p,v in meta['provenance']['source_file_sha256'].items()})
    pins.update({CP/p: v for p,v in meta['file_sha256'].items()})
    selection = PRE/'selection.json'
    assert read(PRE/'selection.json.seal.json')['selection_sha256'] == sha(selection)
    analysis = read(CP/'analysis.json')
    assert analysis['selection_freeze']['selection_sha256'] == sha(selection)
    assert analysis['selection_freeze']['frozen_before_candidate_or_full_new_shell_dressed_validation']
    assert sorted(int(k.rsplit('k',1)[1]) for k in analysis['retained_basis_cases']) == [300,320,340,360,380]
    for p,digest in pins.items():
        assert sha(p) == digest, str(p)
    assert meta['validation']['maximum_eigenpair_residual'] <= 1e-7
    assert meta['validation']['orthogonality_residual_inf'] <= 1e-8
    return meta, {'status':'PASS', 'configuration_sha256':expected,
                  'manifest_files_verified':len(manifest['files']),
                  'source_pins':{str(p.relative_to(ROOT)):d for p,d in pins.items()},
                  'boundary':'Local SHA-256 linkage, not a signature or independent timestamp.'}


def local_action(c, name, n, x2, x3, n2, n3):
    if name == 'n1': return c*n[None,None,:]
    if name == 'x2': return c*x2[None,:,None]
    if name == 'x3': return c*x3[:,None,None]
    if name == 'n2': return np.einsum('ab,cbn->can', n2, c)
    if name == 'n3': return np.einsum('ab,bcn->acn', n3, c)
    raise ValueError(name)


def metrics(p, n, x2, x3):
    pn, p2, p3 = p.sum(axis=(0,1)), p.sum(axis=(0,2)), p.sum(axis=(1,2))
    p23 = p.sum(axis=2)
    out = {}
    for name,prob in [('joint',p),('spatial_joint',p23),('charge',pn),('theta2',p2),('theta3',p3)]:
        ipr = float(np.sum(prob**2))
        out[name+'_ipr'] = ipr
        out[name+'_pr'] = 1/ipr
        out[name+'_normalized_pr'] = 1/ipr/prob.size
    area = (x2[1]-x2[0])*(x3[1]-x3[0])
    out['spatial_continuous_ipr_per_rad2'] = out['spatial_joint_ipr']/area
    out['spatial_pr_area_rad2'] = out['spatial_joint_pr']*area
    out['charge_edge_minus'] = float(pn[0])
    out['charge_edge_plus'] = float(pn[-1])
    out['charge_edge'] = float(pn[0]+pn[-1])
    for name,prob in [('theta2',p2),('theta3',p3)]:
        for k in (1,2):
            out[f'{name}_edge{k}_minus'] = float(prob[:k].sum())
            out[f'{name}_edge{k}_plus'] = float(prob[-k:].sum())
            out[f'{name}_edge{k}'] = float(prob[:k].sum()+prob[-k:].sum())
    for k in (1,2):
        mask = np.zeros(p23.shape, dtype=bool)
        mask[:k,:]=mask[-k:,:]=mask[:,:k]=mask[:,-k:]=True
        out[f'spatial_union_edge{k}'] = float(p23[mask].sum())
    for name,prob,x in [('charge',pn,n),('theta2',p2,x2),('theta3',p3,x3)]:
        mean, second = float(prob@x), float(prob@(x*x))
        out[name+'_mean'] = mean
        out[name+'_second_moment'] = second
        out[name+'_std'] = float(np.sqrt(max(0,second-mean**2)))
    out['theta2_theta3_covariance'] = float(np.einsum('ab,a,b->',p23,x3,x2)-out['theta2_mean']*out['theta3_mean'])
    out['grid_phi_mean'] = out['theta3_mean']+.5*out['theta2_mean']
    out['grid_phi_std'] = float(np.sqrt(max(0,out['theta3_std']**2+.25*out['theta2_std']**2+out['theta2_theta3_covariance'])))
    out['charge_marginal'] = pn.tolist()
    out['theta2_marginal'] = p2.tolist()
    out['theta3_marginal'] = p3.tolist()
    return out


def synthetic_tests():
    # Distinct axis sizes prevent equal-size spatial axes hiding swaps.
    c = np.zeros((4,3,2),complex); c[3,1,0]=1j
    assert np.argmax(abs(c.ravel())) == ((3*3)+1)*2
    d = np.arange(2)+1
    assert np.array_equal((np.kron(np.eye(4),np.kron(np.eye(3),np.diag(d)))@c.ravel()).reshape(c.shape),c*d)
    v = np.arange(24).reshape(4,3,2)+1j*np.arange(24,0,-1).reshape(4,3,2)
    v /= np.linalg.norm(v)
    p = abs(v)**2
    assert all(abs(p.sum(axis=axes).sum()-1)<1e-14 for axes in ((0,1),(0,2),(1,2)))
    assert abs(np.sum(abs(v/np.sqrt(.3*.4))**2)*.3*.4-1)<1e-14
    o = np.array([[1,2j],[-2j,3]])
    acted = v@o.T
    full = np.kron(np.eye(4),np.kron(np.eye(3),o))
    assert np.allclose(acted.ravel(),full@v.ravel())
    assert np.allclose(np.vdot(v,acted),v.ravel().conj()@full@v.ravel())
    a,b = 1+2j,3-4j
    n4=np.array([[0,-1j],[1j,0]]); t4=np.array([[0,1],[1,0]])
    block=a*n4+b*t4
    assert np.allclose(block.conj().T,a.conjugate()*n4+b.conjugate()*t4)
    assert not np.isclose(abs(a*n4[0,1]+b*t4[0,1])**2,abs(a*n4[0,1])**2+abs(b*t4[0,1])**2)
    w=np.stack([v.ravel(),v.ravel()[::-1]],axis=1)
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    assert np.allclose(np.sum(abs(w@u)**2,axis=1),np.sum(abs(w)**2,axis=1))
    return ['unequal-axis one-hot C indexing','normalized marginal sums','constant rectangle quadrature',
            'complex local versus Kronecker contraction','complex coherent bilinear interference/conjugation',
            'unitary-invariant ensemble density']


def run(out):
    started=time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180 second budget')))
    signal.alarm(180)
    as_limit='2 GiB'
    try:
        resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    except (ValueError,OSError) as exc:
        as_limit='UNAVAILABLE: '+str(exc)
    def monitor():
        while True:
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform!='darwin': peak*=1024
            if peak>(1<<30):
                print('STOP: 1 GiB resident-memory cap exceeded',file=sys.stderr,flush=True)
                os._exit(70)
            time.sleep(.05)
    threading.Thread(target=monitor,daemon=True).start()
    out.mkdir(parents=True,exist_ok=True)
    tests=synthetic_tests()
    meta,auth=authenticate()
    # Import definitions only. Gridium4Mode and all solve/assembly paths stay unused.
    sys.path.insert(0,str(ROOT))
    from Circuit_Objs.qchard_gridium_netlist import _coeffs, _grid_ops, _charge_ops
    params=meta['configuration']['physical_parameters']
    ec,k,_,_=_coeffs(**{key:params[key] for key in ('EJ','EC','EL','ELK','EJS','ECS','eps_J','eps_LK','eC','eP')})
    n,_,_,_=_charge_ops(4)
    x2,n2,_,_=_grid_ops(71,11)
    x3,n3,_,_=_grid_ops(71,14)
    n2,n3=n2.toarray(),n3.toarray()
    with np.load(CP/'stage1_eigensystem.npz',allow_pickle=False) as z:
        e,v,res=z['eigenvalues_GHz'],z['eigenvectors'],z['eigenpair_residuals']
    assert v.shape==(45369,380) and e.shape==(380,) and np.all(np.diff(e)>0)
    assert np.isfinite(v).all() and np.isfinite(e).all()
    assert np.array_equal(res,np.asarray(meta['stage1_solver_diagnostics']['eigenpair_residuals']))
    norms=np.einsum('ij,ij->j',v.conj(),v).real
    normerr=float(max(abs(norms-1)))
    orth=float(np.linalg.norm(v.conj().T@v-np.eye(380),ord=np.inf))
    assert normerr<1e-10 and orth<1e-8
    with np.load(CP/'projected_operators.npz',allow_pickle=False) as z:
        ops={name:z[name] for name in ('n1','n2','n3','x2','x3')}
    herm={name:float(np.max(abs(o-o.conj().T))) for name,o in ops.items()}
    assert all(o.shape==(380,380) for o in ops.values()) and max(herm.values())<1e-10
    # One-vector actions keep peak memory small; validate all 380 bra ranks.
    direct={name:0.0 for name in ops}
    vc=v.conj().T
    for r in SELECTED:
        c=v[:,r].reshape(71,71,9)
        for name in ops:
            col=vc@local_action(c,name,n,x2,x3,n2,n3).ravel()
            direct[name]=max(direct[name],float(np.max(abs(col-ops[name][:,r]))))
    del vc
    assert max(direct.values())<1e-10
    a=sum(8*ec[i,3]*ops['n'+str(i+1)] for i in range(3))
    b=k[0,2]*ops['x2']+k[1,2]*ops['x3']
    A4,B4=4*ec[3,3],.5*k[2,2]
    f4=(A4/(4*B4))**.25; g4=1/(2*f4)
    annih=np.diag(np.sqrt(np.arange(1,8)),1)
    th4=f4*(annih+annih.T); no4=1j*g4*(annih.T-annih)
    w4=2*np.sqrt(A4*B4)*(np.arange(8)+.5)
    # Coherent sum BEFORE magnitude; no incoherent operator replacement.
    h01=a*no4[0,1]+b*th4[0,1]
    h10=a*no4[1,0]+b*th4[1,0]
    strength=abs(a)**2*np.linalg.norm(no4)**2+abs(b)**2*np.linalg.norm(th4)**2+2*np.real(a.conj()*b*np.vdot(no4,th4))
    assert np.allclose(h10,h01.conj().T,atol=1e-12)
    cluster_ids=np.cumsum(np.r_[0,np.diff(e)>.001])
    rows=[]; marginalerr=0.; reshapeerr=0.
    for r in range(300,380):
        c=v[:,r].reshape(71,71,9)
        reshapeerr=max(reshapeerr,abs(float(np.vdot(c,c).real)-norms[r]))
        p=abs(c)**2/norms[r]
        m=metrics(p,n,x2,x3)
        marginalerr=max(marginalerr,*(abs(sum(m[name+'_marginal'])-1) for name in ('charge','theta2','theta3')))
        role='focal' if r in FOCAL else ('neighbor' if r in NEIGHBORS else ('deterministic_control' if r in CONTROLS else 'band_background'))
        row=dict(rank=r,role=role,energy_GHz=float(e[r]),norm_squared=float(norms[r]),saved_residual_GHz=float(res[r]),
                 gap_below_MHz=float((e[r]-e[r-1])*1000),gap_above_MHz=float((e[r+1]-e[r])*1000) if r<379 else None,
                 minimum_neighbor_gap_MHz=float(min(e[r]-e[r-1],e[r+1]-e[r] if r<379 else np.inf)*1000),
                 gap_residual_ratio_below=float((e[r]-e[r-1])/(res[r]+res[r-1])),
                 gap_residual_ratio_above=float((e[r+1]-e[r])/(res[r]+res[r+1])) if r<379 else None,
                 cluster_start=int(np.flatnonzero(cluster_ids==cluster_ids[r])[0]),
                 cluster_end=int(np.flatnonzero(cluster_ids==cluster_ids[r])[-1]))
        row['numerically_ambiguous']=min(row['gap_residual_ratio_below'],row['gap_residual_ratio_above'] or np.inf)<=10
        row.update(m)
        for prefix in (11,40,100,300,(r//20)*20):
            tag=f'p{prefix}'
            den=e[r]-e[:prefix]+w4[1]-w4[0]
            assert min(den)>0
            st=strength[:prefix,r]
            row[tag+'_block_strength_GHz2']=float(st.sum())
            row[tag+'_strongest_rank']=int(np.argmax(st))
            row[tag+'_strongest_block_norm_GHz']=float(np.sqrt(max(st)))
            row[tag+'_vacuum_proxy_GHz']=float(np.sum(abs(h01[:prefix,r])**2/den))
            row[tag+'_denominator_min_GHz']=float(min(den))
            row[tag+'_denominator_max_GHz']=float(max(den))
        # Consistent fixed schema, own-shell-prefix diagnostics stored separately.
        own=(r//20)*20
        row['own_shell_prefix']=own
        for suffix in ('block_strength_GHz2','strongest_rank','strongest_block_norm_GHz','vacuum_proxy_GHz','denominator_min_GHz','denominator_max_GHz'):
            row['own_shell_'+suffix]=row[f'p{own}_'+suffix]
            if own!=300: del row[f'p{own}_'+suffix]
        rows.append(row)
    assert marginalerr<1e-10 and reshapeerr<1e-10
    # Rotation-invariant density of selected complete finite subspaces.
    subspaces=[]
    for r in FOCAL:
        groups=[('fixed_window',list(range(r-2,r+3))),('close_cluster',np.flatnonzero(cluster_ids==cluster_ids[r]).tolist())]
        for kind,ranks in groups:
            p=sum(abs(v[:,s].reshape(71,71,9))**2 for s in ranks)/len(ranks)
            assert abs(p.sum()-1)<1e-10
            p=p/p.sum()
            subspaces.append(dict(focal=r,kind=kind,ranks=ranks,
                                  external_gap_below_MHz=float((e[ranks[0]]-e[ranks[0]-1])*1000) if ranks[0]>0 else None,
                                  external_gap_above_MHz=float((e[ranks[-1]+1]-e[ranks[-1]])*1000) if ranks[-1]<379 else None,
                                  **metrics(p,n,x2,x3)))
    del v
    # Detailed complex elements: fixed p<40 for every band state.
    complexrows=[]
    for r in range(300,380):
        for p in range(40):
            row=dict(low_rank=p,high_rank=r,stage1_gap_GHz=float(e[r]-e[p]),
                     vacuum_denominator_GHz=float(e[r]-e[p]+w4[1]-w4[0]))
            for name,mat in [*ops.items(),('A_GHz',a),('B_GHz_per_rad',b),('H_p0_r1_GHz',h01),('H_p1_r0_GHz',h10)]:
                z=mat[p,r]
                row[name+'_real']=float(z.real); row[name+'_imag']=float(z.imag); row[name+'_abs']=float(abs(z))
            block=a[p,r]*no4+b[p,r]*th4
            row['block_frobenius_GHz']=float(np.linalg.norm(block))
            assert np.isclose(np.linalg.norm(block)**2,strength[p,r],atol=1e-10)
            complexrows.append(row)
    # Bring in saved exact scalar evidence; no dressed-vector reconstruction.
    d=read(PRE/'k380_metrics.json'); historical=read(REV/'DIAGNOSTICS.json')
    for record in historical['rank_results']:
        assert abs(e[record['rank']]-record['k380_stage1_energy_GHz'])<1e-12
    schur=[]
    for shell in ('S1','S2'):
        sh=d['shell_diagnostics'][shell]
        # Saved rank_range endpoints are INCLUSIVE, unlike Python range.
        lo,hi=sh['rank_range']
        assert (lo,hi)==((340,359) if shell=='S1' else (360,379))
        assert len(sh['states']['0']['ranks'])==hi-lo+1==20
        for r in range(lo,hi+1):
            entries={s:next(v for v in sh['states'][str(s)]['ranks'] if v['rank']==r) for s in (0,1,2,3,7,8)}
            weight=sum(abs(v['exact_self_energy_MHz']) for v in entries.values())
            row=rows[r-300]
            row['schur_shell']=shell; row['schur_absolute_weight_MHz']=weight
            row['schur_fraction']=weight/sh['aggregate_rank_participation']['absolute_total_MHz']
            schur.append(dict(rank=r,shell=shell,absolute_weight_MHz=weight,fraction=row['schur_fraction'],targets=entries))
        assert abs(sum(s['absolute_weight_MHz'] for s in schur if s['shell']==shell)-sh['aggregate_rank_participation']['absolute_total_MHz'])<1e-10
    for row in rows:
        row.setdefault('schur_shell','');row.setdefault('schur_absolute_weight_MHz',None);row.setdefault('schur_fraction',None)
    corr=[]
    fixed=['charge_edge','theta2_edge2','theta3_edge2','joint_pr','theta2_std','theta3_std','p300_block_strength_GHz2','p300_vacuum_proxy_GHz']
    for shell in ('S1','S2','union'):
        subset=[r for r in rows if r['schur_shell'] and (shell=='union' or r['schur_shell']==shell)]
        for name in fixed:
            rho=spearmanr([r[name] for r in subset],[r['schur_absolute_weight_MHz'] for r in subset]).statistic
            corr.append(dict(shell=shell,metric=name,n=len(subset),spearman_rho=float(rho)))
    # Enforce ranks by count of strictly larger values; ties share rank.
    for row in rows:
        for name in fixed:
            row[name+'_band_descending_rank']=1+sum(o[name]>row[name] for o in rows)
    serial=[]
    for row in rows:
        serial.append({key:json.dumps(value,separators=(',',':')) if isinstance(value,list) else value for key,value in row.items()})
    csvwrite(out/'FINGERPRINTS.csv',serial)
    csvwrite(out/'OPERATOR_COUPLINGS.csv',complexrows)
    # Audit storage of the very same computed blocks, including strongest
    # partners outside the frozen detailed p<40 CSV. No new metric or selection.
    np.savez(out/'OPERATOR_BLOCKS.npz',low_ranks=np.arange(300),high_ranks=np.arange(300,380),
             A_GHz=a[:300,300:],B_GHz_per_rad=b[:300,300:],
             H_p0_r1_GHz=h01[:300,300:],H_p1_r0_GHz=h10[:300,300:])
    write(out/'DIAGNOSTICS.json',dict(plan_sha256=sha(HERE/'RESEARCH_PLAN.md'),states=rows,subspaces=subspaces,
          schur=schur,correlations=corr,historical_rank328=historical['rank_results'][0],
          coefficients=dict(ECmat=ec.tolist(),K=k.tolist(),oscillator_spacing_GHz=float(w4[1]-w4[0]),f4=float(f4),g4=float(g4))))
    write(out/'SOURCE_AUTHENTICATION.json',auth)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin': peak*=1024
    assert peak <= (1<<30), peak
    validation=dict(status='PASS',synthetic_tests=tests,plan_sha256=sha(HERE/'RESEARCH_PLAN.md'),
                    maximum_norm_error=normerr,orthogonality_inf=orth,maximum_marginal_sum_error=marginalerr,
                    maximum_reshape_norm_error=reshapeerr,operator_hermiticity_max_errors=herm,
                    operator_projection_max_errors=direct,operator_columns_checked=list(SELECTED),
                    saved_maximum_eigenpair_residual_GHz=float(max(res)),residuals_recomputed=False,
                    state_rows=len(rows),complex_coupling_rows=len(complexrows),
                    elapsed_seconds=time.monotonic()-started,peak_resident_bytes=peak,new_eigensolves=0)
    validation['address_space_limit']=as_limit
    validation['resident_watchdog_interval_seconds']=.05
    write(out/'RUN_VALIDATION.json',validation)
    signal.alarm(0)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    run(args.output)
