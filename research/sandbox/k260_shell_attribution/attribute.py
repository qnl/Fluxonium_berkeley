"""Frozen finite-Stage-2 diagnostic; never builds or solves Stage 1."""
from pathlib import Path
import sys, json, hashlib, subprocess, csv
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner

CP=ROOT/'research/checkpoints/four-mode-asymmetric-n71-k260'
STATES=[0,1,2,3,7,8]
KS=[180,200,220,240,260]
FREQ={'f01':(0,1),'f07':(0,7),'f08':(0,8),'f78':(7,8),'f81':(1,8),
      'f72':(2,7),'f21':(1,2),'f03':(0,3),'f38':(3,8)}
METHOD='921f264e6e62c6233f1906592f282f28ab1d389aa055417c47e05ac62303ec80'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def snapshot():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    return {n:sha(ROOT/n) for n in names if n and (ROOT/n).is_file()}

def serial(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v))

def solve(h,indices,ref=None):
    e,v=eigh(h,subset_by_index=(0,10),driver='evr')
    embedded=np.zeros((2080,11),complex); embedded[indices]=v
    if ref is not None:
        ov=embedded.conj().T@ref
        rows,cols=linear_sum_assignment(-abs(ov))
        assignment=np.empty(11,int); assignment[cols]=rows
        conf=runner.assignment_confidence_diagnostics(ov,assignment)
        e=e[assignment]; v=v[:,assignment]; embedded=embedded[:,assignment]
        diag=np.diag(embedded.conj().T@ref)
        phase=np.exp(1j*np.angle(diag))
        v=v*phase; embedded=embedded*phase
    else:
        conf=[]
    residual=float(np.max(np.linalg.norm(h@v-v*e,axis=0)))
    assert residual<1e-9, residual
    return dict(e=e,v=v,embedded=embedded,confidence=conf,residual=residual)

def participation(terms):
    a=abs(terms); total=float(a.sum())
    if total==0: return dict(top3_fraction=0.,effective_count=0.,count90=0)
    w=a/total; order=np.argsort(-w)
    return dict(top3_fraction=float(w[order[:3]].sum()),effective_count=float(1/(w@w)),
                count90=int(np.searchsorted(np.cumsum(w[order]),.9)+1))

def attribute(h,cases,a,b):
    na=a*8; nb=b*8
    hp=h[:na,:na]; hq=h[na:nb,na:nb]; t=h[na:nb,:na]
    lam,u=eigh(hq)
    out={}; rankrows=[]; moderows=[]
    for s in STATES:
        old=cases[a]['v'][:,s]; v=cases[b]['v'][:,s]
        p=v[:na]; q=v[na:nb]; pn=float(np.vdot(p,p).real)
        E=float(cases[b]['e'][s]); E0=float(cases[a]['e'][s]); delta=E-E0
        cp=u.conj().T@(t@p); c0=u.conj().T@(t@old)
        den=E-lam; den0=E0-lam
        assert min(abs(den))>1e-12, 'Unresolved Schur pole'
        qr=u@(cp/den)
        exactterms=abs(cp)**2/den/pn
        ptterms=abs(c0)**2/den0
        eta=abs(c0)/abs(den0)
        near=(abs(den0)<=.010)|(eta>=.10)
        pt=float(ptterms.sum()); sigma=float(exactterms.sum())
        relax=float(np.vdot(p,hp@p-E0*p).real/pn)
        qerr=float(np.linalg.norm(q-qr))
        hefferr=float(np.linalg.norm(hp@p+t.conj().T@qr-E*p))
        identity_error=float(delta-sigma-relax)
        assert qerr<=1e-8 and hefferr<=1e-9 and abs(identity_error)<=1e-9
        rankterms=np.real((t@p).conj()*q).reshape(b-a,8).sum(axis=1)/pn
        assert abs(rankterms.sum()-sigma)<=1e-9
        coupling_rank=np.linalg.norm((t@old).reshape(b-a,8),axis=1)
        qw=float(np.vdot(q,q).real)
        fractions=participation(ptterms)
        nearfrac=float(abs(ptterms[near]).sum()/abs(ptterms).sum())
        controlled=abs(pt-delta)<=max(.2*abs(delta),1e-6) and max(eta)<.1 and qw<.01
        safe=all(c['state_specific_claim_safe'] for c in cases[b]['confidence'] if c['reference_state']==s)
        if not safe: cl='D'
        elif nearfrac>=.5 and fractions['top3_fraction']>=.7: cl='A'
        elif nearfrac<.1 and fractions['top3_fraction']<=.5 and fractions['effective_count']>=8 and controlled: cl='B'
        else: cl='C'
        top=np.argsort(-abs(ptterms))[:12]
        details=[]
        for j in top:
            weights=abs(u[:,j].reshape(b-a,8))**2
            rw=weights.sum(axis=1); dom=np.argsort(-rw)[:4]
            details.append(dict(mode=int(j),epsilon_GHz=float(lam[j]),coupling_GHz=float(abs(c0[j])),
                denominator_GHz=float(den0[j]),eta=float(eta[j]),pt_MHz=float(ptterms[j]*1000),
                fraction=float(abs(ptterms[j])/abs(ptterms).sum()),
                dominant_ranks=[dict(rank=int(a+r),weight=float(rw[r])) for r in dom]))
        out[str(s)]=dict(observed_MHz=1000*delta,second_order_MHz=1000*pt,
            second_order_error_MHz=1000*(pt-delta),exact_self_energy_MHz=1000*sigma,
            relaxation_MHz=1000*relax,exact_reconstructed_MHz=1000*(sigma+relax),
            reconstruction_error_MHz=1000*identity_error,schur_residual_GHz=hefferr,q_reconstruction_error=qerr,
            shell_probability=qw,min_abs_denominator_GHz=float(min(abs(den0))),
            maximum_eta=float(max(eta)),near_resonant_fraction=nearfrac,
            second_order_controlled=bool(controlled),modal_participation=fractions,
            exact_rank_participation=participation(rankterms),mechanism=cl,top_modes=details,
            top_ranks=[dict(rank=int(a+r),exact_self_energy_MHz=float(rankterms[r]*1000),
                coupling_norm_GHz=float(coupling_rank[r]),Q_weight=float(np.sum(abs(q.reshape(b-a,8)[r])**2)))
                for r in np.argsort(-abs(rankterms))[:10]])
        for r in range(b-a):
            rankrows.append(dict(step=f'{a}->{b}',state=s,rank=a+r,self_energy_MHz=float(rankterms[r]*1000),
                coupling_norm_GHz=float(coupling_rank[r]),Q_weight=float(np.sum(abs(q.reshape(b-a,8)[r])**2))))
        for j in range(len(lam)):
            moderows.append(dict(step=f'{a}->{b}',state=s,mode=j,epsilon_GHz=float(lam[j]),
                denominator_GHz=float(den0[j]),coupling_GHz=float(abs(c0[j])),eta=float(eta[j]),
                near_resonant=bool(near[j]),second_order_MHz=float(ptterms[j]*1000),exact_self_energy_MHz=float(exactterms[j]*1000)))
    return out,rankrows,moderows

def main():
    assert sha(HERE/'methodology.md')==METHOD
    baseline=snapshot()
    checkpoints={str(p.relative_to(ROOT)):sha(p) for root in [CP,
        ROOT/'research/checkpoints/four-mode-asymmetric-n71-k180',
        ROOT/'research/checkpoints/four-mode-asymmetric-n71-k220',
        ROOT/'research/checkpoints/2026-09-26-asymmetric-n51'] for p in root.rglob('*') if p.is_file()}
    meta=json.loads((CP/'checkpoint.json').read_text())
    assert meta['configuration_sha256']=='990dad04b52deeef9d90908c6c19ce38e917d7f5b7fab32f0e39699c3c83c4bb'
    assert sha(ROOT/'research/sandbox/k260_preregistration/criteria.json')=='8f42b3c10453a048238dea23d28e4a590accacbd87ad5fe54439cca1af799be9'
    for name,digest in meta['file_sha256'].items(): assert sha(CP/name)==digest
    saved=json.loads((CP/'analysis.json').read_text())['retained_basis_cases']
    ret=json.loads((ROOT/'research/sandbox/k260_preregistration/k260_return_analysis/k260_analysis_summary.json').read_text())
    with np.load(CP/'stage1_eigensystem.npz') as z:
        w=z['eigenvalues_GHz']; vecshape=z['eigenvectors'].shape
    with np.load(CP/'projected_operators.npz') as z: ops={n:z[n] for n in z.files}
    p=meta['configuration']['physical_parameters']
    ec,k,_,_=_coeffs(p['EJ'],p['EC'],p['EL'],p['ELK'],p['EJS'],p['ECS'],p['eps_J'],p['eps_LK'],p['eC'],p['eP'])
    A,B=4*ec[3,3],.5*k[2,2]; f=(A/(4*B))**.25; g=1/(2*f)
    a=np.diag(np.sqrt(np.arange(1,8)),1); x=f*(a+a.T); momentum=1j*g*(a.T-a)
    w4=2*np.sqrt(A*B)*(np.arange(8)+.5)
    h=np.kron(np.diag(w),np.eye(8))+np.kron(np.eye(260),np.diag(w4))
    h=h.astype(complex)
    for n in range(3): h+=8*ec[n,3]*np.kron(ops[f'n{n+1}'],momentum)
    h+=k[0,2]*np.kron(ops['x2'],x)+k[1,2]*np.kron(ops['x3'],x)
    herm=float(np.max(abs(h-h.conj().T))); assert herm<1e-12
    h=(h+h.conj().T)/2
    op=np.kron(.5*ops['x2']+ops['x3'],np.eye(8))
    print('Reconstructing finite Stage-2 states only',flush=True)
    cases={260:solve(h,np.arange(2080))}
    ref=cases[260]['embedded']
    for n in KS[:-1]: cases[n]=solve(h[:n*8,:n*8],np.arange(n*8),ref)
    checks={}; selectivity=[]; collisions={}
    for n,c in cases.items():
        c['matrix']=c['embedded'].conj().T@op@c['embedded']
        old=saved[f'nested_prefix_k{n}']
        ee=float(max(abs(c['e']-old['energies_GHz'])))
        oe=float(np.max(abs(abs(c['matrix'])-old['grid_phi_abs'])))
        assert ee<1e-10 and oe<1e-10,(n,ee,oe)
        checks[str(n)]=dict(energy_error_GHz=ee,operator_error=oe,eigenpair_residual_GHz=c['residual'],confidence=c['confidence'])
        freqs={name:float(abs(c['e'][j]-c['e'][i])*1000) for name,(i,j) in FREQ.items()}
        selectivity.append(dict(k=n,f01_MHz=freqs['f01'],f78_MHz=freqs['f78'],
            difference_MHz=freqs['f78']-freqs['f01'],g01=float(abs(c['matrix'][0,1])),g78=float(abs(c['matrix'][7,8])),
            outer_difference_MHz=freqs['f07']-freqs['f81']))
        near={}
        for tone in ['f07','f81']:
            wanted=tuple(sorted(FREQ[tone])); cand=[]
            for i in [0,1]:
                for j in range(i+1,11):
                    if (i,j)==wanted: continue
                    freq=abs(c['e'][j]-c['e'][i])*1000
                    cand.append(dict(pair=[i,j],frequency_MHz=float(freq),detuning_MHz=float(freq-freqs[tone]),
                        grid_phi=float(abs(c['matrix'][i,j]))))
            near[tone]=sorted(cand,key=lambda r:abs(r['detuning_MHz']))[:5]
        collisions[str(n)]=near
    print('Reconstruction validated; attributing adjacent shells',flush=True)
    shifts={}; ranks=[]; modes=[]; transitions={}; opcor={}
    for lo,hi in zip(KS[:-1],KS[1:]):
        label=f'{lo}->{hi}'
        states,rr,mm=attribute(h,cases,lo,hi); shifts[label]=states; ranks+=rr; modes+=mm
        transitions[label]={}
        for name,(i,j) in FREQ.items():
            sign=np.sign(cases[lo]['e'][j]-cases[lo]['e'][i])
            obs=(abs(cases[hi]['e'][j]-cases[hi]['e'][i])-abs(cases[lo]['e'][j]-cases[lo]['e'][i]))*1000
            pt=sign*(states[str(j)]['second_order_MHz']-states[str(i)]['second_order_MHz'])
            exact=sign*(states[str(j)]['exact_reconstructed_MHz']-states[str(i)]['exact_reconstructed_MHz'])
            transitions[label][name]=dict(observed_MHz=float(obs),second_order_MHz=float(pt),
                second_order_error_MHz=float(pt-obs),exact_MHz=float(exact),reconstruction_error_MHz=float(exact-obs))
        # Align new vector phases to old for a meaningful PP/direct split.
        v=cases[hi]['v'].copy(); old=cases[lo]['v']; dim=lo*8
        phase=np.exp(-1j*np.angle(np.diag(old.conj().T@v[:dim])))
        v*=phase
        pp=v[:dim]; qq=v[dim:]; o=op[:hi*8,:hi*8]
        oldm=old.conj().T@o[:dim,:dim]@old
        relax=pp.conj().T@o[:dim,:dim]@pp-oldm
        direct=pp.conj().T@o[:dim,dim:]@qq+qq.conj().T@o[dim:,:dim]@pp+qq.conj().T@o[dim:,dim:]@qq
        change=v.conj().T@o@v-oldm
        assert np.max(abs(change-relax-direct))<1e-10
        opcor[label]={f'g{i}{j}':dict(total_real=float(change[i,j].real),total_imag=float(change[i,j].imag),
            PP_relaxation_real=float(relax[i,j].real),PP_relaxation_imag=float(relax[i,j].imag),
            direct_shell_real=float(direct[i,j].real),direct_shell_imag=float(direct[i,j].imag),
            direct_shell_magnitude=float(abs(direct[i,j])),absolute_magnitude_change=float(abs((oldm+change)[i,j])-abs(oldm[i,j])))
            for i,j in [(0,7),(7,8),(8,1),(0,1)]}
        print(label,'validated',flush=True)
    print('Resolving Q1-Q2 nonadditivity by finite-block counterfactuals',flush=True)
    combined,combined_ranks,combined_modes=attribute(h,cases,220,260)
    ranks+=combined_ranks; modes+=combined_modes
    ids=np.r_[np.arange(1760),np.arange(1920,2080)]
    pq2=solve(h[np.ix_(ids,ids)],ids,ref)
    no=h.copy(); no[1760:1920,1920:]=0; no[1920:,1760:1920]=0
    nocross=solve(no,np.arange(2080),ref)
    cross={}
    for s in STATES:
        base=cases[220]['e'][s]; e1=cases[240]['e'][s]; e2=pq2['e'][s]; full=cases[260]['e'][s]; en=nocross['e'][s]
        cross[str(s)]=dict(Q1_standalone_MHz=float((e1-base)*1000),Q2_standalone_MHz=float((e2-base)*1000),
            full_MHz=float((full-base)*1000),nonadditivity_MHz=float((full-e1-e2+base)*1000),
            explicit_Q1Q2_effect_MHz=float((full-en)*1000),P_mediated_nonadditivity_MHz=float((en-e1-e2+base)*1000))
    central=[shifts['220->240'][str(s)]['mechanism'] for s in [7,8]]
    if 'D' in central: overall='D'
    elif central==['A','A']: overall='A'
    elif central==['B','B'] and all(shifts['220->240'][str(s)]['near_resonant_fraction']<.5 for s in [0,1]): overall='B'
    else: overall='C'
    assert snapshot()==baseline,'Tracked file mutation'
    assert all(sha(ROOT/n)==v for n,v in checkpoints.items()),'Checkpoint mutation'
    result=dict(methodology_sha256=METHOD,configuration_sha256=meta['configuration_sha256'],
        checkpoint_sha256=sha(CP/'checkpoint.json'),data_feasibility='SUFFICIENT',
        feasibility=dict(stage1_vector_shape=vecshape,projected_operators={n:list(a.shape) for n,a in ops.items()},
            Stage2_saved=False,Stage2_reconstructible=True,dimensions=dict(P=1760,Q1=160,Q2=160,full=2080)),
        effective_hamiltonian='VALIDATED',hermiticity_error=herm,reconstruction_checks=checks,
        state_attributions=shifts,combined_Q1_Q2_attribution=combined,transition_attributions=transitions,cross_shell=cross,
        Q1Q2_coupling_frobenius_GHz=float(np.linalg.norm(h[1760:1920,1920:])),
        counterfactual_tracking=dict(P_plus_Q2=pq2['confidence'],no_Q1Q2=nocross['confidence']),
        selectivity=sorted(selectivity,key=lambda r:r['k']),near_collisions=collisions,operator_corrections=opcor,
        operator_blocks=ret['operator_blocks'],subspace_diagnostics=ret['subspace_diagnostics'],
        state_overlaps_vs_k260=ret['state_overlaps_vs_k260'],pathway_series=ret['values_and_trends']['pathway_scores'],
        path_ordering=ret['path_ordering'],frozen_endpoint_comparisons=ret['endpoint_comparisons'],
        dominant_mechanism=overall,tracked_files_unchanged=True,checkpoints_unchanged=True,
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT).decode())
    (HERE/'summary.json').write_text(json.dumps(result,indent=2,default=serial,allow_nan=False)+'\n')
    tables=[]
    for step,rs in shifts.items():
        for s,r in rs.items():
            tables.append(dict(step=step,kind='state',quantity=s,observed_MHz=r['observed_MHz'],
                predicted_second_order_MHz=r['second_order_MHz'],second_order_error_MHz=r['second_order_error_MHz'],
                exact_MHz=r['exact_reconstructed_MHz'],reconstruction_error_MHz=r['reconstruction_error_MHz']))
    for step,rs in transitions.items():
        for n,r in rs.items():
            tables.append(dict(step=step,kind='transition',quantity=n,observed_MHz=r['observed_MHz'],
                predicted_second_order_MHz=r['second_order_MHz'],second_order_error_MHz=r['second_order_error_MHz'],
                exact_MHz=r['exact_MHz'],reconstruction_error_MHz=r['reconstruction_error_MHz']))
    for name,rows in [('attribution_tables.csv',tables),('rank_attributions.csv',ranks),('shell_modes.csv',modes)]:
        with (HERE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (HERE/'preservation.json').write_text(json.dumps(dict(tracked=baseline,checkpoints=checkpoints),indent=2)+'\n')
    print('Completed; dominant mechanism',overall,flush=True)

if __name__=='__main__': main()
