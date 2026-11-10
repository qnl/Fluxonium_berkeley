"""Post-result finite-Stage-2 attribution; never runs a Stage-1 eigensolve."""
from pathlib import Path
from itertools import combinations
import csv, hashlib, json, subprocess, sys

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from Circuit_Objs.qchard_gridium_netlist import _coeffs
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner

CP = ROOT / 'research/checkpoints/four-mode-asymmetric-n71-k300'
METHOD = 'b5e65484dfe546755199a4606d4e27aa1145be36e1aca7d4eae413ccd69d6dbf'
CONFIG = 'b7596cd4a31727912f570fdd6be6fcd02ec8d97cf067c8e6df13b99fa984760b'
CHECKPOINT = 'a48ab94ed6b958e7e4422825ca1670784f8ed797f72cf011db32824b35f1145f'
SELECTION = 'ef27780c0c8b684f50b611774d2b5b7c61adaa5e076d9c7bc86d015af59a024e'
SEAL = 'b82151d79efab2b77f713b0908ac4c57ef5319a479102895adf1c5ec7f047bb9'
STATES = [0, 1, 2, 3, 7, 8]
FREQ = {'f01': (0,1), 'f07': (0,7), 'f08': (0,8), 'f78': (7,8),
        'f81': (1,8), 'f72': (2,7), 'f21': (1,2), 'f03': (0,3), 'f38': (3,8)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def tracked_snapshot():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    return {n: sha(ROOT / n) for n in names if n and (ROOT / n).is_file()}


def tree_snapshot(root):
    return {str(p.relative_to(ROOT)): sha(p) for p in root.rglob('*') if p.is_file()}


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x))


def solve(mat, global_indices, reference=None):
    e, v = eigh(mat, subset_by_index=(0, 10), driver='evr')
    embedded = np.zeros((2400, 11), complex)
    embedded[np.asarray(global_indices)] = v
    confidence = []
    if reference is not None:
        overlap = embedded.conj().T @ reference
        rows, cols = linear_sum_assignment(-abs(overlap))
        assignment = np.empty(11, int); assignment[cols] = rows
        confidence = runner.assignment_confidence_diagnostics(overlap, assignment)
        e, v, embedded = e[assignment], v[:, assignment], embedded[:, assignment]
        phase = np.exp(1j * np.angle(np.diag(embedded.conj().T @ reference)))
        v, embedded = v * phase, embedded * phase
    residual = float(np.max(np.linalg.norm(mat @ v - v * e, axis=0)))
    if residual > 1e-9: raise RuntimeError(f'Stage-2 residual {residual}')
    return {'e': e, 'v': v, 'embedded': embedded, 'confidence': confidence,
            'residual': residual, 'indices': np.asarray(global_indices)}


def count_fraction(values, threshold):
    a = abs(np.asarray(values, float)); total = float(a.sum())
    if total == 0: return 0
    return int(np.searchsorted(np.cumsum(np.sort(a)[::-1]) / total, threshold) + 1)


def participation(values):
    a = abs(np.asarray(values, float)); total = float(a.sum())
    if total == 0:
        return {'top3_fraction': 0., 'effective_count': 0., 'count50': 0,
                'count80': 0, 'count90': 0, 'absolute_total_MHz': 0.}
    w = a / total
    return {'top3_fraction': float(np.sort(w)[::-1][:3].sum()),
            'effective_count': float(1 / (w @ w)),
            'count50': count_fraction(values, .5), 'count80': count_fraction(values, .8),
            'count90': count_fraction(values, .9), 'absolute_total_MHz': total * 1000}


def mechanism(nearfrac, modal, rank, safe):
    if not safe: return 'E'
    if nearfrac >= .5 and (modal['top3_fraction'] >= .7 or modal['count80'] <= 5): return 'A'
    if nearfrac < .1 and rank['top3_fraction'] >= .7 and rank['count80'] <= 5: return 'B'
    if (nearfrac < .1 and rank['top3_fraction'] <= .5 and modal['top3_fraction'] <= .5
            and rank['count80'] >= 8 and modal['count80'] >= 8): return 'C'
    return 'D'


def attribute(local_h, p_dim, rank_labels, base, endpoint, label):
    hp, hq = local_h[:p_dim, :p_dim], local_h[p_dim:, p_dim:]
    t = local_h[p_dim:, :p_dim]
    lam, u = eigh(hq)
    results, rank_rows, mode_rows = {}, [], []
    for s in STATES:
        old, v = base['v'][:, s], endpoint['v'][:, s]
        p, q = v[:p_dim], v[p_dim:]
        pn = float(np.vdot(p, p).real)
        E0, E = float(base['e'][s]), float(endpoint['e'][s])
        c0, cp = u.conj().T @ (t @ old), u.conj().T @ (t @ p)
        den0, den = E0 - lam, E - lam
        if np.min(abs(den)) <= 1e-12: raise RuntimeError('Unresolved Schur pole')
        qr = u @ (cp / den)
        pt_terms = abs(c0) ** 2 / den0
        exact_mode_terms = abs(cp) ** 2 / den / pn
        eta = abs(c0) / abs(den0)
        near = (abs(den0) <= .010) | (eta >= .10)
        sigma = float(exact_mode_terms.sum())
        relax = float(np.vdot(p, hp @ p - E0 * p).real / pn)
        delta = E - E0
        qerr = float(np.linalg.norm(q - qr))
        schur = float(np.linalg.norm(hp @ p + t.conj().T @ qr - E * p))
        identity = float(delta - sigma - relax)
        if qerr > 1e-8 or schur > 1e-9 or abs(identity) > 1e-9:
            raise RuntimeError((label, s, qerr, schur, identity))
        rank_terms = np.real((t @ p).conj() * q).reshape(len(rank_labels), 8).sum(1) / pn
        if abs(rank_terms.sum() - sigma) > 1e-9: raise RuntimeError('rank sum failure')
        coupling_rank = np.linalg.norm((t @ old).reshape(len(rank_labels), 8), axis=1)
        qw_rank = (abs(q.reshape(len(rank_labels), 8)) ** 2).sum(1)
        modal_p, rank_p = participation(pt_terms), participation(rank_terms)
        nearfrac = float(abs(pt_terms[near]).sum() / abs(pt_terms).sum())
        qw = float(np.vdot(q, q).real)
        pt = float(pt_terms.sum())
        controlled = (abs(pt - delta) <= max(.2 * abs(delta), 1e-6)
                      and float(max(eta)) < .1 and qw < .01)
        safe = all(c['state_specific_claim_safe'] for c in endpoint['confidence']
                   if c['reference_state'] == s) if endpoint['confidence'] else True
        cl = mechanism(nearfrac, modal_p, rank_p, safe)
        top_modes = []
        for j in np.argsort(-abs(exact_mode_terms))[:12]:
            rw = (abs(u[:, j].reshape(len(rank_labels), 8)) ** 2).sum(1)
            dom = np.argsort(-rw)[:4]
            top_modes.append({'mode': int(j), 'epsilon_GHz': float(lam[j]),
                'coupling_GHz': float(abs(c0[j])), 'denominator_GHz': float(den0[j]),
                'eta': float(eta[j]), 'near_resonant': bool(near[j]),
                'second_order_MHz': float(pt_terms[j] * 1000),
                'exact_self_energy_MHz': float(exact_mode_terms[j] * 1000),
                'dominant_ranks': [{'rank': int(rank_labels[r]), 'weight': float(rw[r])}
                                   for r in dom]})
        top_ranks = [{'rank': int(rank_labels[r]),
                      'exact_self_energy_MHz': float(rank_terms[r] * 1000),
                      'coupling_norm_GHz': float(coupling_rank[r]),
                      'Q_probability': float(qw_rank[r])}
                     for r in np.argsort(-abs(rank_terms))[:10]]
        results[str(s)] = {
            'observed_MHz': float(delta * 1000), 'exact_self_energy_MHz': sigma * 1000,
            'P_relaxation_MHz': relax * 1000, 'exact_identity_MHz': (sigma + relax) * 1000,
            'second_order_MHz': pt * 1000, 'second_order_error_MHz': (pt - delta) * 1000,
            'second_order_relative_error': float(abs(pt-delta)/max(abs(delta), 1e-6)),
            'second_order_controlled': bool(controlled), 'shell_probability': qw,
            'minimum_abs_denominator_GHz': float(min(abs(den0))), 'maximum_eta': float(max(eta)),
            'near_resonant_fraction': nearfrac, 'modal_participation': modal_p,
            'rank_participation': rank_p, 'mechanism': cl, 'q_error': qerr,
            'schur_residual_GHz': schur, 'identity_error_GHz': identity,
            'top_ranks': top_ranks, 'top_modes': top_modes}
        for r, rank in enumerate(rank_labels):
            rank_rows.append({'shell': label, 'state': s, 'rank': int(rank),
                'exact_self_energy_MHz': float(rank_terms[r] * 1000),
                'coupling_norm_GHz': float(coupling_rank[r]),
                'Q_probability': float(qw_rank[r])})
        for j in range(len(lam)):
            mode_rows.append({'shell': label, 'state': s, 'mode': j,
                'epsilon_GHz': float(lam[j]), 'denominator_GHz': float(den0[j]),
                'coupling_GHz': float(abs(c0[j])), 'eta': float(eta[j]),
                'near_resonant': bool(near[j]), 'second_order_MHz': float(pt_terms[j]*1000),
                'exact_self_energy_MHz': float(exact_mode_terms[j]*1000)})
    return results, rank_rows, mode_rows


def frequency(case, pair):
    i, j = pair
    return float(abs(case['e'][j] - case['e'][i]) * 1000)


def transition_attribution(states, base, endpoint):
    out = {}
    for name, (i, j) in FREQ.items():
        observed = frequency(endpoint, (i,j)) - frequency(base, (i,j))
        sign = float(np.sign(base['e'][j] - base['e'][i]))
        def diff(key): return sign * (states[str(j)][key] - states[str(i)][key])
        out[name] = {'observed_MHz': observed,
            'exact_self_energy_MHz': diff('exact_self_energy_MHz'),
            'P_relaxation_MHz': diff('P_relaxation_MHz'),
            'exact_identity_MHz': diff('exact_identity_MHz'),
            'second_order_MHz': diff('second_order_MHz'),
            'second_order_error_MHz': diff('second_order_MHz') - observed}
    return out


def energy_nonadditivity(base, q1, q2, full, nocross):
    out = {'states': {}, 'transitions': {}}
    for s in STATES:
        e0, e1, e2, ef, en = (x['e'][s] for x in (base, q1, q2, full, nocross))
        out['states'][str(s)] = {'Q1_alone_MHz': float((e1-e0)*1000),
            'Q2_alone_MHz': float((e2-e0)*1000), 'full_MHz': float((ef-e0)*1000),
            'total_nonadditivity_MHz': float((ef-e1-e2+e0)*1000),
            'explicit_Q1Q2_MHz': float((ef-en)*1000),
            'P_mediated_remainder_MHz': float((en-e1-e2+e0)*1000)}
    for name, pair in FREQ.items():
        f0, f1, f2, ff, fn = (frequency(x, pair) for x in (base,q1,q2,full,nocross))
        out['transitions'][name] = {'Q1_alone_MHz': f1-f0, 'Q2_alone_MHz': f2-f0,
            'full_MHz': ff-f0, 'total_nonadditivity_MHz': ff-f1-f2+f0,
            'explicit_Q1Q2_MHz': ff-fn, 'P_mediated_remainder_MHz': fn-f1-f2+f0}
    return out


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    if sha(HERE/'methodology.md') != METHOD: raise RuntimeError('methodology hash changed')
    tracked_before = tracked_snapshot()
    checkpoint_roots = [CP, ROOT/'research/checkpoints/four-mode-asymmetric-n71-k260',
        ROOT/'research/checkpoints/four-mode-asymmetric-n71-k220',
        ROOT/'research/checkpoints/four-mode-asymmetric-n71-k180',
        ROOT/'research/checkpoints/2026-09-26-asymmetric-n51']
    checkpoints_before = {}
    for root in checkpoint_roots: checkpoints_before.update(tree_snapshot(root))

    # Fresh authentication of immutable saved artifacts.
    auth0 = json.loads((ROOT/'research/sandbox/k300_results/authentication.json').read_text())
    meta = json.loads((CP/'checkpoint.json').read_text())
    complete = json.loads((CP/'COMPLETE').read_text())
    analysis = json.loads((CP/'analysis.json').read_text())
    if auth0['status'] != 'PASS' or not auth0['payload_hashes_verified']: raise RuntimeError('prior auth failed')
    if sha(CP/'checkpoint.json') != CHECKPOINT or meta['configuration_sha256'] != CONFIG: raise RuntimeError('checkpoint auth')
    if complete['checkpoint_sha256'] != CHECKPOINT or complete['configuration_sha256'] != CONFIG: raise RuntimeError('COMPLETE auth')
    for name, digest in meta['file_sha256'].items():
        if sha(CP/name) != digest: raise RuntimeError(f'payload hash {name}')
    sel_path = ROOT/'research/sandbox/k300_preregistration/selection.json'
    seal_path = ROOT/'research/sandbox/k300_preregistration/selection.json.seal.json'
    if sha(sel_path) != SELECTION or sha(seal_path) != SEAL: raise RuntimeError('selection hashes')
    freeze = analysis['held_out_selection_freeze']
    if freeze['selection_sha256'] != SELECTION or freeze['seal_sha256'] != SEAL or not freeze['frozen_before_full_stage2']:
        raise RuntimeError('selection checkpoint link')
    prefixes = sorted(int(k.rsplit('k',1)[1]) for k in analysis['retained_basis_cases'])
    if prefixes != [180,200,220,240,260,280,300]: raise RuntimeError('nested prefixes')
    for name, record in auth0['historical_references'].items():
        if sha(ROOT/'research/checkpoints'/name/'checkpoint.json') != record['checkpoint_json_sha256']:
            raise RuntimeError(f'historical reference {name}')

    with np.load(CP/'stage1_eigensystem.npz') as z:
        values = z['eigenvalues_GHz'].copy(); vectors = z['eigenvectors'].copy()
    orth = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(300), ord=np.inf))
    max_stage1_residual = float(max(meta['stage1_solver_diagnostics']['eigenpair_residuals']))
    if orth > 1e-9 or max_stage1_residual > 1e-8: raise RuntimeError('Stage-1 numerical auth')
    del vectors
    with np.load(CP/'projected_operators.npz') as z: ops = {n:z[n] for n in z.files}

    p = meta['configuration']['physical_parameters']
    ec, k, _, _ = _coeffs(p['EJ'],p['EC'],p['EL'],p['ELK'],p['EJS'],p['ECS'],
                           p['eps_J'],p['eps_LK'],p['eC'],p['eP'])
    A, B = 4*ec[3,3], .5*k[2,2]
    scale_x, scale_p = (A/(4*B))**.25, 1/(2*(A/(4*B))**.25)
    a = np.diag(np.sqrt(np.arange(1,8)),1)
    x4, p4 = scale_x*(a+a.T), 1j*scale_p*(a.T-a)
    w4 = 2*np.sqrt(A*B)*(np.arange(8)+.5)
    h = np.kron(np.diag(values),np.eye(8)) + np.kron(np.eye(300),np.diag(w4))
    h = h.astype(complex)
    for n in range(3): h += 8*ec[n,3]*np.kron(ops[f'n{n+1}'],p4)
    h += k[0,2]*np.kron(ops['x2'],x4) + k[1,2]*np.kron(ops['x3'],x4)
    herm = float(np.max(abs(h-h.conj().T)))
    if herm > 1e-12: raise RuntimeError(f'Hermiticity {herm}')
    h = (h+h.conj().T)/2
    grid_phi = np.kron(.5*ops['x2']+ops['x3'],np.eye(8))

    print('Solving authenticated finite Stage-2 reconstructions', flush=True)
    full = solve(h, np.arange(2400))
    cases = {300: full}
    for n in (280,260): cases[n] = solve(h[:8*n,:8*n], np.arange(8*n), full['embedded'])
    reconstruction = {}
    saved = analysis['retained_basis_cases']
    for n, case in cases.items():
        mat = case['embedded'].conj().T @ grid_phi @ case['embedded']
        record = saved[f'nested_prefix_k{n}']
        ee = float(max(abs(case['e']-record['energies_GHz'])))
        oe = float(np.max(abs(abs(mat)-record['grid_phi_abs'])))
        if ee > 1e-10 or oe > 1e-10: raise RuntimeError(('reconstruction',n,ee,oe))
        case['grid_phi'] = mat
        reconstruction[str(n)] = {'energy_error_GHz':ee,'grid_phi_error':oe,
            'eigenpair_residual_GHz':case['residual'],'tracking':case['confidence']}

    # Central final-shell attribution, plus isolated held-out shells for selector agreement.
    central, central_ranks, central_modes = attribute(h,2240,list(range(280,300)),cases[280],full,'P280_Q280_299')
    s1_h = h[:2240,:2240]
    s1, s1_ranks, s1_modes = attribute(s1_h,2080,list(range(260,280)),cases[260],cases[280],'S1_isolated_260_279')
    s2_indices = np.r_[np.arange(2080),np.arange(2240,2400)]
    s2_h = h[np.ix_(s2_indices,s2_indices)]
    s2_full = solve(s2_h,s2_indices,full['embedded'])
    s2, s2_ranks, s2_modes = attribute(s2_h,2080,list(range(280,300)),cases[260],s2_full,'S2_isolated_280_299')
    rank_rows = central_ranks+s1_ranks+s2_ranks
    mode_rows = central_modes+s1_modes+s2_modes
    central_trans = transition_attribution(central,cases[280],full)
    s1_trans = transition_attribution(s1,cases[260],cases[280])
    s2_trans = transition_attribution(s2,cases[260],s2_full)

    print('Computing Q1/Q2 and influential-rank nonadditivity',flush=True)
    q1 = solve(h[:2320,:2320],np.arange(2320),full['embedded'])
    q2_indices = np.r_[np.arange(2240),np.arange(2320,2400)]
    q2 = solve(h[np.ix_(q2_indices,q2_indices)],q2_indices,full['embedded'])
    no_cross_h = h.copy(); no_cross_h[2240:2320,2320:]=0; no_cross_h[2320:,2240:2320]=0
    no_cross = solve(no_cross_h,np.arange(2400),full['embedded'])
    nonadd = energy_nonadditivity(cases[280],q1,q2,full,no_cross)
    nonadd['Q1Q2_coupling_frobenius_GHz'] = float(np.linalg.norm(h[2240:2320,2320:]))

    aggregate = {r:0. for r in range(280,300)}
    for row in central_ranks: aggregate[row['rank']] += abs(row['exact_self_energy_MHz'])
    influential = sorted(aggregate,key=lambda r:(-aggregate[r],r))[:3]
    subsets = {}
    for size in (1,2,3):
        for sub in combinations(influential,size):
            idx = np.r_[np.arange(2240),np.concatenate([np.arange(r*8,(r+1)*8) for r in sub])]
            subsets[sub] = solve(h[np.ix_(idx,idx)],idx,full['embedded'])
    addback = {'influential_ranks':influential,'aggregate_absolute_self_energy_MHz':aggregate,'states':{},'transitions':{}}
    for s in STATES:
        e0=cases[280]['e'][s]
        effects={','.join(map(str,k)):float((v['e'][s]-e0)*1000) for k,v in subsets.items()}
        pairint={','.join(map(str,pair)):float((subsets[pair]['e'][s]-subsets[(pair[0],)]['e'][s]-subsets[(pair[1],)]['e'][s]+e0)*1000)
                 for pair in combinations(influential,2)}
        a,b,c=influential
        triple=float((subsets[(a,b,c)]['e'][s]-subsets[(a,b)]['e'][s]-subsets[(a,c)]['e'][s]-subsets[(b,c)]['e'][s]
                     +subsets[(a,)]['e'][s]+subsets[(b,)]['e'][s]+subsets[(c,)]['e'][s]-e0)*1000)
        addback['states'][str(s)]={'effects_MHz':effects,'pair_interactions_MHz':pairint,'triple_interaction_MHz':triple}
    for name,pair in FREQ.items():
        f0=frequency(cases[280],pair)
        effects={','.join(map(str,k)):frequency(v,pair)-f0 for k,v in subsets.items()}
        pairint={','.join(map(str,pa)):frequency(subsets[pa],pair)-frequency(subsets[(pa[0],)],pair)-frequency(subsets[(pa[1],)],pair)+f0
                 for pa in combinations(influential,2)}
        a,b,c=influential
        triple=(frequency(subsets[(a,b,c)],pair)-frequency(subsets[(a,b)],pair)-frequency(subsets[(a,c)],pair)-frequency(subsets[(b,c)],pair)
                +frequency(subsets[(a,)],pair)+frequency(subsets[(b,)],pair)+frequency(subsets[(c,)],pair)-f0)
        addback['transitions'][name]={'effects_MHz':effects,'pair_interactions_MHz':pairint,'triple_interaction_MHz':triple}

    # Frozen selector alignment and S2/b10 exception.
    selection=json.loads(sel_path.read_text()); validation=json.loads((ROOT/'research/sandbox/k300_results/validation_metrics.json').read_text())
    selector_rows=[]; selector_summary={}
    isolated_rows={'S1':s1_ranks,'S2':s2_ranks}
    for shell in ('S1','S2'):
        data=selection['shells'][shell]
        exact={r:0. for r in data['order']}
        signed={r:{str(s):0. for s in STATES} for r in data['order']}
        for row in isolated_rows[shell]:
            exact[row['rank']]+=abs(row['exact_self_energy_MHz']); signed[row['rank']][str(row['state'])]=row['exact_self_energy_MHz']
        total=sum(exact.values()); order_index={r:i+1 for i,r in enumerate(data['order'])}
        gain={x['rank']:x['gain'] for x in data['greedy_trace']}
        feature={x['rank']:x for x in data['features']}
        exact_order=sorted(exact,key=lambda r:(-exact[r],r))
        selector_summary[shell]={'physics_order':data['order'],'exact_aggregate_order':exact_order,
            'top5_overlap_count':len(set(data['order'][:5])&set(exact_order[:5])),'budgets':{}}
        controls={m:{r:i+1 for i,r in enumerate(selection['spaces'][f'{shell}_{m}_15'][260:])}
                  for m in ('prefix','spread','spread_reverse')}
        for budget in (3,6,10,15):
            chosen=set(data['order'][:budget]); capture=sum(exact[r] for r in chosen)/total
            selector_summary[shell]['budgets'][str(budget)]={'absolute_exact_rank_weight_capture':capture,
                'selected_exact_top_ranks':sorted(chosen&set(exact_order[:budget]))}
        for rank in sorted(exact):
            groups={tuple(g['states']):g for g in feature[rank]['groups']}
            row={'shell':shell,'rank':rank,'physics_order':order_index[rank],'physics_gain':gain[rank],
                 'exact_aggregate_abs_self_energy_MHz':exact[rank],
                 'exact_aggregate_fraction':exact[rank]/total}
            for s in STATES: row[f'exact_state_{s}_MHz']=signed[rank][str(s)]
            for pair,key in [((0,1),'01'),((2,3),'23'),((7,8),'78')]:
                g=groups[pair]
                row.update({f'group_{key}_trace_GHz':g['trace_GHz'],f'group_{key}_coupling_GHz':g['coupling_norm_GHz'],
                    f'group_{key}_min_den_GHz':g['minimum_denominator_GHz'],f'group_{key}_max_eta':g['maximum_eta']})
            for b in (3,6,10,15): row[f'physics_b{b}']=rank in set(data['order'][:b])
            for m,orders in controls.items(): row[f'{m}_order']=orders.get(rank)
            selector_rows.append(row)

    def solve_rank_set(ranks, ref):
        idx=np.r_[np.arange(2080),np.concatenate([np.arange(r*8,(r+1)*8) for r in sorted(ranks)])]
        return solve(h[np.ix_(idx,idx)],idx,ref)
    physics_ranks=selection['shells']['S2']['order'][:10]
    reverse_ranks=selection['spaces']['S2_spread_reverse_10'][260:]
    physics_case=solve_rank_set(physics_ranks,s2_full['embedded'])
    reverse_case=solve_rank_set(reverse_ranks,s2_full['embedded'])
    def exception_case(case,ranks,method):
        state={}
        for s in (7,8):
            state[str(s)]={'correction_from_base_MHz':float((case['e'][s]-cases[260]['e'][s])*1000),
                           'endpoint_correction_MHz':float((s2_full['e'][s]-cases[260]['e'][s])*1000),
                           'endpoint_error_MHz':float((case['e'][s]-s2_full['e'][s])*1000)}
        errors={n:frequency(case,p)-frequency(s2_full,p) for n,p in FREQ.items()}
        cancellation=1-abs(errors['f78'])/(abs(errors['f07'])+abs(errors['f08']))
        saved=float(validation['comparisons']['S2']['10'][method]['components']['carrier:f78']['value'])
        if abs(errors['f78']-saved)>1e-7: raise RuntimeError(('selector reconstruction',method,errors['f78'],saved))
        return {'ranks':ranks,'states':state,'transition_errors_MHz':errors,
                'f78_common_mode_cancellation_fraction':cancellation,'saved_f78_error_MHz':saved}
    exception={'physics':exception_case(physics_case,physics_ranks,'physics'),
               'reverse_spread':exception_case(reverse_case,reverse_ranks,'spread_reverse')}
    exception['common_ranks']=sorted(set(physics_ranks)&set(reverse_ranks))
    exception['physics_only']=sorted(set(physics_ranks)-set(reverse_ranks))
    exception['reverse_only']=sorted(set(reverse_ranks)-set(physics_ranks))
    iso_s2_by_rank={r:{str(s):0. for s in STATES} for r in range(280,300)}
    for row in s2_ranks: iso_s2_by_rank[row['rank']][str(row['state'])]=row['exact_self_energy_MHz']
    for key,ranks in [('physics_only',exception['physics_only']),('reverse_only',exception['reverse_only'])]:
        exception[key+'_schur_sums_MHz']={str(s):sum(iso_s2_by_rank[r][str(s)] for r in ranks) for s in STATES}
    exception['set_schur_f78_MHz']={key:sum(iso_s2_by_rank[r]['8']-iso_s2_by_rank[r]['7'] for r in ranks)
                                    for key,ranks in [('physics',physics_ranks),('reverse_spread',reverse_ranks)]}

    # Same-parent collision/selectivity table.
    freq_rows=[]
    for n in (260,280,300):
        case=cases[n]
        row={'k':n}
        for name,pair in FREQ.items(): row[name+'_MHz']=frequency(case,pair)
        row['f78_minus_f01_MHz']=row['f78_MHz']-row['f01_MHz']
        row['outer_f07_minus_f81_MHz']=row['f07_MHz']-row['f81_MHz']
        row['g01']=float(abs(case['grid_phi'][0,1])); row['g78']=float(abs(case['grid_phi'][7,8]))
        freq_rows.append(row)

    central_labels=[central[str(s)]['mechanism'] for s in (7,8)]
    overall='E' if 'E' in central_labels else central_labels[0] if central_labels[0]==central_labels[1] else 'D'
    authentication={'status':'PASS','complete':True,'configuration_sha256':CONFIG,
        'checkpoint_sha256':CHECKPOINT,'payload_hashes_verified':True,'selection_sha256':SELECTION,
        'selection_seal_sha256':SEAL,'nested_prefixes':prefixes,'max_stage1_residual':max_stage1_residual,
        'orthogonality_residual_inf':orth,'historical_references_unchanged':True}

    if tracked_snapshot()!=tracked_before: raise RuntimeError('tracked file mutation during analysis')
    checkpoints_after={}
    for root in checkpoint_roots: checkpoints_after.update(tree_snapshot(root))
    if checkpoints_after!=checkpoints_before: raise RuntimeError('checkpoint mutation during analysis')
    result={'schema':1,'methodology_sha256':METHOD,'authentication':authentication,
        'data_feasibility':'SUFFICIENT','schur_effective_hamiltonian':'VALIDATED',
        'dimensions':{'P':2240,'Q':160,'full':2400,'Q1':80,'Q2':80},
        'hermiticity_error':herm,'reconstruction':reconstruction,
        'central_final_shell':{'states':central,'transitions':central_trans,'mechanism':overall},
        'isolated_selector_shells':{'S1':{'states':s1,'transitions':s1_trans},
                                    'S2':{'states':s2,'transitions':s2_trans}},
        'Q1_Q2_nonadditivity':nonadd,'influential_rank_addbacks':addback,
        'selector_alignment':selector_summary,'S2_budget10_f78_exception':exception,
        'frequency_selectivity':freq_rows,'frozen_selector_validation':'BORDERLINE',
        'retained_basis_convergence':'FAIL','tracked_files_unchanged':True,
        'authenticated_checkpoints_unchanged':True,
        'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT).decode()}
    (HERE/'summary.json').write_text(json.dumps(result,indent=2,default=serial,allow_nan=False)+'\n')
    write_csv(HERE/'rank_attributions.csv',rank_rows)
    write_csv(HERE/'shell_mode_contributions.csv',mode_rows)
    write_csv(HERE/'selector_diagnostics.csv',selector_rows)
    write_csv(HERE/'frequency_selectivity.csv',freq_rows)
    (HERE/'preservation.json').write_text(json.dumps({'tracked':tracked_before,'checkpoints':checkpoints_before},indent=2)+'\n')
    print('Completed: mechanism',overall,'influential ranks',influential,flush=True)


if __name__ == '__main__': main()
