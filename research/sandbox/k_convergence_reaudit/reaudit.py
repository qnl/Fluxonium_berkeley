"""Saved-data re-audit. No Hamiltonian construction, reconstruction, or eigensolve.

Run with PYTHONDONTWRITEBYTECODE=1 and one BLAS thread. All writes stay HERE.
The validated tracker is reused; only Gram whitening uses an equivalent SVD
adapter, avoiding even its small Gram eigendecomposition in this no-eigensolve run.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / 'subspace_tracking'
POST = HERE.parent / 'n71_retained_subspace_postprocess.json'
CP71 = ROOT / 'research/checkpoints/four-mode-asymmetric-n71-k180'
CP51 = ROOT / 'research/checkpoints/2026-09-26-asymmetric-n51'
PATHS = [(0, 7, 8, 1), (0, 7, 2, 1), (0, 3, 8, 1)]
CARRIERS = [(0, 1), (0, 7), (0, 8), (7, 8), (1, 8), (2, 7), (1, 2), (0, 3), (3, 8)]
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path); INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def serial(x):
    if isinstance(x, np.ndarray): return serial(x.tolist())
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, complex): return {'real': x.real, 'imag': x.imag}
    if isinstance(x, dict): return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    return x


def save(name, obj):
    (HERE / name).write_text(json.dumps(serial(obj), indent=2, allow_nan=False) + '\n')


def table(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(serial(v)) if isinstance(v, (list, dict, tuple, np.ndarray)) else v for k, v in row.items()})


def forbidden(*args, **kwargs):
    raise RuntimeError('No new eigensolve permitted in this saved-data audit.')


def svd_frame(u, rcond=1e-12):
    if not np.isfinite(rcond) or not 0 < rcond < 1: raise ValueError('Invalid rcond')
    u = tracker._matrix(u); g = u.conj().T @ u
    z, s, _ = la.svd((g + g.conj().T) / 2)
    if s[-1] <= rcond * s[0]: raise ValueError('Rank-deficient subspace frame')
    c = (z / np.sqrt(s)) @ z.conj().T
    return u @ c, c, dict(gram_eigenvalues=s[::-1], gram_error=float(la.norm(g - np.eye(len(s)), 2)), gram_condition=float(s[0] / s[-1]))


spec = importlib.util.spec_from_file_location('reaudit_validated_tracking', OLD / 'tracking.py')
tracker = importlib.util.module_from_spec(spec); sys.modules[spec.name] = tracker; spec.loader.exec_module(tracker)
tracker.orthonormal_frame = svd_frame
for module, names in [(np.linalg, ['eig', 'eigh', 'eigvals', 'eigvalsh']),
                      (la, ['eig', 'eigh', 'eigvals', 'eigvalsh', 'eigh_tridiagonal', 'eigvalsh_tridiagonal']),
                      (sla, ['eigs', 'eigsh', 'lobpcg'])]:
    for name in names: setattr(module, name, forbidden)


def load(name):
    path = OLD / (name + '.npz'); INPUTS[str(path.relative_to(ROOT))] = sha(path)
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def authenticate():
    prov = read(OLD / 'provenance.json')
    assert sha(CP71 / 'checkpoint.json') == prov['N71_checkpoint_sha256']
    for name, h in prov['N71_payload_sha256'].items(): assert sha(CP71 / name) == h
    for name, h in prov['N51_file_sha256'].items(): assert sha(CP51 / name) == h
    assert sha(POST) == prov['retained_selection_file_sha256']
    manifest = read(OLD / 'resume_manifest.json')
    for name, h in manifest.items():
        if name.endswith('.npz'): assert sha(OLD / name) == h, name
    INPUTS[str((OLD / 'tracking.py').relative_to(ROOT))] = sha(OLD / 'tracking.py')
    read(HERE.parent / 'k220_preregistered_criteria.json')


def band(value, pass_max, borderline_max):
    return 'PASS' if abs(value) <= pass_max else 'BORDERLINE' if abs(value) <= borderline_max else 'FAIL'


def amplitude_band(a):
    return 'PASS' if a >= .999 else 'BORDERLINE' if a >= .995 else 'FAIL'


def compact_subspace(d):
    return {k: v for k, v in d.items() if k not in ['reference_frame', 'candidate_frame', 'reference_transform', 'candidate_transform', 'overlap']}


def tracking(a, b):
    return tracker.track_states(a['vectors'], b['vectors'], a['energy'] - a['energy'][0], b['energy'] - b['energy'][0])


def all_paths(phi):
    rows = []
    for a in range(2, 11):
        for b in range(2, 11):
            if a == b: continue
            edges = [float(abs(phi[0, a])), float(abs(phi[a, b])), float(abs(phi[b, 1]))]
            rows.append(dict(path=[0, a, b, 1], edge_magnitudes=edges, score_product=float(np.prod(edges))))
    return sorted(rows, key=lambda r: -r['score_product'])


def operator_block(a, b, ai, bi, aj, bj):
    left = tracker.track_subspaces(a['vectors'][:, ai], b['vectors'][:, bi])
    right = tracker.track_subspaces(a['vectors'][:, aj], b['vectors'][:, bj])
    ar = left['reference_transform'].conj().T @ a['phi'][np.ix_(ai, aj)] @ right['reference_transform']
    br = left['candidate_transform'].conj().T @ b['phi'][np.ix_(bi, bj)] @ right['candidate_transform']
    comp = tracker.compare_blocks(ar, br, left['rotation'], right['rotation'])
    rf = comp['reference']['frobenius']; ef = comp['candidate']['frobenius']; df = comp['aligned_difference_frobenius']
    row = dict(reference_states=ai, endpoint_states=bi, target_states=aj, endpoint_target_states=bj,
        reference_singular_values=comp['reference']['singular_values'], endpoint_singular_values=comp['candidate']['singular_values'],
        reference_frobenius=rf, endpoint_frobenius=ef, reference_operator_norm=comp['reference']['operator_norm'],
        endpoint_operator_norm=comp['candidate']['operator_norm'], aligned_difference_frobenius=df,
        aligned_difference_operator=comp['aligned_difference_operator'],
        aligned_relative_to_case=None if rf < 1e-8 else df / rf,
        aligned_relative_to_endpoint=None if ef < 1e-8 else df / ef,
        target_minimum_singular_value=right['minimum_singular_value'],
        strict_operator_band='NEAR_ZERO' if ef < 1e-8 else band(df / ef, .03, .05))
    detail = dict(reference_block=ar, endpoint_block=br, aligned_endpoint_block=comp['aligned_candidate'],
        left_rotation=left['rotation'], right_rotation=right['rotation'])
    return row, detail


def main():
    authenticate()
    endpoints = {71: load('N71_k180'), 51: load('N51_k180')}
    state_rows, cluster_rows, op_rows, low_rows, path_rows, case_rows = [], [], [], [], [], []
    block_details, cases, tracks, adjacent_rows, adjacent_ops = {}, {}, {}, [], []
    names = [f'N71_priority_k{k}' for k in [170, 172, 175, 177, 179]] + [f'asymmetric_priority_k{k}' for k in [170, 172, 175]] + ['N71_core160']
    for name in names:
        a = load(name); b = endpoints[71]; t = tracking(a, b); tracks[name] = t
        family = 'N71_priority' if name.startswith('N71_priority') else 'asymmetric_priority' if name.startswith('asymmetric') else 'energy'
        k = len(a['indices']); perm = t['assignment']; inverse = np.argsort(perm)
        clusters = tracker.ambiguity_clusters(t, a['vectors'], b['vectors'])
        ambiguous = {i for c in clusters for i in c['reference_states']}
        for row in t['rows']:
            i, j = row['reference'], row['candidate']; scores = t['probability'][:, j].copy(); scores[i] = -1
            state_rows.append(dict(case=name, grid=71, family=family, k=k, evidence='vectors_Hungarian',
                **row, column_competitor_state=int(np.argmax(scores)),
                column_probability_margin=row['assigned_probability'] - row['column_competitor_probability'],
                energy_displacement_MHz=1000 * row['energy_displacement'],
                state_accuracy_band=amplitude_band(row['assigned_amplitude']),
                identity_class='ambiguous_cluster' if i in ambiguous else 'clean' if row['assigned_amplitude'] >= .999 else 'borderline',
                energy_rank_assignment_changed_vs_previous_k=None, cycle_inconsistent_vs_previous_k=None))
        sets = [(c['reference_states'], 'automatic', c['status']) for c in clusters]
        for ids, label in [([45, 46], 'pair45_46_followup'), ([7, 8], 'control78'), (list(range(11)), 'low11')]:
            if not any(ids == x[0] for x in sets): sets.append((ids, label, 'monitor_not_automatic_ambiguity'))
        for ids, origin, status in sets:
            dest = perm[ids].tolist(); d = tracker.track_subspaces(a['vectors'][:, ids], b['vectors'][:, dest])
            cid = name + ':' + ','.join(map(str, ids))
            cluster_rows.append(dict(case=name, grid=71, family=family, k=k, cluster_id=cid, origin=origin,
                reference_states=ids, endpoint_states=dest, status=status,
                reference_excitation_energies_GHz=a['energy'][ids]-a['energy'][0],
                endpoint_excitation_energies_GHz=b['energy'][dest]-b['energy'][0],
                cluster_mean_excitation_delta_MHz=float(1000*(np.mean(a['energy'][ids]-a['energy'][0])-np.mean(b['energy'][dest]-b['energy'][0]))),
                **compact_subspace(d)))
            groups = {'internal': ids, 'to_logical01': [0, 1], 'to_low11': list(range(11)),
                      'to_stored_complement': [j for j in range(48) if j not in ids]}
            for label, target in groups.items():
                if not target: continue
                row, detail = operator_block(a, b, ids, dest, target, perm[target].tolist())
                key = cid + ':' + label; block_details[key] = detail
                op_rows.append(dict(case=name, grid=71, family=family, k=k, object_type='subspace_block',
                    cluster_id=cid, block=label, origin=origin, **row))
        # Individual low-energy labels are permitted only after tracking gates.
        ids = inverse[:11]; assert not any(t['rows'][i]['ambiguous'] for i in ids)
        p = a['phi'][np.ix_(ids, ids)]
        # Phase-align the CASE vectors toward endpoint for signed edge diagnostics.
        s = np.diag(b['vectors'][:, :11].conj().T @ a['vectors'][:, ids])
        w = np.diag(s.conj() / abs(s)); p = w.conj().T @ p @ w
        cases[(71, family, k)] = dict(name=name, grid=71, family=family, k=k,
            evidence='vectors_Hungarian_signed_operator', energy=a['energy'][ids],
            probability=np.array([t['rows'][i]['assigned_probability'] for i in ids]),
            assignment=ids.tolist(), phi=p, paths=all_paths(p), confidence='individually_verified', indices=a['indices'].tolist())
        case_rows.append(dict(case=name, grid=71, family=family, k=k, evidence='vectors_48_signed_phi', stored_states=48,
            clean=sum(r['assigned_amplitude'] >= .999 and r['reference'] not in ambiguous for r in t['rows']),
            borderline=sum(r['assigned_amplitude'] < .999 and r['reference'] not in ambiguous for r in t['rows']),
            ambiguous_states=sorted(ambiguous), index_exchanges=[r['reference'] for r in t['rows'] if r['energy_rank_changed']],
            minimum_probability=min(r['assigned_probability'] for r in t['rows']), automatic_clusters=len(clusters)))
    # Adjacent-k consistency is checked by composition through actual vectors,
    # not by pretending that equal energy ranks are physical identity labels.
    for family in ['N71_priority', 'asymmetric_priority']:
        series = sorted([c for c in cases.values() if c['family'] == family], key=lambda c: c['k'])
        series.append(dict(name='N71_k180',k=180))
        for previous, current in zip(series[:-1], series[1:]):
            a, b = load(previous['name']), load(current['name']); step = tracking(a, b)
            pa = tracks[previous['name']]['assignment']
            pb = np.arange(48) if current['k']==180 else tracks[current['name']]['assignment']
            for r in step['rows']:
                i, j = r['reference'], r['candidate']; inconsistent = int(pa[i]) != int(pb[j])
                adjacent_rows.append(dict(family=family, k_from=previous['k'], k_to=current['k'], **r,
                    endpoint_from=int(pa[i]), endpoint_to=int(pb[j]), cycle_inconsistent=inconsistent))
            for row in state_rows:
                if row['case'] == current['name']:
                    i = row['reference']; source = int(np.where(step['assignment'] == i)[0][0])
                    row['energy_rank_assignment_changed_vs_previous_k'] = bool(pa[i] != pb[i])
                    row['cycle_inconsistent_vs_previous_k'] = bool(pa[source] != pb[i])
            if family=='N71_priority':
                for label,ids,target in [('pair45_46_to_low11',[45,46],list(range(11))),('control78_internal',[7,8],[7,8]),('low11_internal',list(range(11)),list(range(11)))]:
                    # Define each subspace by endpoint descendants at each k;
                    # endpoint ambiguity remains a cluster, not a state label.
                    ia=np.argsort(pa); ib=np.argsort(pb)
                    row,detail=operator_block(a,b,ia[ids].tolist(),ib[ids].tolist(),ia[target].tolist(),ib[target].tolist())
                    adjacent_ops.append(dict(family=family,k_from=previous['k'],k_to=current['k'],block=label,**row))
    for grid,b in endpoints.items():
        t=tracking(b,b)
        for row in t['rows']:
            state_rows.append(dict(case=f'N{grid}_k180',grid=grid,family='endpoint',k=180,evidence='endpoint_self_check',**row,
                identity_class='endpoint_self_check',state_accuracy_band='SELF_NOT_CONVERGENCE',energy_displacement_MHz=1000*row['energy_displacement']))
    post = read(POST)
    summary_sources = []
    for label, values in post['candidates'].items():
        fam, grid = label.split('_on_n'); family = 'N71_priority' if fam == 'n71_priority' else 'common_priority'
        for key, c in values.items(): summary_sources.append((int(grid), family, c, str(POST.relative_to(ROOT))))
    summary_sources += [(51, 'energy', post['n51_core160'], str(POST.relative_to(ROOT)))]
    historical = read(ROOT / 'Figures/Rabi_3Photon/four_mode_asymmetric_shell_basis_audit.json')
    for label, c in historical['candidate_cases'].items():
        family = 'asymmetric_priority' if label.startswith('asymmetric') else 'energy' if label.startswith('energy') else 'inherited_rejected'
        summary_sources.append((51, family, c, 'Figures/Rabi_3Photon/four_mode_asymmetric_shell_basis_audit.json'))
    other = read(ROOT / 'Figures/Rabi_3Photon/four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected.json')
    for label, original in other['cases'].items():
        if label == 'energy_k180': continue
        c = dict(original); c['pathways'] = c['grid_phi_pathways']
        c['state_assignment_candidate_for_reference'] = c['state_assignment_candidate_row_for_reference_state']
        family = 'energy' if label.startswith('energy') else 'inherited_rejected'
        summary_sources.append((51, family, c, 'Figures/Rabi_3Photon/four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected.json'))
    for grid, family, c, source in summary_sources:
        k = len(c['retained_indices']); key = (grid, family, k)
        if key in cases:
            assert cases[key]['indices'] == c['retained_indices']
            assert np.max(abs(cases[key]['energy'] - np.array(c['energies_GHz']))) < 1e-7
            continue
        name = f'N{grid}_{family}_k{k}'; confidence = c.get('tracking_confidence')
        edges = {}
        for path in c['pathways']:
            for (i, j), value in zip(zip(path['path'][:-1], path['path'][1:]), path['edge_magnitudes']): edges[tuple(sorted((i, j)))] = value
        for group in ['primary_path_coupling_errors', 'logical_coupling_errors', 'audited_logical_leakage_coupling_errors']:
            for e in c.get(group, []): edges[tuple(sorted(e['transition']))] = e['candidate']
        if 'grid_phi_abs' in c:
            for i in range(11):
                for j in range(i + 1, 11): edges[i, j] = c['grid_phi_abs'][i][j]
        cases[key] = dict(name=name, grid=grid, family=family, k=k, evidence='saved_summary_only', source=source,
            energy=np.array(c['energies_GHz']), probability=np.array(c['state_overlaps'])**2,
            assignment=c['state_assignment_candidate_for_reference'], paths=c['pathways'], edges=edges,
            confidence=confidence, indices=c['retained_indices'])
        for endpoint, (idx, amp) in enumerate(zip(c['state_assignment_candidate_for_reference'], c['state_overlaps'])):
            state_rows.append(dict(case=name, grid=grid, family=family, k=k, evidence='saved_summary_only',
                reference=idx, candidate=endpoint, assigned_probability=amp**2, assigned_amplitude=amp,
                energy_displacement_MHz=float(1000 * ((endpoints[grid]['energy'][endpoint] - endpoints[grid]['energy'][0]) - (c['energies_GHz'][endpoint] - c['energies_GHz'][0]))),
                energy_rank_changed=idx != endpoint, identity_class='not_reauditable_summary_only',
                state_accuracy_band=amplitude_band(amp), saved_aggregate_confidence=confidence,
                reasons=['full_overlap_matrix_and_vectors_unavailable']))
        case_rows.append(dict(case=name, grid=grid, family=family, k=k, evidence='saved_summary_only', stored_states=11,
            minimum_probability=min(c['state_overlaps'])**2, automatic_clusters=None, source=source))
    # Endpoint rows for each family make every trend explicit; self-agreement
    # is a normalization check, never independent convergence evidence.
    for grid, family in sorted({(g, f) for g, f, k in cases}):
        b = endpoints[grid]
        cases[grid, family, 180] = dict(name=f'N{grid}_{family}_k180', grid=grid, family=family, k=180,
            evidence='endpoint_self_check', energy=b['energy'][:11], probability=np.ones(11), assignment=list(range(11)),
            phi=b['phi'][:11, :11], paths=all_paths(b['phi']), confidence='endpoint', indices=list(range(180)))
    observables = {}
    for key, c in sorted(cases.items()):
        grid, family, k = key; b = endpoints[grid]; e = c['energy']; be = b['energy'][:11]
        base = dict(case=c['name'], grid=grid, family=family, k=k, evidence=c['evidence'])
        obs = {}; edge_errors = {}
        for i in range(11):
            delta = 1000 * ((e[i] - e[0]) - (be[i] - be[0])); obs[f'E{i}_excitation_MHz'] = 1000 * (e[i] - e[0])
            low_rows.append(dict(**base, object_type='state', observable=f'E{i}', state=i, case_energy_GHz=e[i],
                endpoint_energy_GHz=be[i], absolute_energy_delta_MHz=1000 * (e[i] - be[i]),
                case_excitation_GHz=e[i] - e[0], endpoint_excitation_GHz=be[i] - be[0], delta_MHz=delta,
                assigned_probability=c['probability'][i], assigned_amplitude=np.sqrt(c['probability'][i]),
                candidate_energy_index=c['assignment'][i], label_evidence=c['confidence'], strict_band=band(delta, 1, 5)))
        for i, j in CARRIERS:
            value = abs(e[j] - e[i]) * 1000; ref = abs(be[j] - be[i]) * 1000; fine = (i, j) == (7, 8)
            obs[f'f{i}_{j}_MHz'] = value
            low_rows.append(dict(**base, object_type='carrier', observable=f'f{i}_{j}', case_frequency_MHz=value,
                endpoint_frequency_MHz=ref, delta_MHz=value-ref, strict_band=band(value-ref, .5 if fine else 1, 1 if fine else 5),
                practical_band=band(value-ref, 1 if fine else 5, 2 if fine else 10)))
        edges = c.get('edges', {(i, j): float(abs(c['phi'][i, j])) for i in range(11) for j in range(i+1, 11)} if 'phi' in c else {})
        for (i, j), value in sorted(edges.items()):
            ref = float(abs(b['phi'][i, j])); relative = None if ref < 1e-8 else (value-ref)/ref
            obs[f'g{i}_{j}'] = value; edge_errors[i, j] = relative
            signed = c['phi'][i, j] if 'phi' in c and c['evidence'] != 'endpoint_self_check' else None
            op_rows.append(dict(**base, object_type='individual_edge_magnitude', edge=[i, j], case_magnitude=value,
                endpoint_magnitude=ref, magnitude_delta=value-ref, relative_to_endpoint=relative,
                aligned_case_real=None if signed is None else float(signed.real), aligned_case_imag=None if signed is None else float(signed.imag),
                endpoint_real=float(b['phi'][i, j].real), endpoint_imag=float(b['phi'][i, j].imag),
                strict_operator_band='NEAR_ZERO' if relative is None else band(relative, .03, .05),
                minimum_endpoint_state_amplitude=float(np.sqrt(min(c['probability'][i], c['probability'][j]))),
                carrier_error_MHz=float(1000*(abs(e[j]-e[i])-abs(be[j]-be[i])))))
        ranklist = [tuple(p['path']) for p in c['paths']]; reflist = all_paths(b['phi']); refranks = [tuple(p['path']) for p in reflist]
        for path in PATHS:
            ref = reflist[refranks.index(path)]; rank = ranklist.index(path)+1 if path in ranklist else None
            if rank is None:
                path_rows.append(dict(**base, path=path, status='not_in_saved_top_list', rank=None)); continue
            p = c['paths'][rank-1]; value = p['score_product']; relative = value/ref['score_product']-1
            errs = [x/y-1 for x,y in zip(p['edge_magnitudes'], ref['edge_magnitudes'])]
            cancellation = sum(abs(x) for x in errs) - abs(sum(errs))
            obs['path_' + ''.join(map(str,path))] = value
            row = dict(**base, path=path, edge_magnitudes=p['edge_magnitudes'], edge_relative_errors=errs,
                product=value, endpoint_product=ref['score_product'], relative_to_endpoint=relative,
                rank=rank, endpoint_rank=refranks.index(path)+1, rank_changed=rank != refranks.index(path)+1,
                rank_scope='all_72_simple_paths_0_a_b_1_a_b_in_2_to_10' if 'phi' in c else 'saved_sorted_top10',
                maximum_edge_relative_error=max(abs(x) for x in errs), first_order_cancellation=sum(abs(x) for x in errs)>1e-12 and cancellation>0,
                cancellation_fraction=0 if sum(abs(x) for x in errs)<1e-12 else cancellation/sum(abs(x) for x in errs),
                strict_product_band=band(relative,.05,.10), strict_edges_band=band(max(abs(x) for x in errs),.03,.05),
                carrier_errors_MHz=[float(1000*(abs(e[j]-e[i])-abs(be[j]-be[i]))) for i,j in zip(path[:-1],path[1:])],
                maximum_carrier_error_MHz=max(float(1000*abs(abs(e[j]-e[i])-abs(be[j]-be[i]))) for i,j in zip(path[:-1],path[1:])),
                label_validation='individually_verified' if c['evidence']=='vectors_Hungarian_signed_operator' else 'endpoint_self_check' if k==180 else 'saved_tracking_only_not_reauditable',
                minimum_state_amplitude=float(np.sqrt(min(c['probability'][i] for i in path))))
            if 'phi' in c:
                z=c['phi'][path[0],path[1]]*c['phi'][path[1],path[2]]*c['phi'][path[2],path[3]]
                row.update(aligned_product_real=float(z.real),aligned_product_imag=float(z.imag))
            path_rows.append(row)
        observables[key] = obs
    trends = []
    for grid, family in sorted({(g,f) for g,f,k in observables}):
        ks=sorted(k for g,f,k in observables if (g,f)==(grid,family))
        common=set.intersection(*(set(observables[grid,family,k]) for k in ks))
        for name in sorted(common):
            vals=np.array([observables[grid,family,k][name] for k in ks]); errors=vals-vals[-1]; diffs=np.diff(vals)
            tol=max(1e-12,abs(vals[-1])*1e-10)
            monotone=bool(np.all(np.diff(abs(errors))<=tol)); sign_changes=int(np.sum(diffs[:-1]*diffs[1:] < -tol**2))
            # Descriptive definitions, not asymptotic convergence certificates.
            late_idx=[ks.index(k) for k in [177,179,180] if k in ks]
            late=vals[late_idx] if len(late_idx)==3 else None
            limit=.5 if name=='f7_8_MHz' else 1 if name.endswith('MHz') else abs(vals[-1])*(.05 if name.startswith('path') else .03)
            plateau=None if late is None else bool(np.ptp(late)<=max(limit,1e-12))
            shrinking=None if late is None else bool(abs(late[2]-late[1])<=abs(late[1]-late[0])+tol)
            slope_shrinking=None if late is None else bool(abs(late[2]-late[1])<=abs(late[1]-late[0])/2+tol)
            trends.append(dict(grid=grid,family=family,observable=name,ks=ks,values=vals,endpoint_differences=errors,
                adjacent_increments=diffs,increment_per_added_state=diffs/np.diff(ks),monotonic_endpoint_approach=monotone,
                step_direction_reversals=sign_changes,trend='monotonic_endpoint_approach' if monotone else 'nonmonotonic',
                largest_step_from=ks[int(np.argmax(abs(diffs)))],largest_step_to=ks[int(np.argmax(abs(diffs)))+1],
                late_plateau_at_descriptive_strict_tolerance=plateau,late_increments_shrinking=shrinking,late_slopes_shrinking=slope_shrinking))
    table('state_tracking.csv',state_rows);table('ambiguous_clusters.csv',cluster_rows);table('operator_convergence.csv',op_rows)
    table('low_energy_convergence.csv',low_rows);table('pathway_convergence.csv',path_rows);table('convergence_trends.csv',trends)
    table('case_inventory.csv',case_rows);table('adjacent_state_tracking.csv',adjacent_rows)
    table('adjacent_operator_blocks.csv',adjacent_ops);save('operator_blocks.json',block_details)
    spatial=read(OLD/'N51_to_N71_low11_tracking.json')
    save('spatial_saved_evidence.json',dict(source='subspace_tracking/N51_to_N71_low11_tracking.json',
        rows=spatial['tracking']['rows'],note='Existing common-grid physical interpolation comparison reused; no new spatial eigensolve.'))
    counterexamples=[r for r in op_rows if r['object_type']=='individual_edge_magnitude' and r.get('relative_to_endpoint') is not None
        and r['endpoint_magnitude']>=.01 and abs(r['relative_to_endpoint'])>.03
        and (r['minimum_endpoint_state_amplitude']>.999 or abs(r['carrier_error_MHz'])<1)]
    save('summary.json',dict(no_new_eigensolves=True,tracker=str((OLD/'tracking.py').relative_to(ROOT)),
        whitening='SVD-only symmetric Gram whitening adapter; unchanged Hungarian/ambiguity/subspace/alignment logic',
        endpoint='same-grid k180, not ground truth',input_sha256=INPUTS,case_inventory=case_rows,
        counts=dict(state_rows=len(state_rows),automatic_clusters=sum(r['origin']=='automatic' for r in cluster_rows),
            block_comparisons=len(block_details),low_rows=len(low_rows),path_rows=len(path_rows)),
        counterexamples=counterexamples,adjacent_cycle_inconsistencies=[r for r in adjacent_rows if r['cycle_inconsistent']],
        numerical_tolerances=dict(relative_norm_floor=1e-8,material_edge_relative=.03,material_edge_minimum_magnitude=.01),
        coverage_limit='N51 reduced/common-priority vectors and signed matrices unavailable; their scalar summaries are not a new full subspace audit.',
        excluded=['k220 mock fixtures','older physically different checkpoint families'],
        units='energies GHz, ordinary transition frequencies MHz, grid_phi in saved dimensionless phase convention'))
    print(json.dumps(dict(cases=len(cases),state_rows=len(state_rows),clusters=len(cluster_rows),operator_rows=len(op_rows),counterexamples=len(counterexamples))))


if __name__ == '__main__': main()
