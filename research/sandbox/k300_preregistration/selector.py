"""Frozen feature-only selector. No Stage-1 solver and no full-parent solve."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import eigh

GROUPS = ((0, 1), (2, 3), (7, 8))
BUDGETS = (3, 6, 10, 15)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def serial(x):
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {'real': x.real.tolist(), 'imag': x.imag.tolist()}
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def write_new(path, data):
    # Exclusive creation: a failed freeze is inspected, never silently replaced.
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, default=serial, allow_nan=False)
        f.write('\n')


def indices(ranks):
    return (8 * np.asarray(ranks)[:, None] + np.arange(8)).ravel()


def build_hamiltonian(values, operators, ec, stiffness):
    """Existing Stage-2 formula, rank-major/mode4-minor ordering, GHz."""
    n = len(values)
    a4, b4 = 4 * ec[3, 3], .5 * stiffness[2, 2]
    f4 = (a4 / (4 * b4)) ** .25
    ladder = np.diag(np.sqrt(np.arange(1, 8)), 1)
    x4 = f4 * (ladder + ladder.T)
    p4 = 1j / (2 * f4) * (ladder.T - ladder)
    h = np.kron(np.diag(values), np.eye(8)).astype(complex)
    h += np.kron(np.eye(n), np.diag(2 * np.sqrt(a4 * b4) * (np.arange(8) + .5)))
    for j in range(3):
        h += 8 * ec[j, 3] * np.kron(operators[f'n{j+1}'], p4)
    h += stiffness[0, 2] * np.kron(operators['x2'], x4)
    h += stiffness[1, 2] * np.kron(operators['x3'], x4)
    if np.max(abs(h - h.conj().T)) > 1e-12:
        raise ValueError('Hamiltonian Hermiticity failed')
    return (h + h.conj().T) / 2, np.kron(.5 * operators['x2'] + operators['x3'], np.eye(8))


def importance(h, base_e, base_v, candidates):
    """Positive second-order block importance at each pair's mean energy.

    base_v contains only base-space rows; its 11 columns carry tracked labels.
    No exact candidate-space dressed information enters this function.
    """
    if base_v.ndim != 2 or base_v.shape[1] < max(max(g) for g in GROUPS) + 1:
        raise ValueError('Base eigenvectors do not contain all target labels')
    if len(base_e) != base_v.shape[1]:
        raise ValueError('Base energy/vector count mismatch')
    base_dim = base_v.shape[0]
    raw, details = [], []
    for r in candidates:
        ix = indices([r])
        lam, u = eigh(h[np.ix_(ix, ix)])  # only an isolated 8x8 block
        by_group = []
        row = {'rank': int(r), 'groups': []}
        for group in GROUPS:
            b = h[ix, :base_dim] @ base_v[:, group]
            z = u.conj().T @ b
            mu = float(np.mean(base_e[list(group)]))
            den = lam - mu
            # Positive off-resonant model; fail closed, do not clip denominators.
            eta = float(np.max(np.linalg.norm(z, axis=1) / abs(den)))
            if np.min(den) <= .010 or eta >= .10:
                raise ValueError(f'Off-resonant selector invalid: rank {r}, pair {group}')
            a = z.conj().T @ (z / den[:, None])
            a = (a + a.conj().T) / 2
            if np.min(np.linalg.eigvalsh(a)) < -1e-13:
                raise ValueError(f'Importance matrix is not positive semidefinite: rank {r}')
            by_group.append(a)
            row['groups'].append({'states': group, 'trace_GHz': float(np.trace(a).real),
                                  'coupling_norm_GHz': float(np.linalg.norm(b)),
                                  'minimum_denominator_GHz': float(min(den)), 'maximum_eta': eta})
        raw.append(by_group)
        details.append(row)
    return np.asarray(raw), details


def normalize(raw):
    """Whiten each two-dimensional target sector by its entire shell response."""
    result = np.empty_like(raw)
    floors = []
    for g in range(3):
        w = raw[:, g].sum(axis=0)
        lam, u = eigh((w + w.conj().T) / 2)
        if float(min(lam)) < -1e-12:
            raise ValueError('Shell importance is not positive semidefinite')
        floor = max(1e-14, 1e-10 * float(max(lam)))
        inv = (u * (1 / np.sqrt(np.maximum(lam, floor)))) @ u.conj().T
        result[:, g] = inv @ raw[:, g] @ inv
        floors.append({'eigenvalues_GHz': lam.tolist(), 'floor_GHz': floor})
    return result, floors


def objective(c):
    total = 0.0
    for a in c:
        eigenvalues = np.linalg.eigvalsh((a + a.conj().T) / 2)
        if float(min(eigenvalues)) < -1e-12:
            raise ValueError('Coverage matrix is not positive semidefinite')
        total += float(np.log1p(6 * np.maximum(eigenvalues, 0)).sum())
    return total / 6


def greedy(raw, ranks):
    normalized, floors = normalize(raw)
    selected, trace = [], []
    covered = np.zeros((3, 2, 2), complex)
    remaining = set(range(len(ranks)))
    while remaining:
        gains = {i: float(objective(covered + normalized[i]) - objective(covered)) for i in remaining}
        best_gain = max(gains.values())
        tied = [i for i in remaining if best_gain - gains[i] <= 1e-12]
        best = min(tied, key=lambda i: ranks[i])
        remaining.remove(best)
        selected.append(int(ranks[best]))
        covered += normalized[best]
        trace.append({'rank': int(ranks[best]), 'gain': gains[best],
                      'coverage_eigenvalues': [np.linalg.eigvalsh(c).tolist() for c in covered]})
    return {'order': selected, 'greedy_trace': trace, 'normalization': floors}


def spread_order(start):
    # Fixed nested farthest-point design on 20 equally spaced ranks.
    return [start + i for i in (0, 19, 9, 14, 4, 2, 6, 11, 16, 1, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18)]


def freeze(h, values, base_e, base_v, base, out, provenance):
    """Freeze BOTH shell selections before any validation-space diagonalization."""
    if len(values) != base + 40 or h.shape != (len(values)*8,)*2:
        raise ValueError('Expected exactly two held-out 20-rank shells')
    if base_v.shape != (base*8, 11) or np.asarray(base_e).shape != (11,):
        raise ValueError('Unexpected base eigensystem dimensions')
    # Refuse arbitrary rank identities at small Stage-1 gaps, including boundaries.
    if np.min(np.diff(values[base-1:])) <= 1e-6:
        raise ValueError('Near-degenerate candidate Stage-1 ranks: preregister a cluster method separately')
    payload = {'schema': 1, 'base': base, 'provenance': provenance, 'shells': {}, 'spaces': {}}
    for label, start in (('S1', base), ('S2', base+20)):
        ranks = list(range(start, start+20))
        raw, details = importance(h, base_e, base_v, ranks)
        result = greedy(raw, ranks)
        result['features'] = details
        result['raw_importance'] = raw
        payload['shells'][label] = result
        orders = {'physics': result['order'], 'prefix': ranks, 'spread': spread_order(start),
                  'spread_reverse': [2*start+19-r for r in spread_order(start)]}
        for method, order in orders.items():
            for b in BUDGETS:
                payload['spaces'][f'{label}_{method}_{b}'] = list(range(base)) + sorted(order[:b])
        payload['spaces'][f'{label}_full'] = list(range(base)) + ranks
    for method in ('physics', 'prefix', 'spread', 'spread_reverse'):
        for b in BUDGETS:
            additions = payload['spaces'][f'S1_{method}_{b}'][base:] + payload['spaces'][f'S2_{method}_{b}'][base:]
            payload['spaces'][f'union_{method}_{b}'] = list(range(base)) + sorted(additions)
    # Genuine global prefix, matched to union's total added count (not shell-balanced).
    for b in BUDGETS:
        payload['spaces'][f'union_global_prefix_{b}'] = list(range(base + 2*b))
    payload['spaces']['base'] = list(range(base))
    payload['spaces']['union_full'] = list(range(base+40))
    write_new(out, payload)
    write_new(str(out)+'.seal.json', {'selection_sha256': sha(out), 'provenance': provenance})
    return payload
