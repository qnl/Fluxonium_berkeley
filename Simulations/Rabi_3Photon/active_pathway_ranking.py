"""Three-photon grid_phi pathway ranking at gridium_params.ACTIVE_PHYSICS.

For each case, solves Gridium4Mode, takes |<i|grid_phi|j>| over the lowest
NLEV states and ranks every 0 -> a -> b -> 1 path by the product of its three
edge magnitudes.  Also reports the direct 0-1 element and the frequency of
each edge, so frequency collisions (for example a middle tone near f01) are
visible.

Cases:
    sym      ACTIVE_PHYSICS (symmetric), default cutoffs
    asym     ACTIVE_PHYSICS with eps_J=0.10, eps_LK=0.05, default cutoffs
    sym_chk  ACTIVE_PHYSICS (symmetric), combined ACTIVE_CUTOFFS_CHECK

    python -m Simulations.Rabi_3Photon.active_pathway_ranking [--cases ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode
from Simulations.Rabi_3Photon import gridium_params as gp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Figures' / 'Rabi_3Photon'
NLEV = 16
TOP = 8

CASES = {
    'sym': ('symmetric, default cutoffs', gp.physics(), gp.cutoffs('default')),
    'asym': ('eps_J=0.10 eps_LK=0.05, default cutoffs',
             gp.physics(eps_J=0.10, eps_LK=0.05), gp.cutoffs('default')),
    'sym_chk': ('symmetric, CHECK cutoffs', gp.physics(),
                gp.cutoffs('check')),
}


def rank_paths(G, freqs_MHz, top=TOP):
    n = G.shape[0]
    rows = []
    for a, b in itertools.product(range(2, n), repeat=2):
        if a == b:
            continue
        edges = (G[0, a], G[a, b], G[b, 1])
        score = float(np.prod(edges))
        if score <= 1e-12:
            continue
        rows.append(dict(
            path=[0, a, b, 1], score=score,
            edges=[float(x) for x in edges],
            edge_freqs_MHz=[float(abs(freqs_MHz[a] - freqs_MHz[0])),
                            float(abs(freqs_MHz[b] - freqs_MHz[a])),
                            float(abs(freqs_MHz[1] - freqs_MHz[b]))],
        ))
    rows.sort(key=lambda r: r['score'], reverse=True)
    return rows[:top]


def run_case(key):
    label, phys, cut = CASES[key]
    t0 = time.perf_counter()
    m = Gridium4Mode(**phys, **cut, nlev=NLEV,
                     stage1_permc_spec='MMD_AT_PLUS_A')
    e = np.asarray(m.levels(), dtype=float)
    G = np.abs(np.asarray(m.grid_phi().full()))
    f = (e - e[0]) * 1e3
    paths = rank_paths(G, f)
    return dict(
        key=key, label=label, physics=phys, cutoffs=cut,
        seconds=time.perf_counter() - t0,
        transitions_MHz=f.tolist(),
        abs_G01=float(G[0, 1]),
        f01_MHz=float(f[1]),
        grid_phi_abs=G.tolist(),
        top_paths=paths,
    )


def print_case(r):
    print('=' * 86)
    print('%s   (%.0f s)' % (r['label'], r['seconds']))
    f = np.array(r['transitions_MHz'])
    print('  levels (MHz from 0): ' + ', '.join(
        '%d:%.1f' % (i, x) for i, x in enumerate(f)))
    print('  direct |<0|G|1>| = %.5f at f01 = %.4f MHz'
          % (r['abs_G01'], r['f01_MHz']))
    print('  rank  path            score     edges (|G|)                 '
          'edge freqs (MHz)')
    for i, p in enumerate(r['top_paths'], 1):
        print('  %2d   %-14s  %8.4f  %-26s  %s'
              % (i, '->'.join(map(str, p['path'])), p['score'],
                 ', '.join('%.3f' % x for x in p['edges']),
                 ', '.join('%.2f' % x for x in p['edge_freqs_MHz'])))
    #ratio of the best three-photon product to the direct element, as a
    #crude measure of how much the direct path competes
    if r['top_paths']:
        best = r['top_paths'][0]['score']
        print('  best path score / direct |G01| = %.3g'
              % (best / max(r['abs_G01'], 1e-300)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', default=['sym', 'asym', 'sym_chk'],
                   choices=sorted(CASES))
    args = p.parse_args(argv)

    results = []
    for key in args.cases:
        r = run_case(key)
        results.append(r)
        print_case(r)
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'active_pathway_ranking.json').open('w') as fh:
            json.dump(results, fh, indent=2)
    print('Wrote %s' % (OUT / 'active_pathway_ranking.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
