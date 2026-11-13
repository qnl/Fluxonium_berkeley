"""Cutoff convergence check for gridium_params.ACTIVE_PHYSICS.

Solves the active point (paper regime d + eps_K=2.5, eps_p=5.5, symmetric) at
a ladder of cutoffs and reports low transition frequencies, f01, and the
grid_phi elements that matter for control.  Each rung changes one cutoff
family relative to the previous one, so the source of any change is clear.

    python -m Simulations.Rabi_3Photon.active_cutoff_check [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode
from Simulations.Rabi_3Photon import gridium_params as gp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Figures' / 'Rabi_3Photon'
NLEV = 12

LADDER = [
    ('fast',               gp.cutoffs('fast')),
    ('fast nkeep120',      gp.cutoffs('fast', nkeep=120)),
    ('fast nkeep120 N4=6', gp.cutoffs('fast', nkeep=120, N4=6)),
    ('default',            gp.cutoffs('default')),
    ('default nkeep160',   gp.cutoffs('default', nkeep=160)),
    ('default N4=8',       gp.cutoffs('default', N4=8)),
    ('default n1max=5',    gp.cutoffs('default', n1max=5)),
    ('wider L, N=101',     gp.cutoffs('default', N2=101, L2=26.0,
                                      N3=101, L3=30.0)),
]


def solve(cut):
    t0 = time.perf_counter()
    m = Gridium4Mode(**gp.ACTIVE_PHYSICS, **cut, nlev=NLEV,
                     stage1_permc_spec='MMD_AT_PLUS_A')
    e = np.asarray(m.levels(), dtype=float)
    G = np.abs(np.asarray(m.grid_phi().full()))
    return dict(
        seconds=time.perf_counter() - t0,
        transitions_MHz=((e - e[0]) * 1e3).tolist(),
        f01_MHz=float((e[1] - e[0]) * 1e3),
        abs_G01=float(G[0, 1]),
        #largest couplings out of |0> and |1>, labelled by target
        top_from_0=sorted(((float(G[0, j]), j) for j in range(2, NLEV)),
                          reverse=True)[:4],
        top_from_1=sorted(((float(G[1, j]), j) for j in range(2, NLEV)),
                          reverse=True)[:4],
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='Only the first four rungs.')
    args = p.parse_args(argv)
    ladder = LADDER[:4] if args.quick else LADDER

    rows = []
    for label, cut in ladder:
        r = solve(cut)
        r.update(label=label, cutoffs=cut)
        rows.append(r)
        tr = r['transitions_MHz']
        print('%-20s f01=%9.4f MHz |G01|=%.5f  f02..f05=%s MHz  (%.0f s)'
              % (label, r['f01_MHz'], r['abs_G01'],
                 np.array2string(np.array(tr[2:6]), precision=2),
                 r['seconds']), flush=True)
        print('%-20s   top |<0|G|j>|: %s'
              % ('', ', '.join('%d:%.3f' % (j, v) for v, j in r['top_from_0'])))
        print('%-20s   top |<1|G|j>|: %s'
              % ('', ', '.join('%d:%.3f' % (j, v) for v, j in r['top_from_1'])),
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'active_cutoff_check.json'
    with path.open('w') as f:
        json.dump({'physics': gp.ACTIVE_PHYSICS, 'rows': rows}, f, indent=2)
    print('Wrote %s' % path.relative_to(ROOT))


if __name__ == '__main__':
    main()
