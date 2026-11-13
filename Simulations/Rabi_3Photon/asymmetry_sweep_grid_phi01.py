"""Sweep KITE asymmetry and record the logical ``grid_phi`` matrix element.

Scientific question
-------------------
At the benchmark point ``eps_J=0.10``, ``eps_LK=0.05`` the four-mode
``grid_phi`` operator has a large intra-doublet element
``|<0|grid_phi|1>| ~ 0.5`` (see
``Figures/Rabi_3Photon/four_mode_grid_phi_matrix_reduced_diagnostic.csv``).
Naive perturbative intuition says a small asymmetry should leave that element
near zero, because the symmetric model forbids a single-photon logical
transition.  Both behaviours are possible:

* If the symmetric logical splitting ``Delta_0`` is large compared with the
  asymmetry-induced intra-doublet coupling ``V ~ eps``, then the element grows
  linearly in ``eps`` and stays small.
* If the protected point is (near-)degenerate, ``Delta_0 -> 0``, then even a
  tiny ``eps`` fully mixes the doublet.  The element saturates at O(1) while
  the splitting ``f01`` grows linearly in ``eps``.

This script measures which regime the netlist-derived model is in by sweeping
the asymmetries down toward zero and recording ``f01`` and
``|<0|grid_phi|1>|``.

It is a bounded diagnostic at deliberately small cutoffs.  It is NOT a
convergence study, NOT a gate result, and the absolute numbers inherit the
cutoff error of the reduced basis.  Only the *trend* versus ``eps`` is the
intended output.  The default cutoffs match the reduced diagnostic used by
``four_mode_control_audit.py`` so the ``eps_J=0.10`` end point can be compared
with the existing tracked artifact.

Usage
-----
Fast default sweep (reduced cutoffs, scales both asymmetries together)::

    python -m Simulations.Rabi_3Photon.asymmetry_sweep_grid_phi01

Sweep only the junction asymmetry, holding ``eps_LK`` fixed at its benchmark
value::

    python -m Simulations.Rabi_3Photon.asymmetry_sweep_grid_phi01 --mode eps_J

Larger cutoffs to check that the trend is not a reduced-basis artifact::

    python -m Simulations.Rabi_3Photon.asymmetry_sweep_grid_phi01 \
        --N2 41 --N3 41 --N4 6 --nkeep 160
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode
from Simulations.Rabi_3Photon import gridium_params as gp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'Figures' / 'Rabi_3Photon'

#active physical point (paper regime d, eps_K=2.5, eps_p=5.5) minus the
#asymmetries, which are what this script sweeps
BENCHMARK_PHYSICS = {k: v for k, v in gp.ACTIVE_PHYSICS.items()
                     if k not in ('eps_J', 'eps_LK')}

#asymmetry scale at sweep factor 1.0.  The active point is symmetric, so this
#is the upper end of the paper's fabrication range (SI Note 12: 1-15%), the
#same values earlier work used as its fixed benchmark.
BENCHMARK_EPS_J = 0.10
BENCHMARK_EPS_LK = 0.05

#exploration cutoffs for the active regime
REDUCED_CUTOFFS = gp.cutoffs('fast')

#asymmetry scale factors, spanning three decades so linear growth and
#saturation are easy to tell apart
DEFAULT_SCALES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0)

#number of low-energy states to request; enough to also see f78 and the
#0-7-8-1 pathway edges
NLEV = 12


def _eps_pair(scale, mode):
    """Return the ``(eps_J, eps_LK)`` pair for one sweep point.

    ``scale`` multiplies the benchmark asymmetry, so ``scale=1.0`` reproduces
    the benchmark point and ``scale=0.0`` is the exactly symmetric model.
    """
    if mode == 'both':
        return BENCHMARK_EPS_J * scale, BENCHMARK_EPS_LK * scale
    if mode == 'eps_J':
        return BENCHMARK_EPS_J * scale, BENCHMARK_EPS_LK
    if mode == 'eps_LK':
        return BENCHMARK_EPS_J, BENCHMARK_EPS_LK * scale
    raise ValueError('Unknown sweep mode: %s' % mode)


def _sweep_point(scale, mode, cutoffs):
    """Solve one asymmetry point and extract the logical observables."""
    eps_J, eps_LK = _eps_pair(scale, mode)
    model = Gridium4Mode(**BENCHMARK_PHYSICS, eps_J=eps_J, eps_LK=eps_LK,
                         **cutoffs, nlev=NLEV)

    started = time.perf_counter()
    energies = np.asarray(model.levels(), dtype=float)
    matrix = np.abs(np.asarray(model.grid_phi().full()))
    elapsed = time.perf_counter() - started

    #GHz -> MHz for the small intra-doublet splittings
    f01_MHz = (energies[1] - energies[0]) * 1e3
    f78_MHz = (energies[8] - energies[7]) * 1e3

    return {
        'scale': scale,
        'eps_J': eps_J,
        'eps_LK': eps_LK,
        #the quantity under test: single-photon logical coupling
        'abs_grid_phi_01': matrix[0, 1],
        'f01_MHz': f01_MHz,
        #the middle tone of the 0-7-8-1 pathway, for the collision check
        'f78_MHz': f78_MHz,
        'f78_minus_f01_MHz': f78_MHz - f01_MHz,
        #primary pathway edges, to see whether the three-photon route also
        #depends on the asymmetry
        'abs_grid_phi_07': matrix[0, 7],
        'abs_grid_phi_78': matrix[7, 8],
        'abs_grid_phi_81': matrix[8, 1],
        #the historical one-mode ordering 0->5->4->1, for comparison
        'abs_grid_phi_05': matrix[0, 5],
        'abs_grid_phi_54': matrix[5, 4],
        'abs_grid_phi_41': matrix[4, 1],
        'f07_MHz': (energies[7] - energies[0]) * 1e3,
        'seconds': elapsed,
    }


def _ratio_table(rows):
    """Compare measured growth with exact linear-in-``eps`` growth.

    For each nonzero scale this reports the element divided by the scale,
    normalized to the benchmark point.  Constant values near 1.0 mean linear
    growth (the perturbative expectation).  Values that blow up as the scale
    decreases mean saturation (near-degenerate mixing).
    """
    benchmark = next((r for r in rows if r['scale'] == 1.0), None)
    if benchmark is None or benchmark['abs_grid_phi_01'] == 0.0:
        return []

    table = []
    for row in rows:
        if row['scale'] <= 0.0:
            continue
        linear_prediction = benchmark['abs_grid_phi_01'] * row['scale']
        table.append({
            'scale': row['scale'],
            'abs_grid_phi_01': row['abs_grid_phi_01'],
            'linear_prediction': linear_prediction,
            #>> 1 means the element is far larger than linear scaling allows
            'measured_over_linear': row['abs_grid_phi_01'] / linear_prediction,
        })
    return table


def _classify(rows):
    """Apply the three pre-stated outcomes to the measured sweep."""
    symmetric = next((r for r in rows if r['scale'] == 0.0), None)
    smallest = min((r for r in rows if r['scale'] > 0.0),
                   key=lambda r: r['scale'], default=None)
    benchmark = next((r for r in rows if r['scale'] == 1.0), None)

    if symmetric is None or smallest is None or benchmark is None:
        return 'INCOMPLETE: sweep is missing the symmetric, smallest, or benchmark point.'

    #a nonzero element at eps=0 means the asymmetry is not what enables the
    #transition.  four_mode_parity_check.py shows why: inversion parity is
    #exact at this bias (and is NOT broken by eps_J/eps_LK), grid_phi is
    #parity-odd, and logical states 0/1 are a (+,-) parity doublet, so the
    #direct element is symmetry-allowed.
    if symmetric['abs_grid_phi_01'] > 1e-6:
        return (
            'DIRECT LOGICAL COUPLING PRESENT AT eps=0: '
            '|<0|grid_phi|1>| = %.6g in the exactly symmetric model, so the '
            'asymmetry does not create it. States 0/1 are an opposite-parity '
            'doublet under exact inversion, which allows a parity-odd '
            'grid_phi element; see four_mode_parity_check.py.'
            % symmetric['abs_grid_phi_01']
        )

    ratios = _ratio_table(rows)
    worst = max(ratios, key=lambda r: r['measured_over_linear'], default=None)
    if worst is None:
        return 'INCOMPLETE: no nonzero-scale points to compare.'

    #saturation: at small eps the element vastly exceeds linear scaling
    if worst['measured_over_linear'] > 10.0:
        return (
            'NEAR-DEGENERATE MIXING: at scale %.4g the element is %.1fx larger '
            'than linear-in-eps scaling predicts. Even a small asymmetry fully '
            'mixes the logical doublet, so the single-photon logical coupling '
            'is not suppressed by small eps.'
            % (worst['scale'], worst['measured_over_linear'])
        )

    return (
        'PERTURBATIVE: the element tracks linear-in-eps scaling within %.1fx '
        'across the sweep. The large benchmark element follows from eps_J=0.10 '
        'being a large asymmetry, matching the original intuition.'
        % worst['measured_over_linear']
    )


def sweep(mode='both', scales=DEFAULT_SCALES, cutoffs=None,
          output_dir=DEFAULT_OUTPUT, write_outputs=True):
    """Run the asymmetry sweep and return its rows plus a verdict."""
    cutoffs = dict(REDUCED_CUTOFFS) if cutoffs is None else dict(cutoffs)
    rows = []

    for scale in scales:
        row = _sweep_point(scale, mode, cutoffs)
        rows.append(row)
        print('scale=%-7.4g eps_J=%-7.4g eps_LK=%-7.4g '
              '|grid_phi_01|=%-12.6g f01=%-12.6g MHz  (%.1f s)'
              % (row['scale'], row['eps_J'], row['eps_LK'],
                 row['abs_grid_phi_01'], row['f01_MHz'], row['seconds']),
              flush=True)

    verdict = _classify(rows)

    if write_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = 'asymmetry_sweep_grid_phi01_%s_reduced_diagnostic' % mode

        csv_path = output_dir / ('%s.csv' % stem)
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        json_path = output_dir / ('%s.json' % stem)
        with json_path.open('w') as stream:
            json.dump({
                'question': ('Does |<0|grid_phi|1>| scale linearly in the KITE '
                             'asymmetry, or saturate at O(1)?'),
                'sweep_mode': mode,
                'physical_point': {
                    key: value for key, value in BENCHMARK_PHYSICS.items()
                },
                'benchmark_eps_J': BENCHMARK_EPS_J,
                'benchmark_eps_LK': BENCHMARK_EPS_LK,
                'cutoffs': cutoffs,
                'nlev': NLEV,
                'rows': rows,
                'linear_scaling_comparison': _ratio_table(rows),
                'verdict': verdict,
                'limitations': [
                    'Reduced cutoffs: absolute values carry basis-truncation error.',
                    'Not a convergence study and not a gate result.',
                    'States 7/8 are known to be unconverged in retained basis.',
                    'Only the trend versus eps is the intended conclusion.',
                ],
            }, stream, indent=2)

        print('\nWrote %s' % csv_path.relative_to(ROOT))
        print('Wrote %s' % json_path.relative_to(ROOT))

    return rows, verdict


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Sweep KITE asymmetry and record |<0|grid_phi|1>| and f01.')
    parser.add_argument('--mode', choices=('both', 'eps_J', 'eps_LK'),
                        default='both',
                        help='Scale both asymmetries together (default), or '
                             'scale one while holding the other at benchmark.')
    parser.add_argument('--scales', type=float, nargs='+', default=None,
                        help='Asymmetry scale factors; 1.0 is the benchmark '
                             'point and 0.0 is the symmetric model.')
    parser.add_argument('--n1max', type=int, default=REDUCED_CUTOFFS['n1max'])
    parser.add_argument('--N2', type=int, default=REDUCED_CUTOFFS['N2'])
    parser.add_argument('--N3', type=int, default=REDUCED_CUTOFFS['N3'])
    parser.add_argument('--L2', type=float, default=REDUCED_CUTOFFS['L2'])
    parser.add_argument('--L3', type=float, default=REDUCED_CUTOFFS['L3'])
    parser.add_argument('--N4', type=int, default=REDUCED_CUTOFFS['N4'])
    parser.add_argument('--nkeep', type=int, default=REDUCED_CUTOFFS['nkeep'])
    parser.add_argument('--no-write', action='store_true',
                        help='Print results without writing artifacts.')
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse_arguments(argv)
    cutoffs = dict(n1max=arguments.n1max, N2=arguments.N2, L2=arguments.L2,
                   N3=arguments.N3, L3=arguments.L3, N4=arguments.N4,
                   nkeep=arguments.nkeep)
    scales = (DEFAULT_SCALES if arguments.scales is None
              else tuple(arguments.scales))

    print('Asymmetry sweep of the logical grid_phi matrix element')
    print('mode=%s  cutoffs=%s  nlev=%d\n' % (arguments.mode, cutoffs, NLEV))

    rows, verdict = sweep(mode=arguments.mode, scales=scales, cutoffs=cutoffs,
                          write_outputs=not arguments.no_write)

    print('\nLinear-scaling comparison (measured / linear-in-eps prediction):')
    for entry in _ratio_table(rows):
        print('  scale=%-7.4g measured=%-12.6g linear=%-12.6g ratio=%.3g'
              % (entry['scale'], entry['abs_grid_phi_01'],
                 entry['linear_prediction'], entry['measured_over_linear']))

    print('\nVERDICT: %s' % verdict)
    return rows, verdict


if __name__ == '__main__':
    main()
