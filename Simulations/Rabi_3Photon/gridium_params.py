"""Shared Gridium4Mode parameter sets for active simulation scripts.

Source: Nguyen et al., "The superconducting grid-states qubit",
arXiv:2509.14656 (local copy: references/gridium_2509.14656v1.pdf).

ACTIVE  (``ACTIVE_PHYSICS``)
    Hardest parameter regime in the paper: Table S1 regime (d),
        EJ=10, EC=0.5, EL=ELK=0.2, EJS=4, ECS=8   (h*GHz).
    The paper describes it as "even higher inductance [that] implements the GKP
    Hamiltonian better, offering larger grid support in the phase basis".

    Capacitive imperfections, both included, as on the physical device:
        eC = eps_K = 2.5  cross-KITE capacitance between the two outer pads
        eP = eps_p = 5.5  stray capacitance on the superinductor/KITE node
    These are the Table S2 values.  Table S2 applies them one at a time; the
    HFSS design in SI Note 6 has both together, [EC, eps_p, eps_K] ~
    [0.75, 5.43, 2.45].
    In ``Circuit_Objs.qchard_gridium_netlist._coeffs``, ``eC`` adds 1/eC to the
    C(1,2) cross capacitance and ``eP`` adds 1/eP to the node-4 capacitance,
    using the same e^2/2C normalization as EC.  That is exactly eps_K and
    eps_p.

    Symmetric: eps_J = eps_LK = 0.
    Bias: the protected point (phi_ext, theta_ext) = (0, pi), ng = 0.

LEGACY  (``LEGACY_PHYSICS``)
    The parameters used for all four-mode work up to 2026-11-06: regime (a)
    with eC=5.5, eP=10, eps_J=0.10, eps_LK=0.05.  Kept so the frozen spatial71
    runner, the retained-basis/shell audits, their checkpoints and the netlist
    validation tests remain reproducible.  Those modules intentionally keep
    their own hard-coded copies and do NOT import from here.

Cutoffs
    Regime (d) needs wider extended-phase windows than regime (a): EL=0.2
    gives a much broader envelope.  Checked with
    ``Simulations/Rabi_3Photon/active_cutoff_check.py`` (results in
    ``Figures/Rabi_3Photon/active_cutoff_check.json``), changing one cutoff
    family per step from the default:

    * |<0|grid_phi|1>| = 0.152 to within ~1% under nkeep 120->160, N4 6->8,
      n1max 4->5, and N2=N3 81->101 with L2/L3 22/26->26/30.  Dominant
      couplings |<0|G|2>| ~ 2.96 and |<0|G|11>| ~ 1.22 to within ~0.3%.
    * f01 ~ 0.11-0.12 MHz (exponentially small; ~8% spread between rungs).
    * f02/f03 doublet ~ 2554-2556 MHz, stable to ~2 MHz.
    * f04/f05 doublet: N4 is the slow cutoff.  N4=4 -> 6 -> 8 gives about
      4421 -> 3960 -> 3915 MHz, so N4 >= 8 is needed for anything involving
      states 4/5, and N4=8 itself still needs checking.
    * N4=4 is NOT adequate: it also inflates |G01| to ~0.188.

    ``ACTIVE_CUTOFFS`` (N4=8) is the default for quantitative work.
    ``ACTIVE_CUTOFFS_FAST`` (N=61, N4=6, nkeep=120; ~3.5 min) agrees with the
    default to ~1% on |G01| and ~5 MHz on f02-f05; use it for exploration.
    ``ACTIVE_CUTOFFS_CHECK`` (N=101, L 26/30, N4=10, nkeep=160) was run on
    2026-11-09 (``active_pathway_ranking.py``; ~17 min, ~5 GB peak).  Against
    the default it gives |G01| 0.1513 -> 0.1507, f02 2554.3 -> 2551.7 MHz,
    f04 3914.9 -> 3909.0 MHz, f10 5512.9 -> 5510.4 MHz, and the same
    pathway ranking with path scores within ~1-4%.  The default is adequate
    for ranking and coupling work; f04/f05 still drift by ~6 MHz from N4=8 to
    N4=10.

Pathway ranking at ACTIVE_PHYSICS (default cutoffs, symmetric)
    Lowest levels (MHz from |0>): 1:0.12 | 2,3: 2554 | 4,5: 3915 |
    6,7: 5003-5005 | 8,9: 5071 | 10,11: 5513.
    Top 0->a->b->1 paths by |grid_phi| product:
      0->2->3->1    10.3  (edges 2.96, 1.17, 2.96; middle tone ~0.04 MHz)
      0->2->10->1    3.6  (2.96, 1.00, 1.22; tones 2554 / 2959 / 5513 MHz)
      0->11->3->1    3.6  (mirror of the above)
    Direct |<0|grid_phi|1>| = 0.151; best path / direct ~ 68.
    With eps_J=0.10, eps_LK=0.05: same top three, direct 0.075, ratio ~145.
    The top path's middle edge 2<->3 is an intra-doublet transition at
    ~40 kHz, close to f01 ~ 120 kHz, so it is the same kind of near-collision
    seen in the legacy 0->7->8->1 path.  0->2->10->1 avoids it.
"""

from __future__ import annotations

import numpy as np


#paper Table S1 regime (d): the hardest regime
REGIME_D = dict(EJ=10.0, EC=0.5, EL=0.2, ELK=0.2, EJS=4.0, ECS=8.0)

#paper Table S2 capacitive imperfections, applied together as on the device
PAPER_CAPACITANCES = dict(eC=2.5, eP=5.5)

PROTECTED_BIAS = dict(ng=0.0, phi_ext=0.0, theta_ext=float(np.pi))

SYMMETRIC = dict(eps_J=0.0, eps_LK=0.0)

#full constructor-ready physical parameter set for Gridium4Mode
ACTIVE_PHYSICS = dict(**REGIME_D, **PAPER_CAPACITANCES, **SYMMETRIC,
                      **PROTECTED_BIAS)

#numerical cutoffs for ACTIVE_PHYSICS
ACTIVE_CUTOFFS = dict(n1max=4, N2=81, L2=22.0, N3=81, L3=26.0, N4=8,
                      nkeep=120)
ACTIVE_CUTOFFS_CHECK = dict(n1max=4, N2=101, L2=26.0, N3=101, L3=30.0,
                            N4=10, nkeep=160)
ACTIVE_CUTOFFS_FAST = dict(n1max=4, N2=61, L2=20.0, N3=61, L3=24.0, N4=6,
                           nkeep=120)

#historical parameters (do not use for new work)
LEGACY_PHYSICS = dict(
    EJ=5.0, EC=0.5, EL=1.0, ELK=1.0, EJS=4.0, ECS=8.0,
    eC=5.5, eP=10.0, eps_J=0.10, eps_LK=0.05,
    ng=0.0, phi_ext=0.0, theta_ext=float(np.pi),
)
LEGACY_CUTOFFS = dict(n1max=4, N2=31, L2=11.0, N3=31, L3=14.0, N4=4,
                      nkeep=120)


def physics(**overrides):
    """Return a copy of ACTIVE_PHYSICS with optional overrides."""
    unknown = set(overrides) - set(ACTIVE_PHYSICS)
    if unknown:
        raise KeyError('Unknown Gridium4Mode parameter(s): %s'
                       % ', '.join(sorted(unknown)))
    return dict(ACTIVE_PHYSICS, **overrides)


def cutoffs(kind='default', **overrides):
    """Return a copy of a cutoff set ('default', 'check', 'fast')."""
    base = {'default': ACTIVE_CUTOFFS, 'check': ACTIVE_CUTOFFS_CHECK,
            'fast': ACTIVE_CUTOFFS_FAST}[kind]
    return dict(base, **overrides)


def circuit_only(params):
    """Strip the bias/asymmetry keys, leaving the 8 circuit energies."""
    keys = ('EJ', 'EC', 'EL', 'ELK', 'EJS', 'ECS', 'eC', 'eP')
    return {k: params[k] for k in keys}
