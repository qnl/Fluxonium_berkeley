# Asymmetric four-mode Gridium model-threat audit

## Scope

This audit uses the paper Supplementary Notes 4, 5, 6, and 11; the repository
implementation notes; `Circuit_Objs/qchard_gridium_netlist.py`; and the
authenticated N=71 k=180 checkpoint.  It does not solve a modified Stage-1
Hamiltonian.  Quantitative results below are first-order responses projected
into the saved 180 x 8 Stage-2 space.

## Implemented Hamiltonian

The code constructs

\[
H_C=4\sum_{ij}(E_C^{\rm mat})_{ij}n_i n_j,
\qquad E_C^{\rm mat}=(M^T C_{\rm node}M)^{-1},
\]

with

\[
C_{\rm node}=\frac{e_0e_0^T+e_1e_1^T}{E_C}
+\frac{e_2e_2^T}{E_{CS}}
+\frac{(e_0-e_1)(e_0-e_1)^T}{e_C}
+\frac{e_3e_3^T}{e_P}.
\]

Here `eC` and `eP` are energy-like inverse-capacitance parameters corresponding
to the paper's cross-KITE and cross-circuit capacitances.  At the current
point, in the code's inverse-energy capacitance units,

```text
Cnode = [[ 2.181818, -0.181818, 0,     0    ],
         [-0.181818,  2.181818, 0,     0    ],
         [ 0,          0,        0.125, 0    ],
         [ 0,          0,        0,     0.100]]
```

and

```text
ECmat = [[ 0.461538, -0.423077, -0.038462, -0.461538],
         [-0.423077,  0.846154, -0.423077,  0.423077],
         [-0.038462, -0.423077,  8.461538,  0.038462],
         [-0.461538,  0.423077,  0.038462, 10.461538]].
```

Thus changing any physical capacitor generally changes several diagonal and
off-diagonal charge terms after inversion; it is not only a single-mode
frequency shift.

The Josephson terms are

\[
-E_J(1-\epsilon_J)\cos\theta_1
-E_J(1+\epsilon_J)\cos(\theta_1+\theta_2+\vartheta_{\rm ext})
-E_{JS}\cos(\theta_1+\theta_2+\theta_3+\phi_{\rm ext}).
\]

The inductive part is `0.5*x.T*K*x`, with

\[
K=E_{LK}(1-\epsilon_{LK})v_1v_1^T
 +E_{LK}(1+\epsilon_{LK})v_2v_2^T+E_Lv_3v_3^T,
\]

where `x=(theta2,theta3,theta4)`, `v1=(0,0,-1)`,
`v2=(1,0,-1)`, and `v3=(1,1,-1)`.  The baseline matrix is

```text
K = [[ 2.05, 1.00, -2.05],
     [ 1.00, 1.00, -1.00],
     [-2.05,-1.00,  3.00]].
```

## Present-term audit

| Present term | Physical origin and modes | Expected effect |
|---|---|---|
| `EC` | Equal capacitances of the two KITE junction branches; enters the full inverse capacitance matrix | Energies, charge-mode mixing, and all `grid_phi` elements. State 8 is more responsive than states 2/3 at the local point. |
| `ECS` | Capacitance of the S/QPS junction branch | Strong energy and mixing threat for state 8: +1% changes `f78` by -4.15 MHz and the two outer leading-path carriers by about 24-28 MHz. |
| `eC` | Cross-KITE capacitance between the outer pads (`epsilon_K`-like parameter) | Renormalizes differential-mode charging and every transformed charge coupling. Local +1% effect is small, but the paper-supported 2.5-to-5.5 GHz span is not a local perturbation and can materially shift `f78`. |
| `eP` | Cross-circuit/node-to-ground stray capacitance creating the fourth mode (`epsilon_p`-like parameter) | Changes mode-4 frequency and charge hybridization. States 7/8 have mean mode-4 occupations about 0.40, but the local +1% response is modest. |
| `EJ, eps_J` | Mean and imbalance of the two KITE Josephson junctions; couples compact `theta1` to `theta2` | Changes energies, wavefunctions, selection-rule suppression, and `grid_phi`. `eps_J` is the largest local coupling threat to the leading path. |
| `EJS` | S/QPS junction Josephson term involving `theta1+theta2+theta3` | Shifts and mixes extended/global-phase excitations; affects energies and all path edges. |
| `ELK, eps_LK` | Mean and imbalance of the two KITE inductors; couples `theta2` and parasitic `theta4` | Energies and mixing. The paper warns sufficiently large asymmetry can hybridize slow and fast modes. +0.01 `eps_LK` moves the leading outer carriers by about 13 MHz. |
| `EL` | S-branch/superinductor energy; couples `theta2+theta3` to `theta4` | Energies, phase localization, mode mixing, and `grid_phi`. +1% moves leading outer carriers by about 14-15 MHz. |
| `ng` | Offset charge on the compact island | First-order energy and `|grid_phi|` slopes vanish at the chosen symmetric point, but the first-order state rotation is larger for 7/8 than 2/3/6. Second-order response is required. |
| `phi_ext`, `theta_ext` | External flux biases in the S junction and second KITE junction terms | First-order scalar slopes vanish at the protected point, while near-doublet eigenvectors rotate strongly. They require degenerate/subspace tracking rather than naive statewise derivatives. |

## Source-supported omitted or incompletely specified threats

| Priority | Threat | Source grounding | Relevance |
|---:|---|---|---|
| 1 | Distributed superinductor/array modes | Supplementary Notes 6 and 11 give array self-capacitance, ground capacitance, Eq. S33, and examples of array modes near 4.83 GHz and 5.5-6.5 GHz. The present Hamiltonian uses lumped inductors and one `Cp` node mode whose bare frequency is 15.845 GHz. | Very high. States 7/8 lie at 6.275/6.298 GHz, directly in a paper-identified array-mode band. A missing near-resonant mode can cause avoided crossings, change state identity, and reorder pathways. |
| 2 | Residual capacitance-matrix entries from the real electrode geometry | The paper says stray electric fields are unavoidable and extracts pair capacitances with HFSS. The code retains only equal junction caps, one outer-pad cross-cap, and one node-ground cap. | High. Extra entries change the entire inverse capacitance matrix and can alter energies, mixing, selection rules, and all path elements. |
| 3 | KITE junction-capacitance asymmetry | The two KITE junction capacitances are equal in the code, while the same physical junctions have source-supported Josephson asymmetry. Capacitance asymmetry is physically plausible but has no numerical tolerance in the paper. | High for selection rules/path ordering because it breaks the kinetic symmetry independently of `eps_J`. |
| 4 | Uncertainty in `eC` and `eP` | Both capacitors are explicitly in S20/S30. Paper examples/design extraction give approximately `epsilon_K/h=2.45-2.5 GHz` and `epsilon_p/h=5.43-5.5 GHz`; other fitted devices differ. | High for quantitative higher-level energies; locally modest for leading-path edges, but broad source-supported changes require full solves. |
| 5 | Exact combined four-mode-plus-asymmetry model | The paper prints S20/S30 and S31/S32 separately, not one combined equation. The repository implementation is a tested netlist construction, but author confirmation of the combined physical model remains absent. | Epistemic high: implementation consistency does not prove that the chosen combined model is the intended experimental model. |
| 6 | Physical microwave-port operator | `grid_phi=theta3+theta2/2` is an abstract source-derived coordinate, not a calibrated line coupling. | Does not change static energies, but can invalidate coupling-product rankings if the real port couples to a different combination. |

No additional mutual inductance or nonlinear element is proposed: the sources
do not define one.  Coherent phase slips in the junction arrays are discussed
as a decoherence mechanism, but cannot be added to this Hamiltonian without a
separate sourced reduction.

### Ranked relevance to requested observables

`VH/H/M/L` mean very high/high/medium/low expected relevance.  These are
physics threat categories, not a combined numerical score.

| Rank | Threat | A: states 7/8 | B: `f78` | C: `g07/g78/g81` | D: alternative paths | E: ordering |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Omitted distributed array mode near 5-7 GHz | VH | VH | VH | H | VH |
| 2 | Residual/full capacitance matrix, including KITE capacitance asymmetry | H | H | H | H | H |
| 3 | `eps_J` uncertainty | H, especially state 8 | H | VH | H | H |
| 4 | Broad `eC/eP` uncertainty | H | H | M locally, potentially H broadly | H | H |
| 5 | Fitted `EJ/ECS/EC/EL/ELK/EJS` uncertainty | H | H | M-H | H | M-H |
| 6 | `eps_LK` uncertainty | M-H | M-H | M | M-H | M |
| 7 | Unknown physical port operator | None for static states | None | VH | VH | VH |

## Explicit capacitance modifications

For any added capacitor between existing nodes `a,b`, the source-consistent
linear capacitance update is

\[
\delta C_{\rm node}=c_{ab}(e_a-e_b)(e_a-e_b)^T,
\]

and an added node-to-ground capacitor is

\[
\delta C_{\rm node}=c_{ag}e_ae_a^T.
\]

The first-order inverse-matrix change is

\[
\delta E_C^{\rm mat}=-E_C^{\rm mat}
M^T\delta C_{\rm node}M E_C^{\rm mat},
\qquad
\delta H_C=4\sum_{ij}\delta(E_C^{\rm mat})_{ij}n_i n_j.
\]

KITE junction-capacitance asymmetry can be introduced before inversion as

\[
\frac{e_0e_0^T}{E_{C1}}+\frac{e_1e_1^T}{E_{C2}},
\quad C_{J1,2}=C_J(1\pm\delta_C).
\]

An array mode cannot be represented reliably by one arbitrary bilinear term.
It requires enlarging the circuit capacitance/inductance matrices using the
distributed array normal mode and its derived coupling to the Gridium nodes.

## First-order checkpoint susceptibilities

The following normalization is used: +1% for positive energy parameters and
+0.01 absolute for each asymmetry.

| Perturbation | `f78` change | Leading score | `0-3-8-1` score | `0-7-2-1` score | Control score |
|---|---:|---:|---:|---:|---:|
| `EJ` +1% | +4.38 MHz | +0.66% | +1.21% | +1.39% | +17.67% |
| `EC` +1% | +1.58 MHz | -0.30% | +1.47% | +1.90% | +8.63% |
| `EL` +1% | -1.76 MHz | -0.89% | -1.59% | -1.48% | -7.08% |
| `ELK` +1% | -1.72 MHz | -0.20% | -0.75% | -0.84% | -6.45% |
| `EJS` +1% | +1.65 MHz | +0.13% | -0.62% | +0.19% | +4.30% |
| `ECS` +1% | -4.15 MHz | +0.44% | +0.29% | -1.12% | -16.98% |
| `eC` +1% | +0.37 MHz | -0.05% | +0.15% | +0.18% | +1.15% |
| `eP` +1% | -0.12 MHz | +0.21% | -0.15% | -0.21% | -1.23% |
| `eps_J` +0.01 | +3.04 MHz | +3.82% | +3.18% | -2.02% | -26.87% |
| `eps_LK` +0.01 | -1.06 MHz | +0.16% | +1.59% | +1.22% | +5.13% |

For the leading path, `eps_J+0.01` changes `(g07,g78,g81)` by approximately
`(+0.007%, +1.694%, +2.119%)`.  It is therefore the clearest existing-parameter
threat to the coupling product.  `EJ` and `ECS` are the strongest 1%-normalized
threats to the already-small 7-8 splitting.

The lower-state control path is not automatically robust: its weak edges make
its relative score highly susceptible even though states 3/6 are less mixed.

## Pathway implications

- `0-7-8-1` remains locally stable under small `eC/eP` changes, but is exposed
  to an omitted array mode near 6.3 GHz, `eps_J`, and the fragile `f78` split.
- `0-3-8-1` removes state 7 and the `f78` middle tone, but retains state 8,
  which has the largest local state response for several parameter directions.
  It is not protected against the array-mode threat.
- `0-7-2-1` removes state 8 and `f78`.  Of the two alternatives it is the more
  plausible hedge against state-8-specific or array-hybridization problems,
  but state 7 remains at 6.275 GHz and its weak `g72/g21` edges can move.
- `0-3-6-1` uses cleaner states but its score is about 102 times weaker and its
  weak edges have large relative parameter susceptibility.  It remains a
  diagnostic control rather than a robust gate winner.

## Recommended future sweep

1. Sweep `eC/h = {2.5, 3.5, 5.5, 7.5, 10}` GHz.  The 2.5-GHz endpoint is
   directly paper-supported; the upper points approach the smaller-capacitance
   limit around the current 5.5-GHz baseline.
2. Sweep `eP/h = {5.5, 7.5, 10, 15, 20}` GHz for the current regime.  The
   5.5-GHz value is paper-supported and the upper points test node-mode
   decoupling.  Much lower fitted values belong to different parameter regimes
   and should not be transplanted without a complete device parameter set.
3. Sweep `|eps_J|` through `0, 0.025, 0.05, 0.075, 0.10`, including relative
   sign combinations with `eps_LK`.  Ten percent is the paper's explicit
   simulated range.
4. Sweep `|eps_LK|` over the same grid.  The present point is 0.05.

Before assigning numerical ranges to other capacitance-matrix entries, obtain
an EM-extracted Maxwell capacitance matrix or fabrication data.  A 1/5/10%
matrix perturbation can be used as a mathematical sensitivity study, but is
not source-justified as a physical tolerance.

Each sweep must track state/subspace overlaps; all path carrier frequencies;
each individual `grid_phi` edge and path product; `f78`; state 7/8 mode
participation; spectral crowding; direct `g01`; pathway ordering; and
retained-basis, mode-4, and any new array-mode cutoff convergence.

## Limitations

- First-order derivatives are local and cannot be extrapolated across the
  broad `eC/eP` ranges or through avoided crossings.
- The response omits Stage-1 ranks above 179; k=220 convergence is pending.
- Bias-point scalar derivatives vanish by symmetry while eigenvectors rotate;
  second-order or degenerate-subspace analysis is required.
- No saved operator describes an omitted array mode or a general residual
  capacitance entry, so those threats cannot be quantified from the checkpoint.
- Static `grid_phi` susceptibilities do not give pulse-fidelity susceptibility.
- A modified capacitance matrix, a distributed array mode, or a different
  combined four-mode-asymmetry derivation requires rebuilding and
  diagonalizing the modified Hamiltonian with state/subspace tracking.
