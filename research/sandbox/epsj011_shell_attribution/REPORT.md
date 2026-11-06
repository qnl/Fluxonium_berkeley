# Comparative shell-resolved Schur/Feshbach attribution at epsilon_J=0.11

## Required outcomes

**SCHUR RECONSTRUCTION: PASS**

**NEW-POINT SHELL MECHANISM: B — concentrated off-resonant virtual shell.**
For states 7 and 8, the leading three ranks carry 91.326% and 89.593% of
absolute exact shell self-energy; two ranks reach 80%, three/four ranks reach
90%, the near-resonant fraction is zero, maximum eta is 0.01777/0.01773, and
the shell probability is 0.001194/0.001182. Ranks 292 and 290 alone carry
84.417% of the six-state aggregate absolute rank attribution.

**CROSS-PARAMETER MECHANISM: SUPPORTED.**
Both epsilon_J=0.10 and 0.11 are resolved class B mechanisms. Overlap tracking
confirms the identity of every rank in the 280--299 shell. Ranks 292 and 290
remain dominant. The third aggregate contributor changes from rank 287 to
rank 289, but this is a quantitative redistribution inside the same
off-resonant mechanism, not a mechanism change.

**SELECTOR ATTRIBUTION: systematic coverage, quantitatively supported.**
For the final S2 shell, the frozen physics ordering captures 88.329%, 98.337%,
99.507%, and 99.776% of aggregate absolute exact Schur weight at budgets
3, 6, 10, and 15. Its top-five overlap with the exact ordering is 5/5.
For S1 the corresponding capture is 69.582%, 85.350%, 93.339%, and 97.730%,
with top-five overlap 4/5. No selector was recomputed or refit.

**UNION/b10 f78 EXCEPTION: B — cancellation of larger individual errors.**
The physics union has raw state-7/state-8 errors +0.254622/+0.475006 MHz and
therefore f78 error +0.220384 MHz. The ordinary global-prefix control has
raw errors +54.479177/+54.338596 MHz and therefore f78 error -0.140580 MHz.
The prefix's smaller differential error is produced by 0.1349% residual
after cancellation of individual errors 115--214 times larger than the
corresponding physics-union state errors. It is not better individual-state
or shell recovery. The
frozen selector verdict remains BORDERLINE.

**FREQUENCY SELECTIVITY:** the epsilon_J=0.10 k300 near-collision does not
persist at epsilon_J=0.11. At k300, f78-f01 changes from +0.021648 MHz to
+1.260224 MHz, while the sign remains positive. This is a static spectral
diagnostic only.

**NEXT RESEARCH ACTION:** preregister one authenticated N71 epsilon_J=0.11
larger-parent calculation with Stage-1 k340 and retain k300 as a nested
endpoint. This directly tests whether the dominant-rank and f78-cancellation
conclusions survive the still-failed retained-basis convergence before
spending effort on more parameter points or gate work.

**GATE WORK: HOLD**

**SPATIAL CONVERGENCE: HOLD**

## Scope, provenance, and recovery

This analysis used the already authenticated N71 k300 checkpoints at
epsilon_J=0.11 and epsilon_J=0.10. No Stage-1 eigensolve was run. The
epsilon_J=0.11 authentication remains PASS with configuration SHA-256
`8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646`,
checkpoint SHA-256
`20a58be10ad8408ce479de259e1d2f3562cc46222e0eebc939d6d695803516bd`,
and frozen-selection SHA-256
`17ec7b93304099399f152cb34e819fa4aedc5133d23bc3c04c3b7b1f6292a931`.
All checkpoint payload hashes and the selection seal pass.

The recovered `methodology.md` and `input_hashes.json` were frozen before
the new shell couplings, denominators, occupations, attributions, and
counterfactual addbacks were examined. Their hashes are respectively
`1df2a8d19a7b11c4ea35195f63650eb31c0b1baff4f57dc03e4577adf3bcdbbc`
and `86cea52b6585b692c0d79ca7a7b26d82a965520a9fdad9dd3b8f8b7a9471d84c`.
This remains post-result mechanistic analysis: the endpoint and frozen
selector-validation outcomes were already known, so no prospective
preregistration claim is made.

The interrupted implementation was recovered rather than replaced. It
reuses the validated epsilon_J=0.10 construction, uses signed denominators
E-epsilon, performs solves rather than explicit inverses, and tracks reduced
states by maximum total absolute overlap. Its P/Q dimensions are 2240/160
(full 2400), with Q1/Q2 dimensions 80/80. Source compilation succeeds.

## Reconstruction and numerical audit

| finite case | max energy error (GHz) | grid_phi error | eigenpair residual (GHz) |
|---|---:|---:|---:|
| k260 | 0 | 3.331e-15 | 2.397e-13 |
| k280 | 0 | 2.442e-15 | 2.975e-13 |
| k300 | 0 | 3.775e-15 | 2.801e-13 |

The full Hamiltonian Hermiticity error is 6.413e-13. All Schur effective
equation, exact shift-identity, and eliminated-vector checks pass their
frozen tolerances. Reduced-state tracking is safe: the worst assigned
overlap is 0.998118 (k280 state 8), and the largest competitor/assigned
ratio is 0.03907.

Cross-parameter raw-grid tracking was performed rather than equating rank
indices. Across ranks 240--299, minimum assigned overlap is 0.951459,
maximum competitor ratio 0.284121, and minimum margin 0.688410. Rank swaps
270/271 and 278/279 are detected. Within the attribution shell 280--299,
all numerical labels track to the same old physical rank; minimum overlap is
0.961629 and every match is safe.

All five required CSV outputs contain finite values. Their row counts are
360 rank attributions, 2880 shell-mode contributions, 120 cross-parameter
comparisons, 40 selector diagnostics, and 6 frequency-selectivity records.

## Exact new-point shell attribution

All values below are k280 to k300 changes in MHz. `Exact self-energy` is the
energy-dependent Schur term; `P relaxation` completes the exact identity.

| state | observed | exact self-energy | P relaxation | second order | second-order error | relative error | Q probability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -2.301194 | -2.669056 | +0.367862 | -1.985589 | +0.315605 | 13.715% | 4.298e-5 |
| 1 | -2.330950 | -2.703075 | +0.372125 | -2.011682 | +0.319268 | 13.697% | 4.358e-5 |
| 2 | -1.064726 | -1.247107 | +0.182382 | -0.910454 | +0.154271 | 14.489% | 2.071e-5 |
| 3 | -1.451993 | -1.703493 | +0.251500 | -1.239842 | +0.212151 | 14.611% | 2.859e-5 |
| 7 | -54.479177 | -66.671027 | +12.191851 | -44.551135 | +9.928041 | 18.224% | 1.194e-3 |
| 8 | -54.338596 | -66.022450 | +11.683853 | -44.681321 | +9.657276 | 17.772% | 1.182e-3 |

Every state passes the frozen second-order-control diagnostic. Minimum
absolute denominators are 43.087/43.071, 39.664/39.427, and 36.850/36.833
GHz for state pairs 0/1, 2/3, and 7/8. No near-resonant mode is present.
Second order captures the mechanism and sign of individual state shifts but
does not replace the exact energy-dependent result, especially for f78 where
differential cancellation amplifies its relative error.

### Carrier decomposition

| carrier | observed | exact self-energy | P relaxation | second order | second-order error |
|---|---:|---:|---:|---:|---:|
| f01 | -0.029756 | -0.034019 | +0.004263 | -0.026093 | +0.003663 |
| f07 | -52.177982 | -64.001971 | +11.823988 | -42.565546 | +9.612436 |
| f08 | -52.037402 | -63.353393 | +11.315991 | -42.695732 | +9.341671 |
| f78 | +0.140580 | +0.648578 | -0.507997 | -0.130185 | -0.270766 |
| f81 | -52.007646 | -63.319375 | +11.311728 | -42.669639 | +9.338007 |
| f72 | -53.414451 | -65.423920 | +12.009469 | -43.640681 | +9.773770 |
| f21 | +1.266225 | +1.455967 | -0.189743 | +1.101228 | -0.164997 |
| f03 | +0.849202 | +0.965564 | -0.116362 | +0.745747 | -0.103454 |
| f38 | -52.886604 | -64.318957 | +11.432353 | -43.441479 | +9.445125 |

For f78, the +0.648578 MHz exact self-energy is reduced to +0.140580 MHz by
-0.507997 MHz of P-space relaxation. The fixed-vector second-order result
has the wrong differential sign (-0.130185 MHz) even though it is controlled
for each individual state; this is a cancellation-sensitive observable.

### Dominant ranks and dressed modes

The six-state aggregate absolute exact rank weights total 141.016208 MHz:

| rank | aggregate weight (MHz) | fraction |
|---:|---:|---:|
| 292 | 62.599516 | 44.392% |
| 290 | 56.442319 | 40.025% |
| 289 | 7.714764 | 5.471% |
| 287 | 6.159704 | 4.368% |
| 286 | 5.401560 | 3.830% |
| 291 | 1.058019 | 0.750% |

For state 7, ranks 292/290/289 contribute -29.478107/-27.822027/-3.587837
MHz to exact self-energy. For state 8 they contribute
-29.768075/-25.589906/-3.793415 MHz. The state-7 rank thresholds are 2, 2,
and 3 ranks for 50%, 80%, and 90%; state 8 requires 2, 2, and 4.

The invariant dressed-mode attribution is also concentrated but less so.
For state 7, the three leading modes contribute -16.736430, -12.664999,
and -8.034230 MHz at denominators -52.991, -68.669, and -37.020 GHz; their
couplings are 0.941745, 0.932574, and 0.545367 GHz. For state 8 the leading
three contributions are -16.607563, -12.732793, and -8.073842 MHz at
denominators -52.825, -68.837, and -37.104 GHz, with couplings 0.936644,
0.936211, and 0.547334 GHz. The top-three mode fractions are 84.029% and
83.736%; three modes reach 80% and four reach 90% for both states.

### Nonadditivity and exact addbacks

The Q1--Q2 coupling Frobenius norm is 23.983569 GHz. Exact f78 shifts from
Q1 alone, Q2 alone, and full Q are -1.271811, +1.222616, and +0.140580 MHz.
The +0.189775 MHz nonadditivity comprises -0.032230 MHz explicit Q1--Q2
coupling and +0.222005 MHz P-mediated remainder. For f07/f08 the total
nonadditivities are -1.250333/-1.060558 MHz.

Exact nonlinear addbacks select ranks 292, 290, and 289. Their singleton
effects on f07 are -18.763843, -17.655546, and -2.314332 MHz; on f08 they
are -19.374024, -16.012541, and -2.492529 MHz; and on f78 they are
-0.610180, +1.643005, and -0.178198 MHz. The 292+290 pair gives
-44.281843/-42.990818/+1.291025 MHz for f07/f08/f78, including interaction
terms -7.862454/-7.604253/+0.258200 MHz. The three-rank addback gives
-47.444539/-46.323444/+1.121095 MHz. These interactions demonstrate why
individual shell contributions cannot be assumed additive.

## Cross-parameter mechanism

At epsilon_J=0.10 the state-7/state-8 observed final-shell shifts were
-54.873826/-52.652829 MHz, compared with -54.479177/-54.338596 MHz at
epsilon_J=0.11. Thus the differential f78 shell shift contracts from
+2.220997 to +0.140580 MHz even though the two large individual shifts stay
near -54 MHz.

The dominant exact rank self-energies remain comparable:

| state/rank | epsilon_J=0.10 (MHz) | epsilon_J=0.11 (MHz) |
|---|---:|---:|
| 7/292 | -30.001065 | -29.478107 |
| 7/290 | -29.220289 | -27.822027 |
| 7/289 | -2.448542 | -3.587837 |
| 8/292 | -29.043156 | -29.768075 |
| 8/290 | -25.810681 | -25.589906 |
| 8/289 | -2.518631 | -3.793415 |

The group-78 coupling for rank 292 is effectively unchanged
(1.530785 to 1.530446 GHz), and its minimum denominator is unchanged to
0.001 GHz. Rank 290 coupling decreases 1.466652 to 1.440481 GHz (-1.78%),
again with an effectively unchanged denominator. Rank 289 coupling grows
0.462470 to 0.558934 GHz (+20.86%) while its denominator changes only
37.678725 to 37.690897 GHz. This coupling increase, rather than a resonance,
promotes rank 289 above rank 287 in aggregate exact attribution. The top
three aggregate fraction decreases modestly from the old shell but remains
89.6--91.3% in the two central states, satisfying the same class B criteria.

The f78 addback anatomy changes quantitatively: at epsilon_J=0.10 rank
292/290/287 singletons give +0.055530/+2.266081/-0.581614 MHz; at
epsilon_J=0.11 rank 292/290/289 give -0.610180/+1.643005/-0.178198 MHz.
This explains the reduced differential shift without changing the broad
virtual-shell mechanism.

## Frozen-selector interpretation

The exact final-shell aggregate ordering starts
292, 290, 289, 287, 286; the frozen physics ordering starts
292, 290, 286, 289, 287. The selector therefore identifies all five leading
ranks without refitting. At b10 it includes nine of the leading exact ranks
and captures 99.507% of aggregate absolute exact weight. The only material
b10 omission is the residual 0.493%, not a missed dominant mechanism.

For completeness, the b10 carrier errors in MHz are:

| selector | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | -0.000386 | +0.019818 | +0.087566 | +0.067748 | +0.087952 | -0.012107 | +0.032311 | +0.038628 | +0.048938 |
| S2 | +0.000131 | +0.213761 | +0.358397 | +0.144636 | +0.358267 | +0.207155 | +0.006475 | +0.013603 | +0.344795 |
| union | -0.000094 | +0.234589 | +0.454973 | +0.220384 | +0.455068 | +0.191330 | +0.043354 | +0.057516 | +0.397458 |

At b15, maximum absolute carrier errors fall to 0.049711, 0.161192, and
0.209442 MHz for S1, S2, and union. The b10 primary/comparator pathway
relative errors are:

| selector | 0781 (rank) | 0721 (rank) | 0381 (rank) |
|---|---:|---:|---:|
| S1 | 9.052e-5 (1) | 9.269e-6 (3) | 2.193e-4 (4) |
| S2 | 7.667e-5 (1) | 2.450e-4 (3) | 4.019e-4 (5) |
| union | 1.375e-4 (1) | 3.374e-4 (3) | 1.556e-4 (5) |

All pathway orderings pass and all state tracking is safe. The frozen
evaluation's family error ratios versus ordinary prefix are below one for
all six groups and all three selectors. For S2 they range from 0.00478 to
0.00752, and for union from 0.00481 to 0.00850. This, together with exact
Schur-weight capture and zero frozen physics-cancellation flags at b10/b15,
supports systematic rather than accidental selector success.

The one formal PASS miss remains union/carrier:f78 no-harm. The union physics
set contains ten important S2 ranks absent from the prefix set, including
282, 286--293, and 298. The prefix contains only S1/global-prefix ranks in
260--279. Its state-7/state-8 errors nearly cancel despite omitting the entire
dominant final shell. The correct scientific interpretation is therefore
excellent comparator differential cancellation, not a selector mechanism
failure. The frozen classifications are unchanged: selector BORDERLINE,
multi-target transfer PARTIAL, and retained-basis convergence FAIL.

## Static frequency selectivity

| epsilon_J | k | f01 (MHz) | f78 (MHz) | f78-f01 (MHz) | f07 (MHz) | f81 (MHz) | f07-f81 (MHz) | g01 | g78 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.11 | 260 | 16.029950 | 17.849718 | +1.819767 | 6237.082332 | 6238.902099 | -1.819767 | 0.448236 | 1.173209 |
| 0.11 | 280 | 16.035194 | 17.125081 | +1.089887 | 6236.909808 | 6237.999696 | -1.089887 | 0.448182 | 1.173816 |
| 0.11 | 300 | 16.005438 | 17.265661 | +1.260224 | 6184.731826 | 6185.992050 | -1.260224 | 0.440616 | 1.257032 |
| 0.10 | 260 | 15.855811 | 14.568530 | -1.287281 | 6243.253114 | 6241.965833 | +1.287281 | 0.447948 | 1.155801 |
| 0.10 | 280 | 15.862073 | 13.632570 | -2.229503 | 6243.055864 | 6240.826361 | +2.229503 | 0.447897 | 1.156882 |
| 0.10 | 300 | 15.831920 | 15.853567 | +0.021648 | 6190.483521 | 6190.505168 | -0.021648 | 0.440333 | 1.232142 |

At epsilon_J=0.11 the k300 edge is no longer nearly degenerate: the
separation is 1.260 MHz rather than 0.0216 MHz. The selector preserves the
primary 0781 pathway as rank 1 across all reported budgets, but this static
evidence does not establish time-domain selectivity or gate fidelity.

## Scientific conclusions and preservation

The virtual-shell mechanism transfers. It remains a small-occupation,
off-resonant, strongly concentrated virtual correction dominated by the
same tracked ranks 292 and 290. The selector identifies genuinely dominant
contributions at epsilon_J=0.11 and its success is supported by exact weight
capture, target-family improvements, and stable pathway ordering. The
remaining BORDERLINE miss is explained by unusually strong control-prefix
cancellation in f78, not better recovery of either state.

This is adequate for further sandboxed exploratory numerical studies, but
not for production-precision or gate work. The authenticated retained-basis
classification is still FAIL, and fixed N71/four-mode results do not prove
spatial, domain, n1max, N4, or omitted-mode convergence.

Preservation checks pass: no new Stage-1 solve; no authenticated checkpoint,
frozen selector, criteria, or scientific-threshold modification; no tracked
production-file modification by this analysis; no EmilianoUPDATE.md update;
no commit or push; and no figures generated. The pending figure backlog is
unchanged. Repository status still contains the pre-existing modifications
to `EmilianoUPDATE.md`, `Simulations/Cphase/main.ipynb`, and
`Simulations/Cphase/workflow_funcs.py`; this task did not alter them.
