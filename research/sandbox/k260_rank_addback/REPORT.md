# Gridium k260 rank-addback and ablation experiment

## Decision

**RECONSTRUCTION: PASS**

**SELECTED-BASIS STRATEGY: PROMISING**

The three-rank selection `{223,225,230}` recovers most of the k220-to-k240
reorganization for states 7/8 and the primary carriers at only three added
Stage-1 ranks. It preserves state identity, pathway ordering, and the relevant
`grid_phi` operators. Both prespecified equal-cost controls perform far worse.

The result is multi-target but not universal: it transfers to logical states
0/1, not comparator states 2/3. It is an exact same-parent result and therefore
does not establish retained-basis convergence. Ranks 258/259 remain active near
the k260 boundary. The formal retained-basis verdict remains **FAIL**.

## Preregistration and data authentication

The complete design was frozen before any selected-space diagonalization in
`preregistration.json` with SHA-256:

`cdde5cd5c520f80719cfc417c90270fc40508df44276809725a03825ed088c7b`

The hash is also recorded in `PREREGISTRATION.sha256` and hard-coded in the
analysis script. It fixes the 14 retained spaces, two controls, all observables,
recovery and nonadditivity formulas, inherited tolerances, and decision rule.

The authenticated k260 checkpoint passed `COMPLETE`, configuration, payload,
and checkpoint-hash validation. Its saved Stage-1 eigensystem has shape
`(45369,260)`; `n1`, `n2`, `n3`, `x2`, and `x3` projections each have shape
`(260,260)`. Those data are sufficient to form any selected principal
Stage-1 subspace, with all eight mode-4 states per rank. No Stage-1 Hamiltonian
or eigenvector was recomputed.

The selected dimensions range from 1760 for P through 2080 for full k260.
These are finite Stage-2 calculations using the existing parent.

## Reconstruction gate

Contiguous P=k220, P+Q1=k240, and full k260 reproduce the authenticated nested
results before any selection result is interpreted.

| Case | Max energy error (GHz) | Max `|grid_phi|` error | Eigenpair residual (GHz) | Status |
|---|---:|---:|---:|---|
| P / k220 | 0 | 1.776e-15 | 2.884e-13 | PASS |
| P+Q1 / k240 | 0 | 2.442e-15 | 2.476e-13 | PASS |
| Full k260 | 0 | 2.887e-15 | 3.528e-13 | PASS |

The reconstructed Hamiltonian Hermiticity error is `3.801e-13`, below the
frozen `1e-12` limit.

## Spectral comparison

Frequencies are MHz. Q1 recovery compares every selected space with the exact
k220-to-k240 change. Agreement with k240 is not convergence beyond k260.

| Case | Added ranks | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 | f78-f01 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P | none | 15.908195 | 6272.819042 | 6292.853928 | 20.034886 | 6276.945733 | 2822.096724 | 3434.814124 | 3681.799676 | 2611.054252 | 4.126691 |
| 223 | 223 | 15.873583 | 6265.373545 | 6283.197366 | 17.823821 | 6267.323783 | 2814.334585 | 3435.165378 | 3682.106139 | 2601.091227 | 1.950238 |
| 230 | 230 | 15.903188 | 6262.389941 | 6281.740186 | 19.350245 | 6265.836998 | 2811.157532 | 3435.329220 | 3682.307559 | 2599.432627 | 3.447057 |
| 225 | 225 | 15.894707 | 6271.294390 | 6290.924066 | 19.629676 | 6275.029359 | 2820.544612 | 3434.855071 | 3681.859809 | 2609.064257 | 3.734968 |
| 223+230 | 223,230 | 15.863252 | 6252.430394 | 6269.231524 | 16.801130 | 6253.368272 | 2800.791839 | 3435.775303 | 3682.705524 | 2586.526000 | 0.937879 |
| Dominant three | 223,225,230 | 15.846884 | 6249.882814 | 6266.168813 | 16.285999 | 6250.321929 | 2798.167033 | 3435.868897 | 3682.818392 | 2583.350421 | 0.439116 |
| Complement 17 | Q1 except dominant three | 15.927074 | 6270.338408 | 6289.528715 | 19.190308 | 6273.601642 | 2822.160973 | 3432.250361 | 3678.740065 | 2610.788650 | 3.263234 |
| Full Q1 / k240 | 220–239 | 15.864131 | 6246.176785 | 6261.563920 | 15.387135 | 6245.699789 | 2796.971743 | 3433.340911 | 3679.790146 | 2581.773775 | -0.476995 |
| Q1+258 | Q1,258 | 15.859938 | 6245.514299 | 6259.944933 | 14.430634 | 6244.084995 | 2796.521915 | 3433.132446 | 3679.447160 | 2580.497773 | -1.429303 |
| Q1+259 | Q1,259 | 15.854096 | 6245.502919 | 6260.006114 | 14.503195 | 6244.152018 | 2796.564108 | 3433.084715 | 3679.314228 | 2580.691887 | -1.350901 |
| Q1+258+259 | Q1,258,259 | 15.849797 | 6244.820024 | 6258.338681 | 13.518657 | 6242.488884 | 2796.100094 | 3432.870133 | 3678.960453 | 2579.378228 | -2.331139 |
| Full k260 | 220–259 | 15.855811 | 6243.253114 | 6257.821644 | 14.568530 | 6241.965833 | 2794.963818 | 3432.433485 | 3678.193165 | 2579.628479 | -1.287281 |
| Prefix control | 220,221,222 | 15.932096 | 6271.557062 | 6291.173531 | 19.616469 | 6275.241435 | 2822.862877 | 3432.762089 | 3679.511673 | 2611.661858 | 3.684373 |
| Spread control | 226,232,238 | 15.906428 | 6272.664197 | 6292.069854 | 19.405657 | 6276.163426 | 2822.120600 | 3434.637169 | 3681.456047 | 2610.613807 | 3.499229 |

## Dominant-three recovery

The descriptive recovery fraction is

`1 - |selected effect - full Q1 effect| / |full Q1 effect|`.

It is not clipped and is not a convergence classification.

| Quantity | Full Q1 change | Dominant-three change | Recovery | Error versus k240 |
|---|---:|---:|---:|---:|
| State 0 energy | -1.409978 MHz | -1.174046 MHz | 83.267% | +0.235932 MHz |
| State 1 energy | -1.454042 MHz | -1.235357 MHz | 84.960% | +0.218685 MHz |
| State 2 energy | -2.927255 MHz | -0.180584 MHz | 6.169% | +2.746671 MHz |
| State 3 energy | -3.419509 MHz | -0.155330 MHz | 4.542% | +3.264179 MHz |
| State 7 energy | -28.052236 MHz | -24.110275 MHz | 85.948% | +3.941961 MHz |
| State 8 energy | -32.699986 MHz | -27.859161 MHz | 85.196% | +4.840825 MHz |
| f07 | -26.642257 MHz | -22.936229 MHz | 86.090% | +3.706029 MHz |
| f08 | -31.290008 MHz | -26.685115 MHz | 85.283% | +4.604893 MHz |
| f78 | -4.647750 MHz | -3.748886 MHz | 80.660% | +0.898864 MHz |
| f81 | -31.245944 MHz | -26.623804 MHz | 85.207% | +4.622140 MHz |
| f72 | -25.124981 MHz | -23.929691 MHz | 95.243% | +1.195290 MHz |
| f21 | -1.473213 MHz | +1.054774 MHz | -71.597% | +2.527986 MHz |
| f03 | -2.009530 MHz | +1.018716 MHz | -50.694% | +3.028246 MHz |
| f38 | -29.280477 MHz | -27.703831 MHz | 94.615% | +1.576646 MHz |
| f78-f01 | -4.603686 MHz | -3.687575 MHz | 80.100% | +0.916111 MHz |

The frozen 80% target is met for states 7/8 and f07/f08/f78. Every carrier is
practical PASS versus k240. f01 is strict PASS; the other eight are strict
BORDERLINE. The largest carrier error is 4.622140 MHz.

This answers the transfer question sharply: the selection captures the
logical states 0/1 and the primary-state shifts, but not states 2/3. The
dominant selection moves f21 and f03 in the opposite direction from full Q1.
The selected strategy is therefore pathway-informed, not a universal compact
basis for every low-energy observable.

## Equal-cost controls and accuracy per rank

| Selection | State-7 recovery | State-8 recovery | f07 recovery | f08 recovery | f78 recovery | Primary worst error |
|---|---:|---:|---:|---:|---:|---:|
| Dominant three | 85.948% | 85.196% | 86.090% | 85.283% | 80.660% | 4.604893 MHz |
| Prefix `{220,221,222}` | 4.735% | 5.342% | 4.737% | 5.370% | 9.003% | 29.609611 MHz |
| Spread `{226,232,238}` | 0.621% | 2.457% | 0.581% | 2.506% | 13.538% | 30.505934 MHz |

The dominant-three accuracy gains per added rank are 7.645 MHz/rank for f07,
8.895 for f08, and 1.250 for f78. The prefix control provides only 0.421,
0.560, and 0.139 MHz/rank; the spread control provides 0.052, 0.261, and
0.210 MHz/rank. The dominant selection beats both controls under every frozen
strategy comparison.

The complementary 17 ranks alone leave a 29.015 MHz worst carrier error and
recover only 2.669/3.514 MHz of the state-7/8 shifts. Three informed ranks are
substantially more efficient for the primary mechanism than those 17 ranks.

## Nonadditivity and ablation

The following are exact selected-space effects in MHz, not self-energy shares.

| Quantity | Effect 223 | Effect 230 | Effect 223+230 | Pair nonadditivity | Effect 225 | Dominant-three effect | 225-with-pair interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| State 7 | -7.764757 | -10.941517 | -21.310370 | -2.604096 | -1.722236 | -24.110275 | -1.077669 |
| State 8 | -9.975822 | -11.626157 | -24.544125 | -2.942145 | -2.127446 | -27.859161 | -1.187590 |
| f07 | -7.445497 | -10.429101 | -20.388649 | -2.514050 | -1.524652 | -22.936229 | -1.022928 |
| f08 | -9.656562 | -11.113742 | -23.622404 | -2.852100 | -1.929862 | -26.685115 | -1.132849 |
| f78 | -2.211065 | -0.684641 | -3.233755 | -0.338050 | -0.405210 | -3.748886 | -0.109921 |

Ranks 223 and 230 are not additive: their joint state shifts are another
2.60–2.94 MHz below the sum of single-rank calculations. Adding rank 225 after
the pair produces 2.80/3.32 MHz more lowering for states 7/8, compared with
1.72/2.13 MHz from rank 225 alone. The grouped interaction contributes the
remaining -1.08/-1.19 MHz. This validates the user's warning that attributed
self-energy shares are not standalone addback predictions.

For states 7/8, dominant-three effects are -24.110/-27.859 MHz, complement-17
effects are -2.669/-3.514 MHz, and full Q1 effects are -28.052/-32.700 MHz.
The residual dominant/complement nonadditivity is -1.273/-1.327 MHz. For
f07/f08/f78 it is -1.225/-1.280/-0.054 MHz.

## State and operator validation

The dominant-three space safely tracks all required states. Against k240:

| Diagnostic | Result | Frozen classification |
|---|---:|---|
| State-7 overlap | 0.999952238 | PASS |
| State-8 overlap | 0.999937398 | PASS |
| `{0,1,7,8}` minimum singular value | 0.999936773 | PASS |
| Maximum principal angle | 0.644305 degrees | Supporting |

Against k260, state-7/state-8 overlaps are 0.999901687/0.999854273, and the
subspace minimum singular value is 0.999853201. All tracking-ambiguity gates
pass. The primary/0721/0381 ranks are `1/3/5`, identical to k240 and k260.

**OPERATOR ACCURACY: PASS**

| Quantity | Dominant three | Full k240 | Relative error |
|---|---:|---:|---:|
| g07 | 1.911083322 | 1.910605408 | +0.0250% |
| g78 | 1.155356229 | 1.156867287 | -0.1306% |
| g81 | 1.783710762 | 1.781096767 | +0.1468% |
| Primary score | 3.938401292 | 3.936788277 | +0.0410% |
| 0721 score | 0.317411309 | 0.317564675 | -0.0483% |
| 0381 score | 0.242932592 | 0.243329215 | -0.1630% |
| Full block Frobenius norm | 4.091495199 | 4.089519178 | +0.0483% |
| Logical-to-`{7,8}` Frobenius norm | 2.614165937 | 2.612033445 | +0.0816% |
| Internal `{7,8}` Frobenius norm | 1.633920449 | 1.636057408 | -0.1306% |

Every edge, path score, block norm, and ordered singular value is strict PASS.
The worst operator component is g38 at 0.2853% relative error, well below the
3% strict threshold. No selected space is being accepted on energies while
hiding an operator failure.

## Active upper boundary

Ranks 258 and 259 remain individually influential:

| Quantity | Rank 258 effect on k240 | Rank 259 effect | Joint effect | Pair nonadditivity | Full k240→k260 |
|---|---:|---:|---:|---:|---:|
| State 7 | -0.677641 | -0.696353 | -1.394749 | -0.020756 | -3.122168 |
| State 8 | -1.634142 | -1.580293 | -3.263227 | -0.048792 | -3.940773 |
| f07 | -0.662486 | -0.673866 | -1.356761 | -0.020409 | -2.923671 |
| f08 | -1.618987 | -1.557806 | -3.225239 | -0.048446 | -3.742276 |
| f78 | -0.956501 | -0.883940 | -1.868478 | -0.028037 | -0.818605 |

The pair is nearly additive, but it does not reproduce the whole Q2 shell.
For f78 the pair shift is more than twice the final full-shell shift, showing
substantial cancellation from the other Q2 ranks and their interactions.
The active rank-259 boundary and cancellation mean a larger independent parent
is required before any convergence claim.

## Scientific answers

1. **Do the three ranks capture most of the central reorganization?** Yes for
   states 7/8, states 0/1, primary carriers, f72, and f38: roughly 80–95%.
2. **Is state identity preserved?** Yes; all overlap, subspace, and ambiguity
   gates pass.
3. **Are important operators reproduced?** Yes; every frozen operator metric
   is strict PASS.
4. **Is it more accurate per rank than prefix truncation?** Yes, by roughly an
   order of magnitude or more for the primary carriers.
5. **Does it transfer to all requested states?** No. It transfers to 0/1/7/8,
   but captures only 6.2%/4.5% for states 2/3 and gives wrong-signed f21/f03
   effects.
6. **Is the upper boundary still active?** Yes. Ranks 258/259 have MHz-scale
   effects and the remaining shell causes important cancellation.
7. **Is physics-informed selection justified?** It is **PROMISING** for
   independent multi-target validation, not yet justified as a production
   basis or convergence substitute.
8. **Is a larger parent required for convergence?** Yes. The same-parent
   addback cannot test ranks beyond 259.

## Next research action

**Preregister one held-out N71 k300 parent experiment with unchanged physics.**
Before launch, freeze a multi-target coupling-based selector derived from k260
that covers states 0,1,2,3,7,8 rather than only the primary pathway. Apply the
algorithm unchanged to the two new 20-rank shells and compare it against
equal-cost prefix and prespecified spread controls for every mandatory carrier,
tracked subspace, edge, pathway score, and operator block.

k300 is recommended here for a specific reason: it supplies two held-out
20-rank shells beyond the active rank-259 boundary, allowing the selector to be
tested against both tail behavior and accidental one-shell cancellation. It is
not recommended merely because k260 failed. This experiment was not executed.

## Preservation and limits

k260 remains an endpoint, not ground truth. The experiment tests selected
retained Stage-1 spaces only at fixed N71 in the fixed four-mode model. It does
not establish spatial/domain, n1max, N4, omitted-array-mode, physical-port, or
X90/X180 convergence or validation.

Before/after hashes confirm no tracked file or authenticated checkpoint was
changed. No Stage-1 eigensolve, production edit, `EmilianoUPDATE.md` edit,
commit, or push occurred.

**GATE WORK: HOLD**

**SPATIAL CONVERGENCE: HOLD**
