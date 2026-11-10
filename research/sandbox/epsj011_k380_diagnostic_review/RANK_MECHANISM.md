# Dominant virtual ranks

Source keys and units are preserved in DIAGNOSTICS.json. Rank indices are zero-based. Stage-1 energies and couplings are GHz (H/h, no 2pi). Scalar self-energies and response matrices below are MHz.

## Exact scalar decomposition

For each tracked full-shell eigenstate, d=H_QP p, q=(E-H_QQ)^(-1)d, and sigma=Re(d†q)/(p†p). The saved rank partition is sigma_r=Re(d_r†q_r)/(p†p). It includes cross-rank resolvent mixing through q. Sum_r sigma_r=sigma. The exact energy identity is deltaE=sigma+P-space relaxation.

A rank's aggregate absolute weight sums |sigma_r| over six target states. It is neither an eigenvalue shift nor a separately exact transition correction. The pair differences below compare scalar attributions evaluated at different state-specific energies and vectors; they are descriptive partitions, not rank-ablation results.

## Rank 328 (new_S4, parent k340)

Stage-1 energy: 45.9914976933105 GHz. Neighbor gaps: 0.168942242 / 0.0252319703 GHz. Absolute attribution: 3.23702767 MHz, 79.454191541% of this shell.

| Target | Signed rank scalar (MHz) | Target-projected P-Q coupling (GHz) | Rank Q probability |
| --- | --- | --- | --- |
| 0 | -0.0936963922 | 0.0860998221 | 1.25885815e-06 |
| 1 | -0.0625756539 | 0.0748781726 | 7.40935062e-07 |
| 2 | -0.325962544 | 0.155337541 | 4.64491661e-06 |
| 3 | -0.291808043 | 0.14359266 | 4.45350677e-06 |
| 7 | -1.09253071 | 0.299575264 | 1.42433232e-05 |
| 8 | -1.37045433 | 0.338625497 | 1.73916477e-05 |

| Pair | Mean rank scalar (MHz) | Scalar difference j−i (MHz) | Exact whole-shell pair shift (MHz) |
| --- | --- | --- | --- |
| [0, 1] | -0.078136023 | 0.0311207383 | 0.0305974447 |
| [2, 3] | -0.308885293 | 0.0341545019 | -0.0391470987 |
| [7, 8] | -1.23149252 | -0.277923624 | -0.236537103 |

### Saved selector response: base P300

These are the frozen positive isolated-rank 2×2 response A_r=B†(H_rr−mu_pair)^(-1)B, with mu_pair the base-pair mean energy. The corresponding isolated second-order correction has sign −A_r. This is the saved selector model, not the exact full-shell Schur matrix. Its response is unwhitened. Raw complex entries remain in DIAGNOSTICS.json.

| Pair | Trace (MHz) | Coupling norm (GHz) | Min positive denominator (GHz) | Max eta | abs(A01) (MHz) |
| --- | --- | --- | --- | --- | --- |
| [0, 1] | 0.12379273 | 0.101237172 | 46.686871 | 0.000841142936 | 5.03762744e-13 |
| [2, 3] | 0.473193437 | 0.184318117 | 43.1520093 | 0.0015586579 | 2.66900941e-13 |
| [7, 8] | 1.84453198 | 0.391613057 | 40.501509 | 0.00269135193 | 2.53194754e-12 |

Exact shell attribution uses P320; the selector uses P300. Their reference vectors and energy arguments differ even when those dimensions match. In particular, rank362's selector uses P340 whereas exact S2 uses P360. Rank328's selector uses P300 whereas exact historical S4 uses P320.

## Rank 354 (S1, parent k380)

Stage-1 energy: 47.3851359527375 GHz. Neighbor gaps: 0.104376842 / 0.0377081649 GHz. Absolute attribution: 9.78472146 MHz, 66.823078574% of this shell.

| Target | Signed rank scalar (MHz) | Target-projected P-Q coupling (GHz) | Rank Q probability |
| --- | --- | --- | --- |
| 0 | -0.196451743 | 0.126120435 | 2.56234446e-06 |
| 1 | -0.193869143 | 0.126600351 | 2.50936502e-06 |
| 2 | -0.0840078266 | 0.0772279813 | 1.23970174e-06 |
| 3 | -0.102443935 | 0.0878506901 | 1.45678026e-06 |
| 7 | -4.34801792 | 0.578107933 | 6.11940358e-05 |
| 8 | -4.85993089 | 0.613442326 | 6.68141624e-05 |

| Pair | Mean rank scalar (MHz) | Scalar difference j−i (MHz) | Exact whole-shell pair shift (MHz) |
| --- | --- | --- | --- |
| [0, 1] | -0.195160443 | 0.00258260068 | -0.0167935626 |
| [2, 3] | -0.093225881 | -0.0184361088 | -0.148046401 |
| [7, 8] | -4.60397441 | -0.511912975 | -1.06542825 |

### Saved selector response: base P340

These are the frozen positive isolated-rank 2×2 response A_r=B†(H_rr−mu_pair)^(-1)B, with mu_pair the base-pair mean energy. The corresponding isolated second-order correction has sign −A_r. This is the saved selector model, not the exact full-shell Schur matrix. Its response is unwhitened. Raw complex entries remain in DIAGNOSTICS.json.

| Pair | Trace (MHz) | Coupling norm (GHz) | Min positive denominator (GHz) | Max eta | abs(A01) (MHz) |
| --- | --- | --- | --- | --- | --- |
| [0, 1] | 0.321063855 | 0.161413372 | 48.0806268 | 0.00117402621 | 2.00159892e-13 |
| [2, 3] | 0.142892488 | 0.103205617 | 44.5464085 | 0.000850756748 | 2.49672886e-13 |
| [7, 8] | 7.15932168 | 0.740825761 | 41.8970051 | 0.00549083129 | 1.03763315e-11 |

Exact shell attribution uses P340; the selector uses P340. Their reference vectors and energy arguments differ even when those dimensions match. In particular, rank362's selector uses P340 whereas exact S2 uses P360. Rank328's selector uses P300 whereas exact historical S4 uses P320.

## Rank 362 (S2, parent k380)

Stage-1 energy: 47.7349448123736 GHz. Neighbor gaps: 0.0193269096 / 0.0687582968 GHz. Absolute attribution: 15.1388736 MHz, 88.992448042% of this shell.

| Target | Signed rank scalar (MHz) | Target-projected P-Q coupling (GHz) | Rank Q probability |
| --- | --- | --- | --- |
| 0 | -0.27072295 | 0.150596966 | 3.44797252e-06 |
| 1 | -0.290577071 | 0.15445184 | 3.73608582e-06 |
| 2 | -0.0513074259 | 0.0665211647 | 6.36476965e-07 |
| 3 | -0.0872520096 | 0.0847886974 | 1.12328731e-06 |
| 7 | -6.75990663 | 0.726947092 | 9.19743457e-05 |
| 8 | -7.6791075 | 0.77054403 | 0.000107138581 |

| Pair | Mean rank scalar (MHz) | Scalar difference j−i (MHz) | Exact whole-shell pair shift (MHz) |
| --- | --- | --- | --- |
| [0, 1] | -0.28065001 | -0.0198541208 | -0.0164425107 |
| [2, 3] | -0.0692797178 | -0.0359445837 | -0.0533547139 |
| [7, 8] | -7.21950707 | -0.919200877 | -0.606498937 |

### Saved selector response: base P340

These are the frozen positive isolated-rank 2×2 response A_r=B†(H_rr−mu_pair)^(-1)B, with mu_pair the base-pair mean energy. The corresponding isolated second-order correction has sign −A_r. This is the saved selector model, not the exact full-shell Schur matrix. Its response is unwhitened. Raw complex entries remain in DIAGNOSTICS.json.

| Pair | Trace (MHz) | Coupling norm (GHz) | Min positive denominator (GHz) | Max eta | abs(A01) (MHz) |
| --- | --- | --- | --- | --- | --- |
| [0, 1] | 0.348720654 | 0.170198939 | 48.4304357 | 0.00120950744 | 3.2380176e-13 |
| [2, 3] | 0.084984363 | 0.0807219093 | 44.8962173 | 0.000713064215 | 1.82399058e-13 |
| [7, 8] | 8.1094657 | 0.795584092 | 42.246814 | 0.00609571886 | 1.37593331e-11 |

Exact shell attribution uses P360; the selector uses P340. Their reference vectors and energy arguments differ even when those dimensions match. In particular, rank362's selector uses P340 whereas exact S2 uses P360. Rank328's selector uses P300 whereas exact historical S4 uses P320.

## Rank identity

The old k340 rank328 overlaps most strongly with k380 rank328. Absolute same-rank overlap is 1; strongest competitor is 3.1300394e-12. The overlap is phase invariant and uses the same physical grid. This supports identity continuity for rank328 without adopting a new classification threshold.

Ranks354/362 are fixed sorted identities in one k380 parent with positive saved neighbor gaps. Cross-parent stability for them is INCOMPLETE because the k340 checkpoint stops at rank339. Do not interpret the recurring concentration as the same eigenstate migrating from rank328 to354 to362.

## Mechanism supported by these records

The dominant ranks have much larger target-projected couplings to states 7/8 than to the lower pairs. Large denominators suppress their admixture, but energy corrections scale with coupling squared divided by an energy denominator, while occupation scales with the squared coupling/denominator ratio. Thus small occupation and stable state identity can coexist with multi-MHz differential shifts. This is an explanation of the measured finite-parent Schur identities, not an extrapolation to an infinite tail.

The saved isolated-rank within-pair off-diagonal responses are extremely small compared with their diagonal entries. That supports unequal diagonal responses within this limited diagnostic. Exact full-shell off-diagonal matrices, cross-pair mixing, and an energy scan are INCOMPLETE: the checkpoint does not save dressed Stage-2 eigenvectors or the necessary contractions. Scalar P-space relaxation is evidence that p changes, not a reconstruction of those missing matrix entries.
