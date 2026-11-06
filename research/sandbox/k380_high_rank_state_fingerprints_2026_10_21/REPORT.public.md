> Publication copy: local machine paths have been replaced by repository-relative paths or `python`. Scientific statements and values are unchanged. The original file and its SHA-256 are recorded in [the publication manifest](../../publication/MANIFEST.json).

# High-rank virtual-state physical fingerprints — 2026-10-21

**SOURCE AUTHENTICATION: PASS. STATE CHARACTERIZATION: PARTIAL. NEW EIGENSOLVES: ZERO.**

The saved-data characterization is complete for every frozen observable. Overall characterization is PARTIAL because basis sensitivity under cutoff changes, cross-parent stability of ranks 354/362 and dressed operator-channel attribution cannot be computed from this saved parent. No essential requested finite-parent metric is missing.

The three influential ranks share unusually compact charge support, narrow theta2 distributions and low joint participation. They are the three lowest charge-edge weights, three smallest theta2 standard deviations, and three smallest coefficient participation ratios among ranks 300–379. Strong virtual response therefore does not coincide with unusually large charge-cutoff support in this comparison. Their theta3 distributions are broad, but boundary probability is not consistently enhanced. The saved spectrum does not indicate close-neighbor ambiguity at any focal rank. These are observations in one finite discretization, not a mode assignment or a convergence proof.

## Verified representation and meaning

`Circuit_Objs/qchard_gridium_netlist.py:114–154, 191–239, 417–476` supplies the basis and operators. `kron3(A, B, C)=C⊗B⊗A` gives i=(i3*71+i2)*9+i1; C-order reshape is (theta3, theta2, n_theta1). Independent production interpolation uses exactly this order. Stage-1 has 45369 coordinates; the eight mode-4 oscillator states enter Stage-2, not this reshape.

Charge indices -4…4 are Cooper-pair-number labels conjugate to compact theta1, at ng=0. They are not the physical derived grid charge, which production defines using -n_theta1+n_theta3+n_theta4. In the netlist transformation, n_theta1 is conjugate to the common compact coordinate; coordinate labels must not be silently identified with the older three-mode paper basis. The user’s phi2/phi3 boundary request is evaluated on the actual implementation’s theta2/theta3 grids.

theta2 spans [-11, 11] rad, theta3 spans [-14, 14] rad, including endpoints, with spacings 22/70 and 28/70. The standard eigenproblem has mass I. Stored coefficients have Euclidean norm; a continuum-density convention divides by sqrt(h2*h3) and uses equal rectangle weights at all nodes. Extra trapezoid endpoint weights would change the saved representation. Finite-difference stencils terminate at the grid edges; these are not periodic spatial boundaries. Charge raising terminates at ±4. Both edge observables are meaningful finite-basis probabilities.

Means, spreads and participation describe these verified coordinates. No compact-angle spread, spatial lobe identity, symmetry quantum number, full circuit normal-mode assignment or physical microwave-port interpretation is inferred.

## Frozen comparisons and state structure

Focal ranks: (328, 354, 362). Fixed neighbors: (326, 327, 329, 330, 352, 353, 355, 356, 360, 361, 363, 364). Deterministic controls: (300, 310, 320, 330, 340, 350, 360, 370). Every rank 300–379 receives identical metrics, including all four 20-state energy bands. Selection and definitions were frozen in RESEARCH_PLAN.md before state metrics.

| Rank | E (GHz) | P(n=±4) | sigma(theta2), rad | sigma(theta3), rad | Joint PR | Charge PR | Nearest gap, MHz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 328 | 45.991497693 | 0.00193052043 | 1.772167 | 6.603001 | 2563.319 | 4.005974 | 25.231970 |
| 354 | 47.385135953 | 0.00725824123 | 2.144274 | 6.449513 | 1576.443 | 3.232014 | 37.708165 |
| 362 | 47.734944812 | 0.00215927883 | 2.309048 | 6.524013 | 2429.926 | 3.605669 | 19.326910 |

| Metric | Band median | Band minimum | Band maximum |
| --- | --- | --- | --- |
| charge_edge | 0.164624977 | 0.00193052043 | 0.394941234 |
| theta2_std | 4.45535523 | 1.77216742 | 5.85911168 |
| theta3_std | 6.17090043 | 5.76189847 | 6.67878286 |
| joint_pr | 6812.00618 | 1576.44283 | 9295.48576 |
| theta2_edge2 | 1.57301709e-05 | 1.67431237e-08 | 0.00062928782 |
| theta3_edge2 | 0.000721055376 | 0.000347826426 | 0.00197102526 |

| 20-state band | Median charge edge | Median theta2 spread | Median theta3 spread | Median joint PR |
| --- | --- | --- | --- | --- |
| 300–319 | 0.1423725 | 4.416689 | 6.064905 | 6843.163 |
| 320–339 | 0.161302 | 4.524788 | 6.195909 | 6587.552 |
| 340–359 | 0.2042746 | 4.258027 | 6.122462 | 6602.502 |
| 360–379 | 0.1299192 | 4.605905 | 6.224867 | 7253.953 |

| Deterministic control | Charge edge | theta2 spread | theta3 spread | Joint PR |
| --- | --- | --- | --- | --- |
| 300 | 0.2921559 | 4.454524 | 5.865374 | 6662.748 |
| 310 | 0.1373263 | 5.149691 | 6.135533 | 6724.732 |
| 320 | 0.1285512 | 5.180625 | 6.324279 | 7644.406 |
| 330 | 0.346836 | 3.995706 | 5.853852 | 6352.904 |
| 340 | 0.1615374 | 4.26891 | 6.003198 | 8680.27 |
| 350 | 0.3515573 | 4.519314 | 5.972895 | 6613.739 |
| 360 | 0.08918502 | 4.832675 | 6.252058 | 7361.931 |
| 370 | 0.1040531 | 5.17606 | 6.403593 | 7529.356 |

Charge populations are balanced between ±n to numerical precision. Rank 328 has its largest charge populations at n=±1 (32.3233% each); rank 354 has 49.8896% at n=0; rank 362 has 34.3979% each at n=±1. This describes populations, not wavefunction parity or a quantum-number assignment. Coordinate means are near zero; widths and joint probability structure distinguish them.

The theta3 spreads are in the upper part of the band (third, ninth and fifth largest), whereas theta2 spreads are the three smallest. Joint PR is basis dependent; its small value alone is not a coordinate-independent localization claim. Charge-summed spatial PR areas are 91.2884, 106.568, 108.073 rad², respectively.

### Local comparisons

| Rank | P(n=±4) | sigma(theta2) | Joint PR | theta2 edge2 | theta3 edge2 |
| --- | --- | --- | --- | --- | --- |
| 326 | 0.12345 | 5.17526 | 6339.22 | 8.8394e-06 | 0.0011662 |
| 327 | 0.307058 | 4.43079 | 7738.24 | 3.9925e-05 | 0.00076617 |
| 328 | 0.00193052 | 1.77217 | 2563.32 | 3.6526e-08 | 0.00034783 |
| 329 | 0.105686 | 4.75948 | 5694.35 | 1.6904e-05 | 0.0017355 |
| 330 | 0.346836 | 3.99571 | 6352.90 | 1.4135e-05 | 0.00064495 |
| 352 | 0.256187 | 4.05664 | 7132.18 | 2.7721e-06 | 0.00097775 |
| 353 | 0.0848588 | 2.89492 | 5636.44 | 2.8652e-06 | 0.00053781 |
| 354 | 0.00725824 | 2.14427 | 1576.44 | 6.3015e-07 | 0.0006873 |
| 355 | 0.169873 | 5.12974 | 6030.11 | 0.00038539 | 0.00087501 |
| 356 | 0.227438 | 5.48158 | 7214.39 | 0.00050197 | 0.00052805 |
| 360 | 0.089185 | 4.83268 | 7361.93 | 0.00028056 | 0.0010164 |
| 361 | 0.0836406 | 4.67016 | 7131.48 | 8.386e-05 | 0.00094972 |
| 362 | 0.00215928 | 2.30905 | 2429.93 | 1.6743e-08 | 0.00091446 |
| 363 | 0.0774915 | 4.33862 | 7224.42 | 0.00010144 | 0.0007249 |
| 364 | 0.134429 | 4.14716 | 6986.36 | 1.0312e-05 | 0.00086444 |

The contrast is individual-state structure rather than a property shared by each adjacent five-state window. Equal-weight window mixtures have theta2 spreads 4.19938, 4.14239, 4.15979 rad, much broader than the corresponding focal states. Their charge-edge probabilities are 17.6992%, 14.9123%, 7.73811%. Complete window ensemble densities and all marginals are saved in DIAGNOSTICS.json. Mixing a whole window changes the observable to a subspace quantity; its individual-vector PR cannot be substituted for the focal PR.

## Charge and spatial boundaries

| Rank | Charge edge, % | theta2 edge1 | theta2 edge2 | theta3 edge1 | theta3 edge2 |
| --- | --- | --- | --- | --- | --- |
| 328 | 0.193052 | 4.73969263e-09 | 3.65255434e-08 | 6.17654062e-05 | 0.000347826426 |
| 354 | 0.725824 | 8.62775412e-08 | 6.30150527e-07 | 0.000122274382 | 0.000687300467 |
| 362 | 0.215928 | 2.24195891e-09 | 1.67431237e-08 | 0.000164053819 | 0.000914461428 |

Charge-edge probabilities are 0.193052%, 0.725824%, 0.215928%, versus a 16.4625% band median. Thus the focal states are not unusually charge-edge-loaded. Many comparison states have substantial outer-charge weight (up to 39.4941%); the surrounding high-energy basis should not be described as uniformly safe simply because these focal states have lower edge weight.

The focal theta2 two-node weights are exceptionally low (second-lowest, fifth-lowest and lowest). Theta3 two-node weights are 0.0347826%, 0.0687300%, 0.0914461%; the band median is 0.0721055% and range is 0.0347826–0.197103%. Rank 328 has the smallest theta3 edge weight despite a broad theta3 variance. Rank 354 is near the median and rank 362 is moderately above it, still within neighboring/band values. A large spread need not imply the largest endpoint density.

Finite edge support is observed in both spatial coordinates and in charge. There is no shared anomalously large boundary-support signature that explains all three influential states. This does not prove weak energy sensitivity to cutoff changes: boundary probability supplies neither an infinite-basis error estimate nor a test of spatial resolution, and a state with little edge weight can still respond indirectly through other states. All actual cutoff/domain/grid sensitivity tests remain INCOMPLETE and spatial convergence remains HOLD.

## Operator-mediated coupling

The five saved projections are dimension 380×380, O_pr=<p|O|r>=V_p† O V_r. The actual cross-rank Stage-2 block is H_pr=A_pr n4+B_pr theta4, with A=8(-6/13 n1+11/26 n2+1/26 n3) and B=-2.05 x2-x3. These coefficients are obtained from the authenticated production `_coeffs`, not inferred from a drive model. Oscillator spacing is 15.8454070026 GHz, theta4 scale 1.6250849723 and charge scale 0.3076762191.

OPERATOR_COUPLINGS.csv retains real/imaginary/absolute values of each operator, A, B and both oscillator off-diagonal elements for p<40 and every band state (3200 rows). OPERATOR_BLOCKS.npz also stores the same combined complex amplitudes for every p<300, including the strongest partners; it is audit storage, not a new selection rule. Fixed prefixes p<11, 40, 100, 300 and each own-shell lower prefix are summarized in FINGERPRINTS.csv. Stage-1 low ranks are not dressed low-state labels.

| High rank | sum p<300 \|\|H_pr\|\|F², GHz² | Band order, largest first | Strongest p | \|\|block\|\|F, GHz | Bare-vacuum proxy, GHz | Proxy band order |
| --- | --- | --- | --- | --- | --- | --- |
| 328 | 3538.175283 | 1 | 211 | 50.490433 | 3.144264 | 14 |
| 354 | 3220.337258 | 5 | 224 | 42.881510 | 2.766214 | 37 |
| 362 | 3284.350379 | 2 | 230 | 50.577738 | 2.837222 | 34 |

The unweighted coupling strengths rank first, fifth, second across the 80-state band. Strongest partners are intermediate retained ranks 211, 224, 230, not the first eleven Stage-1 states. The p<11 total block strengths are only 0.264462, 0.244446, 0.219224 GHz²; the low-only sector cannot stand in for the dressed retained vector. The block Frobenius norm sums all eight oscillator levels, including excited levels, and is not a target-state coupling or a perturbative-smallness measure.

| r / p | A_pr, GHz | B_pr, GHz/rad | H_(p, 0),(r, 1), GHz | H_(p, 1),(r, 0), GHz |
| --- | --- | --- | --- | --- |
| 328 / 211 | +0.773475+1.656949i | +3.749009-1.750063i | +6.602262-3.081981i | +5.582655-2.606021i |
| 354 / 224 | +1.078760-1.275008i | -2.681061-2.268395i | -4.749241-4.018243i | -3.964662-3.354425i |
| 362 / 230 | +0.412978+1.899907i | +4.048150-0.879937i | +7.163144-1.557036i | +5.994032-1.302909i |

At those strongest blocks, |B_pr theta4_01| exceeds |A_pr n4_01|. Their complex sum enhances the (0, 1) oscillator matrix element and partly cancels in the (1, 0) element (row, column indices). The saved phases are arbitrary under independent Stage-1 rephasing, but relative phases of operator terms within the same block are meaningful. Absolute values were taken only after coherent contractions. This identifies coordinate-mediated structure in these bare blocks, not the operator-channel composition of the dressed Schur response.

The frozen bare-vacuum proxy sums |H_(p, 0),(r, 1)|²/(E_r-E_p+w4_1-w4_0) over p in a fixed prefix. Its large GHz values are sums over many different bare initial states, not a predicted physical level shift. Its band rankings 14, 37, 34 do not reproduce Schur dominance. The own-shell denominators span 16.7648–60.6508, 16.6324–62.0444, 16.0042–62.3942 GHz. These are bare denominators and must not replace the saved dressed-shell denominators.

## Connection with authenticated Schur evidence

| Parent/shell | Rank | Aggregate \|rank scalars\|, MHz | Fraction of shell | Target 7 coupling, GHz | Target 8 coupling, GHz |
| --- | --- | --- | --- | --- | --- |
| k340/new_S4 | 328 | 3.237027672 | 79.454192% | 0.299575 | 0.338625 |
| k380/S1 | 354 | 9.784721460 | 66.823079% | 0.578108 | 0.613442 |
| k380/S2 | 362 | 15.138873587 | 88.992448% | 0.726947 | 0.770544 |

S1 and S2 aggregate absolute totals reproduce 14.642728 and 17.011414 MHz to the published rounding. The large saved target 7/8 couplings and small rank Q probabilities support off-resonant virtual response: rank 354 target 7/8 probabilities are 6.1194e-5/6.6814e-5; rank 362 gives 9.1974e-5/1.07139e-4. Earlier authenticated analysis found no near-resonant shell weight. Exact scalar attribution includes mixed-shell resolvent effects. Aggregate absolute weights are not exact signed transition-energy shifts; even signed rank-scalar differences are not rank-ablation experiments. No Schur calculation was rerun.

The saved isolated-rank selector response supplies another denominator convention. For the target pair (7, 8), its minimum positive denominators are:

| Rank | Selector P prefix | Minimum isolated-rank denominator, GHz |
| --- | --- | --- |
| 328 | 300 | 40.501509020 |
| 354 | 340 | 41.897005102 |
| 362 | 340 | 42.246813962 |

Those isolated-rank values are off-resonant context, not full-shell Schur denominators. In particular, rank 362 selector P340 differs from exact S2 P360; rank 328 selector P300 differs from historical exact P320. Their reference vectors and energy arguments must remain distinct.

Rank 328 attribution comes from the separate k340 parent, with P320, while this fingerprint uses its authenticated k380 vector. The prior saved same-rank overlap is 1.0000000000000002 (roundoff), with strongest competitor 3.13e-12. This identity evidence was read, not recomputed. Ranks 354/362 do not exist in the k340 archive; cross-parent continuity for them remains INCOMPLETE.

Across all 40 S1/S2 ranks, the fixed descriptive Spearman correlations with absolute Schur weight are:

| Metric | S1 (20 ranks) | S2 (20 ranks) | Union (40 ranks) |
| --- | --- | --- | --- |
| charge_edge | -0.512782 | -0.469173 | -0.430582 |
| theta2_edge2 | -0.566917 | -0.505263 | -0.586492 |
| theta3_edge2 | -0.022556 | -0.248120 | -0.205253 |
| joint_pr | -0.311278 | -0.538346 | -0.418386 |
| theta2_std | -0.774436 | -0.678195 | -0.750094 |
| theta3_std | +0.082707 | -0.120301 | -0.031520 |
| p300_block_strength_GHz2 | +0.809023 | +0.705263 | +0.764916 |
| p300_vacuum_proxy_GHz | +0.765414 | +0.711278 | +0.756285 |

The strongest fixed associations are stronger block coupling and smaller theta2 spread; theta3 spread and boundary support correlate weakly. Negative charge-edge association runs opposite to a simple edge-loaded-artifact account. These descriptive correlations are dependent finite-shell observations, not causal evidence or a predictive selector validation. They do not isolate which operator term produces a dressed target response.

## Spectral neighbors and stability

| Rank | Gap below, MHz | Gap above, MHz | Minimum gap / summed saved residuals |
| --- | --- | --- | --- |
| 328 | 168.942242224 | 25.231970324 | 5.56873e+09 |
| 354 | 104.376842379 | 37.708164902 | 8.61204e+09 |
| 362 | 19.326909576 | 68.758296814 | 4.15186e+09 |

All focal clusters are singletons under the frozen 1-MHz rule; no rank 300–379 enters such a cluster or the residual-scaled ambiguity flag. Finite-parent individual-state fingerprints are therefore resolved against the saved solver residuals. Five-state ensemble diagnostics provide rotation-invariant alternatives, but these windows are not numerically degenerate multiplets. This does not establish stability under a different grid, charge cutoff, circuit parameter or parent beyondk380. Saved residuals certify numerical resolution of this finite matrix, not physical basis convergence.

## Conclusions, limitations and next method decision

- Definitive within the saved finite parent: basis ordering and normalization; compact charge support and unusually narrow theta2 structure of the three focal ranks; low joint PR; lack of shared enhanced boundary weight; strong retained-sector bilinear blocks; finite spectral resolution.
- Preliminary interpretation: a recurring coordinate/charge structural family with favorable operator overlap is consistent with concentrated virtual response. Similar marginals do not establish a shared normal mode, excitation quantum number or microscopic cause.
- INCOMPLETE: cutoff/domain/resolution derivatives or convergence bounds; missing dressed-vector channel contractions and exact dressed subspace responses; new cross-parent stability of 354/362; unseen ranks beyond 379; any calibrated experimental drive interpretation.

These results favor testing an effective retained-space method against known finite-parent targets rather than diagnosing the influential states as merely charge-cutoff-loaded. They do not clear retained-basis convergence or identify a safe larger-k endpoint. The bare-vacuum proxy is not accurate enough conceptually to replace target-aware dressed response, regardless of its descriptive correlation.

**One recommended next experiment (not executed):** when separately authorized, run the sealed restricted v2 eleven-state matrix Schur benchmark for S340, S360, U340 against its frozen finite-parent targets and tolerances, without refitting. This tests whether retaining coherent target-dependent matrix response addresses the observed concentrated off-resonant influence. Prelaunch review and any required dressed solves need separate authorization. This recommendation does not release the current HOLD.

**V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.**

## Artifacts and reproduction

FINGERPRINTS.csv contains 80 state rows, all normalized marginals (JSON arrays inside CSV fields), both-sided edge probabilities, IPR/PR/moments, gaps, controls and coupling summaries. OPERATOR_COUPLINGS.csv contains 3200 complex element records. DIAGNOSTICS.json supplies subspaces, full correlations, saved Schur records and circuit coefficients. OPERATOR_BLOCKS.npz retains combined complex audit amplitudes. None duplicates the large eigenvector array.

Use the existing pinned Python environment; no installation is needed. From the repository root:

```shell
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/analyze.py
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/verify_reproduction.py
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/preservation.py verify
```

build_report.py regenerates the Markdown narrative after numerical outputs. Runtime/performance records vary between runs; scientific CSV/JSON/NPZ artifacts must reproduce byte for byte. See NUMERICAL_VALIDATION.md and SOURCE_PROVENANCE.md.
