> Publication copy: local machine paths have been replaced by repository-relative paths or `python`. Scientific statements and values are unchanged. The original file and its SHA-256 are recorded in [the publication manifest](../../publication/MANIFEST.json).

# Coupling strength versus virtual-state influence — 2026-10-21

**SOURCE AUTHENTICATION: PASS. ANALYSIS: PARTIAL. NEW EIGENSOLVES: ZERO.**

All 35 available frozen predictors were evaluated for both complete shells and both dominant-removed populations. Three further requested classes remain unavailable or not applicable; exact dressed channel/interference attribution and the missing S2 P360 selector response are explicitly INCOMPLETE. No previous fingerprint calculation or sealed benchmark was rerun.

**The apparent dominant-rank disagreement partly disappears when the populations match.** Rank 354 is fifth by T300 among all ranks 300–379 but first within S1; rank 362 is second overall but first within S2. Total bare coupling identifies each shell leader. It does not explain the much larger ratios of their Schur weights to those of their runners-up. Target-specific saved coupling information explains those differences far better than the tested bare energy-denominator weights.

**BARE COUPLING AS PREDICTOR: PARTIAL. DENOMINATOR WEIGHTING: MIXED.** Bare T300 has useful within-shell rank associations but substantial errors. Matched weighting worsens S1 ranks and gives tiny changes in S2; results depend on the particular denominator. Saved target-aware coupling already has near-perfect rank association before weighting. Correlation is not a causal decomposition or held-out validation.

## Sources, normalization and comparability

The completed k380 parent has epsilon_J=0.11, epsilon_LK=0.05, n1max=4, N2=N3=71, L2=11, L3=14, N4=8, Stage-1 dimension 45369 and retained count 380. Sorted ranks are zero-based. Energies and Hamiltonian amplitudes are H/h in GHz; reported attribution W is MHz. No 2pi or changed energy zero enters. Stage-1 coefficients use the previously verified Euclidean inner product and complex projections <p|O|r>. Only saved energies, small projected operators and scalar/2×2 response records were loaded; eigenvectors were not loaded.

S1 is ranks 340–359 with exact Schur P340; S2 is 360–379 with P360. Each contains all 20 ranks. W_r=sum_s |sigma_r,s| for s=0,1,2,3,7,8. The saved scalar is Re(d_r†q_r)/(p†p), with cross-rank resolvent mixing already inside q_r. W is an aggregate absolute attribution, not a signed energy or transition correction. Shell totals reproduce 14.642728 and 17.011414 MHz at published rounding. No new Schur contractions were computed.

Total bare coupling T300 sums ||A_pr n4+B_pr theta4||F² over p<300 and all eight mode-4 states. Tactual instead uses p<340 or p<360. A=8 sum_i EC[i,3] n_i and B=-2.05 x2-x3 are the production circuit conventions. The two are different partner populations. Mode-4 spacing is 15.8454070026 GHz.

Saved selector features use base P340 for both shells, three target pairs, and isolated-rank denominators at pair-mean energies. Raw response traces are unwhitened; no greedy score or refitted weighting was used. Saved exact coupling norms equal ||d_r||/sqrt(p†p), as established by inspection of analyze_k380.py:83–101. They already refer to normalized P directions; no second normalization was applied. Those exact target norms are explanatory, outcome-adjacent data, not predictors available without full-shell dressed information.

Definitions were frozen before the new calculations in RESEARCH_PLAN.md and PREDICTOR_DEFINITIONS.md. This is explicitly retrospective. Rank 328 remains a separate k340 context record and enters no correlation or recovery population.

## Main comparisons

| Shell | Predictor | Pearson raw | Pearson log10 | Spearman | Top3 recovery | Leader position |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | T300 | 0.549169 | 0.765951 | 0.809023 | 2/3 | 1 |
| S1 | Tactual | 0.402152 | 0.577582 | 0.603008 | 2/3 | 3 |
| S1 | Dgap | 0.355870 | 0.709949 | 0.718797 | 1/3 | 5 |
| S1 | Dplus | 0.512506 | 0.757911 | 0.769925 | 2/3 | 1 |
| S1 | V300 | 0.527595 | 0.767788 | 0.769925 | 2/3 | 1 |
| S1 | Vplus | 0.483594 | 0.757747 | 0.765414 | 2/3 | 2 |
| S1 | Tref | 0.542176 | 0.765780 | 0.787970 | 2/3 | 1 |
| S1 | BaseC | 0.999918 | 0.998970 | 0.996992 | 3/3 | 1 |
| S1 | BaseMinDen | 0.999943 | 0.999008 | 0.993985 | 3/3 | 1 |
| S1 | BaseTrace | 0.999973 | 0.999123 | 0.996992 | 3/3 | 1 |
| S1 | ExactC | 0.999870 | 0.999491 | 0.993985 | 3/3 | 1 |
| S1 | ExactRef | 0.999885 | 0.999462 | 0.992481 | 3/3 | 1 |
| S2 | T300 | 0.555901 | 0.684194 | 0.705263 | 3/3 | 1 |
| S2 | Tactual | 0.591245 | 0.698087 | 0.451128 | 3/3 | 1 |
| S2 | Dgap | 0.534764 | 0.688772 | 0.700752 | 3/3 | 2 |
| S2 | Dplus | 0.547142 | 0.685358 | 0.706767 | 3/3 | 1 |
| S2 | V300 | 0.556352 | 0.691991 | 0.705263 | 3/3 | 1 |
| S2 | Vplus | 0.547217 | 0.693457 | 0.711278 | 3/3 | 1 |
| S2 | Tref | 0.567342 | 0.685417 | 0.706767 | 3/3 | 1 |
| S2 | BaseC | 0.999998 | 0.998128 | 0.989474 | 3/3 | 1 |
| S2 | BaseMinDen | 0.999997 | 0.998306 | 0.990977 | 3/3 | 1 |
| S2 | BaseTrace | 0.999962 | 0.996821 | 0.990977 | 3/3 | 1 |
| S2 | ExactC | 0.999994 | 0.999764 | 0.996992 | 3/3 | 1 |
| S2 | ExactRef | 0.999998 | 0.999852 | 0.996992 | 3/3 | 1 |

The actual top-three sets are S1={354,344,343} and S2={362,379,375}. T300 chooses {354,344,341} in S1 and {362,379,375} in S2. Thus it misses rank 343 in S1, while recovering all three in S2. A good leading set does not imply accurate weights or reliable ordering of weak contributors.

### All planned predictors, including unfavorable results

Each cell below gives Spearman / recovered top3 count / position of the true dominant state. Complete raw/log Pearson, all ranks, coverage, disagreement lists and dominant-removed scores are in RANKING_ANALYSIS.csv. Every available predictor covers 20/20 states per shell; the leave-dominant-out versions cover 19/19. All values used in log correlations are strictly positive, so no pseudocount or zero filtering affects these actual populations.

| Class | Predictor | S1: rho / top3 / leader | S2: rho / top3 / leader |
| --- | --- | --- | --- |
| bare | T300 | +0.809023 / 2/3 / 1 | +0.705263 / 3/3 / 1 |
| bare | Tactual | +0.603008 / 2/3 / 3 | +0.451128 / 3/3 / 1 |
| bare | G0_11 | +0.798496 / 2/3 / 1 | +0.754887 / 3/3 / 1 |
| bare | G11_40 | +0.666165 / 1/3 / 9 | +0.735338 / 2/3 / 1 |
| bare | G40_100 | +0.697744 / 2/3 / 2 | +0.341353 / 1/3 / 4 |
| bare | G100_200 | +0.306767 / 1/3 / 19 | -0.076692 / 0/3 / 9 |
| bare | G200_300 | +0.815038 / 2/3 / 1 | +0.705263 / 3/3 / 1 |
| bare | Dgap | +0.718797 / 1/3 / 5 | +0.700752 / 3/3 / 2 |
| bare | Dplus | +0.769925 / 2/3 / 1 | +0.706767 / 3/3 / 1 |
| bare | V300 | +0.769925 / 2/3 / 1 | +0.705263 / 3/3 / 1 |
| bare | Vplus | +0.765414 / 2/3 / 2 | +0.711278 / 3/3 / 1 |
| bare | N1 | +0.403008 / 1/3 / 9 | -0.094737 / 0/3 / 5 |
| bare | N2 | +0.621053 / 1/3 / 4 | +0.712782 / 3/3 / 1 |
| bare | N3 | +0.815038 / 3/3 / 1 | +0.693233 / 3/3 / 1 |
| bare | X2 | +0.339850 / 0/3 / 8 | +0.514286 / 2/3 / 1 |
| bare | X3 | +0.786466 / 3/3 / 1 | +0.676692 / 3/3 / 1 |
| bare | Charge | +0.630075 / 1/3 / 8 | +0.679699 / 3/3 / 3 |
| bare | Phase | +0.809023 / 2/3 / 1 | +0.705263 / 3/3 / 1 |
| bare | RefInverse | +0.207519 / 0/3 / 15 | -0.225564 / 1/3 / 3 |
| bare | Tref | +0.787970 / 2/3 / 1 | +0.706767 / 3/3 / 1 |
| bare | Concentration | +0.577444 / 1/3 / 1 | +0.458647 / 2/3 / 1 |
| saved_base_target | BaseC | +0.996992 / 3/3 / 1 | +0.989474 / 3/3 / 1 |
| saved_base_target | BaseC01 | +0.932331 / 3/3 / 1 | +0.926316 / 3/3 / 1 |
| saved_base_target | BaseC23 | +0.842105 / 1/3 / 4 | +0.786466 / 3/3 / 2 |
| saved_base_target | BaseC78 | +0.969925 / 2/3 / 1 | +0.906767 / 3/3 / 1 |
| saved_base_target | BaseMinDen | +0.993985 / 3/3 / 1 | +0.990977 / 3/3 / 1 |
| saved_base_target | BaseTrace | +0.996992 / 3/3 / 1 | +0.990977 / 3/3 / 1 |
| saved_base_target | BaseTrace01 | +0.938346 / 3/3 / 1 | +0.924812 / 3/3 / 1 |
| saved_base_target | BaseTrace23 | +0.849624 / 2/3 / 3 | +0.763910 / 3/3 / 2 |
| saved_base_target | BaseTrace78 | +0.969925 / 2/3 / 1 | +0.900752 / 3/3 / 1 |
| saved_exact_target | ExactC | +0.993985 / 3/3 / 1 | +0.996992 / 3/3 / 1 |
| saved_exact_target | ExactC01 | +0.932331 / 3/3 / 1 | +0.936842 / 3/3 / 1 |
| saved_exact_target | ExactC23 | +0.846617 / 1/3 / 4 | +0.809023 / 3/3 / 2 |
| saved_exact_target | ExactC78 | +0.966917 / 2/3 / 1 | +0.921805 / 3/3 / 1 |
| saved_exact_target | ExactRef | +0.992481 / 3/3 / 1 | +0.996992 / 3/3 / 1 |

| Unavailable class | S1 | S2 | Reason |
| --- | --- | --- | --- |
| DressedChannel | INCOMPLETE | INCOMPLETE | Missing target-conditioned individual-channel contractions |
| DressedInterference | INCOMPLETE | INCOMPLETE | Missing complex dressed d_r,q_r/channel contractions |
| MatchedP360Selector | NOT APPLICABLE | INCOMPLETE | Only the P340 selector response was saved |

The N3 and X3 channel rankings can recover S1 top3 even though their amplitudes differ by orders of magnitude. N3 has only about 0.16 GHz² at each leader, compared with roughly 3200 GHz² in the coherent phase channel. A high correlation of a small channel is not evidence that it causes the energy response. No channel was selected after inspection to replace the frozen total metric.

## Sensitivity to one dominant contributor

Remove exactly rank 354 from S1 and rank 362 from S2, without changing predictors. The new observed top-three sets are {344,343,353} and {379,375,374}; these are formed from the remaining 19 states. This is an influence check, not cross-validation.

| Shell | Predictor | Raw Pearson full → removed | Spearman full → removed | Recovery full → removed |
| --- | --- | --- | --- | --- |
| S1 | T300 | 0.549169 → 0.750934 | 0.809023 → 0.777193 | 2/3 → 2/3 |
| S1 | Dgap | 0.355870 → 0.552271 | 0.718797 → 0.705263 | 1/3 → 1/3 |
| S1 | Dplus | 0.512506 → 0.718550 | 0.769925 → 0.731579 | 2/3 → 2/3 |
| S1 | Vplus | 0.483594 → 0.692352 | 0.765414 → 0.733333 | 2/3 → 2/3 |
| S1 | BaseC | 0.999918 → 0.999400 | 0.996992 → 0.996491 | 3/3 → 3/3 |
| S1 | BaseTrace | 0.999973 → 0.999531 | 0.996992 → 0.996491 | 3/3 → 3/3 |
| S1 | ExactC | 0.999870 → 0.999267 | 0.993985 → 0.992982 | 3/3 → 3/3 |
| S2 | T300 | 0.555901 → 0.743693 | 0.705263 → 0.656140 | 3/3 → 3/3 |
| S2 | Dgap | 0.534764 → 0.736737 | 0.700752 → 0.652632 | 3/3 → 3/3 |
| S2 | Dplus | 0.547142 → 0.742491 | 0.706767 → 0.657895 | 3/3 → 3/3 |
| S2 | Vplus | 0.547217 → 0.739721 | 0.711278 → 0.663158 | 3/3 → 3/3 |
| S2 | BaseC | 0.999998 → 0.999871 | 0.989474 → 0.987719 | 3/3 → 3/3 |
| S2 | BaseTrace | 0.999962 → 0.999789 | 0.990977 → 0.989474 | 3/3 → 3/3 |
| S2 | ExactC | 0.999994 → 0.999830 | 0.996992 → 0.996491 | 3/3 → 3/3 |

T300 rank association remains moderate-to-strong after removing the leader, while its raw Pearson increases. The dominant point is therefore not solely manufacturing the bare association; rather, bare norms substantially under-represent its exceptionally large relative target response. Saved base-target associations remain very high even without that point (BaseC Spearman 0.996491/0.987719). This still measures the same small retrospective finite populations.

## Why denominators do not resolve the bare mismatch

Dgap and Dplus preserve the T300 numerator and partner set, weighting by E_r-E_p and E_r-E_p+omega, respectively. V300 and Vplus are a separate matched pair using only |H_(p,0),(r,1)|². Tref weights T300 by the mean reciprocal separation from the six saved dressed reference energies, including the oscillator vacuum energy E_r+omega/2. No one of these scalar denominators is an exact dressed Schur denominator.

For S1, all three matched weighting comparisons lower Spearman: T300→Dgap/Dplus/Tref is 0.809023→0.718797/0.769925/0.787970; V300→Vplus is 0.769925→0.765414. Dgap moves the true leader to position 5 and reduces top3 recovery to 1/3. For S2, T300→Dgap is slightly worse (0.705263→0.700752), whereas Dplus/Tref increase it only to 0.706767; Vplus reaches 0.711278. These small changes persist similarly after removing the leader and do not explain the magnitude contrast.

The energy-only RefInverse baseline has Spearman +0.207519 in S1 and -0.225564 in S2. A small denominator by itself is not a successful ranking explanation. Low bare separations to intermediate partners can overweight states such as rank 340, which Dgap promotes from observed position 13 to predicted position 1.

| Shell | Saved selector pair | Minimum denominator range, GHz |
| --- | --- | --- |
| S1 | 01 | 47.309525–48.271652 |
| S1 | 23 | 43.775306–44.737434 |
| S1 | 78 | 41.125903–42.088031 |
| S2 | 01 | 48.393725–49.542310 |
| S2 | 23 | 44.859506–46.008092 |
| S2 | 78 | 42.210103–43.358689 |

These minimum isolated-rank denominators vary narrowly within a shell; they are bounds/context, not full mode-resolved denominator distributions. Saved full-shell minimum absolute denominators across targets are 40.892638–47.087057 GHz for S1 and 42.021538–48.209219 GHz for S2. These shell quantities are not rank-resolved. No near-resonant leader explanation follows.

BaseC already has Spearman 0.996992/0.989474; replacing it with the saved denominator-resolved BaseTrace gives 0.996992/0.990977. Thus the major gain is associated with target conditioning, rather than the incremental denominator refinement measured by these comparisons. The denominator still sets the physical scale of a virtual response; this ranking result does not make it dispensable in an effective Hamiltonian.

## Dominant contributors: quantified comparisons

### Rank 354: S1

Rank 354 carries 9.784721460 MHz (66.823079% of shell W). Compared with runner-up rank 344, its T300 is only 1.028751× larger, but W is 6.043333× larger. Saved BaseC differs by 5.699414× and ExactC by 5.686591×. Mean reciprocal reference separation differs by 0.988374×. This supports target-specific coupling as the measured explanation for the scale contrast, rather than an unusually favorable denominator.

Targets 7/8 account for 94.105375% of this rank's absolute attribution, compared with 31.709068% for the runner-up. The saved exact normalized couplings to targets 7/8 are 0.578108/0.613442 GHz.

Its strongest bare partner is rank 224, carrying 57.100352% of T300. Top three partners carry 86.880385%; effective partner count is 2.565026. The p=200–299 group contributes 99.540871% of T300. This concentration is descriptive; its stand-alone concentration predictor has only Spearman 0.577444.

The coherent phase channel contributes 99.192338% of the all-oscillator Frobenius strength. Charge-channel cancellation is -59.640261% of its incoherent channel sum; phase-channel cross terms add 8.641577%. For the vacuum matrix element, charge-phase cross terms add 17.774846% of its incoherent sum. These are exact bare-block statements; they do not identify dressed target-channel interference.

Saved BaseTrace is 7.623278 MHz, versus W=9.784721 MHz. Its smaller magnitude is not a prediction of an exact signed shift: the isolated-rank/base-vector approximation omits full-shell mixing, energy arguments and P-vector changes. No fitted rescaling was used. Both use P340 but different target vectors/energy arguments.

### Rank 362: S2

Rank 362 carries 15.138873587 MHz (88.992448% of shell W). Compared with runner-up rank 379, its T300 is only 1.011666× larger, but W is 12.240280× larger. Saved BaseC differs by 12.005851× and ExactC by 11.886569×. Mean reciprocal reference separation differs by 1.024754×. This supports target-specific coupling as the measured explanation for the scale contrast, rather than an unusually favorable denominator.

Targets 7/8 account for 95.377070% of this rank's absolute attribution, compared with 65.973187% for the runner-up. The saved exact normalized couplings to targets 7/8 are 0.726947/0.770544 GHz.

Its strongest bare partner is rank 230, carrying 77.887779% of T300. Top three partners carry 90.028674%; effective partner count is 1.621142. The p=200–299 group contributes 99.414820% of T300. This concentration is descriptive; its stand-alone concentration predictor has only Spearman 0.458647.

The coherent phase channel contributes 99.138618% of the all-oscillator Frobenius strength. Charge-channel cancellation is -54.362363% of its incoherent channel sum; phase-channel cross terms add 7.961962%. For the vacuum matrix element, charge-phase cross terms add 18.039010% of its incoherent sum. These are exact bare-block statements; they do not identify dressed target-channel interference.

Saved BaseTrace is 8.543171 MHz, versus W=15.138874 MHz. Its smaller magnitude is not a prediction of an exact signed shift: the isolated-rank/base-vector approximation omits full-shell mixing, energy arguments and P-vector changes. No fitted rescaling was used. The selector uses P340 while exact S2 uses P360, adding a retained-reference mismatch.

## Strong, weak and disagreeing states

| Shell | Subset | Rank | W, MHz | T300 position | Dplus position | BaseC position | BaseTrace position |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | strongest three | 354 | 9.78472146 | 1 | 1 | 1 | 1 |
| S1 | strongest three | 344 | 1.61909364 | 2 | 3 | 2 | 2 |
| S1 | strongest three | 343 | 1.35737658 | 4 | 4 | 3 | 3 |
| S1 | weakest three | 358 | 0.000611117287 | 9 | 8 | 19 | 19 |
| S1 | weakest three | 349 | 0.000541878211 | 18 | 18 | 18 | 18 |
| S1 | weakest three | 346 | 0.00033347078 | 19 | 19 | 20 | 20 |
| S2 | strongest three | 362 | 15.1388736 | 1 | 1 | 1 | 1 |
| S2 | strongest three | 379 | 1.23680777 | 2 | 2 | 2 | 2 |
| S2 | strongest three | 375 | 0.444073525 | 3 | 3 | 3 | 3 |
| S2 | weakest three | 363 | 0.00120375556 | 7 | 7 | 19 | 18 |
| S2 | weakest three | 370 | 0.000836413328 | 16 | 16 | 18 | 19 |
| S2 | weakest three | 369 | 0.00041215757 | 17 | 17 | 20 | 20 |

| Shell | Predictor | Three largest disagreements (rank: predicted / observed) |
| --- | --- | --- |
| S1 | T300 | 358: 9 / 18; 345: 15 / 7; 340: 6 / 13 |
| S1 | Dgap | 340: 1 / 13; 358: 9 / 18; 345: 12 / 7 |
| S1 | Dplus | 358: 8 / 18; 340: 5 / 13; 345: 15 / 7 |
| S1 | Vplus | 358: 8 / 18; 340: 5 / 13; 345: 14 / 7 |
| S1 | BaseTrace | 340: 12 / 13; 349: 18 / 19; 357: 13 / 12 |
| S2 | T300 | 363: 7 / 18; 368: 5 / 16; 366: 19 / 13 |
| S2 | Dgap | 363: 7 / 18; 368: 5 / 16; 366: 19 / 13 |
| S2 | Dplus | 363: 7 / 18; 368: 5 / 16; 366: 19 / 13 |
| S2 | Vplus | 363: 7 / 18; 368: 6 / 16; 366: 19 / 13 |
| S2 | BaseTrace | 365: 15 / 17; 360: 10 / 11; 364: 6 / 7 |

Examples: rank 358 is ninth by T300 but eighteenth by S1 attribution. S2 ranks 368 and 363 are fifth and seventh by T300 but sixteenth and eighteenth by attribution. Broad bare norms count couplings to directions that need not participate appreciably in the six tracked targets. The saved base-target couplings largely resolve these discrepancies; their remaining precise dynamical origin cannot be assigned to a particular dressed operator channel without missing contractions.

## What complex interference is and is not supported

The saved complex projected operators permit exact bare charge-channel and spatial-channel coherent sums. Over the complete eight-state oscillator Frobenius trace, charge–phase cross terms vanish because Tr(n4†theta4)=0; individual oscillator matrix elements can still interfere. This distinction is validated synthetically and numerically. The reported bare cancellation/enhancement percentages are normalized by the corresponding positive incoherent sums.

Exact Schur records contain only scalar signed attributions and coupling norms, not per-channel complex d_r,q_r. The saved selector 2×2 matrices retain within-target-pair complex response, but do not decompose it into the five operator channels or full-shell cross-rank paths. Therefore a dressed interference explanation of residual rank errors is INCOMPLETE. Absolute-only attribution must not be used to infer its sign or pathway.

## Rank 328: historical context only

Historical parent k340/new_S4, exact P320: rank 328 has aggregate absolute weight 3.237027672 MHz and 79.454192% of its shell. Its saved selector response used P300. The k380 fingerprint uses a different parent, with the previously authenticated near-unit same-rank overlap as context. Historical norms/traces are preserved in CONTEXT_RANK328.json; no correlation, top3 test or denominator comparison mixes this historical attribution into S1/S2.

## Scientific conclusion and next question

Total bare coupling is a useful coarse within-shell screening variable for these finite data, but it cannot supply target-conditioned virtual weights. The global fifth/second rankings do not imply failure to find the shell leaders: both are first in their own shell. The remaining disagreement concerns magnitudes, partner groups and weak-state ordering.

Saved target-aware norms and isolated-rank responses quantitatively support future physics-informed retention. Their strong retrospective association survives removal of each dominant state. This is not out-of-sample validation, a newly validated selector, an exact correction model, an infinite-tail bound or a basis-convergence result. Individual bare channels and partner concentration are correlated structural diagnostics, not established causal mechanisms.

**Next research question:** can a target-aware selector frozen at P340 retain its ordering accuracy on an independently authorized, previously unseen shell, with unchanged targets, thresholds and a predeclared top3/weak-state evaluation? The current analysis cannot answer that from reused S1/S2 data. This is a recommendation only; no new parent, dressed solve or benchmark is authorized or executed here.

**V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.**

## Figures, validation and reproduction

[Coupling versus attribution](coupling_vs_attribution.pdf) shows all 20 states per shell against T300, matched Dplus, and saved BaseTrace, with log axes and no fitted lines. Orange marks the dominant contributor. [Ranking comparison](predictor_rank_comparison.pdf) shows all within-shell positions for observed W, T300, Dplus, BaseC and BaseTrace. Companion PNGs use 200 dpi; PDFs contain vector graphics. Both figures remain solely in this new sandbox directory. They do not extend the existing public package.

Eight synthetic tests passed. All prior T300/Vplus/W comparisons agree exactly (maximum absolute difference 0.0). Operator Hermiticity and selector raw-matrix traces pass. All 40 ranks and 240 target records are present; all 152 planned full/removed available/missing evaluation rows are retained. Six scientific/provenance outputs reproduce byte for byte in a second process.

Main process: 0.897203s, peak 154.203 MiB; repeat: 0.898607s, peak 154.656 MiB. Limits were 180s/512MiB with a 50-ms resident watchdog and single-thread BLAS. Zero Stage-1 eigenvectors loaded, zero eigensolves, zero new dressed contractions.

A targeted 104-file input/preservation check passed, including prior reports, consumed checkpoint files, sealed manifests/seals, protected user edits and Git index/status. The earlier 456-file audit was not repeated. See SOURCE_AUTHENTICATION.json and SOURCE_PROVENANCE.md.

Reproduce from repository root (existing environments; no dependency installation):

```shell
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_coupling_vs_schur_2026_10_21/analyze.py
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_coupling_vs_schur_2026_10_21/verify_reproduction.py
/usr/local/bin/Rscript --vanilla research/sandbox/k380_coupling_vs_schur_2026_10_21/figures.R research/sandbox/k380_coupling_vs_schur_2026_10_21
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_coupling_vs_schur_2026_10_21/build_report.py
```

The figure script uses base R PDF and native Quartz PNG graphics. PDFs were rendered separately with macOS sips for visual QA because Poppler was unavailable. No installation was needed. The first Cairo PNG attempt encountered missing existing X11 libraries; Quartz produced the final PNGs successfully. Final validation caught and removed a new implicit Rplots.pdf at the repository root from that failed export. The script now fails closed on any unexpected implicit graphics device. Final additions are confined to this sandbox. Numerical outputs are independent of plotting.
