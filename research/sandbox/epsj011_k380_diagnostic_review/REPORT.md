# k380 post-execution scientific diagnostic

SOURCE AUTHENTICATION: PASS. SPECTRAL DECOMPOSITION: COMPLETE. NEW EIGENSOLVES: ZERO. PRODUCTION MODIFICATIONS: NONE.

## Official results remain unchanged

Execution and checkpoint authentication succeeded; the frozen evaluator runs successfully. Retained-basis convergence is FAIL, tail classification is D_NONMONOTONIC_UNRESOLVED, and the secondary selector is PASS. Earlier independent selector BORDERLINE, multi-target PARTIAL, and cross-parameter mechanism SUPPORTED classifications are not revised.

## Observation: the spectral tail does not shrink uniformly

| Transition | 300→320 | 320→340 | 340→360 | 360→380 |
| --- | --- | --- | --- | --- |
| f01 | -0.000814631 | +0.030597445 | -0.016793563 | -0.016442511 |
| f07 | -0.434168321 | -1.032701369 | -4.529117818 | -6.131698601 |
| f08 | -0.714571720 | -1.269238472 | -5.594546068 | -6.738197538 |
| f78 | -0.280403399 | -0.236537103 | -1.065428251 | -0.606498937 |
| f81 | -0.713757089 | -1.299835917 | -5.577752506 | -6.721755027 |
| f72 | -0.193826526 | -0.726502779 | -3.939156098 | -6.109228464 |
| f21 | -0.239527164 | -0.336796034 | -0.573168158 | -0.006027627 |
| f03 | -0.364913815 | -0.345345689 | -0.738008121 | -0.075824851 |
| f38 | -0.349657904 | -0.923892783 | -4.856537948 | -6.662372687 |

Signed entries above are upper-minus-lower MHz, opposite to frozen comparison component signs. All 36 rows, absolute magnitudes, growth ratios, cumulative changes, raw state shifts, pair-common shifts, excitation shifts, and frozen flags are in SPECTRAL_DECOMPOSITION.csv. No rounding from previous prose was used.

| Shell | Maximum absolute carrier correction (MHz) |
| --- | --- |
| k300_to_k320 | 0.714571720 |
| k320_to_k340 | 1.299835917 |
| k340_to_k360 | 5.594546068 |
| k360_to_k380 | 6.738197538 |

Five of nine carrier increments grow in the final shell; four decrease (f01,f78,f21,f03). Across 66 mandatory components, 42 grow and 24 are nonincreasing under the frozen test. There are no meaningful carrier sign reversals or two-shell carrier cancellation flags between the two new shells. Within-comparison f78 cancellation is severe in both new shells and cumulative k340→k380. Cumulative k340→k380 has seven practical FAIL components. These facts preserve the official FAIL/D verdict even though selected corrections shrink.

### Common and differential state-energy changes

Raw eigenvalue shifts use the same saved Hamiltonian energy zero. A pair-common shift is (deltaEi+deltaEj)/2; its differential is deltaEj−deltaEi. Excitation shifts subtract deltaE0 and are invariant to a shared energy-zero change. The CSV includes both conventions so a raw common-mode shift is not confused with stable excitations.

| Comparison | deltaE0 | deltaE7 | deltaE8 | Six-target common mean |
| --- | --- | --- | --- | --- |
| k300_to_k320 | -0.024104247 | -0.458272568 | -0.738675966 | -0.316572627 |
| k320_to_k340 | -0.108378376 | -1.14107974 | -1.37761685 | -0.595526155 |
| k340_to_k360 | -0.23575675 | -4.76487457 | -5.83030282 | -2.1471613 |
| k360_to_k380 | -0.274618082 | -6.40631668 | -7.01281562 | -2.43872369 |

## f78 cancellation and selectivity

| k | f01 | f07 | f08 | f78 | f78−f01 |
| --- | --- | --- | --- | --- | --- |
| 340 | 16.035220517 | 6183.264956229 | 6200.013677057 | 16.748720828 | 0.713500311 |
| 360 | 16.018426954 | 6178.735838411 | 6194.419130989 | 15.683292577 | -0.335134377 |
| 380 | 16.001984443 | 6172.604139810 | 6187.680933451 | 15.076793640 | -0.925190803 |

All quantities are MHz. f78=f08−f07 is reconstructed directly from saved frequencies. The maximum identity residual is 6.96331881e-13 MHz.

For k340→k360, deltaf07=−4.529117818 and deltaf08=−5.594546068 MHz, giving deltaf78=−1.065428251 MHz. For k360→k380, deltaf07=−6.131698601 and deltaf08=−6.738197538 MHz, giving deltaf78=−0.606498937 MHz. The latter is only 4.712539% of the sum of the absolute state 7/8 excitation errors. It is a cancellation-sensitive splitting, not evidence of convergence of either state.

f78−f01 changes from +0.713500311 through −0.335134377 to −0.925190803 MHz as the retained prefix changes. All f78 values remain positive: this is a reversal in the relative ordering of two transition frequencies in finite truncations, not an observed physical level crossing as a device parameter changes. Any future frequency-selective control study must resolve uncertainty in both carriers and their difference before using the small separation as a reliable selectivity scale. No gate fidelity or physical-port conclusion follows.

## Recurring concentrated off-resonant contributions

| Rank / parent | Stage-1 energy (GHz) | Absolute rank weight (MHz) | Shell fraction |
| --- | --- | --- | --- |
| 328 / k340 | 45.9914977 | 3.23702767 | 79.454192% |
| 354 / k380 | 47.385136 | 9.78472146 | 66.823079% |
| 362 / k380 | 47.7349448 | 15.1388736 | 88.992448% |

Historical rank328 is from the authenticated k340 parent. Ranks354/362 and all four spectral increments above are from k380. The k380 common prefixes reproduce historical spectral corrections closely, but separate parents must not be treated as bitwise identical. RANK_MECHANISM.md gives exact reference bases, signed contributions, pair sensitivities, selector response matrices, denominator conventions, and rank tracking.

## Exact Schur audit and second-order limitations

For each target, deltaE = sigma(E,p) + relaxation. Rank sums and, where saved, dressed-mode sums reproduce sigma within the unchanged frozen Schur identity tolerance. Reported q reconstruction and Schur residuals also pass. Per-state audit residuals are recorded in DIAGNOSTICS.json.

At fixed normalized p direction, d sigma/dE = −sum_m |g_m|²/(E−lambda_m)² = −||q||²/||p||². For S1/S2 this derivative is evaluated by scalar algebra from saved modes and independently from Q probability. It is not the total derivative of an eigenvalue or a scanned effective Hamiltonian, and it does not capture changing p. Large self-energy magnitudes therefore do not imply strong local energy dependence or proximity to a pole.

| Shell / target | Exact shift | Exact sigma | P relaxation | Saved fixed-vector second order | Second order−exact shift |
| --- | --- | --- | --- | --- | --- |
| k380/S1/7 | -4.76487457 | -5.42212458 | 0.657250014 | -4.18860326 | 0.576271307 |
| k380/S1/8 | -5.83030282 | -6.63889707 | 0.808594249 | -5.1213248 | 0.708978021 |
| k380/S2/7 | -6.40631668 | -7.49276359 | 1.0864469 | -5.47897491 | 0.927341771 |
| k380/S2/8 | -7.01281562 | -8.15780412 | 1.1449885 | -6.03045001 | 0.982365609 |

All entries above are MHz. The fixed-vector approximation gets the scale and sign but underestimates important shifts by about 10–15%, reaching almost 1 MHz. It is useful as a ranking feature, not an adequate stand-alone correction at the frozen accuracy scales. Exact sigma alone is also not deltaE: positive P relaxation cancels roughly 10–15% of |sigma|. Saved nonadditivity reaches 0.220605779 MHz for an S2 carrier; explicit cross-half coupling and P-mediated contributions are separately recorded. Independent-rank additivity is not an exact model of observed shifts.

Dressed shell-mode concentration is reported separately from rank concentration. Its eigenmode weights include H_QQ mixing; dominant rank identity must not be substituted for a particular dressed mode. The diagnostic JSON records aggregate mode concentration and leading modes without a new diagonalization.


### Accuracy and perturbative control are different questions

The new-shell fixed-vector relative shift discrepancies span 9.643–14.475%. They satisfy the earlier shell-attribution methodology's numerical 20% error, eta<0.10 and Q-probability<0.01 checks. Thus these data do not demonstrate a breakdown of off-resonant perturbation theory. They demonstrate insufficient accuracy for a stand-alone correction at the present sub-MHz/MHz criteria and the need to include relaxation and nonadditivity. No new official perturbative verdict is introduced.

The maximum magnitude of the fixed-p self-energy slope is 8.9993e-5 in S1 and 1.13563e-4 in S2. This supports weak local dependence on the energy argument at the saved eigenstates, while leaving changes of p and matrix-valued mixing unresolved. Full-shell near-resonant modal weight is zero in both shells.

The top-three aggregate dressed-mode fractions are 50.300195% (S1) and 68.171149% (S2), with effective mode counts 10.204358 and 5.610445. These are much less concentrated than the corresponding rank fractions 87.150372% and 98.873351%. A single Stage-1 rank includes eight mode-4 product states; it is not one dressed shell mode or one physical resonance. Stage-1 energy alone is not a Schur denominator: denominators include the shell Hamiltonian and target energy.

For S2 f78, rank362's scalar difference is −0.919200877 MHz, other ranks contribute +0.254160343 MHz in this scalar partition, and differential P relaxation contributes +0.058541597 MHz. Together they recover −0.606498937 MHz. This illustrates why assigning that exact transition shift to rank362 alone is invalid. All target-pair decompositions are in SCHUR_PAIR_DECOMPOSITION.json.

## Why stable operators do not imply stable spectra

Demonstrated: all 12 hard gates pass; target tracking passes; all 32 operator components and all 3 pathway-score components pass strict thresholds in each primary comparison. The cumulative minimum target overlap is 0.999762806, maximum principal angle 1.256322 degrees, maximum operator relative error 1.369015%, and primary pathway rank remains 1. The relative response of states 7/8 to newly retained ranks is substantially larger than that of states 0/1.

Interpretation: small off-resonant admixtures can leave vector overlaps and low-state operator matrix elements nearly unchanged while shifting diagonal energies by MHz. Transition errors measure differences of these state-dependent shifts; they need not follow the stability of matrix elements. This interpretation is supported by small saved shell probabilities and the exact Schur identity. A full effective-Hamiltonian mixing mechanism is not reconstructed from absent matrices.

## Scientific decision

k380 does not establish improved retained-basis convergence because newly included shells produce larger differential corrections and concentrated virtual responses recur. This suggests that contiguous energy-prefix retention can be inefficient for these targets; the frozen secondary selector PASS provides independent within-parent evidence for testing response-aware retention. It does not establish a tail bound or rescue convergence.

Before another parent, existing data can test which target pairs carry the response, quantify perturbative bias and relaxation, verify cancellation, compare frozen equal-budget selector/control results, and benchmark any proposed scalar approximation against already saved exact shifts. It cannot determine unseen ranks or fully validate a new matrix-valued effective model.

A larger solve would be justified only by a reviewed prospective protocol stating what unresolved hypothesis it discriminates, fixed predictions/stopping rules, how new shells are selected without outcome tuning, unchanged or separately preregistered criteria, and renewed resource checks. It must not be justified by an assumed plateau or automatic continuation. NUMERICAL_STRATEGY.md ranks method investigations by their evidential value and validation gaps, not price alone.

## Incomplete analyses

- Exact full-shell off-diagonal self-energy matrices and their energy scans require saved dressed vectors/resolvent contractions that are absent.
- Rank354/362 cross-parent Stage-1 identity stability cannot be tested against k340.
- Rank328 historical individual mode lists and fixed-vector second-order diagnostics were not saved.
- No bound or prediction for omitted ranks above379 is established.

No figures were necessary. No large arrays were copied. Sources and preservation evidence are in SOURCE_PROVENANCE.md and SOURCE_HASHES.json.

NEXT EXPERIMENT (recommendation only, NOT AUTHORIZED): preregister a finite-parent benchmark of a Schur-corrected retained model against the already frozen selector and controls, with held-out shells and no refitting. First specify how missing dressed-vector contractions will be obtained and separately authorize any required eigensolve. No such benchmark is executed here.

GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
