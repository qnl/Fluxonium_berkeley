> Publication copy: three coefficient expressions are formatted as code so their brackets are not interpreted as file links. Definitions and numerical values are unchanged. Original SHA-256 is recorded in ../../publication/MANIFEST.json.

# Frozen predictor definitions

All energies are H/h in GHz, not angular frequencies. No2pi enters. W is MHz.
Stored Stage-1 projections O_pr=<p|O|r> have normalized Euclidean eigenvectors.
Mode4 uses production N4=8 oscillator operators n4 and theta4. From the verified
production coefficients, A_pr=8 sum_i `EC[i,3](n_i)_pr` and
B_pr=`K[0,2](x2)_pr`+`K[1,2](x3)_pr`. H_pr=A_pr n4+B_pr theta4 (8×8).
Form complex sums before squared magnitudes. Absolute phases are arbitrary;
cross terms for the same p,r are gauge invariant. All rankings descend.

Define t_pr=||H_pr||F² (GHz²), v_pr=|H_(p,0),(r,1)|² (GHz²),
delta_pr=E_r-E_p>0, omega=w4[1]-w4[0], and P_actual=340(S1),360(S2).
Reference targets are exactly s∈{0,1,2,3,7,8}. E_s is the saved dressed energy
at the shell's lower prefix P_actual, using the same Hamiltonian zero. For bare
reference separation d_rs=E_r+w4[0]-E_s, use the mode4 vacuum energy explicitly.
Require positive denominators; fail closed, never clip or take absolute values.
These d_rs are bare-product-to-dressed-reference separations, not eigenvalues of
H_QQ-E_s. None is an exact dressed Schur denominator.

## Bare predictors (21)

| ID | Definition | Units / normalization |
| --- | --- | --- |
| T300 | sum_(p<300) t_pr | GHz²; same definition as prior total bare strength |
| Tactual | sum_(p<P_actual) t_pr | GHz²; own exact partition's full retained prefix |
| G0_11 | sum_(0≤p<11) t_pr | GHz² |
| G11_40 | sum_(11≤p<40) t_pr | GHz² |
| G40_100 | sum_(40≤p<100) t_pr | GHz² |
| G100_200 | sum_(100≤p<200) t_pr | GHz² |
| G200_300 | sum_(200≤p<300) t_pr | GHz² |
| Dgap | sum_(p<300) t_pr/delta_pr | GHz; matches T300 numerator and partners |
| Dplus | sum_(p<300) t_pr/(delta_pr+omega) | GHz; matches T300 numerator and partners |
| V300 | sum_(p<300) v_pr | GHz²; mode4 vacuum numerator |
| Vplus | sum_(p<300) v_pr/(delta_pr+omega) | GHz; exact prior bare-vacuum proxy |
| N1 | sum_(p<300) ||8 EC[0,3] n1_pr n4||F² | GHz²; individual weighted operator channel |
| N2 | same, index1 and n2 | GHz² |
| N3 | same, index2 and n3 | GHz² |
| X2 | sum_(p<300) ||K[0,2] x2_pr theta4||F² | GHz² |
| X3 | same, K[1,2] and x3 | GHz² |
| Charge | sum_(p<300) ||A_pr n4||F² | GHz²; coherent charge sum |
| Phase | sum_(p<300) ||B_pr theta4||F² | GHz²; coherent spatial sum |
| RefInverse | mean_s 1/d_rs | GHz^-1; energy-only baseline |
| Tref | T300*mean_s 1/d_rs | GHz; matched common-reference denominator weighting |
| Concentration | sum_(p<300)(t_pr/T300)² | dimensionless IPR; high=concentrated |

Additionally save descriptive maximum-partner fraction, top-three fraction,
effective partner count1/Concentration and strongest partner rank. These are not
extra ranked candidate predictors. Save channel cross terms Charge-(N1+N2+N3),
Phase-(X2+X3), T300-Charge-Phase and analogous vacuum cross terms. Normalize
each by its positive incoherent sum; report null if the normalization is zero.
Individual channel strengths do not add to the coherent answer without cross
terms. No dressed-state interference is inferred from these bare quantities.

## Saved base-target predictors (9)

The sealed selection.json has three groups g=(0,1),(2,3),(7,8), baseP340 for both
S1/S2. Let c_rg be the saved `coupling_norm_GHz` (Frobenius norm of coupling of
one isolated rank's8 rows to two base dressed columns), m_rg the saved minimum
positive isolated-rank denominator, and R_rg the saved complex unwhitened2×2
positive response B†(H_rr-mu_g)^-1 B, unitsGHz. No calculation of this response is
repeated. No whitened greedy-selector score is substituted for a scalar trace.

| ID | Definition | Units |
| --- | --- | --- |
| BaseC | sum_g c_rg² | GHz² |
| BaseC01 | c_r,01² | GHz² |
| BaseC23 | c_r,23² | GHz² |
| BaseC78 | c_r,78² | GHz² |
| BaseMinDen | sum_g c_rg²/m_rg | GHz; upper bound for sum positive isolated responses, not exact trace |
| BaseTrace | sum_g Re trace(R_rg) | GHz |
| BaseTrace01 | Re trace(R_r,01) | GHz |
| BaseTrace23 | Re trace(R_r,23) | GHz |
| BaseTrace78 | Re trace(R_r,78) | GHz |

The target information came from an earlier P340 dressed solve, is already saved,
and is available without new reconstruction. It is more informed than any bare
group of Stage-1 ranks. S2 exact attribution uses P360, so its reference mismatch
must remain explicit. Do not treat it as new independent predictive validation.

## Saved exact-target explanatory diagnostics (5)

Let a_rs be the saved exact shell `coupling_norm_GHz` for rankr and targets.
Code inspection of `analyze_k380.py:83–101` establishes
a_rs=||d_rs||/sqrt(p_s†p_s), d_s=H_QP p_s, using the shell's full-state P
component. Thus this is coupling to the normalized direction of that component.
Its stored normalization is retained; do not divide by the P probability again.

| ID | Definition | Units |
| --- | --- | --- |
| ExactC | sum_s a_rs² | GHz² |
| ExactC01 | sum_(s=0,1) a_rs² | GHz² |
| ExactC23 | sum_(s=2,3) a_rs² | GHz² |
| ExactC78 | sum_(s=7,8) a_rs² | GHz² |
| ExactRef | sum_s a_rs²/d_rs | GHz; explanatory bare denominator proxy |

These use already saved full-shell target information. Their association with W
is explanatory and outcome-adjacent, not a screening test without dressed data.
Also save W fractions by pair, per-target signed sigma, coupling norms and rank
probabilities. Do not derive interference or signed transition shifts from W.

## Missing predictors (3): explicitly INCOMPLETE

`DressedChannel`: target-conditioned decomposition by the five individual
operator channels needs unsaved dressed vectors/contractions.
`DressedInterference`: exact cross-channel/cross-rank full-shell contractions
need unsaved complex d_r,q_r and operator-channel contributions.
`MatchedP360Selector`: target-aware isolated-rank P360 response for S2 was not
saved; cannot reconstruct it. For S1 this is not applicable (P340 is matched),
and no additional P360 predictor is invented for S1.

## Evaluation conventions

Every available predictor is evaluated separately in S1 and S2, with full20 and
dominant-removed19 populations. Pearson on raw values and on positive log10
values; Spearman using average ties; top3 recovery=count of intersecting top3/3;
dominant rank position; mean/max absolute rank error. Deterministic sorting uses
descending value then ascending Stage-1 rank. Missing outputs are blank/null,
never zero. Scientific correlations and rankings are unit-invariant under a
uniform positive scale; no within-shell normalization is fitted. Attribution
totals/shares remain separate from predictor quantities and units.
