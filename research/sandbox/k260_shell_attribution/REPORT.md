# Gridium k260 shell-resolved attribution


## Result

DATA FEASIBILITY: **SUFFICIENT** for exact attribution inside the saved finite
k260 parent. EFFECTIVE HAMILTONIAN: **VALIDATED**. DOMINANT MECHANISM: **C**.

The large 220->240 spectral change is an off-resonant virtual correction with
strong Stage-1 rank concentration and a broader dressed-shell spectral spread.
It is not explained by a near-resonant omitted shell state under the frozen
diagnostic definition. Ranks 230, 223, and 225 dominate states 7/8; the central
shell couples substantially more strongly than its predecessors. Finite
P-space relaxation is important. The original retained-basis verdict remains
FAIL, and gate work and spatial convergence remain HOLD.

The C label follows the preregistered rule: state 7 is B (distributed by the
modal criterion), state 8 is C (intermediate modal concentration), so their
joint mechanism is C. Here “mixed” means rank concentration plus distributed
off-resonant dressing and relaxation. It does **not** mean a mixture of
resonant and off-resonant terms; the measured near-resonant fraction is zero.

## Preregistration and reproducibility

`methodology.md` was written and hashed before new attribution calculations.
Its SHA-256 is
`921f264e6e62c6233f1906592f282f28ab1d389aa055417c47e05ac62303ec80`.
The diagnostic checks this hash. No existing convergence threshold was changed.

All ranks are zero-based saved Stage-1 indices. Energies and couplings are in
GHz; shift tables are in MHz. Full precision is in `summary.json`,
`attribution_tables.csv`, `rank_attributions.csv`, and `shell_modes.csv`.

Reproduction command (Stage-2 finite-block work only):

```bash
bash research/sandbox/k220_environment_reconstruction/python.sh \
  research/sandbox/k260_shell_attribution/attribute.py
```

`finalize_report.py` renders the stored results without any diagonalization.

## Data feasibility

Stored eigenvalues (260) and eigenvectors (45369 x 260) are available.
The five projected operators n1/n2/n3/x2/x3 each have shape 260 x 260.
The stored physical coefficients and mode-4 cutoff suffice to reconstruct
the complete Stage-2 Hamiltonian and grid_phi=(x2/2+x3) tensor I8.
Stage-2 Hamiltonian components and retained vectors are not separately saved,
but are reconstructible by the existing production formula. Coupling blocks
are exact slices of that reconstructed finite matrix. No physical couplings
were guessed. No Stage-1 Hamiltonian was assembled or diagonalized.

The reconstructed lowest eleven energies and absolute operator entries match
the saved nested analysis. Saved Stage-1 real-space vectors are unnecessary
for this finite-block attribution. Ranks beyond 259 remain unavailable; no
claim is made about their tail or any infinite-basis limit.

## Effective Hamiltonian and numerical validation

P has dimension 220 x 8 = 1760, Q1 and Q2 each 20 x 8 = 160, and their union
has dimension 320. The full Stage-2 dimension is 2080. Tensor ordering is
Stage-1 rank followed by mode-4 occupation.

For an exact dressed eigenpair with components p,q,

    q = (E I - H_QQ)^(-1) H_QP p
    H_eff(E) = H_PP + H_PQ (E I - H_QQ)^(-1) H_QP

The plus sign is correct; shell levels above E produce negative denominators
and lower energies. Q1-Q2 coupling is included when eliminating their union.
The numerical implementation uses the spectral resolvent, without forming an
explicit inverse. The exact shift identity is

    E - E_old = <p|Sigma(E)|p>/<p|p>
                + <p|(H_PP-E_old)|p>/<p|p>.

The second term is P-space relaxation. Rank contributions sum to self energy,
not to the entire energy shift until relaxation is included. These are signed
contributions in the saved basis and are not invariant under arbitrary rank
rotations. Dressed-shell spectral participation supplies a complementary
description (individual degenerate eigenmode shares also depend on basis
choice within any degenerate shell subspace).

This is an exact **a posteriori reconstruction at the observed dressed E**,
not an independent root prediction. The separately reported second-order
estimate uses the old eigenstate and old energy and is genuinely approximate.
All low-lying states are diagonalized together, preserving finite-block
near-degenerate mixing; unsafe assignments would have blocked individual
claims. All required assignments, including counterfactuals, are safe.


| Check | Measured maximum | Frozen limit |
|---|---|---|
| Saved energy error (GHz) | 1.563e-13 | 1e-10 |
| Saved |grid_phi| error | 5.925e-13 | 1e-10 |
| Stage-2 eigenpair residual (GHz) | 2.870e-13 | 1e-9 |
| Adjacent Schur residual (GHz) | 2.771e-13 | 1e-9 |
| Adjacent Q-vector error | 9.664e-16 | 1e-8 |
| Adjacent energy-identity error (MHz) | 9.407e-11 | 1e-6 |
| Combined Q1+Q2 identity error (MHz) | 5.146e-11 | 1e-6 |


## State shifts across every adjacent shell

All values are MHz. Exact reconstructed shifts equal observed shifts to the errors above. SO is the old-state second-order estimate; its error is SO minus observed.


| Step | State | Observed | SO | SO error | Exact self energy | P relaxation | Mechanism |
|---|---|---|---|---|---|---|---|
| 180->200 | 0 | -0.551468 | -0.497310 | 0.054159 | -0.611679 | 0.060211 | B |
| 180->200 | 1 | -0.484607 | -0.437163 | 0.047444 | -0.537309 | 0.052703 | B |
| 180->200 | 2 | -2.049351 | -1.811321 | 0.238031 | -2.319112 | 0.269761 | B |
| 180->200 | 3 | -2.814520 | -2.489129 | 0.325391 | -3.183307 | 0.368787 | B |
| 180->200 | 7 | -1.846086 | -1.618955 | 0.227130 | -2.106785 | 0.260700 | B |
| 180->200 | 8 | -3.730688 | -3.260401 | 0.470287 | -4.272871 | 0.542183 | B |
| 200->220 | 0 | -0.156569 | -0.145865 | 0.010704 | -0.168158 | 0.011589 | B |
| 200->220 | 1 | -0.172506 | -0.160750 | 0.011756 | -0.185237 | 0.012731 | B |
| 200->220 | 2 | -3.949936 | -3.455429 | 0.494507 | -4.517247 | 0.567310 | C |
| 200->220 | 3 | -4.136196 | -3.622111 | 0.514085 | -4.726082 | 0.589886 | C |
| 200->220 | 7 | -1.056980 | -0.966564 | 0.090416 | -1.158025 | 0.101045 | B |
| 200->220 | 8 | -1.870929 | -1.690905 | 0.180024 | -2.073009 | 0.202080 | B |
| 220->240 | 0 | -1.409978 | -1.215148 | 0.194830 | -1.637021 | 0.227043 | C |
| 220->240 | 1 | -1.454042 | -1.250014 | 0.204028 | -1.691993 | 0.237951 | C |
| 220->240 | 2 | -2.927255 | -2.578744 | 0.348511 | -3.324991 | 0.397737 | C |
| 220->240 | 3 | -3.419509 | -3.017562 | 0.401947 | -3.877102 | 0.457593 | C |
| 220->240 | 7 | -28.052236 | -22.601340 | 5.450896 | -34.844483 | 6.792247 | B |
| 220->240 | 8 | -32.699986 | -26.605591 | 6.094395 | -40.193026 | 7.493040 | C |
| 240->260 | 0 | -0.198497 | -0.179848 | 0.018648 | -0.219142 | 0.020645 | B |
| 240->260 | 1 | -0.206817 | -0.188850 | 0.017967 | -0.226624 | 0.019807 | B |
| 240->260 | 2 | -1.114243 | -0.990304 | 0.123939 | -1.254191 | 0.139949 | B |
| 240->260 | 3 | -1.795477 | -1.587128 | 0.208349 | -2.031838 | 0.236361 | B |
| 240->260 | 7 | -3.122168 | -2.732955 | 0.389212 | -3.567928 | 0.445760 | C |
| 240->260 | 8 | -3.940773 | -3.468398 | 0.472375 | -4.479618 | 0.538845 | C |


## Transition attribution

Carrier shifts are differences of the tracked state-energy shifts; cancellation is retained. Exact reconstruction errors for every row are in `attribution_tables.csv` (maximum about 1.1e-10 MHz).


| Step | Carrier | Observed MHz | SO MHz | SO error MHz |
|---|---|---|---|---|
| 180->200 | f01 | 0.066862 | 0.060147 | -0.006715 |
| 180->200 | f07 | -1.294617 | -1.121646 | 0.172972 |
| 180->200 | f08 | -3.179220 | -2.763091 | 0.416129 |
| 180->200 | f78 | -1.884603 | -1.641446 | 0.243157 |
| 180->200 | f81 | -3.246082 | -2.823238 | 0.422844 |
| 180->200 | f72 | 0.203265 | 0.192365 | -0.010900 |
| 180->200 | f21 | -1.564745 | -1.374158 | 0.190587 |
| 180->200 | f03 | -2.263052 | -1.991820 | 0.271232 |
| 180->200 | f38 | -0.916168 | -0.771272 | 0.144896 |
| 200->220 | f01 | -0.015937 | -0.014885 | 0.001052 |
| 200->220 | f07 | -0.900411 | -0.820698 | 0.079713 |
| 200->220 | f08 | -1.714360 | -1.545040 | 0.169320 |
| 200->220 | f78 | -0.813949 | -0.724342 | 0.089607 |
| 200->220 | f81 | -1.698423 | -1.530155 | 0.168268 |
| 200->220 | f72 | 2.892956 | 2.488866 | -0.404091 |
| 200->220 | f21 | -3.777431 | -3.294679 | 0.482751 |
| 200->220 | f03 | -3.979627 | -3.476246 | 0.503381 |
| 200->220 | f38 | 2.265267 | 1.931206 | -0.334061 |
| 220->240 | f01 | -0.044064 | -0.034866 | 0.009198 |
| 220->240 | f07 | -26.642257 | -21.386192 | 5.256065 |
| 220->240 | f08 | -31.290008 | -25.390443 | 5.899564 |
| 220->240 | f78 | -4.647750 | -4.004251 | 0.643499 |
| 220->240 | f81 | -31.245944 | -25.355577 | 5.890367 |
| 220->240 | f72 | -25.124981 | -20.022596 | 5.102385 |
| 220->240 | f21 | -1.473213 | -1.328730 | 0.144483 |
| 220->240 | f03 | -2.009530 | -1.802414 | 0.207117 |
| 220->240 | f38 | -29.280477 | -23.588029 | 5.692448 |
| 240->260 | f01 | -0.008320 | -0.009002 | -0.000682 |
| 240->260 | f07 | -2.923671 | -2.553107 | 0.370564 |
| 240->260 | f08 | -3.742276 | -3.288550 | 0.453726 |
| 240->260 | f78 | -0.818605 | -0.735443 | 0.083162 |
| 240->260 | f81 | -3.733956 | -3.279548 | 0.454408 |
| 240->260 | f72 | -2.007925 | -1.742651 | 0.265273 |
| 240->260 | f21 | -0.907426 | -0.801454 | 0.105972 |
| 240->260 | f03 | -1.596980 | -1.407280 | 0.189701 |
| 240->260 | f38 | -2.145296 | -1.881270 | 0.264026 |


## Why ranks 220–239 matter

For states 7/8, the shell probabilities are only 0.064906%/0.075298%, yet the
exact energy shifts are -28.052236/-32.699986 MHz. Logical states 0/1 shift
only -1.409978/-1.454042 MHz. Their unequal corrections produce the large
outer-carrier shifts and the -4.647750 MHz change in f78.

The minimum isolated-shell denominators for states 7/8 are 32.517046 and
32.497011 GHz, vastly larger than the frozen 0.010 GHz near-resonance scale.
Maximum coupling/denominator ratios are 0.00913 and 0.01110, below 0.10.
Every measured adjacent-shell near-resonant contribution fraction is zero.
This conclusion concerns the stated partition; it is not a statement about
uncomputed ranks beyond 259.

The enhanced numerator, rather than a small denominator, explains the jump.
Total coupling norms ||H_QP psi_old|| in GHz are:


| Step | State0 | State1 | State7 | State8 |
|---|---|---|---|---|
| 180->200 | 0.184660 | 0.175162 | 0.314557 | 0.449816 |
| 200->220 | 0.103105 | 0.108412 | 0.233688 | 0.326823 |
| 220->240 | 0.293246 | 0.296760 | 1.149129 | 1.242372 |
| 240->260 | 0.119072 | 0.121649 | 0.431929 | 0.478767 |



Exact central-shell rank shares for the primary states (self energy, before adding P relaxation):


| State | Rank | Self energy MHz | Old-state coupling norm GHz | Shell weight |
|---|---|---|---|---|
| 7 | 230 | -16.192968 | 0.790544 | 3.001e-04 |
| 7 | 223 | -11.851075 | 0.664436 | 2.194e-04 |
| 7 | 225 | -2.908925 | 0.320885 | 5.333e-05 |
| 7 | 221 | -1.389642 | 0.230807 | 2.570e-05 |
| 7 | 231 | -1.061953 | 0.202303 | 2.095e-05 |
| 8 | 230 | -17.266152 | 0.813043 | 3.154e-04 |
| 8 | 223 | -14.616357 | 0.747393 | 2.807e-04 |
| 8 | 225 | -3.405819 | 0.355572 | 6.381e-05 |
| 8 | 221 | -1.840153 | 0.269559 | 3.543e-05 |
| 8 | 226 | -0.947097 | 0.206141 | 1.681e-05 |



Ranks 230/223/225 supply 88.832%/87.797% of absolute exact self energy for
states 7/8. Their contributions are not standalone addback energy shifts:
they include the self-consistent coupled-shell solution and must not be added
again to the reported exact shift. Logical states0/1 emphasize the same ranks;
comparator states2/3 emphasize ranks220/222/227 instead.

The largest dressed-shell contributions are:


| State | Shell mode | |coupling| GHz | E_old-epsilon GHz | SO contribution MHz | SO share |
|---|---|---|---|---|---|
| 7 | 22 | 0.442555 | -48.471944 | -4.040585 | 17.878% |
| 7 | 47 | 0.455104 | -64.662480 | -3.203085 | 14.172% |
| 7 | 21 | 0.383666 | -48.446244 | -3.038411 | 13.443% |
| 8 | 43 | 0.713382 | -64.294752 | -7.915322 | 29.751% |
| 8 | 1 | 0.354100 | -32.584026 | -3.848099 | 14.463% |
| 8 | 27 | 0.391241 | -48.813616 | -3.135800 | 11.786% |



The frozen mode-based decision uses top-three shares and effective participation:


| Central-shell state | Top3 modal share | Effective modes | Modes for 90% | SO error / shift | Class |
|---|---|---|---|---|---|
| 0 | 52.344% | 6.827399 | 10 | 13.818% | C |
| 1 | 55.669% | 8.148141 | 12 | 14.032% | C |
| 2 | 65.101% | 4.899825 | 15 | 11.906% | C |
| 3 | 57.783% | 5.198680 | 16 | 11.755% | C |
| 7 | 45.493% | 9.497784 | 11 | 19.431% | B |
| 8 | 56.000% | 7.053051 | 11 | 18.637% | C |



Both primary states require eleven dressed shell modes to account for 90%.
Their top-three shares are 45.493% and 56.000%. The latter misses the frozen
distributed criterion (<=50% and effective count>=8), so overall C is retained.
There is no basis for relabeling this result A or relaxing the preregistration
to force B. The mixed label is operational, not a new physical phase.

Second-order estimates meet the preregistered 20% diagnostic accuracy bound,
but miss state7/8 shifts by 5.451/6.094 MHz. They are not adequate for precision
calibration or a convergence claim. Exact finite-block treatment is essential
for the residuals of interest. P relaxation contributes +6.792/+7.493 MHz to
exact self energies -34.844/-40.193 MHz. This decomposition is not identical
to second-order plus a single correction; the exact self energy uses the
relaxed p and dressed E.

In Q2, states7/8 emphasize ranks251/258/259 and 258/259/256 respectively.
The rank259 boundary remains active. That warns that a larger-parent tail is
unknown; it does not authorize extrapolating a smooth decay beyond k260.

## Q1–Q2 coupling and nonadditivity

The P+Q2 counterfactual skips Q1. The no-cross counterfactual includes both
shells but sets only H_Q1Q2 and its adjoint to zero. All remaining matrix
entries are unchanged. These diagnostic counterfactuals have safe tracking.

    Full correction = standalone Q1 + standalone Q2
                      + explicit Q1-Q2 effect + P-mediated nonadditivity.

Here explicit effect = E_full-E_no_cross; the last term is
E_no_cross-E_PQ1-E_PQ2+E_P. It includes mixing through P and energy dependence.
Only this defined decomposition is used; separate shells are not assumed
additive. Values below are MHz.


| State | Q1 alone | Q2 alone | Explicit cross | P-mediated remainder | Full 220->260 |
|---|---|---|---|---|---|
| 0 | -1.409978 | -0.134161 | -0.049773 | -0.014563 | -1.608475 |
| 1 | -1.454042 | -0.145391 | -0.047703 | -0.013722 | -1.660859 |
| 2 | -2.927255 | -0.841619 | -0.205684 | -0.066940 | -4.041498 |
| 3 | -3.419509 | -1.369638 | -0.326525 | -0.099314 | -5.214986 |
| 7 | -28.052236 | -1.865323 | -0.931852 | -0.324992 | -31.174403 |
| 8 | -32.699986 | -2.798407 | -0.825876 | -0.316489 | -36.640758 |



For states7/8 explicit cross coupling contributes -0.931852/-0.825876 MHz;
the remainder is -0.324992/-0.316489 MHz. Cross-shell effects matter at
sub-MHz/MHz precision but cannot explain the already-large P->P+Q1 jump.
For f07/f08/f78, total nonadditivity is -1.192508/-1.078029/+0.114479 MHz.


## Drive selectivity

No gate optimization or driven evolution was performed. Frequencies are MHz; grid_phi magnitudes are in the existing operator convention.


| k | f01 | f78 | f78-f01 | g01 | g78 | f07-f81 |
|---|---|---|---|---|---|---|
| 180 | 15.857270 | 22.733437 | 6.876168 | 0.453425 | 1.142340 | -6.876168 |
| 200 | 15.924131 | 20.848834 | 4.924703 | 0.452727 | 1.140375 | -4.924703 |
| 220 | 15.908195 | 20.034886 | 4.126691 | 0.452495 | 1.140265 | -4.126691 |
| 240 | 15.864131 | 15.387135 | -0.476995 | 0.448356 | 1.156867 | 0.476995 |
| 260 | 15.855811 | 14.568530 | -1.287281 | 0.447948 | 1.155801 | 1.287281 |



The intended middle transition crosses the logical splitting between k220 and
k240. Its separation is only 0.477 MHz at k240 and 1.287 MHz at k260, compared
with 4.127 MHz at k220. g01 remains appreciable (~0.448 at k260); the proximity
is not removed by a vanishing logical matrix element. g01/g78 is about 0.388.

The outer-carrier identity f07-f81 = f01-f78 means the same crossing also
brings the two strong outer transitions into near collision. At k260 they are
separated by 1.287281 MHz with g07=1.909889 and g81=1.777150. Those transitions
are desired on their respective legs but unwanted targets of each other's tone.

The nearest additional logical-origin transitions are 0->8 and 1->7:
their offsets from outer tones are about 14.57–15.86 MHz at k260, but their
grid_phi magnitudes are only about 1.13e-11 and 9.47e-12 in this model. The
appreciable 1->6 coupling (~0.4871) is about 253–254 MHz from the outer tones.
The nearest-five scan for both tones at all five k values is stored in
summary.json. Frequency proximity alone does not establish gate fidelity,
leakage, or a physical-port response.

## Operators and subspaces

The following reuse the validated same-parent operator and tracking results;
the newly reconstructed values agree to the tolerance above. F denotes a
Frobenius norm. Duplicated singular values are listed as pairs where equal to
the shown precision, not discarded from the numerical results.


| Quantity | k180 | k200 | k220 | k240 | k260 |
|---|---|---|---|---|---|
| Full {0,1,7,8} F | 4.086598 | 4.082916 | 4.081871 | 4.089519 | 4.084720 |
| Logical->{7,8} F | 2.615264 | 2.613366 | 2.612638 | 2.612033 | 2.608819 |
| Internal {7,8} F | 1.615513 | 1.612733 | 1.612578 | 1.636057 | 1.634549 |
| Full singular values | 2.362544 / 2.362544 / 1.663889 / 1.663889 | 2.360299 / 2.360299 / 1.662555 / 1.662555 | 2.359765 / 2.359765 / 1.662030 / 1.662030 | 2.372922 / 2.372922 / 1.652672 / 1.652672 | 2.370460 / 2.370460 / 1.650269 / 1.650269 |
| Logical singular values | 1.906482 / 1.790232 | 1.906114 / 1.787852 | 1.905232 / 1.787728 | 1.910605 / 1.781097 | 1.909889 / 1.777150 |
| Internal singular values | 1.142340 / 1.142340 | 1.140375 / 1.140375 | 1.140265 / 1.140265 | 1.156867 / 1.156867 | 1.155801 / 1.155801 |
| State7 overlap | 0.999135751 | 0.999224320 | 0.999243232 | 0.999960157 | 1.000000000 |
| State8 overlap | 0.998803514 | 0.998893774 | 0.998854305 | 0.999941855 | 1.000000000 |
| Subspace minimum SV | 0.998794263 | 0.998885848 | 0.998847125 | 0.999941298 | 1.000000000 |
| Maximum principal angle (deg) | 2.813893 | 2.704894 | 2.751506 | 0.620823 | 0.000000 |



No required assignment is ambiguous. The full subspace k220->k260 overlap is
strict BORDERLINE (0.998847125), and k240->k260 is strict PASS (0.999941298).
State8 follows the same pattern; state7 passes both. This is measurable
dressing without loss of the tracked identity.

All operator-block magnitudes pass frozen strict relative criteria for both
endpoint comparisons, but the operator redistribution is not zero. In the
central step |g07| grows by 0.005374, |g78| by 0.016602, and |g81| falls by
0.006631. The exact complex operator split after phase alignment is

    delta O = (p_new^dag O_PP p_new - O_old)
              + p_new^dag O_PQ q + q^dag O_QP p_new + q^dag O_QQ q.

The first term is PP relaxation; the remaining three are direct shell terms.
Their sum matches the full complex change within 1e-10. The direct-shell
magnitudes for g07/g78/g81 are 0.002037/0.011291/0.002225. These correspond to
approximately 0.107%/0.990%/0.124% of their k220 edge magnitudes. In particular,
the direct g78 contribution is substantial within its small net change;
for g81 the direct contribution opposes part of the PP relaxation. Complex
contributions are recorded separately, rather than treating absolute values
as additive. High overlap therefore does not imply identical operators.

## Pathway reassessment

The primary remains global rank1 at every k; 0721 remains rank3. 0381 changes
rank4 to rank5 at k240, giving the existing ordering BORDERLINE. Target-path
ordering among these three never changes.


| Path score | k180 | k200 | k220 | k240 | k260 |
|---|---|---|---|---|---|
| 0-7-8-1 | 3.898859 | 3.886224 | 3.883784 | 3.936788 | 3.922971 |
| 0-7-2-1 | 0.327142 | 0.324498 | 0.324697 | 0.317565 | 0.317275 |
| 0-3-8-1 | 0.254077 | 0.251611 | 0.252399 | 0.243329 | 0.242901 |



All three score changes pass the frozen strict endpoint thresholds. Primary
edges change by 0.244%/1.344%/0.595% from k220 to k260; its score changes by
0.999%. The mechanism remains structurally meaningful and deserves further
study. Its carrier frequencies remain unsuitable as converged precision
calibration data. Both comparators remain useful stress tests: their shell
response emphasizes different Stage-1 ranks. No evidence here warrants
replacing the primary mechanism with a comparator merely on score stability.

## Next research action
Preregister a same-parent rank-addback/ablation experiment: starting from k220, add ranks 223 and 230 separately and together, then add 225, always with all eight mode-4 states; compare against k240 and a held-out complement of ranks 220–239. Use ranks 258/259 as an independent Q2 stress test, and measure all mandatory carrier, state/subspace, and operator errors. Use the saved k260 parent only.

This recommendation is not executed in this task. It directly tests whether
the rank shares are causally captured by a small retained subset rather than
being inseparable collective dressing. The held-out complementary shell is
necessary to expose selection bias and omitted interactions.

| Option | Information gained | Cost / limitation |
|---|---|---|
| Larger k280/k300 parent | New data beyond the active rank259 boundary; eventually needed to test the tail | New Stage-1 solve; alone it does not test the central rank-attribution mechanism |
| Physics-informed retained selection | Tests whether energy-prefix ordering wastes basis states | Requires validation against held-out shells; must not cherry-pick only the primary path |
| Targeted influential-rank addback/ablation | Directly tests ranks223/230/225 with existing data and comparators | Lowest-cost discriminating experiment; cannot certify the beyond-k260 tail |
| Further effective-Hamiltonian diagnostics | Can partition relaxation within P and identify feedback channels | Useful follow-up if addback fails; the present exact identity already resolves the numerical shell shift |

The recommended addback has the greatest immediate information per numerical
cost. It is a diagnostic experiment, not a replacement convergence criterion.

## Scope and preservation

k260 remains an endpoint, not ground truth. This is fixed-N71, fixed-four-mode
retained Stage-1 attribution. It establishes no spatial/domain/n1max/N4
convergence, omitted-array-mode completeness, physical-device/port validity,
or X90/X180 gate validation.

All tracked-file contents and all files in the k260, k220, k180, and N51
checkpoints were hashed before and after the calculation and remained identical.
No Stage-1 solve, production edit, EmilianoUPDATE.md edit, commit, or push was
performed. Existing tracked status:

```text
MM EmilianoUPDATE.md
M  Simulations/Cphase/main.ipynb
M  Simulations/Cphase/workflow_funcs.py
```

GATE WORK: **HOLD**

SPATIAL CONVERGENCE: **HOLD**
