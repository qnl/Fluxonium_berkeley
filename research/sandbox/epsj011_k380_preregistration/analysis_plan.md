# Frozen analysis plan

## Authentication first

Verify the preregistration manifest/seal, pinned historical sources and payloads,
new COMPLETE-to-checkpoint link, exact fingerprint and nested sizes, payload
hashes, selection seal and checkpoint linkage, and parent quality before analysis.
All outputs use exclusive creation. Never overwrite a failed analysis or checkpoint.
No Stage-1 solve is called by the analyzer.

## Observables and exact decision rules

Report f01,f07,f08,f78,f81,f72,f21,f03,f38; f78-f01; six target raw, centered and
excitation energies (state-0 excitation is identically zero); target overlaps,
assignment confidence, all five subspace singular spectra/principal angles;
all eight grid_phi edges; six operator blocks with Frobenius norms and singular
values; pathway scores/ranks for 0781,0721,0381; ordering, reversals and cancellation.
Hamiltonian energies are GHz (H/h, no 2pi); frequency/error displays convert to
MHz by multiplying by 1000. Comparison component errors use lower-k minus upper-k;
explicit f78-f01 increments use upper-k minus lower-k. Label both conventions.

Inherited component thresholds, inclusive at boundaries:

| Component | Strict PASS / BORDERLINE boundary | Practical PASS / BORDERLINE boundary |
| --- | --- | --- |
| Standard absolute energy/carrier error, MHz | <=1 / <=5 | <=5 / <=10 |
| f78 absolute error, MHz | <=0.5 / <=1 | <=1 / <=2 |
| State/subspace overlap | >=0.999 / >=0.995 | >=0.995 / >=0.990 |
| Operator relative error | <=0.03 / <=0.05 | <=0.05 / <=0.10 |
| Path score relative error | <=0.05 / <=0.10 | <=0.10 / <=0.20 |

Outside the second boundary is FAIL. Relative-error zero floor is exactly 1e-12,
copied from the authenticated predecessor; a nonzero change against a numerical
zero reference is not silently regularized. Nonfinite outputs stop analysis.

Overall FAIL has precedence: any hard gate, any primary practical FAIL, any
final-shell strict FAIL, any mandatory final error exceeding first error +1e-12,
meaningful reversal/two-shell cancellation, flagged same-comparison cancellation,
or primary-pathway ordering FAIL. No FAIL trigger but a strict miss or comparator
rank change is BORDERLINE. Otherwise PASS. Every first-shell and cumulative
comparison remains in the decision; a small last increment cannot erase them.
All 66 inherited component identities must be present; missing data stop evaluation.
f78-f01 and principal angles are mandatory diagnostics with no invented new threshold.

The inherited subspace classification uses singular values of the absolute
overlap block. Also report genuine principal angles from the complex overlap
block explicitly, without replacing the inherited criterion or conflating the two.
Tracking gates remain assigned>=0.90, competitor ratio<0.50, margin>0.25.
All-component nonincrease is the actual predecessor evaluator rule, stronger than
the shorthand referring only to historically problematic carriers. Same-shell
cancellation/path ordering are now explicitly enforced as required by the stated
scientific rules. These clarifications are frozen before results; no old artifact changes.

## Exact Schur/Feshbach diagnostics

S1 uses P=0–339, Q=340–359, full k360; S2 uses P=0–359, Q=360–379, full k380.
For each target solve q=(E I-H_QQ)^(-1)H_QP p by linear solve. Report observed
signed raw energy shift, exact self-energy, P relaxation, their identity residual,
Q probability, q-vector error, rank contributions/couplings/occupation and dressed
shell-mode contributions/couplings/signed denominators/eta. Report fixed-vector
second order and its discrepancy with the exact observed shift. Absolute weight
is a concentration diagnostic, not a signed energy shift.

Near resonance: |denominator|<=0.010 GHz OR eta>=0.10. Concentration: top-three
fractions, effective rank count 1/sum(normalized_weight^2), counts to 50/80/90%.
Sort aggregate rank weights descending with stable lower-rank tie order.
Report exact-mode near-resonant weight across all six targets.
Quality thresholds: Hermiticity <=1e-12, Stage-2/Schur/shift identity <=1e-9 GHz,
q-vector error<=1e-8; Stage-1 residual<=1e-7, orthogonality infinity norm<=1e-8.

Nonadditivity uses fixed first-ten/last-ten rank halves of each shell, independent
of outcomes. Compare exact P+Q1, P+Q2, P+Q1+Q2, and full with cross-half coupling
zeroed. Separate total nonadditivity, explicit cross-half contribution and
P-mediated remainder. These are descriptive counterfactuals and never feed selection.

## Secondary validation and interpretation

Use inherited k340 secondary evaluator families and b10/b15 logic, budgets
3/6/10/15, physics/prefix/spread/reverse-spread, separately for S1, isolated S2
on P340 and union. The inherited union prefix control is shell-balanced; label it
as such, not a global ordinary prefix. The frozen selector also records global
prefix identities, but they do not replace the inherited secondary comparison.
Its outcome cannot change convergence or the earlier independent BORDERLINE,
multi-target PARTIAL, and cross-parameter mechanism SUPPORTED classifications.

Report frozen verdicts before interpretation; no threshold tuning, new figures,
gate simulation, automatic next expansion, or changes to authenticated artifacts.
