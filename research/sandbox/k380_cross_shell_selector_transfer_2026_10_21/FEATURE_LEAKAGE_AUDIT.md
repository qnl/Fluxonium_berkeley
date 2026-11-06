# Feature leakage audit

**FEATURE LEAKAGE: LIMITATIONS.** No evaluation-label input was found in the
historically sealed physics order. The current comparison is retrospective,
the simple controls were defined after historical outcomes existed, and S2 lacks
base-compatible labels. These limitations prevent independent transfer claims;
they are not evidence that the historical selector used its answer.

Categories requested by the work block: A=available before candidate evaluation;
B=retained-base information; C=allowed candidate feature information;
D=dressed information requiring calculation unless already saved compatibly;
E=same Schur contractions as the evaluation labels (prohibited as a predictor).
Categories can overlap because A describes timing and B/C describe origin.

| Input or quantity | Category | Provenance and treatment |
| --- | --- | --- |
| Target pairs (0,1),(2,3),(7,8), budgets3/6/10/15, logdet scale6, floors/ties, zero operator weight | A | Fixed historical method/source and k380 criteria, not tuned here |
| Base-only eleven dressed vectors/energies, tracked target pairs, pair-mean mu | A+B; D for recomputation now | The historical selection hook solves only P340 before feature freezing. Vectors are not saved for a new feature calculation, but the derived matrices and final order are saved with matching provenance |
| Original-k340 reference used to track the nested P340 base | A+B | Prior base identity anchor, not candidate-shell dressed targets; the matching process precedes freezing |
| k380 Stage-1 eigenvalues and five projected operators, including candidate ranks | C | Allowed to be known after Stage-1 and before candidate Stage-2 evaluation; shared model data do not equal outcome labels |
| Coherent H_QrP U_g and H_QrQr; isolated8×8 block denominators | A+B+C | Historically computed at the base target directions; no candidate-space dressed eigenvectors or exact Schur labels enter |
| Raw complex2×2 A_rg, full-shell feature sum and whitening | A+B+C | Candidate-shell adaptation uses its allowed features, not attributions. Use the historical matrix-greedy order, never replace it with trace sorting |
| Sealed complete greedy order and selected budget lists | A | Direct current ranking source; no new contraction, whitening or greedy reranking |
| T300/Tactual bare strengths and Vplus/Dplus controls | C; fixed retrospectively | Already computed from saved Stage-1 operators/energies; original formulas fixed before this budget comparison, but after historical labels existed. Not historical pre-result predictions |
| Prefix/spread/reverse-spread identities | A | Historical geometry/energy controls; no target outcome input. S2 P360 baseline support is explicitly rebased |
| WP2 BaseTrace or isolated-rank scalar trace sorting | B+C origin, excluded here | Not inherently derived from exact labels under the verified provenance, but it is not the historical selector. User prohibits treating this alternative scalar ranking as predictive transfer |
| WP2 BaseC scalar ranking | B+C origin, excluded here | Saved base-target norm association is not the whitened greedy selector; no substitution for the registered rule |
| P360-based S2 selector targets/features/order | D: unavailable | No saved P360 selector order. Generating one requires prohibited new dressed work |
| Exact rank sigma, full-shell target coupling norms, probabilities, ExactC, ExactRef | E | Evaluation or outcome-adjacent quantities; never supplied to the ranking producer |
| Full-shell channel/cross-rank complex Schur contractions | D+E: unavailable/prohibited | Not saved, not reconstructed, not used |

## Evidence for historical separation

The selector source/method were sealed on2026-10-08. The k380 preregistration
sealed on2026-10-18 pins exactly those same source/method hashes and mandates
selection before candidate/full-new-shell validation. The selection hook writes
`selector.freeze(...)` and its SHA/seal record before calling `analyze_parent`;
the latter starts the new full-parent Stage-2 reference calculation. Later
`analyze_k380.py` creates exact S1/S2 Schur labels. Checkpoint analysis links the
selection and seal, marks selection-before-validation true, and records false
for use of candidate/full-shell dressed results and old selected-rank lists.
Metrics link the identical selection SHA. Source authentication passes.

This is affirmative local code-and-artifact evidence for pre-label selection,
not merely a claim that an algorithm name was old. It is not a signed or
externally timestamped execution trace. Historical prior outcomes informed the
original method; these k380 labels did not enter its stated feature path.
The October21 scalar alternatives are explicitly later exploratory quantities.

## Current executable separation

`bare_input` retains only shell,rank,T300,Tactual,Vplus,Dplus. Neither W nor
target-aware fields reach `make_orders`. Stored physics/control identities are
validated against sealed budget support. `RANKINGS_FROZEN.json` is written and
hashed before parsing `k380_metrics.json` for Schur labels; evaluation cannot
change it. A synthetic test changes W,ExactC and BaseTrace to adversarial values
without changing any whitelisted input or ranking. Outcome fields appear only
in the subsequent evaluation stage. No isolated-rank scalar trace order is made.

No E input is accepted as an independent predictor. Source separation alone
does not make reused S1 data a new blind validation set, and it cannot correct
the P340/P360 mismatch for S2. See SOURCE_COMPATIBILITY.md.
