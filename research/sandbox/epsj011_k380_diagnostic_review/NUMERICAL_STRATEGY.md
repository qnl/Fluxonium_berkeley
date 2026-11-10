# Numerical strategy assessment

All options are research proposals. No new method is declared validated, no additional solve is authorized, and the official FAIL / D_NONMONOTONIC_UNRESOLVED / secondary PASS results remain unchanged. Ranking reflects the uncertainty addressed and available falsification tests, not computational price alone.

## 1. C — Validate a Schur/resolvent correction strategy

**Uncertainty:** Whether a retained model can reproduce target-dependent energy shifts and state changes without retaining all high ranks explicitly, and which approximation causes the measured perturbative bias.

**Evidence:** Exact finite-parent Schur identities pass, whereas saved fixed-vector second-order shifts miss state7/8 corrections by 0.576/0.709 MHz in S1 and 0.927/0.982 MHz in S2. Positive P-space relaxation cancels a material part of the negative self-energy. Saved S2 nonadditivity reaches 0.221 MHz for a carrier. Small fixed-p energy derivatives do not remove these vector-relaxation and cross-shell effects.

**Required validation:** Preregister the energy argument, P/Q partition, treatment of off-diagonal elements and cross-rank coupling, root/state tracking, and error metrics. Compare against exact already authenticated finite-parent endpoints, including raw/centered/excitation energies, f78 cancellation, operators, and held-out shells. Distinguish scalar perturbation, exact full-shell resolvent, and a self-consistent matrix model. Freeze all choices before any validation used to tune a method. A benchmark inside k380 does not validate an infinite tail.

**Burden:** The saved scalar audit needs only sums. A new Q-block resolvent contraction would use 160-dimensional shell blocks and matrix products, with cost roughly O(Q³) for factorization plus contractions; it is not a new Stage-1 solve. A self-consistent effective P-space eigenproblem can still be expensive (P dimensions 2720/2880), depending on the formulation. No implementation or timing is promised here.

**Principal failure:** Neglecting cross-rank mixing, changing P vectors, or cancellation can yield accurate-looking splittings but wrong states. Treating the isolated-rank selector response as the exact full-shell self-energy is invalid.

**Existing-data test without a new expensive solve:** Yes for scalar identities, fixed-vector bias, relaxation, local fixed-p derivatives, and already saved nonadditivity. Full matrix validation is **INCOMPLETE** because dressed eigenvectors/contractions were not saved. Obtaining them through any new large eigensolve requires separate authorization.

## 2. B — Investigate response-aware retention using the frozen selector

**Uncertainty:** Whether a target-aware finite retained space captures virtual influence more efficiently than a contiguous energy prefix at matched computational cost.

**Evidence:** The unchanged selector gives secondary PASS in S1, S2, and union contexts. Its saved pair responses emphasize states7/8 for ranks354/362, and the physical rank attribution is strongly concentrated. State selection and its seal precede validation. This is affirmative finite-parent evidence, independent of whether contiguous-prefix convergence passes.

**Required validation:** Reuse the exact source hash, target pairs, whitening, regularization, weights, budgets, tie-breaking and controls. Preserve individual state recovery and cancellation checks. Evaluate held-out shells/parents before extrapolating. Any altered feature/weight/budget would be a new preregistered algorithm; it cannot retrospectively improve these verdicts. Independent parameter validation remains BORDERLINE, multi-target transfer PARTIAL.

**Burden:** Reviewing saved budgets and controls is cheap and needs no eigensolve. Computing candidate Stage-1 states still requires an authenticated parent. New validation spaces may each require a dense Stage-2 solve and are not authorized by this review.

**Principal failure:** A finite parent can make a selector look successful while unseen omitted states remain important. A fixed target list can miss other states/operators; near-degenerate rank rotations can undermine rank-based features. Retained dimension is not the only cost: response construction and available parent information count.

**Existing-data test without a new expensive solve:** Yes, all frozen selector/control metrics and selection response matrices are saved. They do not test influence above rank379 or validate changed algorithms.

## 3. D — Develop a convergence-tail diagnostic

**Uncertainty:** Whether omitted-state sensitivity can be bounded or prospectively flagged, rather than inferred from the last observed increment.

**Evidence:** Maximum carrier corrections increase 0.715→1.300→5.595→6.738 MHz. Successive shells reveal different concentrated contributors. Last-step f78 stability is partly cancellation. Neither a shrinking scalar nor a small Q probability supplies a bound on future shells.

**Required validation:** Define a diagnostic and predictions before inspecting held-out shells. Test false reassurance on historical failures, use all target states/observables, and distinguish a heuristic warning from a rigorous error bound. A rigorous resolvent bound needs a justified omitted-space spectral gap and coupling norm, including coupling from the approximate P-space state to the truly omitted space.

**Burden:** Back-testing scalar warning measures on the four saved increments is cheap. Rigorous beyond380 bounds may require new operator estimates or spectral information and could approach the burden of a new solve. No geometric/power-law fit is performed.

**Principal failure:** Sparse influential ranks and cancellation defeat smooth-tail extrapolation. Four observed increments cannot certify an asymptotic regime or a uniform omitted-space gap.

**Existing-data test without a new expensive solve:** Partial. Historical false-negative tests and in-parent sensitivity are available. Beyond380 error bounds are **INCOMPLETE**; no saved diagnostics contain that missing space.

## 4. E — Investigate alternative basis construction

**Uncertainty:** Whether the staged basis itself causes target-relevant response to appear inefficiently at high rank, and whether a representation can reduce cost without losing scientific observables.

**Evidence:** A single Stage-1 rank represents eight rank-major/mode4-minor product states; high-rank virtual response is strongly target-dependent. Rank concentration is substantially stronger than dressed-mode concentration. Energy ordering alone does not order virtual importance.

**Required validation:** State the proposed basis and variational/effective-model semantics. Preserve the Hamiltonian, units and physical grid. Compare state/projector and operator recovery as well as energies at matched memory/time. Verify Hermiticity, symmetry and state assignment; handle near-degenerate clusters by subspaces rather than arbitrary rank labels. Separate a representation change from a model change and from spatial convergence.

**Burden:** Reviewing saved operator rows and response structure is cheap. Constructing and validating a new contracted, adaptive or block basis can require substantial implementation and fresh solves. Costs are method-dependent and unmeasured.

**Principal failure:** Basis optimization may overfit selected states or silently omit channels, break variational interpretation, or hide spatial/model truncation error.

**Existing-data test without a new expensive solve:** Partial. Existing matrices and saved state-level diagnostics can motivate and constrain a proposal. A new basis is not validated without new comparisons, some of which would require separate authorization.

## 5. A — Increase k with the existing method

**Uncertainty:** What the next contiguous retained shell actually contributes and whether this finite sequence eventually develops a plateau.

**Evidence:** The authenticated sequence is reliable but still failing; no result predicts the next influential rank. An additional parent would directly sample an unseen interval. It would not by itself resolve asymptotic convergence.

**Required validation:** Separate authorization; a prospectively sealed protocol, hypothesis, finite endpoint, prefix/shell comparisons, selector chronology, stopping conditions, resource/lock checks and unchanged or newly preregistered criteria. The design must distinguish competing explanations rather than assume another parent will converge. No automatic continuation after a failure.

**Burden:** Another Stage-1 eigenproblem and its Stage-2 analysis are required. At fixed N71 the sparse matrix dimension/LU structure may remain the same, but Arnoldi and eigenvector costs grow with k. k380 measured 275.227 s for its guarded runner; frozen post-analysis took several minutes. These measurements are context, not a k400 runtime or memory promise.

**Principal failure:** A costly extra shell can reproduce the same irregularity, or a fortuitously small shell can be misread as convergence. Finite-parent success would still not prove an infinite tail, spatial convergence, model completeness or gate fidelity.

**Existing-data test without a new expensive solve:** No for k400 or larger results. Existing data can assess the experimental design and motivation only.

## Recommended next experiment — NOT AUTHORIZED

Preregister a matched-cost finite-parent benchmark of a Schur-corrected retained model against the unchanged selector and controls, with held-out shells and no refitting. First identify the missing dressed-vector contractions, how they would be obtained, and whether that requires separately authorized expensive Stage-2 work. This investigates the observed perturbative/relaxation failure directly; a larger parent remains a separate experimental choice.

GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
