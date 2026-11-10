# N71 k220 retained-basis preregistration

Date frozen: 2026-09-29, before inspection of an authenticated k220 result.

This preregistration concerns three nested Stage-1 retained spaces, k=180,
k=200, and k=220, derived from one authenticated N=71 k220 parent.  The k220
endpoint is the comparison endpoint, not ground truth.

The machine-readable source of truth is
`research/sandbox/k220_preregistered_criteria.json`.  The postprocessor records
that file's SHA-256 in every result so later threshold edits are visible.

## Paths

1. `0 -> 7 -> 8 -> 1`: primary coupling baseline.
2. `0 -> 7 -> 2 -> 1`: primary model-risk comparator.
3. `0 -> 3 -> 8 -> 1`: secondary robustness comparator.

## Classification protocol

Each k180 and k200 observable is compared independently with the same k220
endpoint.  The reported three-point status is the worse of the two endpoint
comparisons.  Thus a stable k200 result does not hide an inadequate k180
space.

| Observable | Strict PASS / BORDERLINE / FAIL | Practical PASS / BORDERLINE / FAIL | Basis |
| --- | --- | --- | --- |
| f07, f08, and ordinary carrier frequencies | <=1 / 1–5 / >5 MHz | <=5 / 5–10 / >10 MHz | The repository's production transition criterion is 5 MHz.  The strict 1 MHz band asks for a clear plateau below that limit and is small relative to the known 6.876 MHz close spacing. |
| Fine 7–8 carrier | <=0.5 / 0.5–1 / >1 MHz | <=1 / 1–2 / >2 MHz | The repository uses 0.5 MHz for protected-doublet splittings; the 7–8 carrier is itself a small, control-sensitive splitting. |
| Assigned state overlap | >=0.999 / 0.995–0.999 / <0.995 | >=0.995 / 0.990–0.995 / <0.990 | N51-to-N71 overlaps for states 7/8 are above 0.999, while the unresolved k175 evidence reached only about 0.993 for state 8.  The runner's 0.90 gate is only a warning against making state-specific claims, not a convergence tolerance. |
| Each nonzero target `|grid_phi|` edge | <=3% / 3–5% / >5% | <=5% / 5–10% / >10% | The existing drive-response convergence test uses 3% relative tolerance.  Its 2 MHz absolute floor has incompatible units and is not reused for dimensionless `grid_phi`. |
| Path-product score | <=5% / 5–10% / >10% | <=10% / 10–20% / >20% | Three independent 3% edge changes can compound to about 9%; 10% therefore bounds ordinary compounded edge tolerance.  The 5% strict band distinguishes a genuinely stable conclusion. |

Boundary values belong to the less severe class.  Relative changes use the
k220 magnitude as denominator.  None of the preregistered path edges is
near-zero; if a future reference edge is zero, the postprocessor refuses a
relative classification instead of manufacturing one.

Path ordering is categorical:

- PASS: the primary remains global rank 1, all three full ranks stay fixed,
  and their mutual order stays fixed.
- BORDERLINE: the primary stays global rank 1 and remains the strongest target,
  but a comparator changes full rank or the comparators exchange order.
- FAIL: the primary loses global rank 1, another target becomes the strongest
  target, or a required target cannot be ranked.

An optional competing-overlap ratio >=0.50, overlap margin <=0.25, or assigned
overlap <0.90 invokes the runner's ambiguity gate.  State-specific conclusions
must then be deferred in favor of principal-angle/subspace diagnostics.  The
current retained-case schema does not provide the first two diagnostics.

Parent quality is a hard prerequisite: maximum eigenpair residual <=1e-7,
orthogonality infinity residual <=1e-8, exact frozen fingerprint
`69ee54299c70336f00b06c18250747c2e600b2f48fb9b9365f1651ce57997c39`,
verified file hashes, and a matching COMPLETE marker.

## Decision tree

1. **k180 approximately equals k200 approximately equals k220 for everything.**
   Accept retained-basis stability at fixed N=71/four-mode physics.  Carry all
   three paths forward to the separate model-risk study, with `0-7-8-1` still
   the coupling baseline.  Do not call the spatial model or a gate validated.

2. **Energies converge but `grid_phi` elements do not.**
   Treat control/pathway conclusions as unconverged.  Increase the retained
   parent and nested spaces or inspect omitted Stage-1 weight/operator
   projection.  Do not choose a path from energy convergence alone.

3. **Assigned overlaps are high but pathway scores move materially.**
   Inspect individual edges, phases if available, and competing/subspace
   overlaps.  High diagonal overlap can coexist with control-relevant operator
   rotation.  Classify the affected pathway BORDERLINE or FAIL according to
   its edge/score bands.

4. **`0-7-8-1` destabilizes while a comparator stabilizes.**
   Retain the stable comparator alongside the primary physics question.  Do
   not discard the primary until a larger parent determines whether its change
   is truncation or state-tracking ambiguity.  Prioritize the stable comparator
   for later array-mode/model-risk tests.

5. **All three pathways move materially.**
   Conclude that k220 has exposed unresolved retained-basis dependence.  Build
   a larger same-parent study before pulse work; no pathway recommendation is
   numerically mature.

6. **Pathway ordering changes.**
   If only the two comparators exchange, mark ordering BORDERLINE and retain
   both.  If the primary loses rank 1 or ceases to lead the target set, mark
   FAIL and defer pathway selection until a larger-parent comparison.

7. **Ambiguous or near-degenerate tracking appears.**
   Stop individual-label interpretation.  Track the competition-connected
   subspace with singular values/principal angles, report basis-invariant
   `grid_phi` block norms, and only restore individual paths after labels are
   uniquely resolved.  Request a schema extension if these diagnostics are not
   present in the returned analysis.

## Scientific guardrails

- Agreement with k220 establishes only retained-basis stability at fixed N=71
  and in the fixed four-mode model.
- It does not establish spatial convergence.
- It does not establish robustness to omitted distributed array modes or other
  parasitics.
- It does not validate a gate.

## Postprocessor contract

`analyze_k220_return.py` accepts either a completed checkpoint directory or a
lone `analysis.json`.  Directory input verifies COMPLETE, the configuration
fingerprint, checkpoint and artifact hashes, residuals, and orthogonality.
JSON-only input is analyzed but explicitly remains unauthenticated and lacks
parent-solver quality.

Required scientific fields are `profile`, `configuration_sha256`,
`n71_reference_k220`, and retained k180/k200/k220 cases containing
`retained_indices`, `energies_GHz`, `state_overlaps`, and `grid_phi_abs`.
Missing fields cause a clear nonzero failure.

Current schema ambiguities:

- `thomas_observables` stores f07/f08 and primary-path edges only.  The
  postprocessor therefore derives every requested carrier and edge from the
  complete energies and `grid_phi_abs` matrix.
- `state_overlaps` is positional and is understood to be the runner's
  assignment against k220; the field name does not encode that endpoint.
- Same-parent retained cases do not store competing overlaps, overlap margins,
  or principal-angle/subspace diagnostics.  Assigned overlaps cannot alone
  settle near-degenerate tracking.
- Authentication and solver-quality fields live outside `analysis.json`, so a
  lone JSON cannot establish them.
