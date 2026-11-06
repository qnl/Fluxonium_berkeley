# Blockers and minimal next-step tree

This describes future branches; none is executed by this audit. “Pending” reflects the saved evidence, not a newly checked remote-job status. No message to Thomas or external service was sent.

| Blocker | What is missing | What resolves it | What does not |
|---|---|---|---|
| Thomas/Lawrencium or equivalent compute access | Authenticated N71 k220 parent and declared nested reductions | Run/return the frozen job on suitable resources; authenticate COMPLETE, fingerprints, hashes, residuals and masks | Recomputing old k180 reductions, mock PASS, silently starting a local large solve |
| Combined physical-model interpretation | Reviewed correspondence between the netlist, source ambiguities and intended device | Thomas/author clarification or reviewed full circuit derivation | More accurate diagonalization of the same assumed model |
| Device array data | Branch, geometry/linear matrices, normalized mode participation, terminations, possible already-fitted capacitance contributions | Device EM extraction, measured sidebands/avoided gaps plus a consistent reduction | Frequency alone, a borrowed device's coupling, an arbitrary fifth oscillator |
| Physical control port | Operator mixture, line transfer at roughly 16–23 MHz and GHz, amplitude/phase calibration, limits | Device/line model or calibrated measurements | Abstract grid_phi amplitude or algebraic controllability |
| Spatial and other cutoffs | Independent charge/grid/domain/N4 and driven-window accuracy | Targeted convergence ladder after isolating retained-space effects | Clean state assignments, small tails, or a k-only plateau |
| Decoherence and gate objective | Intermediate-state lifetimes/noise operators, accepted fixed target/frame and error budget | Explicit open-system and hardware specification | Inferring dissipative fidelity from final coherent leakage |

## Common entry step

Keep the frozen k220 criteria unchanged. Prepare an optional signed-operator/vector sidecar contract so new ambiguity can be resolved and loop phases checked. Record any added diagnostic thresholds **before** interpreting new outputs; they supplement rather than retroactively alter the preregistration. Distinguish the new parent's nested k180 from the old standalone k180 checkpoint and compare their matched states if both are available. See ledger C28–C37 and the [frozen protocol](../k220_preregistered_analysis.md).

## A — k220 agrees with k180 and k200

Authenticate first; classify each nested case independently against k220 under both frozen strict and practical criteria. Require matching labels or adequate subspaces, target carriers, each edge, products and full ranks. Inspect signed internal/outgoing blocks when supplied, particularly 7/8-to-logical/guard states and 45/46-to-low11. Magnitude-only PASS cannot answer an absent signed-block question.

**Next calculation:** a targeted remaining spatial/N4/dynamical-window check for the observables and prescribed pulse actually to be carried forward; then a small fixed-pulse evaluation under a declared target and hardware model, if those inputs exist. **Question:** does retained-space agreement survive the independent numerical cutoffs and the specified physical drive? Agreement does not establish an infinite-k limit or physical completeness. Do not immediately begin broad optimization.

## B — k220 moves materially

Separate identity failure, carrier drift, operator drift and integration issues. If individuals become ambiguous, compare cluster projectors plus signed operator blocks and transform the cluster Hamiltonian consistently. If k200 agrees with k220 but k180 does not, reject the smaller retained case for that observable; do not automatically require a still-larger solve. If both smaller cases remain sensitive, use the authenticated residual/omitted-state structure to choose a justified larger-parent or better-retained-space calculation.

**Next calculation:** localize the changed channels or competing subspace before choosing more compute. **Question:** is the shift a numerical identity/representation issue or genuine sensitivity to newly represented states? Reassess the three paths separately. Stable comparators may proceed as controls; no aggregate score should hide a failed edge or carrier. Recompute affected fixed pulses only after the static input is accepted.

## C — device array data become available

Check the specific device and branch, coordinate normalization, static inductance recovery and possible capacitance double counting. Reduce the full circuit matrices before inverting capacitance. Retain the first internal mode with its derived coupling, and check against additional internal modes where justified. State/subspace tracking must include old Gridium states tensored with array excitations; compare outgoing operators as well as projector continuity.

**Next calculation:** a validated linear network reduction followed by a targeted dressed-spectrum/avoided-crossing calculation when resources and authorization permit. **Question:** does the extracted array actually hybridize the relevant states and change the real control operator/pathways? Re-evaluate model-specific controllability rather than inheriting su(48). Determine whether port coupling directly reaches the array. C is an independent branch and can expose model error even if A passed.

## D — Thomas or external compute remains unavailable

Separate person-dependent information from compute access. Another suitably resourced machine can supply the frozen numerical run; it cannot supply missing device facts. With only saved data, finish interface/schema checks, compare existing artifacts, and define target/metric requirements. Maintain the unresolved status rather than substituting a guessed array coupling or calling k180 converged.

**Next calculation:** only bounded saved-matrix algebra or a specific software contract test that addresses a new question. **Question:** is the workflow ready to consume new evidence without mixing frames, masks or metrics? Do not repeat already completed audits to create the appearance of progress.

## E — no array data become available

Continue explicitly conditional four-mode research if it answers a new well-defined question. A normalized mathematical perturbation or coupling-agnostic toy can test methods, but must not be presented as a device prediction, physical error bar or evidence of pathway survival. A predictive five-mode comparison remains blocked.

**Next calculation:** after numerical convergence and a drive/target specification, a bounded comparison of existing pulse mechanisms under common constraints; if those inputs are absent, no new dynamics is justified yet. **Question:** what follows from the declared four-mode control problem alone? Report that physical completeness is unresolved regardless of the numerical outcome.

## Work not worth compute/credits now

- Repeating Stage2 reconstruction or diagonalization merely to regenerate already saved vectors, static products, characterization or overlap tables.
- Repeating a generic full Lie closure on the same matrices, especially 48 states; the independent certificate already answers that question.
- Broad pulse optimization before a fixed target, feasible waveform constraints and accepted numerical/model inputs exist.
- Re-optimizing the 70 ns result with best-axis scoring while claiming a fixed-X objective, or comparing pulse families with unequal undeclared amplitude limits.
- Scanning guessed array frequencies and independent arbitrary couplings as if they were device tolerances.
- Broad eC/eP/asymmetry sweeps justified solely by local derivatives or examples from different devices; use device questions and tracking first.
- More arbitrary scalar robustness rankings, raw-product rankings or dense plots that add no new observable beyond existing tables.
- Treating sub-1e−9 integration refinement as useful while state-window/parent/model errors are larger.
- Rerunning a mock k220 return for scientific confidence, or changing frozen thresholds after inspecting a real return.
- Launching a larger parent automatically when k200/220 already answer a narrowly defined retained-space question; choose new work based on the actual failed diagnostic.

None of these exclusions forbids targeted work once its missing input or scientific question is supplied.
