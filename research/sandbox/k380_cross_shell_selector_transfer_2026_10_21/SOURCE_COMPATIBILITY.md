# Source and base compatibility

**BASE-COMPATIBLE LABELS: PARTIAL.** S1 is evaluable; the frozen S2 selector is
not evaluable against an exact matching saved Schur attribution.

| Comparison | Selection base/support | Available attribution labels | Finding |
| --- | --- | --- | --- |
| Historical physics S1 | P=[0,340), candidates[340,360), full endpoint[0,360) | Exact P340,Q[340,360), endpointk360 | Compatible same-parent/base/support; evaluate saved order |
| Historical physics S2 | P=[0,340), candidates[360,380), full endpoint[0,340) union[360,380) | Exact P360,Q[360,380), endpoint[0,380) | INCOMPLETE: retained P and full endpoint differ |
| Hypothetical refreshed-P360 selector S2 | P=[0,360), candidates[360,380) | Exact P360 labels exist | INCOMPLETE: no saved matching P360 physics features/order; no new reconstruction |
| S2 bare/prefix/spread baselines | Base-independent rule, selected support explicitly P360 plus candidates | Exact P360,Q[360,380) | Valid baseline-only ranking/capture, not a P340 selector transfer evaluation |
| Historical rank328 | k340 parent, exactP320; its selectorP300 | Earlier separate-parent attribution | Context already documented; excluded from this test |

All intervals are half-open and use zero-based Stage-1 rank indices. Every
retained rank has eight mode-4 oscillator states. All current labels/selection
identities concern the authenticated k380 parent with configuration
`8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`.

## Why S1 scalar capture is meaningful but not an energy ablation

S1 has the same P340 in the selector and exact label decomposition, the same
candidate ranks and full finite endpointk360. The selector uses base eigenvectors
and pair-mean isolated denominators; labels use the actual full-endpoint P
directions and mixed-shell resolvent. Different approximate reference vectors
are expected in a prediction/evaluation comparison and are not label leakage.

For a selected subset S, sum_(r in S)W_r / sum_(all Q)W_r is exactly the fraction
of this fixed endpoint's aggregate absolute attribution assigned to S. It does
not equal the correction after solving P plus S: both P vectors and the Q
resolvent can change upon truncation. Signed transition shifts, gate performance
and selected-space energy convergence cannot be inferred from capture alone.

## What is missing for S2

The frozen selector did not refresh P after S1. In contrast, exact S2 Schur
labels use P360, which includes340–359. This changes H_PP, the target P
directions, H_QP and the finite full endpoint. Equal candidate rank names do not
make the response functions or target vectors identical.

No exact rank-attribution record exists here for P340 with isolated Q360–379.
The prior targeted dependency review already established that the full historical
noncontiguous S2 endpoint metrics and endpoint-specific assignment were not
persisted. Existing historical S2 selected-space comparison summaries exist;
they are not a complete independent endpoint record and contain no exact
P340-based per-rank Schur labels. A full endpoint record alone would also not
supply the missing rank contractions.

The inspected analyzer constructs its noncontiguous S2 endpoint in memory but
serializes only `selector_secondary/comparisons`, not a matching endpoint
attribution object. Its `shell_diagnostics/S2` is explicitly P360. Checkpoint
analysis retains only contiguous prefixes300,320,340,360,380. This local source
inspection confirms the already recorded compatibility issue; no broad new
historical search or dependency audit was conducted.

No label was reverse-engineered from target errors, isolated-rank traces or
missing dressed vectors. The missing endpoint is not newly authenticated.
No P360 attribution was silently assigned to the P340 physics order. Its four
budget rows preserve selected identities but leave all evaluation metrics null.

## Generalization categories

- **Historical label-independent selection:** supported locally by sealed source,
  recorded selection chronology and explicit exclusion of candidate outcomes.
- **Historical independent validation:** no new claim. Earlier frozen verdicts
  are unchanged; this run adds no independent parameter/device/parent data.
- **Retrospective compatible S1 evaluation:** supported, using newly specified
  rank-capture diagnostics on previously known labels.
- **Retrospective cross-shell transfer S1↔S2 with matching labels:** INCOMPLETE.
- **Earlier WP2 in-sample association:** remains explanatory; scalar BaseC/Trace
  correlations are not the historical greedy algorithm or new validation.
- **Internal algebraic consistency:** selection SHA links, stored-order/budget
  support checks and metrics sums pass, without new scientific solves.
- **Insufficient data:** exact P340-based S2 labels, or an independently saved
  P360 physics selector with matching frozen provenance, are unavailable.

No new selector PASS, historical endpoint recovery, protocol alteration or
release of v1/v2 HOLD follows from the valid S1 results.
