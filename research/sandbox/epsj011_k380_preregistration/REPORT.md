# k380 preregistration assessment

Prepared 2026-10-11 from authenticated results; no k380 result exists.
See criteria.json for the exact configuration and numerical rules,
preregistration_seal.json for fingerprints, and execution_plan.md for the
separate authorization boundary. All files are sandbox-only.

## Evidence and scientific justification

Authenticated k340 remains retained-basis FAIL and tail D — NONMONOTONIC UNRESOLVED.
Maximum carrier corrections are 0.714572, 1.299836, and 2.013593 MHz for 300→320,
320→340, and 300→340. All practical components pass; 13 strict misses are
BORDERLINE and none strict FAIL. Seven of nine final carrier increments grow;
f01 reverses and f78 has same-shell cancellation. The evaluator has 61 failure
entries: 60 nonshrinking component entries plus the f01 reversal/cancellation
entry. This corrects shorthand counting without changing any historical verdict.

Historical S1/S2 aggregate weights are 2.142150477769782/4.074080434450945 MHz.
Rank 328 supplies about 79.5% of the final shell; rank 332 follows. The primary
path remains rank1, the k340 secondary selector passes, and f78-f01 is +0.713500 MHz.
Historical hashes, raw comparisons and shell diagnostics are pinned in
historical_evidence.json. Historical numerical audits were not rerun.

One k380 expansion is scientifically justified: it adds two equal-width shells
to test whether influential directions persist above 340. It distinguishes
isolated, recurring concentrated, smaller distributed, and unresolved behavior
within this range. It cannot prove an asymptotic tail law from two new shells.
Prospective hypothesis definitions retain inherited concentration/resonance
thresholds but address recurrence of the smaller rank-328-type correction,
not only recurrence of >10 MHz events. Historical criteria are unchanged.

Meaningful plateau evidence requires all three primary comparisons strict PASS,
stable tracking/pathway order, nonincreasing error in every mandatory component,
and no meaningful reversal/cancellation. A tiny final shell alone is insufficient.
Recurring concentrated influence, nonadditivity, cancellation, unsafe tracking,
or renewed nonmonotonicity calls for shell attribution and model-reduction review
before proposing another expansion. This preregistration authorizes no k400.

Even a complete PASS leaves spatial/domain/N4/n1max and model adequacy unresolved,
does not validate a device or control port, and establishes no logical gate.
GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD. k380 is a finite endpoint.

## Implementation and compatibility

The generic production runner is reused unchanged. Sandbox files register the
profile, freeze the exact selector before validation, authenticate, analyze and
evaluate. criteria.json contains stage1_orthogonality_inf_max=1e-8 and the exact
inherited zero_reference rule with floor=1e-12. Neither known post-solve key error
requires a runtime adapter. Static key-access tests and a complete mocked analyzer
exercise every orchestrated comparison/selector context without research solves.

Additional tests cover threshold boundaries, first-shell failures, cancellation,
concentration hypotheses, Schur identity on a tiny synthetic diagonal matrix,
authorization rejection, locks, and no-solve profile manifests. Genuine complex
principal angles are explicitly distinguished from the inherited absolute-block
subspace classification. See analysis_plan.md for prospective enforcement details.

Final cheap-test and guarded dry-preflight records are saved separately. They
certify preparation, not numerical results. No selector rank identities have
been fabricated; they must be frozen from the future parent before validation.

Next action: review this sealed protocol and, only if approved separately,
authorize one guarded k380 run after renewed safety gates.
