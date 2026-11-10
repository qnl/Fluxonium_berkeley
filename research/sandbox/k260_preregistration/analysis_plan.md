# k260 retained-basis analysis plan

## Scope and inputs

The future analysis is restricted to same-parent Stage-1 prefix spaces
`k=180,200,220,240,260` reconstructed from one authenticated N71 k260 parent.
The fixed model, spatial cutoffs, solver settings, and thread settings are those
recorded in `criteria.json`; only the requested parent eigenpair count changes
from 220 to 260. The k260 endpoint is a comparison endpoint, not ground truth.

The analyzer will first authenticate the checkpoint, its `COMPLETE` marker,
configuration fingerprint, source provenance, saved-array hashes, parent solver
diagnostics, and exact prefix retained indices. It will also reauthenticate the
historical N51, N71 k180, and N71 k220 references and require the preregistered
k220 checkpoint hash.

No Stage-1 Hamiltonian or sparse factorization is reconstructed. The analyzer
uses the saved k260 Stage-1 eigenpairs and projected operators to solve only the
small Stage-2 problems needed for consistent state/subspace diagnostics.

## Required report

For every mandatory quantity, report the five values in rank order and:

- adjacent signed changes for 180->200, 200->220, 220->240, and 240->260;
- total signed changes for 180->260 and 220->260;
- absolute successive-step ratios;
- monotonicity and every sign reversal;
- independent strict and practical classifications for k220 versus k260 and
  k240 versus k260.

Mandatory quantities are:

- carriers `f07`, `f08`, `f78`, `f81`, `f72`, `f21`, `f03`, and `f38`;
- endpoint state overlaps for states 2, 3, 7, and 8, plus assignment overlap,
  competitor ratio, and overlap margin gates;
- the minimum singular value for the tracked `{0,1,7,8}` subspace;
- `g07`, `g78`, `g81`, `g72`, `g21`, `g03`, and `g38`;
- pathway scores `0-7-8-1`, `0-7-2-1`, and `0-3-8-1`;
- Frobenius norms and every singular value of the full `{0,1,7,8}` operator
  block, logical-to-`{7,8}` block, and internal `{7,8}` block;
- target-path ordering, parent residual/orthogonality quality, and checkpoint
  authentication.

Shell weights and boundary indicators recovered from saved vectors are
supporting diagnostics only. Any extrapolation is non-authoritative and may not
replace the endpoint classifications.

## Classification and trend logic

The analyzer reads the frozen thresholds and decision rule from `criteria.json`.
It does not tune thresholds from k260 results. Component status is calculated
before the overall verdict, with `FAIL` taking precedence over `BORDERLINE` and
`BORDERLINE` over `PASS`.

The plateau decision uses the full five-point series. A small 240->260 step
cannot erase an earlier sign reversal or growing 220->240 increment. A reversal
may be resolved only by an explicit basis-invariant subspace explanation; it
must remain visible in the report. The three verdicts are:

- **PASS A:** convincing approach to a retained-basis plateau under every
  condition in `plateau_decision_rule.PASS_A_convincing_approach`.
- **BORDERLINE B:** still converging but not yet sufficient under every
  applicable condition in `BORDERLINE_B_still_converging`.
- **FAIL C:** non-monotonic or unresolved behavior when any condition in
  `FAIL_C_nonmonotonic_or_unresolved` is met.

## Tool and outputs

After a successful future run, execute:

```bash
bash research/sandbox/k220_environment_reconstruction/python.sh \
  research/sandbox/k260_preregistration/analyze_k260_return.py \
  research/checkpoints/four-mode-asymmetric-n71-k260 \
  --output-dir research/sandbox/k260_preregistration/k260_return_analysis
```

The analyzer writes a machine-readable JSON result and flat CSV summaries in
the requested output directory. Its schema self-test is safe before k260 data
exist:

```bash
bash research/sandbox/k220_environment_reconstruction/python.sh \
  research/sandbox/k260_preregistration/analyze_k260_return.py \
  --schema-self-test
```

The self-test checks classification boundaries and report assembly only; it
does not invent or simulate scientific k260 results.

## Scientific limits

This study addresses retained Stage-1 convergence only at fixed N71 and within
the fixed four-mode model. It does not establish spatial/domain, `n1max`, or
`N4` convergence; resolve omitted array-mode physics; validate a physical
microwave port; or validate X90/X180.
