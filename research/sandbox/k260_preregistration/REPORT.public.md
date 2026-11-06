> Publication copy: local machine paths have been replaced by repository-relative paths or `python`. Scientific statements and values are unchanged. The original file and its SHA-256 are recorded in [the publication manifest](../../publication/MANIFEST.json).

# Gridium k260 retained-basis preregistration

## Status

**K260 PREREGISTRATION STATUS: READY**

The experiment is preregistered before any k260 result exists. No large
eigensolve was run, k260 was not launched, no authenticated checkpoint was
changed, and no tracked production file was edited for this preparation.

## Authoritative k220 finding carried forward

The authenticated k220 postmortem found that k180 is not on the retained-basis
plateau. Its exact overall failure driver was the k180-to-k220 `f78` change of
2.698552 MHz, beyond both the frozen strict fine-carrier 1 MHz borderline
ceiling and practical 2 MHz ceiling. `f21` changed 5.342176 MHz and `f03`
6.242679 MHz, causing strict failures and practical borderline results for the
0721 and 0381 comparator paths. Global carriers were strict borderline and
practical pass. State overlaps, all important matrix elements, all pathway
scores, ordering, and parent solver quality passed strict limits.

The reason to request k260 rather than stop at k240 is diagnostic, not a change
of tolerance. `f72`, `f21`, `f03`, `g07`, and `g38` had growing last increments;
`f38`, `g72`, and both comparator scores reversed sign. The primary-frequency
tail was consistent with many weak virtual-shell contributions, while ranks
218--219 still had material participation. States 7/8 remained unambiguous and
the `{0,1,7,8}` operator blocks drifted only weakly. Two new equal 20-rank
increments are therefore needed to distinguish shrinkage from cancellation.

## Runner capability decision

The tracked runner cannot select k260 directly from its unchanged CLI because
its profile registry contains only `n71-k180` and `n71-k220`. No tracked change
is required, however. Once a profile is registered, the production computation
is generic in parent size and retained cases. The sandbox-only
`k260_profile.py` injects the explicit profile in memory before invoking that
unchanged implementation.

| Capability inspected | Result |
|---|---|
| Parent size and retained list | Only profile registration is hard-coded; computation uses configured values |
| Checkpoint schema and fingerprint | Generic; fingerprint is canonical fixed configuration with `k=260` |
| Stage-1 vector storage | Shapes derive from requested parent `k` |
| Nested Stage-2 construction | Generic prefix indices; expected dimension is `260 x 8 = 2080` |
| Operator projections and pathway report | Generic over parent and retained cases |
| State overlap tracking | Generic, with k260 endpoint and the existing assignment gates |
| Residual and orthogonality report | Generic and covered by unchanged quality limits |
| `COMPLETE`, hashes, and atomic publication | Generic; final directory is promoted only after validation |
| Partial output and overwrite protection | Generic unique hidden partial, refusal to overwrite final output |

This route preserves scientific semantics and adds auditable sandbox
provenance without changing production code. The guard additionally binds the
exact retained-case list and frozen criteria hash because retained cases are
not themselves part of the runner's configuration fingerprint.

## Proposed experiment

Use one parent with `N2=N3=71`, `L2=11`, `L3=14`, `N4=8`, `n1max=4`, `k=260`,
`sigma=-24`, `which=LM`, `tol=1e-8`, and `MMD_AT_PLUS_A`. Preserve all physical
parameters, numerical conventions, and thread settings from k220. Analyze
same-parent prefix spaces `k=180,200,220,240,260`.

- Expected fingerprint:
  `990dad04b52deeef9d90908c6c19ce38e917d7f5b7fab32f0e39699c3c83c4bb`
- Future checkpoint:
  `research/checkpoints/four-mode-asymmetric-n71-k260`
- Frozen criteria: `criteria.json`
- Automated return analysis: `analyze_k260_return.py`

## Frozen criteria and plateau rule

All k220 thresholds are carried unchanged. Standard carriers use strict
PASS/BORDERLINE ceilings of 1/5 MHz and practical ceilings of 5/10 MHz. Fine
`f78` uses 0.5/1 MHz strict and 1/2 MHz practical. State and subspace overlaps
use 0.999/0.995 strict and 0.995/0.990 practical minima. Grid-phi edges and
basis-invariant operator blocks use 3%/5% strict and 5%/10% practical ceilings.
Path scores use 5%/10% strict and 10%/20% practical ceilings. Parent residual
and orthogonality limits remain `1e-7` and `1e-8`.

Every mandatory metric is classified independently for k220->k260 total
change and k240->k260 final increment. Mandatory gates comprise all specified
carriers, states 2/3/7/8 and tracked subspace, seven grid-phi edges, three path
scores, three basis-invariant operator blocks, path ordering, tracking
confidence, parent quality, and checkpoint authentication. Adjacent-step
trends, shell weights, and boundary indicators support the diagnosis but do not
override a mandatory gate.

The preregistered overall rule is:

- **PASS A — convincing approach:** k220 and k240 both strictly pass every
  mandatory endpoint comparison; ordering, tracking, subspace, quality, and
  authentication pass; previously non-passing carriers shrink over the last
  two equal steps with consistent sign or an explicit basis-invariant
  explanation; operator blocks strictly pass.
- **BORDERLINE B — still converging:** no practical mandatory failure and the
  final k240 comparison is at worst strict borderline, but a strict issue or
  failure to demonstrate shrinking remains, without ambiguity, quality/order
  failure, or unexplained new final reversal.
- **FAIL C — non-monotonic/unresolved:** any practical mandatory failure, any
  strict k240 failure, a growing or reversing final step in a historically
  problematic carrier without a basis-invariant explanation, or any
  authentication, quality, ordering, or required tracking/subspace failure.

FAIL takes precedence over BORDERLINE, then PASS. A small last step cannot
erase earlier non-monotonic behavior.

## Resource feasibility and future command

Because the fixed N71 sparse problem and factorization are unchanged, the
increase chiefly affects ARPACK and eigenvector storage. From the authenticated
k220 timing of 146.12 s and approximately 1.80 GiB observed live RSS, plan for
3--5 minutes (conservative window 3--7 minutes) and about 2--3 GiB RSS while
reserving 4--6.5 GiB headroom. The existing dedicated environment and guarded
atomic execution method remain suitable.

After a separate launch review, the exact command is:

```bash
bash research/sandbox/k260_preregistration/execute_after_review.sh
```

It was not run during preregistration.

## Scientific limits

k260 will be another endpoint, not ground truth. This study addresses retained
Stage-1 convergence only at fixed N71 and in the fixed four-mode model. It does
not establish spatial/domain, `n1max`, or `N4` convergence. It does not resolve
omitted array-mode physics, validate a physical microwave port, or validate
X90/X180.
