# Gridium epsilon_J=0.11 k340 preregistration

## Decision

**K340 PREREGISTRATION: READY**

**TRACKED CODE CHANGES: NOT REQUIRED**

No k340 calculation or new Stage-1 eigensolve was run. Authenticated
checkpoints and tracked production files were not modified. No figure was
generated.

## Authenticated evidence

The existing epsilon_J=0.11 k300 checkpoint authenticates with configuration
SHA-256 `8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646`
and checkpoint SHA-256
`20a58be10ad8408ce479de259e1d2f3562cc46222e0eebc939d6d695803516bd`.
Its Stage-1 maximum residual is 3.0423e-12 and orthogonality residual is
2.1585e-11, both well inside the frozen gates.

The exact authenticated k280-to-k300 carrier increments in MHz are:

| f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.0298 | -52.1780 | -52.0374 | +0.1406 | -52.0076 | -53.4145 | +1.2662 | +0.8492 | -52.8866 |

This supports the existing retained-basis `FAIL`. The shell reconstruction is
PASS; ranks 290/292 carry 84.417% of aggregate absolute attribution, rank 289
is third after a 20.86% coupling increase, no near-resonant mechanism was
found, and the exact frozen selector captures 99.5071% of S2 absolute Schur
weight at budget 10. The union/b10 f78 exception is cancellation of roughly
54 MHz individual prefix state errors, not individual-state recovery.

## Frozen configuration

epsilon_J=0.11, epsilon_LK=0.05; EJ=5, EC=0.5, EL=1, ELK=1, EJS=4,
ECS=8, eC=5.5, eP=10, ng=0, phi_ext=0, theta_ext=pi; n1max=4,
N2=N3=71, L2=11, L3=14, N4=8; k=340, sigma=-24, which=LM,
tol=1e-8, ncv=None, v0=None, MMD_AT_PLUS_A.

**Configuration fingerprint:**
`10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0`.

Same-parent prefixes are exactly 260, 280, 300, 320, 340. Original k300,
nested k300, k320, and k340 remain distinct objects.

## Frozen criteria

Strict/practical tolerances are unchanged: standard frequencies and centered/
excitation energies 1/5 MHz strict and 5/10 MHz practical; f78 0.5/1 MHz
strict and 1/2 MHz practical; overlaps 0.999/0.995 strict and 0.995/0.990
practical; operator relative errors 3%/5% strict and 5%/10% practical; pathway
scores 5%/10% strict and 10%/20% practical.

**Criteria SHA-256:**
`ffdbe09ae353c0eb6746a95c67b7c6dd1461db1a6c3e2d1260b61f831c0e4c72`.

**Preregistration manifest SHA-256:** recorded in
`preregistration_seal.json`; the guard recomputes it before every dry or live
invocation.

Mandatory checks include all nine carriers, f78-f01, states 0/1/2/3/7/8,
five subspaces, g01/g07/g78/g81, six grid_phi blocks and singular values,
0781/0721/0381 scores, parent quality, original-to-nested k300 tracking,
trend/cancellation, and exact new-shell occupation/coupling diagnostics.

## Scientific decision rule

PASS requires strict success for k300-to-k320, k320-to-k340, and k300-to-k340
on every mandatory component, shrinking final errors, and no reversal or
cancellation. BORDERLINE requires no practical failure, final-shell results no
worse than strict BORDERLINE, shrinking trends, and no unresolved cancellation,
with at least one strict miss. FAIL has precedence for any hard failure,
practical failure in any primary comparison, final-shell strict failure,
nonshrinking historically problematic carrier, or meaningful reversal/
cancellation.

Therefore a small k320-to-k340 result cannot erase a large k300-to-k320 jump.
A PASS supports only a plateau within the tested fixed-N71 range. It does not
establish convergence beyond k340, k340 as ground truth, or other spatial/model
convergence.

## Resource estimate

The sparse matrix and LU structure are unchanged from k300; k340 enlarges the
requested eigenspace and saved projections. Estimate 5--8 minutes for Stage 1
and 10--25 minutes total guarded wall time including analysis/publication.
Estimate 3--4 GiB working memory; reserve 6--8 GiB. Estimated checkpoint size
is 250--290 MiB. These are scaling estimates, not k340 measurements.

## Feasibility and validation

The generic runner can be registered entirely from this sandbox and already
provides single-run execution, hidden partials, atomic publication, COMPLETE-
last authentication, residual checks, and checkpoint hashes. The sandbox adds
the k340 profile, exact pre-validation selector freeze, original/nested k300
tracking, same-parent analysis, Schur diagnostics, evaluator, and read-only
guard. Cheap syntax, hash, profile, checkpoint, no-solve, and mock-boundary
tests are recorded in `test_results.json`.

## Next action

Conduct one prelaunch review of the sealed manifest, current memory pressure,
unique output paths, lock state, and authorization token. If every gate still
passes, authorize exactly one guarded k340 parent execution; otherwise revise
the preregistration before producing data.

**GATE WORK: HOLD**

**SPATIAL CONVERGENCE: HOLD**
