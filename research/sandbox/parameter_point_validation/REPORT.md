# Gridium independent parameter-point selector preregistration

## Decision

**PARAMETER-POINT PREREGISTRATION: READY**  
**SELECTOR REUSE: EXACT**  
**OVERFITTING PROTECTION: PASS**  
**TRACKED MODIFICATIONS REQUIRED: NO**

No new-point Hamiltonian was assembled, no Stage-1 eigensolve or new-point
dressed-space calculation was run, and no new scientific result was produced.
The parameter point, configuration, metrics, controls, cancellation rules,
decision precedence, and exact selector source are fixed before execution.

## Frozen parameter point

The held-out point is `eps_J=0.11`, `eps_LK=0.05`. Every other physical
parameter remains EJ=5, EC=0.5, EL=1, ELK=1, EJS=4, ECS=8, eC=5.5, eP=10,
ng=0, phi_ext=0, theta_ext=pi. The frozen configuration fingerprint is
`8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646`.

This is scientifically appropriate. `eps_J` is the existing dimensionless
KITE-junction imbalance in `EJ1,2=EJ_mean(1+/-eps_J)`, not a new Hamiltonian
term. Previously generated local sensitivity data report that +0.01 changes
f78 by approximately +3.04 MHz, the primary 0781 score by +3.82%, and g78/g81
by +1.694%/+2.119%. The shift is therefore large enough to challenge the
sensitive 7/8 sector while remaining a local perturbation within the same
four-mode model. It was proposed before inspecting any new-point result.

This tests parameter transfer within one mathematical model. It is not an
independent device, fabrication distribution, spatial-convergence test, or
physical-port validation.

## Exact frozen selector

The implementation is imported byte-for-byte from
`k300_preregistration/selector.py`, hash
`57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2`.
The specification hash is
`2055c7d54d81ada97379f4941ef08354dbb3cbf9378aa2b10dde1349ca2ff7cc`.

For each rank r and target pair g=(0,1),(2,3),(7,8), the algorithm constructs

    A_rg = B_rg^dagger (D_r-mu_g I)^(-1) B_rg,

whitens by the full-shell `W_g=sum_r A_rg`, and greedily maximizes

    F(S)=(1/6) sum_g log det[I_2+6 sum_(r in S) W_g^(-1/2)A_rgW_g^(-1/2)].

Equal pair weights, scale 6, floors 1e-14 GHz/1e-10 relative, eta and pole
gates, lower-rank tie break within 1e-12, independent shell ranking, and
budgets 3/6/10/15 are unchanged. grid_phi remains a validation outcome with
zero selection weight.

The new point uses its own Stage-1 eigenvalues, projections, and P260 dressed
states. Authenticated original k260 states only transfer physical labels by
raw-grid overlap. All eleven low states must pass the old tracking gates.
Previously successful ranks 223/225/230/290/292 are prohibited inputs and are
not encoded in the adapter.

## Leakage and overfitting protection

Both S1 and S2 orderings are exclusively written and sealed after the single
future Stage-1 solve but before any candidate-containing or full new-point
Stage-2 diagonalization. Only new-point P260 and isolated 8x8 candidate blocks
are permitted feature calculations. The checkpoint records the selection
hash and freeze timing.

The point, selector source, target pairs, objective, weights, floors, feature
gates, budgets, comparators, metrics, thresholds, and verdict precedence are
all sealed. A failure does not authorize another point, new weights, a rank
override, or threshold relaxation. The old rank lists and all new-point
endpoint/selected errors are forbidden selection inputs.

## Validation experiment

Run one future N71 k300 parent with N2=N3=71, L2=11, L3=14, N4=8, n1max=4,
sigma=-24, which=LM, tol=1e-8, and MMD_AT_PLUS_A. Preserve nested prefixes
180/200/220/240/260/280/300. S1 is ranks 260–279 and S2 280–299, both scored
against P=ranks 0–259.

At budgets 3/6/10/15 compare:

- exact frozen physics selector;
- equal-cost shell energy prefix;
- fixed spread and reverse-spread controls;
- for the union, ordinary global prefix and shell-balanced prefix.

Evaluate S1 and counterfactual S2 independently, then their union. Budget 10
is primary and budget 15 confirms stability.

Mandatory measurements include all nine carriers, states 0/1/2/3/7/8, five
subspaces, eight grid_phi edges, all norms/singular values of six operator
blocks, and scores/ranks of 0781/0721/0381. Success is balanced across the
01/23/78 families; improvement confined to states 7/8 is insufficient.

## Strengthened cancellation check

For candidate/endpoint raw state error `d_s`, subtract
`c=mean(d_0,d_1,d_2,d_3,d_7,d_8)` and report every centered error `z_s=d_s-c`.
Also report excitation error `x_s=d_s-d_0`. For transition i→j, flag
cancellation when

    |x_j-x_i|/(|x_i|+|x_j|) <= 0.25,

the denominator exceeds 1e-6 MHz, and either excitation error exceeds 1 MHz.
The flag is severe above 5 MHz. Centered and excitation errors use the carried
1/5 MHz strict and 5/10 MHz practical thresholds.

A flagged small transition is not counted as genuine shell recovery. Any
physics flag prevents PASS; a severe physics flag is FAIL. This method detects
the known reverse-spread S2/f78 example with ratio 0.000419 while leaving the
old BORDERLINE verdict untouched.

## Frozen verdicts

`criteria.json` gives exact precedence for three separate outcomes:

- PARAMETER-POINT SELECTOR VALIDATION: PASS/BORDERLINE/FAIL;
- MULTI-TARGET TRANSFER: SUPPORTED/PARTIAL/NOT SUPPORTED;
- RETAINED-BASIS CONVERGENCE: PASS/BORDERLINE/FAIL.

Selector PASS requires hard-gate success, no practical failures, strict b15
success, at least 20% b10 improvement over the best equal-cost control in all
families/contexts, no cancellation or no-harm miss, stable b15 errors, and
path-order preservation. Transfer SUPPORTED additionally requires the strong
condition independently for 01,23,78 in S1,S2,union. Convergence uses only the
new-point nested-prefix rule and remains distinct from endpoint agreement.

## Implementation and numerical feasibility

The existing generic sparse Stage-1 solver, payload writer, hidden partial,
atomic publication, and COMPLETE-last protocol can be reused through a
sandbox profile. A sandbox analysis replaces the production N51 comparison:
using an original-parameter N51 Stage-1 payload with new-point couplings would
be scientifically invalid. The replacement computes only same-parent nested
new-point results and requires no tracked code change.

Cheap tests passed for the exact selector hash, configuration fingerprint,
unitary pair invariance, deterministic ties, classifier precedence, known
cancellation detection, profile schema, and a dry manifest. The dry manifest
has dimension 45,369, Stage-2 dimension 2400, exact prefixes, and the unique
output `four-mode-asymmetric-n71-epsj011-k300`. Tests explicitly record that
no new-point matrix, eigensolve, or result was used.

Original k300 timing was 360.23 s internal (61.98 s factorization, 265.19 s
eigsh), with about 1004 s guarded wall time. Plan 6–10 minutes for the parent,
conservatively 5–17 minutes guarded wall time, plus 2–7 minutes selected-space
analysis. Peak RSS was unavailable; retain the 2.5–3.5 GiB working estimate
and reserve 5–8 GiB. Expected checkpoint size is 220–260 MiB.

## Operational decision

**NEXT ACTION:** conduct a separate prelaunch review of the sealed manifest,
filesystem/process gates, memory pressure, and unique output paths; if every
gate passes, execute the sandbox launcher exactly once without changing this
preregistration.

**GATE WORK: HOLD**  
**SPATIAL CONVERGENCE: HOLD**

The k260 shell-attribution and rank-addback figure plans are preserved. A
future k300 selector/cancellation comparison is added to the backlog. No graph
was generated.
