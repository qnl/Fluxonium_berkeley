# Frozen comparative epsilon_J=0.11 shell-attribution methodology

Frozen before inspecting any new epsilon_J=0.11 shell couplings,
denominators, occupations, Schur rank/mode attributions, or counterfactual
rank addbacks. The authenticated endpoint and frozen selector-validation
outcomes are already known. This is post-result mechanistic analysis, not an
independent prospective prediction and not a new convergence test.

## Authenticated finite inputs

Use only the authenticated N71 k300 checkpoints at epsilon_J=0.11 and 0.10,
their saved Stage-1 eigensystems and projected operators, their nested
Stage-2 analyses, and the already frozen selection/validation artifacts.
Verify COMPLETE, configuration/checkpoint/payload hashes, selection seals and
checkpoint links, exact prefixes 180/200/220/240/260/280/300, and numerical
quality before calculation. No Stage-1 eigensolve is permitted.

Reconstruct the epsilon_J=0.11 Stage-2 Hamiltonian using the unchanged
production formula in GHz (H/h, no 2pi):

    H = diag(E_stage1) tensor I8 + I tensor H4
        + sum_i 8 EC[i,3] n_i tensor p4
        + K[0,2] x2 tensor x4 + K[1,2] x3 tensor x4.

Tensor order is Stage-1 rank followed by mode-4 number 0..7. Define P as
ranks 0..279 (2240 dimensions), Q as ranks 280..299 (160), and full as 2400.
For nonadditivity, Q1 is 280..289 and Q2 is 290..299. The exact Schur operator
and eliminated component are

    H_eff(E) = H_PP + H_PQ (E I - H_QQ)^(-1) H_QP,
    q = (E I - H_QQ)^(-1) H_QP p.

Signed denominators are E-epsilon. Use shell diagonalization/linear solves,
never an explicit inverse. Retain Q1-Q2 coupling except in the explicitly
labeled no-cross diagnostic.

## Reconstruction and state tracking

Solve only finite Stage-2 matrices for the lowest eleven dressed states.
Track reduced and counterfactual cases to the authenticated epsilon_J=0.11
k300 endpoint by maximum-total-absolute-overlap assignment after zero
padding. The frozen gates are assigned overlap >=0.90, competitor/assigned
ratio <0.50, and overlap margin >0.25. Mandatory states are 0,1,2,3,7,8;
mandatory transitions are f01,f07,f08,f78,f81,f72,f21,f03,f38.

Reconstructed k260, k280, and k300 energies and absolute grid_phi matrices
must match saved results within 1e-10 GHz/operator units. Hermiticity max
entry must be <=1e-12. Stage-2 eigenpair, Schur effective-equation, and exact
shift-identity errors must be <=1e-9 GHz; q-vector error must be <=1e-8.

For cross-parameter Stage-1 rank identity, compare raw-grid eigenvectors with
phase-insensitive overlaps. Match epsilon_J=0.11 ranks 240..299 against all
authenticated epsilon_J=0.10 ranks by one-to-one maximum-overlap assignment.
A rank-specific comparison is safe only when assigned overlap >=0.90,
competitor ratio <0.50, and margin >0.25. Otherwise restrict conclusions to
the tracked shell subspace and do not equate numeric rank labels.

## State, rank, and mode attribution

For each tracked state report the exact k280->k300 shift and identity

    deltaE = <p|Sigma(E)|p>/<p|p>
             + <p|(H_PP-E_280)|p>/<p|p>.

The terms are exact shell self-energy and P-space relaxation. Also report the
fixed-vector second-order estimate

    sum_mu |<mu_Q|H_QP|psi_280>|^2/(E_280-epsilon_mu)

and its signed error. Report Q occupation, coupling strengths, signed
denominators, eta=|coupling/denominator|, exact rank contributions
Re[(H_QP p)_r^* q_r]/<p|p>, and invariant shell-eigenmode contributions.
Rank attribution is basis-dependent; shell-mode attribution is invariant to
unitary rotations of Q.

For absolute rank/mode contributions, sort descending with lower rank/mode as
the tie break. Report minimum counts reaching 50%, 80%, and 90%, top-three
fraction, and inverse participation 1/sum(w^2). Signed sums are authoritative;
absolute fractions diagnose concentration and cancellation.

Near resonance is |E_280-epsilon_mu|<=0.010 GHz OR eta>=0.10. A fixed-vector
second-order diagnostic is controlled only when its shift error is <=20% of
the exact shift (absolute floor 1e-6 GHz), max eta<0.10, and Q probability
<0.01.

Mechanism labels reproduce the validated epsilon_J=0.10 definitions:

- A: near-resonant modes carry >=50% of absolute second-order weight and
  either the top three modes carry >=70% or <=5 modes reach 80%.
- B: near-resonant fraction <10%, exact rank top-three fraction >=70%, and
  <=5 ranks reach 80% (concentrated off-resonant ranks).
- C: near-resonant fraction <10%, both rank and mode top-three fractions
  <=50%, and >=8 ranks and >=8 modes reach 80% (distributed).
- D: every resolved mixed/conflicting case not satisfying A/B/C.
- E: reconstruction, elimination, or tracking failure prevents resolution.

The overall final-shell label is the common state-7/state-8 label when they
agree; otherwise D, unless either is E. Cross-parameter mechanism is
SUPPORTED only if both points are resolved in the same A/B/C class, rank
identity is safe, and concentration/coupling-denominator evidence supports
the same mechanism. It is PARTIAL for a shared broad mechanism with material
rank/concentration changes; otherwise NOT SUPPORTED.

## Nonadditivity and selector diagnostics

Compute exact P+Q1, P+Q2, full P+Q1+Q2, and full with Q1-Q2 blocks zeroed.
Report total nonadditivity, explicit Q1-Q2 contribution, and P-mediated
remainder for each state/transition. The no-cross matrix is diagnostic only.

Define influential ranks after attribution as the three Q ranks with largest
sum of absolute exact rank self-energy over states 0,1,2,3,7,8. Solve P plus
each singleton, pair, and triple; report Möbius interaction terms. These are
descriptive exact addbacks, not prospective selection.

For frozen-selector interpretation, compare its unchanged S1/S2 rank orders
at budgets 3/6/10/15 with exact isolated-shell rank attribution. Report
captured absolute Schur weight, target-group feature coverage, carrier and
pathway errors, and omitted important ranks without refitting.

For union/b10 f78, reconstruct P260 plus the frozen union physics set and the
ordinary global-prefix set, verifying +0.220384 and -0.140580 MHz against the
saved metrics. Report raw and excitation state-7/8 errors, differential f78,
selected-rank differences, captured/omitted signed and absolute isolated
Schur weight, and exact subset results. Classify the exception as genuine
differential recovery, cancellation of larger individual errors, mixed, or
unresolved. Do not change the frozen BORDERLINE verdict.

## Frequency selectivity and scope

Report k260/k280/k300 f01, f78, f78-f01, f07, f81, f07-f81, g01, and g78 for
both parameter points. This is a static spectral diagnostic only; do not
infer time-domain gate fidelity or run gate optimization.

All new outputs go only to research/sandbox/epsj011_shell_attribution. Verify
tracked-file and authenticated-checkpoint hashes before and after. Preserve
the figure backlog and generate no figures. This fixed-N71, fixed-four-mode
analysis does not establish spatial/domain/n1max/N4/omitted-mode convergence.
