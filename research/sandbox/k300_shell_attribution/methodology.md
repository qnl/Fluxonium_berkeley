# Frozen k300 final-shell attribution methodology

Frozen on 2026-10-08 before inspecting any new k300 shell couplings,
denominators, occupations, rank/mode attributions, or counterfactual addbacks.
The authenticated endpoint and frozen selector validation are already known;
this is a post-result diagnostic protocol, not a prospective prediction and
not a new retained-basis convergence test.

## Authenticated inputs and finite Hamiltonian

Use only the authenticated N71 k300 checkpoint, its saved Stage-1 eigenvalues
and projected operators, its nested dressed-space analysis, and the already
frozen k300 selection/validation outputs. Verify COMPLETE, the configuration
fingerprint, checkpoint and payload hashes, the frozen selection seal/link,
exact nested prefixes, residual/orthogonality limits, and the unchanged N51,
k180, k220, and k260 references using the existing authentication record plus
fresh hashes. No Stage-1 eigensolve is permitted.

Reconstruct Stage 2 using the unchanged production formula and units (GHz,
H/h; no 2pi):

    H = diag(E_stage1) tensor I8 + I tensor H4
        + sum_i 8 EC[i,3] n_i tensor p4
        + K[0,2] x2 tensor x4 + K[1,2] x3 tensor x4.

Tensor order is Stage-1 rank followed by mode-4 number 0..7. Define P as
ranks 0..279 (2240 dimensions), Q as ranks 280..299 (160), and full as 2400.
For nonadditivity only, Q1 is 280..289 and Q2 is 290..299. The exact Schur
operator is

    H_eff(E) = H_PP + H_PQ (E I - H_QQ)^(-1) H_QP,

with q=(E I-H_QQ)^(-1)H_QP p. Signed denominators are E-epsilon. Use shell
diagonalization or linear solves, never an explicit inverse. Q1/Q2 mutual
coupling is retained except in the explicitly labeled no-cross diagnostic.

## Reconstruction and state tracking

Solve only finite Stage-2 matrices for the lowest eleven dressed states.
Track reduced/counterfactual states to the authenticated k300 endpoint by
maximum-total-absolute-overlap assignment after zero padding. Reuse the frozen
tracking gates: assigned overlap >=0.90, competitor ratio <0.50, and margin
>0.25. If a gate fails, qualify state-specific claims and report the relevant
subspace. Mandatory states are 0,1,2,3,7,8 and mandatory transitions are f01,
f07,f08,f78,f81,f72,f21,f03,f38.

Reconstructed nested k280 and k300 energies and absolute grid_phi entries must
match saved values within 1e-10 GHz/operator units. Hermiticity max entry must
be <=1e-12. Stage-2 eigenpair, Schur effective-equation, and exact shift
identity errors must be <=1e-9 GHz; reconstructed q-vector error <=1e-8.
These are the validated k260 tolerances and are not loosened.

## State, rank, and mode attribution

For each tracked state, report the observed k280->k300 shift and the exact
identity

    deltaE = <p|Sigma(E)|p>/<p|p>
             + <p|(H_PP-E_280)|p>/<p|p>,

where the terms are exact shell self-energy and P-space relaxation. Also
report the fixed-vector second-order estimate

    sum_mu |<mu_Q|H_QP|psi_280>|^2/(E_280-epsilon_mu)

and its signed error relative to the observed shift. Report Q occupation,
coupling strengths, signed denominators, eta=|coupling/denominator|, exact
rank contributions Re[(H_QP p)_r^* q_r]/<p|p>, and invariant shell-mode
contributions. Rank attribution is basis-dependent; shell eigenmode
attribution is invariant under a unitary change of Q coordinates.

For absolute rank or mode contributions, sort descending with lower rank/mode
as deterministic tie-break and report the minimum count reaching 50%, 80%,
and 90%, top-three fraction, and inverse participation 1/sum(w^2). Signed
sums remain authoritative; absolute fractions diagnose concentration and may
expose cancellation.

Near resonance is fixed at |E_280-epsilon_mu|<=0.010 GHz OR eta>=0.10. A
second-order diagnostic is controlled only if its shift error is <=20% of the
exact shift (absolute floor 1e-6 GHz), max eta<0.10, and Q probability<0.01.

Mechanism labels are fixed as follows. A: near-resonant modes contribute
>=50% of absolute second-order weight and either the top three modes carry
>=70% or <=5 modes reach 80%. B: near-resonant fraction <10%, and exact rank
weight is concentrated (top three ranks >=70% and <=5 ranks reach 80%). C:
near-resonant fraction <10% and the contribution is distributed in both
descriptions (rank and mode top-three fractions <=50%, with >=8 ranks and
>=8 modes required for 80%). D: every numerically resolved case not satisfying
A/B/C, including mixed rank/mode concentration, appreciable but non-dominant
resonance, or conflicting states. E: reconstruction, elimination, or tracking
failure prevents resolution. The overall final-shell label is the common
label of states 7/8 only when they agree; otherwise D, unless either is E.
Supporting states remain explicit. Transition shifts are signed differences
of state shifts, so cancellation is reported rather than assigned a separate
mechanism label.

## Nonadditivity and selector diagnostics

Compute exact P+Q1, P+Q2, full P+Q1+Q2, and a full matrix with Q1-Q2 blocks
zeroed. For each state and transition report

    total nonadditivity = E_full-E_PQ1-E_PQ2+E_P,
    explicit Q1Q2       = E_full-E_no_cross,
    P-mediated remainder= E_no_cross-E_PQ1-E_PQ2+E_P.

The no-cross matrix is diagnostic only. It is not a physical-model result.

Define influential ranks after attribution as the three Q ranks with largest
sum of absolute exact rank self-energy over states 0,1,2,3,7,8, ties to lower
rank. For these exactly specified post-attribution ranks, solve P plus each
singleton, each pair, and the triple. Report singleton/pair/triple shifts and
Möbius interaction terms. This is descriptive exact addback, not prospective
selection and not interchangeable with Schur self-energy attribution.

For selector interpretation, preserve the frozen physics, prefix, spread,
and reverse-spread rank orders and scores. Compare their selected ranks with
exact rank attribution for both held-out shells using existing S1 k260->k280
attribution and the new S2 attribution. Report group coverage and denominator/
coupling diagnostics without refitting weights. For S2 budget 10, reconstruct
P260 plus the frozen physics and reverse-spread sets, verify their saved f78
errors, and decompose state-7/state-8 shifts. Differences in exact subset
shifts, omitted exact rank weight, and signed cancellation diagnose the f78
exception; no selector revision is made.

## Collision and scope

Report k260/k280/k300 f01, f78, f78-f01, g01, g78, and outer separation
f07-f81 from authenticated same-parent nested spaces. The separation diagnoses
spectral proximity only. Coupling strengths inform possible driven crosstalk;
no time-domain fidelity or gate impossibility is inferred.

All new outputs go only to research/sandbox/k300_shell_attribution. Verify
tracked-file and authenticated-checkpoint hashes before and after. This study
addresses fixed-N71, fixed-four-mode retained Stage-1 attribution only. It does
not establish spatial/domain/n1max/N4/omitted-mode convergence or gate fidelity.
