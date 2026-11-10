# Work Block 4 — P340 / S2 base-matched validation

New retrospective study, October 23, 2026 (America/Los_Angeles). This is separate
from sealed v1/v2; neither benchmark nor its prelaunch is invoked. Authorization:
at most two new Stage-2 dressed solves, zero Stage-1 solves; no automatic retries.

P=[0,340); Q=[360,380); U=P union Q. Ranks [340,360) are excluded. Rank-major,
mode4-minor C ordering, eight oscillator states per rank. P/U dimensions are
2720/2880; Q dimension is 160. Request 11 tracked dressed states. H/h and E/h
are GHz, not angular frequencies; convert final scalars to MHz with 1000 only.
Preserve the source Hamiltonian and its energy zero. Five channels enter as a
complex coherent sum: 8 EC[i,3] n_i tensor n4, K[0,2] x2 tensor theta4,
K[1,2] x3 tensor theta4. grid_phi=(x3+x2/2) tensor I8 has zero selector weight.

## Frozen selector

Use the original sealed order, never regenerate it from labels. Target pairs
are (0,1),(2,3),(7,8). B_rg=H_QrP U_g, mu_g=mean base pair energy,
A_rg=B_rg†(D_r-mu_g I)^-1 B_rg. W_g=sum_r A_rg, T_g=W_g^-1/2 with
eigenvalue floor max(1e-14 GHz,1e-10 lambda_max); C_rg=T_g A_rg T_g.
Greedy gain maximizes (1/6) sum_g logdet(I2+6 sum_selected C_rg).
Gain ties within 1e-12 choose lower rank. Isolated denominators must exceed
0.010 GHz, eta<0.10; no pole clipping. Budgets 3,6,10,15. Full original features,
normalization, hashes, control definitions and exact orders are recorded in
FROZEN_SELECTOR_RECORD.json before any new attribution calculation.
The frozen bare T300, Dplus, Vplus controls use p<300 and remain comparable.
The saved S2 Tactual uses p<360 and is ineligible for this P340 comparison.
Prefix/spread controls retain P340 plus only their selected S2 ranks. Bare
controls were defined after historical outcomes and are retrospective controls.

## Exact historical attribution, distinct from second order

Source: authenticated analyze_k380.py::shell_diagnostic, lines 65–151, and
diagnostic-review RANK_MECHANISM.md. For each full-U tracked state s, split its
complex vector into p_s=P psi_s, q_s=Q psi_s and use full-U E_s. Define
d_s=H_QP p_s and predicted q_s=(E_s I-H_QQ)^-1 d_s. Historical exact per-rank
scalar sigma_rs=1000 Re[d_rs† q_rs]/(p_s†p_s), retaining all eight entries of
each rank and all cross-rank resolvent mixing. W_r=sum_{s=0,1,2,3,7,8}|sigma_rs|.
Keep signed and complex intermediate values before Re/absolute aggregation.
Modal denominators are E_s-lambda_m (signed), not positive selector gaps.
State families are the three frozen pairs; every rank 360..379 remains present.

The P340 base eigenvectors u_s and energies e_s identify tracked base states
and separately define fixed-base second order:
1000 (H_QP u_s)†(e_s I-H_QQ)^-1(H_QP u_s). They are NOT substituted for
the full-U P component or full-U energy in the primary historical attribution.
The exact identity is E_s-e_s=sigma_s/1000+
[p_s†H_PP p_s/(p_s†p_s)-e_s]. The second term is P relaxation.
Thus exact per-rank sigma, fixed-base second order, collective matrix
H_PQ(E I-H_QQ)^-1 H_QP and directly diagonalized U energies remain distinct.
The noncontiguous gather must replace historical contiguous slices, preserving
operator ordering and original source ranks. No unimplemented formula is claimed
numerically verified; all new contractions remain blocked if any launch gate fails.

## Required validation and scope

Reuse an exact authenticated compatible vector cache if found before either
solve. Historical sources say dressed vectors were not saved; no exhaustive
cache search is claimed if execution stops at resource preflight. Do not load
raw Stage-1 eigenvectors, assemble full Stage-1 H or full k380 dressed H.
Validate Hermiticity, rank/tensor identity, independent tensor action, residuals,
orthogonality, assignment confidence, overlaps, near-degenerate subspaces,
grid_phi, transitions, units/zero and phase invariance. Validate P340 against
authenticated nested-prefix P340 energies/operators. U has no independent
historical endpoint: only numerical consistency can be established there.
No energy-index-only reassignment or silent gate omission is permitted.

Top-k recovery is intersection with descending W (lower-rank tie break), k=budget;
capture=sum_selected W/sum_all W. Report dominant inclusion and all controls.
These are attribution metrics, not energy accuracy. Selected-space accuracy is
conditional on supported small reduced problems, with individual/excitation and
all nine carrier errors, f01, f78, f78-f01, and constituent cancellation retained.
No additional large dressed solve is authorized for selected subspaces.

Before numerical launch require authenticated inputs/method, validated method
implementation, resource headroom, one numerical process, BLAS threads=1,
exclusive lock, atomic checkpoints, hard process memory ceiling 4 GiB, 900 s
total numerical time, and 1 GiB new output. The historical watchdog samples RSS
after allocations; it does not by itself establish this hard ceiling. Failure
stops numerical work and preserves evidence. No alternative experiment follows.

V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
