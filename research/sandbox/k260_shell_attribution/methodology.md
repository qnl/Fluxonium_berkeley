# Frozen shell-attribution methodology

Preregistered 2026-10-08 before calculating new shell attribution. Existing
convergence results are known; these diagnostic definitions are fixed before
examining couplings, denominators, participation, or shell counterfactuals.
This does not alter the k260 convergence criteria or verdict.

## Data and finite Hamiltonian

Use only authenticated k260 Stage-1 eigenvalues, projected n1/n2/n3/x2/x3,
configuration, and saved nested analysis. Stage-1 vectors are available but
not needed to form Stage-2 blocks. Stage-2 vectors/Hamiltonian components are
not saved explicitly; reconstruct them from the unchanged _stage2_templates
formula. All energies are in GHz (H/h); no 2pi enters static calculations.
The tensor order is Stage-1 rank then mode-4 number 0..7.

P=ranks 0..219 (1760 dimensions), Q1=220..239 (160), Q2=240..259 (160).
Full dimension is 2080. For each adjacent prefix a->b, P is 0..a-1 and Q is
a..b-1, always tensored with all eight oscillator states.

H_eff(E)=H_PP+H_PQ(EI-H_QQ)^(-1)H_QP.
From the lower block equation q=(EI-H_QQ)^(-1)H_QP p. Denominators E-epsilon
are signed; shell levels above E give negative self energy. Use shell
diagonalization and linear solves, not an explicit inverse. Treat Q1+Q2 as
one coupled Q for the exact full correction. Independent Q1/Q2 self energies
are not additive when H_Q1Q2 is nonzero.

## Measurements and tracking

Recover lowest 11 states for all five prefixes; track to the k260 endpoint
by maximum-total-absolute-overlap assignment after zero padding. Prioritize
states 0,1,2,3,7,8 and transitions f01,f07,f08,f78,f81,f72,f21,f03,f38.
Reuse existing tracking gates (overlap>=0.90, competitor ratio<0.50,
margin>0.25); if crossed, qualify individual attribution and report the
{0,1,7,8} subspace. Near-degenerate states are solved together in the exact
finite Hamiltonian; no independent scalar PT result substitutes for them.

For each a->b and tracked state report observed shift and:

1. Exact Schur reconstruction q and effective-equation residual.
2. Exact shift identity deltaE = <p|Sigma(E)|p>/<p|p>
   + <p|(H_PP-E_a)|p>/<p|p>. The second term is P-space relaxation.
3. Second-order estimate sum_mu |<mu_Q|H_QP|psi_a>|^2/(E_a-epsilon_mu).
   Report its error without substituting it for the exact result.
4. Exact signed rank contributions Re[p^dag H_P,r q_r]/<p|p>, plus separate
   relaxation. These sum to exact self energy, not generally the entire shift.
   Rank attribution is tied to the saved Stage-1 eigenbasis, not invariant
   under rotations of degenerate Stage-1 states. Also report invariant shell
   spectral contributions |<mu|H_QP p>|^2/[(E-epsilon_mu)<p|p>].
5. Coupling strengths, denominators, Q weights, largest fractions, effective
   participation 1/sum(w_mu^2), and count reaching 90% absolute contribution.

Compute both standalone P+Q1 and P+Q2, full P+Q1+Q2, and the full matrix with
only Q1-Q2 couplings set to zero. This last matrix is a labeled diagnostic
counterfactual, never a modified physical model result. Decompose nonadditivity:
full-PQ1-PQ2+P = (full-no_cross) + (no_cross-PQ1-PQ2+P).
First term isolates explicit Q1-Q2 coupling; second contains nonadditive
P-mediated mixing/energy dependence. Never add these again to the full shift.

## Fixed numerical and interpretation criteria

- Match saved energies and absolute grid_phi entries within 1e-10 GHz or
  absolute operator units, respectively. Hermiticity max entry <=1e-12.
- Stage-2 eigenpair, Schur effective-equation, and exact shift-identity errors
  <=1e-9 GHz; reconstructed Q-vector error <=1e-8 in Euclidean norm.
- Near-resonant diagnostic: |E_a-epsilon_mu|<=0.010 GHz OR
  |<mu|H_QP psi_a>|/|E_a-epsilon_mu|>=0.10. Zero denominators are flagged;
  exact finite-block solution is authoritative. PT is controlled for this
  diagnostic only if shift error <=20% (absolute floor 1e-6 GHz), maximum
  coupling/denominator ratio<0.10, and shell probability<0.01.
- Concentrated spectral contribution: top three shell eigenmodes carry >=70%
  of absolute second-order contribution. Distributed: top three <=50% and
  effective participation>=8. Report Stage-1 rank concentration separately.
- State A: near-resonant terms carry >=50% and concentration criterion holds.
- State B: near-resonant fraction<10%, distributed criterion holds, and PT
  passes the controlled-diagnostic check.
- State C: resolved intermediate/mixed cases, including concentrated but
  off-resonant effects or distributed contributions with appreciable mixing.
  Explicitly state which of these produces C; do not imply a resonance.
- State D: reconstruction failure, singular unresolved elimination, or unsafe
  state tracking that prevents a subspace-resolved interpretation.
- Central-shell overall: A if both states7/8 are A; B if both are B and logical
  states0/1 do not introduce a dominant near-resonant countereffect; C for a
  resolved combination or intermediate cases; D if exact validation fails or
  central attribution cannot be resolved. Supporting states2/3 remain visible.
  Report transitions as signed differences of state contributions, including
  cancellation; a state-level mechanism need not explain a transition without
  this subtraction.

## Operators, selectivity, and scope

Measure the full {0,1,7,8}, logical->{7,8}, and internal{7,8} grid_phi blocks:
Frobenius norms, singular values, individual edges, state overlaps, and
principal angles. Split each adjacent operator change into PP relaxation and
PQ+QP+QQ direct shell terms after phase alignment; verify their sum.
Reuse frozen convergence classifications; no new operator tolerance is fitted.

Report f01, f78, f78-f01 for every prefix, g01/g78, and all logical-to-states
0..10 transitions near f07/f81. List nearest five unwanted pairs per carrier
without suppressing small/forbidden matrix elements; include their coupling
magnitudes so spectral proximity is not mistaken for appreciable excitation.
No pulse optimization or driven simulation is implied.

Only fixed-N71, fixed-four-mode, retained Stage-1 attribution is addressed.
k260 is an endpoint, not ground truth. Spatial/domain/n1max/N4 convergence,
omitted modes, physical ports, and X90/X180 validation remain unestablished.
