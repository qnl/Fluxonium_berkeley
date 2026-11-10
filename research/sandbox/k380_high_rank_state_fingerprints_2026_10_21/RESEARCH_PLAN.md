# Frozen high-rank state fingerprint diagnostics

Frozen 2026-10-21 before reading state metrics. This is a new saved-data diagnostic,
not a modification of any benchmark, selection, threshold, Hamiltonian or checkpoint.

## Sources and representation established by code inspection

Use only the completed epsilon_J=0.11 k380 parent. Authenticate COMPLETE, payload
hashes, canonical configuration, original production source pins, k380 sealed
manifest and linked diagnostic-review sources. Do not rerun historical audits.
Configuration fingerprint must be
`8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`.

Production `_assemble_nonlinear_sector.kron3(A,B,C)=C⊗B⊗A` proves the index
`i=((i3*71)+i2)*9+i1`. Reshape C order `(71,71,9)` = (theta3,theta2,n1).
This is independently documented by production `interpolate_raw_states`.
The user calls the spatial coordinates phi2/phi3; this report uses the actual
code's theta2/theta3, not paper-coordinate identifications.
Charge labels are integer n_theta1=-4,...,4, conjugate to compact theta1;
ng=0. They are not oscillator levels or the derived grid charge.
The node transformation in `_coeffs` is varphi=M theta; theta2=varphi2-varphi1,
theta3=varphi3-varphi2 and theta4=varphi4-varphi1 in node-row numbering.
Coordinates are dimensionless reduced phases (radians). theta2∈[-11,11],
theta3∈[-14,14], both inclusive; h2=22/70, h3=28/70.
Stage-1 is the three-coordinate sector, dimension 9*71*71=45369; N4=8 enters
only Stage-2. Mass matrix is I. Coefficients have Euclidean normalization.
For an optional continuous-density interpretation psi=c/sqrt(h2*h3), use equal
rectangle weights h2*h3 at every node, including endpoints, consistent with
production interpolation. No trapezoid reweighting is permitted.

## Fixed state selection and comparisons

Focal zero-based energy-sorted ranks: 328,354,362. Neighbors: focal±1,±2.
Deterministic controls: 300,310,320,330,340,350,360,370.
Calculate the same metrics for every rank 300..379 as the surrounding-band
background, so comparisons cannot be cherry-picked. Also calculate full-parent
norm/orthogonality and gaps from all 380 saved ranks. No data-dependent controls.
Report focal positions within the full 80-state band, local five-state windows,
and each 20-state interval [300,320),[320,340),[340,360),[360,380).

## Probability, participation and coordinate diagnostics

For each state s, retain original q_s=sum|v_s|²; p=|reshape(v_s)|²/q_s.
Charge marginal is sum over both spatial axes; spatial marginals sum over
the other spatial axis and charge. Save all nine/71/71 probabilities.
Charge edge = p_n(-4)+p_n(+4), with separate sides. Spatial edge(k) is the
sum of the first k and last k marginal entries for k=1,2; also save each side
and the union of either spatial boundary (no double counting corners).
These diagnose finite-basis support, not an infinite-basis error bound.
For joint coefficients and each marginal, IPR=sum p², PR=1/IPR and normalized
PR=PR/number of bins. Joint spatial PR uses charge-summed 71×71 density.
Continuous spatial IPR=sum p23²/(h2*h3), PR area=(h2*h3)/sum p23².
Compute means, second moments and standard deviations for n1,theta2,theta3;
spatial covariance and grid_phi=theta3+theta2/2 moments. No compact-angle
variance or wavefunction amplitude localization inference.

## Spectral and subspace diagnostics

Save E_s (H/h in GHz), gaps below/above, residuals, gap/(res_s+res_neighbor).
Frozen descriptive close-cluster criterion: adjacent gaps <=0.001 GHz (1 MHz),
transitive on all 380 ranks. A numerical ambiguity flag also tests gap <=
10*(res_s+res_neighbor). These are descriptive criteria, not convergence gates.
For every close cluster intersecting a focal rank and every fixed focal±2
window, compute ensemble density sum_s |v_s|²/m and corresponding marginals,
edges and coordinate moments. This is invariant under unitary rotations of
the complete chosen subspace; single-vector PR is not invariant. Report external
gaps of windows. A single parent cannot establish cross-grid eigenstate stability.

## Operators and denominators

Read saved 380×380 projections n1,n2,n3,x2,x3 = V† O V. Hermiticity check
max|O−O†| <=1e-10. Independently reproduce focal/neighbor/control columns against
all saved vectors using local diagonal or 71×71 finite-difference actions;
do not build a full Stage-1 operator or Hamiltonian. Tolerance 1e-10 absolute.
Production Stage-2 cross-rank block is
H_pr = A_pr*n4 + B_pr*theta4,
A=8*sum_i EC[i,3]*n_i, B=K[0,2]*x2+K[1,2]*x3.
Use production `_coeffs` (only a 4×4 inverse), production mode-4 formulas and
all complex phases before summation. No eigensolve, inverse resolvent or dressed
vector construction. Save individual complex matrix elements <p|O|r>, A_pr,
B_pr and H_(p,0),(r,1), H_(p,1),(r,0) for p=0..39 and r=300..379, plus
8×8 block Frobenius strengths. Relative phases across operator terms in a
fixed p,r block are meaningful; absolute eigenvector phases are arbitrary.

For each high rank report summed squared block Frobenius coupling to fixed
prefixes p<11,40,100,300 and its own 20-rank-band lower prefix; strongest
partner rank in each prefix. Also compute the vacuum bare-state proxy
sum_{p<prefix}|H_(p,0),(r,1)|²/(E_r−E_p+w4[1]−w4[0]) in GHz.
The denominator uses saved Stage-1 energies plus analytic oscillator energies;
it is positive for these lower prefixes, not a dressed Schur denominator.
Save strengths/denominator ranges and rankings across all 80 ranks.
No bare proxy is labeled an exact state shift or exact Schur contraction.

## Existing Schur evidence

Read authenticated saved S1/S2 rank scalars (sum six-target absolute weights),
target-projected couplings, probabilities and denominator context. Compare all
40 shell ranks with structural metrics and bare coupling proxies using
descriptive Spearman correlations (fixed metrics: charge edge, theta2/3 two-node
edges, joint PR, coordinate spreads, p<300 block strength and vacuum proxy).
No p-value or causal claim. Rank328 historical attribution is a separate k340
parent, used only through authenticated review records and their saved identity
overlap; do not relabel it a new k380 Schur result. Exact dressed-vector-based
operator attribution and rotated-subspace response are INCOMPLETE if vectors
are absent; do not reconstruct them or rerun Schur.

## Validation and resource limits

Synthetic unequal-axis one-hot indexing; complex local-operator versus Kronecker
contraction; normalized marginals/constant quadrature; Hermitian complex bilinear
block and conjugation; ensemble invariance under a fixed complex unitary.
Norm/marginal/reshape tolerance 1e-10; orthogonality infinity norm <=1e-8 (original
quality rule). Recompute no eigenpair residual: authenticate saved residuals.
Repeat post-processing in a temporary output directory and require identical
numerical CSV/JSON hashes. Freeze this file's SHA-256 in every run.

Single-thread BLAS. Wall-time cap 180 seconds per analysis process, address-space
cap 2 GiB where supported and resident-memory cap <=1 GiB. On macOS, if
RLIMIT_AS rejects the limit, use a 50-ms resident-memory watchdog instead and
record the unavailable address-space limit. This platform fallback was added
after the initial process rejected RLIMIT_AS, before any state metrics ran.
Stop if exceeded; mark essential
unavailable work INCOMPLETE. Load the ~263 MiB vector array once, process one
state or one selected operator action at a time, no D×D arrays, Hamiltonians,
new spatial calculations or research eigensolves. Record measured resource use.
Preserve hashes of tracked files, Git index, notes, all checkpoints, existing
figures and v1/v2/preregistration artifacts. All new files stay in this directory.
No staging, commits or pushes. V2 PRELAUNCH, GATE WORK and SPATIAL CONVERGENCE HOLD.
