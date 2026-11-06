# Source-grounded first-array-mode extension for asymmetric Gridium

## Executive conclusion

A unique five-mode Hamiltonian cannot yet be derived for the current
asymmetric benchmark.  The paper supplies the distributed-array topology and
the first-mode dispersion, but it does not identify which physical inductor
array produces the reported mode, nor does it provide the normalized mode
shape or terminal participation needed to couple that mode to the current
four node coordinates.

The clean extension is therefore **not** an oscillator appended to the solved
four-mode Hamiltonian.  It is a circuit reduction: replace the relevant
lumped inductive branch by its `L_JA`-`C_JA`-`C_g` array network, transform that
network together with the existing nodes, and retain its first internal
normal mode.  The coupling then follows from off-diagonal blocks of the
enlarged capacitance and stiffness matrices.

It is scientifically justified now to implement only a parameterized matrix
reduction/scaffold.  It is not justified to instantiate a physical array mode
or publish pathway conclusions until the missing device/EM information is
provided.

## 1. Source-derived facts

1. Supplementary Fig. S10 models a Josephson-junction array with junction
   inductance `L_JA`, junction self-capacitance `C_JA`, and capacitance `C_g`
   from each array island to ground.
2. Its first internal mode is the `k=1` distributed plasma mode, with

   \[
   \omega_k=\omega_0\sqrt{
   \frac{1-\cos(\pi k/N)}
   {C_g/(2C_{JA})+1-\cos(\pi k/N)}} ,
   \qquad
   \omega_0=(L_{JA}C_{JA})^{-1/2}.
   \]

3. The supplement gives material/design estimates `C_JA approximately 45
   fF/um^2 times junction area`, `C_g approximately 30-40 aF`, and example
   critical-current densities from 300 to 750 nA/um^2.  These are examples,
   not the parameters of the present Table-S1-like benchmark.
4. The paper reports an array mode near 4.83 GHz for one protected device,
   expects 5.5-6.5 GHz for another device, and says later devices move array
   modes above 8 GHz.  These statements are device-specific and do not select
   a frequency for the current checkpoint.
5. The present implementation replaces physical arrays by three lumped
   inductive branches.  In `x=(theta2,theta3,theta4)`, their vectors are

   \[
   b_{K1}=(0,0,-1),\quad
   b_{K2}=(1,0,-1),\quad
   b_L=(1,1,-1),
   \]

   with stiffness contributions

   \[
   E_{LK1}b_{K1}b_{K1}^T+E_{LK2}b_{K2}b_{K2}^T+E_Lb_Lb_L^T.
   \]

6. The paper does not state which of these physical arrays, or which
   common/differential combination of them, produces the reported spectral
   array mode.

## 2. Physical degree of freedom

The new degree of freedom is an internal charge/phase wave along a junction
array.  It is distinct from the total endpoint phase across the array: the
latter is the low-frequency coordinate represented by the present lumped
inductance, whereas the `k=1` mode has nonuniform phase drops and oscillating
charges on internal islands.

For an array with node fluxes
`Phi=(Phi_0,...,Phi_N)`, define the edge-difference matrix `D` by
`(D Phi)_j=Phi_j-Phi_(j-1)`.  In physical-flux units the source-supported
quadratic array Lagrangian is

\[
T_{arr}=\frac12\dot\Phi^T C_{arr}\dot\Phi,
\quad
V_{arr}=\frac12\Phi^T K_{arr}\Phi,
\]

with

\[
C_{arr}=C_{JA}D^TD+C_gG,\qquad
K_{arr}=L_{JA}^{-1}D^TD.
\]

`G` places the ground capacitances on the internal array islands.  End-pad or
termination capacitances, if present in the device extraction, must be added
explicitly rather than absorbed into a guessed coupling.

## 3. Coupling to the current coordinates

Let the array endpoints be existing circuit nodes `r,s`.  The current node
phases obey `Phi_node=M theta`, with the same transformation `M` used by
`qchard_gridium_netlist.py`.  Resolve the array-node vector as

\[
\Phi_{arr}=P_{rs}M\theta+u_1q_a,
\]

where:

- `P_rs` interpolates the two physical endpoint phases across the chain;
- `u_1` is the normalized first internal mode with zero endpoint displacement;
- `q_a` is its amplitude.

Substitution gives the source-consistent reduced matrix blocks

\[
\widetilde C_{arr}=
\begin{pmatrix}
P^TC_{arr}P & P^TC_{arr}u_1\\
u_1^TC_{arr}P & u_1^TC_{arr}u_1
\end{pmatrix},
\]

\[
\widetilde K_{arr}=
\begin{pmatrix}
P^TK_{arr}P & P^TK_{arr}u_1\\
u_1^TK_{arr}P & u_1^TK_{arr}u_1
\end{pmatrix}.
\]

The off-diagonal blocks are the coupling.  They must be calculated from the
mode shape and endpoint embedding; they are not an independent free constant.

If the selected branch has present stiffness `E_b b b^T`, the enlarged
five-coordinate matrices have the schematic structure

\[
C^{(5)}=
\begin{pmatrix}
C^{(4)}+\Delta C_{\theta\theta} & c_{\theta a}\\
c_{\theta a}^T & C_{aa}
\end{pmatrix},
\]

\[
K^{(5)}=
\begin{pmatrix}
K^{(4)}-E_bb b^T+K_{\theta\theta}^{arr} & k_{\theta a}\\
k_{\theta a}^T & K_{aa}
\end{pmatrix}.
\]

The subtraction prevents double counting of the lumped inductor.  A correct
reduction must recover `E_b b b^T` in the static low-frequency limit.  The
array capacitance block is added before matrix inversion, because its
off-diagonal charge couplings follow from the inverse of the complete
capacitance matrix.

In the repository's canonical convention the proposed Hamiltonian is

\[
H_5=4\,\mathbf n^T E_C^{(5)}\mathbf n
+\frac12\mathbf x^T K^{(5)}\mathbf x
+U_J(\theta_1,\theta_2,\theta_3),
\qquad E_C^{(5)}\propto(C^{(5)})^{-1},
\]

where `x=(theta1,theta2,theta3,theta4,q_a)` and the existing Gridium
Josephson terms are unchanged.  At the level supported by Eq. S33, `q_a` is
harmonic.  Array-junction Kerr or coherent phase slips are not included
without a separate derivation.

### Candidate branch mappings from the current netlist

| If the mode belongs to | Existing branch coordinate | Matrix vector |
|---|---|---|
| first KITE inductor array | `-theta4` | `b_K1=(0,0,-1)` |
| second KITE inductor array | `theta2-theta4` | `b_K2=(1,0,-1)` |
| S/superinductor array | `theta2+theta3-theta4` | `b_L=(1,1,-1)` |

The supplement's design example discusses constructing `E_L`, which suggests
the S/superinductor branch is a serious candidate.  That is not enough to
identify the array mode observed in Supplementary Note 11; the physical mode
could also be a collective combination of multiple arrays.

### Reasonable inferences, not source facts

- The `E_L` branch should be tested first if no better device information is
  available, because that is the branch used in the supplement's explicit
  array-design example.  It must not be declared the observed-mode branch.
- A mode within a few tens of MHz of states 7/8 is likely to require
  subspace-level avoided-crossing analysis if its extracted endpoint
  participation is nonzero.
- Keeping `grid_phi` tensored with the array identity is the cleanest
  diagnostic convention, but it is not a physical port model.

## 4. What the source determines and what it does not

| Quantity | Status |
|---|---|
| Array topology `L_JA`, `C_JA`, `C_g` | Source determined |
| Harmonic first-mode dispersion Eq. S33 | Source determined |
| Current lumped branch vectors and node transform | Repository determined |
| Reported 4.83 GHz, 5.5-6.5 GHz, and above-8-GHz examples | Source determined but device-specific |
| Which branch produces the relevant mode | Unknown |
| `N`, `L_JA`, `C_JA`, `C_g` for the current checkpoint/device | Unknown |
| End-pad and termination capacitances | Unknown |
| Normalized mode shape and endpoint flux/charge participation | Unknown |
| Effective `C_aa` and `K_aa` separately | Unknown; frequency fixes only their ratio/product under a normalization |
| Off-diagonal charge and phase coupling blocks | Unknown until mode shape/embedding is supplied |
| Whether fitted `eC/eP/EC/ECS` already absorb part of the array capacitance | Unknown; double-counting risk |
| Array-mode Kerr/nonlinearity and linewidth | Unknown |
| Coupling of the physical microwave port to the array coordinate | Unknown |

## 5. Relationship to current mode 4

Current mode 4 is the S20 `phi` node coordinate made dynamical by finite
cross-circuit capacitance `C_p`.  In the present checkpoint its bare harmonic
frequency is approximately 15.845 GHz.  It describes motion of an existing
circuit node shared by the lumped inductive branches.

The array mode is an internal oscillation of junction-island phases inside an
inductor chain.  Mode 4 therefore does not contain this degree of freedom.
It can contain low-frequency renormalization from parameters fitted to the
real device, and the physical normal modes may mix node-4 and array motion,
but no internal array coordinate exists in the current Hilbert space.

## 6. Expected near-resonant hybridization

In the zero-coupling product basis, an array frequency near 6.3 GHz creates
at least the states

\[
|0,1_a\rangle\quad\text{near }6.3\ \text{GHz},
\qquad
|1,1_a\rangle\quad\text{near }6.3\ \text{GHz}+f_{01}.
\]

These lie close to the existing `|7,0_a>` and `|8,0_a>` states at 6.275 and
6.298 GHz.  The relevant avoided-crossing problem may therefore be a
four-state cluster, not a single two-level crossing.  Its gaps and selection
rules depend on `c_theta_a`, `k_theta_a`, and the branch symmetry.

Mathematical consequences if the coupling is nonzero:

- energy-index labels 7 and 8 will exchange character across crossings;
- array occupation and Gridium state character will be shared;
- `f78` may cease to represent the same pair of physical states;
- `grid_phi` edges can brighten, darken, or migrate to array-dressed states;
- new three-step pathways through array-like states may outrank all current
  candidates.

No avoided-crossing magnitude can be estimated without the missing
participation/coupling blocks.

## 7. Required state and subspace tracking

1. Define the decoupled reference by either zero extracted coupling or a
   source-supported far-detuned device point.
2. Embed each original four-mode state as `|j,0_a>` and track ordinary states
   by full Hilbert-space overlap only away from crossings.
3. Near a crossing, track the whole ambiguous cluster with spectral
   projectors, principal angles, and SVD/Procrustes alignment; do not force a
   one-to-one Hungarian assignment when singular values show ambiguity.
4. Record array occupation, overlap with every `|j,0_a>` reference, and
   branch-flux participation for each dressed state.
5. Use energy labels only for display.  Scientific labels must refer to
   descendants/subspaces, such as the `old-7/array` cluster.
6. Track the logical `{0,1}` subspace as a two-dimensional projector so its
   basis remains stable at the protected near-degeneracy.

## 8. Pathway comparison in the enlarged model

Initially embed the existing abstract control coordinate as

\[
G_5=(\theta_3+\theta_2/2)\otimes I_a.
\]

This isolates changes caused by eigenstate hybridization.  It must not be
called the physical drive operator unless Thomas supplies the port-mode
participation; a real line may couple directly to `q_a` as well.

For each dressed parameter point:

1. compare the paths through tracked descendants/subspaces, never raw indices;
2. retain `0->7->8->1`, `0->7->2->1`, and `0->3->8->1` as provenance labels;
3. compute all active three-step paths including new array-dressed states;
4. report each individual edge, carrier, crowding, array occupation, and path
   product rather than only a winning product;
5. compare path stability across the entire ambiguous subspace when a unique
   state assignment is impossible.

All three current candidates remain exposed: the first uses both 7 and 8, the
second still uses state 7, and the third still uses state 8.  The enlarged
model must decide which state has lower participation in the physical array
branch; current state character alone cannot decide that.

## 9. Future implementation and validation sequence

### Phase A: obtain a physical array reduction

1. Identify the physical array/branch and device corresponding to the mode.
2. Obtain either `(N,L_JA,C_JA,C_g,terminations)` or a linearized EM/black-box
   model giving the mode frequency and normalized node-flux eigenvector.
3. Build the full linear array matrices and verify Eq. S33 for the uniform
   limit.
4. Verify that eliminating internal coordinates recovers the present static
   `E_L` or `E_LK` branch without double counting fitted capacitances.
5. Reduce to the first mode and compare the reduced linear eigenfrequencies
   and terminal participation against the full array network.  Include the
   second array mode in this check before declaring a one-mode reduction safe.

### Phase B: numerical convergence

- Array oscillator Fock cutoff: `N_a=2,4,6,8`; accept only after energies,
  avoided gaps, overlaps, and path elements converge.
- Existing fourth-mode cutoff: repeat the current `N4=8` result with larger
  cutoffs because mode 4 and the array mode may hybridize.
- Retained Stage-1 basis: same-grid `k=180,200,220`, with k220 as the parent.
- Spatial basis: repeat the established N51-to-N71 comparison after modifying
  the physical matrices.
- Number of stored final levels: increase beyond 11 so every array sideband
  and crowding competitor in the control band is present.
- Array-mode-count convergence: compare one retained internal mode against at
  least two; this is different from increasing the Fock cutoff of one mode.

If the array reduction changes the existing capacitance/stiffness blocks, the
modified Stage-1 Hamiltonian must be rebuilt.  A cheap Stage-2-only appended
oscillator can be used only as a mathematical diagnostic, not the validated
physical result.

### Phase C: source-supported frequency cases

Use the actual extracted device frequency if supplied.  For a documented
sensitivity comparison only, the paper supports examining the reported
4.83-GHz case, the reported/anticipated 5.5-6.5-GHz band including the
7/8 crossings, and a device-specific above-8-GHz detached case.  Do not assign
or sweep a coupling magnitude independently of the extracted mode
participation.

### Required observables

- full energies and avoided-crossing gaps;
- principal angles and singular values for tracked ambiguous subspaces;
- overlaps with `|j,0_a>` and array occupation;
- branch-flux and charge participation;
- `f07`, `f08`, `f78`, and every carrier in the three candidate paths;
- every individual `grid_phi` edge and path product;
- direct logical coupling, low-11-plus-sideband spectral crowding, and pathway
  ordering;
- mode-4 and array cutoff errors, k180/200/220 errors, N51/N71 changes, and
  one-versus-two-array-mode changes.

## 10. Questions we need Thomas to answer before implementing the array mode

1. Which physical array produces the relevant mode: the `E_L` S-branch
   superinductor, either `E_LK` KITE array, or a collective combination?
2. Which reported device should the current benchmark represent, and should
   we use the 4.83-GHz observation, the 5.5-6.5-GHz expectation, or a later
   above-8-GHz device extraction?
3. For each relevant array, what are `N`, junction area, critical-current
   density or `L_JA`, junction capacitance `C_JA`, island-to-ground `C_g`, and
   end/termination capacitances?
4. Can you provide the EM/black-box eigenmode frequency and normalized node
   flux/charge eigenvector, or equivalently the terminal participation of the
   first array mode?
5. Can you provide the Maxwell capacitance matrix or port admittance around
   the array resonance so we can avoid double counting `eC`, `eP`, `EC`, and
   `ECS`?
6. If multiple arrays participate, what is the relative phase and amplitude
   of their motion in the observed mode?
7. Is there a measured or simulated avoided-crossing gap/sideband strength
   that constrains the coupling?
8. Should the first implementation remain harmonic as in Eq. S33, or is an
   array-mode Kerr/nonlinearity required?
9. Does the microwave control port couple directly to the array mode, and if
   so, what is its port participation relative to the existing `grid_phi`
   coordinate?
10. What overlap/subspace and convergence tolerances should qualify the
    enlarged model for pathway decisions?

## Exact limitations

- The paper's frequency examples cannot be transferred uniquely to the
  current regime-A-like checkpoint.
- A mode frequency alone does not determine effective capacitance,
  inductance, zero-point amplitude, or coupling.
- The branch identity and normalized mode shape are absent.
- The current combined four-mode-plus-asymmetry model already has a documented
  source-boundary ambiguity; adding an array mode increases that need for an
  explicit reviewed circuit derivation.
- No static checkpoint calculation can predict the avoided crossings because
  the omitted coordinate and its coupling operator are absent.
- A production implementation made before the questions above are answered
  would necessarily contain invented physics.
