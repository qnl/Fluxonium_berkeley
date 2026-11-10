**Gridium state and subspace tracking — completed sandbox study**

**PASS:** all 19 numerical-method tests pass. The saved Gridium results support stable low-energy labels in the tested cases and provide a genuine ambiguous high-energy pair. **They do not establish operator convergence for that pair:** N71-priority k170 states 45/46 have mean subspace overlap 0.9833 with k180, yet their aligned `grid_phi` coupling block to states 0–10 changes by **22.1%** in Frobenius norm.

The resumed work reused every saved eigensystem. It performed **zero new Gridium eigensolves**, no optimization, and no production edits, commit, or push. The earlier completed validation used 11 small Stage-2 reconstructions, at most 1440×1440, from authenticated saved Stage-1 data; those calculations were not repeated. The synthetic example below uses only 4×4 matrices.

**What is being tracked.** These questions are different:

| Question | Appropriate object | When a label is meaningful |
|---|---|---|
| Which new eigenstate corresponds to one old eigenstate? | One-to-one assignment maximizing squared overlaps | Assigned overlap dominates alternatives and remains stable as the sweep step/window changes |
| Does a nearly degenerate cluster persist? | Projector onto its span; principal angles | The whole cluster is captured, even if individual eigenvectors rotate |
| Where did a specified physical character go? | Observable/projector weights, or an observable-defined frame inside the cluster | The defining observable resolves that character; these frame vectors need not be energy eigenstates |
| What happens when ordering changes? | Store both physical ancestry and current energy indices | Energy rank is metadata, never the matching criterion |

A smooth adiabatic eigenstate and a descendant defined by fixed physical character are not necessarily the same object through an avoided crossing. Abandon unique individual ancestry when overlaps compete, rather than forcing a scientifically certain label from the Hungarian permutation.

**Reusable module and audit.** [tracking.py](tracking.py) supplies `track_states`, `track_subspaces`, `ambiguity_clusters`, `compare_blocks`, `block_invariants`, `observable_frame`, `embed_retained`, and `pathway_signature`. Inputs are **column vectors in an explicitly common physical basis**. There is no implicit padding or truncation to make incompatible vectors appear comparable.

For individual matching it normalizes columns without mixing states, computes S=U†V and Oij=|Sij|², and implements both global-largest-available-edge greedy matching and Hungarian maximum-total-probability assignment. Outputs include assigned probability, best alternative, probability and amplitude margins, column competition, energy displacement, ordering changes, and ambiguity reasons. The saved field `second_best_probability` means the best alternative **excluding the assigned candidate**; it may exceed the assigned score when a global assignment must sacrifice a row's preferred match. Rectangular windows can contain unmatched references, and global energy indices must be supplied to compare unequal windows' ordering.

The code audit confirmed the squared-overlap objective, complex matrix conventions, tensor embedding, SVD orientation, and separate left/right operator rotations. It fixed six edge cases: custom policy inheritance; unassigned/column competitors; misleading singleton “energy watch” classification; certification of a nonoptimal greedy assignment; insufficiently strict unitarity checking; and stale global-index flags in the saved Stage-1 shell metadata. The last correction changed flags only for shell rows corresponding to states 171, 177, and 178, whose global indices did not change. No overlaps, energies, or assignments were recalculated. Details: [code_audit.json](code_audit.json).

**Subspace and operator definitions.** Spatial interpolation is generally not an isometry. The module reports the Gram matrix distortion and uses symmetric whitening Q=U(U†U)^−1/2 for subspace calculations, rejecting deficient frames. It does not silently apply this mixing to individual eigenstate matching. The 10⁻¹² relative Gram-eigenvalue cutoff is a numerical rank safeguard, not a physical accuracy criterion.

For orthonormal frames QU,QV, singular values σ of QU†QV are cosines of principal angles. With ranks r,s,

\[
\|P_U-P_V\|_F^2=r+s-2\sum_j\sigma_j^2.
\]

For equal ranks the operator norm is sin θmax; for unequal ranks it is one. The chordal distance is the Frobenius distance divided by √2. Mean overlap is Σσ²/r for equal ranks; unequal-rank reports give both directional coverages and the symmetric value 2Σσ²/(r+s). A minimum singular value of one alone does **not** prove equality of unequal-rank spaces. Small angles are evaluated with residuals to avoid loss of accuracy near σ=1.

If QU†QV=LΣR†, the Procrustes rotation W=RL† aligns QVW toward QU. An internal candidate block becomes W†AVW. A rectangular block between two clusters requires **both** endpoint rotations, WL†AVWR. Singular values, Frobenius norm, and operator norm are invariant under these basis rotations; internal Hermitian blocks additionally have invariant eigenvalues, trace, and trace of the square. Aligned differences use the geometric correspondence and are reported separately. Rank-deficient cross-overlap makes the alignment nonunique and must not restore labels. Definitions agree with the standard [principal-angle](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html) and [unitary Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html) constructions, against which the tests check representative cases.

**Real ambiguous pair: N71-priority k170 → k180.** This is the saved `N71_priority_k170` retained basis, distinct from the older `asymmetric_priority_k170` selection. Hungarian matching proposes 45→46 and 46→45; both identities are flagged ambiguous.

| Reference → candidate | Assigned probability | Best alternative probability | Probability margin | Ground-referenced energy displacement |
|---|---:|---:|---:|---:|
| 45 → 46 | 0.539224 | 0.441147 | 0.098077 | −7.02025 MHz |
| 46 → 45 | 0.546129 | 0.440124 | 0.106005 | −100.77829 MHz |

For the pair as a whole, singular values are **0.993992925, 0.989243350**, principal angles **6.28329°, 8.41136°**, mean overlap **0.983312170**, projector Frobenius distance **0.258362769**, and projector operator distance **0.146279169**. The worst-direction retained probability is 0.9786024. This supports treating the pair collectively; it does not mean its projector is identical or its couplings have converged.

The following are **signed** `grid_phi` blocks with the pair on the left and the stated tracked target space on the right. Reference is k170, candidate is k180. Rotations and whitening are applied consistently at both endpoints. Units are the stored `grid_phi` units; relative differences use the k170 Frobenius norm.

| Pair → target block | Singular values, k170 → k180 | Frobenius norm, k170 → k180 | Operator norm, k170 → k180 | Aligned ΔF / Δop | Relative ΔF |
|---|---|---|---|---|---:|
| Logical `{0,1}` | (0.354907, ≈0) → (0.310592, ≈0) | 0.354907 → 0.310592 | 0.354907 → 0.310592 | 0.0451943 / 0.0451943 | **12.73%** |
| States 0–10 | (0.383841, 0.0885916) → (0.351336, 0.0509533) | 0.393932 → 0.355011 | 0.383841 → 0.351336 | 0.0871543 / 0.0853290 | **22.12%** |
| Other 46 stored states | (1.69403, 0.712758) → (1.83679, 0.682707) | 1.83787 → 1.95957 | 1.69403 → 1.83679 | 0.254905 / 0.232208 | **13.87%** |

The internal 45/46 block is at numerical scale: its Frobenius norm is 1.07×10⁻¹¹ → 1.86×10⁻¹². Relative errors there are meaningless; its smallness says nothing about convergence of outgoing blocks. “Other 46” means the complement **inside the saved 48-state window**, not all physical states.

For the low-11 block, matching individual phases alone gives ΔF=0.272353; allowing the cluster rotation reduces it to 0.0871543. Thus a substantial part of the apparent matrix-entry change is basis rotation, but a material change remains. Its Frobenius norm decreases 9.88%, and its smaller singular value decreases about 42.5%; no phase convention can remove these invariant changes. **Operator convergence and control-pathway convergence are not established.** See [complete block diagnostics](pair45_46_operator_analysis.json) and [CSV table](pair45_46_operator_blocks.csv).

**Stable low-energy controls and states 7/8.** The saved validation covered nine retained-basis selections and N51→N71 spatial refinement. These results were reused, not recomputed. Representative assigned probabilities are:

| Comparison → N71 k180 | P0 | P1 | P7 | P8 | Mean overlap of 7/8 span |
|---|---:|---:|---:|---:|---:|
| N51 k180, spatial embedding | 0.999588 | 0.999596 | 0.999186 | 0.998136 | 0.998661 |
| N71-priority k170 | 0.999998 | 0.999997 | 0.999966 | 0.999951 | 0.999959 |
| N71-priority k172 | 0.999999 | 0.999999 | 0.999980 | 0.999979 | 0.999979 |
| N71-priority k175 | 1.000000 | 1.000000 | 0.999990 | 0.999987 | 0.999989 |
| N71-priority k177 | 1.000000 | 1.000000 | 0.999995 | 0.999996 | 0.999995 |
| N71-priority k179 | 1.000000 | 1.000000 | 0.999999 | 0.999999 | 0.999999 |

None of these 0/1/7/8 labels is flagged ambiguous by the documented policy. The older asymmetric-priority k170/k172 selections are less accurate—minimum low-11 probability about 0.9678—but their low-state assignments still do not show the strong competition of the 45/46 example. Closeness of energies alone does not make 7/8 ambiguous in the evidence actually available.

Operator accuracy is reported independently. For N71-priority k170→k180, aligned relative differences are 0.2161% for the internal 7/8 block and 0.0480% for its block to `{0,1}`; at k179 they decrease to 0.00292% and 0.00218%. Spatial N51→N71 gives 2.51% and 1.16% in the **common target-grid, whitened comparison**. Those spatial values include interpolation effects and must not be confused with a native-eigenbasis comparison of checkpoint magnitudes. No gate-fidelity criterion is inferred. [Low-energy summary](low_energy_summary.csv), [all assignments](state_assignments.csv), [subspaces](subspaces.csv), and [operator comparisons](operator_comparisons.csv).

The independently reconstructed spatial overlap amplitudes had matched the saved analysis to 2.47×10⁻¹³. Interpolated column norms range from 0.97284 to 0.98893; even after individual normalization, the low-11 Gram error is 0.02892. This is why principal angles cannot safely be computed by simply taking an SVD of normalized-column overlaps without orthonormalizing the subspaces. A further saved Stage-1 shell example has forced poor matches, including reference 179→candidate 111 with probability about 0.00186. Its assigned 20-state span has σmin≈0.04508: grouping does not rescue missing descendants. Enlarge the candidate window or refine the model/step. No new Stage-1 solution was attempted.

**Synthetic avoided crossing: software validation only.** [synthetic_crossing.py](synthetic_crossing.py) uses

\[
H(x)=\operatorname{diag}(-3,x,-x,3)+0.08(|1\rangle\langle2|+|2\rangle\langle1|),
\]

with dimensionless parameters and a fixed complex Hermitian test operator on the central pair. The energy-ordered branches do not cross; their physical character exchanges. The energy index carrying the largest bare-|1⟩ character changes from 1 to 2. A coarse step from x=−0.08 to +0.08 has all four individual overlap probabilities 0.5, while the cluster's singular values remain one. Fine consecutive steps retain at least 0.996139 individual probability along adiabatic branches. Thus ambiguity depends on step size and on whether one is following an eigenstate or physical character.

An observable-defined bare descendant stays continuous but is not an energy eigenvector near the crossing. Procrustes-aligned test-operator error is below 9.81×10⁻¹⁶, and descendant projector error below 4.72×10⁻¹⁶. The cluster span is fixed by construction; its zero principal angles validate the implementation, not a new Gridium phenomenon. [Figure](synthetic_avoided_crossing.png), [sweep table](synthetic_sweep.csv), [checks](synthetic_summary.json).

**Conservative ambiguity criteria.** The default policy preserves the existing Gridium audit's **amplitude** cutoffs: assigned amplitude ≥0.9; amplitude margin ≥0.25; competitor/assigned amplitude <0.5. In probability form these become P≥0.81 and competitor ratio <0.25, while the margin must remain √Passigned−√Pother≥0.25. Squaring the margin itself is incorrect. Row and column competition are both checked. These are transparent warning thresholds inherited from prior work, not universal scientific confidence levels or convergence tolerances.

Recommended use:

1. Retain individual labels only when overlap, row/column margin, window coverage, and step-size stability all support them. An assignment change under refinement triggers review, not an automatic physical identity swap.
2. Form competition-connected clusters and include competing unassigned candidates. If a cluster fails worst-direction coverage, enlarge the window or reduce the step; do not call it resolved because its *mean* overlap is high.
3. Treat an energy gap comparable to measured/estimated step shifts as a joint-monitoring warning. Supply those shift scales explicitly, after removing a common energy offset. A shift is not automatically an error bound. Energy crowding alone does not overrule strong overlap evidence.
4. Separate a **tracking threshold** from the desired numerical accuracy. The 45/46 pair passes worst-direction probability thresholds 0.81 and 0.95, but fails 0.99. Its operator blocks must still meet separately chosen absolute/relative tolerances. Relative comparisons are suppressed at numerical-zero norms.
5. If overlap-score uncertainty is bounded by ε, the best/second-best assignment-score gap exceeding 2mε certifies that matching against such perturbations, where m is the matched count. Without a justified ε this is a diagnostic gap, not a confidence probability.
6. Restore individual labels only after competing weights separate, the observable definition is resolved, and correspondence persists under a smaller step and a larger window. Require several consecutive checks as a declared workflow policy; no universal count is inferred from these data.

**Workflow for subsequent sweeps.**

| Sweep | Explicit embedding and procedure | Required checks |
|---|---|---|
| Retained k | Embed `(retained parent, oscillator)` using actual retained indices into the common parent×oscillator basis; never pad a prefix by guesswork | Hungarian map, margins, reordered indices, cluster angles, internal/outgoing operator blocks, omitted-window capture |
| N51/N71 spatial | Reconstruct saved states in `(theta3,theta2,n1,N4)`; interpolate wavefunction amplitudes with the √quadrature-weight conversion on identical domains; normalize only for state matching, whiten for subspaces | Interpolation norms/Gram error, physical-coordinate operator convention, both endpoint rotations, forward/refined-grid consistency when data permit |
| Parameter sweep | Track both consecutive points and a fixed reference; record their different ancestry questions; adapt step size at crowding/competition | Candidate guard states, energy and observable continuity, agreement after step halving, possible path-dependent gauge/holonomy rather than a presumed global frame |
| Array-mode enlargement | Establish the new physical tensor basis and an explicit embedding of old states tensored with array states; use the actual enlarged Hamiltonian and actual control operator | Include mixed array excitations, external cluster gap, projector capture, array participation, and operator-block convergence before restoring labels |

For every case, archive energies, vectors/embedding, complex overlaps, assignments and alternatives, principal angles, Gram diagnostics, rotations, operator invariants, policy parameters, and input hashes. Keep energy eigenvectors and the aligned cluster frame separately: a Procrustes-rotated frame generally has **R†diag(E)R**, not a diagonal Hamiltonian.

For a future cluster seeded by `|7,0a⟩`, `|8,0a⟩`, `|0,1a⟩`, `|1,1a⟩`, track the four-dimensional span when two-state windows lose weight to the array descendants. Expand beyond four if guard states compete. Report each energy eigenstate's array-number distribution, ⟨na⟩, and bare-product projector weights; inside an ambiguous frame, additionally report basis-invariant spectra and traces of Q†naQ and Q†Πna≥1Q. Their trace divided by cluster dimension is a cluster average, not a unique eigenstate occupation. Array number alone can leave twofold ambiguity; additional physically specified ancestry observables may help, but noncommuting or degenerate projected observables do not define unique simultaneous labels. Do not invent the new mode's Hamiltonian, coupling, or assume the control remains Φ⊗I without a supplied model.

**Pathway continuity.** Replace fixed-index `0→7→8→1` comparisons by a tracked logical frame L and intermediate cluster C. Report C†ΦL, C†ΦC, their singular values/norms, and aligned differences. The diagnostic word (C†ΦL)†(C†ΦC)(C†ΦL) is invariant under rotations inside C, whereas a selected three-entry product generally is not. With justified elimination, use full matrix resolvents of the aligned cluster Hamiltonian, transforming them consistently with its operator blocks; specify photon orderings and detunings before interpreting a result as a control rate. Continuity of C or of unweighted operator norms alone does not establish continuity of the driven pathway.

**Files, tests, and limitations.** New completion files are `complete_analysis.py`, `synthetic_crossing.py`, `verify_results.py`, this report, the 45/46 JSON/CSV diagnostics, low-energy summary, synthetic JSON/CSV/PNG, and audit/preservation records. Existing `tracking.py` and `test_tracking.py` were corrected/extended; two shell metadata outputs and test logs were updated. `validate_gridium.py` and all cached NPZ eigensystems were preserved during the resumed work.

The 19 tests cover complex phases/permutations, greedy failure, assignment-gap certification, rectangular windows, known principal angles and projector norms, unequal ranks, nonorthogonal/rank-deficient frames, complex Procrustes and rectangular blocks, embeddings, custom policies and missing competitors, observable degeneracy, invariant pathway words, explicit energy warnings, and the avoided crossing. See [tests.log](tests.log), [verification.json](verification.json).

```sh
# Tests and artifact checks only; no Gridium eigensolve.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/subspace_tracking/test_tracking.py
python research/sandbox/subspace_tracking/verify_results.py
```

`complete_analysis.py` reproduces the new operator diagnostics from cached inputs. `synthetic_crossing.py` reproduces the dimensionless unit example. **Do not rerun `validate_gridium.py` just to use these outputs:** it is the earlier reconstruction driver. Spatial interpolation error, finite stored windows, observable choices, and absent array-mode data remain limitations. No physical parity, pulse fidelity, or model completeness is inferred from this tracker.
