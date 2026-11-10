## Work Block — 2026-09-07

Completed: Set up the Gridium research workflow and established the division between research planning/interpretation and repository implementation. Validated the IdealGridium Python/QuTiP environment and grounded the working notes in the Gridium paper and Thomas's guidance. Audited the existing `Rabi_3Photon` implementation and identified that the intended three-photon control mechanism, pathway, tone structure, and drive operator still needed to be pinned down before making substantive changes.

Next: Clarify the intended three-photon control scheme with Thomas, then turn those answers into the first bounded and reproducible control-validation task.

### Questions for Thomas

- Is the intended three-photon logical gate driven through flux modulation rather than charge modulation?
- Should the control machinery support three independently tunable drive frequencies from the beginning?
- Is there a preferred intermediate-state pathway that should be treated as the initial reference implementation?


## Work Block — 2026-09-08

Completed: Characterized the candidate three-photon pathways and the protected-point selection rules of the symmetric IdealGridium model. Identified important degeneracies and showed that several superficially similar interpretations of the control scheme lead to materially different implementations and matrix elements. This made it clear that the control physics should be resolved before optimizing a gate numerically.

Next: Use Thomas's clarification of the physical drive, pathway, and symmetry-breaking intent to define the first validated three-tone control experiment.

### Questions for Thomas

- Which physical flux coordinate/operator is intended to drive the logical gate?
- Is the first three-tone pathway intended to connect the logical states through the 0/5/4/1 manifold, or should a different intermediate manifold be used?
- Should perfect symmetry be treated only as a reference point, with slight fabrication asymmetry introduced intentionally for control?


## Work Block — 2026-09-15

Completed: Built and validated the three-tone flux-drive control workflow in symmetric IdealGridium and obtained a converged X180 candidate with 98.67% logical fidelity, 0.47% final leakage, approximately 0.50% peak leakage, and a 21.5 ns gate time. Tightened the time-domain solver enough to separate numerical error from coherent gate error and found that the remaining infidelity is primarily coherent axis/angle error rather than leakage. The result established a reliable symmetric baseline for testing X90 control and eventual symmetry breaking.

Next: Use the validated symmetric X180 as the control baseline, construct an X90 gate with the same methodology, and then determine how slight symmetry breaking affects fidelity, matrix elements, and leakage before moving to the full four-mode model.

### Questions for Thomas

- Can you confirm that `0 <-> 5`, `5 <-> 4`, `4 <-> 1` is the intended first three-flux-tone pathway?
- For the first symmetry-breaking study, which physical asymmetry should we vary first, and is there a preferred nominal magnitude or fabrication-relevant range?


## Work Block — 2026-09-16

Completed: Calibrated and validated a symmetric IdealGridium X90 candidate with 99.23% logical fidelity, 0.061% final leakage, and 0.198% peak leakage, and added robust eigenstate tracking for near-degenerate manifolds. Tested a simple external-flux offset as a symmetry-breaking proxy and found that it rotates/mixes the logical doublet rather than cleanly representing the fabrication asymmetry we want to study. Implemented and diagnosed the symmetric three-mode S23 model, finding that its fully extended representation has a common-coordinate/topology problem rather than a simple cutoff problem, then merged the validated work into `four-mode-asym-gridium-tests` as requested. Audited the shared four-mode code and identified `qchard_gridium_netlist.Gridium4Mode` as the authoritative candidate model; the initial bounded validation suite gave 13 passed and 5 intentionally skipped production-scale checks, while `N4=6 -> 8` was already stable and Stage-1 retained-state convergence remained the main numerical issue.

Next: Validate Stage-1 retention in the netlist-derived four-mode model before attempting spatial-basis convergence or porting the X90/X180 control workflow. Keep four-mode gate optimization on hold until the physical flux-drive operator is established.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-17

Completed: Continued validation of `Gridium4Mode` and isolated the Stage-1 shift-invert eigensolver as the dominant numerical bottleneck rather than Hamiltonian construction. Instrumentation showed that sparse LU factorization and fill-in dominated runtime; the default COLAMD ordering produced about 73.1x fill, while `MMD_AT_PLUS_A` reduced this to about 47.8x and substantially reduced memory and inverse-solve cost. Verified that the new ordering preserves the same Hamiltonian physics: at `k=100`, Stage-1 eigenvalues agreed to approximately `5.3e-13 GHz`, downstream energies agreed within approximately `0.000441 MHz`, and eigenpair/orthogonality residuals remained near numerical precision. A bounded `k=120` study still showed retained-space convergence failures, and a successful `k=140` solve was checkpointed for later state-overlap, operator, and cutoff-boundary analysis rather than rerunning expensive calculations.

Next: Resume only from the saved `k=140` artifacts, determine whether the retained-state error is ordinary slow Rayleigh-Ritz convergence or a more structured coupling problem, and use that evidence to decide whether a bounded `k=160` extension is scientifically justified.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-18

Completed: Extended the four-mode retained-state study through `k=180` and found that energy-ranked Stage-1 truncation can miss physically important higher-energy states: the `170 -> 180` step shifted `f06` by about 57.53 MHz even though the logical doublets remained well tracked. Post-processing traced the effect primarily to a small cluster around Stage-1 states 172/174/175, including a strong fourth-mode-assisted near resonance involving state 172, and showed that the dominant shell coupling is carried mainly through the `x2/x3` sector. A separate local Gridium sandbox then tested blind residual-, coupling-, channel-, QoI-, and operator-response enrichment methods using only the saved `k=180` parent data; the best blind constructions matched or beat the diagnostic hand basis spectrally at the same retained dimension, reaching about 2.81 MHz maximum first-six transition error and 0.363 MHz `f06` error, while operator-response/Krylov enrichment improved `d_theta` from roughly 20.6 MHz/rad for ordinary energy truncation to about 4.3 MHz/rad but still did not match the hand basis. Also productionized the validated optional `MMD_AT_PLUS_A` solver path and expanded the bounded physical-validation suite to 21 passed and 5 intentionally skipped tests covering flux periodicity, finite-cutoff charge-periodicity behavior, Hermiticity, and state-resolved `d_phi`/`d_theta` periodicity, with no physical inconsistency found.

Next: Stop brute-force energy-only `k > 180` growth and treat retained-subspace construction as the remaining numerical problem, especially blind representation of `d_theta` response. Once that issue is sufficiently controlled, resume `n1max/N2/N3` spatial-basis convergence and independent operator/model validation, then port the validated three-tone X90/X180 workflow to the asymmetric four-mode model.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-25–2026-09-26

Completed: Incorporated Thomas's clarification that the current abstract four-mode control coordinate is `Gridium4Mode.grid_phi()`, with `phi_grid = theta3 + theta2/2`; `d_theta()` and `d_phi()` are external-flux response/sensitivity operators rather than the default microwave-drive proxy, so poor `d_theta` convergence is not a blocker for current gate work. Reproduced the frozen one-mode gates without reoptimization. X90 has duration `39.8007574 ns`, gate fidelity `0.9923100794`, process fidelity `0.9884651191`, final leakage `0.061239%`, and peak leakage `0.197513%`. X180 has duration `21.4963121 ns`, gate fidelity `0.9866983593`, process fidelity `0.9800475389`, final leakage `0.471441%`, and peak leakage `0.498001%`. Added permanent state-population, leakage, drive-trace, intermediate-population, convergence, and compiled CSV artifacts for these frozen validations; convergence plots now use gate infidelity `1-F_gate` rather than misleading offset notation.

---
Thomas Response:

- What do you mean by frozen one-mode gates?

---

Generalized the multitone workflow so `IdealGridium` uses `phi()` and `Gridium4Mode` uses `grid_phi()`, with every tone coupled through the same full operator and the existing per-target matrix-element normalization preserved. A reduced-basis four-mode three-tone smoke test demonstrated working `grid_phi` propagation; this was strictly a control-plumbing diagnostic, not a gate claim and not convergence evidence.

Independent review caught two serious problems in the initial retained-basis analysis: a symmetric Sep-18 k180 checkpoint had accidentally been reused as an asymmetric N=51 reference, and noncontiguous shell-state overlap embedding was incorrect. The affected numerical conclusions were withdrawn before finalizing the science. State tracking was corrected so noncontiguous retained coordinates are embedded at their actual Stage-1 ranks and overlap assignments consistently reorder eigenvectors, energies, transitions, tensors, and `grid_phi`; regression tests cover both noncontiguous embedding and reordered states.

Generated a fresh asymmetric N=51 k180 reference with `EJ=5`, `EC=0.5`, `EL=1`, `ELK=1`, `EJS=4`, `ECS=8`, `eC=5.5`, `eP=10`, `eps_J=0.10`, `eps_LK=0.05`, `ng=0`, `phi_ext=0`, `theta_ext=pi`, `n1max=4`, `N2=N3=51`, `L2=11`, `L3=14`, `N4=8`, Stage-1 `k=180`, `sigma=-24`, `which=LM`, `tol=1e-8`, and `MMD_AT_PLUS_A`. Its Stage-1 dimension is `23,409`, matrix nnz is `599,625`, LU nnz is `28,667,572`, fill ratio is `47.809x`, factorization time is `66.346 s`, ARPACK time is `170.118 s`, and the solve used `736` inverse solves. The maximum eigenpair residual is `5.433e-12`, median residual is `8.512e-13`, and orthogonality residual is `3.247e-12`. This is now the trustworthy asymmetric N=51 oracle.

---

Thomas Response:

- Just so we are on the same page, can you create a full explaination of what each variable here corresponds to in the model?

---

The inherited symmetric shell-informed k170 basis failed for the asymmetric model, with about `109.9 MHz` maximum transition error, about `13.22%` error on the important `7->8` `grid_phi` edge, and material leakage-coupling errors. An asymmetry-specific shell audit identified Stage-1 state 170 as the dominant omitted contribution: removing it caused about `105.369 MHz` transition error and up to `0.144713` absolute primary-path matrix-element error. The selected asymmetric basis is `0..159 + {162,163,165,167,168,170,171,172,176,177}`. Against full asymmetric k180 at N=51, it gives `3.055 MHz` maximum transition error, `0.999938` minimum tracked overlap, `0.005236` absolute / `0.452%` maximum primary-path `grid_phi` error, and `0.009153` absolute / `2.13%` maximum audited logical/leakage `grid_phi` error. The leading pathway remains `0 -> 7 -> 8 -> 1`, with full-k180 primary edges `0->7 = 1.911695`, `7->8 = 1.157822`, and `8->1 = 1.815039`. This k170 basis is controlled enough for the next spatial-convergence calculation at N=71; it does not establish spatial convergence and does not make the system ready for X90/X180 optimization.

---

Thomas Response:

- Why are we now looking at the 0->7->8->1 transition path? This is the first I am hearing of this. I am sure the cross parity manifold coupling is larger, but generally speaking it is harder to trust that these higher energy states are valid given various spurious capacitances in the physical circuit that are not captured by the model, let alone convergence issues that may arise in the existing model. Where can I find the convergence tests for states 7 and 8 in these models?

---

Solver/resource work confirmed `MMD_AT_PLUS_A` as the best tested SuperLU ordering. A local N=71 k180 solve was not practical on the current machine, and N=101 was rejected by resource preflight. Created and independently reviewed a portable external N=71 runner with explicit `--dry-run`/`--execute` modes, frozen physical/numerical settings, cryptographic pinning of the verified N=51 oracle, atomic incomplete/complete checkpoints, output hashing and checkpoint authentication, same-solve k170/k172/k175 analysis, overlap-based N=51-to-N=71 tracking, near-degenerate-subspace tracking, assignment-confidence diagnostics, SVD/principal-angle diagnostics, basis-invariant `grid_phi` subspace metrics, and `resolved`/`ambiguous`/`deferred` pathway claim gating. The runner is `READY FOR EXTERNAL N=71`. The durable scientific checkpoints are `956869f` (`Clarify Gridium control semantics and add gate validation artifacts`) and `f9fdd829d780d9d88f9919af4f2b3891e5bc5782` (`Validate asymmetric four-mode grid_phi workflow`); both are on `upstream/four-mode-asym-gridium-tests`.

Next: Run the frozen asymmetric N=71 k180 reference on a higher-memory machine using the reviewed runner, compare N=51 to N=71 with overlap/subspace tracking, and validate the asymmetry-specific retained basis there. Do not begin four-mode X90/X180 optimization until spatial convergence is understood.

---

Thomas Response:

- Why are you choosing specifically N=71 and N=51? Moreover, can we just treat one very long k run (let's say k on the order of 200+) as the ground truth, save the diagonalized system up to whichever levels nicely converge at that point, and avoid all the quibbling over specific cuttoffs?

---

### Questions for Thomas

- Does the asymmetry-specific k170 basis with full k180 as the reference look reasonable for the N=71 spatial-convergence check?
- Is there a preferred higher-memory machine or cluster for the N=71 run?

---

Thomas Response:

1. Where can I find these outputs?
2. Yes. I can run simulations on machines which have up to ~1Tb of RAM if need be. This is through Lawrence Berkeley National Lab's Lawrencium cluster.

---

## Work Block — 2026-09-27

Completed: Successfully validated the asymmetric four-mode `N=71, k=180` reference. N=51→N=71 state tracking remained clean, and the dominant `grid_phi` pathway stayed `0 -> 7 -> 8 -> 1`. The old raw-index k170 basis failed because important shell states moved in rank, but a common physics-informed k172 basis works well at both N=51 and N=71. The first bounded four-mode X90 test showed that the intended pathway is activated, but the inherited pulse is not a viable gate: the full three-tone pulse gave `F_gate≈0.366` with \~28% leakage trapped in states 7 and 8, while the middle tone alone produced substantial logical rotation with negligible leakage. The main problem is now pulse-area, phase, and interference control rather than spatial convergence.

Next: Build a driven four-state `{0,1,7,8}` model, compare it with the full propagation, run tone ablations, and determine how the direct logical drive interferes with the mediated pathway before any broad optimization.

### Questions for Thomas

- Does the common physics-informed k172 basis look reasonable for the next control studies?
- Should we continue treating the three-tone mediated pathway as primary, or intentionally exploit the direct logical coupling from the middle tone?

---

Thomas Response:

1. Sorry, where can I find this?
2. Let's keep down this pathway for a short while longer. Can you explain what changed between your early reporting of extremeley high fidelity gates and your reporting now of poor fidelity?

Other questions:

- What do you mean in some figures by "smoke traces"
- Are you using this k=180 and N=51 simulation for referencing errors in energy levels in the histogram plots? Are you doing that because of something I said, or is there good reason to believe things have converged by that cutoff?
- Why does the x90_state_populations.png graph look so much worse than the x180_state_populations.png graph? Moreover, for the reported fidelity, can you take it to be the peak fidelity rather than the final fidelity? We can and should always cutoff our simulation at whichever time point reports the highest fidelity, which in this case, appears to be sooner than the last time step.

---

## Work Block — 2026-09-28

Completed: I went back through the four-mode results specifically around the concerns you raised. A few of our previous statements were too strong. `k=180` should be treated as our current reference endpoint rather than ground truth, and the `N=51 -> 71` comparison shows that states 7 and 8 are encouragingly stable but not yet fully spatially converged or validated against omitted circuit parasitics. The `0 -> 7 -> 8 -> 1` pathway was not chosen beforehand for a physical reason; it emerged as the strongest three-step pathway under the four-mode `grid_phi` matrix-element ranking. The earlier `0 -> 5 -> 4 -> 1` pathway belonged to the symmetric one-mode `IdealGridium` model, and I should have made that transition explicit.

I also audited the gate results. The ~99.23% X90 and ~98.67% X180 results were optimized one-mode gates, while the recent ~36.6% X90 result was an unoptimized transplant into the asymmetric four-mode model, so those results are not directly comparable. Rerunning the frozen one-mode trajectories showed that X90 reaches `F_gate = 0.999967` at `32.2405 ns` and X180 reaches `F_gate = 0.999987` at `16.2472 ns`, before their nominal endpoints. I am treating those as candidate stopping points rather than validated shorter gates because changing `T_gate` changes the pulse envelope and requires revalidation.

The tracked `Figures/Rabi_3Photon/README.md` now serves as an output manifest. I also stopped using “smoke traces” for research-facing results; those were diagnostic/control-plumbing propagations, not validated gates.

### Responses to your questions

- **Model variables:** 

`EJ/EC/EL/ELK/EJS/ECS` are circuit energy scales; `eC/eP` parameterize capacitance terms; `eps_J/eps_LK` are asymmetries; `ng/phi_ext/theta_ext` are biases; `n1max/N2/N3/L2/L3/N4` are numerical cutoffs; `k/sigma/which/tol/MMD_AT_PLUS_A` are eigensolver settings. `eC=5.5`, `eP=10`, `eps_J=0.10`, and `eps_LK=0.05` are model/test assumptions rather than fitted device values.

- **Why `0 -> 7 -> 8 -> 1`?** 

It is the dominant three-step `grid_phi` path in the current asymmetric four-mode model at both N=51 and N=71. That makes it a strong numerical candidate, not yet a physically validated pathway.

- **States 7 and 8 convergence:** 

N=51 -> 71 gives overlaps of approximately `0.99959` and `0.99907`, while their transition frequencies move by approximately `3.84 MHz` and `6.20 MHz`. I therefore would not yet call them fully converged.

- **Why N=51/N=71 and k=180?** 

N=51 and N=71 were practical finite-difference resolutions, not a formal convergence sequence. Likewise, k=180 was simply our largest available same-model reference; it had not been independently checked against larger k.

- **Why does X90 look worse than X180?** 

It is longer and passes through a near-swap before rotating back toward the intended X90 operation, so the population trajectory looks more oscillatory even though its leakage is lower than X180.

### Larger-k convergence test

I finished and tested a frozen `n71-k220` runner specifically to address the retained-cutoff concern. It performs one N=71 Stage-1 solve at `k=220` and compares nested `k=180`, `200`, and `220` retained spaces from that same parent. It reports `f07`, `f08`, state-7/8 overlaps, the three `grid_phi` elements along `0 -> 7 -> 8 -> 1`, and pathway rankings. The focused runner/state-tracking test suite passes `33/33`.

The runner is on my feature branch at:

`48ca7b719726ae878622922fbbf2caa8f6673935`

I also packaged the authenticated N51 reference needed for the run:

`gridium-n51-reference-2026-09-26.tar.gz` (~64 MB)

SHA-256:

`e6007d9cf989bf6e770685c6c957dce232839aefc460a2eafe3bb13b2aeff8bd`

### Lawrencium run

After I send the N51 archive, extract it from the repository root so this path exists:

`research/checkpoints/2026-09-26-asymmetric-n51`

Then run the guarded dry run first:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Simulations.Rabi_3Photon.run_four_mode_spatial71_reference \
  --profile n71-k220 \
  --dry-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Simulations.Rabi_3Photon.run_four_mode_spatial71_reference \
  --profile n71-k220 \
  --execute \
  --output-root research/checkpoints \
  --n51-checkpoint research/checkpoints/2026-09-26-asymmetric-n51 \
  2>&1 | tee lawrencium-k220.log
```


## Work Block — 2026-09-29

My Credits reset tommorow so I decided to do some investigated work

Completed: While waiting for the Lawrencium `k=220` run, I used the existing authenticated N51/N71 checkpoints for several sandbox-only robustness studies. States 7 and 8 are numerically well contained within the current charge, spatial, and mode-4 cutoffs, but they are substantially more multimode and spatially extended than the logical states, and their small splitting is comparatively sensitive. The original `0 -> 7 -> 8 -> 1` route remains the clear coupling-product winner, but two useful robustness comparators emerged: `0 -> 7 -> 2 -> 1` (~12x weaker, avoids state 8 and the near `f78/f01` collision) and `0 -> 3 -> 8 -> 1` (~15x weaker, very stable from N51 to N71 and also avoids the middle-tone collision).

The most important model-completeness concern is now an omitted distributed superinductor-array mode. The paper reports plausible array modes in roughly the 4.8–6.5 GHz range, overlapping states 7/8 near 6.3 GHz. The current mode-4 coordinate is a distinct ~15.8 GHz endpoint/node mode and does not represent this internal array excitation. A predictive five-mode model therefore should not be built by guessing an oscillator coupling; we need the physical array identity and either device-specific ladder parameters or an extracted mode eigenvector/participation.

I also preregistered strict/practical criteria for the pending `k=180/200/220` comparison and built a sandbox postprocessor so the convergence decision is fixed before seeing the result. The k220 test will establish retained-basis stability only at fixed N71 in the present four-mode model; it will not establish spatial convergence or robustness to the omitted array mode.

Next: Wait for the authenticated N71 `k=220` Lawrencium result and evaluate all three candidate pathways against the preregistered criteria. If retained-basis stability is confirmed, move to spatial/domain convergence and then model/parasitic sensitivity before returning to four-mode gate optimization.

### Questions for Thomas

- Do you know which physical superinductor array produces the reported low-frequency array mode?
- Is there an EM/black-box extraction, normalized mode eigenvector/participation, or device-specific ladder data available for that mode?


## Work Block — 2026-09-30–2026-10-09

### Research objective and present decision

The central objective remains to develop and validate physically meaningful, high-fidelity logical X90/X180 control in the asymmetric superconducting four-mode Gridium model. The work in this block did not optimize or validate a gate. It tested whether the finite Stage-1 retained basis is trustworthy enough for that control problem, investigated why ordinary energy-prefix truncation fails, and tested a physics-informed alternative at an independently chosen Hamiltonian parameter point.

The current frozen decisions are:

| Question | Decision |
| --- | --- |
| Retained-basis convergence | **FAIL** |
| Frozen selector validation at `epsilon_J=0.11` | **BORDERLINE** |
| Multi-target transfer to `epsilon_J=0.11` | **PARTIAL** |
| Cross-parameter shell mechanism | **SUPPORTED** |
| Four-mode X90/X180 gate work | **HOLD** |
| Spatial convergence | **HOLD** |

These decisions refer to different questions and must not be conflated. Accurate state tracking does not by itself establish accurate operators or transition frequencies; accurate physics-selected subspaces do not make the ordinary energy prefix converged; retained-basis convergence does not establish spatial or physical-model convergence; and none of the static results below validates a physical device or a time-domain gate.

### Retained-basis progression at `epsilon_J=0.10`

The authenticated N71 sequence used one four-mode model and nested Stage-1 energy prefixes through `k=300`. The resulting transition frequencies are:

| k | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 180 | 15.857270 | 6275.014071 | 6297.747508 | 22.733437 | 6281.890238 | 2819.000502 | 3440.156299 | 3688.042355 | 2609.705153 |
| 200 | 15.924131 | 6273.719453 | 6294.568288 | 20.848834 | 6278.644156 | 2819.203767 | 3438.591555 | 3685.779303 | 2608.788985 |
| 220 | 15.908195 | 6272.819042 | 6292.853928 | 20.034886 | 6276.945733 | 2822.096724 | 3434.814124 | 3681.799676 | 2611.054252 |
| 240 | 15.864131 | 6246.176785 | 6261.563920 | 15.387135 | 6245.699789 | 2796.971743 | 3433.340911 | 3679.790146 | 2581.773775 |
| 260 | 15.855811 | 6243.253114 | 6257.821644 | 14.568530 | 6241.965833 | 2794.963818 | 3432.433485 | 3678.193165 | 2579.628479 |
| 280 | 15.862073 | 6243.055864 | 6256.688434 | 13.632570 | 6240.826361 | 2795.614153 | 3431.579637 | 3677.182032 | 2579.506402 |
| 300 | 15.831920 | 6190.483521 | 6206.337088 | 15.853567 | 6190.505168 | 2741.842182 | 3432.809419 | 3677.999330 | 2528.337758 |

All values are MHz. The original N71 `k=180` endpoint was therefore not a convergence certificate. The `k220 -> k240` step changed `f07`, `f08`, and `f78` by `-26.642257`, `-31.290008`, and `-4.647750 MHz`. After smaller `k240 -> k260` and `k260 -> k280` changes, the `k280 -> k300` step changed them by `-52.572343`, `-50.351346`, and `+2.220997 MHz`. Several carriers reversed direction, and the final shell was larger than the preceding shell for the important outer transitions. The progression is nonmonotonic and the final increment is not a small tail.

State identity and the dominant `0 -> 7 -> 8 -> 1` pathway remained trackable through these changes. That is useful but narrower evidence: overlaps answer whether the same state is being followed, whereas frequency differences test its energy, `grid_phi` comparisons test the control operator, and the retained-basis verdict asks whether all frozen observables stabilize together. At `k280 -> k300`, states 7/8 remained identifiable, yet their overlaps were only `0.998963/0.997201`, `g78` and the internal `{7,8}` operator block changed by `6.108%`, and the primary pathway score changed by `6.409%`. The authenticated conclusion is therefore **RETAINED-BASIS CONVERGENCE: FAIL**. `k=300` is another finite endpoint, not ground truth.

Traceable results: `research/checkpoints/four-mode-asymmetric-n71-k220/`, `research/checkpoints/four-mode-asymmetric-n71-k260/`, `research/checkpoints/four-mode-asymmetric-n71-k300/`, and `research/sandbox/k300_results/REPORT.md`.

### Why the large shells matter: Schur/Feshbach attribution

The saved parent eigensystems were sufficient to reconstruct the finite Stage-2 Hamiltonians and partition them into an included `P` space and a newly added `Q` shell. The energy-dependent Schur/Feshbach self-energy quantifies how virtual excursions into `Q` dress the low states. Exact reconstruction also separates the shell self-energy from relaxation of the dressed vector inside `P`; the two are both needed to reproduce the observed finite-shell energy shift.

For `epsilon_J=0.10`, ranks 292 and 290 account for approximately `86.75%` of the aggregate absolute `k280 -> k300` rank-attribution weight. Rank 287 is the smaller third contributor. For `epsilon_J=0.11`, ranks 292 and 290 still dominate, carrying approximately `84.42%`, while rank 289 becomes third. Cross-parameter raw-grid overlap tracking authenticated the relevant shell identities. Rank 289's group-78 coupling grows by approximately `20.86%` (`0.462470 -> 0.558934 GHz`) while its denominator changes only from `37.678725` to `37.690897 GHz`; this is a coupling redistribution, not a new resonance.

The mechanism is a concentrated, off-resonant virtual shell. At both parameter points the important denominators remain tens of GHz and the frozen near-resonant fraction is zero. Large effects arise from strong coherent coupling of a few ranks to the dressed target subspace, especially states 7/8, not from a near-resonant avoided crossing.

The exact analysis also explains why a fixed-vector second-order estimate is informative but not authoritative. At `epsilon_J=0.10`, states 7/8 receive exact shell self-energies of about `-67.17/-63.59 MHz`, partly offset by `+12.30/+10.94 MHz` of `P`-space relaxation. Ranks 292 and 290 interact nonadditively: their joint exact addback is not the sum of their singleton addbacks. At `epsilon_J=0.11`, the fixed-vector second-order estimate even gives the wrong sign for the small differential `f78` shell shift because it subtracts two much larger, nearly common-mode state corrections. This does not invalidate controlled second-order diagnostics for individual states; it shows why energy dependence, `P` relaxation, and rank interactions matter for cancellation-sensitive transitions.

The frozen conclusion **CROSS-PARAMETER SHELL MECHANISM: SUPPORTED** means that the same broad class-B mechanism—a few strongly coupled, off-resonant virtual Stage-1 ranks—survives the `epsilon_J=0.10 -> 0.11` change, with quantitative redistribution among subleading ranks. It does not establish a converged infinite retained basis, spatial convergence, completeness of the four-mode circuit model, or device-level validity.

Traceable analyses: `research/sandbox/k260_shell_attribution/REPORT.md`, `research/sandbox/k260_rank_addback/REPORT.md`, `research/sandbox/k300_shell_attribution/REPORT.md`, and `research/sandbox/epsj011_shell_attribution/REPORT.md`.

### Physics-informed retained-basis selection

The attribution results exposed the failure mode of a pure energy prefix: a small number of high-rank, off-resonant states can have large virtual influence even when their Stage-1 energies are not especially low. An initial same-parent rank-addback test at `k=260` selected ranks `{223,225,230}` and recovered roughly `85%` of the important state-7/8 and outer-carrier `k220 -> k240` reorganization with three ranks; equal-cost prefix and spread controls recovered very little. That experiment was promising but explicitly non-universal—it transferred to logical states 0/1 but not comparator states 2/3—and did not establish convergence.

The subsequent frozen selector therefore treats three target pairs equally: `(0,1)`, `(2,3)`, and `(7,8)`. For each candidate rank and pair it constructs the allowed isolated-shell Schur-response matrix from the candidate coupling block and its energy denominator. Each target group's aggregate response is used to whiten its features, preventing a numerically large group from dominating. A deterministic greedy log-determinant coverage objective then selects ranks that add complementary response directions across all three groups. `grid_phi` is reserved for validation and has zero selection weight. The exact implementation, weights, floors, tie handling, and budgets `3/6/10/15` were frozen before selected-space outcomes were inspected.

The selector identified the actual high-impact ranks:

| Parameter point | S2 exact shell-weight capture at budgets 3 / 6 / 10 / 15 |
| --- | --- |
| `epsilon_J=0.10` | `90.50% / 98.70% / 99.46% / 99.77%` |
| `epsilon_J=0.11` | `88.329% / 98.337% / 99.507% / 99.776%` |

The same frozen selector formula was reused at `epsilon_J=0.11` without refitting its weights, targets, or selection rules. This is strong evidence that coupling/denominator-aware selection captures influential virtual directions more efficiently than ordinary prefix truncation within the tested model. It is a model-reduction result, not a substitute for demonstrating convergence of the full energy-prefix calculation.

### Independent `epsilon_J=0.11` validation

The held-out point changed only `epsilon_J` from `0.10` to `0.11`; `epsilon_LK=0.05` and all other physical and numerical settings remained fixed. Its single guarded N71 `k=300` Stage-1 solve completed, atomically published, and authenticated. The checkpoint has configuration SHA-256 `8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646`, checkpoint SHA-256 `20a58be10ad8408ce479de259e1d2f3562cc46222e0eebc939d6d695803516bd`, maximum eigenpair residual `3.0423e-12`, and orthogonality residual `2.1585e-11`.

Applied independently at the new point, all 66 mandatory physics-selected components passed their strict accuracy thresholds. All three target groups improved strongly relative to equal-cost energy-prefix and control selections; state and subspace tracking passed; `grid_phi` operator and pathway errors remained small; no physics-selector cancellation flags appeared at budgets 10 or 15; and `0 -> 7 -> 8 -> 1` remained the rank-1 static pathway. These are encouraging transfer results.

The frozen classifications nevertheless remain exactly:

- **SELECTOR VALIDATION: BORDERLINE** because the union/budget-10 `f78` no-harm clause was missed.
- **MULTI-TARGET TRANSFER: PARTIAL** because the frozen strong condition requires more than broad improvement when the selector verdict itself is BORDERLINE.
- **CROSS-PARAMETER SHELL MECHANISM: SUPPORTED** because the independently tracked shell retains the same concentrated off-resonant mechanism.
- **RETAINED-BASIS CONVERGENCE: FAIL** because the ordinary `k280 -> k300` prefix still contains large carrier, energy, and operator changes, reversals, and nonshrinking increments.

The authenticated checkpoint is `research/checkpoints/four-mode-asymmetric-n71-epsj011-k300/`; frozen classifications are in `research/sandbox/parameter_point_validation/evaluation.json` and authentication in `research/sandbox/parameter_point_validation/authentication.json`.

### The `f78` cancellation exception

At `epsilon_J=0.11`, the union/budget-10 physics selection has `f78` error `+0.220384 MHz`, while the ordinary global prefix has the apparently better error `-0.140580 MHz`. The prefix result is misleading in isolation: its raw state-7 and state-8 errors are `+54.479177` and `+54.338596 MHz`, whose near-common-mode cancellation leaves the small transition error. The physics selection's corresponding state errors are only `+0.254622` and `+0.475006 MHz`.

Thus the prefix does not recover either state well; it obtains a favorable difference between two errors roughly two orders of magnitude larger. A good transition-frequency match alone is not sufficient evidence of reliable state-energy or subspace recovery. The preregistered no-harm rule intentionally did not permit this explanation to erase the miss after seeing the result, so the official selector verdict remains **BORDERLINE**.

### Static selectivity and implications for control

At the finite `k=300` endpoints, the middle/logical detuning changes materially:

| Parameter point | `f78-f01` |
| --- | ---: |
| `epsilon_J=0.10` | `+0.021648 MHz` |
| `epsilon_J=0.11` | `+1.260224 MHz` |

This demonstrates strong parameter sensitivity in a control-relevant spectral separation. The near-collision at `epsilon_J=0.10`, together with nonzero `grid_phi` couplings, is a warning for future frequency-selective pulse design and potential logical/middle-transition crosstalk. The movement at `epsilon_J=0.11` shows that the collision is not fixed across this parameter perturbation. These are static observations from unconverged finite retained bases: they do not establish an experimental addressability window, microwave-port response, or time-domain gate fidelity.

### Sealed `epsilon_J=0.11`, `k=340` experiment

The next retained-basis experiment is fully preregistered and has passed its separate October 9 prelaunch review. It is intentionally not authorized or executed by this documentation update.

| Item | Frozen value or status |
| --- | --- |
| Physical point | `epsilon_J=0.11`, `epsilon_LK=0.05` |
| Spatial/basis settings | `N2=N3=71`, `L2=11`, `L3=14`, `N4=8`, `n1max=4` |
| Stage-1 parent | `k=340`; `sigma=-24`, `which=LM`, `tol=1e-8`, `MMD_AT_PLUS_A` |
| Nested prefixes | `260, 280, 300, 320, 340` |
| New shells | ranks `300–319` and `320–339` |
| Configuration fingerprint | `10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0` |
| Frozen criteria SHA-256 | `ffdbe09ae353c0eb6746a95c67b7c6dd1461db1a6c3e2d1260b61f831c0e4c72` |
| Preregistration manifest SHA-256 | `dd26b9b819e53859a807ef99aab7d96e57a2a7cda6b6862187846518b7924d52` |
| Preregistration | **READY** |
| Prelaunch safety | **PASS** |
| Execution | **NOT AUTHORIZED** |
| Post-solve authentication | **NOT RUN** |
| k340 convergence | **NOT EVALUATED** |
| k340 shell diagnosis | **NOT AVAILABLE** |

All 15 prelaunch gates passed. The three frozen hashes matched; historical checkpoints authenticated; the dedicated Python `3.12.11` environment was valid; no competing k340 solver, final or incomplete checkpoint, execution locke, or conflicting output/log existed; memory pressure was healthy with approximately `33–36%` free memory; pages throttled remained `0`; atomic publication and output isolation were confirmed; all 8 cheap tests passed; and the sealed no-solve dry preflight passed. The guarded launcher requires separate explicit execution authorization. That authorization was not supplied during the review.

Consequently, no k340 Stage-1 eigensolve has been launched, no k340 spectrum exists, and no k340 convergence or shell-mechanism result exists. The scientific purpose of the sealed run is to distinguish an isolated influential `k280 -> k300` shell from continuing high-impact contributions above rank 300. Frozen comparisons require `k300 -> k320`, `k320 -> k340`, and `k300 -> k340`; a small final increment cannot erase a large earlier failure. Even a favorable result would support only a plateau within the tested fixed-N71 range, not convergence beyond k340 or ground-truth status for k340.

The sealed materials are in `research/sandbox/epsj011_k340_preregistration/`, especially `REPORT.md`, `execution_plan.md`, `criteria.json`, and `preregistration_seal.json`.

### Current limitations and next action

No validated high-fidelity four-mode X90/X180 gate has yet been demonstrated. Remaining limitations include unresolved virtual contributions above the current endpoint; failed retained-basis convergence; unresolved spatial/discretization convergence; finite four-mode model scope and the possible omitted distributed array modes; no validated mapping from `grid_phi` to a physical microwave port; and no converged time-domain logical-gate demonstration. These limitations prevent a present validation claim; they do not prove that a physical Gridium gate is impossible.

Next: (1) obtain explicit authorization for the already sealed k340 calculation; (2) run the existing guarded k340 parent exactly once, subject to its renewed safety gates; (3) authenticate the published output; (4) apply only the frozen convergence analysis; and (5) choose the following experiment from the observed shell behavior. Preserve the existing research-figure backlog. Until those steps are complete, **GATE WORK: HOLD** and **SPATIAL CONVERGENCE: HOLD**.


## Work Block — 2026-10-10 -> 2026-10-20

Completed: Consolidated the authenticated k340 results and completed the separately authorized k380 experiment on October 18. The k340 checkpoint completed on October 9; its earlier pre-execution description above remains a historical snapshot. Both experiments completed successfully and authenticated, but their frozen retained-basis verdict is **FAIL**, with shell classification **D_NONMONOTONIC_UNRESOLVED**. The k380 parent's maximum mandatory-carrier corrections across k300→320, k320→340, k340→360 and k360→380 are **0.714572, 1.299836, 5.594546 and 6.738198 MHz**. The final maximum grows, five of nine final carrier corrections increase, and cumulative k340→380 has seven practical component failures. Ranks354 and362 dominate the two new off-resonant shells, continuing the concentrated virtual-response pattern previously seen at rank328. Target tracking, grid_phi structure and the rank-1 static 0→7→8→1 pathway remain comparatively stable; they do not establish spectral convergence.

Scientific conclusion: Retained-basis convergence has **not been established**; the official verdict remains FAIL. Finite-prefix f78−f01 changes sign, and severe cancellation can hide large state-excitation changes. The independent selector verdict remains **BORDERLINE**, multi-target transfer **PARTIAL**, and the k380 secondary selector **PASS**. These are separate findings. No validated high-fidelity asymmetric four-mode gate, calibrated physical microwave-port mapping or spatial-convergence result is claimed.

Current status: Benchmark v1 remains sealed with complete execution **BLOCKED** by the missing independent historical selector-S2 endpoint. Replay would not recover historical independence. A separate restricted Schur-only v2 benchmark on S340/S360/U340 is now **READY, SEALED and AUTHENTICATED**, with implementation complete, **68 synthetic tests passing**, **110 source pins** and **31 manifest-pinned files**. It compares scalar, diagonal, full eleven-state matrix and energy-dependent eleven-state approximations. Actual-parent validation and benchmark execution are **NOT RUN**; numerical execution is not authorized. Final v2 prelaunch review is **ON HOLD at user request**.

The standalone [October 20 Research Work Block](docs/gridium-research-update-2026-10/WORK_BLOCK_2026-10-20.md) gives the numerical tables, interpretation, evidence locations and links to the six existing reviewed PDFs. Those figures cover earlier work through k340 and do not plot the new k380 result.

Next: Await Thomas's input. Conduct a separate prelaunch review only after user authorization, and seek separate numerical-execution authorization afterward. No new Stage-1 parent is required for the proposed restricted study. **GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.**

### Questions for Thomas

- Does the restricted finite-parent Schur comparison look like the most useful diagnostic before another contiguous-k expansion?
- Which circuit-theory or Hamiltonian-reduction checks would help interpret the recurring concentrated off-resonant contributions, including sensitivity to missing parasitics or distributed modes?
- Given the agreed abstract grid_phi coordinate, what device-level or calibration evidence should anchor the eventual physical microwave-port mapping before control claims?

---

## October 24, 2026 — Work Blocks 1–5 Research Update

### Executive Summary

Completed a systematic investigation of high-rank virtual-state contributions, physics-informed state selection, cross-shell attribution, and actual selected-space spectral accuracy in the asymmetric four-mode Gridium model.

The central result is that noncontiguous, physically informed state retention can reproduce the finite-parent dressed spectrum much more accurately than ordinary energy-prefix retention at comparable selected-state budgets.

However, simpler bare-coupling selectors remain competitive. The frozen scientific verdict is **MIXED_OR_NOT_IMPROVED**, not unconditional selector superiority.

### 1. Retained-Basis Convergence: k380

The authenticated k380 Stage-1 parent was used to investigate the retained-basis tail.

The maximum mandatory-carrier changes were:

| Shell | Maximum carrier correction (MHz) |
|---|---:|
| k300 -> k320 | 0.714572 |
| k320 -> k340 | 1.299836 |
| k340 -> k360 | 5.594546 |
| k360 -> k380 | 6.738198 |

The increasing corrections demonstrate unresolved, nonmonotonic spectral behavior.

The relative f78-f01 transition changes sign across finite truncations. This is truncation dependence, not evidence for a physically established crossing.

**Conclusion: RETAINED-BASIS CONVERGENCE FAIL.**

The k380 model is an authenticated finite-parent numerical reference, not a converged infinite-basis result.

### 2. High-Rank Virtual-State Mechanism

High-rank states 328, 354, and 362 were identified as especially influential virtual contributors.

State-fingerprint analysis found that their charge-edge and spatial-edge weights are relatively small. Their importance is therefore not explained simply by concentration at numerical cutoff boundaries.

Coupling-versus-Schur analysis showed strong target-dependent behavior.

Target-aware coupling tracks Schur attribution much more closely than aggregate bare coupling.

This supports a physical interpretation involving unusually strong coupling to specific dressed target states, especially states 7 and 8.

These observations are mechanistic evidence within the tested finite parent; they do not establish basis convergence or universal predictor performance.

### 3. Work Block 3 — Cross-Shell Selection

The frozen physics-informed selector was evaluated retrospectively against historical S1 attribution.

| Selection budget | S1 attribution captured |
|---|---:|
| 3 | 87.150372% |
| 6 | 98.058058% |
| 10 | 99.326400% |
| 15 | 99.946458% |

Results indicated strong identification of influential S1 shell states.

The study did not yet establish selected-space spectral accuracy or prospective cross-parent transfer.

### 4. Work Block 4 — Base-Matched S2 Validation

**Execution: SUCCESS — October 24, 2026.**

Using:

- P340 = Stage-1 ranks [0,340)
- Q_S2 = Stage-1 ranks [360,380)
- U_S2 = P340 union Q_S2

The frozen selector was evaluated against base-matched S2 Schur attribution.

| Budget | S1 capture | S2 capture |
|---|---:|---:|
| 3 | 87.150372% | 98.680013% |
| 6 | 98.058058% | 99.289072% |
| 10 | 99.326400% | 99.816194% |
| 15 | 99.946458% | 99.912058% |

The three dominant selected S2 ranks were:

**362, 379, 375.**

Together they captured 98.680013% of absolute S2 attribution.

Total absolute S2 attribution was 12.607858748 MHz, with rank 362 dominating the distribution.

The experiment completed two large Stage-2 eigensolves, one small Q decomposition, and six shifted linear solves.

- Runtime: 24.959 seconds.
- Peak sampled RSS: approximately 1.199 GiB.
- Numerical validation: PASS.
- Source and output authentication: PASS.
- Preservation: PASS.

**Conclusion:** The frozen selector successfully captures concentrated S2 virtual-state attribution, but bare-coupling and denominator controls are also competitive.

### 5. Work Block 5 — Actual Selected-Space Spectral Accuracy

**Execution: SUCCESS — October 24, 2026.**

This was the first direct test of actual dressed-spectrum accuracy using the selected spaces.

The experiment completed:

- 19 unique selected-space eigensolves.
- 28 frozen method/budget comparisons.
- Eleven requested eigenpairs per unique space.
- Zero Stage-1 eigensolves.
- Zero retries.

All results were compared against the authenticated finite U_S2 reference.

#### Maximum Excitation-Energy Errors

All errors below are in MHz. Lower is better.

| Budget | Physics F | Energy prefix E | Bare T/D/V |
|---|---:|---:|---:|
| 3 | 0.279658 | 0.535358 | 0.279658 |
| 6 | 0.115900 | 0.528191 | 0.133264 |
| 10 | 0.070881 | 0.523126 | 0.079541 |
| 15 | 0.019042 | 0.509421 | 0.028006 |

Physics-informed selection achieved the lowest maximum excitation error at every budget, tying the bare controls at budget 3 because they selected identical spaces.

Under the complete frozen multi-metric evaluation:

- **Versus energy prefix:** IMPROVED at budgets 6, 10, and 15.
- **Budget 3:** Did not satisfy the centered-state-10 no-harm condition.
- **Versus bare coupling:** Consistent superiority was NOT established.
- **Overall verdict:** MIXED_OR_NOT_IMPROVED.

Several bare-coupling comparisons were better on specific carrier-error metrics, despite physics-informed selection having a smaller maximum excitation error.

All physics-selected spaces passed the registered strict finite-parent accuracy thresholds.

Five cancellation flags were observed among spread/reverse-spread controls. None occurred for the physics-informed selections.

Numerical execution results:

- Runtime: 213.313 seconds.
- Peak sampled RSS: approximately 1.398 GiB.
- Numerical validation: PASS.
- Source and output authentication: PASS.
- Preservation: PASS.

Historical screening costs remain unavailable. No end-to-end computational speedup is claimed.

### 6. Scientific Interpretation

The completed investigation supports three findings:

1. High-rank virtual states can strongly influence the dressed spectrum, even when they are far from the retained energy range.
2. Physics-informed, noncontiguous retention can produce accurate finite-parent compression while ordinary energy-prefix retention remains substantially less accurate.
3. More sophisticated physics-informed ranking does not yet demonstrate consistent superiority over simpler bare-coupling selection.

Importantly, attribution capture alone is not proof of spectral accuracy. Work Block 5 supplied the previously missing direct numerical comparison.

The results do NOT demonstrate:

- Convergence of the full asymmetric four-mode Hamiltonian.
- Convergence in spatial or charge-domain truncation.
- A general-purpose superior state-selection algorithm.
- End-to-end computational speedup.
- A validated high-fidelity asymmetric four-mode gate.

### 7. Illustrated Research Handoff for Thomas

Prepared a comprehensive illustrated report containing:

- Eight vector PDF figures.
- Eight matching high-resolution PNG figures.
- Retained-basis convergence diagnostics.
- Virtual-state attribution and coupling analysis.
- S1/S2 selector comparisons.
- Selected-space spectral-accuracy comparisons.
- Individual-state and carrier-error diagnostics.
- Cancellation and retained-space diagrams.

Local report:

`docs/gridium-research-update-2026-10/work-blocks-1-5/THOMAS_RESEARCH_UPDATE.md`

Figure index:

`docs/gridium-research-update-2026-10/work-blocks-1-5/README.md`

The handoff package contains 37 new files and has been verified locally.

**Publication status: PREPARED, NOT YET VERIFIED PUBLISHED.**

The official GitHub destination is:

`SciGeek2000/2q_gridium`

Branch:

`four-mode-asym-gridium-tests`

The last verified upstream commit is:

`bce946ece96f1f715a29135a6382efe36e0300f1`

Publication was interrupted by Codex usage limits. Resume only after checking local Git status and preserving unrelated staged changes.

### 8. Next Research Direction

The recommended next scientific experiment is independent finite-parent transfer validation of the unchanged selector against bare-coupling controls.

Before launching:

1. Publish the October 24 illustrated report.
2. Share findings with Thomas.
3. Request feedback on the most informative independent parent, parameter point, or shell.
4. Preregister the transfer experiment with unchanged selection rules.
5. Obtain new authorization before additional numerical execution.

### 9. Current Research Holds

| Research activity | Status |
|---|---|
| Work Blocks 1–5 | COMPLETE |
| October 24 Thomas handoff | PREPARED, publication pending |
| Independent selector transfer | PROPOSED |
| Schur-v2 benchmark execution | HOLD |
| Asymmetric four-mode gate work | HOLD |
| Spatial convergence | HOLD |

**Overall scientific verdict: MIXED_OR_NOT_IMPROVED.**

**Current priority: Publish the validated scientific handoff and obtain Thomas's feedback before beginning another experiment.**

## Work Block — 2026-10-25–2026-10-29

Completed: Developed and reviewed inactive k420 native A/B/C integration candidates, verified the frozen ten-case Stage-2 workload and phase-handoff contracts, and exercised memory monitoring, independent watchdogs, telemetry validity/freshness, STOP handling and cleanup with bounded metadata workers. Accelerated full-duration audit tests retained 6,001 snapshots over a virtual 1,500-second budget, with approximately 5.50 MB of journal evidence. Subsequent independent checks exposed journal-integrity and time-of-check/time-of-use defects at finalization, downstream dispatch and terminal publication. Defined Route A around immutable authenticated admission state, coordinated STOP/dispatch transitions and protected publication, but its stronger publisher boundary remains unavailable under the current same-user macOS ownership arrangement; no protected native slice was completed or accepted. Two local memory-admission snapshots failed the unchanged free-memory and availability-proxy gates. Zero new scientific eigensolves were performed.

Next: Find authorized compute resources, establish an appropriate numerical environment and explicitly justified execution trust assumptions, and verify monitoring, admission and provenance before requesting separate authorization to resume the frozen k380 → k400 → k420 convergence experiment. Follow up on Thomas's previously reported Lawrencium access; no current external account/allocation for this k420 run has been verified.

![Previously computed maximum absolute changes among nine dressed carrier frequencies across four same-parent retained-prefix increments](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/fig01_retained_basis_convergence.png)

*Figure 1. Prior authenticated k380 results: the largest absolute change among nine registered dressed carrier frequencies rises across the tested nested-prefix increments. All prefixes use the same k380 Stage-1 parent at ε_J=0.11 and fixed N71. Convergence remains unestablished; no k400/k420 values are plotted.* [Source validation](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/FIGURE_VALIDATION.md#figure-1).

## Stoppage Update 02 — 2026-10-29

The genuine k420 convergence experiment has not started. Its purpose remains to evaluate k380 → k400 → k420 using one new k420 Stage-1 parent and the frozen ten-case Stage-2 workload, followed by actual scientific convergence evaluation. Existing retained-basis convergence remains unresolved; the authenticated k380 finite-parent results do not establish a converged spectrum.

The integration work developed inactive native A/B/C candidates and exercised memory monitoring, watchdogs, telemetry, phase handoffs and cleanup with metadata workers. Accelerated audit tests covered the full virtual 1,500-second budget and retained 6,001 snapshots; this was simulated-time retention evidence, not a live scientific run. Independent checks then demonstrated three acceptance races: journal corruption after verification could permit false successful completion; Stage B could dispatch before corrupted evidence was rejected; and terminal receipt contents could change between verification and publication. Later rejection did not undo an incorrect launch.

Route A was designed around immutable authenticated evidence, coordinated STOP/admission/dispatch transitions and a protected publisher. It was not successfully implemented or accepted. Its stronger publication guarantee requires excluding an uncooperative filesystem writer from authoritative artifacts. The current same-user macOS ownership arrangement does not provide that exclusion, and no separate-principal publisher is provisioned. An atomic rename or read-only descriptor alone does not seal mutable source contents.

The local 18-GiB Mac also failed both admission snapshots. The first observed approximately 62 MiB physically free and a 7.34 GiB availability proxy. The quiet-session check observed 0.801575 GiB physically free and a 7.282257 GiB availability proxy. Frozen requirements remain at least 1 GiB physically free and 12 GiB availability proxy. Both checks failed; pressure percentages were not substituted for those quantities. Current local execution is impractical under the unchanged policy, without establishing that this hardware is universally incapable of the calculation.

![Two panels comparing October 29 availability-proxy and physically free memory observations with the unchanged 12 GiB and 1 GiB admission minima](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/fig02_host_memory_admission.png)

*Figure 2. October 29 observations on the Apple M3 Pro with 18 GiB unified memory. (a) Canonical availability proxy; (b) physically free memory. Both sessions fail both frozen gates. Pressure percentages are not substitutes, and these observations do not establish universal hardware incapacity.* [Source validation](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/FIGURE_VALIDATION.md#figure-2).

There are no new k400/k420 numerical results, no new convergence claim and no new four-mode gate validation. Existing scientific results, checkpoints and failure provenance remain preserved. Engineering tests are not substitutes for scientific convergence evidence. Schur-v2 prelaunch, gate work and spatial convergence remain HOLD.

Recovery now depends on identifying authorized external compute access, establishing the appropriate environment and explicitly justified trust assumptions, and verifying monitoring, admission and provenance. Thomas's earlier Lawrencium offer is a lead; external compute access for this experiment has not yet been verified. Separate authorization is required before real calculations. The next scientific result must come from performing the frozen k420 experiment and reporting its actual convergence behavior to Thomas.

![Planned k420 scientific workflow with separate local-memory and protected-publication blockers feeding execution admission, followed by unexecuted Stages A, B and C](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/fig03_k420_execution_roadmap.png)

*Figure 3. Planned scientific workflow. Local memory admission and the unavailable protected native boundary block progress; external compute access remains unverified. Real execution requires separate authorization. The final output must be a tested convergence verdict, not an assumed converged spectrum.* [Source validation](docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/FIGURE_VALIDATION.md#figure-3).
