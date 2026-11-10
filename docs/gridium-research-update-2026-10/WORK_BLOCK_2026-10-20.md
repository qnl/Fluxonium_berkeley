# Gridium Research Work Block — October 10–20, 2026

## Research objective

Our goal remains reliable logical X90/X180 control in the asymmetric four-mode
Gridium model. This block addressed its numerical foundation: whether retained
Stage-1 states reproduce the relevant dressed energies and control operators,
and how to represent influential omitted states efficiently. No four-mode gate
was optimized or validated during this block.

The main result is that the new k380 endpoint **does not establish retained-basis
convergence**. We have consequently prepared a restricted Schur-method benchmark,
but have not executed it. V2 prelaunch, gate work and spatial convergence are
**ON HOLD**. This update supersedes the execution-status statements in the
earlier October 9 synthesis; those statements remain historical snapshots.

## Completed numerical experiments

The k340 experiment completed successfully on October 9, according to its UTC
checkpoint receipt, and is included here as the baseline for subsequent work.
Execution and authentication passed; the frozen conclusions are
**RETAINED-BASIS CONVERGENCE: FAIL** and **D_NONMONOTONIC_UNRESOLVED**. Its final
shell was larger than its predecessor despite comparatively stable state
identity and operators.

The separately preregistered k380 experiment completed on **October 18**. Its
checkpoint contains 380 Stage-1 eigenvalues, with maximum eigenpair residual
approximately **5.919e-8** and orthogonality residual **1.328e-11**, within the
frozen numerical-quality limits. Checkpoint authentication and the frozen
evaluation completed successfully. Numerical solver quality passed while
retained-basis convergence again **FAILed**; the shell classification remains
**D_NONMONOTONIC_UNRESOLVED**. These are different tests of reliability.

Both studies use the same physical point, epsilon_J=0.11 and epsilon_LK=0.05,
with N2=N3=71, n1max=4 and N4=8. Here k counts retained Stage-1 states;
the subsequent dressed calculation includes all eight mode-4 states per rank.

## Scientific results

The k380 parent's nested prefixes give the following maximum absolute
corrections among the nine mandatory carriers:

| Prefix increment | Maximum correction, MHz |
|---|---:|
| k300→320 | 0.714572 |
| k320→340 | 1.299836 |
| k340→360 | 5.594546 |
| k360→380 | 6.738198 |

The first two rounded values also agree with the earlier k340 study. Separate
parents remain distinct numerical objects. The final maximum grows rather
than shrinks; **five of nine** carrier corrections increase in magnitude in
the last comparison. The cumulative k340→380 comparison has **seven practical
component failures**, including five carriers and two excitation energies.

Control-relevant selectivity also depends on the retained basis:

| Nested prefix in the k380 parent | f78−f01, MHz |
|---|---:|
| k300 | +1.260224 |
| k340 | +0.713500 |
| k360 | −0.335134 |
| k380 | −0.925191 |

This sign change reverses the relative ordering of two transition frequencies
in finite truncations. It is not a converged physical crossing or a device
parameter sweep. In the final shell, Δf08=−6.738198 MHz and Δf07=−6.131699 MHz,
while their difference gives Δf78=−0.606499 MHz. The smaller splitting correction
conceals large constituent excitation corrections; the frozen cancellation
diagnostic labels this case severe.

## Interpretation of virtual-state contributions

The saved-data Schur/Feshbach analysis identifies a recurring, concentrated
off-resonant response. In the k340 parent, rank328 dominated the final shell.
The k380 diagnostic finds:

| Added shell | Aggregate absolute Schur weight, MHz | Dominant rank | Fraction |
|---|---:|---:|---:|
| 340–359, diagnostic S1 | 14.642728 | 354 | 66.82% |
| 360–379, diagnostic S2 | 17.011414 | 362 | 88.99% |

No near-resonant shell weight was identified under the frozen diagnostic.
These ranks describe virtual contributions in the finite Hamiltonian, not
identified physical circuit resonances. An absolute attribution weight aggregates
magnitudes across target states; it is **not an exact signed transition shift**.
The observed energy change also includes relaxation of the dressed vector
within the retained P space. Contributions are target-dependent and need not
add independently across ranks.

Small off-resonant admixtures can shift energies materially while leaving
overlaps and low-state operator matrix elements comparatively stable. Consistent
with this interpretation, target tracking and grid_phi diagnostics remain stable
relative to the spectral changes, and the static 0→7→8→1 pathway remains rank1.
This supports a finite-model mechanism, not convergence or robustness to omitted
circuit parasitics.

## Physics-informed retention and Schur methods

Coupling-aware retention remains promising, but its frozen outcomes answer
different questions: independent parameter-point selector validation is
**BORDERLINE**, multi-target transfer is **PARTIAL**, and the k380 secondary
selector evaluation is **PASS**. The latter does not replace the earlier
verdicts or certify retained-basis convergence.

We are investigating effective-Hamiltonian corrections before proposing another
contiguous-k expansion because recurring influential ranks suggest that retaining
states by energy alone can be inefficient. Saved fixed-vector second-order
corrections miss exact state shifts by as much as **0.708978 MHz** in diagnostic
S1 and **0.982366 MHz** in S2. Their off-resonant perturbative checks remain
satisfied; these discrepancies motivate an accuracy comparison, not a claim
that perturbation theory has broken down.

The proposed methods preserve distinct approximations: scalar corrections keep
each base vector fixed; diagonal corrections use one common energy argument;
full eleven-state matrices retain off-diagonal mixing within eleven dressed P
directions; energy-dependent eleven-state models additionally retain explicit
Q support. Neither eleven-state method is an exact full-P Schur solution.
Their accuracy on k380 has **not yet been measured**.

The original benchmark v1 is sealed but its complete execution is **BLOCKED**:
the full independent historical selector-S2 endpoint record was not preserved.
A deterministic numerical replay is possible, but cannot become independent
historical validation of itself. That selector endpoint retains
[0,340)∪[360,380); it is not the contiguous Schur S360 reference.

The separate restricted v2 scientific design is **SUPPORTED**. Its
preregistration is **READY, SEALED and AUTHENTICATED**, with implementation
complete, **68 synthetic tests passing**, **110 authenticated source pins** and
**31 manifest-pinned files**. Its registered partitions are:

| Cell | P ranks | Q ranks |
|---|---|---|
| S340 | [0,340) | [340,360) |
| S360 | [0,360) | [360,380) |
| U340 | [0,340) | [340,380) |

Future reconstruction would require three dressed solves of dimensions
2720/2880/3040, each targeting eleven eigenpairs; no new Stage-1 parent is needed.
Separate tensor-action and direct-solve checks complement historical contiguous
references, but share the finite model and input projections. Synthetic success
does not establish actual-parent validation. **Benchmark execution: NOT RUN.
Final prelaunch review: ON HOLD at user request.**

## Current limitations

No retained-basis plateau or bound above k380 has been established. Spatial/domain
convergence, finite-model adequacy and omitted distributed-array-mode sensitivity
remain unresolved. No calibrated physical microwave-port mapping or validated
high-fidelity asymmetric four-mode X90/X180 gate has been demonstrated. Stable
static pathways alone do not answer these questions.

## Next proposed research step

Await Thomas's input on the restricted benchmark and its physical interpretation.
A separate prelaunch review should occur **only after user authorization**;
any numerical execution would require its own explicit authorization and renewed
resource gates. No automatic reconstruction, S2 replay, k400 expansion or gate
work follows from this documentation update.

## Questions for Thomas

1. Does the restricted finite-parent Schur comparison look like the most useful
   next diagnostic before another contiguous-k expansion?
2. Which circuit-theory or Hamiltonian-reduction checks would you prioritize to
   interpret the recurring concentrated off-resonant contributions, particularly
   when considering missing parasitics or distributed modes?
3. Given the agreed abstract coordinate grid_phi=theta3+theta2/2, what device-level
   or calibration evidence should anchor its eventual physical microwave-port
   mapping before we make control claims?

## Evidence and provenance

The six existing reviewed PDFs provide background through k340, **not new k380
plots**: [A2](figures/A2_adjacent_shell_frequency_corrections_v2.pdf),
[B3](figures/B3_k300_cross_parameter_schur_attribution_v2.pdf),
[C2a](figures/C2a_selector_vs_controls_epsj010_v2.pdf),
[C2b](figures/C2b_selector_vs_controls_epsj011_v2.pdf),
[C3](figures/C3_f78_error_cancellation_v2.pdf), and
[D1](figures/D1_k340_spectral_convergence_v2.pdf). See their
[captions](FIGURE_CAPTIONS.md) and [figure provenance](PROVENANCE.md).

The following are **local authenticated artifacts, not public GitHub links**;
raw checkpoints and sandbox implementations are not included in this publication:

| Local source under `research/` | Evidence |
|---|---|
| `checkpoints/four-mode-asymmetric-n71-epsj011-k{340,380}/` | `COMPLETE`, checkpoint configuration/quality and payload hashes |
| `sandbox/epsj011_k{340,380}_preregistration/` | Final `evaluation.json`; `k340_metrics.json` / `k380_metrics.json`, including prefix, shell and cancellation records |
| `sandbox/epsj011_k380_diagnostic_review/` | `DIAGNOSTICS.json`, `SOURCE_HASHES.json`, `VERIFICATION.json`: saved-data attribution and spectral audit |
| `sandbox/epsj011_k380_{benchmark_execution_readiness,selector_s2_dependency_review,benchmark_v2_design}/` | Independent dependency and scientific-scope reviews |
| `sandbox/epsj011_k380_schur_benchmark_v2_preregistration/` | Sealed criteria, implementation, tests and `authentication.json` |

V2 manifest SHA-256:
`1d78eed131efcba2708be3286c384cb08e178384c1c9e46a1bf15aa4819827e5`.
Seal SHA-256:
`7aa8b6def0f539a08ce3241fd2534a3472f0ffe729c389f7bfbb7e92ea55acb8`.
These authenticate local byte integrity, not independent physical validity.

**V2 PRELAUNCH: ON HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.**
