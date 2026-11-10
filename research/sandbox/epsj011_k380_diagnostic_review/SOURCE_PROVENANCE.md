# Source provenance and preservation

Review date: 2026-10-18. Function: saved-data numerical diagnostics. Audience: scientific direction/research review. Domain: superconducting-qubit scientific computing. No experimental line or device claim is inferred.

## Authentication

The existing `authenticate_existing.authenticate_new` function was called on the k380 checkpoint and selection, without launching any guard or solver. It passed historical authentication, COMPLETE linkage, configuration, payload hashes, selection seal/linkage and parent quality. All 87 k380-manifest-pinned files were independently rehashed. The frozen evaluator was reproduced in memory from saved metrics without writing its outputs and exactly matched every saved verdict and selector ratio.

Key SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Configuration fingerprint | `8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e` |
| Frozen criteria | `a2318603f3e2b5cefcb68873425c7873d0a6243fc7017949352129c3beeb616a` |
| Preregistration manifest | `95f33014769195605ef871f62fee229d0417f610fb6e96dd44df71e710707656` |
| k380 checkpoint metadata | `399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8` |
| k380 metrics | `aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e` |
| k380 evaluation | `43903489aba94442dc868dae2eafa319de82f3cf3e3441db6144d6568c4a91aa` |
| k380 execution receipt | `c7ff154d805ded48367a28ac5137901ad9567d755372ed69bae64bc3c5c36d0b` |

The metrics/evaluation/receipt hashes match the previously reported execution outputs; those post-execution files are not members of the pre-execution manifest. Hash linkage detects later byte changes. It is not a digital signature or independent timestamp. Exact source and preservation hashes, including older artifacts, are in [SOURCE_HASHES.json](SOURCE_HASHES.json).

## Authoritative records and locators

| Source | Used fields / purpose |
| --- | --- |
| [k380 metrics](../epsj011_k380_preregistration/k380_metrics.json) | `prefix_metrics/{k}/energies_GHz`, `carriers_MHz`, `comparisons`, `trend`, `shell_diagnostics/S1,S2/states/{state}/ranks`, `dressed_shell_modes`, `nonadditivity`, selector metrics |
| [k380 evaluation](../epsj011_k380_preregistration/evaluation.json) | Official FAIL, D_NONMONOTONIC_UNRESOLVED, secondary PASS; unchanged |
| [k380 receipt](../epsj011_k380_preregistration/execution_receipt.json) | Successful guarded execution and measured wall time |
| [criteria](../epsj011_k380_preregistration/criteria.json), [manifest](../epsj011_k380_preregistration/preregistration_manifest.json), [seal](../epsj011_k380_preregistration/preregistration_seal.json) | Scientific rules, expected hashes, chronology and configuration |
| [k380 checkpoint](../../checkpoints/four-mode-asymmetric-n71-epsj011-k380/checkpoint.json), COMPLETE and payload archives | Authentication; Stage-1 energies and the existing rank328 overlap comparison |
| [k380 checkpoint analysis](../../checkpoints/four-mode-asymmetric-n71-epsj011-k380/analysis.json) | Selection-before-validation linkage; nested-prefix metrics; original-k340 tracking |
| [k340 metrics](../epsj011_k340_preregistration/k340_metrics.json) | `shell_diagnostics/new_S4` supplies historical rank328 exact scalar attribution, couplings, probabilities, relaxation and denominator/mode summaries |
| [k340 checkpoint](../../checkpoints/four-mode-asymmetric-n71-epsj011-k340/checkpoint.json) and eigenarchive | Authenticated historical reference; rank328 Stage-1 energy/vector |
| [k340 selection](../epsj011_k340_preregistration/selection.json) and [k380 selection](../epsj011_k380_preregistration/selection.json) | Frozen raw isolated-rank 2×2 selector responses, per-pair denominators and couplings; seals authenticate the files |
| [historical evidence](../epsj011_k380_preregistration/historical_evidence.json) | Earlier pinned results and source-map chain |
| [earlier shell attribution](../epsj011_shell_attribution/summary.json) and [methodology](../epsj011_shell_attribution/methodology.md) | Authenticated earlier methodological context; this earlier k300 study is not the source of rank328's later k340 attribution |
| [k380 analysis method](../epsj011_k380_preregistration/analyze_k380.py), [analysis plan](../epsj011_k380_preregistration/analysis_plan.md), [frozen selector](../k300_preregistration/selector.py) | Exact definitions, sign/unit conventions and separation of selector model from full-shell Schur diagnostics |

## Calculations performed

Only saved-value sums/differences/ratios, complex 2×2 entry reads, mode-weight aggregation, and one Stage-1 cross-parent overlap product were added. No Hamiltonian was built and no eigenproblem, inverse, root solve or new resolvent solve was performed. The existing frozen evaluator performs only classification arithmetic here. Archive arrays were read in memory and were not copied into the review directory.

`build_review.py` regenerates the JSON/report/rank outputs in a clean destination. `build_spectral_csv.mjs` loads the processed table into the runtime Artifact Tool, checks that typed values are unchanged, and serializes a CSV. The CSV intentionally contains no formulas or styling, and no XLSX or figure is generated. Rows retain binary64 source precision rather than the rounding displayed in Markdown. Null first-shell growth values export as empty fields, not zero. Boolean columns are explicit true/false.

The CSV is long-form: 36 records, nine carriers × four adjacent comparisons. `signed_upper_minus_lower_MHz` is the new reporting convention; `frozen_lower_minus_upper_MHz` preserves the opposite frozen component convention. Every row includes a JSON pointer into the metrics source. Frequency magnitudes and signed state differences coincide because these saved tracked target transitions have positive energy ordering; this was checked from the saved energies. Raw pair-common shifts use the fixed saved energy zero, while excitation shifts remove the ground-state shift. Diagnostic ratios do not create or replace scientific decision thresholds.

For scalar Schur verification the frozen 1e-9 GHz identity tolerance is reused. The separate derivative equality uses ordinary floating-point numerical-consistency tolerances (relative1e-9, absolute1e-15), not a scientific acceptance criterion or a change to the frozen verdict. Full residual values are retained rather than only Boolean summaries.

## Limits

Exact full-shell off-diagonal self-energy matrices and scans in energy are **INCOMPLETE**: dressed Stage-2 eigenvectors/contractions were not saved. The selector's raw isolated-rank response matrices are available but have different P spaces/energy arguments and cannot substitute for them. Historical rank328 has no saved per-mode list or fixed-vector second-order diagnostics. Rank354/362 cross-parent continuity and all beyond380 tail bounds are also **INCOMPLETE**.

Rank328's cross-parent absolute overlap is numerically1 to floating-point normalization precision (the unrounded value is slightly above1). No new physical identity threshold is introduced. Sorted within-parent rank labels are distinct from tracked dressed-state labels; response concentration recurring at different ranks does not mean one eigenstate is migrating between ranks.

## Preservation

SOURCE_HASHES.json records a 439-file baseline covering tracked files, the Git index, protected notes/backups, existing figures, manifest-pinned sources, and all top-level k340/k380 checkpoint and preregistration files. Final verification compares these hashes and Git status. Only this new diagnostic directory receives outputs. See VERIFICATION.json for the final comparison and OUTPUT_HASHES.json for deliverable hashes.

Initial Git status:

```text
M  Simulations/Cphase/main.ipynb
M  Simulations/Cphase/workflow_funcs.py
?? EmilianoUPDATE-unsaved-backup.md
```

No existing checkpoint, criterion, preregistration, production file, note, backup or figure is changed. No staging, commit or push is part of this review.

GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
