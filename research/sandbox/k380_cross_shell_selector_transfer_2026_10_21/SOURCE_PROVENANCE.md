# Source provenance and preservation

**SOURCE AUTHENTICATION: PASS. PRESERVATION: PASS.**

Configuration SHA-256: `8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`.
Historical algorithm SHA-256: `57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2`.
Historical method SHA-256: `2055c7d54d81ada97379f4941ef08354dbb3cbf9378aa2b10dde1349ca2ff7cc`.
Frozen selection SHA-256: `6720cb7b2213962bf490ec6096ebd88e0b66e79ec7a2c25dee33116fc02eb8b7`.
Current protocol SHA-256: `f8446e05b26e81204d422d6ecf8406b90419350f4c1e767f6964019d6716a8ac`.
Current frozen-order snapshot SHA-256: `df9f9037282a88fb7d05bb6cf85bed9350390b8d089658e50b800e3ecf610209`.

Direct authentication checks only required small inputs: selector/source-method pins in original October8 and k380 October18 manifests; criteria/seal linkage; selection/seal; checkpoint COMPLETE/configuration/analysis linkage; selection-before-validation flags/provenance; exact metrics hash/selection link; consumed prior output manifests; and the existing targeted S2 dependency finding. Checkpoint Stage-1/operator arrays are not loaded or rehashed. Their earlier authenticated payload linkage is explicitly reused.

| Source | Purpose | SHA-256 |
| --- | --- | --- |
| research/sandbox/k300_preregistration/selector.py | Exact historical implementation | 57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2 |
| research/sandbox/k300_preregistration/selector_method.md | Equations, whitening, regularization, targets | 2055c7d54d81ada97379f4941ef08354dbb3cbf9378aa2b10dde1349ca2ff7cc |
| research/sandbox/epsj011_k380_preregistration/criteria.json | Registered P340 and budgets | a2318603f3e2b5cefcb68873425c7873d0a6243fc7017949352129c3beeb616a |
| research/sandbox/epsj011_k380_preregistration/selection.json | Frozen full orders and budget support | 6720cb7b2213962bf490ec6096ebd88e0b66e79ec7a2c25dee33116fc02eb8b7 |
| research/sandbox/epsj011_k380_preregistration/selection.json.seal.json | Selection integrity/provenance | 06c175d947c74896e9c4231ec2c46b461e524cdeb89a5a364b969cc6a93d7712 |
| research/sandbox/epsj011_k380_preregistration/selection_hook.py | Pre-validation call order | 46fc9478bc7ac1cc03cbd2cc27d117eeb1a98d69da11da508ae1f51d532934d3 |
| research/sandbox/epsj011_k380_preregistration/parent_analysis.py | Full-reference entry point after freeze | 48a8abf25b408c1a21f26a73fd6f5633abfc2148fe8083be9981523bdb14bf18 |
| research/sandbox/epsj011_k380_preregistration/analyze_k380.py | P340/P360 label and serialization semantics | 1f444cdf8134b02813bff201f664c810e267d821d10e0cd2a393fd7ded4509f9 |
| research/sandbox/epsj011_k380_preregistration/k380_metrics.json | Exact saved attribution labels | aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/checkpoint.json | Authenticated completed parent | 399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/analysis.json | Selection/linkage and contiguous endpoint support | 49b447ea8df6c6889e7cf4b2d5c5f7eef7cc4696821ac1a3f2d7265dc361787b |
| research/sandbox/k380_coupling_vs_schur_2026_10_21/SHELL_COMPARISON.csv | Whitelisted bare controls only | d1aef885a0c8f829f7677e9088b308140c30b2f16d1d0ad1dd5c0f2a0448a0ae |
| research/sandbox/k380_coupling_vs_schur_2026_10_21/RANKING_ANALYSIS.csv | Reuse four completed control correlations | a71e98125c36339b561d420d7c083ddc7f246821e58ee1700669f1ccab182521 |
| research/sandbox/epsj011_k380_selector_s2_dependency_review/SELECTOR_S2_DEPENDENCY_REVIEW.md | Previously established isolated-S2 endpoint absence | 718050d4452b7651088b0663fc3d4a14683cd33bfaf5119f23c6924027d75704 |

Chronology evidence is local: pinned code seals both shell orders before candidate/full-new-shell dressed validation, and linked checkpoint/metrics records identify the same selection. This is stronger than trusting a historical name, but is not an external timestamp or signature. No historical independent-validation claim is newly awarded.

Units and conventions remain the original H/h GHz, zero-based Stage-1 ranks and rank-major/mode4-minor retained support. Existing complex matrix features are neither replaced with magnitudes nor recomputed; only the sealed matrix-greedy order is used. Scalar trace and exact target-coupling alternatives are excluded from the current predictive ranking.

SOURCE_AUTHENTICATION.json lists 47 narrow before/after hashes. The set contains only consumed historical inputs, their relevant seals/reports, user-edited notes/Cphase files, original v1/v2 seals/manifests and Git index. No large prior audit was replayed. All hashes and Git status match. No pre-existing report, checkpoint, simulation file, note, criterion, figure or sealed artifact is written.

Pre-existing Git status, preserved exactly:

```text
 M EmilianoUPDATE.md
M  Simulations/Cphase/main.ipynb
M  Simulations/Cphase/workflow_funcs.py
?? EmilianoUPDATE-unsaved-backup.md
```

Only the new requested sandbox contains repository additions. No Stage-1/Stage-2 or isolated-block eigensolve, raw-wavefunction analysis, dressed-vector reconstruction, response/Schur contraction, whitening rerun, sealed benchmark execution, Git staging, commit or push was performed. NUMERICAL_VALIDATION.json and REPRODUCIBILITY.json record completed checks. OUTPUT_HASHES.json covers final deliverables.

V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
