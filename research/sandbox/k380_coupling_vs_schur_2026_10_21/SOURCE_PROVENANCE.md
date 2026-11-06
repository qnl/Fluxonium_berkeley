# Source provenance and preservation

**SOURCE AUTHENTICATION: PASS. PRESERVATION: PASS.**

Configuration SHA-256: `8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`.
Frozen plan SHA-256: `2d0653be74e185b71e293ef279b2269540379d923f7b84c6b2940d93000565ed`.
Frozen definitions SHA-256: `da91ade67881fcb7b0490c61b479370d0bc6a82fa38c9ffdad0455961fb4146b`.

The completed fingerprint package was consumed without rerunning analyze.py or its456-file audit. Every artifact in its OUTPUT_HASHES.json was checked. Direct authentication checks k380 checkpoint metadata/COMPLETE linkage, canonical configuration equality to sealed criteria, payload bytes, relevant production/formula source pins and selection seal. The already authenticated prior source chain is reused; unrelated historical audits and evaluators were not repeated.

| Input | Role | SHA-256 |
| --- | --- | --- |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/checkpoint.json | Completed numerical configuration | 399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/stage1_eigensystem.npz | Only eigenvalues member loaded | 8975937187c070097a8e0c7001671c31ae1027446c05dfc6e295beafba923164 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/projected_operators.npz | Five complex380×380 operator projections | a01255f861cce247c677aecd64a5b746942032c4f82416aeb2f8843bf1219063 |
| research/sandbox/epsj011_k380_preregistration/k380_metrics.json | Exact saved S1/S2 rank attribution and target norms | aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e |
| research/sandbox/epsj011_k380_preregistration/selection.json | Previously saved baseP340 target norms and raw responses | 6720cb7b2213962bf490ec6096ebd88e0b66e79ec7a2c25dee33116fc02eb8b7 |
| research/sandbox/epsj011_k380_diagnostic_review/DIAGNOSTICS.json | Rank328 historical provenance/context | 07b4682611a6fb5468e4a1789675decc88331dc256b4664aaaf10801b64d59a7 |
| research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/DIAGNOSTICS.json | Completed fingerprints/circuit coefficients | 30cd6bb7d81198fcbceef6b0436e35d658cd099e8995e6e367e682a19a401eba |
| research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/OUTPUT_HASHES.json | Prior package integrity root | 95a709835abbb06bd87df14d15c50aba002b6c058357dc2cb99176282ca3bce2 |

Authoritative formula sources: Circuit_Objs/qchard_gridium_netlist.py for the circuit A*n4+B*theta4 block; research/sandbox/k300_preregistration/selector.py for saved target groups, base coupling norm and raw positive isolated-rank response; epsj011_k380_preregistration/analyze_k380.py for exact scalar attribution and ||d_r||/sqrt(p†p) coupling normalization. These definitions were inspected, never executed to form new dressed data.

S1 and S2 share one k380 Stage-1 parent but have different exact partitions P340/P360. Saved selector responses for both use P340. Historical rank328 belongs to k340/new_S4, P320, with a separate selector P300. No incompatible parents or partitions are silently pooled. All ranks are zero-based, energies GHz H/h and absolute attribution MHz. Complex sums precede norms.

SOURCE_AUTHENTICATION.json records 104 targeted hashes, checked before and after analysis. This targeted set includes consumed inputs, previous fingerprint/review reports, protected user edits, Git index and sealed artifacts/manifests/reports. It is not a repetition of the previous456-file broad audit. All hashes and Git status were unchanged.

Pre-existing Git status, preserved exactly:

```text
 M EmilianoUPDATE.md
M  Simulations/Cphase/main.ipynb
M  Simulations/Cphase/workflow_funcs.py
?? EmilianoUPDATE-unsaved-backup.md
```

Final repository additions are confined to this new sandbox directory. A new implicit Rplots.pdf created at the repository root by the failed Cairo export was detected and removed during final validation; no prior file was overwritten. The plotting script now fails closed on implicit graphics devices. Existing reports, checkpoints, production, Cphase, notes, figures and sealed v1/v2 criteria remain unchanged. No staging, commits, pushes, Stage-1/2 eigensolves, dressed-vector reconstruction or v2 benchmark execution occurred. SHA-256 linkage provides local integrity, not a digital signature or independent physics validation.

REPRODUCIBILITY.json records byte-identical CSV/JSON outputs across two processes. RUN_VALIDATION.json lists synthetic and numerical checks and resource use. PDFs/PNGs are visualization artifacts; their export metadata need not be byte-identical. OUTPUT_HASHES.json provides final deliverable integrity.

V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
