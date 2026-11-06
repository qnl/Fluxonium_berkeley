# Source authentication and preservation

**SOURCE AUTHENTICATION: PASS. PRESERVATION: PASS.**

Configuration SHA-256: `8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`. Frozen diagnostic-plan SHA-256: `b7e9cfdec01e21be9dd30b7b14cf45bb9557290f57e0e4cb19616092ec974117`.

Direct authentication covers completed status, COMPLETE→checkpoint SHA linkage, canonical JSON configuration fingerprint, exact equality to sealed k380 configuration, all checkpoint payload hashes, original production source hashes, all 87 preregistration manifest pins, selection seal and selection-before-validation linkage, expected prefixes 300/320/340/360/380, and saved parent-quality thresholds. No historical authentication chain, benchmark evaluator or solver was rerun.

The previously authenticated diagnostic review is linked by fixed hashes of DIAGNOSTICS.json and SOURCE_HASHES.json. Historical rank 328 attribution and its cross-parent overlap are read from that record, explicitly retaining parent k340. k380 S1/S2 rank scalars come directly from the pinned k380_metrics.json. Local hash linkage is not a digital signature, independent timestamp or proof of model validity.

| Source | SHA-256 |
| --- | --- |
| Circuit_Objs/qchard_gridium_netlist.py | 85f6760ecf43ba2a44c638eadd0586393fb9e30dc8a28ac8f63f654dc23b6315 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/checkpoint.json | 399d3806566f7479a66227ff128bd37b80710f5ec44cbee49b27697df46101a8 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/projected_operators.npz | a01255f861cce247c677aecd64a5b746942032c4f82416aeb2f8843bf1219063 |
| research/checkpoints/four-mode-asymmetric-n71-epsj011-k380/stage1_eigensystem.npz | 8975937187c070097a8e0c7001671c31ae1027446c05dfc6e295beafba923164 |
| research/sandbox/epsj011_k380_diagnostic_review/DIAGNOSTICS.json | 07b4682611a6fb5468e4a1789675decc88331dc256b4664aaaf10801b64d59a7 |
| research/sandbox/epsj011_k380_preregistration/evaluation.json | 43903489aba94442dc868dae2eafa319de82f3cf3e3441db6144d6568c4a91aa |
| research/sandbox/epsj011_k380_preregistration/k380_metrics.json | aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e |

SOURCE_AUTHENTICATION.json lists every checked source pin. RESEARCH_PLAN.md freezes definitions; the same plan hash is included in numerical diagnostics and validation. Sources for representation: production `_charge_ops`, `_grid_ops`, `_coeffs`, `_assemble_nonlinear_sector.kron3`, `_solve` projection/coupling code; production reference runner `interpolate_raw_states` and checkpoint serialization. Units and coordinate statements are verified code observations, not new claims about the physical device.

PRESERVATION_BASELINE.json contains 456 protected file hashes: all tracked files, Git index, notes/backups, all existing checkpoints, figures, and existing k380 sandbox directories including originalv1, sealedv2 and prerequisite review/preregistration records. Final verification compares contents and original Git status. No new files appeared within those protected boundaries.

Pre-existing Git status (retained exactly):

```text
 M EmilianoUPDATE.md
M  Simulations/Cphase/main.ipynb
M  Simulations/Cphase/workflow_funcs.py
?? EmilianoUPDATE-unsaved-backup.md
```

Only the new requested diagnostic directory receives repository writes. No production simulation, Cphase file, EmilianoUPDATE.md, checkpoint, benchmark, prior research figure, Git index or frozen preregistration changed. No staging, commits or pushes. V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
