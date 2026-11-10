# Rabi_3Photon validation artifacts

This directory contains durable evidence from frozen, serious
`IdealGridium` X90 and X180 validation runs. The control model couples all
tones through the abstract global phase coordinate `IdealGridium.phi()`.
The historical YAML value `drive_type: flux` does not identify a calibrated
experimental flux line.

For each target gate:

- `*_state_populations.png` shows energy-eigenstate populations for logical
  inputs `|0>` and `|1>`, including the `0-5-4-1` pathway levels.
- `*_intermediate_populations.png` expands the vertical scale for `|4>`,
  `|5>`, and the other nonlogical retained levels with the largest transient
  populations in the simulated trajectories.
- `*_leakage.png` shows state-resolved and logical-input-average leakage.
- `*_drive_traces.png` shows the three actual scalar coefficients passed to
  the time-dependent solver, including amplitude scaling.
- `*_convergence.png` summarizes solver/sampling, retained-level, and LC-cutoff
  validation using absolute gate infidelity, `1 - F_gate`, alongside leakage.
- `*_summary.csv` is a one-row reproducibility and Pareto-analysis record.
- `*_timeseries.csv` contains populations, leakage, and drive coefficients at
  every saved time.
- `*_convergence.csv` contains the numerical convergence table.

`validated_gate_runs.csv` combines the per-gate summary rows so fidelity,
duration, and leakage tradeoffs can be plotted without rerunning simulations.

Regenerate the frozen artifacts from the repository root with:

```shell
python -m Simulations.Rabi_3Photon.generate_validation_artifacts
```

The generator reruns validation from the frozen experiment YAMLs; it does not
re-optimize either gate.

Regenerate only the phase-sensitive fidelity-versus-time audit with:

```shell
python -m Simulations.Rabi_3Photon.generate_fidelity_vs_time_artifacts
```

That command evaluates the existing frozen pulses at the validated
8-samples/ns output grid and at a denser 32-samples/ns grid. It does not run a
search or alter either candidate.

## Output manifest

This table distinguishes Git-visible conclusions from raw or derived evidence
that currently exists only on the machine where it was generated. Paths are
relative to the repository root unless an absolute path is shown.

| Result | Exact path | Git visibility | Scientific question answered |
|---|---|---|---|
| Frozen one-mode X90 validation | All under `Figures/Rabi_3Photon/`: `x90_summary.csv`, `x90_timeseries.csv`, `x90_convergence.csv`, `x90_state_populations.png`, `x90_intermediate_populations.png`, `x90_leakage.png`, `x90_drive_traces.png`, `x90_convergence.png` | Tracked | What fidelity, leakage, pulse, and numerical-convergence evidence supports the symmetric `IdealGridium` X90 candidate? |
| Frozen one-mode X180 validation | All under `Figures/Rabi_3Photon/`: `x180_summary.csv`, `x180_timeseries.csv`, `x180_convergence.csv`, `x180_state_populations.png`, `x180_intermediate_populations.png`, `x180_leakage.png`, `x180_drive_traces.png`, `x180_convergence.png` | Tracked | What fidelity, leakage, pulse, and numerical-convergence evidence supports the symmetric `IdealGridium` X180 candidate? |
| One-mode peak-versus-final fidelity audit | All under `Figures/Rabi_3Photon/`: `gate_fidelity_peak_summary.csv`, `x90_fidelity_vs_time.csv`, `x90_fidelity_vs_time.png`, `x180_fidelity_vs_time.csv`, `x180_fidelity_vs_time.png` | Untracked and not ignored; intended for this evidence commit; generated without changing the frozen pulses | Does either validated pulse reach a better phase-sensitive logical fidelity before its nominal final time, and is the sampled peak stable from 8 to 32 samples/ns? |
| Asymmetric N=51 Stage-1 reference | All under `research/checkpoints/2026-09-26-asymmetric-n51/`: `checkpoint.json`, `projected_operators.npz`, `stage1_eigensystem.npz` | Local/ignored; Thomas cannot access the raw files from GitHub | What is the N51 `k=180` finite-grid reference eigensystem and projected-operator dataset? |
| N=51 retained-basis audit | Both under `Figures/Rabi_3Photon/`: `four_mode_asymmetric_shell_basis_audit.json`, `four_mode_asymmetric_shell_basis_audit.png` | Tracked derived evidence | How do the asymmetric `k=160/170/172/175/177/179` retained spaces compare with the same-grid `k=180` endpoint? |
| Rejected inherited N=51 shell | All under `Figures/Rabi_3Photon/`: `four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected.json`, `four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected_summary.csv`, `four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected_pathways.png` | Tracked derived evidence | Does a retained shell inherited from the symmetric model remain accurate for the asymmetric model? It does not. |
| Asymmetric N=71 Stage-1 reference | All under `research/checkpoints/four-mode-asymmetric-n71-k180/`: `COMPLETE`, `checkpoint.json`, `run_state.json`, `stage1_eigensystem.npz`, `projected_operators.npz`, `analysis.json` | Local/ignored; Thomas cannot access the raw files or analysis from GitHub | What is the authenticated N71 `k=180` reference, and what are its solver/residual diagnostics? |
| N51-to-N71 state tracking and spatial comparison | `research/checkpoints/four-mode-asymmetric-n71-k180/analysis.json`, key `n51_to_n71_spatial_comparison` | Local/ignored; Thomas cannot access it from GitHub | Do states 7 and 8 remain identifiable, how much do `f07`/`f08` and the three `grid_phi` pathway elements shift, and does pathway ordering survive? |
| Common retained-basis analysis, including common `k=172` | `research/sandbox/n71_retained_subspace_postprocess.json`, generated by `research/sandbox/gridium_n71_retained_postprocess.py` | Local/ignored; Thomas cannot access it from GitHub | Can one common retained-space construction reproduce both N51 and N71 references, including the `0-7-8-1` observables? |
| Reduced four-mode control-plumbing diagnostic | All under `Figures/Rabi_3Photon/`: `four_mode_grid_phi_spectrum_reduced_diagnostic.csv`, `four_mode_grid_phi_matrix_reduced_diagnostic.csv`, `four_mode_grid_phi_matrix_heatmap_reduced_diagnostic.png`, `four_mode_three_tone_smoke_drive_traces_reduced_diagnostic.png`, `four_mode_three_tone_smoke_population_leakage_reduced_diagnostic.png` | Tracked, but deliberately reduced-model diagnostic only | Are `grid_phi`, drive normalization, multitone assembly, and propagation plumbing operational? |
| Recent unoptimized four-mode X90 pulse-transplant diagnostic | All under `/private/tmp/gridium_four_mode_x90_smoke_20260927/`: `all_run_summaries.json`, `full_three_tone_nlev24.{json,npz}`, `full_three_tone_nlev48.{json,npz}`, `middle_only_nlev24.{json,npz}`, `middle_only_nlev48.{json,npz}`, `model_and_zero_drive.json` | Local only and outside Git; Thomas cannot access it from GitHub | What happens when the one-mode X90 amplitudes/duration are transferred without four-mode optimization, and is the low fidelity stable against final-level expansion? |

### Small derived evidence that should be promoted

Do not copy the large Stage-1 eigensystems into Git. Instead, generate and
review small tracked derivatives containing their source-checkpoint hashes:

- an N51-to-N71 state-tracking CSV/JSON with state assignments, overlaps,
  `f07`/`f08` shifts, the three pathway-element changes, and pathway ranks;
- an N51/N71 common-retained-basis CSV/JSON covering `k=170`, `172`, `175`,
  `177`, and `179`, with state-7/8 overlaps and transition/matrix-element
  errors;
- a compact N71 solver/provenance JSON with the frozen configuration hash,
  code revision, matrix size, LU statistics, residuals, and raw-file hashes;
- a compact four-mode pulse-diagnostic summary CSV/JSON and research-facing
  fidelity/leakage figures, labeled as an unoptimized pulse transplant rather
  than a gate result.

Until those derivatives are promoted, conclusions that depend on the ignored
N71 checkpoint, common-basis postprocessing, or `/private/tmp` diagnostic must
be accompanied by an explicit local-only qualification.
