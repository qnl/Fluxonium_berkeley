# Gridium research handoff for Thomas

This is the review entry point for the historical four-mode numerical studies.
The publication prepared on November 6, 2026 adds saved results and provenance;
it performs no scientific eigensolves, gate optimization, or figure regeneration.

**Retained-basis convergence: FAIL.** The k340/k380 tail classification remains
**D_NONMONOTONIC_UNRESOLVED**. Selector validation remains **BORDERLINE**,
multi-target transfer **PARTIAL**, and the cross-parameter shell mechanism
**SUPPORTED**. The complete WB5 verdict is **MIXED_OR_NOT_IMPROVED**.
**Gate work, spatial convergence, and Schur-v2 prelaunch remain HOLD.**
No validated asymmetric four-mode gate is claimed.

These are results of the previous finite Hamiltonian and recorded parameter
settings. Thomas's pending physics/parameter changes require a separate review;
the archived results are not predictions for the intended physical device.

## Start here

1. [October 24 executive summary](../docs/gridium-research-update-2026-10/work-blocks-1-5/THOMAS_EXECUTIVE_SUMMARY.md)
   and [illustrated report](../docs/gridium-research-update-2026-10/work-blocks-1-5/THOMAS_RESEARCH_UPDATE.md).
   Both were already on the upstream research branch before this publication.
2. [October 29 research/stoppage update](../EmilianoUPDATE.md#work-block--2026-10-252026-10-29)
   and [three figures with supporting scalar data](../docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/).
   These were also already published. k400/k420 scientific results do not exist.
3. [Exact publication file list](publication/RELEASE_FILES.txt),
   [file inventory and hashes](publication/INVENTORY.csv), and
   [publication audit](publication/PUBLICATION_REVIEW.md).

## Scientific results and supporting data

| Topic | Review material | Machine-readable evidence |
| --- | --- | --- |
| k180-k380 parents | [Checkpoint/configuration index](publication/CHECKPOINTS.md) | Seven original `analysis.json` summaries and eight `PUBLICATION_METADATA.json` configuration companions, linked in the index |
| k260/k300 convergence and selection | [k260 shell attribution](sandbox/k260_shell_attribution/REPORT.md), [rank addback](sandbox/k260_rank_addback/REPORT.md), [k300 results](sandbox/k300_results/REPORT.md) | [k300 classification](sandbox/k300_results/classification.json), [adjacent changes](sandbox/k300_results/adjacent_changes.csv), [metrics](sandbox/k300_results/validation_metrics.json) |
| Independent parameter point | [Selection provenance](sandbox/parameter_point_validation/selector_provenance.md) | [Frozen evaluation](sandbox/parameter_point_validation/evaluation.json), [carrier comparison](sandbox/parameter_point_validation/epsj011_vs_epsj010_carriers.csv) |
| k340/k380 convergence | [October 20 synthesis](../docs/gridium-research-update-2026-10/WORK_BLOCK_2026-10-20.md) | [k340 evaluation](sandbox/epsj011_k340_preregistration/evaluation.json), [k340 metrics](sandbox/epsj011_k340_preregistration/k340_metrics.json), [k380 evaluation](sandbox/epsj011_k380_preregistration/evaluation.json), [k380 metrics](sandbox/epsj011_k380_preregistration/k380_metrics.json) |
| High-rank Schur/Feshbach attribution | [k300 attribution](sandbox/k300_shell_attribution/REPORT.md), [cross-parameter attribution](sandbox/epsj011_shell_attribution/REPORT.md), [k380 diagnostic](sandbox/epsj011_k380_diagnostic_review/REPORT.md) | [Rank attribution](sandbox/k300_shell_attribution/rank_attributions.csv), [cross-parameter comparison](sandbox/epsj011_shell_attribution/cross_parameter_comparison.csv), [spectral decomposition](sandbox/epsj011_k380_diagnostic_review/SPECTRAL_DECOMPOSITION.csv) |
| WB1 coupling versus attribution | [Report, portable copy](sandbox/k380_coupling_vs_schur_2026_10_21/REPORT.public.md) | [Results](sandbox/k380_coupling_vs_schur_2026_10_21/RESULTS.json), [target diagnostics](sandbox/k380_coupling_vs_schur_2026_10_21/TARGET_DIAGNOSTICS.csv) |
| WB2 high-rank fingerprints | [Report, portable copy](sandbox/k380_high_rank_state_fingerprints_2026_10_21/REPORT.public.md) | [Fingerprints](sandbox/k380_high_rank_state_fingerprints_2026_10_21/FINGERPRINTS.csv), [operator couplings](sandbox/k380_high_rank_state_fingerprints_2026_10_21/OPERATOR_COUPLINGS.csv) |
| WB3 retrospective transfer | [Report, portable copy](sandbox/k380_cross_shell_selector_transfer_2026_10_21/SCIENTIFIC_REPORT.public.md) | [Transfer results](sandbox/k380_cross_shell_selector_transfer_2026_10_21/TRANSFER_RESULTS.csv), [summary](sandbox/k380_cross_shell_selector_transfer_2026_10_21/TRANSFER_SUMMARY.json) |
| WB4 matched-P340 S2 attribution | [Scientific results](sandbox/k380_s2_final_authorization_execution_2026_10_24/SCIENTIFIC_RESULTS.public.md) | [Numerical summary, portable copy](sandbox/k380_s2_final_authorization_execution_2026_10_24/NUMERICAL_SUMMARY.public.json), [S2 attribution](sandbox/k380_p340_s2_base_matched_validation_continuation_001/S2_ATTRIBUTION.csv) |
| WB5 actual selected-space spectral accuracy | [Complete scientific results, portable copy](sandbox/k380_wb5_final_authorization_execution_2026_10_24/SCIENTIFIC_RESULTS.public.md) | [Comparison summary](sandbox/k380_wb5_final_authorization_execution_2026_10_24/COMPARISON_SUMMARY.csv), [target errors](sandbox/k380_wb5_final_authorization_execution_2026_10_24/TARGET_ERRORS.csv), [carriers/cancellation](sandbox/k380_wb5_final_authorization_execution_2026_10_24/CARRIERS_AND_CANCELLATION.csv), [numerical summary](sandbox/k380_wb5_final_authorization_execution_2026_10_24/NUMERICAL_SUMMARY.public.json) |
| Eigenstate/subspace tracking | [Method and saved-data report](sandbox/subspace_tracking/REPORT.md), [retained-basis re-audit](sandbox/k_convergence_reaudit/REPORT.md) | [Assignments](sandbox/subspace_tracking/state_assignments.csv), [operator comparisons](sandbox/subspace_tracking/operator_comparisons.csv), [convergence table](sandbox/k_convergence_reaudit/low_energy_convergence.csv) |
| Saved transition and signed control matrices | [Export definitions](publication/LOW_ENERGY_EXPORTS.md) | [N51 k180](publication/low_energy/N51_k180.json), [N71 k180](publication/low_energy/N71_k180.json), [other retained masks](publication/low_energy/) |
| Scientific figures | [Six earlier figures](../docs/gridium-research-update-2026-10/FIGURE_CAPTIONS.md), [October 24 figure provenance](../docs/gridium-research-update-2026-10/work-blocks-1-5/FIGURE_CAPTIONS_AND_PROVENANCE.md) | [Original plotting CSVs](sandbox/gridium_figures_2026_10/data/), [October 24 figure data](../docs/gridium-research-update-2026-10/work-blocks-1-5/data/), [October 29 figure data](../docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/figure_data.json) |

The original historical paths are retained where the files are suitable for Git.
Files named `.public.md` or `.public.json` are new publication copies replacing
local absolute paths with repository-relative paths or `python`. One copy also
links its portable numerical companion, and the predictor-definition copy formats
three algebra expressions as code to prevent accidental Markdown links. Original
files remain untouched and excluded. Their original hashes and the exact
transformation are in [MANIFEST.json](publication/MANIFEST.json). JSON numerical
fields were checked unchanged. No plots were regenerated.

## Interpretation and historical status

- Completed numerical results are governed by the saved final evaluations and
  their finite-parent scope. Numerical solver quality is distinct from convergence.
- Files called `REPORT.md` within preregistration directories may say READY or
  "no result exists": those are pre-execution snapshots. The later `evaluation.json`
  files and dated scientific handoffs give the completed outcome.
- WB3's historical S2/P360 mismatch remains documented; WB4's matched-P340
  experiment resolves that particular comparison. Neither upgrades full convergence.
- [State characterization](sandbox/state_0178_checkpoint_diagnostics.json),
  [pathway comparison](sandbox/robust_pathway_audit.json), and
  [model-threat analysis](sandbox/model_threat_audit.md) are bounded diagnostics.
  [Array-mode modeling](sandbox/array_mode_extension_audit.md) is a proposal
  requiring device-specific inputs, not an implemented extra physical mode.
- [Cross-study qualifications](sandbox/cross_study_audit/REPORT.md) preserve
  distinctions between pathway products, control metrics, frames, and leakage.
  Its driven/controllability background references are not new gate validation.
- Synthetic engineering tests, host-admission logs, authorization records,
  caches, local patches, and inactive k420 integration code are excluded.
  Historical mentions of method tests are not new numerical-test results.

## Reproducibility boundary

The reports, figures, and numerical summaries can be inspected from this branch.
Full reconstruction still needs external raw arrays. See the exact
[raw-artifact locations, sizes, and SHA-256 values](publication/RAW_ARTIFACTS.csv)
and [retrieval instructions](publication/RETRIEVAL.md). No download location is
invented and no complete fresh-clone numerical reproduction is claimed.

[Dependencies](publication/DEPENDENCIES.csv) identify published targets,
portable-copy aliases, externally required arrays, excluded historical records,
and references absent even from the local archive. Historical hash manifests
can reference excluded files; publishing a manifest does not publish its members.

Source references include the [netlist model](../Circuit_Objs/qchard_gridium_netlist.py),
[retained-basis audit](../Simulations/Rabi_3Photon/four_mode_retained_basis_audit.py),
[N71 runner](../Simulations/Rabi_3Photon/run_four_mode_spatial71_reference.py),
and [portable environment specification](../environment-portable.yml).
Historical source hashes and dirty-generation status are retained in checkpoint
companions; current source may differ from older generation versions.

Verify the publication, without importing simulation code:

```sh
python research/publication/verify_publication.py
```

This checks Git tracking, file hashes/sizes, Markdown paths, and documented
external dependencies. It performs no eigensolves or figure generation.
