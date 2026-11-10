# Gridium Work Blocks 1-5 - October 24 research update

[Illustrated report](THOMAS_RESEARCH_UPDATE.md) | [Executive summary](THOMAS_EXECUTIVE_SUMMARY.md) | [Message draft for Thomas](THOMAS_MESSAGE.md)

Earlier context: [October 20 research work block](../WORK_BLOCK_2026-10-20.md) and [research-update overview](../README.md). Both earlier documents and their figures remain unchanged.

**Frozen scientific verdict: MIXED_OR_NOT_IMPROVED.** The physics-selected spaces accurately reproduce a finite U_S2 reference. That does not establish consistent superiority over bare coupling, physical or retained-basis convergence, an end-to-end speedup, or an asymmetric four-mode gate.

**Proposed next step:** [independent finite-parent transfer validation](NEXT_TRANSFER_EXPERIMENT.md), requiring Thomas's feedback and separate scientific/execution authorization. No experiment is authorized by this package. Schur-v2 prelaunch, gate work and spatial convergence remain **HOLD**.

## Figures

All eight vector PDFs and their original 600-dpi PNGs are included without downsampling. Intended figure width: 182.88 mm (report/two-column width).

| Figure | Vector PDF | 600-dpi PNG |
| --- | --- | --- |
| F01_retained_basis_tail | [PDF](figures/F01_retained_basis_tail.pdf) | [PNG](figures/F01_retained_basis_tail.png) |
| F02_rank_resolved_attribution | [PDF](figures/F02_rank_resolved_attribution.pdf) | [PNG](figures/F02_rank_resolved_attribution.png) |
| F03_S1_S2_selector_capture | [PDF](figures/F03_S1_S2_selector_capture.pdf) | [PNG](figures/F03_S1_S2_selector_capture.png) |
| F04_WB5_spectral_accuracy | [PDF](figures/F04_WB5_spectral_accuracy.pdf) | [PNG](figures/F04_WB5_spectral_accuracy.png) |
| F05_individual_errors_and_frozen_tradeoffs | [PDF](figures/F05_individual_errors_and_frozen_tradeoffs.pdf) | [PNG](figures/F05_individual_errors_and_frozen_tradeoffs.png) |
| F06_frequency_cancellation | [PDF](figures/F06_frequency_cancellation.pdf) | [PNG](figures/F06_frequency_cancellation.png) |
| F07_retained_space_partition | [PDF](figures/F07_retained_space_partition.pdf) | [PNG](figures/F07_retained_space_partition.png) |
| F08_coupling_and_state_fingerprints | [PDF](figures/F08_coupling_and_state_fingerprints.pdf) | [PNG](figures/F08_coupling_and_state_fingerprints.png) |

## Supporting evidence

- [Figure captions and source locators](FIGURE_CAPTIONS_AND_PROVENANCE.md)
- [Numerical evidence and limitations](NUMERICAL_EVIDENCE_AND_LIMITATIONS.md) ([CSV](NUMERICAL_EVIDENCE_AND_LIMITATIONS.csv))
- [Source provenance](SOURCE_PROVENANCE.md) ([recorded hashes](SOURCE_PROVENANCE.json))
- [Original figure readability report](FIGURE_VALIDATION_AND_READABILITY.md) ([figure hashes](FIGURE_VALIDATION_AND_READABILITY.json))
- [Authenticated figure data](data/FIGURE_DATA.json), [spectral increments](data/SPECTRAL_INCREMENTS.csv), [selector capture](data/SELECTOR_CAPTURE.csv), [S1 attribution](data/ATTRIBUTION_S1.csv), [S2 attribution](data/ATTRIBUTION_S2.csv), [fingerprints](data/FINGERPRINT_SUMMARY.csv)
- [Original evidence authentication](EVIDENCE_AUTHENTICATION.json), [source-package manifest](SOURCE_PACKAGE_MANIFEST.json), and [this package manifest](PACKAGE_MANIFEST.json)

The source-package manifest records the complete original local handoff. It is provenance, not a claim that its preparation scripts and temporary QA renders are included in this review copy. Raw scientific arrays and execution-approval records are not bundled. No new eigensolve, calculation, figure regeneration, or change to production/checkpoint files was needed to prepare this copy.
