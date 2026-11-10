# Figure validation and readability

**ORIGINAL FIGURE VISUAL QA: PASS.** These PDFs and PNGs are byte-identical copies of the inspected October 24 exports. No figures were regenerated for this GitHub package. Eight final PDFs were independently rendered with Ghostscript at 180 dpi and visually inspected. All eight PNG exports are 600 dpi. All figures are intended for 182.88-mm report/two-column width, with visible type at least 9 pt. Do not reduce the multi-panel layouts to a 90-mm single column.

| Figure | Vector PDF / fonts | PNG dimensions | Final render review |
| --- | --- | --- | --- |
| F01_retained_basis_tail | PASS; embedded TrueType, zero raster image objects | 4320 x 3600, 600 dpi | PASS |
| F02_rank_resolved_attribution | PASS; embedded TrueType, zero raster image objects | 4320 x 2760, 600 dpi | PASS |
| F03_S1_S2_selector_capture | PASS; embedded TrueType, zero raster image objects | 4320 x 2520, 600 dpi | PASS |
| F04_WB5_spectral_accuracy | PASS; embedded TrueType, zero raster image objects | 4320 x 2520, 600 dpi | PASS |
| F05_individual_errors_and_frozen_tradeoffs | PASS; embedded TrueType, zero raster image objects | 4320 x 4320, 600 dpi | PASS |
| F06_frequency_cancellation | PASS; embedded TrueType, zero raster image objects | 4320 x 2700, 600 dpi | PASS |
| F07_retained_space_partition | PASS; embedded TrueType, zero raster image objects | 4320 x 2460, 600 dpi | PASS |
| F08_coupling_and_state_fingerprints | PASS; embedded TrueType, zero raster image objects | 4320 x 2580, 600 dpi | PASS |

## Layout and rendering checks

No clipped text, overlapping annotations/legends, illegible tick labels or out-of-bounds content remains in the inspected final renders. Automated artist bounds report zero canvas violations after excluding tick objects outside their actual view limits. Each PNG has a nonzero white canvas margin. PDF text extraction succeeds; fonts are embedded and all plotted geometry is vector. Color/marker choices distinguish methods, while captions preserve sign, unit, base and finite-reference conventions.

The first pass revealed an overlapping attribution legend and overlong panel titles. Those were corrected; the attribution annotation was also moved away from the historical dominant marker. The heatmap and colorbar use vector geometry rather than embedded raster backgrounds. Final PDF/PNG and render hashes are recorded in FIGURE_VALIDATION_AND_READABILITY.json.

All nine historical carriers are shown in F01; all eleven target states and all nine carriers are retained in the F05 tradeoff comparison. Data for every budget/control remain available in data/FIGURE_DATA.json and its authenticated primary WB5 result records. F03 explicitly plots uncaptured attribution on a log scale and identifies its complement. No uncertainty error bars are invented for deterministic saved computations.

All 36 spectral increments were independently reproduced from the saved primary spectra. T/D/V shared-space metrics agree exactly, state-7 damage is included alongside states 4/5/8/9, and cancellation differences agree with their constituent errors. The complete frozen MIXED_OR_NOT_IMPROVED verdict remains visible in the report and F04.

**New eigensolves: ZERO.** No Hamiltonian assembly, new Schur contraction, selector refit, gate propagation, physical-model modification, or HOLD release occurred.
