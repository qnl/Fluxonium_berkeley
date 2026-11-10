# Figure captions and primary source provenance

All eight figures are new saved-data visualizations. PDFs are vector exports with embedded fonts; PNGs are 600 dpi. Intended final width: 182.88 mm (two-column/report width). Keep that width or enlarge; do not reduce complex multi-panel figures to a 90-mm column. All visible type is at least 9 pt at intended width.

Every file hash and primary input path is in SOURCE_PROVENANCE.json. Figure-specific plotted values are in data/FIGURE_DATA.json and companion CSVs. No research arrays, Hamiltonians or new eigensolves are required. Error bars are omitted because these are deterministic finite-matrix results, not sampling uncertainty estimates.

## Figure 1

**Retained-basis tail.** (a) Largest absolute carrier increment over all nine registered carriers for each 20-rank nested-prefix addition: 0.714572, 1.299836, 5.594546, 6.738198 MHz. (b) Every signed carrier increment, defined as larger-k minus smaller-k, within the same authenticated k380 Stage-1 parent. The f78 increment magnitude decreases, grows, then decreases, while the largest high-frequency corrections grow. Neither a small f01/f78 change nor stable eigenvectors establishes a shrinking spectral tail. The frozen retained-basis verdict remains FAIL with D_NONMONOTONIC_UNRESOLVED.

Files: `figures/F01_retained_basis_tail.pdf` and `.png`.

Primary source locators: research/sandbox/epsj011_k380_preregistration/k380_metrics.json:/prefix_metrics; research/sandbox/epsj011_k380_diagnostic_review/SPECTRAL_DECOMPOSITION.csv:all 36 rows.

## Figure 2

**Rank-resolved virtual-state attribution.** Bars show W_r=sum over s={0,1,2,3,7,8} of |sigma_r,s|, in MHz, for S1/P340 and the WB4 S2/P340 endpoint. Rank 354 contributes 9.784721460 MHz (66.823079%) in S1; rank 362 contributes 11.181656938 MHz (88.687993%) in matched S2. Open circles separately show historical S2/P360 weights, whose rank-362 value is 15.138873587 MHz and total is 17.011414 MHz. The different bases must not be pooled. W_r is an aggregate absolute scalar partition, not a signed transition shift, rank-ablation result, or a single dressed shell mode.

Files: `figures/F02_rank_resolved_attribution.pdf` and `.png`.

Primary source locators: research/sandbox/k380_s2_final_authorization_execution_2026_10_24/NUMERICAL_SUMMARY.json:/S1_label_evidence/weights_MHz and /S2_rank_attribution; k380_metrics.json:/shell_diagnostics/S2/aggregate_rank_order and aggregate_rank_absolute_weight_MHz.

## Figure 3

**S1/S2 selector capture.** Shown is the uncaptured percentage, 100*(1-capture), on a logarithmic axis; lower is better. Both panels use selector base P340 and compatible attribution labels, but S1 ends at k360 and S2 ends at noncontiguous U_S2. S1 F capture at budgets 3/6/10/15 is 87.150372/98.058058/99.326400/99.946458%; S2 is 98.680013/99.289072/99.816194/99.912058%. D/V overlap in S1; T/D/V share their order in S2. F gains over these controls in S1, but S2 is tied at 3, lower at 6/15 and higher at 10. The historical original S2/P360 control-only comparison is not used to score the P340 selector. All 56 eligible method/budget capture rows, including energy/spread controls not drawn here, are exported in data/SELECTOR_CAPTURE.csv.

Files: `figures/F03_S1_S2_selector_capture.pdf` and `.png`.

Primary source locators: research/sandbox/k380_s2_final_authorization_execution_2026_10_24/NUMERICAL_SUMMARY.json:/S1_historical_rows and /S2_selector_and_controls; original S1 /TRANSFER_SUMMARY.json hash is linked in that summary.

## Figure 4

**WB5 actual spectral accuracy.** Maximum absolute excitation error over all eleven tracked states and maximum carrier error normalized by its frozen strict tolerance (1 MHz generally; 0.5 MHz for f78). F, T/D/V and E are shown at every registered budget. All comparisons use the finite U_S2 reference. F yields the lower worst-state error at budgets 6/10/15; T/D/V have the lower carrier envelope at 6/15. Identical spaces are deduplicated, including F=T=D=V at budget 3. The complete verdict is MIXED_OR_NOT_IMPROVED: an envelope-only improvement must not be substituted for the full frozen relative rule.

Files: `figures/F04_WB5_spectral_accuracy.pdf` and `.png`.

Primary source locators: research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_*_RESULT.json:/core_result/numerical/spectral and /worst_case_errors; RUN_RESULT.json:/relative_assessments.

## Figure 5

**Individual targets, carriers and the frozen no-harm failure.** At budget 15, panels (a) and (b) retain every mandatory state/carrier, including unfavorable F components. F has a smaller state-10 error but is worse on states 4,5,7,8,9 and several carriers; state 0 is zero by the excitation definition. Panel (c) shows the budget-3 mean-centered state-10 errors: F=0.215369845533, E=0.112044261889 MHz. The allowed E+slack bound is 0.212044261889 MHz. F exceeds it by 0.00332558364419 MHz, so the frozen E comparison is MIXED_OR_NOT_IMPROVED despite improved envelopes. Centering uses all eleven raw errors and does not replace raw or ground-referenced reporting.

Files: `figures/F05_individual_errors_and_frozen_tradeoffs.pdf` and `.png`.

Primary source locators: research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_17_RESULT.json and cell_18_RESULT.json:/core_result/numerical/spectral; cell_01_RESULT.json and cell_04_RESULT.json:/core_result/numerical/spectral/centered_state_error_MHz; frozen criteria /METRICS_CRITERIA.json.

## Figure 6

**Dressed-frequency cancellation.** Panel (a) compares signed ground-referenced excitation changes of states 7/8 with f78 for historical S1, historical S2 and WB4 isolated-S2 additions. Each case keeps its own base/endpoint and uses upper-minus-lower. WB4 gives -4.640170316 and -5.070939122 MHz constituent changes but only -0.430768806 MHz in f78. Panel (b) shows all five WB5 cancellation-flagged method/budget cases, using selected-minus-U_S2; S=spread, R=spread_reverse. Their approximately 4-5 MHz constituent errors are not concealed by the roughly 0.55-0.60 MHz f78 errors. No physics/bare F/T/D/V case is flagged. A flagged small difference cannot rescue a favorable verdict.

Files: `figures/F06_frequency_cancellation.pdf` and `.png`.

Primary source locators: research/sandbox/epsj011_k380_diagnostic_review/SPECTRAL_DECOMPOSITION.csv:f78 rows; WB4 NUMERICAL_SUMMARY.json:/eigenenergies_GHz; WB5 cell_*_RESULT.json:/core_result/numerical/spectral/cancellation.

## Figure 7

**Retained-space partition.** Exact interval definitions read from the WB4 run contract: P340=[0,340), excluded ranks [340,360), Q_S2=[360,380), U_S2=P340 union Q_S2. Panel (a) preserves the full rank scale; panel (b) magnifies the boundary. Eight mode-4 oscillator states per retained Stage-1 rank give dimensions 2720, 160 and 2880. U_S2 has 360 retained ranks but is not the contiguous k360 space. This is an authenticated support diagram, not newly computed numerical physics.

Files: `figures/F07_retained_space_partition.pdf` and `.png`.

Primary source locators: research/sandbox/k380_p340_s2_base_matched_validation_continuation_001/RUN_RESULT.json:/partitions; research/sandbox/k380_selected_space_spectral_accuracy_2026_10_24/FROZEN_COMPARISONS.json:/unique_jobs.

## Figure 8

**Coupling and finite-grid state structure.** Panel (a) includes all 40 S1/S2 ranks with saved bare T300=sum_{p<300} ||H_pr||_F^2, covering all eight oscillator states, against historical absolute Schur weight. Historical S1 uses P340 and S2 uses P360; these are explanatory within-shell observations, not matched S2 transfer labels. Bare coupling identifies both shell leaders yet does not reproduce their much stronger relative target response. Panel (b) includes all 80 saved ranks 300-379; focal ranks 328/354/362 have low outer-charge support and unusually narrow theta2 spread. This does not assign a normal mode, prove cutoff safety, or test a changed spatial grid. Rank 328 attribution belongs to a separate historical k340 calculation; only its saved k380 fingerprint appears here.

Files: `figures/F08_coupling_and_state_fingerprints.pdf` and `.png`.

Primary source locators: research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/FINGERPRINTS.csv:/p300_block_strength_GHz2, schur_absolute_weight_MHz, charge_edge, theta2_std; research/sandbox/k380_coupling_vs_schur_2026_10_21/RESULTS.json:/ranking_analysis.

## Claim boundary

These figures support finite-parent numerical observations and retrospective selector comparisons. They do not establish general selector superiority, omitted-tail bounds, spatial/retained-basis convergence, a physical drive calibration, a four-mode asymmetric gate or an end-to-end speedup. Frozen WB5 verdict: MIXED_OR_NOT_IMPROVED. Schur-v2 prelaunch: HOLD. Gate work: HOLD. Spatial convergence: HOLD.
