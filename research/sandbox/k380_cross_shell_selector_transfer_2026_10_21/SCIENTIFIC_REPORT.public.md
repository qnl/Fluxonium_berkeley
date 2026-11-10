> Publication copy: local machine paths have been replaced by repository-relative paths or `python`. Scientific statements and values are unchanged. The original file and its SHA-256 are recorded in [the publication manifest](../../publication/MANIFEST.json).

# Retrospective cross-shell selector transfer — 2026-10-21

**SOURCE AUTHENTICATION: PASS. TRANSFER FEASIBILITY: PARTIAL. FEATURE LEAKAGE: LIMITATIONS. BASE-COMPATIBLE LABELS: PARTIAL. RETROSPECTIVE TRANSFER: INCOMPLETE.**

The historically frozen selector can be evaluated on S1 without generating a ranking from its attribution labels. Its recorded P340 matrix-greedy ordering performs better than the fixed bare/energy controls under the retrospective S1 rank-capture metrics below. Cross-shell transfer to S2 cannot be established: the frozen S2 selector used P340, while saved exact S2 attribution uses P360. We do not score that mismatched selector/label pair.

No Stage-1/Stage-2 eigensolve, isolated-block eigensolve, whitening rerun, dressed-vector reconstruction, new Schur contraction or sealed v2 benchmark execution occurred. Previous fingerprint and correlation investigations were not repeated. The four bare-control methods’ existing correlations are reused directly for both shells; only new recorded-order evaluations and budget captures were calculated.

## Exact historical algorithm and freeze evidence

The method/source were frozen on October 8; the October 18 k380 preregistration pins those exact hashes. The target pairs are (0, 1),(2, 3),(7, 8) among eleven tracked base states. The base is P340 for both candidate shells. Budgets are 3, 6, 10, 15; the historical primary/confirmation budgets are 10/15 and control-operator weight is zero. These are target-state groups, not a drive-transition or gate protocol.

For each rank’s eight-state oscillator block, the historical rule forms the full complex positive 2×2 response A_rg=B_rg†(D_r-mu_g I)^-1 B_rg. It whitens each target-pair response by the entire candidate shell’s W_g=sum_r A_rg, floors whitening eigenvalues at max(1e-14 GHz,1e-10 lambda_max), and greedily maximizes (1/6)sum_g logdet[I2+6 sum_selected C_rg]. Gain ties within 1e-12 choose the smaller rank. All five production coupling channels enter coherently. Isolated denominators must exceed 0.010 GHz and eta must remain below 0.10; those limits are unchanged.

This is a balanced matrix-coverage rule. It is not a descending raw trace, BaseC, ExactC or aggregate absolute attribution ranking. Shell-dependent whitening uses candidate features; it does not use candidate attribution outcomes. No scalar alternative from Work Block 2 was substituted. TRANSFER_PROTOCOL.md records the full equations, all PSD/gap checks and fixed controls.

| Shell | Saved minimum isolated denominator, GHz | Saved maximum eta | Whitening directions floored |
| --- | --- | --- | --- |
| S1 | 41.125903107 | 0.005490831 | 0 |
| S2 | 42.210102873 | 0.006095719 | 0 |

The saved numerical floor values satisfy the unchanged historical formula. This checks persisted metadata; no isolated response or whitening matrix was reconstructed.

The pinned selection hook writes and seals both shell orders before entering analyze_parent, whose first substantive step is the full new-parent Stage-2 reference. Later analysis produces Schur labels. Checkpoint analysis links the same selection/seal hashes and records selection-before-validation, with candidate/full-new-shell results and old rank lists explicitly excluded. This is affirmative local source/artifact evidence of pre-label selection. It is not an independently timestamped or signed execution record.

In the present script, only sealed orders and whitelisted bare columns reach the ranking producer. RANKINGS_FROZEN.json is written before Schur labels are parsed. Exact target norms/attributions and raw response-trace sorting are excluded. See FEATURE_LEAKAGE_AUDIT.md for classifications A–E and limitations.

## Compatibility: the decisive limitation

| Case | Selector/base | Exact saved labels | Evaluation |
| --- | --- | --- | --- |
| S1 | P340,Q340–359 | P340,Q340–359, full k360 | Compatible |
| S2 frozen selector | P340,Q360–379 | P360,Q360–379, full k380 | INCOMPLETE: different P and full endpoint |
| S2 refreshed selector | P360 features/order not saved | P360,Q360–379 | INCOMPLETE: no corresponding frozen ranking |

The isolated S2 selector endpoint retains [0, 340) union[360, 380), omitting 340–359. It is not the contiguous k360 base or the full k380 endpoint. The prior targeted dependency review found its complete historical endpoint metric/assignment record missing; this run neither re-searches broadly nor reconstructs it. Historical selected-space comparison summaries do exist, but they supply neither an independent complete endpoint record nor exact P340-based per-rank Schur labels. Even recovering endpoint energies alone would not recover the required rank contractions.

For a same-base comparison such as S1, base-only approximate selector targets and the actual full-endpoint P directions are expected to differ; this is not leakage. S2 changes the retained base itself, which is a different comparison. SOURCE_COMPATIBILITY.md documents that distinction.

## S1 performance: valid retrospective comparison

All ranks 340–359 are included in the evaluation. W_r is the sum over targets 0, 1, 2, 3, 7, 8 of the absolute saved rank scalar contributions in the full P340+Q endpoint. The total is 14.6427277352016 MHz (binary64 source precision retained in CSV). Captured fractions describe this endpoint’s attribution partition, not the energy change caused by retaining the selected subset alone.

| Budget | Frozen physics additions, in selection order | Top-k recovery | Top3 recovery | Absolute attribution captured | Rank354 included |
| --- | --- | --- | --- | --- | --- |
| 3 | 354, 344, 343 | 3/3 | 3/3 | 87.150372% | yes |
| 6 | 354, 344, 343, 341, 353, 345 | 5/6 | 3/3 | 98.058058% | yes |
| 10 | 354, 344, 343, 341, 353, 345, 347, 352, 350, 342 | 9/10 | 3/3 | 99.326400% | yes |
| 15 | 354, 344, 343, 341, 353, 345, 347, 352, 350, 342, 359, 357, 351, 348, 340 | 15/15 | 3/3 | 99.946458% | yes |

Top-k means k equals the selection budget, with lower Stage-1 rank breaking equal-attribution ties. Top3 is also reported at every budget. The frozen first three ranks 354, 344, 343 are exactly the three leading S1 attribution ranks. The full-order Spearman is 0.960902.

### Every fixed S1 control

| Method | Spearman | Captured %, b3 / b6 / b10 / b15 | Top-k recovery counts | Leader included, b3 / b6 / b10 / b15 |
| --- | --- | --- | --- | --- |
| physics_frozen | +0.960902 | 87.150372 / 98.058058 / 99.326400 / 99.946458 | 3/3 / 5/6 / 9/10 / 15/15 | yes / yes / yes / yes |
| bare_T300 | +0.809023 | 81.071308 / 97.678198 / 99.034303 / 99.897026 | 2/3 / 5/6 / 8/10 / 14/15 | yes / yes / yes / yes |
| bare_Tactual | +0.603008 | 81.071308 / 90.866482 / 92.011399 / 99.516276 | 2/3 / 4/6 / 7/10 / 12/15 | yes / yes / yes / yes |
| denominator_Vplus | +0.765414 | 81.071308 / 97.678198 / 98.530261 / 99.897026 | 2/3 / 5/6 / 7/10 / 14/15 | yes / yes / yes / yes |
| denominator_Dplus | +0.769925 | 81.071308 / 97.678198 / 98.530261 / 99.897026 | 2/3 / 5/6 / 7/10 / 14/15 | yes / yes / yes / yes |
| energy_prefix | +0.207519 | 3.327221 / 24.107046 / 24.646397 / 99.563880 | 0/3 / 3/6 / 5/10 / 13/15 | no / no / no / yes |
| spread | +0.027068 | 0.390889 / 78.334932 / 81.619110 / 92.042269 | 0/3 / 2/6 / 4/10 / 12/15 | no / yes / yes / yes |
| spread_reverse | -0.001504 | 0.554473 / 1.087047 / 17.730663 / 85.257703 | 0/3 / 0/6 / 5/10 / 11/15 | no / no / no / yes |

At budget 3, physics captures 87.150372% versus 81.071308% for T300, Tactual and both denominator controls, and 3.327221% for the energy prefix. It recovers 3/3 top ranks versus 2/3 for those bare/denominator controls. At budget 10 the respective captures are 99.326400% for physics,99.034303% for T300, 98.530261% for the two denominator controls, and 24.646397% for the energy prefix. At budget 15 the energy prefix catches rank 354 and reaches 99.563880%; physics still reaches 99.946458%. The complete budget pattern is retained rather than emphasizing only the strongest contrast.

Physics and T300 both put rank 354 first, so there is no demonstrated advantage over T300 for that single binary dominant-rank test. Physics’s measured advantage is recovery beyond the leader, full-order correlation and captured attribution at the registered budgets. The historical objective balances six target directions, whereas W emphasizes the large 7/8 absolute response; it is not designed to maximize this new scalar capture diagnostic directly.

Bare T300, Tactual, Vplus and Dplus were fixed in completed October 21 exploratory work after the original outcomes existed. They are useful retrospective controls, not historically independent pre-result predictors. Prefix/spread/reverse-spread identities are historical controls. No new historical selector PASS is assigned.

## S2 performance: selector INCOMPLETE

| Budget | Sealed P340 physics additions, identities only | P340-based attribution evaluation |
| --- | --- | --- |
| 3 | 362, 379, 375 | INCOMPLETE |
| 6 | 362, 379, 375, 374, 378, 377 | INCOMPLETE |
| 10 | 362, 379, 375, 374, 378, 377, 364, 373, 361, 376 | INCOMPLETE |
| 15 | 362, 379, 375, 374, 378, 377, 364, 373, 361, 376, 366, 360, 365, 372, 369 | INCOMPLETE |

The four physics rows in TRANSFER_RESULTS.csv have null capture, recovery, correlation and dominant-inclusion fields. Recording the rank identities is not a quantitative evaluation against P360 labels. No P360-based capture number is assigned to the frozen P340 selector, even as a descriptive transfer score.

For completeness, base-independent bare/energy/spread rules can be scored against the actual P360 labels with explicit P360 support. These are CONTROL_ONLY_P360 rows, not selector transfer evidence. Tactual uses p<360; historical prefix/spread patterns are rebased to P360 for this baseline calculation, without reconstructing any endpoint. Their selected identities and all metadata are in the CSV.

| P360 baseline only | Spearman | Captured %, b3 / b6 / b10 / b15 | Top-k recovery counts |
| --- | --- | --- | --- |
| bare_T300 | +0.705263 | 98.873351 / 99.365958 / 99.741839 / 99.948532 | 3/3 / 4/6 / 8/10 / 13/15 |
| bare_Tactual | +0.451128 | 98.873351 / 99.259280 / 99.581543 / 99.837075 | 3/3 / 4/6 / 6/10 / 12/15 |
| denominator_Vplus | +0.711278 | 98.873351 / 99.365958 / 99.741839 / 99.948532 | 3/3 / 4/6 / 8/10 / 13/15 |
| denominator_Dplus | +0.706767 | 98.873351 / 99.365958 / 99.741839 / 99.948532 | 3/3 / 4/6 / 8/10 / 13/15 |
| energy_prefix | -0.225564 | 89.286526 / 89.407308 / 89.454171 / 89.991068 | 1/3 / 2/6 / 3/10 / 10/15 |
| spread | +0.115789 | 7.306276 / 96.779604 / 97.111671 / 97.161935 | 1/3 / 3/6 / 6/10 / 10/15 |
| spread_reverse | -0.099248 | 7.308770 / 9.985968 / 10.164494 / 10.611866 | 1/3 / 2/6 / 5/10 / 10/15 |

Large captured fractions in S2 can be dominated by one contributor. These baseline-only results cannot fill the absent physics comparison or be pooled with S1 magnitudes. No transfer improvement is inferred by comparing S1 physics to S2 bare baselines.

## Computational advantage: structural, conditional and unmeasured

The selector does not need candidate/full-shell dressed eigenvectors or Schur contractions as features. Once base P340 target vectors and projected blocks are available, its historical incremental work is candidate 8×8 isolated block calculations, three 2×2 whitening problems per shell, and 20 greedy steps over small target-pair matrices. Exact rank attribution additionally requires candidate/full-endpoint dressed information and the shell resolvent contractions for the six targets. This supplies a structural reason to expect cheaper incremental screening.

It is not a free bare-operator lookup: obtaining the base targets requires a 2720-dimensional dressed problem for 11 states, and the actual historical hook also performs base tracking against an earlier parent. The original implementation builds a dense 3040-dimensional Stage-2 Hamiltonian, so matrix assembly/storage are not eliminated. S1/S2 full exact endpoints have dimensions 2880/3040, with 160-dimensional shell blocks. A selector run from scratch can therefore still be expensive.

No separate feature-versus-exact timing benchmark exists in the consumed records. The 275.227-second historical receipt covers the entire guarded run, selection and publication, not a selector-only or attribution-only measurement. This run’s sub-second evaluation merely reads previously computed orders and labels; it cannot demonstrate an end-to-end speedup. No speedup factor is claimed.

## Evidence classification and scientific answers

- **Can the selector avoid its answer?** Yes for the historically recorded input path: it uses base targets plus allowed candidate operators and isolated blocks, not exact rank labels. S1’s matching labels permit a valid retrospective test. Direct new feature regeneration is unavailable without the unsaved base vectors; the authenticated stored order is sufficient for this evaluation.
- **Does it improve dominant-state identification?** It identifies S1’s dominant 354 immediately, as T300 also does; it improves top3, ordering and capture relative to the specified controls. No S2 physics performance comparison is valid.
- **Are labels compatible?** S1 yes; frozen S2 no. Different P spaces cannot be repaired by matching rank names or endpoint dimensions.
- **Does performance transfer between shells?** INCOMPLETE from saved compatible labels. Historical pre-label application of the same algorithm to both shells is documented, but its exact P340-based S2 attribution evaluation is absent.
- **What is missing?** Exact P340-based isolated-S2 rank attributions with their tracked-target/endpoint provenance, or a separately frozen and saved P360 selector with matching labels. The historical isolated-S2 endpoint remains incompletely persisted.
- **Does it support future noncontiguous retention?** The S1 result supports testing it as an efficiency hypothesis. Captured absolute weight is not a signed energy correction, selected-space accuracy criterion, convergence proof or gate validation. Earlier frozen classifications and v1/v2 protocols remain unchanged.

The evidence combines historical label-independent selection, one retrospective base-compatible S1 evaluation, reused in-sample control associations and internal algebraic checks. It contains no new independent validation dataset. The aggregate cross-shell transfer classification is INCOMPLETE, rather than a negative empirical failure or an invented PASS.

**One next experiment (recommendation only):** under separate authorization, keep the sealed P340 S2 order fixed and construct a complete P340/Q360–379 exact rank-attribution record with saved tracked endpoint vectors, assignments and provenance. Evaluate the predeclared budgets against those matching labels without reranking. It would resolve the base-compatibility gap, while remaining retrospective; it would not recover an independent historical endpoint or satisfy the original v1 historical-reproduction gate by itself. No such solve or reconstruction is performed here.

**V2 PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.**

## Validation, reproducibility and preservation

All 7 cheap synthetic tests pass: top-k intersection and absolute-weight capture; incompatible-base null metrics; independent Spearman permutation identity; average ties; forbidden-label/trace-column invariance; absolute attribution versus signed cancellation; lower-rank attribution tie rule. Sealed greedy trace ranks match recorded orders, all budget supports match historical lists, attribution sums reproduce the saved shell totals, and forbidden fields do not reach the ranking producer. Missing-base metrics remain null rather than zero.

The CSV contains 64 rows:32 compatible S1 method/budget rows,28 baseline-only P360 rows, and 4 incomplete S2 physics rows. Four numerical/provenance artifacts reproduce byte for byte in a second small process. Main post-processing took 0.199416 s and 40.766 MiB peak RSS; repeat took 0.189414 s and 40.078 MiB. A 60-second alarm and 256-MiB resident watchdog bound each process.

Only 47 required-input/protected-file hashes and original Git status were checked. Prior broad preservation investigations were not rerun; this is the current narrow consumed-input boundary. Checkpoint array authentication is reused from prior evidence; arrays were neither loaded nor rehashed. Existing notes, Cphase changes, reports, checkpoint bytes, criteria, sealed artifacts and Git index remain unchanged. No staging, commits or pushes.

No figures are needed for this compatibility-limited test. All outputs stay in the new requested directory. Numerical definitions/selection metadata are in TRANSFER_PROTOCOL.md, FEATURE_LEAKAGE_AUDIT.md, SOURCE_COMPATIBILITY.md and RANKINGS_FROZEN.json.

Reproduce from the repository root:

```shell
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_cross_shell_selector_transfer_2026_10_21/analyze_transfer.py
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_cross_shell_selector_transfer_2026_10_21/verify_reproduction.py
PYTHONDONTWRITEBYTECODE=1 python research/sandbox/k380_cross_shell_selector_transfer_2026_10_21/build_report.py
```
