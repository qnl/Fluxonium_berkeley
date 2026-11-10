> Publication copy: the numerical-summary link points to its path-sanitized publication copy. Scientific statements and numerical values are unchanged. Original SHA-256 is recorded in ../../publication/MANIFEST.json.

Work Block 4 completed on October 24, 2026 (America/Los_Angeles). Exactly one guarded launcher invocation completed; no retry, fallback, additional experiment, or selected-space solve occurred. The frozen scientific scope and controls were unchanged.

P340=[0,340), Q_S2=[360,380), U_S2=P340 union Q_S2; ranks [340,360) remain excluded. Rank-major, mode4-minor ordering with eight oscillator states per rank was preserved. H/h and E/h are GHz; reported energy corrections use 1000 MHz/GHz and no 2*pi conversion.

The two authorized large Stage-2 solves returned eleven eigenpairs each at dimensions 2720 and 2880. One 160-dimensional Q eigendecomposition and the six registered shifted linear solves ran through the authenticated historical diagnostic. Zero Stage-1 solves occurred. The completion ledger and result schema agree. Supervised runtime was 24.958824542 s, excluding initial supervisor authentication/preflight and final publication. Sampled peak worker-group RSS was 1,287,094,272 bytes (1.198700 GiB); this is a sampled value, not a continuous peak or kernel cap. There were 238 samples, with maximum sampled loop gap 0.131993 s. No safety breach, signal, or cleanup error was recorded; the watchdog disarmed after verified cleanup.

All mandatory live launch gates passed under the shared experiment lock before worker creation. Exact initial telemetry was not separately retained; the last recorded system state was {"disk_free_bytes": 60779433984, "free_percent": 57.0, "throttled_pages": 0}. Output totaled 1,783,069 bytes, below 1 GiB.

Both eigensystems passed the existing numerical gates. P340 reproduced the authenticated historical energies exactly and all 121 absolute grid_phi elements within 3.774758283725532e-15. U_S2 has no independent historical endpoint; its validation establishes numerical consistency with the frozen implementation.

| Case | Max eigenpair residual (GHz) | Orthogonality inf | Max independent H action difference (GHz) | Max independent phi action difference | H anti-Hermitian max (GHz) |
| --- | --- | --- | --- | --- | --- |
| P340 | 4.677087012e-13 | 1.430539544e-14 | 6.469537475e-14 | 1.232734542e-14 | 5.453422730e-13 |
| U_S2 | 3.031518972e-13 | 6.354468564e-15 | 6.390199061e-14 | 1.240984320e-14 | 5.453422730e-13 |

The tracked maps were identity in both directions and all eleven confidence checks passed. Minimum assigned overlap was 0.999925191510. The maximum genuine complex-overlap principal angle over the reported target subspaces was 0.706516690 degrees. Historical absolute-overlap threshold scores are separate diagnostics. Phase-invariance checks passed for both cases.

| State | P340 E/h (GHz) | U_S2 E/h (GHz) | U minus P (MHz) | U assigned overlap |
| --- | --- | --- | --- | --- |
| 0 | 7.219195035371831 | 7.218979362354465 | -0.215673017366 | 0.999998203109 |
| 1 | 7.235230255888524 | 7.235001217712974 | -0.229038175550 | 0.999998077109 |
| 2 | 10.642870232667807 | 10.642653213978232 | -0.217018689575 | 0.999998133070 |
| 3 | 10.879991726276341 | 10.879737215524742 | -0.254510751599 | 0.999997862178 |
| 4 | 11.191198947387257 | 11.189809694532194 | -1.389252855063 | 0.999985631829 |
| 5 | 11.343782867155753 | 11.341939500948452 | -1.843366207300 | 0.999980767754 |
| 6 | 13.176514309429670 | 13.175697228288801 | -0.817081140870 | 0.999973545718 |
| 7 | 13.402459991601063 | 13.397604148267888 | -4.855843333175 | 0.999947051010 |
| 8 | 13.419208712429004 | 13.413922100289550 | -5.286612139454 | 0.999925191510 |
| 9 | 14.105208099282279 | 14.105006271575572 | -0.201827706707 | 0.999998037374 |
| 10 | 14.360722003286718 | 14.360010245986103 | -0.711757300614 | 0.999992486290 |

Primary rank attribution uses the full-U energy and full-U P/Q projections: sigma_rs=1000 Re(d_rs† q_rs)/(p_s†p_s), d_s=H_QP p_s. W_r sums absolute signed sigma over states 0,1,2,3,7,8. All twenty ranks and all eight entries per rank remain present; coherent cross-rank mixing is preserved. Complex p, q, d and per-rank contractions are saved in COMPLEX_INTERMEDIATES.npz.

Total absolute S2 attribution is 12.607858747887 MHz. Rank 362 accounts for 88.68799343%; ranks 362,379,375 together account for 98.68001260%. These are attribution weights, not transition-frequency errors or selected-space spectral accuracy.

| Rank | W_r (MHz) | sigma_0 | sigma_1 | sigma_2 | sigma_3 | sigma_7 | sigma_8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 360 | 0.00499101893718 | -0.000105914291125 | -0.00012961452868 | -0.000624828541893 | -0.00109106169208 | -0.00162706598114 | -0.00141253390227 |
| 361 | 0.0339578289448 | -0.000744848447258 | -0.000841921010154 | -0.000126029155439 | -0.000229324052266 | -0.0224605680222 | -0.00955513825752 |
| 362 | 11.1816569382 | -0.208291524368 | -0.225399504861 | -0.0339577455966 | -0.0604073418059 | -4.9831570514 | -5.6704437702 |
| 363 | 0.00115796910043 | -3.66925208401e-05 | -3.95156762432e-05 | -3.50544152169e-06 | -5.84092575724e-06 | -0.000946717737467 | -0.000125696798597 |
| 364 | 0.0173127528207 | -0.000138520756591 | -0.000159842297842 | -0.00187749269761 | -0.0025635020041 | -0.00464575470498 | -0.00792764035954 |
| 365 | 0.00178386233915 | -1.22280052985e-05 | -1.62175962064e-05 | -0.000495876726111 | -0.000733700316487 | -4.44576039501e-05 | -0.000481382091094 |
| 366 | 0.00284633513247 | -2.61090985354e-05 | -3.39242589158e-05 | -0.000783245189977 | -0.00112463460206 | -3.58258586868e-06 | -0.000874839397115 |
| 367 | 0.00218159666178 | -3.25550579254e-05 | -3.24970974587e-05 | -6.08481304124e-05 | -6.00887888353e-05 | -0.000864703843486 | -0.00113090374367 |
| 368 | 0.00145722889168 | -1.55302237192e-05 | -1.10512861256e-05 | -0.000116306091495 | -0.00019616934013 | -0.000556649004943 | -0.000561522945266 |
| 369 | 0.000713004631847 | -6.24003789692e-07 | -2.40564555303e-07 | -0.000190562503369 | -0.000308876194463 | -7.02052236153e-06 | -0.000205680843308 |
| 370 | 0.00115151954236 | -5.89258786805e-05 | -6.05645378407e-05 | -0.000104975948694 | -0.000176387134068 | -1.95940400198e-06 | -0.000748706639076 |
| 371 | 0.00513935187476 | -7.92592271034e-05 | -7.08872408841e-05 | -2.33236797652e-06 | -1.909577319e-05 | -0.00188519448583 | -0.00308258277978 |
| 372 | 0.00175213296735 | -1.21037051418e-05 | -1.36125768004e-05 | -0.00023401845966 | -0.000257420306793 | -0.000291242446792 | -0.000943735472168 |
| 373 | 0.0122981637375 | -0.000600238655284 | -0.000584058392066 | -0.00142715516271 | -0.0016756903151 | -0.00397165341836 | -0.00403936779395 |
| 374 | 0.0590872771509 | -0.00204870164138 | -0.00288021994717 | -0.0169732803563 | -0.0258279238128 | -0.0110002422717 | -0.000356909121524 |
| 375 | 0.313187775527 | -0.010357321229 | -0.00935594882453 | -0.0393534916502 | -0.046949478163 | -0.127308336234 | -0.079863199427 |
| 376 | 0.00289000626562 | -0.000149264068039 | -0.000147055692411 | -0.000957676214196 | -0.00120943775295 | -0.000333278774396 | -9.32937636329e-05 |
| 377 | 0.00830143291653 | -0.000231858710503 | -0.00034654582386 | -0.00214688619813 | -0.00343509370972 | -0.00202617115911 | -0.000114877315205 |
| 378 | 0.00940066444299 | -0.000383603380838 | -0.000473253032289 | -0.00245310036455 | -0.00364957297356 | -0.0017829385099 | -0.000658196181857 |
| 379 | 0.946591887764 | -0.016521525594 | -0.0136507183302 | -0.137706511863 | -0.132009274526 | -0.402199917439 | -0.244503940012 |

The exact endpoint self-energy, P-space relaxation and observed energy shift are distinct. The fixed-base second-order estimate uses base energies/vectors and is not substituted into primary attribution. All following columns are MHz except shell probability.

| State | Exact self-energy | P relaxation | Observed U-P shift | Fixed-base second order | Second order minus observed | Shell probability |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -0.239847348863 | 0.0241743314868 | -0.215673017366 | -0.193976708959 | 0.0216963084069 | 3.03263457044e-06 |
| 1 | -0.254247193576 | 0.0252090180171 | -0.22903817555 | -0.206392638112 | 0.0226455374376 | 3.24672148142e-06 |
| 2 | -0.23959586866 | 0.0225771791182 | -0.217018689575 | -0.196651495583 | 0.0203671939926 | 3.20861300068e-06 |
| 3 | -0.28192991419 | 0.0274191625831 | -0.254510751599 | -0.229805463093 | 0.0247052885058 | 3.63095217248e-06 |
| 7 | -5.56511450555 | 0.709271172317 | -4.85584333317 | -4.23814940827 | 0.617693924902 | 7.57265315894e-05 |
| 8 | -6.02712391705 | 0.740511777588 | -5.28661213945 | -4.63854054963 | 0.648071589822 | 8.41795093443e-05 |

Maximum exact energy-identity error was 5.793676649546e-14 GHz; maximum P Schur-equation residual was 2.887341695592e-13 GHz; maximum reconstructed-q vector error was 4.169213757704e-16. Minimum absolute modal denominator was 42.013981682456 GHz, maximum eta was 0.006959834316, and near-resonant attributed fraction was zero. Signed modal denominators remain negative and are retained in the raw detail file; no pole clipping was introduced.

Frozen selector and controls below report capture percentages. Top-k recovery counts follow in a separate table. Every physics_frozen prefix includes dominant rank 362. The mismatched historical S2 bare_Tactual (P360) control remains excluded.

| Method | Budget 3 (%) | Budget 6 (%) | Budget 10 (%) | Budget 15 (%) |
| --- | --- | --- | --- | --- |
| bare_T300 | 98.680012604 | 99.297542198 | 99.716470554 | 99.931183253 |
| denominator_Dplus | 98.680012604 | 99.297542198 | 99.716470554 | 99.931183253 |
| denominator_Vplus | 98.680012604 | 99.297542198 | 99.716470554 | 99.931183253 |
| energy_prefix | 88.996918593 | 89.157569062 | 89.214661749 | 89.844653303 |
| physics_frozen | 98.680012604 | 99.289072208 | 99.816193847 | 99.912057501 |
| spread | 7.553193055 | 96.847157981 | 97.202757794 | 97.264086024 |
| spread_reverse | 7.556671163 | 10.120731224 | 10.313579405 | 10.865471477 |

| Method | Top-3 recovered | Top-6 recovered | Top-10 recovered | Top-15 recovered |
| --- | --- | --- | --- | --- |
| bare_T300 | 3 | 5 | 8 | 12 |
| denominator_Dplus | 3 | 5 | 8 | 12 |
| denominator_Vplus | 3 | 5 | 8 | 12 |
| energy_prefix | 1 | 3 | 3 | 10 |
| physics_frozen | 3 | 4 | 9 | 13 |
| spread | 1 | 4 | 6 | 11 |
| spread_reverse | 1 | 2 | 5 | 10 |

The frozen order is [362,379,375,374,378,377,364,373,361,376,366,360,365,372,369,370,371,368,367,363]. The measured descending-attribution order is [362, 379, 375, 374, 361, 364, 373, 378, 377, 371, 360, 376, 366, 367, 365, 372, 368, 363, 370, 369]. Selected prefixes and dominant-inclusion flags for all 28 comparisons are in the authenticated SELECTOR_COMPARISON.json and NUMERICAL_SUMMARY.json.

Historical S1 evidence is reused without rerunning S1. It measures P340 plus [340,360); this run measures P340 plus [360,380). Both now use base-compatible P340 attribution, but their full-U endpoints differ. This is a retrospective cross-shell comparison, not an independent prospective test. S1 total absolute attribution was 14.642727735201628 MHz; S2 is 12.607858747887336 MHz.

| Budget | S1 frozen capture (%) | S2 frozen capture (%) | S2 minus S1 (percentage points) | S1 top-k | S2 top-k |
| --- | --- | --- | --- | --- | --- |
| 3 | 87.150371906 | 98.680012604 | 11.529640698 | 3 | 3 |
| 6 | 98.058058003 | 99.289072208 | 1.231014205 | 5 | 4 |
| 10 | 99.326399702 | 99.816193847 | 0.489794146 | 9 | 9 |
| 15 | 99.946458353 | 99.912057501 | -0.034400852 | 15 | 13 |

The S1 frozen selector exceeded eligible bare/denominator controls at every budget. In S2 all four methods select the same first three ranks. Frozen capture is slightly lower at budgets 6 and 15, and higher at budget 10. This supports strong cross-shell attribution capture, without establishing consistent superiority over those controls. All historical S1 control rows are preserved in NUMERICAL_SUMMARY.json.

Base-to-union transition frequencies follow. Differences are U_S2 minus P340; frequencies themselves use the historical absolute energy-difference convention. The f78-minus-f01 row is scalar subtraction from saved carrier values, requiring no additional solve.

| Carrier | P340 (MHz) | U_S2 (MHz) | U minus P (MHz) |
| --- | --- | --- | --- |
| f01 | 16.035220516693 | 16.021855358510 | -0.013365158184 |
| f07 | 6183.264956229232 | 6178.624785913424 | -4.640170315808 |
| f08 | 6200.013677057173 | 6194.942737935085 | -5.070939122088 |
| f78 | 16.748720827941 | 16.317952021662 | -0.430768806279 |
| f81 | 6183.978456540480 | 6178.920882576576 | -5.057573963903 |
| f72 | 2759.589758933256 | 2754.950934289656 | -4.638824643599 |
| f21 | 3407.639976779283 | 3407.651996265257 | 0.012019485975 |
| f03 | 3660.796690904510 | 3660.757853170277 | -0.038837734233 |
| f38 | 2539.216986152663 | 2534.184884764809 | -5.032101387854 |
| f78_minus_f01 | 0.713500311248 | 0.296096663153 | -0.417403648095 |

The inherited comparison reports severe f78 constituent cancellation: excitation changes for states 7 and 8 are -4.640170315809 and -5.070939122088 MHz relative to the ground-state change, while f78 changes only -0.430768806279 MHz. The saved cancellation ratio is 0.044358351539. A small f78 difference alone would hide larger constituent shifts. The frozen strict comparison labels f08, f38, f81 and excitation 8 FAIL; their practical labels are BORDERLINE. These are base-to-union recovery diagnostics and do not indicate an eigensolver-validation failure. Path ordering remains PASS.

Every saved comparison component is reproduced below with its original value and threshold classifications. Values retain the source component conventions: energy components are MHz; operator/score/block components are relative differences; overlap/subspace components retain their historical conventions. These classifications do not establish selected-space accuracy or spatial convergence.

| Component | Saved value | Strict | Practical |
| --- | --- | --- | --- |
| block:all_targets:frobenius | 0.00038667039979 | PASS | PASS |
| block:all_targets:sv0 | 0.000318586536564 | PASS | PASS |
| block:all_targets:sv1 | 0.000318586536612 | PASS | PASS |
| block:all_targets:sv2 | 0.00200894494045 | PASS | PASS |
| block:all_targets:sv3 | 0.00200894494042 | PASS | PASS |
| block:all_targets:sv4 | 0.00353856928331 | PASS | PASS |
| block:all_targets:sv5 | 0.00353856928315 | PASS | PASS |
| block:comparators_to_78:frobenius | 0.00400820449884 | PASS | PASS |
| block:comparators_to_78:sv0 | 0.00284933885139 | PASS | PASS |
| block:comparators_to_78:sv1 | 0.00574720132841 | PASS | PASS |
| block:internal_78:frobenius | 0.00287001719899 | PASS | PASS |
| block:internal_78:sv0 | 0.00287001719897 | PASS | PASS |
| block:internal_78:sv1 | 0.00287001719901 | PASS | PASS |
| block:logical_to_23:frobenius | 0.000728317484096 | PASS | PASS |
| block:logical_to_23:sv0 | 0.000894084080778 | PASS | PASS |
| block:logical_to_23:sv1 | 0.000550141907419 | PASS | PASS |
| block:logical_to_78:frobenius | 0.000494177363116 | PASS | PASS |
| block:logical_to_78:sv0 | 0.000343071513187 | PASS | PASS |
| block:logical_to_78:sv1 | 0.0014343442661 | PASS | PASS |
| block:primary:frobenius | 9.14398088195e-05 | PASS | PASS |
| block:primary:sv0 | 0.000839998724613 | PASS | PASS |
| block:primary:sv1 | 0.00083999872464 | PASS | PASS |
| block:primary:sv2 | 0.00160518578069 | PASS | PASS |
| block:primary:sv3 | 0.00160518578066 | PASS | PASS |
| carrier:f01 | 0.0133651581837 | PASS | PASS |
| carrier:f03 | 0.0388377342329 | PASS | PASS |
| carrier:f07 | 4.64017031581 | BORDERLINE | PASS |
| carrier:f08 | 5.07093912209 | FAIL | BORDERLINE |
| carrier:f21 | -0.0120194859746 | PASS | PASS |
| carrier:f38 | 5.03210138785 | FAIL | BORDERLINE |
| carrier:f72 | 4.6388246436 | BORDERLINE | PASS |
| carrier:f78 | 0.430768806279 | PASS | PASS |
| carrier:f81 | 5.0575739639 | FAIL | BORDERLINE |
| centered:0 | -1.62744300042 | BORDERLINE | PASS |
| centered:1 | -1.61407784224 | BORDERLINE | PASS |
| centered:2 | -1.62609732821 | BORDERLINE | PASS |
| centered:3 | -1.58860526619 | BORDERLINE | PASS |
| centered:7 | 3.01272731539 | BORDERLINE | PASS |
| centered:8 | 3.44349612167 | BORDERLINE | PASS |
| edge:g01 | 0.00158053652185 | PASS | PASS |
| edge:g03 | 0.000894084080778 | PASS | PASS |
| edge:g07 | 0.000343071513187 | PASS | PASS |
| edge:g21 | 0.000550141907419 | PASS | PASS |
| edge:g38 | 0.00574720132841 | PASS | PASS |
| edge:g72 | 0.00284933885139 | PASS | PASS |
| edge:g78 | 0.00287001719899 | PASS | PASS |
| edge:g81 | 0.0014343442661 | PASS | PASS |
| excitation:1 | 0.0133651581837 | PASS | PASS |
| excitation:2 | 0.00134567220922 | PASS | PASS |
| excitation:3 | 0.0388377342331 | PASS | PASS |
| excitation:7 | 4.64017031581 | BORDERLINE | PASS |
| excitation:8 | 5.07093912209 | FAIL | BORDERLINE |
| overlap:0 | 0.999998203109 | PASS | PASS |
| overlap:1 | 0.999998077109 | PASS | PASS |
| overlap:2 | 0.99999813307 | PASS | PASS |
| overlap:3 | 0.999997862178 | PASS | PASS |
| overlap:7 | 0.99994705101 | PASS | PASS |
| overlap:8 | 0.99992519151 | PASS | PASS |
| score:0381 | 0.00628927670306 | PASS | PASS |
| score:0721 | 0.00195376963888 | PASS | PASS |
| score:0781 | 0.00178236708802 | PASS | PASS |
| subspace:0_1 | 0.999998077109 | PASS | PASS |
| subspace:0_1_2_3_7_8 | 0.999771584173 | PASS | PASS |
| subspace:0_1_7_8 | 0.999788008287 | PASS | PASS |
| subspace:2_3 | 0.999997862178 | PASS | PASS |
| subspace:7_8 | 0.99992519151 | PASS | PASS |

All raw numerical artifacts, including eigenvectors, full modal diagnostics, signed rank contributions, tracking diagnostics, principal subspaces and complex intermediates, remain in [the completed run](../k380_p340_s2_base_matched_validation_continuation_001/). [NUMERICAL_SUMMARY.json](NUMERICAL_SUMMARY.public.json) consolidates the results and historical S1 evidence. The original output manifest SHA-256 is 418874afc97a8cfc017b452c9f7c5f65844944c1a55d8b74f0d97b0ee61cbe0f.

Authentication passed for the approved independent review, amendment v2, addendum v1, all 192 SOURCE_PINS entries and all 602 integration artifact-manifest entries before execution. All 25 run-manifest entries were freshly verified afterward with no mismatches or unmanifested numerical files. Preservation passed for 1,239 snapshotted files and unchanged Git status; production, EmilianoUPDATE.md, Cphase, existing checkpoints and sealed benchmarks were unchanged. Pre-existing user modifications remain as captured in PRESERVATION.json. No commit or push occurred.

Recommendation: retain the frozen S2 selector as a successful retrospective attribution selector and carry its existing [362,379,375] prefix forward as a candidate for a separately authorized selected-space spectral validation. Do not refit its ordering from these outcomes or claim a unique advantage over the eligible bare/denominator controls. Require constituent excitation/carrier and grid_phi checks in that future assessment, because the present f78 cancellation and approximately 5 MHz higher-carrier shifts show that attribution capture alone cannot certify spectral accuracy. No such follow-up was executed here.

Selected-space spectral accuracy remains INCOMPLETE. U_S2 has numerical-consistency validation only; convergence beyond the finite parent, spatial convergence and gate readiness are not established. V2 SCHUR PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD. The single authorized experiment is finished.
