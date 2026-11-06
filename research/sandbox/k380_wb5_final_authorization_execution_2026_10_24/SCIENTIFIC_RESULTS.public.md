> Publication copy: local machine paths have been replaced by repository-relative paths or `python`. Scientific statements and values are unchanged. The original file and its SHA-256 are recorded in [the publication manifest](../../publication/MANIFEST.json).

# Work Block 5 — complete selected-space spectral results

October 24, 2026, America/Los_Angeles. EXECUTION: SUCCESS. One guarded invocation, 19/19 unique spaces, 28/28 comparisons, 19 new Stage-2 eigensolves, zero Stage-1/reference/Q eigensolves. No retry or resumption.

SOURCE AUTHENTICATION: PASS (847 pins). NUMERICAL VALIDATION: PASS. OUTPUT AUTHENTICATION: PASS. The independent review, safety approval and scientific authorization are separately hash-bound in this delivery. Frozen selections, thresholds and evaluation rules were unchanged.

The frozen aggregate scientific verdict is **MIXED_OR_NOT_IMPROVED**. All four physics-selected spaces have strict-PASS target-state, carrier and selectivity errors against the finite U_S2 reference, but the registered relative rule does not establish consistent superiority over every control. The budget-3 F/E comparison fails its centered-state-10 no-harm condition. F and bare T/D/V are identical at budget 3; F does not meet the two-envelope 20% improvement requirement over T/D/V at budgets 6, 10 or 15.

P340=[0,340), candidate Q_S2=[360,380), U_S2=P340 union Q_S2. Ranks [340,360) remain excluded. Each selected space adds exactly the registered budget to P340 in rank-major, oscillator-minor order (eight oscillator states). Energies are GHz; errors and carriers below are MHz. F=physics_frozen, T=bare_T300, D=denominator_Dplus, V=denominator_Vplus, E=energy_prefix, S=spread, R=spread_reverse. T/D/V aliases share one solve at each budget; F shares that solve at budget 3. Aliases are not independent replications.

## Physics versus controls

| Budget | Control | F max excitation error | Control max excitation error | Reduction | F carrier envelope | Control carrier envelope | Frozen verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | E | 0.279658335 | 0.535357895 | 47.7624% | 0.0451266374 | 0.526430454 | MIXED_OR_NOT_IMPROVED |
| 3 | T | 0.279658335 | 0.279658335 | 0.0000% | 0.0451266374 | 0.0451266374 | IDENTICAL_SPACE |
| 6 | E | 0.115900297 | 0.528190762 | 78.0571% | 0.0326565827 | 0.521571294 | IMPROVED |
| 6 | T | 0.115900297 | 0.133263637 | 13.0293% | 0.0326565827 | 0.0312996388 | MIXED_OR_NOT_IMPROVED |
| 10 | E | 0.0708811751 | 0.52312639 | 86.4505% | 0.00856562913 | 0.520224707 | IMPROVED |
| 10 | T | 0.0708811751 | 0.0795412724 | 10.8876% | 0.00856562913 | 0.011174555 | MIXED_OR_NOT_IMPROVED |
| 15 | E | 0.0190416345 | 0.509420552 | 96.2621% | 0.00515776974 | 0.509420552 | IMPROVED |
| 15 | T | 0.0190416345 | 0.0280057449 | 32.0081% | 0.00515776974 | 0.00463080394 | MIXED_OR_NOT_IMPROVED |

Carrier envelopes are normalized by registered strict tolerances (f78: 0.5 MHz; other carriers: 1 MHz). They are dimensionless and cannot be replaced by a favorable single-transition metric.

At budget 3, centered state-10 error is 0.215369845533 MHz for F versus 0.112044261889 MHz for E. The absolute-error increase is 0.103325583644 MHz, above the frozen 0.1-MHz no-harm slack. Both envelopes improve, but that registered failure is retained. Raw and ground-referenced errors remain fully reported; this is not a refitted criterion.

Against E, F meets the complete frozen improvement rule at budgets 6, 10 and 15. Against T/D/V, budget 6 has less than 20% excitation improvement and a slightly worse carrier envelope; budget 10 has less than 20% excitation improvement; budget 15 has more than 20% excitation improvement but a worse carrier envelope. No strict superiority over bare selection is established.

## Descriptive best method at each budget

| Budget | Lowest max target-excitation error | Value (MHz) | Lowest normalized carrier envelope | Value |
| --- | --- | ---: | --- | ---: |
| 3 | F,T,D,V | 0.279658335415 | F,T,D,V | 0.0451266374121 |
| 6 | F | 0.115900296705 | T,D,V | 0.0312996387966 |
| 10 | F | 0.0708811751382 | F | 0.00856562913123 |
| 15 | F | 0.0190416345456 | T,D,V | 0.00463080394297 |

These metric minima are descriptive. They do not introduce an overall winner or replace the frozen relative verdict. In particular T/D/V have the better carrier envelope at budgets 6 and 15, even though F has the lower worst-state excitation error.

## All 28 selections and accuracy summaries

| Budget | Method | Unique cell | Added Stage-1 ranks | WB4 attribution capture | Max excitation error | Max absolute carrier error | f01 signed error | f78 signed error | Selectivity signed error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | F | cell_04 | 362,375,379 | 0.9868001260 | 0.2796583354 | 0.04512663741 | 0.001074125208 | -0.01964303903 | -0.02071716424 |
| 3 | T | cell_04 | 362,375,379 | 0.9868001260 | 0.2796583354 | 0.04512663741 | 0.001074125208 | -0.01964303903 | -0.02071716424 |
| 3 | D | cell_04 | 362,375,379 | 0.9868001260 | 0.2796583354 | 0.04512663741 | 0.001074125208 | -0.01964303903 | -0.02071716424 |
| 3 | V | cell_04 | 362,375,379 | 0.9868001260 | 0.2796583354 | 0.04512663741 | 0.001074125208 | -0.01964303903 | -0.02071716424 |
| 3 | E | cell_01 | 360,361,362 | 0.8899691859 | 0.5353578945 | 0.5264304543 | -0.002755178448 | -0.1964229472 | -0.1936677687 |
| 3 | S | cell_02 | 360,369,379 | 0.0755319306 | 4.923217242 | 4.995245807 | 0.01547687885 | 0.5484627011 | 0.5329858222 |
| 3 | R | cell_03 | 360,370,379 | 0.0755667116 | 4.922837046 | 4.994686646 | 0.01547480424 | 0.5480234827 | 0.5325486785 |
| 6 | F | cell_09 | 362,374,375,377,378,379 | 0.9928907221 | 0.1159002967 | 0.03265658265 | 0.0001219142725 | -0.005952080826 | -0.006073995099 |
| 6 | T | cell_08 | 362,364,368,374,375,379 | 0.9929754220 | 0.1332636373 | 0.0312996388 | 0.0002965096488 | -0.01177101606 | -0.0120675257 |
| 6 | D | cell_08 | 362,364,368,374,375,379 | 0.9929754220 | 0.1332636373 | 0.0312996388 | 0.0002965096488 | -0.01177101606 | -0.0120675257 |
| 6 | V | cell_08 | 362,364,368,374,375,379 | 0.9929754220 | 0.1332636373 | 0.0312996388 | 0.0002965096488 | -0.01177101606 | -0.0120675257 |
| 6 | E | cell_05 | 360,361,362,363,364,365 | 0.8915756906 | 0.5281907617 | 0.5215712936 | -0.002781025316 | -0.1990427834 | -0.1962617581 |
| 6 | S | cell_06 | 360,362,364,369,374,379 | 0.9684715798 | 0.2484210497 | 0.1429459622 | -0.00065208169 | -0.05519943921 | -0.05454735752 |
| 6 | R | cell_07 | 360,365,370,375,377,379 | 0.1012073122 | 4.87365948 | 4.983042171 | 0.01624920243 | 0.5864943394 | 0.5702451369 |
| 10 | F | cell_14 | 361,362,364,373,374,375,376,377,378,379 | 0.9981619385 | 0.07088117514 | 0.008565629131 | 2.731350257e-05 | 0.003001778857 | 0.002974465355 |
| 10 | T | cell_13 | 361,362,363,364,368,374,375,377,378,379 | 0.9971647055 | 0.07954127244 | 0.01117455502 | 1.255988202e-05 | 0.003701848968 | 0.003689289086 |
| 10 | D | cell_13 | 361,362,363,364,368,374,375,377,378,379 | 0.9971647055 | 0.07954127244 | 0.01117455502 | 1.255988202e-05 | 0.003701848968 | 0.003689289086 |
| 10 | V | cell_13 | 361,362,363,364,368,374,375,377,378,379 | 0.9971647055 | 0.07954127244 | 0.01117455502 | 1.255988202e-05 | 0.003701848968 | 0.003689289086 |
| 10 | E | cell_10 | 360,361,362,363,364,365,366,367,368,369 | 0.8921466175 | 0.5231263903 | 0.5202247066 | -0.002782664474 | -0.2002170679 | -0.1974344034 |
| 10 | S | cell_11 | 360,361,362,364,366,369,371,374,376,379 | 0.9720275779 | 0.2276190922 | 0.1223920426 | -0.0007462073075 | -0.04592660016 | -0.04518039285 |
| 10 | R | cell_12 | 360,363,365,368,370,373,375,377,378,379 | 0.1031357940 | 4.870371507 | 4.984029818 | 0.01618359213 | 0.5883747601 | 0.5721911679 |
| 15 | F | cell_18 | 360,361,362,364,365,366,369,372,373,374,375,376,377,378,379 | 0.9991205750 | 0.01904163455 | 0.005157769742 | -7.239309596e-06 | 0.001319196185 | 0.001326435495 |
| 15 | T | cell_17 | 360,361,362,363,364,368,371,372,373,374,375,376,377,378,379 | 0.9993118325 | 0.02800574489 | 0.003093471712 | 1.262121252e-05 | 0.002315401971 | 0.002302780759 |
| 15 | D | cell_17 | 360,361,362,363,364,368,371,372,373,374,375,376,377,378,379 | 0.9993118325 | 0.02800574489 | 0.003093471712 | 1.262121252e-05 | 0.002315401971 | 0.002302780759 |
| 15 | V | cell_17 | 360,361,362,363,364,368,371,372,373,374,375,376,377,378,379 | 0.9993118325 | 0.02800574489 | 0.003093471712 | 1.262121252e-05 | 0.002315401971 | 0.002302780759 |
| 15 | E | cell_15 | 360,361,362,363,364,365,366,367,368,369,370,371,372,373,374 | 0.8984465330 | 0.5094205521 | 0.5094205521 | -0.003517997559 | -0.1953566253 | -0.1918386277 |
| 15 | S | cell_16 | 360,361,362,363,364,365,366,367,368,369,370,371,374,376,379 | 0.9726408602 | 0.2207294532 | 0.1203178195 | -0.0007481520079 | -0.04664578216 | -0.04589763015 |
| 15 | R | cell_19 | 360,363,365,368,369,370,371,372,373,374,375,376,377,378,379 | 0.1086547148 | 4.868416061 | 5.006154102 | 0.01543094975 | 0.5971289807 | 0.5816980309 |

The raw maximum-state errors, centered arrays, original-six-state summaries, strict/practical classifications, selected/reference carrier values and all absolute errors are also present without rounding in NUMERICAL_SUMMARY.json and the CSV tables.

## Individual excitation errors for every mandatory target

Signed errors are [(E_s−E_0)_selected−(E_s−E_0)_U_S2]×1000. State 0 is therefore zero by definition. Raw absolute energies and raw/centered errors are preserved separately in TARGET_ERRORS.csv; no fitted energy offset was applied.

| Budget | Method | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | F | 0 | 0.001074125208 | 0.02251272876 | 0.03547072728 | 0.03083748682 | 0.03787449595 | 0.1513558631 | 0.04512663741 | 0.02548359838 | 0.07777939035 | 0.2796583354 |
| 3 | T | 0 | 0.001074125208 | 0.02251272876 | 0.03547072728 | 0.03083748682 | 0.03787449595 | 0.1513558631 | 0.04512663741 | 0.02548359838 | 0.07777939035 | 0.2796583354 |
| 3 | D | 0 | 0.001074125208 | 0.02251272876 | 0.03547072728 | 0.03083748682 | 0.03787449595 | 0.1513558631 | 0.04512663741 | 0.02548359838 | 0.07777939035 | 0.2796583354 |
| 3 | V | 0 | 0.001074125208 | 0.02251272876 | 0.03547072728 | 0.03083748682 | 0.03787449595 | 0.1513558631 | 0.04512663741 | 0.02548359838 | 0.07777939035 | 0.2796583354 |
| 3 | E | 0 | -0.002755178448 | 0.1509933121 | 0.1626034351 | 0.1978818525 | 0.278493995 | 0.5353578945 | 0.5264304543 | 0.3300075072 | 0.1129141338 | 0.3524414287 |
| 3 | S | 0 | 0.01547687885 | -0.1139607989 | -0.07202856485 | 1.132715962 | 1.564477177 | 0.4053588548 | 4.374754541 | 4.923217242 | -0.05768799709 | 0.4769568 |
| 3 | R | 0 | 0.01547480424 | -0.1138271488 | -0.07184960028 | 1.132077406 | 1.563613241 | 0.4042258741 | 4.374813563 | 4.922837046 | -0.05782894756 | 0.469114496 |
| 6 | F | 0 | 0.0001219142725 | 0.004609721116 | 0.007056147992 | 0.01896037859 | 0.02468831996 | 0.0474716532 | 0.03265658265 | 0.02670450183 | 0.07517212086 | 0.1159002967 |
| 6 | T | 0 | 0.0002965096488 | 0.006695664018 | 0.01078487858 | 0.01690205877 | 0.02248077717 | 0.05978263087 | 0.0312996388 | 0.01952862274 | 0.07377357596 | 0.1332636373 |
| 6 | D | 0 | 0.0002965096488 | 0.006695664018 | 0.01078487858 | 0.01690205877 | 0.02248077717 | 0.05978263087 | 0.0312996388 | 0.01952862274 | 0.07377357596 | 0.1332636373 |
| 6 | V | 0 | 0.0002965096488 | 0.006695664018 | 0.01078487858 | 0.01690205877 | 0.02248077717 | 0.05978263087 | 0.0312996388 | 0.01952862274 | 0.07377357596 | 0.1332636373 |
| 6 | E | 0 | -0.002781025316 | 0.1490302467 | 0.15980885 | 0.1954001958 | 0.2759040201 | 0.5281907617 | 0.5215712936 | 0.3225285101 | 0.1022214376 | 0.3363976944 |
| 6 | S | 0 | -0.00065208169 | 0.0314802847 | 0.04182017722 | 0.1237764074 | 0.171791211 | 0.2480791598 | 0.1429459622 | 0.08774652299 | 0.1133247492 | 0.2484210497 |
| 6 | R | 0 | 0.01624920243 | -0.1430582031 | -0.1093826912 | 1.03835977 | 1.43222178 | 0.2130312465 | 4.28716514 | 4.87365948 | -0.1236704285 | 0.304539302 |
| 10 | F | 0 | 2.731350257e-05 | 0.002098304557 | 0.003364200762 | 0.005112956651 | 0.006537839824 | 0.02086229802 | 0.005563850274 | 0.008565629131 | 0.0404165014 | 0.07088117514 |
| 10 | T | 0 | 1.255988202e-05 | 0.003506345688 | 0.00514092974 | 0.01399648157 | 0.01833404132 | 0.03275865739 | 0.007472706051 | 0.01117455502 | 0.04716177147 | 0.07954127244 |
| 10 | D | 0 | 1.255988202e-05 | 0.003506345688 | 0.00514092974 | 0.01399648157 | 0.01833404132 | 0.03275865739 | 0.007472706051 | 0.01117455502 | 0.04716177147 | 0.07954127244 |
| 10 | V | 0 | 1.255988202e-05 | 0.003506345688 | 0.00514092974 | 0.01399648157 | 0.01833404132 | 0.03275865739 | 0.007472706051 | 0.01117455502 | 0.04716177147 | 0.07954127244 |
| 10 | E | 0 | -0.002782664474 | 0.1480824887 | 0.1583939471 | 0.1936550542 | 0.2735272468 | 0.5231263903 | 0.5202247066 | 0.3200076387 | 0.1004705067 | 0.3225782153 |
| 10 | S | 0 | -0.0007462073075 | 0.03067097832 | 0.04034692103 | 0.1199947853 | 0.1663890699 | 0.2276190922 | 0.1223920426 | 0.07646544243 | 0.08727155015 | 0.1963612684 |
| 10 | R | 0 | 0.01618359213 | -0.1459064554 | -0.1136583109 | 1.027030352 | 1.41738001 | 0.1955986467 | 4.281996747 | 4.870371507 | -0.143625077 | 0.2847734371 |
| 15 | F | 0 | -7.239309596e-06 | 8.160133103e-05 | 0.0002492051019 | 0.003751365106 | 0.005030810878 | 0.004157519668 | 0.003831334248 | 0.005150530433 | 0.01565645351 | 0.01904163455 |
| 15 | T | 0 | 1.262121252e-05 | 0.001377493619 | 0.002081441795 | 0.001060367064 | 0.001318568306 | 0.009350903142 | 0.0007780697411 | 0.003093471713 | 0.002498216404 | 0.02800574489 |
| 15 | D | 0 | 1.262121252e-05 | 0.001377493619 | 0.002081441795 | 0.001060367064 | 0.001318568306 | 0.009350903142 | 0.0007780697411 | 0.003093471713 | 0.002498216404 | 0.02800574489 |
| 15 | V | 0 | 1.262121252e-05 | 0.001377493619 | 0.002081441795 | 0.001060367064 | 0.001318568306 | 0.009350903142 | 0.0007780697411 | 0.003093471713 | 0.002498216404 | 0.02800574489 |
| 15 | E | 0 | -0.003517997559 | 0.1350632731 | 0.1375837066 | 0.1743745343 | 0.2504603503 | 0.4337933905 | 0.5094205521 | 0.3140639268 | 0.08021767925 | 0.166797407 |
| 15 | S | 0 | -0.0007481520079 | 0.03011829216 | 0.03944149994 | 0.1176607376 | 0.1632289521 | 0.2207294532 | 0.1203178195 | 0.07367203738 | 0.07802840567 | 0.1691637262 |
| 15 | R | 0 | 0.01543094975 | -0.1613005492 | -0.137738041 | 1.013929418 | 1.402587815 | 0.09754756326 | 4.27128708 | 4.868416061 | -0.1526803161 | 0.1171846198 |

## Every carrier: signed / absolute error

Each entry is signed error / absolute error, in MHz. CARRIERS_AND_CANCELLATION.csv includes exact selected and reference frequencies, both classifications and cancellation constituents for all 252 carrier comparisons. SELECTIVITY.csv includes both signed and absolute f78−f01 errors and values for all 28 comparisons.

| Budget | Method | f01 | f03 | f07 | f08 | f21 | f38 | f72 | f78 | f81 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | F | 0.00107412521 / 0.00107412521 | 0.0354707273 / 0.0354707273 | 0.0451266374 / 0.0451266374 | 0.0254835984 / 0.0254835984 | 0.0214386035 / 0.0214386035 | -0.0099871289 / 0.0099871289 | 0.0226139087 / 0.0226139087 | -0.019643039 / 0.019643039 | 0.0244094732 / 0.0244094732 |
| 3 | T | 0.00107412521 / 0.00107412521 | 0.0354707273 / 0.0354707273 | 0.0451266374 / 0.0451266374 | 0.0254835984 / 0.0254835984 | 0.0214386035 / 0.0214386035 | -0.0099871289 / 0.0099871289 | 0.0226139087 / 0.0226139087 | -0.019643039 / 0.019643039 | 0.0244094732 / 0.0244094732 |
| 3 | D | 0.00107412521 / 0.00107412521 | 0.0354707273 / 0.0354707273 | 0.0451266374 / 0.0451266374 | 0.0254835984 / 0.0254835984 | 0.0214386035 / 0.0214386035 | -0.0099871289 / 0.0099871289 | 0.0226139087 / 0.0226139087 | -0.019643039 / 0.019643039 | 0.0244094732 / 0.0244094732 |
| 3 | V | 0.00107412521 / 0.00107412521 | 0.0354707273 / 0.0354707273 | 0.0451266374 / 0.0451266374 | 0.0254835984 / 0.0254835984 | 0.0214386035 / 0.0214386035 | -0.0099871289 / 0.0099871289 | 0.0226139087 / 0.0226139087 | -0.019643039 / 0.019643039 | 0.0244094732 / 0.0244094732 |
| 3 | E | -0.00275517845 / 0.00275517845 | 0.162603435 / 0.162603435 | 0.526430454 / 0.526430454 | 0.330007507 / 0.330007507 | 0.15374849 / 0.15374849 | 0.167404072 / 0.167404072 | 0.375437142 / 0.375437142 | -0.196422947 / 0.196422947 | 0.332762686 / 0.332762686 |
| 3 | S | 0.0154768789 / 0.0154768789 | -0.0720285648 / 0.0720285648 | 4.37475454 / 4.37475454 | 4.92321724 / 4.92321724 | -0.129437678 / 0.129437678 | 4.99524581 / 4.99524581 | 4.48871534 / 4.48871534 | 0.548462701 / 0.548462701 | 4.90774036 / 4.90774036 |
| 3 | R | 0.0154748042 / 0.0154748042 | -0.0718496003 / 0.0718496003 | 4.37481356 / 4.37481356 | 4.92283705 / 4.92283705 | -0.129301953 / 0.129301953 | 4.99468665 / 4.99468665 | 4.48864071 / 4.48864071 | 0.548023483 / 0.548023483 | 4.90736224 / 4.90736224 |
| 6 | F | 0.000121914272 / 0.000121914272 | 0.00705614799 / 0.00705614799 | 0.0326565827 / 0.0326565827 | 0.0267045018 / 0.0267045018 | 0.00448780684 / 0.00448780684 | 0.0196483538 / 0.0196483538 | 0.0280468615 / 0.0280468615 | -0.00595208083 / 0.00595208083 | 0.0265825876 / 0.0265825876 |
| 6 | T | 0.000296509649 / 0.000296509649 | 0.0107848786 / 0.0107848786 | 0.0312996388 / 0.0312996388 | 0.0195286227 / 0.0195286227 | 0.00639915437 / 0.00639915437 | 0.00874374416 / 0.00874374416 | 0.0246039748 / 0.0246039748 | -0.0117710161 / 0.0117710161 | 0.0192321131 / 0.0192321131 |
| 6 | D | 0.000296509649 / 0.000296509649 | 0.0107848786 / 0.0107848786 | 0.0312996388 / 0.0312996388 | 0.0195286227 / 0.0195286227 | 0.00639915437 / 0.00639915437 | 0.00874374416 / 0.00874374416 | 0.0246039748 / 0.0246039748 | -0.0117710161 / 0.0117710161 | 0.0192321131 / 0.0192321131 |
| 6 | V | 0.000296509649 / 0.000296509649 | 0.0107848786 / 0.0107848786 | 0.0312996388 / 0.0312996388 | 0.0195286227 / 0.0195286227 | 0.00639915437 / 0.00639915437 | 0.00874374416 / 0.00874374416 | 0.0246039748 / 0.0246039748 | -0.0117710161 / 0.0117710161 | 0.0192321131 / 0.0192321131 |
| 6 | E | -0.00278102532 / 0.00278102532 | 0.15980885 / 0.15980885 | 0.521571294 / 0.521571294 | 0.32252851 / 0.32252851 | 0.151811272 / 0.151811272 | 0.16271966 / 0.16271966 | 0.372541047 / 0.372541047 | -0.199042783 / 0.199042783 | 0.325309535 / 0.325309535 |
| 6 | S | -0.00065208169 / 0.00065208169 | 0.0418201772 / 0.0418201772 | 0.142945962 / 0.142945962 | 0.087746523 / 0.087746523 | 0.0321323664 / 0.0321323664 | 0.0459263458 / 0.0459263458 | 0.111465677 / 0.111465677 | -0.0551994392 / 0.0551994392 | 0.0883986047 / 0.0883986047 |
| 6 | R | 0.0162492024 / 0.0162492024 | -0.109382691 / 0.109382691 | 4.28716514 / 4.28716514 | 4.87365948 / 4.87365948 | -0.159307406 / 0.159307406 | 4.98304217 / 4.98304217 | 4.43022334 / 4.43022334 | 0.586494339 / 0.586494339 | 4.85741028 / 4.85741028 |
| 10 | F | 2.73135026e-05 / 2.73135026e-05 | 0.00336420076 / 0.00336420076 | 0.00556385027 / 0.00556385027 | 0.00856562913 / 0.00856562913 | 0.00207099105 / 0.00207099105 | 0.00520142837 / 0.00520142837 | 0.00346554572 / 0.00346554572 | 0.00300177886 / 0.00300177886 | 0.00853831563 / 0.00853831563 |
| 10 | T | 1.2559882e-05 / 1.2559882e-05 | 0.00514092974 / 0.00514092974 | 0.00747270605 / 0.00747270605 | 0.011174555 / 0.011174555 | 0.00349378581 / 0.00349378581 | 0.00603362528 / 0.00603362528 | 0.00396636036 / 0.00396636036 | 0.00370184897 / 0.00370184897 | 0.0111619951 / 0.0111619951 |
| 10 | D | 1.2559882e-05 / 1.2559882e-05 | 0.00514092974 / 0.00514092974 | 0.00747270605 / 0.00747270605 | 0.011174555 / 0.011174555 | 0.00349378581 / 0.00349378581 | 0.00603362528 / 0.00603362528 | 0.00396636036 / 0.00396636036 | 0.00370184897 / 0.00370184897 | 0.0111619951 / 0.0111619951 |
| 10 | V | 1.2559882e-05 / 1.2559882e-05 | 0.00514092974 / 0.00514092974 | 0.00747270605 / 0.00747270605 | 0.011174555 / 0.011174555 | 0.00349378581 / 0.00349378581 | 0.00603362528 / 0.00603362528 | 0.00396636036 / 0.00396636036 | 0.00370184897 / 0.00370184897 | 0.0111619951 / 0.0111619951 |
| 10 | E | -0.00278266447 / 0.00278266447 | 0.158393947 / 0.158393947 | 0.520224707 / 0.520224707 | 0.320007639 / 0.320007639 | 0.150865153 / 0.150865153 | 0.161613692 / 0.161613692 | 0.372142218 / 0.372142218 | -0.200217068 / 0.200217068 | 0.322790303 / 0.322790303 |
| 10 | S | -0.000746207308 / 0.000746207308 | 0.040346921 / 0.040346921 | 0.122392043 / 0.122392043 | 0.0764654424 / 0.0764654424 | 0.0314171856 / 0.0314171856 | 0.0361185214 / 0.0361185214 | 0.0917210643 / 0.0917210643 | -0.0459266002 / 0.0459266002 | 0.0772116497 / 0.0772116497 |
| 10 | R | 0.0161835921 / 0.0161835921 | -0.113658311 / 0.113658311 | 4.28199675 / 4.28199675 | 4.87037151 / 4.87037151 | -0.162090048 / 0.162090048 | 4.98402982 / 4.98402982 | 4.4279032 / 4.4279032 | 0.58837476 / 0.58837476 | 4.85418791 / 4.85418791 |
| 15 | F | -7.2393096e-06 / 7.2393096e-06 | 0.000249205102 / 0.000249205102 | 0.00383133425 / 0.00383133425 | 0.00515053043 / 0.00515053043 | 8.88406407e-05 / 8.88406407e-05 | 0.00490132533 / 0.00490132533 | 0.00374973292 / 0.00374973292 | 0.00131919619 / 0.00131919619 | 0.00515776974 / 0.00515776974 |
| 15 | T | 1.26212125e-05 / 1.26212125e-05 | 0.00208144179 / 0.00208144179 | 0.000778069741 / 0.000778069741 | 0.00309347171 / 0.00309347171 | 0.00136487241 / 0.00136487241 | 0.00101202992 / 0.00101202992 | -0.000599423878 / 0.000599423878 | 0.00231540197 / 0.00231540197 | 0.0030808505 / 0.0030808505 |
| 15 | D | 1.26212125e-05 / 1.26212125e-05 | 0.00208144179 / 0.00208144179 | 0.000778069741 / 0.000778069741 | 0.00309347171 / 0.00309347171 | 0.00136487241 / 0.00136487241 | 0.00101202992 / 0.00101202992 | -0.000599423878 / 0.000599423878 | 0.00231540197 / 0.00231540197 | 0.0030808505 / 0.0030808505 |
| 15 | V | 1.26212125e-05 / 1.26212125e-05 | 0.00208144179 / 0.00208144179 | 0.000778069741 / 0.000778069741 | 0.00309347171 / 0.00309347171 | 0.00136487241 / 0.00136487241 | 0.00101202992 / 0.00101202992 | -0.000599423878 / 0.000599423878 | 0.00231540197 / 0.00231540197 | 0.0030808505 / 0.0030808505 |
| 15 | E | -0.00351799756 / 0.00351799756 | 0.137583707 / 0.137583707 | 0.509420552 / 0.509420552 | 0.314063927 / 0.314063927 | 0.138581271 / 0.138581271 | 0.17648022 / 0.17648022 | 0.374357279 / 0.374357279 | -0.195356625 / 0.195356625 | 0.317581924 / 0.317581924 |
| 15 | S | -0.000748152008 / 0.000748152008 | 0.0394414999 / 0.0394414999 | 0.12031782 / 0.12031782 | 0.0736720374 / 0.0736720374 | 0.0308664442 / 0.0308664442 | 0.0342305374 / 0.0342305374 | 0.0901995274 / 0.0901995274 | -0.0466457822 / 0.0466457822 | 0.0744201894 / 0.0744201894 |
| 15 | R | 0.0154309498 / 0.0154309498 | -0.137738041 / 0.137738041 | 4.27128708 / 4.27128708 | 4.86841606 / 4.86841606 | -0.176731499 / 0.176731499 | 5.0061541 / 5.0061541 | 4.43258763 / 4.43258763 | 0.597128981 / 0.597128981 | 4.85298511 / 4.85298511 |

## Cancellation and poorly reproduced targets

The largest excitation error across the study is 4.92321724166 MHz: method S, budget 3, state 8. The largest carrier error is 5.00615410169 MHz: R, budget 15, f38 (FAIL strict, BORDERLINE practical).

The weak controls S at budget 3 and R at all budgets retain roughly 4.3–4.9 MHz excitation errors in states 7/8. Their f07, f08, f72, f81 and f38 errors remain large even when f78 appears substantially smaller. Energy-prefix has no >1-MHz target/carrier error, but leaves approximately 0.51–0.54 MHz worst-state error at every budget. For F the worst target is state 10 at every budget; its error falls from 0.2796583354 MHz to 0.01904163455 MHz. Every registered F carrier is below 0.046 MHz at every budget, and below 0.0052 MHz at budget 15. These accuracy statements are against U_S2 only.

| Method | Budget | Carrier | Excitation error i | Excitation error j | Signed carrier error | rho | Severe |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| S | 3 | f78 | 4.37475454058 | 4.92321724166 | 0.548462701087 | 0.05898734842 | False |
| R | 3 | f78 | 4.37481356311 | 4.92283704586 | 0.548023482747 | 0.05894214633 | False |
| R | 6 | f78 | 4.28716514021 | 4.87365947957 | 0.586494339359 | 0.06402200279 | False |
| R | 10 | f78 | 4.28199674681 | 4.87037150688 | 0.588374760067 | 0.06428661345 | False |
| R | 15 | f78 | 4.27128708 | 4.86841606066 | 0.597128980658 | 0.0653335203 | False |

There are 5 flagged method/budget/carrier records and no severe flags. F, T, D, V and E have no flagged cancellation. The absence of a flag does not suppress their individual constituent errors.

## Numerical validation, target tracking and subspaces

All 209 computed eigenpairs passed their registered gates. Maximum residual: 4.21616440441e-13 GHz; maximum orthogonality infinity norm: 1.43329694907e-14; minimum target assigned overlap: 0.999931577077. Maximum independent H-action difference: 6.59772535443e-14 GHz; phi action: 1.32724989282e-14; phi projection: 7.10889595793e-15. Maximum registered subspace principal angle: 0.676488916299 degrees.

All eleven targets in every cell passed forward, reverse and P340 tracking. Saved eigenvectors were independently reloaded to verify rank identities, energy linkage, orthogonality, common-U overlap/phase and exact reproduction of recorded spectral metrics. No H was reassembled and no eigensolve was repeated during reporting. QUALITY_AND_TRACKING.csv contains every target residual/confidence record. COMPLEX_SUBSPACES.csv contains all five complex blocks, singular values, principal angles and clipping flags for every method/budget. The complete precision-preserving numerical records remain in each authenticated cell result.

| Cell | Aliases | Max residual (GHz) | Orthogonality inf | Minimum assigned overlap | Maximum subspace angle (deg) |
| --- | --- | ---: | ---: | ---: | ---: |
| cell_01 | E | 3.436760741e-13 | 6.212523562e-15 | 0.999993661716 | 0.2054800201 |
| cell_02 | S | 2.903180319e-13 | 5.877068825e-15 | 0.999931588623 | 0.6764361888 |
| cell_03 | R | 3.944332855e-13 | 5.326126975e-15 | 0.999931577077 | 0.6764889163 |
| cell_04 | T,D,V,F | 3.6370429e-13 | 4.761626942e-15 | 0.999997504022 | 0.06331508404 |
| cell_05 | E | 3.284529695e-13 | 5.162199778e-15 | 0.999993698453 | 0.2052212958 |
| cell_06 | S | 3.171937562e-13 | 6.194989571e-15 | 0.99999725007 | 0.1072289938 |
| cell_07 | R | 3.482739962e-13 | 9.538555081e-15 | 0.999934906399 | 0.6603343376 |
| cell_08 | T,D,V | 3.239098833e-13 | 9.713545968e-15 | 0.999998825984 | 0.0473607927 |
| cell_09 | F | 3.176628745e-13 | 5.977468653e-15 | 0.999998938369 | 0.04541770403 |
| cell_10 | E | 2.937272352e-13 | 6.501235345e-15 | 0.999993737479 | 0.2050188818 |
| cell_11 | S | 4.216164404e-13 | 4.902107692e-15 | 0.99999779464 | 0.1012650879 |
| cell_12 | R | 3.145956479e-13 | 1.433296949e-14 | 0.999935142472 | 0.6591715827 |
| cell_13 | T,D,V | 2.859625127e-13 | 3.604086987e-15 | 0.999999307393 | 0.02560912185 |
| cell_14 | F | 2.788831931e-13 | 7.920597815e-15 | 0.999999373725 | 0.0226698617 |
| cell_15 | E | 2.924781672e-13 | 5.22443413e-15 | 0.999994718984 | 0.1993870248 |
| cell_16 | S | 3.148264009e-13 | 6.241245247e-15 | 0.999997911824 | 0.1006578064 |
| cell_17 | T,D,V | 3.285463528e-13 | 8.75947754e-15 | 0.999999756106 | 0.01457661587 |
| cell_18 | F | 3.011651562e-13 | 3.71542154e-15 | 0.999999822678 | 0.01700074213 |
| cell_19 | R | 3.252116991e-13 | 5.104580738e-15 | 0.999935332015 | 0.6582229645 |

## Measured computational costs

Launcher elapsed through final directory/manifest publication: **213.312670500 seconds**. Supervisor elapsed before final receipt/manifest publication: 213.279629000 seconds. Initial authentication/live preflight: 0.861921292 seconds. Worker authentication: 0.653632666 seconds; input load/cache inventory: 0.477248791 seconds. Unique marginal cell total: 210.855209420 seconds.

Independent supervisor sampled peak worker-group RSS: **1501085696 bytes = 1.397994995 GiB** across 2100 monitor samples. This is below the 3-GiB supervised trigger and 4-GiB intended budget; it is not a continuous or kernel-enforced memory ceiling. Per-cell peaks below are supervisor samples attributed to observed ACTIVE_CELL, not isolated-process maxima. The worker's persistent-process ru_maxrss is cumulative and is separately retained.

| Cell | Aliases | Gather s | Construction s | Eigensolve s | Evaluation s | Archive publication s | Marginal total s | Sampled group peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cell_01 | E | 0.003057 | 0.931997 | 9.352710 | 0.071572 | 0.000771 | 10.363792 | 0.740463257 |
| cell_02 | S | 0.002685 | 0.744919 | 8.571346 | 0.075035 | 0.000682 | 9.397797 | 0.757598877 |
| cell_03 | R | 0.002751 | 0.740253 | 9.257106 | 0.066535 | 0.000885 | 10.070720 | 0.767715454 |
| cell_04 | T,D,V,F | 0.002583 | 0.728459 | 9.404832 | 0.075232 | 0.000734 | 10.214917 | 0.787445068 |
| cell_05 | E | 0.002558 | 0.973753 | 9.640684 | 0.073811 | 0.000776 | 10.694303 | 1.168579102 |
| cell_06 | S | 0.002601 | 0.883831 | 9.634234 | 0.071586 | 0.000816 | 10.596073 | 1.119262695 |
| cell_07 | R | 0.002572 | 0.856109 | 9.930543 | 0.076058 | 0.000807 | 10.869067 | 1.046218872 |
| cell_08 | T,D,V | 0.002936 | 0.902851 | 9.936217 | 0.074631 | 0.000862 | 10.920510 | 0.796295166 |
| cell_09 | F | 0.002629 | 0.912876 | 10.235192 | 0.072624 | 0.000787 | 11.227014 | 0.786026001 |
| cell_10 | E | 0.002903 | 0.875543 | 10.731362 | 0.070874 | 0.000843 | 11.684433 | 1.365890503 |
| cell_11 | S | 0.002791 | 0.864946 | 10.598508 | 0.103843 | 0.000742 | 11.575116 | 0.795043945 |
| cell_12 | R | 0.003101 | 0.854451 | 10.067120 | 0.079941 | 0.000781 | 11.009001 | 0.795791626 |
| cell_13 | T,D,V | 0.002624 | 0.858058 | 10.526784 | 0.096321 | 0.000862 | 11.488344 | 0.800094604 |
| cell_14 | F | 0.003093 | 0.899289 | 10.310419 | 0.091063 | 0.000785 | 11.308497 | 0.779876709 |
| cell_15 | E | 0.002808 | 0.948608 | 10.495924 | 0.116415 | 0.001026 | 11.568444 | 1.397994995 |
| cell_16 | S | 0.003281 | 0.929438 | 11.146424 | 0.083823 | 0.000907 | 12.167798 | 1.365142822 |
| cell_17 | T,D,V | 0.003400 | 0.954098 | 11.278016 | 0.083754 | 0.000752 | 12.323753 | 1.172058105 |
| cell_18 | F | 0.003220 | 0.882597 | 10.445363 | 0.075397 | 0.000774 | 11.410588 | 1.164093018 |
| cell_19 | R | 0.003467 | 0.889945 | 10.979939 | 0.088354 | 0.000708 | 11.965042 | 1.153594971 |

Measured unique phase totals (seconds): `{"construction_seconds": 16.63201957700585, "diagonalization_seconds": 192.5427225020012, "gather_seconds": 0.05505845799780218, "publication_seconds": 0.015299876995413797, "reusable_input_load_seconds": 0.0, "source_authentication_seconds": 0.0, "total_cell_seconds": 210.8552094200004, "tracking_validation_metrics_seconds": 1.5468693349994282}`.

Previously paid P340/U_S2 reference computations are reused. Shared costs are counted once; T/D/V and the budget-3 F alias are not charged as separate solves. Per-cell total stops before final RESULT JSON publication; full launcher time covers surrounding IO/metadata. Historical screening/base-preparation allocations remain null/unavailable. There is no measured comparable end-to-end speedup; speedup_claim=NOT_ESTABLISHED.

## Attribution capture versus actual spectral accuracy

F capture fractions are 0.9868001260, 0.9928907221, 0.9981619385 and 0.9991205750 at budgets 3/6/10/15. The new calculations establish actual selected-space accuracy: corresponding all-eleven maximum excitation errors are 0.2796583354, 0.1159002967, 0.07088117514 and 0.01904163455 MHz. High attribution capture accompanies good finite-parent spectral accuracy here, but does not establish consistent superiority over bare controls or a universal capture-to-error bound. T/D/V capture is slightly larger at budgets 6/15 yet has larger worst-state error; carrier-envelope rankings differ. Attribution and spectral error remain distinct quantities.

## Authentication, preservation and recommendation

Authenticated output: `research/sandbox/k380_selected_space_spectral_accuracy_execution_001`. OUTPUT_MANIFEST.json SHA-256: `637d2758e657df47cb002bc492e78dd76262ec032d2ca135b0595e0a8cdaf918`; 70 entries verified. EXECUTION_RECEIPT.json SHA-256: `cf07ec6e3e68a7222ec8bf0431fcb09049889e574f13d2eee7d353e1cc8de693`. Output directory size: 10227119 bytes. Watchdog disarmed after verified cleanup; no failed gate was bypassed. The completed run, existing locks and all source artifacts are retained without modification by this report.

Final preservation evidence is recorded in PRESERVATION.json and the delivery seal. No changes are permitted to production, EmilianoUPDATE.md, Cphase, checkpoints, WB4 outputs, frozen comparisons, sealed benchmarks or existing figures. No Git staging, commit or push is performed.

Specific conclusion: the frozen selector achieves strict-PASS selected-space accuracy against U_S2 and lowers worst-state excitation error relative to energy-prefix at all budgets, with full registered improvement at budgets 6/10/15. It does not satisfy the frozen rule for consistent superiority over every control, particularly bare T/D/V. Reporting the all-eleven state-10 and mean-centered errors changes the budget-3 conclusion relative to an envelope-only comparison; those frozen rules remain in force.

Next research action: preregister a separate transfer-validation experiment at a different, independently validated finite parent to test generalization of this unchanged selector against the same bare controls. Do not refit this experiment or treat it as convergence evidence. Such work requires separate scope and authorization.

This retrospective study shares the same finite Stage-1 parent, model parameters, oscillator truncation and inherited assembly/tracking conventions with U_S2. It establishes no infinite-basis/spatial convergence, experimental physical validation, gate readiness or end-to-end speedup. No further scientific execution is performed after this one run.

V2 SCHUR PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
