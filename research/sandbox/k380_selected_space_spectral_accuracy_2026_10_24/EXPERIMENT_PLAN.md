Work Block 5 — selected-space spectral accuracy. October 24, 2026, America/Los_Angeles. Implementation, preregistration and no-solve validation only; no scientific run is authorized.

Question: within the authenticated finite U_S2 parent, does the unchanged physics-informed selector recover the eleven dressed target states more accurately than every eligible equal-budget control? This is retrospective. U_S2 is a finite numerical reference, not infinite-basis ground truth. Work Block 4 conclusions remain intact: attribution capture does not establish spectral accuracy or selector superiority over bare controls.

P340=[0,340); Q_S2=[360,380); U_S2=P340 union Q_S2. Ranks [340,360) remain excluded. Budgets are 3,6,10,15. Additional-state cost is exactly the budget for every method. Energy prefix means lowest ranks within Q_S2, not a global contiguous Stage-1 prefix. Eligible methods are physics_frozen, bare_T300, denominator_Dplus, denominator_Vplus, energy_prefix, spread, spread_reverse. P360-based bare_Tactual is ineligible.

The authoritative 28 ordered selections come from the authenticated WB4 SELECTOR_COMPARISON.json and are checked against FROZEN_SELECTOR_RECORD.json before freezing. FROZEN_COMPARISONS.json preserves every method selection order and its sorted retained set. Deduplication uses exact sorted sets, not method names. It produces the following deterministic nineteen-job schedule. Each job requests eleven eigenpairs with scipy.linalg.eigh, driver=evr, subset_by_index=(0,10), exactly once, with no fallback or retry. Sorting sets changes tensor indexing only into the documented rank-major convention; it does not refit the selection order.

| Job | Budget | Dimension | Additional ranks (exact set) | Methods sharing this solve |
| --- | --- | --- | --- | --- |
| cell_01 | 3 | 2744 | 360,361,362 | energy_prefix |
| cell_02 | 3 | 2744 | 360,369,379 | spread |
| cell_03 | 3 | 2744 | 360,370,379 | spread_reverse |
| cell_04 | 3 | 2744 | 362,375,379 | bare_T300, denominator_Dplus, denominator_Vplus, physics_frozen |
| cell_05 | 6 | 2768 | 360,361,362,363,364,365 | energy_prefix |
| cell_06 | 6 | 2768 | 360,362,364,369,374,379 | spread |
| cell_07 | 6 | 2768 | 360,365,370,375,377,379 | spread_reverse |
| cell_08 | 6 | 2768 | 362,364,368,374,375,379 | bare_T300, denominator_Dplus, denominator_Vplus |
| cell_09 | 6 | 2768 | 362,374,375,377,378,379 | physics_frozen |
| cell_10 | 10 | 2800 | 360,361,362,363,364,365,366,367,368,369 | energy_prefix |
| cell_11 | 10 | 2800 | 360,361,362,364,366,369,371,374,376,379 | spread |
| cell_12 | 10 | 2800 | 360,363,365,368,370,373,375,377,378,379 | spread_reverse |
| cell_13 | 10 | 2800 | 361,362,363,364,368,374,375,377,378,379 | bare_T300, denominator_Dplus, denominator_Vplus |
| cell_14 | 10 | 2800 | 361,362,364,373,374,375,376,377,378,379 | physics_frozen |
| cell_15 | 15 | 2840 | 360,361,362,363,364,365,366,367,368,369,370,371,372,373,374 | energy_prefix |
| cell_16 | 15 | 2840 | 360,361,362,363,364,365,366,367,368,369,370,371,374,376,379 | spread |
| cell_17 | 15 | 2840 | 360,361,362,363,364,368,371,372,373,374,375,376,377,378,379 | bare_T300, denominator_Dplus, denominator_Vplus |
| cell_18 | 15 | 2840 | 360,361,362,364,365,366,369,372,373,374,375,376,377,378,379 | physics_frozen |
| cell_19 | 15 | 2840 | 360,363,365,368,369,370,371,372,373,374,375,376,377,378,379 | spread_reverse |

Reference reuse: authenticate and load WB4 P340.npz and U_S2.npz with their saved state labels and validation records. Reuse the authenticated k380 Stage-1 eigenvalues and five complex projected operators; never load raw Stage-1 eigenvectors. No P340 or U_S2 eigensolve is required. No Q decomposition, shifted linear solve, full k380 dressed matrix, Stage-1 solve, gate simulation or spatial-convergence calculation is planned.

Future guarded sequence: acquire the same experiment-wide lock before source/auth/resource checks; authenticate the new immutable source/criteria/comparison manifest and the separate new user approvals; check live headroom; authenticate/load references and parent inputs; inventory possible compatible selected-eigenvector caches before attempts. Unexpected caches require review rather than recomputation or unverified reuse. Iterate cell_01 through cell_19 serially. Gather only P340 plus that selected set before Hamiltonian assembly. Write an exclusive pre-call ledger, make one dense call, track/validate all eleven states, evaluate every mandatory metric, atomically save eigenpairs/results, then release cell matrices. Stop the study on any failed gate; retain evidence and label unexecuted cells INCOMPLETE. No resume or automatic retry. Publish a study result only after complete schema verification and verified cleanup.

The present package contains no enabled research solver/worker/launcher. wb5.execute_research always rejects, including the prior WB4 token. Adding the nineteen-cell guarded worker, cost instrumentation, per-run identity and result aggregation requires a new integration and source freeze; this is an explicit readiness blocker, not permission to call scipy directly.

Fair comparisons: one result is shared across identical spaces; identical-space comparisons are ties. At budget 3 the physics and bare/denominator methods are necessarily tied. Do not count aliases as independent replications. Report every budget and control, including unfavorable results. Shared reused input costs must be accounted consistently; no speedup claim is supported by this study without comparable measured total costs.

V2 SCHUR PRELAUNCH: HOLD. GATE WORK: HOLD. SPATIAL CONVERGENCE: HOLD.
