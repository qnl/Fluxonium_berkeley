# Thomas: research update after October 20

Follow-up to the [October 20 research work block](../WORK_BLOCK_2026-10-20.md). See the [illustrated report](THOMAS_RESEARCH_UPDATE.md) for all eight figures.

The main advance is a completed test of actual noncontiguous selected-space spectra, following the earlier high-rank attribution studies. The fixed physics-informed selector gives accurate finite-parent spectra with a few added states, but does not demonstrate consistent superiority over simple bare-coupling selection.

**Observed numerical results.** The k300-k380 nested-prefix sequence remains unconverged under its frozen criteria: the largest adjacent carrier corrections grow from 0.715 to 6.738 MHz. Important high-rank virtual contributions recur. S1 rank 354 carries 66.82% of its shell's aggregate absolute attribution; matched-base S2 rank 362 carries 88.69%. Their strong target-conditioned couplings and the saved state fingerprints support studying response-aware retention, without proving a physical mode identity or cutoff stability.

WB4 resolved the earlier S2 retained-base mismatch by calculating P340 plus ranks [360,380), with [340,360) excluded. WB5 then completed 19 distinct selected-space eigensolves and 28 method/budget comparisons against that U_S2 reference. All numerical checks passed.

Maximum errors over all eleven excitation energies are MHz:

| Added states | Physics F | Bare T/D/V | Energy prefix E | F versus E, complete rule |
| ---: | ---: | ---: | ---: | --- |
| 3 | 0.279658 | 0.279658 | 0.535358 | MIXED_OR_NOT_IMPROVED |
| 6 | 0.115900 | 0.133264 | 0.528191 | IMPROVED |
| 10 | 0.070881 | 0.079541 | 0.523126 | IMPROVED |
| 15 | 0.019042 | 0.028006 | 0.509421 | IMPROVED |

F and T/D/V are identical at budget 3. At budgets 6/10/15, F has the smallest worst-state error, while T/D/V have the smaller normalized carrier-error envelope at budgets 6 and 15. At budget 3, F fails the E comparison's mean-centered state-10 no-harm condition. The complete unchanged verdict is **MIXED_OR_NOT_IMPROVED**.

**Physical interpretation.** Strong off-resonant coupling to selected high ranks is consistent with MHz shifts despite very small changes in dressed vectors. A small f78 shift can cancel large constituent excitation shifts. Attribution capture is therefore useful evidence for retention, but is not itself spectral validation or a bound on the omitted basis.

**Proposed future work.** Preregister an independent finite-parent transfer test of this unchanged selector against frozen bare controls, with all targets, carriers, cancellation checks and complete measured costs. That would test generalization beyond these retrospective records. No new experiment is implemented or authorized by this handoff.

No end-to-end speedup, spatial/retained-basis convergence, or asymmetric four-mode gate is established. Schur-v2 prelaunch, gate work and spatial convergence remain **HOLD**. This update adds no eigensolves and preserves the October-20 handoff.
