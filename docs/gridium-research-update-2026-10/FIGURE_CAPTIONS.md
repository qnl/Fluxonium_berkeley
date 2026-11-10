# Reviewed figure captions

Figures are ordered to follow the scientific argument from convergence failure
to mechanism, model reduction, independent transfer, cancellation risk, and
the final k340 result.

## A2 — Adjacent-shell retained-basis corrections

**PDF:** [A2_adjacent_shell_frequency_corrections_v2.pdf](figures/A2_adjacent_shell_frequency_corrections_v2.pdf)

**What is plotted:** Signed adjacent-shell changes on symmetric-log axes and
absolute magnitudes on log axes for all nine mandatory carriers. The
k220→k240 and k280→k300 reorganizations are highlighted without suppressing
smaller corrections.

**Parameter point:** N71 asymmetric four-mode model. `epsilon_J=0.10` uses the
finite k300 parent with nested prefixes 180–300; `epsilon_J=0.11` uses the
finite k340 parent with nested prefixes 260–340.

**Interpretation:** Retained-basis corrections are large and nonmonotonic. The
late influential shells cannot be interpreted as a smoothly shrinking tail.

**Caveats:** The two parameter points have distinct parents and are not one
numerical sequence. Neither finite endpoint is ground truth; magnitude panels
discard sign, which remains visible in the paired panels.

## B3 — Cross-parameter Schur/Feshbach attribution

**PDF:** [B3_k300_cross_parameter_schur_attribution_v2.pdf](figures/B3_k300_cross_parameter_schur_attribution_v2.pdf)

**What is plotted:** Exact aggregate absolute self-energy attribution by rank,
group-78 coupling norms, and minimum absolute denominators for the k280→k300
shell. Ranks 290 and 292 jointly carry 86.75% and 84.42% of the attribution
weight at the two parameter points.

**Parameter point:** Authenticated N71 k300 parents at `epsilon_J=0.10` and
`epsilon_J=0.11`; shell ranks 280–299.

**Interpretation:** The dominant mechanism transfers across `epsilon_J` as a
few strongly coupled, off-resonant virtual states. The subleading contribution
changes through coupling redistribution, not a new resonance.

**Caveats:** Absolute attribution weight is not a signed dressed-energy shift.
The analysis is exact within each saved finite parent, not an infinite-tail
prediction.

## C2a — Selector comparison at `epsilon_J=0.10`

**PDF:** [C2a_selector_vs_controls_epsj010_v2.pdf](figures/C2a_selector_vs_controls_epsj010_v2.pdf)

**What is plotted:** For S1, S2, and their union, filled bars show budget-10
maximum absolute carrier error, black dashes show budget-10 mean absolute
error, and hollow diamonds show budget-15 maximum absolute error. All
available equal-cost controls are retained.

**Parameter point:** Authenticated N71 `epsilon_J=0.10` k300 parent; S1 ranks
260–279 and S2 ranks 280–299.

**Interpretation:** Coupling/denominator-aware selection is substantially more
efficient than conventional truncations in the tested spaces.

**Caveats:** Errors are measured against finite-parent selected-space targets.
The frozen selector verdict remains **BORDERLINE**; this is not an
unconditional convergence claim.

## C2b — Independent selector comparison at `epsilon_J=0.11`

**PDF:** [C2b_selector_vs_controls_epsj011_v2.pdf](figures/C2b_selector_vs_controls_epsj011_v2.pdf)

**What is plotted:** The same S1/S2/union and b10/b15 comparison as C2a, using
the independently selected `epsilon_J=0.11` spaces and every available
equal-cost control.

**Parameter point:** Authenticated N71 `epsilon_J=0.11` k300 parent; S1 ranks
260–279 and S2 ranks 280–299.

**Interpretation:** Strong improvement transfers to the held-out parameter
point without refitting the selector.

**Caveats:** The official outcomes remain **SELECTOR VALIDATION: BORDERLINE**
and **MULTI-TARGET TRANSFER: PARTIAL**. Strong relative performance does not
make the ordinary prefix converged.

## C3 — `f78` common-mode error cancellation

**PDF:** [C3_f78_error_cancellation_v2.pdf](figures/C3_f78_error_cancellation_v2.pdf)

**What is plotted:** Signed raw state-7/8 energy errors, an expanded view of the
physics-selector state errors, signed `f78` transition errors, and positive
absolute magnitudes on a log scale.

**Parameter point:** `epsilon_J=0.11`, authenticated k300 parent, union/budget
10 selected spaces.

**Interpretation:** The ordinary prefix obtains a small `f78` error by
cancelling two approximately +54 MHz state errors. Its smaller splitting error
does not indicate better recovery of either state.

**Caveats:** Raw state errors are candidate-minus-endpoint eigenvalue errors in
one Hamiltonian energy convention; a common energy-zero shift changes them
together. Their difference, the transition-frequency error, is offset
invariant. The selector verdict remains **BORDERLINE**.

## D1 — k340 retained-basis spectral test

**PDF:** [D1_k340_spectral_convergence_v2.pdf](figures/D1_k340_spectral_convergence_v2.pdf)

**What is plotted:** Signed and absolute mandatory-carrier corrections for
k300→k320 and k320→k340, together with the frozen strict PASS limits for
standard carriers and `f78`.

**Parameter point:** Authenticated N71 `epsilon_J=0.11` k340 parent; nested
prefixes 300, 320, and 340 shown from the full 260/280/300/320/340 sequence.

**Interpretation:** The maximum correction increases from 0.714572 to
1.299836 MHz and seven of nine increments grow. Practical component passes do
not override the frozen trend, nonmonotonicity, and reversal/cancellation
rules: **RETAINED-BASIS CONVERGENCE: FAIL**.

**Caveats:** The 0.5 MHz line applies only to cancellation-sensitive `f78`;
the standard strict PASS boundary is 1.0 MHz. k340 is a finite endpoint, not
ground truth, and this result says nothing by itself about spatial or device
convergence.
