# Gridium retained-basis research update — October 2026

**Latest: [October 24 illustrated research update](work-blocks-1-5/README.md)** — Work Blocks 1–5, spectral accuracy and eight scientific figures.

**Previous: [October 20 Research Work Block](WORK_BLOCK_2026-10-20.md)** —
authenticated k380 findings and the sealed, unexecuted restricted Schur v2 study.
The existing figures and sections below document the earlier work through k340.

This package summarizes the completed N71 retained-basis investigation for
the asymmetric four-mode Gridium model. It is intended as a compact scientific
review for Thomas Ersevim; the six reviewed figures are linked from
[FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md), with authentication and scope details
in [PROVENANCE.md](PROVENANCE.md).

## 1. Research objective

The long-term objective is a physically meaningful, high-fidelity logical
X90/X180 control protocol. This work addressed a prerequisite: whether the
finite Stage-1 basis used to construct the dressed low-energy model is stable
enough for control conclusions. It did not optimize or validate a gate.

The frozen outcomes are:

| Question | Classification |
|---|---|
| Selector validation | **BORDERLINE** |
| Multi-target transfer | **PARTIAL** |
| Cross-parameter shell mechanism | **SUPPORTED** |
| Retained-basis convergence | **FAIL** |
| Gate work | **HOLD** |
| Spatial convergence | **HOLD** |

No validated high-fidelity X90/X180 gate exists yet.

## 2. Retained-basis convergence investigation

At `epsilon_J=0.10`, the authenticated N71 sequence through the finite k300
parent showed large, nonmonotonic spectral changes. The k220→k240 and
k280→k300 shells reorganized important outer transitions by tens of MHz;
several carriers reversed direction, and the last increment was not a small
tail. [Figure A2](figures/A2_adjacent_shell_frequency_corrections_v2.pdf)
shows every signed correction together with its absolute magnitude.

Good state tracking did not rescue the convergence verdict. Tracking asks
whether the same states are followed; convergence additionally requires their
energies, control operators, subspaces, and pathway diagnostics to stabilize
under the frozen rules. The authenticated result is **RETAINED-BASIS
CONVERGENCE: FAIL**. The k300 endpoint is not ground truth.

## 3. Schur/Feshbach virtual-state mechanism

Exact finite-parent Schur/Feshbach attribution identified a concentrated
virtual-shell mechanism. For the k280→k300 shell, ranks 290 and 292 carry
86.75% of the aggregate absolute attribution weight at `epsilon_J=0.10` and
84.42% at `epsilon_J=0.11`. Their relevant denominators remain tens of GHz, so
the effect is off-resonant rather than a newly encountered avoided crossing.

[Figure B3](figures/B3_k300_cross_parameter_schur_attribution_v2.pdf) separates
exact attribution weight from coupling and denominator diagnostics. The
subleading rank changes across the two parameter points because the coupling
distribution changes. This supports the frozen classification
**CROSS-PARAMETER SHELL MECHANISM: SUPPORTED**, but does not establish an
infinite-basis limit.

## 4. Physics-informed basis selection

Ordinary energy-prefix retention misses high-rank states that have large
virtual influence. The frozen physics-informed selector instead uses
coupling/denominator-aware Schur-response features for the target pairs
`(0,1)`, `(2,3)`, and `(7,8)`. It was compared against equal-cost energy-prefix,
spread, reverse-spread, and—where available—balanced-prefix controls.

[Figure C2a](figures/C2a_selector_vs_controls_epsj010_v2.pdf) and
[Figure C2b](figures/C2b_selector_vs_controls_epsj011_v2.pdf) show strong
numerical advantages for the physics-informed spaces across the tested S1,
S2, and union cases. The plots retain budget-10 maxima and means and budget-15
maxima, including controls that improve substantially at the larger budget.
These results support efficient model reduction within the tested finite
parents; they do not prove that the ordinary retained-basis sequence is
converged or justify an unconditional selector-success claim.

## 5. Independent `epsilon_J` parameter validation

The selector formula, targets, weights, and decision rules were reused without
refitting at the independently chosen `epsilon_J=0.11` point. The broad
off-resonant mechanism transferred and the physics-selected subspaces improved
all target families relative to equal-cost controls. The frozen classifications
remain **SELECTOR VALIDATION: BORDERLINE** and **MULTI-TARGET TRANSFER:
PARTIAL**.

The decisive exception is illustrated by
[Figure C3](figures/C3_f78_error_cancellation_v2.pdf). At union/budget 10, the
ordinary prefix has raw state-7/8 errors of +54.479177/+54.338596 MHz, yet
their common-mode cancellation leaves an apparently small `f78` error of
-0.140580 MHz. The physics selector has +0.254622/+0.475006 MHz state errors
and a +0.220384 MHz `f78` error. A smaller splitting error therefore does not
imply better recovery of the individual states, and the preregistered no-harm
miss remains in force.

## 6. k340 convergence findings

The completed, authenticated `epsilon_J=0.11` k340 parent provides nested
prefixes 260, 280, 300, 320, and 340 from one Stage-1 parent. The maximum
mandatory-carrier correction increased from 0.714572 MHz for k300→k320 to
1.299836 MHz for k320→k340; seven of nine carrier increments grew.

[Figure D1](figures/D1_k340_spectral_convergence_v2.pdf) shows the signed and
absolute changes with the frozen strict PASS limits: 1.0 MHz for standard
carriers and the tighter 0.5 MHz limit for the cancellation-sensitive `f78`
splitting. Although every practical carrier comparison passes, the final
shell does not shrink and the frozen trend, nonmonotonicity, and
reversal/cancellation rules produce **RETAINED-BASIS CONVERGENCE: FAIL**.
The finite k340 endpoint is not ground truth.

## 7. Current limitations and next research steps

The present results are limited to finite retained bases at fixed N71 in the
current four-mode model. They do not establish convergence beyond k340,
spatial/domain convergence, completeness with respect to possible distributed
array modes, a calibrated physical microwave-port model, device validation,
or time-domain gate fidelity.

The next research step should remain a preregistered retained-basis study
informed by the authenticated k340 tail and rank attribution, without changing
thresholds after inspection. Gate simulation should resume only after a
defensible retained-basis plateau is established. Spatial/domain convergence,
omitted-mode sensitivity, and physical-port mapping must then be addressed
before any high-fidelity gate claim.

**GATE WORK: HOLD**  
**SPATIAL CONVERGENCE: HOLD**
