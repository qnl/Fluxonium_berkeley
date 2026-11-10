# N71 epsilon_J=0.11 k380 scientific preregistration

Date: 2026-10-11. Preregistration only; execution requires separate authorization.

## Question and fixed configuration

Do influential off-resonant virtual directions recur above rank 340, or does
the retained-basis tail become smaller and more distributed? No outcome is assumed.
Use the complete authenticated k340 configuration, changing only solver k to 380:
epsilon_J=0.11, epsilon_LK=0.05, N2=N3=71, L2=11, L3=14, N4=8, n1max=4;
sigma=-24 GHz, which=LM, tol=1e-8, ncv=None, v0=None, MMD_AT_PLUS_A.
All physical parameters, biases, tensor order, Hamiltonian terms and units are
specified exactly in criteria.json. Fingerprint uses the runner's canonical JSON.

One parent supplies nested prefixes 300,320,340,360,380. New shells are
S1=340–359 and S2=360–379 (zero-based Stage-1 ranks).
Primary comparisons: 340→360, 360→380, 340→380. Context: 300→320 and 320→340,
reported both from the new parent and from the authenticated old parent with
source parent explicitly labeled. Original k340 to nested k340 tracking is mandatory.

## Competing hypotheses

Apply the exact prospective rules in criteria.json and evaluate.py, precedence D,B,C,A:

- A: no renewed concentrated shell, both new aggregate absolute weights below
  the historical 320–339 weight, and a nonincreasing final error envelope.
- B: concentrated off-resonant rank influence recurs in either new shell:
  top-three fraction >=70%, at most five ranks to reach 80%, and aggregate
  near-resonant exact-mode weight fraction <10%. Report absolute size separately.
- C: both new shells are distributed (top-three <=50%, at least eight ranks to
  reach 80%), below the historical weight, with S2 no larger than S1 and a
  nonincreasing final error envelope.
- D: hard/tracking/Schur failure, meaningful reversal/cancellation, any growing
  mandatory error, or remaining conflicting/unclassified evidence.

These new prospective labels are adapted to the rank-328-dominated historical
shell, whose aggregate weight was 4.074080434450945 MHz. They do not rewrite
the frozen k340 labels. The earlier influential-shell diagnostic (practical FAIL
or raw state shift >10 MHz) remains a separately reported flag. Requiring that
large-size flag for B would miss recurrence of a small but concentrated shell.
All concentration/resonance numbers come from the validated predecessor methodology.

## Secondary selector

Enable the optional secondary diagnostic prospectively for this run; do not
toggle it after outcomes. Use the exact hash-pinned k300 selector implementation,
unchanged target pairs, feature construction, whitening, group weights, floors,
regularization, log-determinant objective, rank tie breaks, and budgets 3/6/10/15.
Its base is nested P340; each new shell has twenty candidates. No rank-328 list
or any historical rank list selects new ranks.

The future execution hook reconstructs authenticated original P340, tracks
nested P340, and freezes both new rank orders and all selected spaces before
any candidate-containing/full-new-shell dressed diagonalization. Actual selected
identities cannot exist before the future Stage-1 parent; the method and this
chronology are sealed now. If feature validity or rank-gap checks fail, stop;
do not refit or bypass the selector. Validation verdict is separate from convergence.

## Claim boundaries

A PASS means a plateau only over the tested 340/360/380 retained prefixes at
fixed N71 and fixed four-mode model. It does not establish an infinite-basis
limit, spatial/domain/N4/n1max convergence, omitted-mode adequacy, physical-device
validation, a microwave-port map, or a validated X90/X180 gate. Gate work and
spatial convergence remain HOLD. No k400 or larger run is authorized.
