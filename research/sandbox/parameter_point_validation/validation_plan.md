# Independent parameter-point validation plan

## Claims kept separate

1. Parameter-point selector validation asks whether the exact frozen selector
   beats equal-cost controls at eps_J=0.11.
2. Multi-target transfer asks whether improvement covers all of 01, 23, and
   78 rather than only the primary mechanism.
3. Retained-basis convergence asks whether the new-point energy prefixes
   through k300 form a plateau. Agreement with the finite k300 endpoint cannot
   establish behavior beyond rank 299.

No outcome is interpreted as physical-device validation, spatial convergence,
or a gate-fidelity result.

## Parent, endpoints, spaces, and costs

One N71 k300 parent is solved at the frozen parameter point. Necessary nested
prefixes are 180,200,220,240,260,280,300. Prefixes 260/280/300 are mandatory
for the convergence verdict; 180/200/220/240 preserve the established trend
series and detect earlier reversals.

P is ranks 0–259. S1 is 260–279 and S2 is 280–299. Shell endpoints are
P+all S1 and the counterfactual P+all S2; the union endpoint is full k300.
Every retained rank includes all eight mode-4 states.

At budgets b=3,6,10,15, compare equal-cost:

- frozen physics selector;
- shell energy prefix;
- fixed farthest-point spread offsets
  `0,19,9,14,4,2,6,11,16,1,3,5,7,8,10,12,13,15,17,18`;
- the reflected reverse-spread order.

Union selections take b ranks from each shell (2b total). Also compare the
ordinary global energy prefix with 2b added ranks and the shell-balanced
prefix. Budget 10 is primary; 15 is confirmation; 3/6 measure efficiency.
No result may change a budget, ordering, group weight, or comparator.

## Mandatory metrics

For every selected/control space, save raw energies and calculate:

- all nine carriers f01,f07,f08,f78,f81,f72,f21,f03,f38;
- state overlaps for 0,1,2,3,7,8 and minimum singular values of subspaces
  01,23,78,0178, and all six targets;
- grid_phi edges g01,g07,g78,g81,g72,g21,g03,g38;
- Frobenius norms and every singular value of the six frozen operator blocks;
- scores and complete path ranks for 0781,0721,0381;
- common-offset-removed target energy errors and individual excitation-energy
  errors relative to state 0;
- cancellation flags and ratios for all mandatory carriers.

Original k300 strict/practical thresholds are carried forward unchanged:
standard errors 1/5 MHz strict and 5/10 MHz practical; f78 0.5/1 MHz strict
and 1/2 MHz practical; overlap 0.999/0.995 strict and 0.995/0.990 practical;
edge/block relative error 3%/5% strict and 5%/10% practical; score relative
error 5%/10% strict and 10%/20% practical. Assignment and numerical-quality
thresholds are unchanged. Centered and excitation-energy errors use the
existing standard 1/5 and 5/10 MHz thresholds; no looser threshold is added.

## Cancellation diagnostic

For candidate C and its shell endpoint R, define raw state error in MHz
`d_s=1000(E_s^C-E_s^R)` for s in 0,1,2,3,7,8. Remove the arbitrary common
offset `c=mean_s d_s` and report `z_s=d_s-c`. Define individual excitation
error `x_s=1000[(E_s^C-E_0^C)-(E_s^R-E_0^R)]=d_s-d_0` with x_0=0.
For transition i→j, `t_ij=x_j-x_i` (with the same signed orientation used by
the endpoint carrier).

Flag accidental cancellation when

    |t_ij| / (|x_i|+|x_j|) <= 0.25,

the denominator exceeds 1e-6 MHz, and either excitation error exceeds 1 MHz.
A flag is severe when either exceeds 5 MHz. Report the numerator, denominator,
ratio, both excitation errors, and ordinary transition classification.

A small transition error with a flag is not genuine shell recovery. Physics
selector PASS permits no such flag at budgets 10 or 15. A severe physics flag
is FAIL; a non-severe flag prevents PASS. Controls retain flags in the equal-
cost comparison so cancellation cannot make them appear scientifically
superior without disclosure. The old k300 BORDERLINE verdict is unchanged.

## Balanced families and verdicts

Six equal-standing families are used: logical01, comparator23, primary78,
cross-carriers, operators, and pathways. The first three now include their
group's centered-state and excitation-energy errors. Thus success requires
all three target pairs, while numerous 7/8 observables cannot outvote 2/3.

The exact PASS/BORDERLINE/FAIL and SUPPORTED/PARTIAL/NOT SUPPORTED precedence
is machine-readable in `criteria.json`. In brief:

- selector PASS requires all hard gates, no practical failure, strict b15
  success, >=20% b10 RMS improvement over the best equal-cost control in every
  family/context, no physics cancellation, no no-harm miss, no material b15
  regression, and preserved path ordering;
- BORDERLINE requires safe/no-practical-failure behavior and improvement over
  ordinary prefix everywhere but misses at least one PASS clause;
- FAIL is triggered by a hard/practical failure, failure of any family or the
  2/3 group to improve prefix, unsafe identity/path ordering, severe
  cancellation, hidden shell failure, or material budget regression.

Multi-target transfer is SUPPORTED only with selector PASS and <=0.80
best-control ratios for logical01/comparator23/primary78 in S1,S2,union with
no cancellation. It is PARTIAL only if all three improve ordinary prefix but
one misses the stronger condition or the selector is BORDERLINE. Otherwise it
is NOT SUPPORTED.

Retained convergence is classified independently using new-point prefixes.
It cannot be inferred from selector success and cannot establish ranks >299.

## Falsification conditions

The transfer claim is falsified by any unsafe mandatory state, feature-gate
failure, leakage of validation outcomes into selection, inability to reproduce
the sealed ranking, failure to improve the 2/3 family, a separate-shell
failure hidden by union cancellation, severe energy-error cancellation, or a
selector FAIL verdict. The point is not replaced after observing a failure.
