# Epsilon_J=0.11 N71 k340 retained-basis experiment

## Scientific question

Test whether the fixed-N71 retained Stage-1 sequence begins approaching a
plateau beyond k300. This is a retained-basis question inside one already
specified four-mode Hamiltonian. It is not a spatial, mode-count, physical-port,
gate, or model-validation experiment.

The authenticated k280-to-k300 shell changed f07, f08, f81, f72, and f38 by
approximately 52--53 MHz. The experiment therefore treats k300 as a finite
historical endpoint and never as ground truth.

## Frozen parent

- Physical parameters (GHz conventions inherited unchanged): EJ=5, EC=0.5,
  EL=1, ELK=1, EJS=4, ECS=8, eC=5.5, eP=10, epsilon_J=0.11,
  epsilon_LK=0.05, ng=0, phi_ext=0, theta_ext=pi.
- Spatial settings: n1max=4, N2=N3=71, L2=11, L3=14, N4=8.
- Stage-1 solver: k=340, sigma=-24, which=LM, tol=1e-8, ncv=None,
  v0=None, explicit shift-invert with MMD_AT_PLUS_A.
- Configuration fingerprint:
  `10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0`.
- Same-parent prefixes: 260, 280, 300, 320, 340.
- New shells: ranks 300--319 (S3) and 320--339 (S4).

Prefixes 180/200/220/240 are not regenerated. They are unnecessary for the
new decision because authenticated same-parent 260/280/300 data already record
the earlier trend, while the new parent includes 260 and 280 for direct
same-parent context. Adding smaller prefixes would increase Stage-2 analysis
without changing the preregistered post-k300 decision.

## Endpoint distinction and tracking

The analysis must report four distinct objects:

1. authenticated independently generated k300 endpoint;
2. nested k300 inside the future k340 parent;
3. nested k320 inside that parent;
4. k340 endpoint.

No bitwise equality is assumed between the two k300 Stage-1 eigensystems.
Physical state labels transfer by raw-grid maximum-total-absolute-overlap
assignment. All eleven low states must satisfy assigned overlap >=0.90,
competitor/assigned ratio <0.50, and margin >0.25. The six mandatory states
0,1,2,3,7,8 are reported individually and in the five frozen subspaces.

## Mandatory measurements

- f01, f07, f08, f78, f81, f72, f21, f03, f38, and f78-f01;
- state energies and overlaps for 0,1,2,3,7,8;
- subspace singular values for 01, 23, 78, 0178, and 012378;
- g01, g07, g78, g81, with g72/g21/g03/g38 retained as supporting checks;
- Frobenius norms and every singular value of the six frozen grid_phi blocks;
- 0781, 0721, and 0381 scores and pathway ordering;
- k300-to-k320, k320-to-k340, and k300-to-k340 comparisons;
- earlier k260-to-k280 and k280-to-k300 increments for context;
- parent residuals, orthogonality, Stage-2 residuals, and Hermiticity;
- signed trend, reversal, state-error cancellation, and f78-f01 diagnostics.

The strict and practical tolerances are copied exactly from the sealed
epsilon_J=0.11 k300 criteria. No threshold is loosened.

## Frozen primary verdict

PASS requires all hard gates and strict PASS for every mandatory component in
all three primary comparisons, non-growing final-shell error magnitude, and no
meaningful reversal or cancellation.

BORDERLINE requires all hard gates, no practical FAIL in any primary
comparison, a final k320-to-k340 step no worse than strict BORDERLINE for every
mandatory component, shrinking trends, and no unresolved reversal or
cancellation, while at least one strict condition is missed.

FAIL has precedence. It is triggered by any hard-gate failure, any practical
FAIL in any primary comparison, any final-shell strict FAIL, a nonshrinking
historically problematic carrier, or meaningful reversal/cancellation. Thus a
small k320-to-k340 increment cannot erase a large k300-to-k320 jump.

## Tail-hypothesis diagnostic

Each new shell receives an exact Schur/Feshbach diagnostic using signed
denominators and linear solves: exact self-energy, P-space relaxation, shell
occupation, rank and shell-eigenmode couplings, denominators, eta, near-resonant
weight, rank/mode concentration, and aggregate six-state attribution.

The prespecified tail classes are:

- A_ISOLATED: the historical 280--299 shell remains the only influential
  shell; neither new shell is influential and the final error envelope does
  not grow.
- B_REPEATED: at least one new shell is influential.
- C_DISTRIBUTED_WEAK: neither shell individually practical-fails, but the
  cumulative result misses strict tolerance, both shells are distributed
  (top-three <=50%, at least eight ranks for 80%), and the final envelope has
  not decayed by a factor of two.
- D_NONMONOTONIC_UNRESOLVED: hard/tracking/Schur failure, meaningful reversal
  or cancellation, or conflicting evidence prevents a unique A/B/C result.

An influential shell is fixed as one with any practical mandatory FAIL or a
target-state absolute excitation shift above 10 MHz. These definitions do not
assume that ranks near 300 resemble ranks 290/292.

## Secondary selector diagnostic

The exact frozen selector source, hash, objective, pair balancing, floors,
regularization, tie rule, budgets 3/6/10/15, and zero grid_phi selection weight
are unchanged. It operates independently on S3 and S4 with nested k300 as the
base. Selection is written and sealed before any candidate-containing or full
new-shell dressed validation. Its verdict is separate and cannot change the
retained-basis verdict.

## Claim boundary

A measured smaller final increment is only a k320-to-k340 observation. A PASS
may support a plateau within the tested fixed-N71 prefix range. No outcome can
establish convergence beyond k340, k340 as ground truth, or convergence in
N2/N3/L2/L3/N4/n1max, omitted modes, Hamiltonian choice, or experimental
calibration.

**GATE WORK: HOLD**

**SPATIAL CONVERGENCE: HOLD**
