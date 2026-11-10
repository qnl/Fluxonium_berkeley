# Frozen k340 analysis plan

## Authentication and reconstruction

Verify the final COMPLETE-to-checkpoint link, configuration fingerprint,
payload hashes, selection seal/link, source hashes, environment, Stage-1
residuals, and orthogonality before analysis. Reconstruct the Stage-2
Hamiltonian in GHz using the existing rank-major/mode-4-minor formula and
`grid_phi = theta3 + theta2/2`. Hermiticity must be <=1e-12 and Stage-2
residuals <=1e-9 GHz.

Track the independently generated original k300 states to nested k300 using
raw-grid overlaps. Track nested k300 to k340 and every prefix to the k340
endpoint. Report assignments, assigned overlaps, best competitors, ratios,
margins, and all five subspace singular-value diagnostics. Unsafe tracking is
a hard failure, not permission to relabel by inspection.

## Prefix metrics

For k=260,280,300,320,340 report all frozen frequencies, f78-f01, target
energies, target state/subspace overlaps, eight grid_phi edges (four mandatory
plus four supporting), every singular value and Frobenius norm of all six
blocks, all three pathway scores/ranks, and residuals.

Classify k300-to-k320, k320-to-k340, and k300-to-k340 using the unchanged
strict/practical thresholds. Retain k260-to-k280 and k280-to-k300 as context.
For every scalar record the first-new-shell and final-shell error magnitudes
and whether the final value shrinks. For carriers record signed increments,
sign reversals, two-shell cancellation, and individual state excitation errors.

## Shell occupation and coupling

For S3 use P=ranks 0--299 and Q=300--319, with full P+Q=k320. For S4 use
P=0--319 and Q=320--339, with full P+Q=k340. For each target state evaluate

    q = (E I - H_QQ)^(-1) H_QP p

by a linear solve. Report q-vector error, exact signed self-energy,
P-space relaxation, observed shift, shell probability, Schur equation
residual, exact identity error, rank-resolved contribution/coupling/occupation,
and shell-eigenmode contribution/coupling/denominator/eta.

Near resonance is fixed as absolute denominator <=0.010 GHz or eta>=0.10.
Report top-three fraction, effective count, and counts reaching 50%, 80%, and
90% absolute attribution for ranks and invariant shell modes. Aggregate exact
absolute rank attribution over states 0,1,2,3,7,8. Rank attribution is
basis-dependent; mode attribution is invariant within the shell.

## Tail classification

Apply the A_ISOLATED/B_REPEATED/C_DISTRIBUTED_WEAK/
D_NONMONOTONIC_UNRESOLVED definitions verbatim from `criteria.json`. D has
precedence over B, then C, then A. Report the classification separately from
retained-basis convergence.

## Secondary selector

Validate S3, counterfactual isolated S4, and their union at budgets 3/6/10/15
against prefix, spread, and reverse-spread controls. Use the same six equal
weight families as the previous sealed analysis. The unchanged b10 20%
best-control improvement target and b15 confirmation logic produce a separate
PASS/BORDERLINE/FAIL selector result. It cannot rescue a retained-basis FAIL.

## Decision and reporting order

1. hard authentication/numerical gates;
2. retained-basis PASS/BORDERLINE/FAIL;
3. exact trend and cancellation findings;
4. tail-hypothesis class;
5. secondary selector verdict;
6. bounded scientific claims.

No figures are generated. Preserve the existing figure backlog:

1. k260 shell-attribution graphs;
2. k260 rank-addback/ablation graphs;
3. k300 held-out-selector comparisons;
4. cross-parameter selector validation.
