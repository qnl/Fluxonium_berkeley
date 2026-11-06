# Exact frozen-selector provenance

## Algorithm, not old rank lists

The reusable object is the deterministic algorithm in
`research/sandbox/k300_preregistration/selector.py`, SHA-256
`57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2`.
Its mathematical specification is `selector_method.md`, SHA-256
`2055c7d54d81ada97379f4941ef08354dbb3cbf9378aa2b10dde1349ca2ff7cc`.
The parameter-point adapter imports that exact file after checking the hash;
it does not copy or edit the score.

Previously successful ranks 223/225/230 and 290/292 are evidence about old
parents only. They are forbidden selection inputs and are not encoded in any
new ordering. New-point ranks are determined exclusively from allowed
new-point Stage-1 values/projections after the future Stage-1 solve.

## Frozen mathematics

For each shell rank r and target pair g in `(0,1),(2,3),(7,8)`, use the
new-point P=k260 dressed pair `U_g`, its mean energy `mu_g`, the isolated
eight-dimensional candidate block `D_r`, and coupling `B_rg=H_QrP U_g`:

    A_rg = B_rg^dagger (D_r - mu_g I)^(-1) B_rg.

For each pair, `W_g=sum_r A_rg`. If `W_g=V diag(lambda) V^dagger`, the inverse
square-root uses the fixed floor
`max(1e-14 GHz, 1e-10 lambda_max(W_g))`. Define
`C_rg=W_g^(-1/2) A_rg W_g^(-1/2)` and

    F(S) = (1/6) sum_g log det[I_2 + 6 sum_(r in S) C_rg].

Greedily add the rank giving the largest increase in F. Gains tied within
absolute 1e-12 go to the lower rank. Budgets 3/6/10/15 are prefixes of the
complete ordering. S1 and S2 are ranked independently against the same P260
target states and are never refreshed using validation-shell dressed results.

The factor 1/6, scale 6, equal target-pair weights, groups, floors, budgets,
candidate-block construction, and tie rule are unchanged. `grid_phi` has zero
selection weight. All Hamiltonian coupling terms enter coherently. The
selector fails closed if any candidate Stage-1 gap is <=1e-6 GHz, a positive
denominator is <=0.010 GHz, eta is >=0.10, or target tracking is unsafe.

## New-point target identity

The selector must use new-point P260 dressed states; original-parameter
vectors cannot substitute for them. Before selection, those P260 states are
labeled through raw-basis overlap to the authenticated original-parameter
k260 states. This uses the shared physical grid only to transfer labels.
All eleven low states must pass the existing assignment gates. Energy and
operator equality with the original point are neither expected nor tested.

This label-transfer adapter does not change the score's scientific semantics.
It is necessary because the original same-parameter hook required numerical
reconstruction of the old k260 endpoint, which would be invalid at eps_J=0.11.

## Leakage barrier

The future hook may use only:

- the completed new-point Stage-1 eigenvalues, vectors, and projections;
- new-point P260 Stage-2 states;
- isolated 8x8 candidate-rank blocks and their couplings to P260;
- the authenticated original k260 states solely for label transfer.

It exclusively writes and hashes both shell orderings before any P+candidate,
shell endpoint, or full-k300 new-point Stage-2 diagonalization. The checkpoint
records the selection hash and freeze timing. Full/selected validation errors,
operator outcomes, exact shell attribution, and old rank identities are
forbidden selection inputs.

**SELECTOR REUSE: EXACT.** The adapter changes only parameter-point state
labeling and provenance; the frozen mathematical implementation is reused
byte-for-byte.
