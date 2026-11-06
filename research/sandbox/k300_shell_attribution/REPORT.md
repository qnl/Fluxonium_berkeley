# Gridium N71 k300 final-shell Schur attribution

## Decision

**DATA FEASIBILITY: SUFFICIENT**  
**SCHUR RECONSTRUCTION: PASS**  
**DOMINANT SHELL MECHANISM: B — a few off-resonant Stage-1 ranks**

The large k280→k300 response is caused principally by ranks 292 and 290,
with rank 287 a much smaller third contributor. These are not near-resonant
states: all six required states have zero near-resonant second-order weight,
minimum shell denominators of 36.84–43.09 GHz, maximum coupling/denominator
ratio 0.01779, and shell probabilities at most 0.001203. The effect is large
because ranks 292/290 couple strongly and coherently to the dressed target
subspace, especially states 7/8, and because their joint exact addback is
substantially nonadditive.

k300 remains an endpoint, not ground truth. This attribution explains the
observed finite-shell correction; it does not establish retained-basis or
spatial convergence.

## Authentication and reconstruction

Fresh hash and numerical checks passed without a Stage-1 eigensolve:

- COMPLETE, checkpoint `a48ab94e…f1145f`, configuration `b7596cd4…4760b`,
  all payload hashes, and nested prefixes 180/200/220/240/260/280/300 passed.
- Frozen selection `ef27780c…a024e`, seal `b82151d7…47bb9`, and the checkpoint
  linkage showing selection before full Stage 2 passed.
- N51, k180, k220, and k260 checkpoint references remained unchanged.
- Maximum saved Stage-1 residual was 3.259e-12; orthogonality infinity residual
  was 1.562e-11.
- The reconstructed 2400-dimensional Stage-2 Hamiltonian had Hermiticity error
  below reported floating precision. Reconstructed k300 energies/grid_phi
  magnitudes differed from saved values by 1.990e-13 GHz/7.114e-13; k280 by
  1.066e-13/3.898e-13. The k260 auxiliary reconstruction also passed.
- Every Schur q-vector, effective-equation, and exact shift identity met the
  frozen 1e-8/1e-9 tolerances. All state assignments were safe.

The methodology was frozen before new attribution at SHA-256
`b5e65484dfe546755199a4606d4e27aa1145be36e1aca7d4eae413ccd69d6dbf`.
P is ranks 0–279 (2240 dimensions), Q ranks 280–299 (160), with all eight
mode-4 states per rank.

## Exact state attribution

Energies and contributions are MHz. `Σ(E)` is the exact shell self-energy;
`relax` is the exact P-space relaxation term. Their sum reconstructs ΔE.
Counts are the minimum number of absolute contributions reaching 50/80/90%.
Mode counts use invariant dressed-shell modes; rank counts refer to the saved
Stage-1 eigenbasis. All six fixed-vector second-order diagnostics satisfy the
predefined controlled-PT check, but the exact identity is authoritative.

|state|ΔE|Σ(E)|relax|second order|SO error|Q probability|min |den.| (GHz)|max η|ranks 50/80/90|modes 50/80/90|class|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|0|-2.301483|-2.669755|+0.368271|-1.985575|+0.315909|0.0000430|43.093|0.00373|2/2/4|2/3/4|B|
|1|-2.331636|-2.704295|+0.372658|-2.011960|+0.319676|0.0000436|43.077|0.00380|2/2/4|2/3/5|B|
|2|-1.101855|-1.292624|+0.190769|-0.940786|+0.161069|0.0000216|39.645|0.00191|2/3/5|3/6/10|B|
|3|-1.484185|-1.743525|+0.259340|-1.265756|+0.218430|0.0000293|39.416|0.00209|2/3/5|3/6/9|B|
|7|-54.873826|-67.171901|+12.298075|-44.862299|+10.011528|0.0012031|36.850|0.01779|2/2/3|2/3/4|B|
|8|-52.652829|-63.589627|+10.936798|-43.475016|+9.177813|0.0011390|36.836|0.01750|2/2/3|2/3/4|B|

The large carrier shifts are therefore not pure second-order sums. For states
7 and 8, exact negative self-energies of -67.17 and -63.59 MHz are partly
offset by +12.30 and +10.94 MHz of P-space relaxation. The second-order
estimates understate the exact shifts by 10.01 and 9.18 MHz.

## Transition attribution

All entries are signed k280→k300 changes in MHz.

|transition|observed|Σ difference|relax difference|second order|SO error|
|:---|---:|---:|---:|---:|---:|
|f01|-0.030153|-0.034540|+0.004387|-0.026386|+0.003768|
|f07|-52.572343|-64.502147|+11.929804|-42.876724|+9.695619|
|f08|-50.351346|-60.919872|+10.568527|-41.489441|+8.861904|
|f78|+2.220997|+3.582275|-1.361277|+1.387283|-0.833714|
|f81|-50.321193|-60.885332|+10.564139|-41.463056|+8.858137|
|f72|-53.771972|-65.879277|+12.107306|-43.921513|+9.850459|
|f21|+1.229782|+1.411671|-0.181889|+1.071174|-0.158608|
|f03|+0.817298|+0.926229|-0.108931|+0.719819|-0.097479|
|f38|-51.168644|-61.846102|+10.677458|-42.209260|+8.959384|

## Which ranks and shell modes matter

Across states 0/1/2/3/7/8, aggregate absolute exact rank self-energy is
139.172 MHz. Rank 292 contributes 62.496 MHz (44.91%), rank 290 contributes
58.236 MHz (41.84%), and rank 287 contributes 5.990 MHz (4.30%). Thus 292/290
carry 86.75% and the top three carry 91.05%. Rank 299 contributes only
0.00563 MHz (0.0040%): the finite k300 boundary rank itself is inactive in
this diagnostic.

Per-state leading exact rank contributions (MHz) are:

|state|rank 1|rank 2|rank 3|rank 4|rank 5|
|---:|:---|:---|:---|:---|:---|
|0|290 -1.170|292 -1.085|286 -0.140|287 -0.133|289 -0.077|
|1|292 -1.258|290 -1.024|287 -0.144|286 -0.136|289 -0.077|
|2|292 -0.536|290 -0.380|286 -0.149|282 -0.091|293 -0.043|
|3|290 -0.630|292 -0.573|286 -0.214|282 -0.126|293 -0.056|
|7|292 -30.001|290 -29.220|287 -2.623|289 -2.449|286 -2.208|
|8|292 -29.043|290 -25.811|287 -3.083|289 -2.519|286 -2.247|

The leading invariant shell modes are also concentrated. For state 7, the
four largest exact modal contributions are -24.643, -19.250, -10.532, and
-7.198 MHz, with fixed-vector denominators -52.99, -68.67, -37.02, and
-84.70 GHz and couplings 0.942, 0.937, 0.523, and 0.612 GHz. For state 8 the
corresponding values are -23.304, -18.289, -11.073, and -6.082 MHz, with
denominators -52.83, -68.83, -37.10, and -84.49 GHz and couplings 0.925,
0.922, 0.541, and 0.566 GHz. The dominant shell modes are themselves largely
supported on ranks 290/292. No pole or avoided-crossing interpretation is
needed.

The couplings are the coherent sum of the unchanged Stage-2 terms
`8 EC[i,3] n_i⊗p4`, `K[0,2] x2⊗x4`, and `K[1,2] x3⊗x4`; individual channel
norms are not additive energy corrections. The frozen isolated-rank features
correctly exposed the combined strength: for rank 292, coupling norms for
target pairs 01/23/78 were 0.333/0.218/1.531 GHz; for rank 290 they were
0.320/0.205/1.467 GHz. Their minimum positive selector denominators were
44.07/40.51/37.82 and 43.99/40.44/37.75 GHz, respectively.

## Nonadditivity

Splitting Q into Q1=280–289 and Q2=290–299 shows that Q2 generates most of
the outer-carrier response, but the pieces are not exactly additive.

|transition|Q1 alone|Q2 alone|full|total nonadditivity|explicit Q1–Q2|P-mediated remainder|
|:---|---:|---:|---:|---:|---:|---:|
|f07|-5.435592|-46.159986|-52.572343|-0.976765|-0.585546|-0.391220|
|f08|-6.453123|-43.251118|-50.351346|-0.647105|-0.583500|-0.063605|
|f78|-1.017532|+2.908868|+2.220997|+0.329660|+0.002046|+0.327615|
|f21|+0.036334|+1.165687|+1.229782|+0.027761|+0.014653|+0.013107|
|f03|-0.094355|+0.887945|+0.817298|+0.023708|+0.010852|+0.012856|

For f78, the explicit Q1–Q2 block changes the result by only +0.002 MHz;
nearly all +0.330 MHz nonadditivity is P-mediated relaxation/energy dependence.
For f07/f08, explicit Q1–Q2 coupling contributes about -0.586 MHz.

Exact addbacks confirm stronger within-Q2 interaction. Ranks 292 and 290
alone shift f07 by -19.091 and -18.575 MHz, but together give -45.975 MHz:
their pair interaction is -8.310 MHz. For f08 the analogous interaction is
-7.688 MHz. For f78, rank 292 alone gives +0.0555 MHz and rank 290 +2.2661
MHz, while the pair gives +2.9435 MHz, including +0.6218 MHz interaction.
Adding rank 287 produces f07/f08/f78 shifts of -47.832/-45.302/+2.530 MHz,
about 91.0%/90.0%/113.9% of the complete final-shell changes. This overshoot
is why exact rank addback must not be conflated with signed Schur attribution.

## Frozen selector interpretation

The selector identifies genuine virtual contributions, not merely favorable
endpoint cancellation:

- S1: its first five ranks overlap four of the exact top five. Budgets
  3/6/10/15 capture 69.53/85.28/94.02/98.54% of aggregate absolute exact
  isolated-shell rank weight.
- S2: its first five ranks are exactly the exact-attribution top five as a set.
  Budgets 3/6/10/15 capture 90.50/98.70/99.46/99.77%. It ranks 292 and 290
  first and second because both provide strong, denominator-weighted coverage
  across all three target pairs, not only 7/8. Rank 286 supports all groups;
  rank 287 is strong in 01/78 but weak in 23; ranks 289 and 282 restore
  complementary coverage.
- At budget 10, frozen validation errors were S1 f07/f08/f78
  +0.0209/+0.0873/+0.0663 MHz and S2 +0.2359/+0.3904/+0.1546 MHz. The union
  physics selection reduced the ordinary prefix's f07/f08/f78 errors from
  +52.572/+50.351/-2.221 MHz to +0.257/+0.486/+0.230 MHz. All required
  states, operator blocks, and pathways were strict PASS.

This supports the transfer of the coupling/denominator coverage rule from the
training analysis into both held-out shells of the same k300 parent. It is not
independent device-parameter generalization.

## Why reverse spread wins only S2/b10 f78

The frozen physics set is
`292,290,286,287,289,282,293,288,280,291`; reverse spread uses
`280,283,285,288,290,293,295,297,298,299`. They share only 280/288/290/293.
Physics uniquely includes 292/286/287/289/282/291; reverse uniquely includes
283/285/295/297/298/299.

Relative to the isolated S2 endpoint, physics leaves state-7 and state-8
energy errors +0.249718 and +0.404279 MHz. Their difference is the preregistered
+0.154561 MHz f78 error. It is a small systematic undercorrection of both
states, slightly larger for state 8.

Reverse spread leaves enormous but nearly equal state errors +35.238875 and
+35.209341 MHz. These cancel to -0.029534 MHz in f78: 99.956% of the combined
f07/f08 error magnitude cancels, while f07/f08 themselves remain wrong by
+33.833667/+33.804133 MHz. Its reverse-only ranks contribute just -0.0736 and
-0.1337 MHz of isolated Schur self-energy to states 7/8; it succeeds on f78
by common-mode cancellation after omitting rank 292 and most important ranks,
not by reproducing the shell.

Thus the BORDERLINE verdict is preserved exactly. The miss exposes a narrow,
transferable limitation—the positive matrix-coverage objective does not target
the sign of a small differential splitting—but not a broad failure to find
important virtual states. Retrospectively adding a signed f78 term would be
overfitting and is not justified before independent testing.

## Logical/middle spectral collision

|k|f01 (MHz)|f78 (MHz)|f78-f01 (MHz)|g01|g78|f07-f81 (MHz)|
|---:|---:|---:|---:|---:|---:|---:|
|260|15.855811|14.568530|-1.287281|0.447948|1.155801|+1.287281|
|280|15.862073|13.632570|-2.229503|0.447897|1.156882|+2.229503|
|300|15.831920|15.853567|+0.021648|0.440333|1.232142|-0.021648|

The final shell barely changes f01 (-0.0302 MHz) but moves f78 by +2.2210
MHz, reversing the ordering and placing the two transitions within 0.02165
MHz. It also changes g01 from 0.44790 to 0.44033 and g78 from 1.15688 to
1.23214. Algebraically, the outer separation is the negative of f78-f01, so
f07 and f81 are equally close at this endpoint.

This is a spectral near-degeneracy. The nonzero and unequal grid_phi matrix
elements make driven crosstalk a concern, but frequency proximity plus static
matrix elements do not determine pulse-selective dynamics or time-domain gate
fidelity. No gate impossibility or validated X90/X180 conclusion follows.

## Scientific decision and next experiment

1. The large correction is a resolved, off-resonant virtual-shell response
   dominated by strong coherent couplings through ranks 292/290, amplified by
   exact relaxation and rank interaction—not a near-resonant pole.
2. It is concentrated by Stage-1 rank and by shell mode, not distributed.
3. The selector succeeds because its multi-target whitened coverage identifies
   those coupling/denominator blocks and complementary 01/23/78 directions.
4. The S2/f78 exception is primarily cancellation in the reverse-spread
   control plus a small differential undercorrection by physics selection.
5. Further selector fitting is not justified before an independent parameter-
   point test.
6. Rank 299 itself is inactive, so the observed boundary does not by itself
   demand another parent. A larger parent is still eventually required for a
   retained-basis convergence claim because ranks beyond 299 are unobserved.

A k320 run would expose only one more 20-rank increment and could repeat the
same endpoint ambiguity. k340 would provide two increments and is the better
future same-point convergence experiment, but it would not test parameter-
point transfer. Further selected-rank diagnostics now have diminishing value,
and changing the selector after this result would confound validation.

**NEXT RESEARCH ACTION:** preregister one independent N71, fixed-four-mode
parameter-point validation that applies the selector formula unchanged to two
predeclared 20-rank held-out shells, with the parameter point and all budgets
frozen before any candidate-space dressed outcome is inspected.

**GATE WORK: HOLD**  
**SPATIAL CONVERGENCE: HOLD**

The pending publication figure sets for (1) k260 shell attribution and (2)
k260 rank-addback/ablation remain documented in the existing k300 figure plan;
no figures were generated here.

## Artifacts and preservation

Machine-readable results are in `summary.json`; rank and invariant shell-mode
tables are in `rank_attributions.csv` and `shell_mode_contributions.csv`;
selector alignment is in `selector_diagnostics.csv`; the collision series is
in `frequency_selectivity.csv`. `preservation.json` records before-analysis
hashes. No tracked production file or authenticated checkpoint was changed,
and no commit or push was made.
