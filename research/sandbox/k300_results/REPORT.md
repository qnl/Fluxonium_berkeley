# Gridium N71 k300 held-out retained-basis validation

## Final decisions

- **EXECUTION: SUCCESS**
- **AUTHENTICATION: PASS**
- **SELECTOR VALIDATION: BORDERLINE**
- **RETAINED-BASIS CONVERGENCE: FAIL**
- **MULTI-TARGET GENERALIZATION: SUPPORTED**
- **OVERFITTING CONCERNS: REMAINING**
- **GATE WORK: HOLD**
- **SPATIAL CONVERGENCE: HOLD**

k300 is a finite retained-basis endpoint, not ground truth.

## Execution and authentication

The single guarded launch completed and atomically published the checkpoint.
Internal runner time was 360.227 s; external wall time was 1004.16 s.
Factorization took 61.979 s and eigsh 265.188 s. Matrix construction reported
0.120 s and projection 1.296 s. `/usr/bin/time -l` could not query the macOS
kernel after completion, so peak RSS is unavailable. Memory throttling stayed
zero while monitored.

COMPLETE, profile `n71-k300`, configuration fingerprint, checkpoint/payload
hashes, exact prefixes 180/200/220/240/260/280/300, residuals and orthogonality
all authenticate. Maximum eigenpair residual is 3.259e-12 and orthogonality
infinity residual 1.562e-11. N51, N71-k180, k220 and k260 authenticate unchanged.

The selection SHA-256 is
`ef27780c0c8b684f50b611774d2b5b7c61adaa5e076d9c7bc86d015af59a024e`.
The checkpoint records that it was frozen before full Stage-2 evaluation.

The sealed analyzer had one mechanical criteria-key typo after completing its
finite solves but before writing output. The sealed file was preserved. A
recorded runtime wrapper changed only `stage1_orthogonality_inf` to the existing
criteria key `stage1_orthogonality_inf_max`; no threshold, space, result, or
classification rule changed.

## Frozen selected ranks

S1 budget 10: `279,267,268,266,276,261,262,269,265,274`.

S1 budget 15 additionally: `277,260,270,271,275`.

S2 budget 10: `292,290,286,287,289,282,293,288,280,291`.

S2 budget 15 additionally: `285,296,297,281,298`.

The important future-shell ranks were selected from allowed pre-validation
features. In particular, ranks 290/292 were selected first in S2; no old rank
list was replayed.

## Nested prefix spectrum

Frequencies are MHz.

| k | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 | f78-f01 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|180|15.857270|6275.014071|6297.747508|22.733437|6281.890238|2819.000502|3440.156299|3688.042355|2609.705153|+6.876168|
|200|15.924131|6273.719453|6294.568288|20.848834|6278.644156|2819.203767|3438.591555|3685.779303|2608.788985|+4.924703|
|220|15.908195|6272.819042|6292.853928|20.034886|6276.945733|2822.096724|3434.814124|3681.799676|2611.054252|+4.126691|
|240|15.864131|6246.176785|6261.563920|15.387135|6245.699789|2796.971743|3433.340911|3679.790146|2581.773775|-0.476995|
|260|15.855811|6243.253114|6257.821644|14.568530|6241.965833|2794.963818|3432.433485|3678.193165|2579.628479|-1.287281|
|280|15.862073|6243.055864|6256.688434|13.632570|6240.826361|2795.614153|3431.579637|3677.182032|2579.506402|-2.229503|
|300|15.831920|6190.483521|6206.337088|15.853567|6190.505168|2741.842182|3432.809419|3677.999330|2528.337758|+0.021648|

The outer separation `f07-f81` is exactly the negative of `f78-f01`; at k300
it is -0.021648 MHz. The outer carriers and f01/f78 are therefore nearly
degenerate in this endpoint.

## Adjacent shell changes

| step | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|180->200|+0.066862|-1.294617|-3.179220|-1.884603|-3.246082|+0.203265|-1.564745|-2.263052|-0.916168|
|200->220|-0.015937|-0.900411|-1.714360|-0.813949|-1.698423|+2.892956|-3.777431|-3.979627|+2.265267|
|220->240|-0.044064|-26.642257|-31.290008|-4.647750|-31.245944|-25.124981|-1.473213|-2.009530|-29.280477|
|240->260|-0.008320|-2.923671|-3.742276|-0.818605|-3.733956|-2.007925|-0.907426|-1.596980|-2.145296|
|260->280|+0.006262|-0.197250|-1.133211|-0.935960|-1.139472|+0.650335|-0.853847|-1.011134|-0.122077|
|280->300|-0.030153|-52.572343|-50.351346|+2.220997|-50.321193|-53.771972|+1.229782|+0.817298|-51.168644|

The final shell is not a small tail. It dominates f07/f08/f81/f72/f38 and
reverses f01/f78/f72/f21/f03. f21 and f03 meet the frozen accidental-
cancellation flag; their small total k260->k300 changes hide opposing shell
increments. f78 reverses sign without satisfying the numerical cancellation
ratio.

## Physics-selected carrier errors versus each endpoint

Signed errors are MHz. Every entry below is strict PASS.

| context/budget | f01 | f07 | f08 | f78 | f81 | f72 | f21 | f03 | f38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|S1/10|-0.000414|+0.020931|+0.087277|+0.066346|+0.087691|-0.007108|+0.028453|+0.032987|+0.054291|
|S1/15|+0.000009|+0.002592|+0.035892|+0.033299|+0.035883|+0.000561|+0.002023|+0.003594|+0.032298|
|S2/10|+0.000571|+0.235868|+0.390430|+0.154561|+0.389859|+0.229318|+0.005980|+0.014028|+0.376402|
|S2/15|+0.000400|+0.112821|+0.166533|+0.053711|+0.166133|+0.117018|-0.004597|-0.003020|+0.169552|
|union/10|+0.000316|+0.256810|+0.486388|+0.229577|+0.486072|+0.219604|+0.036890|+0.049642|+0.436746|
|union/15|+0.000436|+0.115495|+0.200529|+0.085034|+0.200093|+0.117470|-0.002412|+0.000470|+0.200059|

At the primary budget 10, the ordinary union energy prefix has errors
f07/f08/f78/f21/f03 = +52.572/+50.351/-2.221/-1.230/-0.817 MHz. The physics
selection gives +0.257/+0.486/+0.230/+0.037/+0.050 MHz. The shell-balanced
prefix and spread controls also retain tens-of-MHz outer-carrier errors.

The sole selector-PASS miss is S2/b10 f78: physics error is +0.154561 MHz,
while the reverse-spread control happens to give -0.029534 MHz. That control
simultaneously misses f07/f08 by +33.834/+33.804 MHz. The frozen component
no-harm rule nevertheless makes the overall selector verdict BORDERLINE.

## Multi-target, tracking, operators and pathways

| context/budget | minimum target overlap | minimum required subspace SV | maximum edge error | maximum block error | maximum path-score error |
|---|---:|---:|---:|---:|---:|
|S1/10|0.999999002|0.999989609|0.0161%|0.0195%|0.0219%|
|S1/15|0.999999661|0.999997186|0.0042%|0.0042%|0.0059%|
|S2/10|0.999996187|0.999984401|0.0417%|0.0417%|0.0478%|
|S2/15|0.999998417|0.999994311|0.0349%|0.0349%|0.0465%|
|union/10|0.999994985|0.999977824|0.0241%|0.0426%|0.0296%|
|union/15|0.999998114|0.999993423|0.0282%|0.0282%|0.0368%|

All assignments are safe and all operator/path components are strict PASS.
At k300 the endpoint scores are 0781=4.197953 (rank 1), 0721=0.326815
(rank 4), and 0381=0.245346 (rank 5). Union/b10 reproduces them with relative
errors 0.0188%, 0.0296%, and 0.0176%; union/b15 errors are 0.0065%, 0.0168%,
and 0.0368%. The primary mechanism remains numerically identifiable.

The k280->k300 prefix comparison is very different: state-7/8 overlaps are
0.998963/0.997201, g78 changes 6.108%, the internal-{7,8} block changes 6.108%,
and the primary score changes 6.409%. Identities remain trackable, but the
unselected prefix is not converged.

At budget 10, physics-family RMS errors relative to the best equal-cost control
are at most 0.380 for S1, 0.141 for S2, and 0.195 for the union. The states-2/3
family ratios are 0.333, 0.0153, and 0.1166, respectively. Multi-target
coverage therefore transfers beyond states 7/8.

## Accuracy per retained Stage-1 rank

Mean reduction in absolute error across all nine carriers, MHz per added rank:

| context/budget | physics | ordinary prefix | best relevant comparison |
|---|---:|---:|---:|
|S1/10|0.0629|0.0261|spread 0.0500|
|S1/15|0.0437|0.0179|spread 0.0435|
|S2/10|2.8968|0.3227|reverse spread 1.0089|
|S2/15|1.9386|1.9436|reverse spread 1.7772|
|union/10|1.4434|-0.0027|balanced prefix 0.1695|
|union/15|0.9670|0.1060|balanced prefix 0.9568|

The b15 S2 prefix catches the dominant region and becomes marginally more
efficient on this mean, while physics remains strict-accurate and nested.
The primary preregistered b10 comparison strongly favors the selector.

## Scientific interpretation

**Selector improvement:** strong across all six frozen families and all six
targets. The held-out result supports the coupling/denominator coverage rule.

**Agreement with k300:** every mandatory physics-selected component at b10 and
b15 is strict PASS. Overall validation is still BORDERLINE because the frozen
single-component no-harm clause is missed by S2 f78.

**Retained-basis convergence:** FAIL. k280 is not close to k300 on important
carriers or the internal 7/8 operator block, increments grow, and multiple
sign reversals/cancellations occur. Nothing here establishes convergence past
k300.

The primary 0781 mechanism and states 7/8 remain meaningful because assignment
is safe, the pathway stays rank 1, and the selected spaces reproduce states,
operators and scores extremely accurately. Their absolute carrier frequencies
are not suitable as converged calibration data.

Overfitting concerns remain because validation uses one held-out sector of the
same parent spectrum, the selector verdict is BORDERLINE, and the behavior of
ranks beyond 299 is unknown.

## Next action

Preregister a shell-resolved Schur/Feshbach attribution of the existing
authenticated k280->k300 increment, focused on ranks 280–299, ranks 290/292,
and the isolated S2-f78 control cancellation. Use only the saved k300 parent;
do not launch another Stage-1 solve until that attribution determines whether
the large final-shell response is concentrated, collective, or interaction-
driven.
