"""Render existing attribution outputs. No Hamiltonian assembly or solves."""
from pathlib import Path
import json, csv, math, hashlib, subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
d=json.loads((HERE/'summary.json').read_text())
rr=list(csv.DictReader((HERE/'rank_attributions.csv').open()))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'

def fmt(v): return f'{v:.6f}'
def sci(v): return f'{v:.3e}'

recommendation=('Preregister a same-parent rank-addback/ablation experiment: starting from '
 'k220, add ranks 223 and 230 separately and together, then add 225, always with all eight '
 'mode-4 states; compare against k240 and a held-out complement of ranks 220–239. '
 'Use ranks 258/259 as an independent Q2 stress test, and measure all mandatory '
 'carrier, state/subspace, and operator errors. Use the saved k260 parent only.')
d['interpretation']=dict(
    dominant_mechanism='C',
    description='Off-resonant rank-concentrated virtual corrections spread across multiple dressed shell eigenmodes, with P-space relaxation and smaller cross-shell effects.',
    near_resonance_detected=False,
    limitations='Exact finite-parent attribution; Schur reconstruction is evaluated at observed dressed energies, not an independent prediction. Second-order adequacy at 20% is diagnostic only. States beyond rank259 are unavailable.',
    next_research_action=recommendation,gate_work='HOLD',spatial_convergence='HOLD',retained_basis='FAIL',
    stage1_eigensolve_run=False,gate_optimization_run=False)
(HERE/'summary.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

sections=['# Gridium k260 shell-resolved attribution\n',
'''## Result

DATA FEASIBILITY: **SUFFICIENT** for exact attribution inside the saved finite
k260 parent. EFFECTIVE HAMILTONIAN: **VALIDATED**. DOMINANT MECHANISM: **C**.

The large 220->240 spectral change is an off-resonant virtual correction with
strong Stage-1 rank concentration and a broader dressed-shell spectral spread.
It is not explained by a near-resonant omitted shell state under the frozen
diagnostic definition. Ranks 230, 223, and 225 dominate states 7/8; the central
shell couples substantially more strongly than its predecessors. Finite
P-space relaxation is important. The original retained-basis verdict remains
FAIL, and gate work and spatial convergence remain HOLD.

The C label follows the preregistered rule: state 7 is B (distributed by the
modal criterion), state 8 is C (intermediate modal concentration), so their
joint mechanism is C. Here “mixed” means rank concentration plus distributed
off-resonant dressing and relaxation. It does **not** mean a mixture of
resonant and off-resonant terms; the measured near-resonant fraction is zero.

## Preregistration and reproducibility

`methodology.md` was written and hashed before new attribution calculations.
Its SHA-256 is
`921f264e6e62c6233f1906592f282f28ab1d389aa055417c47e05ac62303ec80`.
The diagnostic checks this hash. No existing convergence threshold was changed.

All ranks are zero-based saved Stage-1 indices. Energies and couplings are in
GHz; shift tables are in MHz. Full precision is in `summary.json`,
`attribution_tables.csv`, `rank_attributions.csv`, and `shell_modes.csv`.

Reproduction command (Stage-2 finite-block work only):

```bash
bash research/sandbox/k220_environment_reconstruction/python.sh \\
  research/sandbox/k260_shell_attribution/attribute.py
```

`finalize_report.py` renders the stored results without any diagonalization.

## Data feasibility

Stored eigenvalues (260) and eigenvectors (45369 x 260) are available.
The five projected operators n1/n2/n3/x2/x3 each have shape 260 x 260.
The stored physical coefficients and mode-4 cutoff suffice to reconstruct
the complete Stage-2 Hamiltonian and grid_phi=(x2/2+x3) tensor I8.
Stage-2 Hamiltonian components and retained vectors are not separately saved,
but are reconstructible by the existing production formula. Coupling blocks
are exact slices of that reconstructed finite matrix. No physical couplings
were guessed. No Stage-1 Hamiltonian was assembled or diagonalized.

The reconstructed lowest eleven energies and absolute operator entries match
the saved nested analysis. Saved Stage-1 real-space vectors are unnecessary
for this finite-block attribution. Ranks beyond 259 remain unavailable; no
claim is made about their tail or any infinite-basis limit.

## Effective Hamiltonian and numerical validation

P has dimension 220 x 8 = 1760, Q1 and Q2 each 20 x 8 = 160, and their union
has dimension 320. The full Stage-2 dimension is 2080. Tensor ordering is
Stage-1 rank followed by mode-4 occupation.

For an exact dressed eigenpair with components p,q,

    q = (E I - H_QQ)^(-1) H_QP p
    H_eff(E) = H_PP + H_PQ (E I - H_QQ)^(-1) H_QP

The plus sign is correct; shell levels above E produce negative denominators
and lower energies. Q1-Q2 coupling is included when eliminating their union.
The numerical implementation uses the spectral resolvent, without forming an
explicit inverse. The exact shift identity is

    E - E_old = <p|Sigma(E)|p>/<p|p>
                + <p|(H_PP-E_old)|p>/<p|p>.

The second term is P-space relaxation. Rank contributions sum to self energy,
not to the entire energy shift until relaxation is included. These are signed
contributions in the saved basis and are not invariant under arbitrary rank
rotations. Dressed-shell spectral participation supplies a complementary
description (individual degenerate eigenmode shares also depend on basis
choice within any degenerate shell subspace).

This is an exact **a posteriori reconstruction at the observed dressed E**,
not an independent root prediction. The separately reported second-order
estimate uses the old eigenstate and old energy and is genuinely approximate.
All low-lying states are diagonalized together, preserving finite-block
near-degenerate mixing; unsafe assignments would have blocked individual
claims. All required assignments, including counterfactuals, are safe.
''']

allstates=[r for step in d['state_attributions'].values() for r in step.values()]
sections.append(table(['Check','Measured maximum','Frozen limit'],[
 ['Saved energy error (GHz)',sci(max(r['energy_error_GHz'] for r in d['reconstruction_checks'].values())),'1e-10'],
 ['Saved |grid_phi| error',sci(max(r['operator_error'] for r in d['reconstruction_checks'].values())),'1e-10'],
 ['Stage-2 eigenpair residual (GHz)',sci(max(r['eigenpair_residual_GHz'] for r in d['reconstruction_checks'].values())),'1e-9'],
 ['Adjacent Schur residual (GHz)',sci(max(r['schur_residual_GHz'] for r in allstates)),'1e-9'],
 ['Adjacent Q-vector error',sci(max(r['q_reconstruction_error'] for r in allstates)),'1e-8'],
 ['Adjacent energy-identity error (MHz)',sci(max(abs(r['reconstruction_error_MHz']) for r in allstates)),'1e-6'],
 ['Combined Q1+Q2 identity error (MHz)',sci(max(abs(r['reconstruction_error_MHz']) for r in d['combined_Q1_Q2_attribution'].values())),'1e-6']]))

sections.append('## State shifts across every adjacent shell\n\nAll values are MHz. Exact reconstructed shifts equal observed shifts to the errors above. SO is the old-state second-order estimate; its error is SO minus observed.\n')
sections.append(table(['Step','State','Observed','SO','SO error','Exact self energy','P relaxation','Mechanism'],
 [[step,s,fmt(r['observed_MHz']),fmt(r['second_order_MHz']),fmt(r['second_order_error_MHz']),
   fmt(r['exact_self_energy_MHz']),fmt(r['relaxation_MHz']),r['mechanism']]
  for step,rs in d['state_attributions'].items() for s,r in rs.items()]))

sections.append('## Transition attribution\n\nCarrier shifts are differences of the tracked state-energy shifts; cancellation is retained. Exact reconstruction errors for every row are in `attribution_tables.csv` (maximum about 1.1e-10 MHz).\n')
sections.append(table(['Step','Carrier','Observed MHz','SO MHz','SO error MHz'],
 [[step,n,fmt(r['observed_MHz']),fmt(r['second_order_MHz']),fmt(r['second_order_error_MHz'])]
  for step,rs in d['transition_attributions'].items() for n,r in rs.items()]))

sections.append('''## Why ranks 220–239 matter

For states 7/8, the shell probabilities are only 0.064906%/0.075298%, yet the
exact energy shifts are -28.052236/-32.699986 MHz. Logical states 0/1 shift
only -1.409978/-1.454042 MHz. Their unequal corrections produce the large
outer-carrier shifts and the -4.647750 MHz change in f78.

The minimum isolated-shell denominators for states 7/8 are 32.517046 and
32.497011 GHz, vastly larger than the frozen 0.010 GHz near-resonance scale.
Maximum coupling/denominator ratios are 0.00913 and 0.01110, below 0.10.
Every measured adjacent-shell near-resonant contribution fraction is zero.
This conclusion concerns the stated partition; it is not a statement about
uncomputed ranks beyond 259.

The enhanced numerator, rather than a small denominator, explains the jump.
Total coupling norms ||H_QP psi_old|| in GHz are:
''')
sections.append(table(['Step','State0','State1','State7','State8'],
 [[step,*[fmt(math.sqrt(sum(float(r['coupling_norm_GHz'])**2 for r in rr if r['step']==step and r['state']==s))) for s in ['0','1','7','8']]]
  for step in d['state_attributions']]))

sections.append('\nExact central-shell rank shares for the primary states (self energy, before adding P relaxation):\n')
sections.append(table(['State','Rank','Self energy MHz','Old-state coupling norm GHz','Shell weight'],
 [[s,r['rank'],fmt(r['exact_self_energy_MHz']),fmt(r['coupling_norm_GHz']),sci(r['Q_weight'])]
  for s in ['7','8'] for r in d['state_attributions']['220->240'][s]['top_ranks'][:5]]))
sections.append('''
Ranks 230/223/225 supply 88.832%/87.797% of absolute exact self energy for
states 7/8. Their contributions are not standalone addback energy shifts:
they include the self-consistent coupled-shell solution and must not be added
again to the reported exact shift. Logical states0/1 emphasize the same ranks;
comparator states2/3 emphasize ranks220/222/227 instead.

The largest dressed-shell contributions are:
''')
sections.append(table(['State','Shell mode','|coupling| GHz','E_old-epsilon GHz','SO contribution MHz','SO share'],
 [[s,r['mode'],fmt(r['coupling_GHz']),fmt(r['denominator_GHz']),fmt(r['pt_MHz']),f"{100*r['fraction']:.3f}%"]
  for s in ['7','8'] for r in d['state_attributions']['220->240'][s]['top_modes'][:3]]))

sections.append('\nThe frozen mode-based decision uses top-three shares and effective participation:\n')
sections.append(table(['Central-shell state','Top3 modal share','Effective modes','Modes for 90%','SO error / shift','Class'],
 [[s,f"{100*r['modal_participation']['top3_fraction']:.3f}%",fmt(r['modal_participation']['effective_count']),
   r['modal_participation']['count90'],f"{100*abs(r['second_order_error_MHz']/r['observed_MHz']):.3f}%",r['mechanism']]
  for s,r in d['state_attributions']['220->240'].items()]))
sections.append('''
Both primary states require eleven dressed shell modes to account for 90%.
Their top-three shares are 45.493% and 56.000%. The latter misses the frozen
distributed criterion (<=50% and effective count>=8), so overall C is retained.
There is no basis for relabeling this result A or relaxing the preregistration
to force B. The mixed label is operational, not a new physical phase.

Second-order estimates meet the preregistered 20% diagnostic accuracy bound,
but miss state7/8 shifts by 5.451/6.094 MHz. They are not adequate for precision
calibration or a convergence claim. Exact finite-block treatment is essential
for the residuals of interest. P relaxation contributes +6.792/+7.493 MHz to
exact self energies -34.844/-40.193 MHz. This decomposition is not identical
to second-order plus a single correction; the exact self energy uses the
relaxed p and dressed E.

In Q2, states7/8 emphasize ranks251/258/259 and 258/259/256 respectively.
The rank259 boundary remains active. That warns that a larger-parent tail is
unknown; it does not authorize extrapolating a smooth decay beyond k260.

## Q1–Q2 coupling and nonadditivity

The P+Q2 counterfactual skips Q1. The no-cross counterfactual includes both
shells but sets only H_Q1Q2 and its adjoint to zero. All remaining matrix
entries are unchanged. These diagnostic counterfactuals have safe tracking.

    Full correction = standalone Q1 + standalone Q2
                      + explicit Q1-Q2 effect + P-mediated nonadditivity.

Here explicit effect = E_full-E_no_cross; the last term is
E_no_cross-E_PQ1-E_PQ2+E_P. It includes mixing through P and energy dependence.
Only this defined decomposition is used; separate shells are not assumed
additive. Values below are MHz.
''')
sections.append(table(['State','Q1 alone','Q2 alone','Explicit cross','P-mediated remainder','Full 220->260'],
 [[s,fmt(r['Q1_standalone_MHz']),fmt(r['Q2_standalone_MHz']),fmt(r['explicit_Q1Q2_effect_MHz']),
   fmt(r['P_mediated_nonadditivity_MHz']),fmt(r['full_MHz'])] for s,r in d['cross_shell'].items()]))
sections.append('''
For states7/8 explicit cross coupling contributes -0.931852/-0.825876 MHz;
the remainder is -0.324992/-0.316489 MHz. Cross-shell effects matter at
sub-MHz/MHz precision but cannot explain the already-large P->P+Q1 jump.
For f07/f08/f78, total nonadditivity is -1.192508/-1.078029/+0.114479 MHz.
''')

sections.append('## Drive selectivity\n\nNo gate optimization or driven evolution was performed. Frequencies are MHz; grid_phi magnitudes are in the existing operator convention.\n')
sections.append(table(['k','f01','f78','f78-f01','g01','g78','f07-f81'],
 [[r['k'],fmt(r['f01_MHz']),fmt(r['f78_MHz']),fmt(r['difference_MHz']),fmt(r['g01']),fmt(r['g78']),fmt(r['outer_difference_MHz'])] for r in d['selectivity']]))
sections.append('''
The intended middle transition crosses the logical splitting between k220 and
k240. Its separation is only 0.477 MHz at k240 and 1.287 MHz at k260, compared
with 4.127 MHz at k220. g01 remains appreciable (~0.448 at k260); the proximity
is not removed by a vanishing logical matrix element. g01/g78 is about 0.388.

The outer-carrier identity f07-f81 = f01-f78 means the same crossing also
brings the two strong outer transitions into near collision. At k260 they are
separated by 1.287281 MHz with g07=1.909889 and g81=1.777150. Those transitions
are desired on their respective legs but unwanted targets of each other's tone.

The nearest additional logical-origin transitions are 0->8 and 1->7:
their offsets from outer tones are about 14.57–15.86 MHz at k260, but their
grid_phi magnitudes are only about 1.13e-11 and 9.47e-12 in this model. The
appreciable 1->6 coupling (~0.4871) is about 253–254 MHz from the outer tones.
The nearest-five scan for both tones at all five k values is stored in
summary.json. Frequency proximity alone does not establish gate fidelity,
leakage, or a physical-port response.

## Operators and subspaces

The following reuse the validated same-parent operator and tracking results;
the newly reconstructed values agree to the tolerance above. F denotes a
Frobenius norm. Duplicated singular values are listed as pairs where equal to
the shown precision, not discarded from the numerical results.
''')
ks=['k180','k200','k220','k240','k260']
sections.append(table(['Quantity',*ks],
 [[name,*[fmt(d['operator_blocks'][k][bn]['frobenius_norm']) for k in ks]]
 for name,bn in [('Full {0,1,7,8} F','grid_phi_0_1_7_8'),('Logical->{7,8} F','grid_phi_logical_to_7_8'),('Internal {7,8} F','grid_phi_internal_7_8')]]+
 [[f'{name} singular values',*[' / '.join(fmt(v) for v in d['operator_blocks'][k][bn]['singular_values']) for k in ks]]
 for name,bn in [('Full','grid_phi_0_1_7_8'),('Logical','grid_phi_logical_to_7_8'),('Internal','grid_phi_internal_7_8')]]+
 [[f'State{s} overlap',*[f"{d['state_overlaps_vs_k260'][k][s]:.9f}" for k in ks]] for s in ['7','8']]+
 [['Subspace minimum SV',*[f"{d['subspace_diagnostics'][k]['minimum_singular_value']:.9f}" for k in ks]],
  ['Maximum principal angle (deg)',*[fmt(d['subspace_diagnostics'][k]['maximum_principal_angle_degrees']) for k in ks]]]))
sections.append('''
No required assignment is ambiguous. The full subspace k220->k260 overlap is
strict BORDERLINE (0.998847125), and k240->k260 is strict PASS (0.999941298).
State8 follows the same pattern; state7 passes both. This is measurable
dressing without loss of the tracked identity.

All operator-block magnitudes pass frozen strict relative criteria for both
endpoint comparisons, but the operator redistribution is not zero. In the
central step |g07| grows by 0.005374, |g78| by 0.016602, and |g81| falls by
0.006631. The exact complex operator split after phase alignment is

    delta O = (p_new^dag O_PP p_new - O_old)
              + p_new^dag O_PQ q + q^dag O_QP p_new + q^dag O_QQ q.

The first term is PP relaxation; the remaining three are direct shell terms.
Their sum matches the full complex change within 1e-10. The direct-shell
magnitudes for g07/g78/g81 are 0.002037/0.011291/0.002225. These correspond to
approximately 0.107%/0.990%/0.124% of their k220 edge magnitudes. In particular,
the direct g78 contribution is substantial within its small net change;
for g81 the direct contribution opposes part of the PP relaxation. Complex
contributions are recorded separately, rather than treating absolute values
as additive. High overlap therefore does not imply identical operators.

## Pathway reassessment

The primary remains global rank1 at every k; 0721 remains rank3. 0381 changes
rank4 to rank5 at k240, giving the existing ordering BORDERLINE. Target-path
ordering among these three never changes.
''')
sections.append(table(['Path score',*ks],[[n,*[fmt(r['values'][k]) for k in ks]] for n,r in d['pathway_series'].items()]))
sections.append('''
All three score changes pass the frozen strict endpoint thresholds. Primary
edges change by 0.244%/1.344%/0.595% from k220 to k260; its score changes by
0.999%. The mechanism remains structurally meaningful and deserves further
study. Its carrier frequencies remain unsuitable as converged precision
calibration data. Both comparators remain useful stress tests: their shell
response emphasizes different Stage-1 ranks. No evidence here warrants
replacing the primary mechanism with a comparator merely on score stability.

## Next research action
'''+recommendation+'''

This recommendation is not executed in this task. It directly tests whether
the rank shares are causally captured by a small retained subset rather than
being inseparable collective dressing. The held-out complementary shell is
necessary to expose selection bias and omitted interactions.

| Option | Information gained | Cost / limitation |
|---|---|---|
| Larger k280/k300 parent | New data beyond the active rank259 boundary; eventually needed to test the tail | New Stage-1 solve; alone it does not test the central rank-attribution mechanism |
| Physics-informed retained selection | Tests whether energy-prefix ordering wastes basis states | Requires validation against held-out shells; must not cherry-pick only the primary path |
| Targeted influential-rank addback/ablation | Directly tests ranks223/230/225 with existing data and comparators | Lowest-cost discriminating experiment; cannot certify the beyond-k260 tail |
| Further effective-Hamiltonian diagnostics | Can partition relaxation within P and identify feedback channels | Useful follow-up if addback fails; the present exact identity already resolves the numerical shell shift |

The recommended addback has the greatest immediate information per numerical
cost. It is a diagnostic experiment, not a replacement convergence criterion.

## Scope and preservation

k260 remains an endpoint, not ground truth. This is fixed-N71, fixed-four-mode
retained Stage-1 attribution. It establishes no spatial/domain/n1max/N4
convergence, omitted-array-mode completeness, physical-device/port validity,
or X90/X180 gate validation.

All tracked-file contents and all files in the k260, k220, k180, and N51
checkpoints were hashed before and after the calculation and remained identical.
No Stage-1 solve, production edit, EmilianoUPDATE.md edit, commit, or push was
performed. Existing tracked status:

```text
'''+d['git_status']+'''```

GATE WORK: **HOLD**

SPATIAL CONVERGENCE: **HOLD**
''')
(HERE/'REPORT.md').write_text('\n\n'.join(sections))
print('Wrote REPORT.md and interpretation in summary.json; no numerical solve.')
