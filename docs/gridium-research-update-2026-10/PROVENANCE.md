# Provenance and scientific scope

This directory is a review package, not a raw-data release. It contains six
reviewed vector PDFs and three Markdown documents. It intentionally excludes
checkpoint arrays, eigensystems, projected-operator archives, intermediate
selected spaces, local execution logs, and the complete research sandbox.

## Common model configuration

The figures use the asymmetric four-mode Gridium model with
`epsilon_LK=0.05`, `N2=N3=71`, `L2=11`, `L3=14`, `N4=8`, and `n1max=4`.
Stage-1 parents use `sigma=-24`, `which=LM`, `tol=1e-8`, and
`MMD_AT_PLUS_A`. Parameter-point differences are stated per figure.

## Source artifact directories

- `research/checkpoints/four-mode-asymmetric-n71-k300/`
- `research/checkpoints/four-mode-asymmetric-n71-epsj011-k300/`
- `research/checkpoints/four-mode-asymmetric-n71-epsj011-k340/`
- `research/sandbox/k300_results/`
- `research/sandbox/k300_shell_attribution/`
- `research/sandbox/epsj011_shell_attribution/`
- `research/sandbox/parameter_point_validation/`
- `research/sandbox/epsj011_k340_preregistration/`
- `research/sandbox/gridium_figures_2026_10/`
- `research/sandbox/gridium_figures_2026_10/reviewed_v2/`

These source locations are listed for repository traceability. The public
figure package does not duplicate their raw numerical contents.

## Figure lineage and finite parents

| Package figure | Original figure identifier | Parameter point and finite parent | Retained sizes/shells |
|---|---|---|---|
| A2 | `A2_adjacent_shell_frequency_corrections` | `epsilon_J=0.10`, k300; `epsilon_J=0.11`, k340 | 180–300 and 260–340 nested prefixes |
| B3 | `B3_k300_cross_parameter_schur_attribution` | both `epsilon_J`, separate k300 parents | shell ranks 280–299 |
| C2a | `C2_selector_vs_equal_cost_controls` | `epsilon_J=0.10`, k300 | S1 260–279, S2 280–299; budgets 10/15 |
| C2b | `C2_selector_vs_equal_cost_controls` | `epsilon_J=0.11`, k300 | S1 260–279, S2 280–299; budgets 10/15 |
| C3 | `C3_f78_error_cancellation` | `epsilon_J=0.11`, k300 | union/budget 10 |
| D1 | `D1_k340_spectral_convergence` | `epsilon_J=0.11`, k340 | prefixes 300/320/340 within 260–340 sequence |

## Verified hashes

The reviewed plotting inputs were unchanged when this package was prepared:

| Artifact | SHA-256 |
|---|---|
| A2 plotted-data CSV | `898d9485728ca80ba72d2a7593bbb64ef69d5cde365c1d1780929512d03f0cc8` |
| B3 plotted-data CSV | `205833fd80b8b8bb06d727e37b35fb39353a242d3a3d64c2b78d03943e565a4d` |
| C2 plotted-data CSV | `dd88124a8dfc1d006c196a3b08ba07fe1dc7ee331cfd311fb169f45ef35e78c7` |
| C3 plotted-data CSV | `3ecfe13ff7ab19dc1f8904ff2949ae8b3732c2b0ba95cdfc747a031927904c0b` |
| D1 plotted-data CSV | `2472511f6aeebcc949f8494563a1ef1059239f24d91833b040860fd584e266f4` |
| Frozen k340 criteria | `ffdbe09ae353c0eb6746a95c67b7c6dd1461db1a6c3e2d1260b61f831c0e4c72` |
| k340 configuration fingerprint | `10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0` |
| k340 preregistration manifest | `dd26b9b819e53859a807ef99aab7d96e57a2a7cda6b6862187846518b7924d52` |
| Completed k340 checkpoint | `a6f30847458108adee7a6776463d39a033672b2d3476af739e59bede91b80d13` |

Copied PDF hashes:

| PDF | SHA-256 |
|---|---|
| `A2_adjacent_shell_frequency_corrections_v2.pdf` | `8bd0d1f536b353185219edeb62e6b595ce6e2cdc1860160edd2f6304c34ff1d2` |
| `B3_k300_cross_parameter_schur_attribution_v2.pdf` | `8981396c61f70e4c9607929ba37e1970aba00304689735122efececbd763b072` |
| `C2a_selector_vs_controls_epsj010_v2.pdf` | `6905a7e21e4d02ceca4bbd1762f61220424a25b47c86af51b42568df6b607655` |
| `C2b_selector_vs_controls_epsj011_v2.pdf` | `e02714cace59609c4cf7db7a958910c101beb7c22c36b8157eee223b2bf778b0` |
| `C3_f78_error_cancellation_v2.pdf` | `f78f5888736744db2dbcc50882c4cf146a8c701ba70ae7069281a2bc4543ab44` |
| `D1_k340_spectral_convergence_v2.pdf` | `6f165e6e2fd575dcc027a39f6977b61a76afbd3806526bd098e8ae4a12d3e2e4` |

## Authentication boundary

The k340 figures are linked to the completed checkpoint, configuration
fingerprint, frozen criteria, and preregistration manifest above. Figure
generation used saved CSV/JSON post-processing artifacts and did not invoke a
Hamiltonian assembly, Stage-1 eigensolver, or new Stage-2 diagonalization.
The copied PDFs are byte-identical to the reviewed source PDFs.

## Scientific limitations

- Every endpoint is finite; neither k300 nor k340 is ground truth.
- Retained-basis convergence is **FAIL** and the k340 tail is classified
  **D — NONMONOTONIC UNRESOLVED**.
- Selector validation is **BORDERLINE**, multi-target transfer is **PARTIAL**,
  and the cross-parameter mechanism is **SUPPORTED**.
- The analysis does not establish spatial/domain convergence, convergence
  beyond k340, completeness of the four-mode model, a calibrated physical
  microwave port, device validation, or time-domain gate fidelity.
- No validated high-fidelity X90/X180 gate is included or implied.

**GATE WORK: HOLD**  
**SPATIAL CONVERGENCE: HOLD**
