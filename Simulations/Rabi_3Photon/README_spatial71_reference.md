# Frozen asymmetric N=71 reference run

This runner performs the next four-mode spatial-convergence point only. It
does not optimize a gate and does not alter the physical Hamiltonian. The
single expensive Stage-1 solve is fixed to the asymmetric Gridium parameters,
N2=N3=71, N4=8, and k=180 encoded in
`run_four_mode_spatial71_reference.py`. The configuration SHA-256 is
`2c996f23c3f4810f9bb7d72f9231f264cda516900b4a7067af2ab53dbd1b868a`.

## Machine and environment

Use a higher-memory Linux or macOS machine with at least 8 GiB genuinely free;
16 GiB free is preferred because sparse LU fill is data-dependent. Clone the
same Git revision and transfer the verified local N=51 directory to:

`research/checkpoints/2026-09-26-asymmetric-n51/`

The runner pins SHA-256 hashes for all three N=51 oracle files. A reformatted,
recomputed, symmetric, or corrupted checkpoint is rejected even when its
metadata and array shapes look plausible. Transfer the files byte-for-byte.

For scale, the verified N=51 solve used a 23,409-dimensional matrix with
599,625 nonzeros. Its shifted LU had 28,667,572 total L+U nonzeros (47.81x
fill), took 66.35 s to factor, and ARPACK took 170.12 s / 736 inverse solves on
the original machine. N=71 raises the matrix dimension to 45,369 (1.94x);
neither LU fill nor runtime scales linearly, which is why the larger free-RAM
margin is required. The N=71 matrix nonzeros, LU fill, timings, and solve count
are measured and recorded rather than estimated as scientific results.

Create the portable environment from the repository root:

```bash
conda env create -f environment-portable.yml
conda activate gridium-idealgridium-py312
```

First verify the immutable job description. This does not assemble a matrix:

```bash
python -m Simulations.Rabi_3Photon.run_four_mode_spatial71_reference --dry-run
```

The dry-run fingerprint must match the value above and the expected Stage-1
dimension must be 45,369.

## Exact execution command

Run once from the repository root on a local POSIX filesystem (not directly in
a network-synchronized folder):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Simulations.Rabi_3Photon.run_four_mode_spatial71_reference \
  --execute \
  --output-root research/checkpoints \
  --n51-checkpoint research/checkpoints/2026-09-26-asymmetric-n51
```

The runner refuses execution unless all three thread variables equal `1`. It
also refuses to overwrite an existing final output directory. Do not change
the frozen constants to fit a machine; move the job to a suitable machine.

## Completion and interruption semantics

During execution, files live under a hidden directory named like:

`research/checkpoints/.four-mode-asymmetric-n71-k180.incomplete-<uuid>/`

An exception or interruption normally leaves that directory without a valid
`COMPLETE` marker. An interruption in the narrow interval after atomic rename
but before marker creation instead leaves the final-named directory without a
valid `COMPLETE` marker. There is intentionally no resume path: archive or
remove either invalid directory only after inspecting `run_state.json` and
`failure.txt`, then rerun the exact command from the beginning.

A checkpoint is valid only when all of the following hold:

- the final directory is
  `research/checkpoints/four-mode-asymmetric-n71-k180/`;
- `checkpoint.json` has `status: complete` and the frozen configuration hash;
- every recorded file hash verifies;
- `COMPLETE` exists, has `status: complete`, carries the same configuration
  hash, and authenticates the exact bytes of `checkpoint.json`.

`COMPLETE` is written last, after finite-value, shape, residual,
orthogonality, configuration, retained-basis, and cross-grid analyses finish.
Partial eigenpairs must never be copied or used as scientific input.

## Output files

Copy the entire completed directory back, including:

- `stage1_eigensystem.npz` — k=180 values, vectors, and residuals;
- `projected_operators.npz` — projected n1/n2/n3/x2/x3 operators;
- `analysis.json` — same-solve k170/k172/k175 audits and overlap-tracked
  N=51→N=71 comparison;
- `checkpoint.json` — full configuration, environment, Git, relevant source
  hashes, solver, LU, timing, residual, and file-hash provenance;
- `run_state.json` — terminal run state;
- `COMPLETE` — last-written validity marker.

Preserve those repository-relative names. The `research/` tree is ignored by
Git; copying the result back does not stage it. Do not copy a hidden
`.incomplete-*` directory as a completed checkpoint.

The runner records pathway rankings and control-matrix changes but deliberately
leaves the A/B/C spatial-convergence classification unset for subsequent
scientific review.

## Degenerate-subspace tracking

Individual eigenvectors are not unique inside a degenerate or nearly
degenerate subspace. The runner therefore retains the Hungarian state
assignment while also recording each state's competing overlap, margin, and
competitor ratio, plus principal-angle/singular-value diagnostics for
competition-connected subspaces. Warning-only claim gates are explicit:
assigned overlap `<0.90`, competitor ratio `>=0.50`, or overlap margin
`<=0.25`. They never relabel states. Low or tied individual overlaps do not by
themselves prove spatial nonconvergence when the corresponding subspace remains
well matched. Pathway claims involving ambiguous states are marked
`ambiguous` or `deferred`, and basis-invariant `grid_phi` subspace strengths are
reported instead; uniquely tracked pathways remain `resolved`.
