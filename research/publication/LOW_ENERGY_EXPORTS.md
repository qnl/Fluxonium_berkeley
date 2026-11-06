# Saved low-energy energies and signed grid_phi matrices

The JSON files in [low_energy/](low_energy/) serialize existing arrays from
`research/sandbox/subspace_tracking/*.npz`. They were copied with
`numpy.load(..., allow_pickle=False)`; no Hamiltonian, eigensolve, tracking
assignment, time propagation, or figure was recomputed. The serialized energies
and reconstructed complex matrices were checked element-for-element against the
saved arrays. The original archive path and SHA-256 are embedded in every file.

- `energy_GHz` is the saved energy ordering, in GHz.
- `grid_phi_real` and `grid_phi_imag` are the real and imaginary parts of the
  original signed matrix, in radians. They must not be replaced by magnitudes
  for coherent-control calculations.
- `retained_stage1_indices` identifies the actual retained mask, not a universal
  physical-state identity.
- `saved_residual_GHz` is copied from the source; it is not a newly measured residual.
- The eigenvector array is excluded and cataloged for separate retrieval.

N51 k180 stores 11 final states; N71 k180 and the N71 retained-mask archives store
48. `N71_priority_*` and `asymmetric_priority_*` are different historical mask
families. See the [tracking report](../sandbox/subspace_tracking/REPORT.md) and
[retained-basis re-audit](../sandbox/k_convergence_reaudit/REPORT.md) for their
definitions, assignments, and limitations.

These are descendants of finite k180 references, not k380 or infinite-basis
results. Higher-parent `analysis.json` files preserve their saved matrix-element
magnitudes and tracking summaries. Signed k220-k380 matrices have not been
invented or reconstructed for this publication.
