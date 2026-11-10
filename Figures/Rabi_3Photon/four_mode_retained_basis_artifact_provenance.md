# Four-mode retained-basis artifact provenance

- `four_mode_asymmetric_shell_basis_audit.json` and its matching `.png` are
  the current final retained-basis result. They use the verified asymmetric
  N=51 k=180 checkpoint, rank shell states 160--179, and select the
  asymmetry-specific k170 basis.
- Files beginning with
  `four_mode_retained_basis_grid_phi_audit_spatial51_inherited_shell_rejected`
  are the fresh asymmetric N=51 inherited-shell rejection generated on
  2026-09-26 with `eps_J=0.10`, `eps_LK=0.05` and corrected noncontiguous
  retained-index tracking. They are trustworthy, but precede selection of the
  new asymmetric k170 shell.
- Earlier retained-basis artifacts were withdrawn and are intentionally not
  committed. One analysis reused a symmetric `eps_J=eps_LK=0` checkpoint
  while labeling it asymmetric; another computed overlaps before the
  noncontiguous retained-index embedding bug was corrected. Neither is valid
  convergence evidence.
- Files ending in `_reduced_diagnostic` use the deliberately reduced
  `N2=N3=31, N4=4` basis. They establish only the `grid_phi` matrix audit and
  multitone smoke-propagation plumbing; they are not retained-basis,
  spatial-convergence, or gate-performance evidence.

The verified fresh Stage-1 checkpoint is stored locally under
`research/checkpoints/2026-09-26-asymmetric-n51/`; its manifest records the
physical parameters, spatial cutoffs, solver settings, runtimes, and residuals.
