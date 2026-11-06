# Numerical validation

**PASS. Reproducibility: PASS. Preservation: PASS. New eigensolves: ZERO.**

| Check | Observed | Limit/context |
| --- | --- | --- |
| 380 vector squared-norm error | 5.36237720894e-14 | 1e-10 |
| 380-vector orthogonality infinity norm | 1.32759669322e-11 | 1e-8 (original) |
| Reshape norm difference | 1.85407245112e-14 | 1e-10 |
| Marginal sum error | 1.84297022088e-14 | 1e-10 |
| Maximum saved operator Hermiticity error | 6.2172885886e-14 | 1e-10 |
| Maximum local-action projection error | 3.59784639923e-14 | 1e-10 |
| Authenticated saved maximum residual, GHz | 5.91912736539e-08 | 1e-7 (original; not recomputed) |
| State / detailed coupling rows | 80 / 3200 | 80 / 3200 |
| Scientific artifacts repeated identically | 5 | 5 |
| Protected file hashes unchanged | 456 | All baseline files + Git index/status |

Projection validation uses every one of 380 bra vectors against each of 21 frozen focal/neighbor/control ket columns for each of five operators. It applies only local diagonal or 71×71 finite-difference operators to one reshaped vector at a time. Matching all projections numerically validates both C-order indexing and conjugation against authenticated saved operators; norm preservation alone would not prove ordering. Full D×D operators and Hamiltonians are never constructed.

Six synthetic checks passed: unequal-axis one-hot C indexing; normalized marginal sums; constant rectangle quadrature; complex local versus Kronecker contraction; complex coherent bilinear interference/conjugation; unitary-invariant ensemble density. These include unequal spatial dimensions to expose swapped-axis bugs, a complex contraction, coherent interference and a fixed-unitary ensemble-density invariance test.

Spectral checks: 380 strictly increasing finite saved energies; exact equality of archive residuals and checkpoint metadata residuals; focal k380 energies match authenticated review records within 1e-12 GHz. Rank ranges in saved Schur diagnostics are inclusive; both complete 20-rank shell sums reproduce their saved aggregates within 1e-10 MHz. All 40 ranks enter the correlations. Denominators use H/h GHz without 2pi; only displayed gaps and Schur scalars convert to MHz.

Main pass: 2.420151 s, peak RSS 682.641 MiB. Repeat: 2.612573 s, peak RSS 675.797 MiB. Each stays within 180 s / 1024 MiB. Single-thread BLAS is forced in code.
Address-space limit: UNAVAILABLE: current limit exceeds maximum limit. macOS rejected RLIMIT_AS before state metrics; the frozen plan records this platform fallback. A 50-ms resident-memory watchdog enforces the 1-GiB cap and SIGALRM enforces 180 seconds. The requested scientific scope was not exceeded.

No full eigenpair residual recomputation is claimed: original residuals and their payload hashes are authenticated. No research eigensolve, Schur rerun, Hamiltonian assembly, resolvent solve, interpolation/new grid or dressed-vector reconstruction was performed. QuTiP emits its existing missing-matplotlib warning on import; no plotting is required. No relevant numerical validation failed; broad unrelated repository tests were not run.

REPRODUCIBILITY.json records byte-identical CSV/JSON/NPZ hashes across an independent second process. Performance JSON is deliberately excluded from byte comparison. RUN_VALIDATION.json contains all residuals and checked ranks.
