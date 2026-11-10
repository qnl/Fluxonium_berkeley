# October 29 Gridium figure validation

**Source authentication: PASS. Scalar/unit validation: PASS. Visual inspection: PASS for the final PNGs and the SVGs rendered with CairoSVG.** These are documentation figures made from existing records. No Hamiltonian, eigensystem, new host snapshot, scientific worker or numerical optimization was computed.

The bundled [figure_data.json](figure_data.json) contains full-precision plotted scalars, metric definitions, test-point configuration, source locators and **13 recomputed source SHA-256 fingerprints**. Its SHA-256 is `2e433e95d5b218f1a5a7a1103481db35c28a59c3b9bfcaba2dd8054acee47d76`; the generator rejects a changed bundle. This is recorded-byte authentication, not a digital signature or independent timestamp. Raw `research/` locators below identify retained local evidence, not files supplied by this compact GitHub figure package.

## Figure 1

**Verified metric:** largest absolute change among the **nine registered dressed carrier frequencies** between adjacent nested Stage-2 prefixes of one k380 Stage-1 parent. With saved energies in GHz, `f_ij(k) = 1000 × |E_j(k) − E_i(k)|` MHz; the plotted envelope is `max_ij |f_ij(upper) − f_ij(lower)|`. It is neither a Stage-1 energy correction nor an error against an established exact spectrum. The saved comparison convention is lower minus upper; the plotted absolute magnitude is independent of that sign.

| Nested prefix increment | Exact saved maximum (MHz) | Carrier attaining maximum | Figure annotation (MHz) |
|---|---:|---|---:|
| k300 → k320 | 0.7145717195062389 | f08 | 0.714572 |
| k320 → k340 | 1.2998359168232128 | f81 | 1.299836 |
| k340 → k360 | 5.594546068481577 | f08 | 5.594546 |
| k360 → k380 | 6.738197537884844 | f08 | 6.738198 |

Primary records: `research/sandbox/epsj011_k380_preregistration/k380_metrics.json`, `/comparisons/<increment>/components/carrier:<name>`, and `/prefix_metrics`; `evaluation.json`; the k380 checkpoint's `checkpoint.json` and `analysis.json`. Metrics SHA-256 **aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e** and evaluation SHA-256 **43903489aba94442dc868dae2eafa319de82f3cf3e3441db6144d6568c4a91aa** match the existing [published provenance anchors](../../work-blocks-1-5/SOURCE_PROVENANCE.md). Checkpoint and analysis hashes also match `k420_native_integration_2026_10_27/HISTORICAL_INPUT_PINS.json`; analysis matches the checkpoint's own payload pin.

Meaning/state matching was checked by reading, not executing, `analysis_tools.py` and `analyze_k380.py`. The historical analyzer matched embedded prefix eigenvectors one-to-one by overlap to the reordered finite k380 endpoint, then reordered energies consistently. Saved adjacent comparisons have `tracking_safe=true`. The nine carriers are f01, f07, f08, f78, f81, f72, f21, f03 and f38; their registered state pairs are included in the bundle. All **36 saved carrier changes** were checked against differences of the saved prefix frequencies; the four maxima and six-decimal annotations agree.

Test point: profile `n71-k380-epsj011-convergence`, configuration SHA-256 `8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e`, εJ=0.11, εLK=0.05, N2=N3=71, n1max=4, N4=8, L2=11, L3=14; remaining physical/solver settings are copied unchanged into the data bundle. Parent generated October 18, 2026; these records already underlie the October 24 handoff. They are **PRIOR evidence**, not October 29 numerical research.

**Outcome: PASS. Caveats:** the increasing maximum envelope does not mean every carrier increases. k380 is a finite endpoint, not ground truth; the frozen retained-basis verdict remains **FAIL**. No k400/k420 point, extrapolation or invented uncertainty bar is present. No spatial, omitted-mode or gate-validation claim is made.

## Figure 2

Primary sources: `research/sandbox/k420_host_feasibility_2026_10_29/HOST_FEASIBILITY.md` and `research/sandbox/k420_quiet_host_feasibility_2026_10_29/HOST_DECISION.md`. Their exact bytes, page counts, timestamps and hardware observations were checked against the data bundle. Thresholds match the unchanged policy SHA-256 `4301635b4fcee242c4ace64d4a4c9560f1b624f50beccacfbc0b7725030a89dd`.

| October 29 snapshot (PDT) | Physically free bytes | Physically free GiB | Availability-proxy bytes | Availability-proxy GiB |
|---|---:|---:|---:|---:|
| First, 10:02:19 | 65,077,248 | 0.06060791015625 | 7,882,309,632 | 7.340972900390625 |
| Quiet-session, 10:08:24 | 860,684,288 | 0.80157470703125 | 7,819,264,000 | 7.282257080078125 |

The page size is 16,384 bytes. Physical free is `free_pages × page_size`; the canonical proxy is `(free + inactive + speculative)_pages × page_size`. Divide bytes by **2³⁰** for GiB. The first free measurement is **62.0625 MiB**, not 0.062 GiB. Six-decimal plotted labels are rounded only for display. Panel A uses the **12 GiB** proxy minimum; panel B uses the separate **1 GiB** physical-free minimum. Both sessions fail both gates.

**Outcome: PASS. Caveats:** one brief native snapshot per session is not an atomic/sustained idle benchmark; significant activity remained during the requested quiet check. Pressure percentages, compressed memory and additional reclaimability are not substituted. The Apple M3 Pro's **18 GiB installed unified memory** is context, not a claim that all 18 GiB were available or that the machine can never host the problem. Native monitoring ABI/timing and scientific memory fit remain unverified.

## Figure 3

Sources: the frozen `k420_native_integration_2026_10_27/WORKLOAD.json`; `k420_execution_architecture_decision_2026_10_29/ARCHITECTURE_DECISION.md`; `k420_protected_native_slice_2026_10_29/IMPLEMENTATION_REPORT.md`; and the two host reports. Full local paths and hashes are in the data bundle.

The only completed scientific input shown is prior authenticated k380 evidence. Environment/admission is **CURRENTLY BLOCKED** by two distinct dependencies: failed observed local memory admission, and the unavailable protected execution/publication boundary under the required same-user adversarial-writer threat model. External access is **UNVERIFIED**. Stage A (one genuine k420 parent), Stage B (exact ten cases with frozen quality/Q/SHIFT requirements) and Stage C (actual k380→k400→k420 assessment) are explicitly **PLANNED**, not performed. Separate real-execution authorization is required. The output is a tested **PASS/BORDERLINE/FAIL** verdict, not an assumed converged spectrum.

**Outcome: PASS. Caveats:** this is a status/roadmap diagram, not proof of an implemented execution architecture or a usable external allocation. Metadata tests do not supply scientific results. No proposed trust-boundary change is approved by the figure.

## Reproduction and visual checks

From the repository root, use a separate plotting environment, leaving the scientific environment unchanged:

```bash
uv run --isolated --with-requirements docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/requirements.txt python docs/gridium-research-update-2026-10/figures/workblock_2026_10_29/generate_figures.py
```

Default reproduction uses the published scalar bundle. In the full local evidence checkout, add `--verify-local-sources` to rehash the 13 originals and recheck saved carrier entries. Optional `--qa-dir /tmp/gridium-figure-qa` writes reduced PNGs and independent SVG renders there; CairoSVG additionally needs its normal Cairo system library. On this Mac the QA invocation used `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. Dependency versions are pinned in [requirements.txt](requirements.txt). No scientific module is imported by the generator.

A clean repeat of the generation command with local-source verification reproduced all six final image files **byte for byte** (SHA-256 comparison: PASS).

| Final artifact pair | PNG dimensions | Export density | Visual checks |
|---|---:|---:|---|
| fig01_retained_basis_convergence | 2,880 × 1,760 | 320 dpi | Full PNG and 900-pixel PNG/SVG: PASS |
| fig02_host_memory_admission | 3,200 × 1,760 | 320 dpi | Full PNG and 900-pixel PNG/SVG: PASS |
| fig03_k420_execution_roadmap | 3,200 × 2,752 | 320 dpi | Full PNG and 900-pixel PNG/SVG: PASS |

Every final PNG was actually opened at full resolution. All three were also viewed at **900 pixels wide**, alongside SVGs independently rasterized with **CairoSVG 2.8.2**. Labels, numbers, panel distinction, captions/status wording, spacing, arrows and boundaries were inspected; no clipped text, overlapping annotations or cropped arrowheads were found. Automated text/box bounds support, but do not substitute for, this inspection.

An initial macOS Quick Look SVG thumbnail displayed font fallback/cropping artifacts. Explicit sans-serif fallbacks and portable “to” labels were added; the final SVGs were validated with CairoSVG rather than treating Quick Look's square thumbnails as authoritative. SVG text remains vector/editable (DejaVu Sans with Arial/sans-serif fallbacks); exact font appearance depends on the viewer. PNG and SVG are exported from the **same Matplotlib figure and authenticated data**, with identical values, axes, thresholds and status geometry; pixel identity between different rasterizers is not claimed. SVGs contain no embedded raster images. [RENDER_CHECKS.json](RENDER_CHECKS.json) records final file hashes, dimensions, density and geometry checks.

The log adds only three image/caption blocks in numerical order; original text, workblock/stoppage headings, Thomas feedback and scientific decisions are preserved. Relative PNG paths and provenance anchors resolve. Publication must separately verify the remote commit and asset bytes; local validation alone is not a publication claim.

Schur-v2 prelaunch: **HOLD**. Gate work: **HOLD**. Spatial convergence: **HOLD**. Scientific execution: **NONE**.
