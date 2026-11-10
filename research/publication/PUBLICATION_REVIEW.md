# November 6 publication review

## Repository state before preparation

- Repository: `SciGeek2000/2q_gridium`; branch: `four-mode-asym-gridium-tests`.
- Local HEAD, tracking ref, and live upstream branch were
  `3c9ab4e285e9f8f723f74044c4f71f4ae203142f`; divergence was zero.
- Origin's research branch was `c69da9d293c11e46df82377c310c7260acb59a77`,
  eight commits behind local HEAD. The publication destination is upstream.
- Two pre-existing staged CPhase formatting/notebook changes and the untracked
  `EmilianoUPDATE-unsaved-backup.md` were preserved and excluded from this commit.
- No uncommitted changes to Thomas's Gridium source files were present.

## Already published versus missing

The October 24 illustrated handoff was published in commit `c6a41af5...`.
The October 29 update and figures were published in `60c82b65...` and
`3c9ab4e2...`. Direct HTTP reads of the October 24 illustrated report and October
29 convergence SVG at the upstream commit returned 200 and byte-identical hashes.
All 58 files in the tracked October documentation package are therefore in the
live upstream tree. No claim is made about Thomas's personal account permissions.

Before preparation, `git ls-files research` was empty. The single `research/`
ignore rule excluded the entire raw/local research tree. The separate `docs/*`
rule allowed only `docs/gridium-research-update-2026-10/**`; local core context
documents and reference PDFs were not published. Historical documents could
mention local paths without making those artifacts available on GitHub.

The exact new publication set is [RELEASE_FILES.txt](RELEASE_FILES.txt).
[INVENTORY.csv](INVENTORY.csv) records payload sizes, hashes, scientific roles,
and transformations. Existing reports, metrics, data, and figures are added
without scientific recomputation. Numerical JSON files exceeding GitHub's inline
preview limit remain downloadable from the file's Raw/download control.

## Safety and preservation

- `.gitignore` uses exact file exceptions with ignored contents at every allowed
  ancestor directory. New arbitrary research files remain ignored.
- `.gitattributes` disables text normalization only for `research/**`, preserving
  historical CRLF CSV bytes and their recorded hashes in a fresh checkout.
- Large checkpoint/eigenvector archives, caches, synthetic engineering outputs,
  local diffs, credentials, authorization files, and machine/session logs remain
  excluded. See [EXCLUSIONS.csv](EXCLUSIONS.csv) and [RETRIEVAL.md](RETRIEVAL.md).
- Original reports/metadata remain untouched. Portable copies replace local path
  strings; one report links its portable numerical companion and three algebra
  expressions are formatted as code to avoid accidental Markdown links.
  Original and published hashes remain distinct and explicit.
- Saved low-energy energies and complex `grid_phi` entries are serialized to JSON
  and checked exactly against their source arrays. Eigenvectors remain external.
- Fresh source/payload pin checks are in [SOURCE_AUTHENTICATION.json](SOURCE_AUTHENTICATION.json).
  They establish byte linkage, not physical validation or independent timestamps.
- The publication changes no Hamiltonian, scientific parameter, result,
  classification, figure, gate target, or execution policy.
- No scientific eigensolve, propagation, optimization, figure regeneration, or
  k420 experiment was performed. No numerical regression pass is claimed.

## Validation and publication boundary

Run `python research/publication/verify_publication.py` from the committed tree.
The standard-library-only validator checks the exact release set against Git,
payload hashes/sizes, local Markdown targets, and documented external dependencies.
Historical links to raw arrays or excluded method/session artifacts are explicitly
cataloged; their presence is not misreported as fresh-clone reproducibility.

Publication preparation includes a focused local commit. Pushing requires the
user's requested approval after reviewing that commit and its exact file list.
Before pushing, recheck the live upstream hash. If it differs from the recorded
base, inspect the new commits and reconcile without overwriting Thomas's work.
Never force-push this handoff.

After approval and a successful upstream recheck:

```sh
git push upstream HEAD:refs/heads/four-mode-asym-gridium-tests
```

Then verify that the live upstream hash equals the approved commit and fetch the
new `research/README.md` and representative metrics through GitHub. Until that
push and verification succeed, the added research files are prepared locally and
must not be described as newly available on GitHub.
