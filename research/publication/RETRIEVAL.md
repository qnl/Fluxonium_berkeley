# External raw artifacts and reproduction requirements

[RAW_ARTIFACTS.csv](RAW_ARTIFACTS.csv) records each excluded checkpoint array,
saved eigenvector archive, large derived operator archive, and original metadata
file needed to authenticate a portable copy. `path` is its exact location
relative to the repository root in the owner's retained local archive. `bytes`
and `sha256` identify the actual file, not an estimated size or regenerated output.

No external data store, public download URL, access entitlement, or transfer
completion has been verified. Request the exact relative paths from the
repository owner and agree on an appropriate transfer/storage channel. Provisioning
that channel is a separate decision; large files are not silently added to Git.

For a numerical reconstruction, obtain the parent `stage1_eigensystem.npz`,
`projected_operators.npz`, original `checkpoint.json`, and any additional saved
Stage-2/vector archives required by the chosen historical analysis. Restore their
relative layout, verify file sizes and SHA-256 values, and use the recorded source
revision/hashes and package versions. Do not assume a current runner reproduces an
older dirty-generation checkpoint merely because its branch name is unchanged.

Before running any script, distinguish saved-array postprocessing from code that
assembles Hamiltonians, performs even small eigensolves, or generates figures.
This publication does not authorize numerical execution. The only handoff check
intended to run without scientific dependencies is
`python research/publication/verify_publication.py`.

Raw arrays are unnecessary for reading the published reports, figure data,
convergence metrics, attribution tables, tracking tables, or signed low-energy
matrix exports. They remain necessary for independent vector-level reproduction.

The raw catalog is a retrieval inventory, not proof of long-term storage,
availability on another machine, independent timestamps, or physical validity.
Private credentials and authorization/session logs are excluded from publication.
