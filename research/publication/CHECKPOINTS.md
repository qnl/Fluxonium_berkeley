# Historical checkpoint configurations

All entries describe the previous finite netlist model. A complete checkpoint is not a convergence certificate or physical-device validation. Original execution metadata and binary arrays remain excluded; the public companions retain configuration and source identity.

| Parent | eps_J | Stage-1 k | Grid N2/N3 | Configuration | Saved analysis |
| --- | ---: | ---: | --- | --- | --- |
| 2026-09-26-asymmetric-n51 | 0.1 | 180 | 51/51 | [metadata](../checkpoints/2026-09-26-asymmetric-n51/PUBLICATION_METADATA.json) | Stage-1 only; see published N51 diagnostic and export |
| four-mode-asymmetric-n71-epsj011-k300 | 0.11 | 300 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-epsj011-k300/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-epsj011-k300/analysis.json) |
| four-mode-asymmetric-n71-epsj011-k340 | 0.11 | 340 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-epsj011-k340/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-epsj011-k340/analysis.json) |
| four-mode-asymmetric-n71-epsj011-k380 | 0.11 | 380 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-epsj011-k380/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-epsj011-k380/analysis.json) |
| four-mode-asymmetric-n71-k180 | 0.1 | 180 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-k180/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-k180/analysis.json) |
| four-mode-asymmetric-n71-k220 | 0.1 | 220 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-k220/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-k220/analysis.json) |
| four-mode-asymmetric-n71-k260 | 0.1 | 260 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-k260/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-k260/analysis.json) |
| four-mode-asymmetric-n71-k300 | 0.1 | 300 | 71/71 | [metadata](../checkpoints/four-mode-asymmetric-n71-k300/PUBLICATION_METADATA.json) | [analysis.json](../checkpoints/four-mode-asymmetric-n71-k300/analysis.json) |

All these asymmetric checkpoints use the six energy entries `[EJ, EC, EL, ELK, EJS, ECS] = [5, 0.5, 1, 1, 4, 8]` GHz, `eC=5.5`, `eP=10`, `eps_LK=0.05`, `ng=0`, `phi_ext=0`, and `theta_ext=pi`. The specific recorded values, cutoffs, retained dimensions, and solver settings in each companion govern interpretation.

The common extended domains are `L2=11`, `L3=14`, with `n1max=4` and `N4=8`. N51 and N71 are different discretizations, not different devices. Retained masks and separate Stage-1 parents must not be conflated.

The [raw catalog](RAW_ARTIFACTS.csv) additionally includes older symmetric k160/k180 payloads where present. Those are separate historical results and are not relabeled as asymmetric parents.

[Fresh source/payload hash checks](SOURCE_AUTHENTICATION.json) establish the stated byte linkage. Historical dirty-generation status and source pins remain in the companions. No numerical validation was rerun.
