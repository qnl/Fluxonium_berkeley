# Source provenance and reproducibility

This is a documentation-only GitHub review copy of the authenticated October 24 sandbox handoff. No scientific calculations or figure generation were repeated while preparing it. All 16 PDF/PNG figure files, all six scalar-data files, and the numerical evidence CSV are byte-identical to their authenticated source exports.

The original sandbox preparation scripts authenticated existing records, extracted saved scalars, generated the figures with Matplotlib, and rendered the PDFs with Ghostscript for visual QA. Those scripts, temporary renders, raw checkpoints, eigenvector archives, approval records and execution infrastructure are not part of this compact review package. Their omission does not remove any requested figure or supporting plotted data.

`SOURCE_PACKAGE_MANIFEST.json` is the unchanged complete manifest of the original sandbox handoff, including local-only artifacts. `PACKAGE_MANIFEST.json` hashes the files actually included here, excluding itself. Markdown navigation and packaging descriptions are adapted for this location; the scientific values and frozen conclusions are unchanged. Source locators below that begin with `research/` identify local authenticated evidence, not public GitHub links.

The [October 20 research work block](../WORK_BLOCK_2026-10-20.md) and [earlier overview](../README.md) remain unchanged.

## Authentication manifests

| Manifest | Verified members | SHA-256 |
| --- | ---: | --- |
| research/sandbox/k380_coupling_vs_schur_2026_10_21/OUTPUT_HASHES.json | 21 | d5437f773673343cbb0f00595efbf541712fc5e7d3ab711b8cef477bfdcf4bde |
| research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/OUTPUT_HASHES.json | 17 | 95a709835abbb06bd87df14d15c50aba002b6c058357dc2cb99176282ca3bce2 |
| research/sandbox/k380_cross_shell_selector_transfer_2026_10_21/OUTPUT_HASHES.json | 15 | 4b527e3d4317a2c49e4159648ad84eef057a813f71d57a663bd9e7676fa1d503 |
| research/sandbox/epsj011_k380_diagnostic_review/OUTPUT_HASHES.json | 12 | 0fd48adac195f2164d5d15e60a2082257ad8e9e65922a0a83794573da001e9f9 |
| research/sandbox/k380_s2_final_authorization_execution_2026_10_24/DELIVERY_MANIFEST.json | 11 | f4c2d91c5eb3fa72f5f6ce03e0d3ad76fe87bf7895889cbd932ed828a3ab390a |
| research/sandbox/k380_p340_s2_base_matched_validation_continuation_001/OUTPUT_MANIFEST.json | 25 | 418874afc97a8cfc017b452c9f7c5f65844944c1a55d8b74f0d97b0ee61cbe0f |
| research/sandbox/k380_wb5_final_authorization_execution_2026_10_24/DELIVERY_MANIFEST.json | 23 | 680ec08930590e12377036940f300856989d445c2aa0fd6f039bc9e1e91a481b |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/OUTPUT_MANIFEST.json | 70 | 637d2758e657df47cb002bc492e78dd76262ec032d2ca135b0595e0a8cdaf918 |

## Primary numerical inputs

| Input path | SHA-256 |
| --- | --- |
| research/sandbox/epsj011_k380_diagnostic_review/SPECTRAL_DECOMPOSITION.csv | 9f586b2a77da32202328f5fe14d4312ad4fc20ba17c034d12eb441113d1348ed |
| research/sandbox/epsj011_k380_preregistration/evaluation.json | 43903489aba94442dc868dae2eafa319de82f3cf3e3441db6144d6568c4a91aa |
| research/sandbox/epsj011_k380_preregistration/k380_metrics.json | aa637be253bc1b6a9f4f89307b42056f108bab637f72832248dce39ddd0e314e |
| research/sandbox/k380_coupling_vs_schur_2026_10_21/RESULTS.json | 6fb471d85d4e182ace1f77f0acd819c22ac7cb7c56590779e1f9f03831c99d1d |
| research/sandbox/k380_cross_shell_selector_transfer_2026_10_21/TRANSFER_SUMMARY.json | bcd401b58a5c0ad2cc3bf8fe438ff6e4355f4055606f1df4fd7fc60924b32d99 |
| research/sandbox/k380_high_rank_state_fingerprints_2026_10_21/FINGERPRINTS.csv | 290ca31d317f403eaa6c4df58a0dc11995f6990f0dcaee98370b3f6a6b305848 |
| research/sandbox/k380_p340_s2_base_matched_validation_continuation_001/RUN_RESULT.json | ea3347ea55e564ee09e6ac3d579f44588ae876378f77d349f3152dc80b4ed824 |
| research/sandbox/k380_s2_final_authorization_execution_2026_10_24/NUMERICAL_SUMMARY.json | a16e807cf6b4a4f6018427dba8a22a0bf7407eff373b25ca2faea058d4ad94a1 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/RUN_RESULT.json | 2dd9d825197f40ec7c299b3511bc275bfdd48fd22a201d7b8f7569ea59339eec |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_01_RESULT.json | 40a2eb9f14484e63eca7e48ef970c5981dd8e4248c05bc7145c7596abf05396c |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_02_RESULT.json | 88dce00f3eb7d0f4a28acd35edb9e310df464dab577312b53fae456aa9b3870e |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_03_RESULT.json | 44118bf113d4cbc9923fea995e715333dc266a94cc453802bdd06188b1007d9c |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_04_RESULT.json | d6833b2e5449bc3ca80e9f1518941c0417502d5ba4fa997d3e4daf4a0afbc42e |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_05_RESULT.json | 87bc0d25e98f01574d412f7a40558a85e4ca5b3602e395e02baa6ca81b1453a0 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_06_RESULT.json | 1a22d993a24c34ce895679ec11373debf6183b41fd15d05239debbddc68b41bb |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_07_RESULT.json | c567b19b5b55392844ef10a852199ba1b3dd9dee72588fb687924021ee351a89 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_08_RESULT.json | eed778fd2051ee958a6dbe9c395763bdab1860d24b95f36236707a234bcde459 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_09_RESULT.json | 09ccf83da07cace4fc5c3711b72d335e4d7eccfa636529d6c460f88fc02df99c |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_10_RESULT.json | a20d9c3eafa6b88c649bdffe0275d52df190f5af6fa8e86cbaedd618ba621e39 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_11_RESULT.json | 1eab2e4dab674b331ebfc7c7e26036661e92ea7fc6b508826b754c8964430955 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_12_RESULT.json | b8e4daf8def62ac6d623e1ebab0ac6281b67e39c3f1d6b56eed8540a605f9041 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_13_RESULT.json | 77df69afb022d6ddc30dcab07750c79d72febe0a90d07503e67238098190a6f2 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_14_RESULT.json | 152c07fd26ae2a512fb9d1907a224aa1c2d534c9c323aa1f29cbf2e39bf1e472 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_15_RESULT.json | ce162e5bbd69e1e4f6f0262c53e4ed885d0e7c2493657e5bf048227ded6dcfb2 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_16_RESULT.json | 378246ab1a6b9fda5b78c8b2e117553c9846b59eb7605c53c5ab5b25cba02af0 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_17_RESULT.json | 2f22824d1e2f9183c7c7a8c0c0451c667da518a299dc6288eaf322460df5c2a6 |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_18_RESULT.json | 40978dccfbf8dc7c841ffe08fa4ee170934b7a237d6f3fd76a71eff454cd89dc |
| research/sandbox/k380_selected_space_spectral_accuracy_execution_001/cell_19_RESULT.json | b3b4c312b7dc548dccd8663144c2409f17d7c69f21fbb30c710a3cf74fdef412 |

## Important boundaries

- Historical k380/S2 attribution uses P360; WB4 matched S2 and WB5 use P340 with the [340,360) gap excluded. Both are labeled wherever shown. S1 and matched S2 share P340 but have different full endpoints.
- The historical fingerprints characterize finite-grid Stage-1 states. Their bare block strengths and exact target-conditioned attributions are different quantities.
- Historical residuals certify finite matrix solutions, not spatial/domain/cutoff convergence. U_S2 remains a finite reference.
- SHA-256 authentication is local recorded-byte linkage, not a signature or independent timestamp. Mutable historical note/Git-index snapshots are not mistaken for current numerical-input hashes; the original sandbox handoff recorded a 2,724-file preservation snapshot. This copy is separately validated without changing that record.
- Frozen scientific thresholds, rankings, method identities, production code and existing figures are preserved. The October-20 handoff remains intact.
- Independent future validation is proposed only. No task here releases a HOLD.
