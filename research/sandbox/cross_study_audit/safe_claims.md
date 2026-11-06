# Statements safe to use today

These formulations are conditional on their explicit scope. Exact evidence and falsification conditions are in [claim_ledger.csv](claim_ledger.csv).

1. **Implementation:** The repository implements and has internally tested a declared asymmetric four-mode netlist and its abstract global-phase control. That is not author/device validation of the combined physical Hamiltonian. [C01–C03]
2. **Numerical reliability:** Authenticated checkpoints and cross-study reproduction support internal reliability. Late N71-priority low-energy observables are stable relative to the saved k180 endpoint at stated tolerances. k180, spatial resolution, oscillator cutoffs and driven-state windows are not all established converged. [C28–C36]
3. **States 7/8:** They are individually distinguishable in the vector-backed saved low-energy comparisons, more mixed across the declared Stage1/mode4 split and broader in theta3 than logical states. They are not broader in every coordinate; cutoff containment is not a convergence proof. [C04–C06, C30]
4. **Ranking:** 0781 maximizes the raw absolute grid_phi product among the enumerated simple low11 paths. That is not a ranking of effective driven rates, fidelity or hardware feasibility. [C07–C08]
5. **Reachability:** The specified stored 2/4/6/11/16/48-state control pairs have full su(N) after trace removal. This supports arbitrary logical SU(2) with unrestricted scalar control and variable duration in those models, without furnishing a fast feasible pulse. [C21–C27]
6. **Gate feasibility:** A prescribed two-carrier 70 ns pulse demonstrates collective shelving and approximately 98.1% best-axis equatorial-pi overlap in a finite closed model. Its large transient logical leakage, unconverged physical basis and absent hardware/noise treatment preclude realistic gate validation. [C17–C20]
7. **Comparators:** 0721 is an important retained-mask and state-8-risk comparator. 0381 supplies a different tradeoff and has smaller saved spatial product drift. Neither is universally robust or immune to an omitted array mode. [C09–C11]
8. **Array mode:** A distributed internal mode is a credible omitted-model threat. Its frequency, branch and participation for this benchmark are unspecified; it is a different coordinate from the implemented Cp node mode. A predictive extension requires a consistent device-specific circuit reduction. [C39–C42]
9. **Realistic X gate:** Not yet validated. The remaining task is broader than improving a pulse: it includes model/numerical convergence, target/frame definition, port calibration and an open-system error budget. [C44]

# Claims to avoid

- “k180 is converged” or “k220 agreement proves the device model.”
- “Spatial convergence is no longer an issue because the labels track.”
- “All k172 bases are equivalent” or “common k172 is precision-certified.”
- “0781 is physically validated / always fastest / highest fidelity.”
- “0721 is the most robust path under every metric.”
- “The 70 ns X gate is validated” or “the sampled neighborhood is a robust 98% fixed-X gate.”
- “1.6% final leakage means the qubit stayed protected” or “99% transient leakage means irreversible 99% loss.”
- “Full controllability implies fast gates, strict-bandwidth reachability or control of the physical infinite circuit.”
- “High-only RWA sectors prove all GHz-only exact waveforms cannot transfer logical population.”
- “A three-step process needs three distinct tones.”
- “145 ns is a quantum speed limit.”
- “High overlap, tiny energy error, or a flat product certifies every operator edge.”
- “Synthetic avoided crossings or tiny integration residuals validate Gridium physics.”
- “The omitted array mode definitely destroys the primary path” or “the 15.8 GHz mode already represents that internal mode.”
- “A measured array frequency fixes its coupling” or “an arbitrary appended oscillator is a predictive device model.”
- “Zero first-order bias slopes mean no noise sensitivity.”
- “Independent sandbox studies are independent experimental confirmations.”
- “A mock k220 PASS is a scientific result.”

These are explicit limits on inference, not evidence that useful gates or favorable model refinement are impossible.
