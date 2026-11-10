# Scientific dependencies

An arrow means “is needed to support the stronger downstream claim,” not “has already passed.” The static ranking and algebraic conclusions are exact properties of their stored inputs; they do not need k220 to be statements about those inputs.

```mermaid
flowchart TD
  MODEL["Declared netlist + source/gauge conventions"] --> CP["Authenticated N51/N71 k180 parents"]
  CP --> CHAR["Saved state characterization"]
  CP --> SIGN["Signed reduced grid_phi + energies"]
  CP --> SEL["Oracle-informed retained masks"]
  SEL --> KOLD["Agreement within same k180 parent"]
  SIGN --> RAW["Low11 raw-product ranking"]
  SIGN --> LIE["Abstract finite-model su(N)"]
  SIGN --> PULSE["Specified closed-system shelving result"]
  SIGN --> PERT["Local derivatives / perturbative generators"]
  CP --> TRACK["Identity and subspace tracking"]
  K220["Authenticated same-parent k180/200/220"] --> TRACK
  TRACK --> OP["Separate signed operator-block checks"]
  TRACK --> ENERGY["Separate carrier convergence"]
  SPATIAL["Spatial, charge, domain and N4 refinement"] --> OP
  SPATIAL --> ENERGY
  MODELREVIEW["Reviewed combined circuit model"] --> PHYS["Device-relevant model"]
  ARRAY["Device array matrices / participation"] --> PHYS
  PORT["Physical port and transfer function"] --> DRIVE["Feasible drive model"]
  PHYS --> DRIVE
  OP --> CONTROL["Converged control calculation"]
  ENERGY --> CONTROL
  DRIVE --> CONTROL
  WINDOW["Driven-state and integration convergence"] --> CONTROL
  PULSE --> CONTROL
  LIE -. "reachability only" .-> CONTROL
  RAW -. "candidate selection only" .-> CONTROL
  CONTROL --> GATE["Realistic fixed-target X-gate claim"]
  NOISE["Noise/relaxation + intermediate-state lifetimes"] --> GATE
  TARGET["Logical frame, fidelity metric and acceptance budget"] --> GATE
```

The apparent cycle “choose mask using k180 → agree with k180 → k180 is validated” is **invalid**. The first two operations are legitimate construction and evaluation on the same endpoint. The third inference requires new evidence.

| Conclusion | Needs k220? | Other dependencies |
|---|---|---|
| Recorded raw products, cycle identity, Lie rank, saved pulse metric | No, when explicitly about the frozen inputs | Input authenticity and correct calculation |
| General shelving conjugation / distinction between state and operator convergence | No | Mathematical assumptions and correct definitions |
| Numerical confidence beyond the old retained parent | Yes, or comparable authenticated larger-parent evidence | Tracking, signed operators; no silent threshold changes |
| Spatial convergence | k220 may stabilize the comparison, but is insufficient | More independent spatial/charge/domain checks |
| Physical array participation or physical port | No amount of k-only work resolves it | Device/EM/experimental data and circuit interpretation |
| Robust device pathway selection | k220 contributes, does not suffice | Physical model, port, error objective and actual perturbations |
| Realistic fixed-target X gate | Numerical acceptance including k220 needed before certification | All model, drive, dynamical, noise and target branches |

Algorithmically independent audits share physics inputs. The independently checked Lie theorem and alternative closure provide methodological corroboration, not independent evidence that the inherited Hamiltonian describes a particular device. Exact ledger citations: C01–C03, C07–C08, C16–C24, C28–C45.
