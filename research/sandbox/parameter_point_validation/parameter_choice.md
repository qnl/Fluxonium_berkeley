# Frozen independent parameter point

Frozen before any Stage-1 or dressed-space result at this point.

## Values

The validation point is identical to the authenticated original N71 k300
configuration except for `eps_J`:

| parameter | training point | validation point |
|---|---:|---:|
| EJ | 5.0 GHz | 5.0 GHz |
| EC | 0.5 GHz | 0.5 GHz |
| EL | 1.0 GHz | 1.0 GHz |
| ELK | 1.0 GHz | 1.0 GHz |
| EJS | 4.0 GHz | 4.0 GHz |
| ECS | 8.0 GHz | 8.0 GHz |
| eC | 5.5 GHz | 5.5 GHz |
| eP | 10.0 GHz | 10.0 GHz |
| eps_J | 0.10 | **0.11** |
| eps_LK | 0.05 | 0.05 |
| ng | 0 | 0 |
| phi_ext | 0 | 0 |
| theta_ext | pi | pi |

The frozen configuration fingerprint is
`8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646`.

## Scientific justification

`eps_J` is already an explicit parameter of the established Hamiltonian:
the two KITE Josephson energies are modeled as
`EJ1,2 = EJ_mean (1 +/- eps_J)`, producing the documented asymmetry term
`2 eps_J EJ sin(phi_Sigma) sin(phi_Delta)`. Changing 0.10 to 0.11 therefore
uses the same model and conventions; it is not a production-model edit.

Existing local fixed-parent sensitivity evidence, generated before this
preregistration, reports that `eps_J +0.01` moves f78 by approximately
+3.04 MHz, the primary 0781 score by +3.82%, the 0381 score by +3.18%, and
the 0721 score by -2.02%. It changes `(g07,g78,g81)` by approximately
`(+0.007%,+1.694%,+2.119%)`. This is large enough to perturb the sensitive
7/8 sector and pathway balance while remaining a local 10% relative change
of the existing asymmetry parameter.

The point was proposed independently of any new-point spectrum. No sweep,
new-point matrix assembly, eigenvalue, or dressed-space outcome was inspected.
It is consequently a meaningful held-out parameter test of selector transfer.
It is not a statistically independent device, a fabrication-tolerance model,
or a claim that 0.11 is experimentally realized.

## Restrictions and stop conditions

The future experiment stops rather than changing the point if the registered
Hamiltonian fingerprint, source hashes, selector feature gates, Stage-1 rank
gap gate, or cross-parameter state tracking fails. No alternate point may be
chosen after such a failure without a new preregistration.
