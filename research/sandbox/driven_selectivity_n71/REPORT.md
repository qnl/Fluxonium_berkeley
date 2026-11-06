**Gridium driven selectivity conditional on the N71 four mode checkpoint**

The 6.8761675 MHz spacing is a serious limitation for independently addressing the three edges of `0→7→8→1`. It is not a demonstrated fundamental limitation on logical control. The same drive can intentionally address both outer transitions, and an explicit, unoptimized two-carrier shelving sequence gives about 98% coherent equatorial π-gate overlap in 70 ns in the 48-state reduction. This example has about 1.6% final leakage and nearly complete transient occupation outside the logical subspace. It establishes a useful mechanism, not a high-fidelity or protected gate. Strictly virtual, weak, spectrally selective control is substantially slower.

**Scope and evidence.** All numerical data originate from `research/checkpoints/four-mode-asymmetric-n71-k180`. `COMPLETE`, the checkpoint metadata, the frozen configuration fingerprint, all three payload hashes, and the circuit coefficient source hash were authenticated. No Stage-1 Hamiltonian was assembled or solved. The saved 180-dimensional Stage-1 eigensystem and projected operators were used to reconstruct the existing 1440-dimensional Stage-2 Hamiltonian; only its low eigenpairs were requested. The reconstruction reproduced the authenticated low-11 energies within 2e-13 GHz and coupling magnitudes within 5e-12. See [provenance.json](provenance.json).

The source's pure NumPy `_coeffs` function was read unchanged through Python AST extraction, avoiding an unavailable QuTiP import. The reconstructed Stage-2 formula is the existing `_components`/`_solve` formula, with additional returned eigenpairs solely for driven-space checks. The stored eigenbasis phases were fixed by making Φ01, Φ07, Φ18 positive. Physical closed-loop signs were retained. This is an eigenvector gauge choice, not a change of control operator.

“Exact” below means direct numerical integration of the specified, finite, closed-system Hamiltonian with all cosine terms, to the reported numerical accuracy. It does not mean an exact physical device model. Perturbative formulas and Fourier/error estimates are identified separately. No experimental port calibration, noise model, acceptance threshold, or bound on available amplitude is assumed.

**Reduced driven Hamiltonian.** Use ordinary frequencies in GHz and time in ns:

\[
 i\dot\psi=2\pi h(t)\psi,\qquad h=H/h_{\rm Planck}
 =\operatorname{diag}(\epsilon_j)+u(t)\Phi,
 \quad u(t)=\sum_{\alpha=a,b,c} A_\alpha(t)
 \cos(2\pi\nu_\alpha t+\varphi_\alpha),
 \quad \Phi=\mathrm{grid\_phi}.
\]

Thus a resonant isolated transition has cyclic Rabi frequency Ωij=A|Φij| and off-diagonal rotating-frame element gij=AΦij/2. A rectangular π pulse has Tπ=1/(2Ω); a sin² envelope of total duration T requires peak Ω=1/T. These amplitudes multiply the abstract global phase operator; they are not calibrated flux-line amplitudes.

The four-state analytic core, ordered `(0,1,7,8)`, is

\[
 h_0=\operatorname{diag}(0,0.015857269552,
 6.275014070727,6.297747507785),
\]
\[
\Phi_{0178}\simeq
\begin{pmatrix}
0&0.453425489&1.906482087&0\\
0.453425489&0&0&1.790232029\\
1.906482087&0&0&-1.142340320\\
0&1.790232029&-1.142340320&0
\end{pmatrix}.
\]

Displayed zeros are below about 2e-11, not imposed zeros in the recovered data. The negative loop product Φ07Φ78Φ81Φ10 is gauge invariant. Replacing the matrix by its absolute values would change interference physics.

Four states suffice for the central crowding and interference derivation. They are not a generally adequate leakage model. All states 0–10 were included in initial dynamics: Φ16=−0.449887, Φ19=0.659349, Φ0,10=0.355814, Φ27=0.859354, and Φ38=0.686032 are appreciable. The signed [low-11 matrix](matrix_low11.csv), [energies](energies.csv), and [48-state arrays](reduced_data.npz) are supplied. Additional states materially affect fast shelving, so the main finite-space numerical examples use 48 states. This is a conservative small validation model, not a proof that 48 states are sufficient for arbitrary stronger drives.

**Transition and contamination table.** Detuning means transition frequency minus the positive carrier frequency. These are calculated numbers from the checkpoint, in MHz. The table lists the intended edges and leading competitors; [channels.csv](channels.csv) enumerates every appreciable low-11 transition for each of the nine pathway/tone assignments, including signed couplings, counter-rotating sum frequencies, and relative Rabi strengths.

| Path and tone | Channel | Transition MHz | Φij in the stated gauge | Detuning MHz | Role |
|---|---|---:|---:|---:|---|
| 0781, a | 0–7 | 6275.014071 | 1.906482 | 0 | intended |
| 0781, a | 1–8 | 6281.890238 | 1.790232 | +6.876168 | outer cross-drive |
| 0781, b | 7–8 | 22.733437 | −1.142340 | 0 | intended |
| 0781, b | 0–1 | 15.857270 | 0.453425 | −6.876168 | direct logical drive |
| 0781, c | 1–8 | 6281.890238 | 1.790232 | 0 | intended |
| 0781, c | 0–7 | 6275.014071 | 1.906482 | −6.876168 | outer cross-drive |
| 0781, a / c | 1–6 | 6004.937992 | −0.449887 | −270.076 / −276.952 | logical leakage |
| 0781, a / c | 1–9 | 6850.326509 | 0.659349 | +575.312 / +568.436 | logical leakage |
| 0781, a / c | 0–10 | 7212.207934 | 0.355814 | +937.194 / +930.318 | logical leakage |
| 0781, b | 4–5 | 163.657416 | −0.421982 | +140.923979 | requires population in 4/5 |
| 0781, b | 2–3 | 232.028786 | 0.243641 | +209.295349 | requires population in 2/3 |
| 0781, b | 6–7 | 254.218809 | 0.287346 | +231.485372 | leakage from occupied 7 |
| 0781, a / c | 8–26 | 6005.164704 | 1.114722 | −269.849 / −276.726 | beyond low 11; occupied 8 can leak |
| 0721, a | 0–7 / 1–8 | as above | as above | 0 / +6.876168 | same outer crowding |
| 0721, b | 2–7 | 2819.000502 | 0.859354 | 0 | intended emission |
| 0721, b | 5–9 | 2734.598766 | 0.131395 | −84.401736 | indirect competitor |
| 0721, b | 3–8 | 2609.705153 | 0.686032 | −209.295349 | indirect competitor |
| 0721, c | 1–2 | 3440.156299 | −0.199679 | 0 | weak intended edge |
| 0721, c | 0–3 | 3688.042355 | −0.206877 | +247.886056 | logical leakage |
| 0381, a | 0–3 | 3688.042355 | −0.206877 | 0 | weak intended edge |
| 0381, a | 2–10 | 3756.194365 | −0.044872 | +68.152010 | indirect competitor |
| 0381, a | 1–2 | 3440.156299 | −0.199679 | −247.886056 | logical leakage |
| 0381, b | 3–8 | 2609.705153 | 0.686032 | 0 | intended |
| 0381, b | 3–6 | 2332.752907 | −0.409383 | −276.952246 | leakage from occupied 3 |
| 0381, c | 1–8 / 0–7 | as above | as above | 0 / −6.876168 | same outer crowding |

Transitions such as 5–9 and 2–10 should not be interpreted as immediately populated errors: their initial states must first acquire population. Conversely, shelving populates 7/8 nearly completely, making their outgoing channels relevant. For the common outer carrier, additional appreciable channels include 7–29 at +735.969 MHz (|Φ|=1.553), 7–31 at +1066.478 MHz (1.308), and 8–32 at +1271.186 MHz (1.611). Off-resonant shifts can matter even when final population in those states is small.

**Rotating frame and Floquet construction.** For path `(0,a,b,1)`, let s=sign(εb−εa), and set

\[
 r_0=0,\quad r_a=\nu_a,\quad r_b=\nu_a+s\nu_b,
 \quad r_1=\nu_a+s\nu_b-\nu_c.
\]

With ψlab=RψR and Rjj=exp(−i2πrjt), the exact transformed Hamiltonian is

\[
 (h_R)_{ij}=(\epsilon_i-r_i)\delta_{ij}
 +\sum_\alpha {A_\alpha\Phi_{ij}\over2}
 \left[e^{i2\pi(r_i-r_j-\nu_\alpha)t-i\varphi_\alpha}
 +e^{i2\pi(r_i-r_j+\nu_\alpha)t+i\varphi_\alpha}\right].
\]

No RWA has been made in this expression. Write Da=εa−νa, Db=εb−νa−sνb, and d=ε1−νa−sνb+νc. The intended rotating couplings are ga=(AaΦa0/2)e^(−iφa), gb=(AbΦba/2)e^(−isφb), and gc=(AcΦ1b/2)e^(+iφc). In path order `(0,a,b,1)`, their graph is

\[
 h_{\rm chain}=\begin{pmatrix}
0&g_a^*&0&0\\
g_a&D_a&g_b^*&0\\
0&g_b&D_b&g_c^*\\
0&0&g_c&d
\end{pmatrix}.
\]

This graph alone is an approximation. For the primary path let δh=νc−νa. Its additional slowly varying terms are

\[
 (h_R)_{70}={\Phi_{70}\over2}
 [A_a e^{-i\varphi_a}+A_c e^{-i(2\pi\delta_h t+\varphi_c)}],
\]
\[
 (h_R)_{81}={\Phi_{81}\over2}
 [A_c e^{-i\varphi_c}+A_a e^{i2\pi\delta_h t-i\varphi_a}],
\]
\[
 (h_R)_{10}^{(b)}={A_b\Phi_{10}\over2}
 [e^{-i(2\pi\delta_h t+\varphi_b)}
 +e^{i2\pi(r_1+\nu_b)t+i\varphi_b}],
\]
\[
 (h_R)_{87}^{(b)}={A_b\Phi_{87}\over2}
 [e^{-i\varphi_b}+e^{i4\pi\nu_b t+i\varphi_b}].
\]

At the bare primary carriers δh=6.876167505 MHz, r1=15.857269552 MHz, r1+νb=38.590706610 MHz, and 2νb=45.466874115 MHz. High-carrier counter-rotating terms oscillate around 12.55 GHz. The slow outer beat and the off-resonant logical term survive a justified coarse high-frequency RWA. They disappear only under an additional secular approximation requiring amplitudes and envelope bandwidth small compared with 6.876 MHz. The 38.591 and 45.467 MHz terms require explicit checking; they are retained in all “full cosine” numerical results.

Equivalently use generalized Sambe states |j,nₐ,nᵦ,n𝚌⟩, with diagonal εj+n·ν and matrix elements AαΦij exp(±iφα)/2 connecting n to n±eα. An absorption uses the −eα Fourier component. The intended resonant logical copy is |1,−1,−s,+1⟩. Distinct generally incommensurate carriers require this multifrequency construction; no artificial short common Floquet period is assumed. Other nearly degenerate photon copies, especially those separated by νc−νa, must be retained or perturbatively eliminated with their actual small denominators.

The primary spectral relation is the exact energy-cycle identity

\[
 f_{81}-f_{07}=f_{78}-f_{01}=6.876167505\ {\rm MHz}.
\]

The graph therefore has a phase-sensitive loop. Relative to the direct logical co-rotating term, the three-step amplitude has phase π−φa+φc+2πδh t in the chosen gauge. Changing φb rotates both amplitudes together in this comparison; it cannot independently reverse one of them. The other relative phase controls interference, but the phase winds over the 145.43 ns beat time.

**Stark shifts, beat terms, and validity.** Off-resonant, nondegenerate second-order Floquet perturbation theory gives

\[
 S_n=\sum_{\alpha,k}{A_\alpha^2|\Phi_{nk}|^2\over4}
 \left[{1\over\epsilon_n-\epsilon_k+\nu_\alpha}
 +{1\over\epsilon_n-\epsilon_k-\nu_\alpha}\right].
\]

Both denominators are needed. Resonant channels are excluded from this formula and treated in the retained dynamical block; plugging a resonant denominator into a Stark formula is invalid. For a slowly varying envelope the formula is local in A(t), with derivative corrections small only when the envelope changes slowly compared with the eliminated gaps.

For the direct logical action of the primary middle tone,

\[
 S_1-S_0=2|g_{01}^{(b)}|^2
 \left[{1\over f_{01}-\nu_b}+{1\over f_{01}+\nu_b}\right].
\]

At peak Ω78=1 MHz, this is −9.415 kHz; the counter-rotating contribution partially cancels the co-rotating Stark shift. Scaling it quadratically to much stronger drive is only an estimate until |g01|≪6.876 MHz is checked. For a resonantly driven 7–8 pair the counter-rotating diagonal correction instead gives a leading splitting shift Ω78²/(4f78); the resonant mixing itself must remain in the block.

Two close outer tones also produce phase-dependent diagonal beat terms. If their detunings from 0–7 are approximately a common D much larger than their splitting, then

\[
 S_0(t)\simeq-{\Phi_{07}^2\over4D}
 [A_a^2+A_c^2+2A_aA_c\cos(2\pi\delta_h t+\varphi_c-\varphi_a)].
\]

For unequal detunings use the symmetrized resolvent denominators. Analogous terms occur for state 1, as well as contributions from other levels. Averaging this beat away on a 20–50 ns pulse is not justified. A differential static Stark term can be compensated by setting d=−(S1−S0), or by a calibrated final logical Z correction; constant detuning does not cancel every time-dependent beat or amplitude-dependent shift.

**Selectivity and time scales.** These are Fourier and perturbative estimates, not a quantum speed limit. With Δ=6.876167505 MHz,

\[
1/\Delta=145.42985\ {m ns},\qquad
1/(2\pi\Delta)=23.14588\ {m ns}.
\]

The latter is one radian of phase discrimination, not spectral resolution. A rectangular envelope has normalized spectral magnitude |sinc(ΔT)| and its first null at 145.43 ns. A sin²/Hann envelope has |sinc(ΔT)/(1−(ΔT)²)| and its first null at 290.86 ns. These are pulse-family nulls, not universal minimum durations. The exact [Fourier table](Fourier_overlap.csv) includes:

| Duration ns | Rectangular amplitude at the neighbor | Hann amplitude at the neighbor |
|---:|---:|---:|
| 20 | 0.9692 | 0.9879 |
| 50 | 0.8166 | 0.9261 |
| 100 | 0.3848 | 0.7299 |
| 145 | 0.0030 | 0.5022 |
| 291 | 0.00048 | 0.00016 |

At fixed target Rabi frequency the unwanted logical Rabi strength is 0.396927 Ω78. Each outer tone's unwanted outer Rabi strength is 0.939024 or 1.064936 times its intended strength. The latter is essentially equally strong. A single off-resonant rectangular two-level drive gives exactly, within its RWA,

\[
 P_{\rm off}(t)={\Omega_{\rm off}^2\over\Delta^2+\Omega_{\rm off}^2}
 \sin^2[\pi t\sqrt{\Delta^2+\Omega_{\rm off}^2}].
\]

Thus weak leakage scales as (Ωoff/Δ)², the amplitude as Ωoff/Δ, and Stark shifts as Ωoff²/Δ. Requiring the maximum of this oscillation to remain below 1% gives a conservative rectangular π-pulse duration of 770 ns for the worse outer direction and 287 ns for the middle/logical competition. At 0.1% these become 2.45 μs and 912 ns. Hann pulses of equal area have twice the peak Ω, so applying the same conservative peak-amplitude criterion doubles these durations. Smooth envelopes can yield much smaller final error; these worst-transient-population criteria are sufficient conditions, not lower bounds.

Independent narrowband selectivity is therefore implausible below 20 ns and still poor at 20–50 ns. Hundreds of ns or longer are natural scales for this specific selective-control approach. Strong coherent collective control is a different regime and is not excluded by 1/Δ. No rigorous lower bound on useful gate duration follows without amplitude, bandwidth, leakage, fidelity, and intermediate-occupation constraints.

**Phase and detuning cancellation.** A constant phase cannot remove the middle tone's first-order logical coupling while preserving its 7–8 coupling: both arise from the same scalar control coefficient. The GHz outer tones do not supply another first-order Fourier component near 15.857 MHz. Likewise a relative phase can cancel two outer amplitudes at an instant, but their 6.876 MHz relative phase then evolves. In the exactly coincident-carrier limit cancellation removes the shared physical field, rather than selecting only one transition.

At first order, the finite-pulse logical transition amplitude is proportional to

\[
 e^{-i\varphi_b}F(f_{01}-\nu_b)
 +e^{+i\varphi_b}F(f_{01}+\nu_b),
\qquad F(x)=\int A_b(t)e^{i2\pi xt}\,dt.
\]

A phase-only exact zero requires the two Fourier magnitudes to match; it is not generic. Spectral notches, timed returns, quadrature shaping, and composite pulses can cancel first-order final amplitudes. Such zeros are pulse-specific and do not generally remove transient population, Stark shifts, higher-order amplitudes, or errors from other levels. Derivative-shaped notches carry an amplitude/bandwidth cost that grows when |Δ|T is small. Small detunings move these conditions and can compensate a Stark shift, but cannot independently move the two bare transitions addressed by the same tone.

There is also an instructive dressed-state cancellation. Under one common high-frequency carrier 50 MHz below f07, the two outer pairs have detunings D0=50 MHz and D1=56.876 MHz. In the four-state high-frequency RWA, tan(2θj)=AHΦouter,j/Dj, and the low dressed-state matrix element is

\[
 \Phi_{\rm dressed}=\Phi_{01}\cos\theta_0\cos\theta_1
 +\Phi_{78}\sin\theta_0\sin\theta_1.
\]

It vanishes at AH=60.5365 MHz, with excited-state weights 30.12% and 26.76%. A ±2% amplitude change gives Φdressed≈+0.005675 or −0.005514. See [dressing_cancellation.json](dressing_cancellation.json). This cancellation is tunable interference, not symmetry protection, and lies outside a weak-dressing expansion. It suppresses the total low-frequency dressed transition, including the desired logical drive; it is not a demonstrated fast X-gate solution. Additional-level shifts would require recalibration.

The numerically useful alternative tested here is population shelving: the middle tone acts principally on the occupied 7/8 manifold while 0/1 is nearly empty. Suppression of its unwanted logical action is then a property of the state trajectory, rather than a zero in Φ01. Finite preparation error leaves residual direct action. This mechanism works over a finite parameter neighborhood, but is not protected against general phase, detuning, or amplitude errors.

**Effective logical Hamiltonian.** On exact intended three-photon resonance, and when the intermediate dressed block has a gap much larger than the couplings connecting it to the logical block, a Feshbach/SW elimination of the isolated chain gives

\[
 h_L\simeq\begin{pmatrix}
-|g_a|^2D_b/Q & (g_ag_bg_c/Q)^*\\
g_ag_bg_c/Q & d-|g_c|^2D_a/Q
\end{pmatrix},\qquad Q=D_aD_b-|g_b|^2.
\]

This is the zero-energy resolvent approximation, exact in internal gb but perturbative in the outer couplings. Its gap condition concerns eigenvalues of the intermediate 2×2 block; a small Q invalidates it. In the fully weak regime κ=gagbgc/(DaDb) is third order, S0=−|ga|²/Da and S1=−|gc|²/Db are second order, and instantaneous virtual occupation is of order |ga/Da|² and |gc/Db|². A sudden turnoff can leave that occupation as leakage; sufficiently slow ramps can return it, subject to their own gap conditions. The logical cyclic X/Y Rabi rate is 2|κ|, Tπ=1/(4|κ|), and Z splitting is d+S1−S0. The phase of κ is arg(Φa0ΦbaΦ1b)−φa−sφb+φc. For Da=Db=0 there is no valid logical-only SW model of this chain.

The isolated chain misses other equally ordered processes. To calculate the resonant three-photon matrix element without discarding other states or time orderings, let q=(νa,sνb,−νc), with Σq=f01. For equal constant amplitudes A, the third-order Sambe entry is

\[
 \kappa^{(3)}={A^3e^{-i(\varphi_a+s\varphi_b-\varphi_c)}\over8}
 \sum_{\pi\in S_3}\sum_{i,j}
 {\Phi_{1j}\Phi_{ji}\Phi_{i0}\over
 [q_{\pi_1}-\epsilon_i]
 [q_{\pi_1}+q_{\pi_2}-\epsilon_j]}.
\]

The sum excludes any intermediate state that belongs to the retained degenerate block; none of those denominators vanishes in the numerical example. All 48 physical states and all six time orderings were used. This calculation is third order, not a converged nonperturbative Floquet solve. Other photon combinations yield different harmonics, including nearby sidebands, and are not represented by this one coefficient. The complete logical dynamics also contains the off-resonant first-order drive, diagonal beat terms, and micromotion; a constant κ alone does not predict a gate until these are controlled.

For a common abstract tone amplitude A=2 MHz, Da=Db=50 MHz, and bare three-photon resonance, the results are:

| Path | Isolated κ kHz | Full third-order κ kHz | Differential Stark kHz | Nominal π time from full κ |
|---|---:|---:|---:|---:|
| 0–7–8–1 | −1.55954 | −0.98834 | −11.79065 | 252.95 μs |
| 0–7–2–1 | −0.13086 | −0.21701 | +16.14310 | 1152.00 μs |
| 0–3–8–1 | −0.10163 | −0.17422 | +21.66809 | 1434.98 μs |

The amplitudes and detunings are a declared common comparison point, not a recommended pulse. These π times assume Stark compensation and isolation of the resonant Fourier component. Without compensation the listed Z shifts exceed the X rates, so the corresponding untuned drive is not an X gate. At this point the largest intended g/D is about 0.038; the primary off-resonant logical g/Δ is about 0.066. Adiabatic ramps and control of omitted higher orders are still required on the long inferred times.

For the primary path two large extra contributions are `0→7→0→1` with tone order a,c,b (+4.79352 kHz), and `0→1→8→1` with order b,a,c (−4.22677 kHz). They partially cancel each other and interfere with the intended −1.55954 kHz contribution. They have denominators involving 6.876 MHz. The [term table](floquet_third_order_terms.json) exposes this explicitly. The stronger primary product remains advantageous in this comparison, but its full effective coupling is about 4.55 times the 0721 value, rather than the 11.92 ratio of the bare products. Exact diagonalization of the isolated static 4×4 graphs agrees with their leading κ within 0.36%, 0.20%, and 0.18%, respectively; that check validates the isolated-chain expansion only.

**Pathway comparison.** The following frequencies are bare, before any Stark compensation:

| Quantity | 0–7–8–1 | 0–7–2–1 | 0–3–8–1 |
|---|---:|---:|---:|
| Carrier frequencies GHz | 6.275014, 0.022733, 6.281890 | 6.275014, 2.819001, 3.440156 | 3.688042, 2.609705, 6.281890 |
| Photon signs along path | +,+,− | +,−,− | +,+,− |
| Smallest separation between its carriers MHz | 6.876 | 621.156 | 1078.337 |
| Weakest intended coupling magnitude | 1.14234 | 0.199679 | 0.206877 |
| Product of intended coupling magnitudes | 3.898859 | 0.327142 | 0.254077 |
| Closest carrier to logical transition MHz | 6.876 | 2803.143 | 2593.848 |
| Unavoidable strong neighboring outer transition MHz | 6.876 | 6.876 | 6.876 |

The comparators remove the low-frequency middle-tone/direct-logical near resonance, and their carrier sources are easily separated spectrally. They do not remove the outer transition spectator. For 0721, a 0–7 tone drives 1–8 with relative strength 0.939; for 0381, a 1–8 tone drives 0–7 with relative strength 1.065. Both logical inputs must be propagated to see this error. Evaluating only transfer from |0⟩ would miss an important part of the problem.

At the same peak coefficient A, the weakest comparator leg needs about 5.7 or 5.5 times the primary's weakest-leg π time. To force the same short ideal-chain duration, the comparator needs a much larger drive coefficient on its weak leg, increasing far-off-resonant shifts and sensitivity to additional levels. For example the prescribed 50 ns comparator pulses reach coefficients around 174 MHz (0721) or 167 MHz (0381), versus about 35 MHz for the primary's middle tone. This is why strong-drive comparator rankings are not established by the low-11 graph.

0721 is more attractive for avoiding the direct low-frequency logical drive and for selective slow control, but it is not unconditionally superior in driven dynamics. It remains slower at equal drive amplitude in the displayed perturbative comparison, and its outer spectator persists. Prior state-containment or basis-robustness advantages remain relevant but do not settle this control tradeoff.

**Numerical pulse definitions and metrics.** Every pulse here has a specified sin² envelope and zero outside its interval. No numerical optimizer was run. The prescribed simultaneous chain family sets peak intended off-diagonal elements `(√3,2,√3)/(2T)`, with phases making the three isolated graph edges positive. Its intended-edge-only RWA gives exact endpoint mirroring at T. Applying its actual scalar three-tone waveform tests the approximation, rather than optimizing it.

The two-carrier shelving example uses a common outer carrier at 6.278452154 GHz. An outer sin² pulse from 0 to 10 ns has peak coefficient 54.102101 MHz and phase 0; the middle pulse from 10 to 60 ns is at 22.733437 MHz, coefficient 17.507917 MHz, phase 0; an outer pulse from 60 to 70 ns repeats the first with phase π. This deliberately drives both outer transitions. The middle pulse phase is defined against absolute time, so delaying a pulse without adjusting its phase changes its carrier waveform.

All reported gates propagate both logical columns. With K=PL U PL in the interaction picture of the bare Hamiltonian, mean final leakage is 1−Tr(K†K)/2. The average overlap with a target unitary V is

\[
 F={\operatorname{Tr}(K^\dagger K)+|\operatorname{Tr}(V^\dagger K)|^2\over6}.
\]

Unless explicitly marked “fixed axis,” F is maximized only over the equatorial axis of a π rotation. It includes loss from leakage and is not merely a population-transfer probability or postselected fidelity. A known axis and free-evolution phase can be calibrated, but this is not a claim of fidelity to a preassigned laboratory X gate. The robustness checks also report overlap with the baseline's fixed axis, so changes of axis are not silently forgiven.

The full-cosine results, fixed pulse definitions, approximation comparisons, and full coherent columns are in [experiments.csv](experiments.csv), [refinements.csv](refinements.csv), [final_checks.csv](final_checks.csv), and the corresponding NPZ files. The 70 ns sequence gives F=0.989344 with 11 states, 0.979591 with 32, and 0.981051 with 48. Final leakage in 48 states is 0.016057. Tightening the solver tolerance changes F by 7.4e-10. This verifies time integration, not Hilbert-space convergence to that precision.

The short-duration failures of the independent-chain prescription are not impossibility proofs: its primary full-cosine low-11 F is 0.254 at 20 ns, 0.436 at 150 ns, and 0.759 at 500 ns. In the explicitly frequency-truncated exploratory model it rises to 0.943 at 1 μs and 0.985 at 2 μs; those last two values are not full-cosine 48-state gate validations. Conversely, the shelving example demonstrates that useful collective behavior need not wait for a 145 ns spectral null. Whether 20–50 ns can meet a specified high-fidelity threshold on the primary pathway remains unresolved by these finite tests.

The finished 48-state, full-cosine comparison is:

| Prescribed protocol | Duration ns | Equatorial π overlap F | Final mean leakage |
|---|---:|---:|---:|
| 0781, simultaneous ideal-chain prescription | 500 | 0.757322 | 0.111346 |
| 0721, simultaneous ideal-chain prescription | 50 | 0.396523 | 0.555111 |
| 0721, simultaneous ideal-chain prescription | 500 | 0.988896 | 0.010785 |
| 0381, simultaneous ideal-chain prescription | 50 | 0.210739 | 0.729684 |
| 0381, simultaneous ideal-chain prescription | 500 | 0.977537 | 0.020139 |
| 0781, two-carrier shelving | 70 | 0.981051 | 0.016057 |

These compare pulse families with equal intended-chain duration, not equal maximum drive coefficient or optimized performance. The 500 ns results strengthen the case for 0721 as a selective-control comparator. They do not demonstrate that it beats a phase-engineered primary pulse under a common hardware constraint.

The RWA check is also quantitative. For primary pulses of 20, 150, and 500 ns, retaining slow terms up to 350 MHz gives F=0.2603, 0.4360, and 0.7585 with both Fourier signs, versus 0.6776, 0.4891, and 0.7667 after dropping counter-rotating terms. The corresponding full-cosine low-11 results are 0.2541, 0.4360, and 0.7585. The low-frequency RWA is particularly unreliable in the shortest case.

The baseline 70 ns sequence has finite tolerance to small changes but no protected cancellation. The small neighborhood uses ±2% common amplitude scaling, ±0.5 MHz middle detuning, and ±0.1 rad middle phase changes. Exact full-cosine checks and fixed-axis overlaps are in [exact_robustness.csv](exact_robustness.csv). Exploratory frequency-cutoff calculations omit off-resonant Stark contributions and overestimate baseline F by about 0.0032; they are not substituted for exact checks.

For the two exact amplitude/detuning corner samples, fixed-axis F is 0.96378 and 0.96703; for middle-phase shifts −0.1 and +0.1 rad it is 0.97172 and 0.96903. The corresponding best-axis overlaps remain between 0.9764 and 0.9822. Thus some sensitivity is a coherent axis error, and the finite neighborhood should not be described as a robust 98% fixed-X gate.

**Is three-tone control necessary?** No, under this abstract control model. The explicit shelving example uses two carrier frequencies. A one-carrier drive near f01 is also allowed because |Φ01|=0.453425 is finite. As a diagnostic, a 10 ns pulse at f01, with the cosine centered at the envelope maximum and its area corrected without an RWA, gives about 0.97597 equatorial π overlap in the 32-state full-cosine calculation, with negligible final leakage. It is not a primary-pathway gate, not an optimized gate, and not evidence of physical line feasibility. It does rule out reading 145 ns as a universal control limit of the stored spectrum.

Two high-frequency tones alone do not supply an ordinary second-order logical Raman transition at this checkpoint: the observed bipartite Φ selection pattern makes Σk Φ1kΦk0 approximately zero. Odd-order action or a direct low-frequency component is needed for that mechanism. One common high carrier plus one low carrier can implement shelving, or dress an already allowed direct logical transition. With a common high carrier, the net frequency of a virtual absorb-low-emit-high sequence is the low frequency; the low tone must then match the dressed logical splitting, rather than automatically staying at the bare 7–8 frequency. “Three photon” does not imply “three distinct carriers.”

**Conclusions and model limitations.** The primary path is intrinsically difficult to treat as three independently addressable edges, but a controllability obstruction has not been found or derived. Its large matrix elements retain a significant rate advantage; they also make outer cross-talk strong. Phase and detuning engineering can compensate coherent errors or create pulse-specific cancellation, while shelving exploits the same cross-drive constructively. None of the cancellations studied is symmetry protected. The 0721 comparator improves low-frequency isolation but keeps one outer crowding problem and a weak intended leg. A useful-duration lower bound cannot be stated without explicit engineering and error constraints; the justified statement is a selective-pulse scale of roughly 0.1–1 μs or longer, with faster collective examples possible.

The following survive modest shifts of common high-energy frequencies: the cycle identity between the two splittings, the need to retain all transitions driven by the same operator, the distinction between real shelving and virtual Raman control, the possibility of shared-carrier operation, and the inadequacy of matrix-element products as gate rankings. The numerical 6.876 MHz value, Fourier times, Stark compensation, interference roots, and gate examples do not survive arbitrary differential shifts unchanged. A few MHz is modest relative to 6 GHz but large relative to the relevant selectivity scale; even a sign change or near-degeneracy of Δ would qualitatively change that analysis.

All conclusions are conditional on the inherited four-mode Hamiltonian and its saved k=180, N71, N4=8 parent. The calculations add no omitted array modes and provide no test of physical model completeness. They do not revalidate high excited states against larger Stage-1/spatial cutoffs. The 32-to-48-state differences and particularly the large comparator truncation dependence at 50 ns preclude precision fast-gate claims. No decoherence, relaxation during shelving, physical control-line transfer function, hardware amplitude bound, or protection-preservation metric is included. There is no rigorous quantum speed limit, no optimized gate, no complete Floquet convergence claim, and no production-code change.

**Reproduction and verification.** Run the following from the repository root; every generated file stays in this directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/experiments.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/refinements.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/perturbation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/final_checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/verification.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/sandbox/driven_selectivity_n71/plots.py
```

The numerical checks include independent cosine-propagator agreement, eigenvector-gauge covariance, the analytic ideal-chain mirror identity, Hamiltonian Hermiticity, checkpoint reproduction, integration tolerance, and retained-state comparisons. Production tests were not run because production code was unchanged; the available Python environment lacks QuTiP, an existing environment limitation. Existing modifications to `EmilianoUPDATE.md`, `Simulations/Cphase/main.ipynb`, and `Simulations/Cphase/workflow_funcs.py` were preserved. No commit or push was made.

The independent propagators agree within 7.8e-12 in amplitude, gauge covariance within 5.4e-14, and the intended-only analytic chain mirror probability is 1 to numerical precision. See [verification.json](verification.json). Plots are [selectivity and driven dynamics](selectivity_and_dynamics.png) and [phase and truncation sensitivity](phase_and_truncation.png); their PDFs are also saved for export.

The additional [48-state channel inventory](channels_from_path_states_48.csv) lists all appreciable one-step channels out of each pathway's occupied states for all three prescribed carriers, including off-resonant channels outside the original low 11. [Workspace verification](workspace_verification.json) records that the initial worktree and staged production diffs were preserved.
