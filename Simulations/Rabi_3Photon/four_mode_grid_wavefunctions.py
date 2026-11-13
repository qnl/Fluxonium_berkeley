"""Are the four-mode logical eigenstates grid (GKP-like) states?

Plots the probability density of the low eigenstates along the global grid
phase ``u = theta3 + theta2/2`` (the coordinate behind ``grid_phi``) and along
the conjugate charge ``n3``.  It also builds the two combinations
``|L> = (|0> + |1>)/sqrt2`` and ``|R> = (|0> - |1>)/sqrt2``.

Reading the output
------------------
GKP-like case: |L> and |R> are interleaved combs in u (peaks at even and odd
multiples of the grid spacing), with almost no overlap.  Then
``|<L|grid_phi|R>|`` is tiny, while ``<0|grid_phi|1>`` (which equals
``(<L|u|L> - <R|u|R>)/2``) is set only by how the two codewords' centers of
mass differ.

Fluxonium-like (weak) case: |L> and |R> are single lumps on either side of
u = 0, i.e. a double-well pair rather than a grid.  Then ``<0|grid_phi|1>`` is
the lump displacement, which is O(1).

Wavefunctions are rebuilt with the same module functions as ``Gridium4Mode``,
summing over the theta1 charge and the theta4 oscillator.  Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from Circuit_Objs import qchard_gridium_netlist as net
from Simulations.Rabi_3Photon import gridium_params as gp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Figures' / 'Rabi_3Photon'

#active point: paper regime (d) with eps_K=2.5, eps_p=5.5 (gridium_params)
BENCH = gp.circuit_only(gp.ACTIVE_PHYSICS)
CUT = gp.cutoffs('fast')
#parameters used before 2026-11-06 (regime a, eC=5.5, eP=10)
LEGACY_BENCH = gp.circuit_only(gp.LEGACY_PHYSICS)
LEGACY_CUT = dict(n1max=4, N2=51, L2=11.0, N3=51, L3=14.0, N4=4, nkeep=60)
NLEV = 4


def solve_states(physics, cut, eps_J=0.0, eps_LK=0.0, ng=0.0,
                 phi_ext=0.0, theta_ext=np.pi, nlev=NLEV):
    """Return energies, full-grid densities, and helpers for the low states."""
    ECm, K, EJ1, EJ2 = net._coeffs(
        physics['EJ'], physics['EC'], physics['EL'], physics['ELK'],
        physics['EJS'], physics['ECS'], eps_J, eps_LK,
        eC=physics['eC'], eP=physics['eP'])
    H1, o = net._assemble_nonlinear_sector(
        ECm[:3, :3], K[:2, :2], EJ1, EJ2, physics['EJS'], ng,
        phi_ext, theta_ext, cut['n1max'], cut['N2'], cut['L2'],
        cut['N3'], cut['L3'])
    sigma = -(EJ1 + EJ2 + physics['EJS'] + 10.0)
    w1, V, _ = net._stage1_eigensolve(H1, k=cut['nkeep'], sigma=sigma)

    N4 = cut['N4']
    A4, B4 = 4 * ECm[3, 3], 0.5 * K[2, 2]
    f4 = (A4 / (4 * B4)) ** 0.25
    g4 = 1.0 / (2 * f4)
    a = np.diag(np.sqrt(np.arange(1, N4)), 1)
    th4 = f4 * (a + a.T)
    n4 = 1j * g4 * (a.T - a)
    w4 = 2 * np.sqrt(A4 * B4) * (np.arange(N4) + 0.5)
    kron3 = o['kron3']

    def proj(op):
        return V.conj().T @ (op @ V)

    n1_p = proj(kron3(sps.diags(o['nvec'] + ng), o['I2'], o['I3']))
    n2_p = proj(kron3(o['I1'], o['n2'], o['I3']))
    n3_p = proj(kron3(o['I1'], o['I2'], o['n3']))
    x2_p = proj(kron3(o['I1'], sps.diags(o['x2']), o['I3']))
    x3_p = proj(kron3(o['I1'], o['I2'], sps.diags(o['x3'])))
    nk = cut['nkeep']
    Hc = np.kron(np.diag(w1), np.eye(N4)) + np.kron(np.eye(nk), np.diag(w4))
    Hc = Hc + 8 * ECm[0, 3] * np.kron(n1_p, n4)
    Hc = Hc + 8 * ECm[1, 3] * np.kron(n2_p, n4)
    Hc = Hc + 8 * ECm[2, 3] * np.kron(n3_p, n4)
    Hc = Hc + K[0, 2] * np.kron(x2_p, th4)
    Hc = Hc + K[1, 2] * np.kron(x3_p, th4)
    Hc = 0.5 * (Hc + Hc.conj().T)
    evals, W = spla.eigh(Hc, subset_by_index=(0, nlev - 1))

    G = W.conj().T @ np.kron(0.5 * x2_p + x3_p, np.eye(N4)) @ W

    N1 = 2 * cut['n1max'] + 1
    shape = (cut['N3'], cut['N2'], N1)   #kron3 index order: (i3, i2, i1)

    def amplitudes(coeffs):
        """Full Stage-1 grid amplitudes for each theta4 level."""
        c = coeffs.reshape(nk, N4)
        return [(V @ c[:, m]).reshape(shape) for m in range(N4)]

    return dict(evals=evals, W=W, G=G, amplitudes=amplitudes,
                x2=o['x2'], x3=o['x3'], N4=N4, nkeep=nk)


def density_2d(amps):
    """Density on the (theta3, theta2) grid, summed over theta1 and theta4."""
    return sum(np.sum(np.abs(a) ** 2, axis=2) for a in amps)  #(N3, N2)


def phase_marginal(amps, x2, x3, bins, smooth=0.25):
    """Density along u = theta3 + theta2/2, summed over theta1 and theta4.

    The (theta2, theta3) grid points project onto u at irregular spacings, so
    a raw histogram shows aliasing ripples.  A Gaussian kernel of width
    ``smooth`` (rad), below the pi grid spacing, removes them.
    """
    dens = density_2d(amps)
    U = x3[:, None] + 0.5 * x2[None, :]
    centers = 0.5 * (bins[1:] + bins[:-1])
    w = np.exp(-0.5 * ((centers[:, None] - U.ravel()[None, :]) / smooth) ** 2)
    p = w @ dens.ravel()
    return p / p.sum(), centers


def charge_marginal(amps, x3):
    """Density along n3, the charge conjugate to theta3 (and to u at fixed theta2)."""
    h = x3[1] - x3[0]
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(x3), d=h))
    dens = 0.0
    for a in amps:
        ak = np.fft.fftshift(np.fft.fft(a, axis=0), axes=0)
        dens = dens + np.sum(np.abs(ak) ** 2, axis=(1, 2))
    return dens / dens.sum(), k


def overlap_of_densities(p, q):
    """Bhattacharyya overlap of two normalized marginals (1 = same, 0 = disjoint)."""
    return float(np.sum(np.sqrt(p * q)))


def analyse(physics, cut, label, umax=14.0, **bias):
    s = solve_states(physics, cut, **bias)
    W, G = s['W'], s['G']

    #Most-localized logical pair: eigenvectors of the 2x2 grid_phi block.
    #This is independent of the arbitrary eigenvector phases, unlike a fixed
    #(|0> +/- |1>)/sqrt2.  By construction <L|G|R> = 0 and <L|G|L>, <R|G|R>
    #are the two block eigenvalues (= +/-|G01| when parity zeroes the diagonal).
    B = 0.5 * (G[:2, :2] + G[:2, :2].conj().T)
    gvals, gvecs = np.linalg.eigh(B)
    R = W[:, :2] @ gvecs[:, 0]
    L = W[:, :2] @ gvecs[:, 1]
    G_LR = gvecs.conj().T @ B @ gvecs

    bins = np.linspace(-umax, umax, int(20 * umax) + 1)
    marg = {}
    for name, vec in (('0', W[:, 0]), ('1', W[:, 1]), ('L', L), ('R', R)):
        amps = s['amplitudes'](vec)
        pu, ucent = phase_marginal(amps, s['x2'], s['x3'], bins)
        pn, kvals = charge_marginal(amps, s['x3'])
        marg[name] = dict(pu=pu, pn=pn, d2=density_2d(amps))
    marg['_x2'], marg['_x3'] = s['x2'], s['x3']

    result = dict(
        label=label, physics=physics, cutoffs=cut, bias=bias,
        f01_MHz=float((s['evals'][1] - s['evals'][0]) * 1e3),
        abs_G01=float(abs(B[0, 1])),
        G_LR=dict(LL=float(np.real(G_LR[0, 0])), RR=float(np.real(G_LR[1, 1])),
                  abs_LR=float(abs(G_LR[0, 1]))),
        overlap_u_L_R=overlap_of_densities(marg['L']['pu'], marg['R']['pu']),
        overlap_u_0_1=overlap_of_densities(marg['0']['pu'], marg['1']['pu']),
    )
    return result, marg, ucent, kvals


def plot(results, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(24, 3.4 * n), squeeze=False)
    for row, (res, marg, u, k) in enumerate(results):
        ax = axes[row, 0]
        ax.plot(u, marg['0']['pu'], label='|0>')
        ax.plot(u, marg['1']['pu'], '--', label='|1>')
        ax.set_title('%s\nenergy eigenstates, f01=%.3g MHz, |<0|G|1>|=%.3f'
                     % (res['label'], res['f01_MHz'], res['abs_G01']), fontsize=9)
        ax.set_xlabel('u = theta3 + theta2/2 (rad)')
        for m in range(-2, 3):
            ax.axvline(m * np.pi, color='0.85', lw=0.6, zorder=0)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.plot(u, marg['L']['pu'], label='|L> (grid_phi block eigvec +)')
        ax.plot(u, marg['R']['pu'], label='|R> (grid_phi block eigvec -)')
        ax.set_title('combinations: u-overlap=%.3f, |<L|G|R>|=%.3f, '
                     '<L|G|L>=%.2f, <R|G|R>=%.2f'
                     % (res['overlap_u_L_R'], res['G_LR']['abs_LR'],
                        res['G_LR']['LL'], res['G_LR']['RR']), fontsize=9)
        ax.set_xlabel('u (rad)')
        for m in range(-2, 3):
            ax.axvline(m * np.pi, color='0.85', lw=0.6, zorder=0)
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        ax.plot(k, marg['0']['pn'], label='|0>')
        ax.plot(k, marg['1']['pn'], '--', label='|1>')
        ax.plot(k, marg['L']['pn'], ':', label='|L>')
        ax.set_xlim(-4, 4)
        ax.set_xlabel('n3 (conjugate charge)')
        ax.set_title('charge marginal', fontsize=9)
        ax.legend(fontsize=8)

        x2, x3 = marg['_x2'], marg['_x3']
        ext = [x2[0], x2[-1], x3[0], x3[-1]]
        for col, name in ((3, 'L'), (4, 'R')):
            ax = axes[row, col]
            ax.imshow(marg[name]['d2'], origin='lower', extent=ext,
                      aspect='auto', cmap='magma')
            #lines of constant u = theta3 + theta2/2 at multiples of pi
            for m in range(-9, 10):
                ax.plot(x2, m * np.pi - 0.5 * x2, color='c', lw=0.5,
                        alpha=0.6)
            ax.set_xlim(x2[0], x2[-1]); ax.set_ylim(x3[0], x3[-1])
            ax.set_xlabel('theta2'); ax.set_ylabel('theta3')
            ax.set_title('|%s|^2 in (theta2, theta3); cyan: u = m*pi' % name,
                         fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


#Paper (arXiv:2509.14656) Table S1 regimes, evaluated in the paper's S23 limit:
#no cross-KITE capacitance (eC -> infinity) and a stiff parasitic node
#(eP large, so mode 4 sits far above the qubit and BO elimination holds).
S23_LIMIT = dict(eC=1.0e6, eP=1.0e3)
_S1_COMMON = dict(EC=0.5, EJS=4.0, ECS=8.0, **S23_LIMIT)
PAPER = {
    'a': dict(_S1_COMMON, EJ=5.0, EL=1.0, ELK=1.0),
    'b': dict(_S1_COMMON, EJ=10.0, EL=1.0, ELK=1.0),
    'c': dict(_S1_COMMON, EJ=10.0, EL=0.5, ELK=0.5),
    'd': dict(_S1_COMMON, EJ=10.0, EL=0.2, ELK=0.2),
    #SI Note 11 four-mode fits to measured devices (eC = eps_K, eP = eps_P)
    'devA': dict(EJ=6.67, EC=0.77, EL=0.30, ELK=0.53, EJS=1.25, ECS=7.99,
                 eC=2.07, eP=3.54),
    'devB': dict(EJ=7.21, EC=0.77, EL=0.73, ELK=0.73, EJS=1.14, ECS=8.13,
                 eC=0.68, eP=1.00),
}
#grids sized to each envelope: smaller EL needs a wider window
CUTS = {
    'a': dict(CUT, L2=11.0, L3=14.0, N2=51, N3=51),
    'b': dict(CUT, L2=11.0, L3=14.0, N2=51, N3=51),
    'c': dict(CUT, L2=16.0, L3=20.0, N2=61, N3=61),
    'd': dict(CUT, L2=22.0, L3=26.0, N2=81, N3=81),
    'd_check': dict(CUT, L2=26.0, L3=30.0, N2=91, N3=91, N4=6),
    'devA': dict(CUT, L2=20.0, L3=24.0, N2=71, N3=71, N4=6),
    'devB': dict(CUT, L2=14.0, L3=18.0, N2=61, N3=61, N4=6),
}
LABELS = {
    'repo': 'ACTIVE: regime d + epsK=2.5 epsP=5.5',
    'legacy': 'legacy benchmark (regime a, eC=5.5 eP=10)',
    'a': 'paper regime a (S23 limit)', 'b': 'paper regime b (S23 limit)',
    'c': 'paper regime c (S23 limit)', 'd': 'paper regime d (S23 limit)',
    'd_check': 'regime d, larger grid + N4=6',
    'devA': 'SI11 fitted device A (4-mode)',
    'devB': 'SI11 fitted device B / ED Fig3 (4-mode)',
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+',
                   default=['repo', 'a', 'b', 'c', 'd'],
                   choices=sorted(LABELS),
                   help='Which parameter sets to run.')
    p.add_argument('--tag', default='',
                   help='Suffix for output file names.')
    args = p.parse_args(argv)

    cases = []
    for name in args.cases:
        if name == 'repo':
            cases.append((LABELS[name], BENCH, CUT))
        elif name == 'legacy':
            cases.append((LABELS[name], LEGACY_BENCH, LEGACY_CUT))
        else:
            key = 'd' if name == 'd_check' else name
            cases.append((LABELS[name], PAPER[key], CUTS[name]))

    results, summary = [], []
    for label, phys, cut in cases:
        umax = cut['L3'] + 0.5 * cut['L2']
        res, marg, u, k = analyse(phys, cut, label, umax=umax)
        results.append((res, marg, u, k))
        summary.append(res)
        print('%-40s f01=%10.4g MHz |<0|G|1>|=%.4f  u-overlap(L,R)=%.3f  '
              '|<L|G|R>|=%.4f  <L|G|L>=%+.3f <R|G|R>=%+.3f'
              % (label, res['f01_MHz'], res['abs_G01'], res['overlap_u_L_R'],
                 res['G_LR']['abs_LR'], res['G_LR']['LL'], res['G_LR']['RR']),
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = 'four_mode_logical_wavefunctions_diagnostic' + (
        '_' + args.tag if args.tag else '')
    fig_path = OUT / (stem + '.png')
    plot(results, fig_path)
    with (OUT / (stem + '.json')).open('w') as f:
        json.dump(summary, f, indent=2, default=float)
    print('Wrote %s' % fig_path.relative_to(ROOT))


if __name__ == '__main__':
    main()
