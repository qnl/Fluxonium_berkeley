"""Check whether summing over theta1 hides a grid in the four-mode states.

``four_mode_grid_wavefunctions.py`` sums the density over the compact theta1
coordinate before projecting onto ``u = theta3 + theta2/2``.  In the paper's
reduced model (SI Note 4) the grid term is ``cos[2(phi2 - phi1)]``, so a comb
along u could shift with theta1 and wash out in the sum.  This script Fourier
transforms the theta1 charge index into a theta1 phase and plots the joint
density in (theta1, u) for |L> and |R>.  A hidden grid would show up as
tilted or offset combs.

Diagnostic only; uses the same solver as the wavefunction script.
"""

from __future__ import annotations

import numpy as np

from Simulations.Rabi_3Photon.four_mode_grid_wavefunctions import (
    OUT, PAPER, CUTS, BENCH, CUT, LEGACY_BENCH, LEGACY_CUT, ROOT,
    solve_states)


def joint_theta1_u(amps, x2, x3, nvec, n_theta1=64, ubins=None, smooth=0.25):
    theta1 = np.linspace(-np.pi, np.pi, n_theta1, endpoint=False)
    phase = np.exp(1j * np.outer(nvec, theta1))           #(N1, n_theta1)
    U = (x3[:, None] + 0.5 * x2[None, :]).ravel()
    centers = 0.5 * (ubins[1:] + ubins[:-1])
    kern = np.exp(-0.5 * ((centers[:, None] - U[None, :]) / smooth) ** 2)
    joint = np.zeros((n_theta1, len(centers)))
    for a in amps:                                          #a: (N3, N2, N1)
        psi = a @ phase                                     #(N3, N2, n_theta1)
        dens = np.abs(psi) ** 2
        joint += (kern @ dens.reshape(-1, n_theta1)).T
    return theta1, centers, joint / joint.sum()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cases = [('legacy benchmark', LEGACY_BENCH, LEGACY_CUT),
             ('ACTIVE regime d+caps', BENCH, CUT)]
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4 * len(cases)),
                             squeeze=False)
    for row, (label, phys, cut) in enumerate(cases):
        s = solve_states(phys, cut)
        W, G = s['W'], s['G']
        B = 0.5 * (G[:2, :2] + G[:2, :2].conj().T)
        _, vecs = np.linalg.eigh(B)
        nvec = np.arange(-cut['n1max'], cut['n1max'] + 1).astype(float)
        umax = cut['L3'] + 0.5 * cut['L2']
        ubins = np.linspace(-umax, umax, int(20 * umax) + 1)
        for col, (name, v) in enumerate((('L', vecs[:, 1]), ('R', vecs[:, 0]))):
            amps = s['amplitudes'](W[:, :2] @ v)
            t1, uc, J = joint_theta1_u(amps, s['x2'], s['x3'], nvec,
                                       ubins=ubins)
            ax = axes[row, col]
            ax.imshow(J.T, origin='lower', aspect='auto', cmap='magma',
                      extent=[t1[0], t1[-1], uc[0], uc[-1]])
            for m in range(-12, 13):
                ax.axhline(m * np.pi, color='c', lw=0.4, alpha=0.5)
            ax.set_ylim(-12, 12)
            ax.set_xlabel('theta1 (rad)'); ax.set_ylabel('u (rad)')
            ax.set_title('%s: |%s|^2 in (theta1, u); cyan: u = m*pi'
                         % (label, name), fontsize=9)
            #how much theta1 structure is there at all
            p1 = J.sum(axis=1)
            print('%-16s %s  theta1 marginal max/min = %.3g'
                  % (label, name, p1.max() / max(p1.min(), 1e-300)),
                  flush=True)
    fig.tight_layout()
    path = OUT / 'four_mode_theta1_conditional_diagnostic.png'
    fig.savefig(path, dpi=140)
    print('Wrote %s' % path.relative_to(ROOT))


if __name__ == '__main__':
    main()
