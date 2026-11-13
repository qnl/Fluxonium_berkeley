"""Check the inversion parity of the low four-mode Gridium states.

Question: do logical states 0 and 1 have opposite parity under the symmetry
that survives at the bias point?  If they do, the parity-odd ``grid_phi``
operator is allowed to connect them directly and no selection rule forbids a
single-photon logical transition.

Candidate symmetry
------------------
Full inversion P: (theta1, theta2, theta3, theta4) -> -(same), with all
charges also reversed.  In ``_assemble_nonlinear_sector`` every term is even
under P when ``ng = 0`` and ``theta_ext, phi_ext`` are each 0 or pi:

* kinetic and quadratic inductive terms are even;
* -EJ2 cos(theta1 + theta2 + theta_ext) maps to itself for theta_ext in {0, pi};
* -EJS cos(theta1 + theta2 + theta3 + phi_ext) maps to itself for phi_ext in {0, pi};
* Stage-2 couplings n_i n4 and x_i theta4 are even when theta4 -> -theta4.

``grid_phi = theta3 + theta2/2`` is P-odd, so <i|grid_phi|j> can be nonzero
only between states of opposite P.

On the numerical grids P is exact: the charge basis runs symmetrically from
-n1max to n1max, and the theta2/theta3 grids are symmetric ``linspace(-L, L,
N)``, so inversion is index reversal.  On the theta4 oscillator, P is (-1)^n.

The script rebuilds the Stage-1/Stage-2 problem with the same module
functions as ``Gridium4Mode``, checks it reproduces the model's energies and
``grid_phi``, and then reports <i|P|i> for the low states.  It also runs the
one-mode ``IdealGridium`` for comparison.

Reduced cutoffs; diagnostic only.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from Circuit_Objs import qchard_gridium_netlist as net
from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode
from Circuit_Objs.qchard_idealgridium import IdealGridium
from Simulations.Rabi_3Photon import gridium_params as gp


#circuit energies of the active point (paper regime d, eps_K=2.5, eps_p=5.5);
#asymmetry and bias are passed per run below
PHYSICS = gp.circuit_only(gp.ACTIVE_PHYSICS)
CUTOFFS = gp.cutoffs('fast', nkeep=120)
NLEV = 12


def _flip(n):
    """Index-reversal matrix: inversion on a symmetric grid/charge basis."""
    return sps.csr_matrix(np.fliplr(np.eye(n)))


def analyse(eps_J, eps_LK, ng=0.0, phi_ext=0.0, theta_ext=np.pi):
    c = CUTOFFS
    ECm, K, EJ1, EJ2 = net._coeffs(
        PHYSICS['EJ'], PHYSICS['EC'], PHYSICS['EL'], PHYSICS['ELK'],
        PHYSICS['EJS'], PHYSICS['ECS'], eps_J, eps_LK,
        eC=PHYSICS['eC'], eP=PHYSICS['eP'])

    H1, o = net._assemble_nonlinear_sector(
        ECm[:3, :3], K[:2, :2], EJ1, EJ2, PHYSICS['EJS'], ng,
        phi_ext, theta_ext, c['n1max'], c['N2'], c['L2'], c['N3'], c['L3'])

    #Stage-1 inversion operator, using the module's own kron ordering
    kron3 = o['kron3']
    P1 = kron3(_flip(2 * c['n1max'] + 1), _flip(c['N2']), _flip(c['N3']))
    comm1 = abs(H1 @ P1 - P1 @ H1).max()

    sigma = -(EJ1 + EJ2 + PHYSICS['EJS'] + 10.0)
    w1, V, _ = net._stage1_eigensolve(H1, k=c['nkeep'], sigma=sigma)

    #Stage-1 parity of each kept state
    P_sector = V.conj().T @ (P1 @ V)
    stage1_parity = np.real(np.diag(P_sector))
    offdiag_sector = np.abs(P_sector - np.diag(np.diag(P_sector))).max()

    #Stage 2, copied from Gridium4Mode._solve
    N4 = c['N4']
    A4, B4 = 4 * ECm[3, 3], 0.5 * K[2, 2]
    f4 = (A4 / (4 * B4)) ** 0.25
    g4 = 1.0 / (2 * f4)
    a = np.diag(np.sqrt(np.arange(1, N4)), 1)
    th4 = f4 * (a + a.T)
    n4 = 1j * g4 * (a.T - a)
    w4 = 2 * np.sqrt(A4 * B4) * (np.arange(N4) + 0.5)

    def proj(op):
        return V.conj().T @ (op @ V)

    n1_p = proj(kron3(sps.diags(o['nvec'] + ng), o['I2'], o['I3']))
    n2_p = proj(kron3(o['I1'], o['n2'], o['I3']))
    n3_p = proj(kron3(o['I1'], o['I2'], o['n3']))
    x2_p = proj(kron3(o['I1'], sps.diags(o['x2']), o['I3']))
    x3_p = proj(kron3(o['I1'], o['I2'], sps.diags(o['x3'])))

    nk = c['nkeep']
    Ik, I4 = np.eye(nk), np.eye(N4)
    Hc = np.kron(np.diag(w1), I4) + np.kron(Ik, np.diag(w4))
    Hc = Hc + 8 * ECm[0, 3] * np.kron(n1_p, n4)
    Hc = Hc + 8 * ECm[1, 3] * np.kron(n2_p, n4)
    Hc = Hc + 8 * ECm[2, 3] * np.kron(n3_p, n4)
    Hc = Hc + K[0, 2] * np.kron(x2_p, th4)
    Hc = Hc + K[1, 2] * np.kron(x3_p, th4)
    Hc = 0.5 * (Hc + Hc.conj().T)

    #full parity in the Stage-2 space
    P4 = np.diag((-1.0) ** np.arange(N4))
    P2 = np.kron(P_sector, P4)
    comm2 = np.abs(Hc @ P2 - P2 @ Hc).max()

    evals, W = spla.eigh(Hc, subset_by_index=(0, NLEV - 1))
    parity = np.real(np.einsum('ai,ab,bi->i', W.conj(), P2, W))

    G_sector = 0.5 * x2_p + x3_p
    G = W.conj().T @ np.kron(G_sector, I4) @ W
    Pgrid = P2 @ np.kron(G_sector, I4) @ P2
    grid_phi_is_odd = np.abs(Pgrid + np.kron(G_sector, I4)).max()

    return dict(
        eps_J=eps_J, eps_LK=eps_LK, phi_ext=phi_ext, theta_ext=theta_ext,
        comm_stage1=float(comm1), comm_stage2=float(comm2),
        stage1_parity_offdiag=float(offdiag_sector),
        stage1_parity_first10=stage1_parity[:10],
        energies_GHz=evals, parity=parity, grid_phi=np.abs(G),
        grid_phi_P_odd_residual=float(grid_phi_is_odd),
    )


def reference_check(result):
    """Confirm the rebuild matches Gridium4Mode itself."""
    model = Gridium4Mode(**PHYSICS, eps_J=result['eps_J'],
                         eps_LK=result['eps_LK'], ng=0.0,
                         phi_ext=result['phi_ext'],
                         theta_ext=result['theta_ext'], **CUTOFFS, nlev=NLEV)
    e_ref = np.asarray(model.levels())
    g_ref = np.abs(np.asarray(model.grid_phi().full()))
    return (float(np.abs(e_ref - result['energies_GHz']).max()),
            float(np.abs(g_ref - result['grid_phi']).max()))


def report(result, label):
    print('=' * 78)
    print(label)
    print('  eps_J=%g eps_LK=%g phi_ext=%.4g theta_ext=%.4g'
          % (result['eps_J'], result['eps_LK'], result['phi_ext'],
             result['theta_ext']))
    print('  max|[H1,P1]|              = %.3e' % result['comm_stage1'])
    print('  max|[H2,P2]|              = %.3e' % result['comm_stage2'])
    print('  Stage-1 P off-diagonal    = %.3e  (0 => kept states are P eigenstates)'
          % result['stage1_parity_offdiag'])
    print('  max|P G P + G|            = %.3e  (0 => grid_phi is P-odd)'
          % result['grid_phi_P_odd_residual'])
    e0 = result['energies_GHz'][0]
    print('  state  f_from_0 (MHz)   <P>     |<0|G|i>|   |<1|G|i>|')
    for i, (e, p) in enumerate(zip(result['energies_GHz'], result['parity'])):
        print('  %3d  %14.4f  %+7.4f  %10.5f  %10.5f'
              % (i, (e - e0) * 1e3, p, result['grid_phi'][0, i],
                 result['grid_phi'][1, i]))


def one_mode_comparison():
    """The one-mode IdealGridium used for the validated X90/X180 gates."""
    print('=' * 78)
    print('One-mode IdealGridium (E_L=0.5, E_C=0.5, E_s=4, E_2J=12, phi_ext=pi)')
    q = IdealGridium(E_L=0.5, E_C=0.5, E_s=4, E_2J=12, ng=0,
                     phi_ext=np.pi, nlev=8, nlev_lc=230, units='GHz')
    e = np.asarray(q.levels())
    phi = np.abs(np.asarray(q.phi().full()))
    print('  state  f_from_0 (MHz)   |<0|phi|i>|  |<1|phi|i>|')
    for i in range(len(e)):
        print('  %3d  %14.4f  %11.5f  %11.5f'
              % (i, (e[i] - e[0]) * 1e3, phi[0, i], phi[1, i]))


def main():
    for eps_J, eps_LK in ((0.0, 0.0), (0.10, 0.05)):
        result = analyse(eps_J, eps_LK)
        report(result, 'Four-mode Gridium4Mode, regime d + eps_K/eps_p')
        de, dg = reference_check(result)
        print('  rebuild vs Gridium4Mode: max dE = %.2e GHz, max d|G| = %.2e'
              % (de, dg))

    #the other protected-looking bias the one-mode model uses
    result = analyse(0.0, 0.0, phi_ext=np.pi, theta_ext=np.pi)
    report(result, 'Four-mode, symmetric, phi_ext = pi (one-mode bias)')

    one_mode_comparison()


if __name__ == '__main__':
    main()
