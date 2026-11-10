"""FourModeAsymGridium: exact four-mode gridium circuit model with KITE asymmetry.

.. deprecated::
    This hand reduction carries a pi/2 external-flux bookkeeping error in its
    compact-frame construction (the PHI_EXT_PROTECTION_OFFSET below is a band-aid: it
    aligns the protected theta_ext=pi column with the paper but leaves the theta_ext=0
    column off by pi/2; no global offset fixes both). Verified against scqubits.Circuit
    built from the paper authors' netlist. Use the corrected, netlist-derived
    ``Circuit_Objs.qchard_gridium_netlist.Gridium4Mode`` instead.

Implements the exact four-mode circuit Hamiltonian of the gridium device:

    H = 2 E_C n_Sigma^2 + 2 E_C_delta n_Delta^2 + 4 E_CS n_S^2 + 4 eps_p n_phi^2          (4)
        - 2 E_J cos(phi_Sigma) cos(phi_Delta) - 2 eps_J E_J sin(phi_Sigma) sin(phi_Delta)
        - E_JS cos(phi_S)                                                                 (5)
        + E_LK [ (phi_Sigma - phi)^2 + (phi_Delta - theta_ext/2)^2 ]
        - 2 eps_LK E_LK (phi_Sigma - phi)(phi_Delta - theta_ext/2)                        (6)
        + E_L (phi_S - phi + phi_ext)^2                                                   (7)

Flux gauge: Eqs. (5)/(7) place the external flux phi_ext in the superinductor. The code uses the
gauge-equivalent form that moves phi_ext onto the QPS (phase slip), -E_JS cos(phi_S - phi_ext) with
E_L (phi_S - phi)^2, i.e. the paper's Eq. S23/S26 form. This is manifestly periodic in phi_ext
(the compact island carries the flux), whereas keeping phi_ext in the superinductor and expanding
the extended node harmonically introduces a linear grid tilt that drifts non-periodically.

Four modes:

1. phi_Sigma: common KITE phase
2. phi_Delta: rhombus circulating mode, confined at theta_ext/2
3. phi_S: QPS junction phase
4. phi: superinductor/KITE node with parasitic capacitance.

The KITE asymmetries: eps_J (junctions) and eps_LK (inductors); at the protection bias they lift the parity
doublet degeneracies and brighten the intra-doublet rungs.

Coordinate transformation (paper's Eqs. S24-S26):

    u1 = phi_S: compact island phase -> charge basis, offset charge ng
    u2 = phi_S - phi_Sigma: extended grid mode -> real-space grid / DVR basis
    u3 = phi_Delta: Delta mode -> harmonic-oscillator basis
    u4 = phi_S - phi: node mode -> harmonic-oscillator basis

Conjugate-charge transformation:
    n_Sigma = -m2,
    n_Delta = m3,
    n_S = m1+m2+m4,
    n_phi = -m4.

The Hamiltonian splits exactly as

    H = H_S1(u1,u2,u3) + H_node(u4) + H_coupling,

    H_coupling = -2 E_LK u2*u4' - 2 eps_LK E_LK (u3 - theta_ext/2)*u4' + 8 E_CS (m1+m2+ng)*m4: coupling terms between (u1, u2, u3) and node operator (u4', m4)

    H_S1: contains only the first three coordinates

    H_node(u4): contains only the fourth mode after shifting its equilibrium position

where u4' = u4 - u4_min and H_node is *exactly* harmonic.

How to diagonalize the Hamiltonian efficiently:

    1. Sparse Lanczos diagonalization of the (island x grid x Delta) sector:

        - The grid mode uses a real-space (DVR) basis so cos/sin(u2) are diagonal and the sector stays sparse;
        - the kinetic term uses 4th-order finite differences.
        - builds the three-mode Hamiltonian and finds only its lowest eigenstates using a sparse eigensolver like Lanczos

    2. Exact coupling of the kept sector eigenstates to the node-mode oscillator; low-energy
       structured Hermitian solve of the coupled Hamiltonian

        - after stage 1, keeps some number of low-energy sector eigenstates
        - then forms product states with n4

Truncation knobs:
    - n_charge,
    - n_grid_pts/grid_range,
    - nlev_delta,
    - nlev_node,
    - nkeep_s1.

Convergence guidance:
    1. increase each cutoff independently,
    2. require stable splittings and matrix elements for the observables being used.

Drive/readout operators (flux derivatives of the Hamiltonian), recomputed at every
(eps_J, eps_LK) point:

    D_theta = dH/d(theta_ext) = -E_LK (phi_Delta - theta_ext/2) + eps_LK E_LK (phi_Sigma - phi)
    D_phi   = dH/d(phi_ext)   =  2 E_L (phi_S - phi + phi_ext)
    phi_2   = phi_S - phi_Sigma = u2
    n_1     = n_S + n_Sigma   =  m1 + m4                       (capacitive drive)

"""

__all__ = ['FourModeAsymGridium', 'device_FourModeAsymGridium_params',
           'std_FourModeAsymGridium_sim_params']

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import qutip as qt
import dill

# SI Note 11 fit of the measured protected device (h*GHz). Model-bound: use with this class only.
device_FourModeAsymGridium_params = {
    'E_J': 7.21,        # KITE junctions (mean)
    'E_C': 0.77,        # KITE junction charging
    'E_L': 0.73,        # superinductor
    'E_LK': 0.73,       # KITE inductors (mean)
    'E_JS': 1.14,       # QPS junction
    'E_CS': 8.13,       # QPS charging
    'E_C_delta': 0.492, # Delta-mode charging, e^2/2(C_J + C_K/2), derived via paper Eq. S30
    'eps_p': 1.0,       # parasitic node capacitance energy
}

# External-flux reference. phi_ext = 0 is the protection bias, matching the paper's Fig. S5 /
# Eq. 2 convention (the QPS charge dispersion is flattest and the (0,1) doublet degenerate here).
# In the raw flux variable the protected point sits a quarter flux quantum away:
# the grid confinement then lands on an extremum of the *emergent* cos(2 phi) quartet lattice
# (which the second-order KITE process produces with a fixed sign). So the flux that actually
# enters the Hamiltonian is phi_ext + pi/2. Verified: without this offset the (0,1) splitting is
# ~1.2 GHz at "phi_ext=0" (an anti-sweet-spot) and dips to ~20 MHz at pi/2; the offset is rigid
# (regimes a and d both minimize at the same pi/2) and matches the effective 1-mode model (Eq. 2).
PHI_EXT_PROTECTION_OFFSET = 0.5 * np.pi

# Standard operating/simulation conditions. Protection bias is (phi_ext, theta_ext) = (0, pi).
std_FourModeAsymGridium_sim_params = {
    'eps_J': 0.0,        # junction asymmetry, E_J1,2 = E_J (1 -/+ eps_J); sweep 0-0.15
    'eps_LK': 0.0,       # inductor asymmetry, E_LK1,2 = E_LK (1 -/+ eps_LK); sweep 0-0.15
    'ng': 0.0,           # island offset charge
    'phi_ext': 0.0,
    'theta_ext': np.pi,
    'nlev': 12,

    #those defaults are not guaranteed to give fully converged result
    #for final verification, increase nlev_delta, nlev_node, and nkeep_s1

    'n_charge': 8,       # island charge cutoff: m1 in [-n_charge, n_charge]
    'n_grid_pts': 181,   # DVR points for the grid mode
    'grid_range': 12.0,  # DVR half-range (rad)
    'nlev_delta': 10,    # HO levels for the Delta mode
    'nlev_node': 12,     # HO levels for the node mode
    'nkeep_s1': 320,     # eigenstates kept from the (island x grid x Delta) sector
    'units': 'GHz',
}


class FourModeAsymGridium(object):

    name = 'Four-Mode Asymmetric Gridium'

    _PHYS = (
             'E_J',
             'E_C',
             'E_L',
             'E_LK',
             'E_JS',
             'E_CS',
             'E_C_delta',
             'eps_p',
             'eps_J',
             'eps_LK',
             'ng',
             'phi_ext',
             'theta_ext',
             'n_charge',
             'n_grid_pts',
             'grid_range',
             'nlev_delta',
             'nlev_node',
             'nkeep_s1'
            )

    def __init__(
                 #physical parameters
                 self,
                 E_J,
                 E_C,
                 E_L,
                 E_LK,
                 E_JS,
                 E_CS,
                 E_C_delta,
                 eps_p,
                 eps_J=0.0,
                 eps_LK=0.0,
                 ng=0.0,
                 nlev=12,
                 phi_ext=0.0,
                 theta_ext=np.pi,

                #numerical cutoffs
                 n_charge=8,
                 n_grid_pts=181,
                 grid_range=12.0,
                 nlev_delta=10,
                 nlev_node=12,
                 nkeep_s1=320,
                 units='GHz',
                 verbose=False
                ):

        nlev = self._positive_int('nlev', nlev)
        n_charge = self._positive_int('n_charge', n_charge)
        n_grid_pts = self._positive_int('n_grid_pts', n_grid_pts)
        nlev_delta = self._positive_int('nlev_delta', nlev_delta)
        nlev_node = self._positive_int('nlev_node', nlev_node)
        nkeep_s1 = self._positive_int('nkeep_s1', nkeep_s1)
        grid_range = float(grid_range)

        if n_grid_pts < 5:
            raise ValueError('n_grid_pts must be at least 5 for the fourth-order finite-difference stencil.')
        if grid_range <= 0.0:
            raise ValueError('grid_range must be positive.')

        self.E_J = E_J
        self.E_C = E_C
        self.E_L = E_L
        self.E_LK = E_LK
        self.E_JS = E_JS
        self.E_CS = E_CS
        self.E_C_delta = E_C_delta
        self.eps_p = eps_p
        self.eps_J = eps_J
        self.eps_LK = eps_LK
        self.ng = ng
        self.phi_ext = phi_ext
        # Internal flux entering Eq. (7); see PHI_EXT_PROTECTION_OFFSET. phi_ext = 0 -> protection.
        self._phi_ext_eff = phi_ext + PHI_EXT_PROTECTION_OFFSET
        self.theta_ext = theta_ext
        self.nlev = nlev
        self.n_charge = n_charge
        self.n_grid_pts = n_grid_pts
        self.grid_range = grid_range
        self.nlev_delta = nlev_delta
        self.nlev_node = nlev_node
        self.nkeep_s1 = nkeep_s1


        self.units = units
        self.verbose = verbose
        self.type = 'qubit'

    @staticmethod
    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('{} must be a positive integer.'.format(name))
        return int(value)

    @staticmethod
    def _require_finite(name, value):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('{} contains NaN/Inf entries.'.format(name))
        return value

    #updates the cahed results if the attribute affects the hamiltonian basis
    #prevents the object from returning stale results after a parameter change
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in self._PHYS:
            object.__setattr__(self, '_eigvals', None)
            object.__setattr__(self, '_ops', None)

    def __str__(self):
        s = ('A four-mode asymmetric-KITE gridium with (h*{u}) E_J={o.E_J}, E_C={o.E_C}, '
             'E_L={o.E_L}, E_LK={o.E_LK}, E_JS={o.E_JS}, E_CS={o.E_CS}, '
             'E_C_delta={o.E_C_delta}, eps_p={o.eps_p}; asymmetries eps_J={o.eps_J}, '
             'eps_LK={o.eps_LK}; ng={o.ng}; bias (phi_ext, theta_ext)=({o.phi_ext}, '
             '{th:.4f}).').format(u=self.units, o=self, th=self.theta_ext)
        return s

    #save the object so the file name identifies the configureation
    def _save_str(self):
        s = ('FourModeAsymEJ{}EC{}EL{}ELK{}EJS{}ECS{}ECD{}ep{}eJ{}eLK{}ng{}px{:.3f}tx{:.3f}'
             'nc{}np{}gr{}nd{}nn{}nk{}').format(
            self.E_J, self.E_C, self.E_L, self.E_LK, self.E_JS, self.E_CS,
            self.E_C_delta, self.eps_p, self.eps_J, self.eps_LK, self.ng,
            self.phi_ext, self.theta_ext, self.n_charge, self.n_grid_pts,
            self.grid_range, self.nlev_delta, self.nlev_node, self.nkeep_s1)
        return s

    def _scale_E_params(self, scaling):
        self.E_J = self.E_J * scaling
        self.E_C = self.E_C * scaling
        self.E_L = self.E_L * scaling
        self.E_LK = self.E_LK * scaling
        self.E_JS = self.E_JS * scaling
        self.E_CS = self.E_CS * scaling
        self.E_C_delta = self.E_C_delta * scaling
        self.eps_p = self.eps_p * scaling
        return


    #basic operators for each of the four modes
    #u1 (compact node): charge basis
    #u2: real-space grid
    #u3 & u4: harmonic oscillator bases
    def _mode_ops(self):
        """Single-mode operators in the compact frame (u1, u2, u3, u4)."""
        ops = {}

        # u1 = phi_S: island, charge basis. e^{+i u1} raises the island charge by one.
        nc = self.n_charge
        N1 = 2 * nc + 1
        S = sps.diags(np.ones(N1 - 1), -1)
        ops['m1'] = sps.diags(np.arange(-nc, nc + 1).astype(float))
        ops['cos1'] = 0.5 * (S + S.T)
        ops['sin1'] = -0.5j * (S - S.T)
        ops['I1'] = sps.identity(N1)

        # u2 = phi_S - phi_Sigma: grid mode, real-space (DVR) grid. cos/sin/x diagonal;
        # momentum via 4th-order central finite differences (Dirichlet boundaries).
        Np_ = self.n_grid_pts
        L = self.grid_range
        x = np.linspace(-L, L, Np_)
        h = x[1] - x[0]
        c1 = 1.0 / (12 * h)
        c2 = 1.0 / (12 * h * h)
        D1 = sps.diags([c1 * np.ones(Np_ - 2), -8 * c1 * np.ones(Np_ - 1),
                        8 * c1 * np.ones(Np_ - 1), -c1 * np.ones(Np_ - 2)], [-2, -1, 1, 2])
        ops['m2'] = -1j * D1
        ops['m2sq'] = sps.diags([c2 * np.ones(Np_ - 2), -16 * c2 * np.ones(Np_ - 1),
                                 30 * c2 * np.ones(Np_), -16 * c2 * np.ones(Np_ - 1),
                                 c2 * np.ones(Np_ - 2)], [-2, -1, 0, 1, 2])
        ops['xd'] = sps.diags(x)
        ops['x2d'] = sps.diags(x * x)
        ops['cxd'] = sps.diags(np.cos(x))
        ops['sxd'] = sps.diags(np.sin(x))
        ops['Ig'] = sps.identity(Np_)

        # u3 = phi_Delta: HO basis for v3 = u3 - theta_ext/2 (confined at the frustration
        # point). cos/sin(u3) built exactly by eigendecomposition of v3.
        N3 = self.nlev_delta
        A3 = 2 * self.E_C_delta          # coefficient of m3^2
        B3 = self.E_LK                   # coefficient of v3^2
        f3 = (A3 / (4 * B3)) ** 0.25
        g3 = 1.0 / (2 * f3)
        a3 = np.diag(np.sqrt(np.arange(1, N3)), 1)
        v3 = f3 * (a3 + a3.T)
        wv, Q = np.linalg.eigh(v3)
        half = 0.5 * self.theta_ext
        ops['v3'] = v3
        ops['m3'] = 1j * g3 * (a3.T - a3)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            cos_u3 = (Q * np.cos(half + wv)) @ Q.T
            sin_u3 = (Q * np.sin(half + wv)) @ Q.T
        ops['cos_u3'] = self._require_finite('cos_u3 operator', cos_u3)
        ops['sin_u3'] = self._require_finite('sin_u3 operator', sin_u3)
        ops['I3'] = np.eye(N3)

        # u4 = phi_S - phi: node mode. Exactly harmonic: A4 m4^2 + B4 v4^2 with
        # v4 = u4 - u4_min; the linear terms cancel by the choice of u4_min.
        N4 = self.nlev_node
        A4 = 4 * (self.eps_p + self.E_CS)  # 4 eps_p + the m4^2 part of 4 E_CS (m1+m2+m4+ng)^2
        B4 = self.E_LK + self.E_L
        f4 = (A4 / (4 * B4)) ** 0.25
        g4 = 1.0 / (2 * f4)
        a4 = np.diag(np.sqrt(np.arange(1, N4)), 1)
        # Periodic (S23) gauge: the external flux is carried by the phase-slip cosine
        # -E_JS cos(phi_S - phi_ext) (assembled in _diagonalize), NOT the superinductor. The node
        # equilibrium is therefore not shifted by flux (u40 = 0). This removes the linear grid tilt
        # that made the old inductor-flux gauge drift non-periodically across phi_ext; the spectrum
        # is now exactly periodic (matching the paper's Eq. S23/S26 and s26_reference).
        u40 = 0.0
        ops['v4'] = f4 * (a4 + a4.T)
        ops['m4'] = 1j * g4 * (a4.T - a4)
        ops['u40'] = u40
        ops['w4'] = 2 * np.sqrt(A4 * B4) * (np.arange(N4) + 0.5)
        return ops


    # hierarchical diagonalization
    #1. builds the single-mode operators using _mode_ops();
    #2. constructs the first three-mode Hamiltonian;
    #3. diagonalizes it and keeps the lowest nkeep_s1 states;
    #4. couples those states to the fourth node mode;
    #5. diagonalizes the final reduced four-mode Hamiltonian;
    #6. stores the resulting energies and projected operators.

    def _diagonalize(self):
        """Two-stage exact diagonalization; caches eigenvalues and projected operators."""
        if self._eigvals is not None:
            return
        import time
        t0 = time.time()
        o = self._mode_ops()
        EJ, EC, EL = self.E_J, self.E_C, self.E_L
        ELK, EJS, ECS = self.E_LK, self.E_JS, self.E_CS
        ECD, eJ, eLK, ng = self.E_C_delta, self.eps_J, self.eps_LK, self.ng
        u40 = o['u40']
        phi_e = self._phi_ext_eff   # external flux; enters the phase-slip cosine (periodic gauge)

        def K(A, B, C):
            return sps.kron(sps.kron(A, B, format='csr'), sps.csr_matrix(C), format='csr')

        # Stage 1: (island x grid x Delta) sector -- every term except the u4 mode
        # diagonalizes the three-mode hamiltonian
        # keeps the lowest nkeep_s1 eigenstates

        ECS_scr = ECS * self.eps_p / (ECS + self.eps_p)
        ELK_scr = ELK * EL / (ELK + EL)
        pg = (K(o['m1'], o['Ig'], o['I3']) + K(o['I1'], o['m2'], o['I3'])
              + ng * K(o['I1'], o['Ig'], o['I3']))
        pg2 = (pg @ pg)
        x2full = K(o['I1'], o['x2d'], o['I3'])
        H1 = (4 * ECS_scr * pg2
              + 2 * EC * K(o['I1'], o['m2sq'], o['I3'])
              + 2 * ECD * K(o['I1'], o['Ig'], o['m3'] @ o['m3'])
              - EJS * (np.cos(phi_e) * K(o['cos1'], o['Ig'], o['I3'])
                       + np.sin(phi_e) * K(o['sin1'], o['Ig'], o['I3']))
              + ELK_scr * x2full
              + ELK * K(o['I1'], o['Ig'], o['v3'] @ o['v3'])
              - 2 * EJ * (K(o['cos1'], o['cxd'], o['cos_u3'])
                          + K(o['sin1'], o['sxd'], o['cos_u3'])))
        if eJ != 0.0:
            # junction-asymmetry term: sin(u1-u2) sin(u3) = (sin u1 cos u2 - cos u1 sin u2) sin u3
            H1 = H1 - 2 * eJ * EJ * (K(o['sin1'], o['cxd'], o['sin_u3'])
                                     - K(o['cos1'], o['sxd'], o['sin_u3']))
        if eLK != 0.0:
            # +2 eps_LK E_LK u2 (u3 - theta/2): the u2 part of the inductor-asymmetry cross term
            H1 = H1 + 2 * eLK * ELK * K(o['I1'], o['xd'], o['v3'])
        if u40 != 0.0:
            # off-protection bias: linear terms from expanding around the node minimum
            H1 = H1 - 2 * ELK * u40 * K(o['I1'], o['xd'], o['I3'])
            if eLK != 0.0:
                H1 = H1 - 2 * eLK * ELK * u40 * K(o['I1'], o['Ig'], o['v3'])

        H1 = (0.5 * (H1 + H1.conj().T)).tocsr()
        if self.verbose:
            print('S1 dim = {}, nnz = {:.2e}, build {:.1f}s'.format(
                H1.shape[0], H1.nnz, time.time() - t0))

        if self.nkeep_s1 >= H1.shape[0]:
            # Small sectors (validation runs): dense diagonalization keeps the full basis,
            # making the hierarchy an exact unitary rearrangement of the four-mode model.
            w1, V1 = np.linalg.eigh(H1.toarray())
            nkeep = H1.shape[0]
        else:
            nkeep = self.nkeep_s1
            # Shift-invert about a point safely below the spectrum (potential >= -2E_J - E_JS)
            # converges the lowest states orders of magnitude faster than plain 'SA' here,
            # because the FD kinetic edge makes the raw spectral range enormous.
            sigma = -(2 * EJ + EJS + 10.0)
            try:
                w1, V1 = spsl.eigsh(H1.tocsc(), k=nkeep, sigma=sigma, which='LM')
            except (MemoryError, RuntimeError):
                w1, V1 = spsl.eigsh(H1, k=nkeep, which='SA')
            order = np.argsort(w1)
            w1, V1 = w1[order], V1[:, order]
        if self.verbose:
            print('S1 eigsh done at {:.1f}s'.format(time.time() - t0))

        def proj(Ofull):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                projected = V1.conj().T @ (Ofull @ V1)
            return self._require_finite('Projected S1 operator', projected)

        u2_p = proj(K(o['I1'], o['xd'], o['I3']))
        v3_p = proj(K(o['I1'], o['Ig'], o['v3']))
        pg_p = proj(pg)
        pg2_p = proj(pg2)
        x2_p = proj(x2full)
        m1_p = proj(K(o['m1'], o['Ig'], o['I3']))
        # Flux-drive operator dH/dphi_ext = -E_JS sin(phi_S - phi_e) (periodic gauge): flux now
        # sits in the phase-slip cosine, so the drive is an island operator, sin(u1 - phi_e) =
        # cos(phi_e) sin(u1) - sin(phi_e) cos(u1).
        dphi_island = np.cos(phi_e) * o['sin1'] - np.sin(phi_e) * o['cos1']
        dphi_p = proj(K(dphi_island, o['Ig'], o['I3']))

        # Stage 2: exact coupling to the (exactly harmonic) node mode
        # Restores the (bare - screened) quadratic residuals from the stage-1 regrouping
        # plus the genuine sector couplings; together with stage 1 this sums exactly to the full Hamiltonian.

        N4 = self.nlev_node
        I4 = np.eye(N4)
        Ik = np.eye(nkeep)
        dim_c = nkeep * N4
        nstore = min(self.nlev, dim_c)

        def herm(A):
            return np.asarray(0.5 * (A + A.conj().T), dtype=np.complex128)

        u2_p = herm(u2_p)
        v3_p = herm(v3_p)
        pg_p = herm(pg_p)
        pg2_p = herm(pg2_p)
        x2_p = herm(x2_p)
        m1_p = herm(m1_p)
        dphi_p = herm(dphi_p)
        v4 = herm(o['v4'])
        m4 = herm(o['m4'])
        w1 = np.asarray(np.real(w1), dtype=float)
        w4 = np.asarray(np.real(o['w4']), dtype=float)

        H_terms = [
            (4 * (ECS - ECS_scr), pg2_p, I4),
            (ELK - ELK_scr, x2_p, I4),
            (-2 * ELK, u2_p, v4),
            (8 * ECS, pg_p, m4),
        ]
        if eLK != 0.0:
            H_terms.append((-2 * eLK * ELK, v3_p, v4))

        def apply_terms(vec, terms, diag_left=None, diag_right=None, scalar=0.0):
            X = np.asarray(vec).reshape((nkeep, N4))
            Y = np.zeros_like(X, dtype=np.complex128)
            if diag_left is not None:
                Y += diag_left[:, None] * X
            if diag_right is not None:
                Y += X * diag_right[None, :]
            if scalar != 0.0:
                Y += scalar * X
            for coeff, A, B in terms:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    contribution = A @ X @ B.T
                Y += coeff * contribution
            return self._require_finite('Structured final-stage matvec', Y).ravel()

        def apply_terms_mat(mat, terms, diag_left=None, diag_right=None, scalar=0.0):
            mat = np.asarray(mat)
            if mat.ndim == 1:
                return apply_terms(mat, terms, diag_left, diag_right, scalar)
            out = np.empty_like(mat, dtype=np.complex128)
            for col in range(mat.shape[1]):
                out[:, col] = apply_terms(mat[:, col], terms, diag_left, diag_right, scalar)
            return out

        # The coupled basis can be large at production cutoffs. For small validation sectors,
        # materialize the dense matrix so hierarchy-vs-brute-force tests remain maximally
        # direct. Otherwise use the same Kronecker terms as a structured LinearOperator and
        # compute only the low-energy eigenpairs required by the public qubit API.

        dense_coupled_dim = 2500

        if dim_c <= dense_coupled_dim:
            Hc = (np.kron(np.diag(w1), I4) + np.kron(Ik, np.diag(w4)))
            for coeff, A, B in H_terms:
                Hc = Hc + coeff * np.kron(A, B)
            Hc = np.asarray(0.5 * (Hc + Hc.conj().T), dtype=np.complex128)
            if not np.all(np.isfinite(Hc)):
                raise FloatingPointError('Final coupled Hamiltonian contains NaN/Inf entries.')
            evals, W = spla.eigh(Hc, subset_by_index=[0, nstore - 1],
                                 driver='evr', check_finite=False)
        else:
            Hlin = spsl.LinearOperator(
                shape=(dim_c, dim_c),
                matvec=lambda x: apply_terms(x, H_terms, w1, w4),
                matmat=lambda X: apply_terms_mat(X, H_terms, w1, w4),
                dtype=np.complex128)
            evals, W = spsl.eigsh(Hlin, k=nstore, which='SA',
                                  tol=1e-10, maxiter=max(1000, 20 * dim_c))
            order = np.argsort(evals)
            evals, W = np.real(evals[order]), W[:, order]

        def project_low_energy(terms, diag_left=None, diag_right=None, scalar=0.0):

            OW = apply_terms_mat(W, terms, diag_left, diag_right, scalar)
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                M = W.conj().T @ OW

            return 0.5 * self._require_finite('Projected final-stage operator',
                                              M + M.conj().T)

        Dth_terms = [(-ELK, v3_p, I4)]

        if eLK != 0.0:
            Dth_terms.append((-eLK * ELK, u2_p, I4))
            Dth_terms.append((eLK * ELK, Ik, v4 + u40 * I4))
        Dph_terms = [(-EJS, dphi_p, I4)]
        phi2_terms = [(1.0, u2_p, I4)]
        n1_terms = [(1.0, m1_p, I4), (1.0, Ik, m4)]

        object.__setattr__(self, '_eigvals', evals)
        object.__setattr__(self, '_ops', {
            'd_theta': project_low_energy(Dth_terms),
            'd_phi': project_low_energy(Dph_terms),
            'phase_grid': project_low_energy(phi2_terms),
            'n_cap': project_low_energy(n1_terms)})
        if self.verbose:
            print('total diagonalization {:.1f}s; kept {} x {} = {} coupled levels'.format(
                time.time() - t0, nkeep, N4, nkeep * N4))

    # computed energy levels of the circuit
    def levels(self, nlev=None, eigvecs=False):
        if eigvecs:
            raise NotImplementedError(
                'Full-space eigenvectors are not stored for the four-mode model; all drive '
                'operators are available pre-projected via d_theta()/d_phi()/n_cap().')
        self._diagonalize()
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > len(self._eigvals):
            raise Exception('`nlev` is out of bounds.')
        return self._eigvals[:nlev]

    # returns one specific energy level by index
    def level(self, level_index):
        return self.levels(nlev=level_index + 1)[level_index]

    #returns transition frequency between two levels
    def freq(self, level1, level2):
        evals = self.levels(nlev=max(level1, level2) + 1)
        return evals[level2] - evals[level1]

    #returns transitions energy between two levels
    def transition_energies(self, lower_level=0, nlev=None):
        if nlev is None:
            nlev = self.nlev
        eigvals = self.levels(nlev=nlev)[lower_level:nlev]
        return eigvals - eigvals[0]

    #circuit hamiltonian in low-energy eigenbasis
    def H(self, nlev=None):
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    #returns identity operator
    def eye(self, nlev=None):
        if nlev is None:
            nlev = self.nlev
        return qt.qeye(nlev)

    #energy splitting between each pair of low-energy doublet
    #E1-E0; E3-E2; E5-E4
    def doublet_splittings(self, n_doublets=3):
        evals = self.levels(nlev=2 * n_doublets)
        return np.array([evals[2 * k + 1] - evals[2 * k] for k in range(n_doublets)])


    #returns one of the stored operators in low-energy eigenbasis
    def _op(self, key, nlev=None):
        self._diagonalize()
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > len(self._eigvals):
            raise Exception('`nlev` is out of bounds.')
        return qt.Qobj(self._ops[key][:nlev, :nlev])

    #differential flux-line drive D_theta = dH/d(theta_ext)
    def d_theta(self, nlev=None):
        return self._op('d_theta', nlev)

    #common flux-line drive D_phi = dH/d(phi_ext).
    def d_phi(self, nlev=None):
        return self._op('d_phi', nlev)

    #capacitive drive n_1 = n_S + n_Sigma = m1 + m4
    def n_cap(self, nlev=None):
        return self._op('n_cap', nlev)

    #returns gird-mode phase coorindta u2 in low-enery eigenbasis: matrix element in the paper
    def phase_grid(self, nlev=None):
        return self._op('phase_grid', nlev)

    def phi(self, nlev=None):
        return self.d_theta(nlev)

    def n(self, nlev=None):
        return self.n_cap(nlev)

    #matrix element of D_theta between two eigenstates: <i|phi|j>
    def phi_ij(self, level1, level2):
        self._diagonalize()
        return self._ops['d_theta'][level1, level2]

    #matrix element of n_1 between two eigenstates: <i|n|j>
    def n_ij(self, level1, level2):
        self._diagonalize()
        return self._ops['n_cap'][level1, level2]

    #matrix element of D_phi between two eigenstates: <i|D_phi|j>
    def d_phi_ij(self, level1, level2):
        self._diagonalize()
        return self._ops['d_phi'][level1, level2]

    #matrix element of phi_2 = phi_S - phi_Sigma between two eigenstates: <i|u2|j>
    def phase_grid_ij(self, level1, level2):
        self._diagonalize()
        return self._ops['phase_grid'][level1, level2]

    def save_obj(self, dir):
        with open(dir + self._save_str(), 'wb') as f:
            dill.dump(self, f)
