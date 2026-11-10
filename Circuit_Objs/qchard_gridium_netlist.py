#four-mode hamiltonian derived from circuit netlist 

from dataclasses import dataclass
import time

import numpy as np
import qutip as qt
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

__all__ = ['Gridium4Mode', 'SpectrumSweepResult']

_STAGE1_NUMERICAL_PARAMETERS = frozenset({
    'stage1_sigma', 'stage1_which', 'stage1_tol', 'stage1_ncv', 'stage1_v0',
    'stage1_permc_spec', 'stage1_collect_diagnostics',
})


@dataclass(frozen=True)

#results container
class SpectrumSweepResult:
    energy_table: np.ndarray
    param_name: str
    param_vals: np.ndarray

#computes the eigenenergy at one point in a parameter sweep 
def _solve_sweep_point(task):
    constructor_kwargs, param_name, param_value, evals_count = task
    point_kwargs = dict(constructor_kwargs)
    point_kwargs[param_name] = param_value
    point_kwargs['nlev'] = max(int(point_kwargs['nlev']), evals_count)
    model = Gridium4Mode(**point_kwargs)
    return model.levels(nlev=evals_count)

#execute list of spectrum calculations either sequentially or concurrently
#parallelize parameter sweep
def _ordered_process_map(tasks, num_cpus):

    #serial path
    if num_cpus == 1:
        return list(map(_solve_sweep_point, tasks))

    #load pathos as the parallelization library 
    try:
        from pathos.pools import ProcessPool
    except ImportError as exc:
        raise ImportError(
            'Parallel netlist sweeps require pathos. Install pathos or use num_cpus=1.'
        ) from exc

    pool = ProcessPool(nodes=num_cpus)
    try:
        results = list(pool.map(_solve_sweep_point, tasks))
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
        pool.join()
        return results
    finally:
        pool.clear()

# Computes the entire spectrum across a four-mode parameter sweep.
class _GridiumSpectrumSweepMixin:

    def _constructor_kwargs(self):
        raise NotImplementedError

    #get the eigenvalues across different parameter points
    def get_spectrum_vs_paramvals(self, param_name, param_vals, evals_count=None,
                                  subtract_ground=False, num_cpus=1):
      
        constructor_kwargs = self._constructor_kwargs()
        if (param_name not in constructor_kwargs or param_name == 'nlev'
                or param_name in _STAGE1_NUMERICAL_PARAMETERS):
            raise ValueError('Unknown or unsupported sweep parameter: %s' % param_name)

        values = np.asarray(param_vals)
        if values.ndim != 1:
            raise ValueError('param_vals must be a one-dimensional sequence.')

        evals_count = self.nlev if evals_count is None else int(evals_count)
        if evals_count < 1:
            raise ValueError('evals_count must be positive.')
        if isinstance(num_cpus, bool) or not isinstance(num_cpus, (int, np.integer)):
            raise ValueError('num_cpus must be a positive integer.')
        if num_cpus < 1:
            raise ValueError('num_cpus must be a positive integer.')

        if len(values) == 0:
            table = np.empty((0, evals_count), dtype=float)
        else:
            worker_count = min(int(num_cpus), len(values))
            tasks = [
                (constructor_kwargs, param_name, value, evals_count)
                for value in values
            ]
            table = np.asarray(_ordered_process_map(tasks, worker_count), dtype=float)

        #converting energies into transitions
        if subtract_ground and len(table):
            table = table - table[:, :1]
        return SpectrumSweepResult(table, 
                                   param_name, 
                                   values.copy())


# basis operators

#constructs numerical operators for the compact coordinate theta1: charge basis 
def _charge_ops(n1max):

    N = 2 * n1max + 1 #dimensions 
    nvec = np.arange(-n1max, n1max + 1).astype(float) #eigenvalues
    R = sps.diags(np.ones(N - 1), -1)          # raising operators: R|n> = |n+1>
    cos1 = (0.5 * (R + R.T)).tocsr()
    sin1 = (-0.5j * (R - R.T)).tocsr()
    return nvec, cos1, sin1, sps.identity(N, format='csr')

#constructs operator for extended phase coordinatesL theta2 & theta3: real-space grid 
def _grid_ops(N, L):
  
    x = np.linspace(-L, L, N) #hilert space
    h = x[1] - x[0] #spacing

    #constructing charge conjugate
    c1 = 1.0 / (12 * h)
    D1 = sps.diags([c1 * np.ones(N - 2), #forth-order central-difference formula
                    -8 * c1 * np.ones(N - 1),
                    8 * c1 * np.ones(N - 1), 
                    -c1 * np.ones(N - 2)
                    ], 
                    [-2, -1, 1, 2]
                )
    n = (-1j) * D1                              # n = -i d/dx, Hermitian

    #n^2
    c2 = 1.0 / (12 * h * h)
    nsq = sps.diags([c2 * np.ones(N - 2), 
                    -16 * c2 * np.ones(N - 1), 
                    30 * c2 * np.ones(N),
                    -16 * c2 * np.ones(N - 1), 
                    c2 * np.ones(N - 2)
                    ], 
                    [-2, -1, 0, 1, 2]
                )

    return x, n.tocsr(), nsq.tocsr(), sps.identity(N, format='csr')


# coefficient derivation

#return hamlitonian coefficients: ECmat, K, EJ * (1 - eps_J), EJ * (1 + eps_J)
def _coeffs(EJ, EC, EL, ELK, EJS, ECS, eps_J, eps_LK, eC, eP):
    C = np.zeros((4, 4))

    #node capacitance matrix 
    C[0, 0] += 1.0 / EC          # JJ(0,1) shunt cap
    C[1, 1] += 1.0 / EC          # JJ(0,2) shunt cap
    C[2, 2] += 1.0 / ECS         # JJs(0,3) QPS junction cap

    C[0, 0] += 1.0 / eC      # cross-KITE C(1,2)
    C[1, 1] += 1.0 / eC
    C[0, 1] -= 1.0 / eC
    C[1, 0] -= 1.0 / eC
    C[3, 3] += 1.0 / eP      # parasitic node C(0,4)

    M = np.array([[1., 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]])

    ECmat = np.linalg.inv(M.T @ C @ M) #coordinate transformation 

    #inductive asymmetry
    ELK1 = ELK * (1 - eps_LK)
    ELK2 = ELK * (1 + eps_LK) #inductive asymmetry

    # inductive branch vector terms
    v1 = np.array([0., 0., -1.])
    v2 = np.array([1., 0., -1.])
    v3 = np.array([1., 1., -1.])
    K = ELK1 * np.outer(v1, v1) + ELK2 * np.outer(v2, v2) + EL * np.outer(v3, v3)

    return ECmat, K, EJ * (1 - eps_J), EJ * (1 + eps_J)


# Four-mode Hamiltonian assembly
#builds the part of the four-mode hamiltonian that does not contain theta4

def _assemble_nonlinear_sector(ECm, K, EJ1, EJ2, EJS, ng, phi_ext, theta_ext,
                               n1max, N2, L2, N3, L3):
    
    #single-mode operators 
    nvec, cos1, sin1, I1 = _charge_ops(n1max)
    x2, n2, n2sq, I2 = _grid_ops(N2, L2)
    x3, n3, n3sq, I3 = _grid_ops(N3, L3)

    #tensor product helper
    def kron3(A, B, C):
        return sps.kron(sps.kron(C, B, format='csr'), A, format='csr')

    #offset charge 
    n1d = sps.diags(nvec + ng)
    n1sqd = sps.diags((nvec + ng) ** 2)

    # Kinetic energy: 4 sum_ij ECm[i,j] n_i n_j; symmetry doubles cross terms.
    H = 4 * ECm[0, 0] * kron3(n1sqd, I2, I3)
    H = H + 4 * ECm[1, 1] * kron3(I1, n2sq, I3)
    H = H + 4 * ECm[2, 2] * kron3(I1, I2, n3sq)
    if ECm[0, 1] != 0.0:
        H = H + 8 * ECm[0, 1] * kron3(n1d, n2, I3)
    if ECm[0, 2] != 0.0:
        H = H + 8 * ECm[0, 2] * kron3(n1d, I2, n3)
    if ECm[1, 2] != 0.0:
        H = H + 8 * ECm[1, 2] * kron3(I1, n2, n3)

    # Theta4-independent part of 0.5 x^T K x, x=(theta2, theta3, theta4).
    H = H + 0.5 * K[0, 0] * kron3(I1, sps.diags(x2 ** 2), I3)
    H = H + 0.5 * K[1, 1] * kron3(I1, I2, sps.diags(x3 ** 2))
    if K[0, 1] != 0.0:
        H = H + K[0, 1] * kron3(I1, sps.diags(x2), sps.diags(x3))

    H = H - EJ1 * kron3(cos1, I2, I3)
    c2t = sps.diags(np.cos(x2 + theta_ext))
    s2t = sps.diags(np.sin(x2 + theta_ext))
    H = H - EJ2 * (kron3(cos1, c2t, I3) - kron3(sin1, s2t, I3))

    c2 = sps.diags(np.cos(x2))
    s2 = sps.diags(np.sin(x2))
    c3p = sps.diags(np.cos(x3 + phi_ext))
    s3p = sps.diags(np.sin(x3 + phi_ext))
    H = H - EJS * (kron3(cos1, c2, c3p) - kron3(cos1, s2, s3p)
                   - kron3(sin1, s2, c3p) - kron3(sin1, c2, s3p))

    H = (0.5 * (H + H.conj().T)).tocsr()
    ops = dict(
        nvec=nvec, x2=x2, x3=x3, I1=I1, I2=I2, I3=I3,
        n2=n2, n3=n3, cos1=cos1, sin1=sin1, kron3=kron3,
    )
    return H, ops


def _stage1_eigensolve(matrix, k, sigma, which='LM', tol=1e-8, ncv=None,
                       v0=None, permc_spec=None, collect_diagnostics=False):
    """Solve the standard Stage-1 eigenproblem, whose mass matrix is ``M=I``.

    When ``permc_spec`` is provided, the explicit shift-invert operator is
    ``OPinv = (A - sigma I)^(-1)``.  The remaining arguments retain their
    ``scipy.sparse.linalg.eigsh`` meanings.
    """
    matrix = matrix.tocsc()
    eigsh_kwargs = dict(
        k=k, sigma=sigma, which=which, tol=tol, ncv=ncv, v0=v0,
    )

    if permc_spec is None:
        started = time.perf_counter() if collect_diagnostics else None
        values, vectors = spsl.eigsh(matrix, **eigsh_kwargs)
        diagnostics = None
        if collect_diagnostics:
            diagnostics = {
                'path': 'scipy_default_shift_invert',
                'factorization_seconds': None,
                'eigsh_seconds': time.perf_counter() - started,
                'opinv_solve_count': None,
                'L_nnz': None,
                'U_nnz': None,
                'LU_nnz': None,
                'fill_ratio': None,
            }
    else:
        shifted = matrix - sigma * sps.identity(
            matrix.shape[0], dtype=matrix.dtype, format='csc',
        )
        started = time.perf_counter() if collect_diagnostics else None
        lu = spsl.splu(shifted, permc_spec=permc_spec)
        factorization_seconds = (
            time.perf_counter() - started if collect_diagnostics else None
        )

        if collect_diagnostics:
            solve_count = 0

            def inverse_solve(vector):
                nonlocal solve_count
                solve_count += 1
                return lu.solve(vector)
        else:
            inverse_solve = lu.solve

        opinv = spsl.LinearOperator(
            matrix.shape, matvec=inverse_solve, dtype=matrix.dtype,
        )
        started = time.perf_counter() if collect_diagnostics else None
        values, vectors = spsl.eigsh(matrix, OPinv=opinv, **eigsh_kwargs)
        diagnostics = None
        if collect_diagnostics:
            lu_nnz = lu.L.nnz + lu.U.nnz
            diagnostics = {
                'path': 'explicit_shift_invert',
                'permc_spec': permc_spec,
                'factorization_seconds': factorization_seconds,
                'eigsh_seconds': time.perf_counter() - started,
                'opinv_solve_count': solve_count,
                'L_nnz': lu.L.nnz,
                'U_nnz': lu.U.nnz,
                'LU_nnz': lu_nnz,
                'fill_ratio': lu_nnz / shifted.nnz,
            }

    order = np.argsort(values)
    return np.real(values[order]), vectors[:, order], diagnostics


class Gridium4Mode(_GridiumSpectrumSweepMixin):
    """Four-mode Gridium model with optional Stage-1 numerical controls.

    The ``stage1_*`` arguments configure only the numerical eigensolver; they
    do not alter the Gridium Hamiltonian or any physical model parameter.
    ``stage1_sigma=None`` selects the existing Gridium-derived default shift,
    rather than SciPy's ordinary no-shift ``sigma=None`` behavior.
    """

    #numerical cutoff: 
    #n1max: compact-charge cutoff
    #N2, L2: resolution and extent for theta2
    #N3, L3: resolution and extent for theta3
    #N4: number of forth-mode oscillator states 
    #nkeep: number of dressed nonlinear-sector states retained
    #nlev: number of final low-energy states stored
    def __init__(self, EJ, EC, EL, ELK, EJS, ECS, eC, eP, eps_J=0.0, eps_LK=0.0,
                 ng=0.0, phi_ext=0.0, theta_ext=np.pi, nlev=6,
                 n1max=4, N2=51, L2=12.0, N3=51, L3=13.0, N4=8, nkeep=100,
                 stage1_sigma=None, stage1_which='LM', stage1_tol=1e-8,
                 stage1_ncv=None, stage1_v0=None, stage1_permc_spec=None,
                 stage1_collect_diagnostics=False):
        self.EJ, self.EC, self.EL, self.ELK, self.EJS, self.ECS = EJ, EC, EL, ELK, EJS, ECS
        self.eC, self.eP = eC, eP
        self.eps_J, self.eps_LK, self.ng = eps_J, eps_LK, ng
        self.phi_ext, self.theta_ext, self.nlev = phi_ext, theta_ext, nlev
        self.n1max, self.N2, self.L2, self.N3, self.L3 = n1max, N2, L2, N3, L3
        self.N4, self.nkeep = N4, nkeep
        self.stage1_sigma = stage1_sigma
        self.stage1_which = stage1_which
        self.stage1_tol = stage1_tol
        self.stage1_ncv = stage1_ncv
        self.stage1_v0 = (
            None if stage1_v0 is None else np.array(stage1_v0, copy=True)
        )
        self.stage1_permc_spec = stage1_permc_spec
        self.stage1_collect_diagnostics = stage1_collect_diagnostics
        self.stage1_solver_diagnostics = None
        self._evals = None
        self._ops = None

    #return constructor input
    def _constructor_kwargs(self):
        return dict(
            EJ=self.EJ, EC=self.EC, EL=self.EL, ELK=self.ELK, EJS=self.EJS, ECS=self.ECS,
            eC=self.eC, eP=self.eP, eps_J=self.eps_J, eps_LK=self.eps_LK, ng=self.ng,
            phi_ext=self.phi_ext, theta_ext=self.theta_ext, nlev=self.nlev,
            n1max=self.n1max, N2=self.N2, L2=self.L2, N3=self.N3, L3=self.L3,
            N4=self.N4, nkeep=self.nkeep,
            stage1_sigma=self.stage1_sigma, stage1_which=self.stage1_which,
            stage1_tol=self.stage1_tol, stage1_ncv=self.stage1_ncv,
            stage1_v0=(
                None if self.stage1_v0 is None else self.stage1_v0.copy()
            ),
            stage1_permc_spec=self.stage1_permc_spec,
            stage1_collect_diagnostics=self.stage1_collect_diagnostics,
        )

    def _solve(self, compute_operators=False):
        self.stage1_solver_diagnostics = None
        #first get the coefficients
        ECm, K, EJ1, EJ2 = _coeffs(self.EJ, self.EC, self.EL, self.ELK, self.EJS, self.ECS,
                                   self.eps_J, self.eps_LK, eC=self.eC, eP=self.eP)

        # Stage 1: nonlinear (theta1, theta2, theta3) sector, containing all
        # four-mode Hamiltonian terms that do not involve theta4 or n4.
        # dimensions: D123 = 9*51*51 = 23409

        H1, o = _assemble_nonlinear_sector(
            ECm[:3, :3], K[:2, :2], EJ1, EJ2, self.EJS, self.ng,
            self.phi_ext, self.theta_ext,
            self.n1max, self.N2, self.L2, self.N3, self.L3,
        )
        default_sigma = -(EJ1 + EJ2 + self.EJS + 10.0)
        sigma = default_sigma if self.stage1_sigma is None else self.stage1_sigma
        nkeep = min(self.nkeep, H1.shape[0]) 
        if nkeep == H1.shape[0]:
            started = time.perf_counter() if self.stage1_collect_diagnostics else None
            w1, V = np.linalg.eigh(H1.toarray())
            if self.stage1_collect_diagnostics:
                self.stage1_solver_diagnostics = {
                    'path': 'dense',
                    'eigh_seconds': time.perf_counter() - started,
                }
        else:
            w1, V, self.stage1_solver_diagnostics = _stage1_eigensolve(
                H1, k=nkeep, sigma=sigma, which=self.stage1_which,
                tol=self.stage1_tol, ncv=self.stage1_ncv, v0=self.stage1_v0,
                permc_spec=self.stage1_permc_spec,
                collect_diagnostics=self.stage1_collect_diagnostics,
            )

        # theta4: exactly harmonic.  A n4^2 + B th4^2, A = 4 EC44, B = 0.5 K44
        A4, B4 = 4 * ECm[3, 3], 0.5 * K[2, 2]
        f4 = (A4 / (4 * B4)) ** 0.25
        g4 = 1.0 / (2 * f4)
        a = np.diag(np.sqrt(np.arange(1, self.N4)), 1)      # annihilation
        th4 = f4 * (a + a.T)
        n4 = 1j * g4 * (a.T - a)
        w4 = 2 * np.sqrt(A4 * B4) * (np.arange(self.N4) + 0.5)

        # projected sector operators for the bilinear couplings
        kron3 = o['kron3']
        def proj(Op):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                projected = V.conj().T @ (Op @ V)
            if not np.all(np.isfinite(projected)):
                raise FloatingPointError('Projected sector operator contains NaN/Inf entries.')
            return projected
        n1full = kron3(sps.diags(o['nvec'] + self.ng), o['I2'], o['I3'])
        n1barefull = kron3(sps.diags(o['nvec']), o['I2'], o['I3'])
        n2full = kron3(o['I1'], o['n2'], o['I3'])
        n3full = kron3(o['I1'], o['I2'], o['n3'])
        x2full = kron3(o['I1'], sps.diags(o['x2']), o['I3'])
        x3full = kron3(o['I1'], o['I2'], sps.diags(o['x3']))

        # Cache every stage-one projection.  These matrices enter both the coupled
        # Hamiltonian and the low-energy observables, so projecting each one once
        # avoids several repeated sparse matrix products in operator calculations.
        n1_p = proj(n1full)
        n1bare_p = proj(n1barefull)
        n2_p = proj(n2full)
        n3_p = proj(n3full)
        x2_p = proj(x2full)
        x3_p = proj(x3full)

        if compute_operators:
            # Since H_J = -E_J cos(argument), differentiation with respect to the
            # dimensionless external-flux phase gives +E_J sin(argument).
            c2t = sps.diags(np.cos(o['x2'] + self.theta_ext))
            s2t = sps.diags(np.sin(o['x2'] + self.theta_ext))
            dtheta_full = EJ2 * (
                kron3(o['sin1'], c2t, o['I3']) + kron3(o['cos1'], s2t, o['I3'])
            )
            c2 = sps.diags(np.cos(o['x2']))
            s2 = sps.diags(np.sin(o['x2']))
            c3p = sps.diags(np.cos(o['x3'] + self.phi_ext))
            s3p = sps.diags(np.sin(o['x3'] + self.phi_ext))
            dphi_full = self.EJS * (
                kron3(o['sin1'], c2, c3p)
                + kron3(o['cos1'], s2, c3p)
                + kron3(o['cos1'], c2, s3p)
                - kron3(o['sin1'], s2, s3p)
            )
            dtheta_sector = proj(dtheta_full)
            dphi_sector = proj(dphi_full)
            #   varphi_2 = varphi_S - varphi_Sigma = theta_3 + theta_2 / 2.
            # The conjugate canonical charge follows from the same linear canonical
            # transformation (including the fourth node mode):
            #   n_grid = -n_theta1 + n_theta3 + n_theta4.
            phase_grid_sector = 0.5 * x2_p + x3_p

        # stage 2: exact coupled Hamiltonian in (kept sector) x (theta4 HO)
        #D_coupled = n_keep * N4
        Ik, I4 = np.eye(nkeep), np.eye(self.N4)
        Hc = np.kron(np.diag(w1), I4) + np.kron(Ik, np.diag(w4))
        Hc = Hc + 8 * ECm[0, 3] * np.kron(n1_p, n4)
        Hc = Hc + 8 * ECm[1, 3] * np.kron(n2_p, n4)
        Hc = Hc + 8 * ECm[2, 3] * np.kron(n3_p, n4)
        Hc = Hc + K[0, 2] * np.kron(x2_p, th4)
        Hc = Hc + K[1, 2] * np.kron(x3_p, th4)
        Hc = 0.5 * (Hc + Hc.conj().T)
        nstore = min(self.nlev, Hc.shape[0])
        if compute_operators:
            evals, W = spla.eigh(
                Hc, subset_by_index=(0, nstore - 1), driver='evr', check_finite=False,
            )

            states = W.reshape(nkeep, self.N4, nstore)

            def finish_projection(applied):
                applied = applied.reshape(nkeep * self.N4, nstore)
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    matrix = W.conj().T @ applied
                matrix = 0.5 * (matrix + matrix.conj().T)
                if not np.all(np.isfinite(matrix)):
                    raise FloatingPointError(
                        'Projected physical operator contains NaN/Inf entries.'
                    )
                return matrix

            def project_sector_to_low_energy(sector_operator):
                applied = np.einsum(
                    'ab,bin->ain', sector_operator, states, optimize=True,
                )
                return finish_projection(applied)

            def project_grid_charge_to_low_energy():
                applied = np.einsum(
                    'ab,bin->ain', -n1bare_p + n3_p, states, optimize=True,
                )
                applied += np.einsum('ij,ajn->ain', n4, states, optimize=True)
                return finish_projection(applied)

            self._ops = {
                'n1': project_sector_to_low_energy(n1bare_p),
                'phase2': project_sector_to_low_energy(x2_p),
                'grid_n': project_grid_charge_to_low_energy(),
                'grid_phi': project_sector_to_low_energy(phase_grid_sector),
                'd_theta': project_sector_to_low_energy(dtheta_sector),
                'd_phi': project_sector_to_low_energy(dphi_sector),
            }
        else:
            evals = spla.eigh(
                Hc, subset_by_index=(0, nstore - 1), driver='evr',
                eigvals_only=True, check_finite=False,
            )
            self._ops = None
        self._evals = np.real(evals)

    def levels(self, nlev=None):
        if self._evals is None:
            self._solve()
        return self._evals[:(nlev or self.nlev)]

    def _op(self, key, nlev=None):
        if self._ops is None:
            self._solve(compute_operators=True)
        nlev = self.nlev if nlev is None else int(nlev)
        if nlev < 1 or nlev > len(self._evals):
            raise ValueError('nlev must be between 1 and %d.' % len(self._evals))
        return qt.Qobj(self._ops[key][:nlev, :nlev])

    def d_theta(self, nlev=None):
        """Return the external-flux response dH/d(theta_ext) (GHz/radian).

        This is a bias-sensitivity observable, not the default abstract
        high-frequency control operator for current Gridium gate work.
        """
        return self._op('d_theta', nlev)

    def d_phi(self, nlev=None):
        """Return the external-flux response dH/d(phi_ext) (GHz/radian).

        This response operator is distinct from :meth:`grid_phi`, the global
        Gridium phase coordinate used as the present abstract control model.
        """
        return self._op('d_phi', nlev)

    def n1(self, nlev=None):
        """Return the raw compact netlist charge n_theta1."""
        return self._op('n1', nlev)

    def phase2(self, nlev=None):
        """Return the raw extended netlist coordinate theta2, in radians."""
        return self._op('phase2', nlev)

    def grid_n(self, nlev=None):
        """Return the effective grid-mode charge conjugate to grid_phi()."""
        return self._op('grid_n', nlev)

    def grid_phi(self, nlev=None):
        """Return the abstract global Gridium phase-control coordinate.

        In the netlist basis this is paper Eq. S24
        ``varphi2 = varphiS - varphiSigma = theta3 + theta2/2``, in radians.
        It is the present conceptual control operator for four-mode gate work;
        it does not by itself specify a calibrated experimental control line.
        """
        return self._op('grid_phi', nlev)
