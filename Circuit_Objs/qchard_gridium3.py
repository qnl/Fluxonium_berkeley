"""Symmetric three-mode Gridium Hamiltonian from paper Eq. S23.

This module implements only the source-grounded, Born--Oppenheimer-reduced
three-mode Hamiltonian.  It does not implement the compact-coordinate form
Eq. S26, fabrication asymmetry, or the four-mode parasitic-capacitance model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigsh

__all__ = [
    'SymmetricThreeModeGridium',
    'table_s1_regime_a',
    'table_s1_regime_b',
    'table_s1_regime_c',
    'table_s1_regime_d',
]


table_s1_regime_a = {
    'E_J': 5.0, 'E_C': 0.5, 'E_L': 1.0, 'E_LK': 1.0,
    'E_JS': 4.0, 'E_CS': 8.0,
}
table_s1_regime_b = {
    'E_J': 10.0, 'E_C': 0.5, 'E_L': 1.0, 'E_LK': 1.0,
    'E_JS': 4.0, 'E_CS': 8.0,
}
table_s1_regime_c = {
    'E_J': 10.0, 'E_C': 0.5, 'E_L': 0.5, 'E_LK': 0.5,
    'E_JS': 4.0, 'E_CS': 8.0,
}
table_s1_regime_d = {
    'E_J': 10.0, 'E_C': 0.5, 'E_L': 0.2, 'E_LK': 0.2,
    'E_JS': 4.0, 'E_CS': 8.0,
}


@dataclass(frozen=True)
class _ModeOperators:
    identity: qt.Qobj
    phi: qt.Qobj
    n: qt.Qobj
    cos_phi: qt.Qobj


class SymmetricThreeModeGridium:
    r"""Symmetric extended-phase realization of paper Eq. S23.

    Energies are stored as ordinary frequencies in GHz, consistently with
    the other qchard circuit objects.  Time-evolution callers must therefore
    apply the repository's established ``2*pi`` conversion when using ns.

    Each canonical pair is represented in an independently truncated
    harmonic-oscillator basis,

    ``phi_j = center_j + scale_j * position`` and
    ``n_j = momentum / scale_j``.

    This is an extended, noncompact phase basis.  ``phase_scales`` and
    ``phase_centers`` affect finite-cutoff convergence only; they are not
    physical parameters and do not redefine the physical coordinates.
    """

    name = 'Symmetric three-mode Gridium (S23)'
    type = 'qubit'
    mode_order = ('sigma', 'delta', 's')

    def __init__(
            self, E_C, E_J, E_L, E_LK, E_JS, E_CS, *,
            phi_ext=0.0, theta_ext=np.pi,
            trunc_sigma=10, trunc_delta=10, trunc_s=10,
            nlev=12, phase_scales=None, phase_centers=None, units='GHz',
            eigensolver_tol=1e-10, eigensolver_maxiter=None):
        self._basis_cache = None
        self._hamiltonian_cache = None
        self._eigvals = None
        self._eigvec_matrix = None
        self._energy_operator_cache = {}

        self._E_C = self._positive_float('E_C', E_C)
        self._E_J = self._positive_float('E_J', E_J)
        self._E_L = self._positive_float('E_L', E_L)
        self._E_LK = self._positive_float('E_LK', E_LK)
        self._E_JS = self._positive_float('E_JS', E_JS)
        self._E_CS = self._positive_float('E_CS', E_CS)
        self._phi_ext = self._finite_float('phi_ext', phi_ext)
        self._theta_ext = self._finite_float('theta_ext', theta_ext)
        self._trunc_sigma = self._positive_int(
            'trunc_sigma', trunc_sigma, minimum=2)
        self._trunc_delta = self._positive_int(
            'trunc_delta', trunc_delta, minimum=2)
        self._trunc_s = self._positive_int(
            'trunc_s', trunc_s, minimum=2)
        self._nlev = self._positive_int('nlev', nlev, minimum=1)
        if self._nlev >= self.hilbert_dimension:
            raise ValueError('nlev must be smaller than the tensor dimension.')
        self._phase_scales_auto = phase_scales is None
        self._phase_scales = self._validate_phase_scales(phase_scales)
        self._phase_centers = self._validate_phase_centers(phase_centers)
        self.units = units
        self.eigensolver_tol = float(eigensolver_tol)
        self.eigensolver_maxiter = eigensolver_maxiter

    @staticmethod
    def _positive_float(name, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('{} must be finite and positive.'.format(name))
        return value

    @staticmethod
    def _finite_float(name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('{} must be finite.'.format(name))
        return value

    @staticmethod
    def _positive_int(name, value, minimum):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(
                '{} must be an integer of at least {}.'.format(name, minimum))
        return int(value)

    def _validate_phase_scales(self, scales):
        if scales is None:
            return self._default_phase_scales()
        array = np.asarray(scales, dtype=float)
        if array.shape != (3,) or np.any(~np.isfinite(array)) or np.any(array <= 0):
            raise ValueError('phase_scales must contain three positive values.')
        return tuple(float(value) for value in array)

    @staticmethod
    def _validate_phase_centers(centers):
        if centers is None:
            return (0.0, 0.0, 0.0)
        array = np.asarray(centers, dtype=float)
        if array.shape != (3,) or np.any(~np.isfinite(array)):
            raise ValueError('phase_centers must contain three finite values.')
        return tuple(float(value) for value in array)

    def _default_phase_scales(self):
        # These balance the kinetic coefficient against the local quadratic
        # curvature of S23 near a cosine minimum.  They are basis choices,
        # not additional terms in the Hamiltonian.
        return (
            (2 * self.E_C / (self.E_J + self.ELprime)) ** 0.25,
            (2 * self.E_C / (self.E_J + self.E_LK)) ** 0.25,
            (4 * self.E_CS / (0.5 * self.E_JS + self.ELprime)) ** 0.25,
        )

    def _reset_cache(self, reset_basis=False):
        if reset_basis:
            self._basis_cache = None
        self._hamiltonian_cache = None
        self._eigvals = None
        self._eigvec_matrix = None
        self._energy_operator_cache = {}

    def _refresh_automatic_phase_scales(self):
        if self._phase_scales_auto:
            self._phase_scales = self._default_phase_scales()

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        self._E_C = self._positive_float('E_C', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        self._E_J = self._positive_float('E_J', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, value):
        self._E_L = self._positive_float('E_L', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def E_LK(self):
        return self._E_LK

    @E_LK.setter
    def E_LK(self, value):
        self._E_LK = self._positive_float('E_LK', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def E_JS(self):
        return self._E_JS

    @E_JS.setter
    def E_JS(self, value):
        self._E_JS = self._positive_float('E_JS', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def E_CS(self):
        return self._E_CS

    @E_CS.setter
    def E_CS(self, value):
        self._E_CS = self._positive_float('E_CS', value)
        self._refresh_automatic_phase_scales()
        self._reset_cache(reset_basis=True)

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value):
        self._phi_ext = self._finite_float('phi_ext', value)
        self._reset_cache()

    @property
    def theta_ext(self):
        return self._theta_ext

    @theta_ext.setter
    def theta_ext(self, value):
        self._theta_ext = self._finite_float('theta_ext', value)
        self._reset_cache()

    @property
    def trunc_sigma(self):
        return self._trunc_sigma

    @trunc_sigma.setter
    def trunc_sigma(self, value):
        self._trunc_sigma = self._positive_int(
            'trunc_sigma', value, minimum=2)
        self._validate_nlev_after_truncation()
        self._reset_cache(reset_basis=True)

    @property
    def trunc_delta(self):
        return self._trunc_delta

    @trunc_delta.setter
    def trunc_delta(self, value):
        self._trunc_delta = self._positive_int(
            'trunc_delta', value, minimum=2)
        self._validate_nlev_after_truncation()
        self._reset_cache(reset_basis=True)

    @property
    def trunc_s(self):
        return self._trunc_s

    @trunc_s.setter
    def trunc_s(self, value):
        self._trunc_s = self._positive_int('trunc_s', value, minimum=2)
        self._validate_nlev_after_truncation()
        self._reset_cache(reset_basis=True)

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        value = self._positive_int('nlev', value, minimum=1)
        if value >= self.hilbert_dimension:
            raise ValueError('nlev must be smaller than the tensor dimension.')
        self._nlev = value
        self._eigvals = None
        self._eigvec_matrix = None
        self._energy_operator_cache = {}

    @property
    def phase_scales(self):
        return self._phase_scales

    @phase_scales.setter
    def phase_scales(self, value):
        self._phase_scales = self._validate_phase_scales(value)
        self._phase_scales_auto = value is None
        self._reset_cache(reset_basis=True)

    @property
    def phase_centers(self):
        return self._phase_centers

    @phase_centers.setter
    def phase_centers(self, value):
        self._phase_centers = self._validate_phase_centers(value)
        self._reset_cache(reset_basis=True)

    @property
    def truncations(self):
        return (self.trunc_sigma, self.trunc_delta, self.trunc_s)

    @property
    def hilbert_dimension(self):
        return int(np.prod(self.truncations))

    @property
    def ELprime(self):
        return 2 * self.E_LK * self.E_L / (2 * self.E_LK + self.E_L)

    def _validate_nlev_after_truncation(self):
        if hasattr(self, '_nlev') and self._nlev >= self.hilbert_dimension:
            raise ValueError('Current nlev is too large for this truncation.')

    @staticmethod
    def _local_mode(dimension, phase_scale, phase_center):
        annihilation = qt.destroy(dimension).to('csr')
        identity = qt.qeye(dimension).to('csr')
        phi = (
            phase_center * identity
            + phase_scale / np.sqrt(2)
            * (annihilation + annihilation.dag())).to('csr')
        n = (
            -1j / (phase_scale * np.sqrt(2))
            * (annihilation - annihilation.dag())).to('csr')
        cos_phi = (0.5 * (
            (1j * phi).expm() + (-1j * phi).expm())).to('csr')
        return _ModeOperators(
            identity=identity,
            phi=phi, n=n, cos_phi=cos_phi)

    def _basis(self):
        if self._basis_cache is None:
            self._basis_cache = tuple(
                self._local_mode(dimension, scale, center)
                for dimension, scale, center in zip(
                    self.truncations, self.phase_scales,
                    self.phase_centers))
        return self._basis_cache

    def _promote(self, sigma=None, delta=None, s=None):
        modes = self._basis()
        return qt.tensor(
            modes[0].identity if sigma is None else sigma,
            modes[1].identity if delta is None else delta,
            modes[2].identity if s is None else s).to('csr')

    def _identity(self):
        return self._promote()

    def _primitive_operator(self, name):
        sigma, delta, s_mode = self._basis()
        primitive = {
            'phi_sigma': lambda: self._promote(sigma=sigma.phi),
            'phi_delta': lambda: self._promote(delta=delta.phi),
            'phi_s': lambda: self._promote(s=s_mode.phi),
            'n_sigma': lambda: self._promote(sigma=sigma.n),
            'n_delta': lambda: self._promote(delta=delta.n),
            'n_s': lambda: self._promote(s=s_mode.n),
        }
        if name in primitive:
            return primitive[name]()
        if name == 'phi_1':
            return self._primitive_operator('phi_s')
        if name == 'phi_2':
            return (
                self._primitive_operator('phi_s')
                - self._primitive_operator('phi_sigma')).to('csr')
        if name == 'phi_3':
            return self._primitive_operator('phi_delta')
        if name == 'n_1':
            return (
                self._primitive_operator('n_s')
                + self._primitive_operator('n_sigma')).to('csr')
        if name == 'n_2':
            return (-self._primitive_operator('n_sigma')).to('csr')
        if name == 'n_3':
            return self._primitive_operator('n_delta')
        if name == 'dH_dphi_ext':
            shifted = s_mode.phi - self.phi_ext * s_mode.identity
            sine = ((shifted * 1j).expm() - (-1j * shifted).expm()) / (2j)
            return (-self.E_JS * self._promote(s=sine)).to('csr')
        if name == 'dH_dtheta_ext':
            return (-self.E_LK * (
                self._primitive_operator('phi_delta')
                - 0.5 * self.theta_ext * self._identity())).to('csr')
        raise KeyError('Unknown three-mode operator {!r}.'.format(name))

    def operator(self, name, *, basis='primitive', nlev=None):
        """Return a named operator in the primitive or energy basis."""
        operator = self._primitive_operator(name)
        if basis == 'primitive':
            return operator
        if basis != 'energy':
            raise ValueError("basis must be 'primitive' or 'energy'.")
        if nlev is None:
            nlev = self.nlev
        nlev = self._checked_nlev(nlev)
        key = (name, nlev)
        if key not in self._energy_operator_cache:
            self._solve_eigensystem(nlev)
            vectors = self._eigvec_matrix[:, :nlev]
            array = operator.data.as_scipy()
            self._energy_operator_cache[key] = qt.Qobj(
                vectors.conj().T @ (array @ vectors))
        return self._energy_operator_cache[key]

    def phi_sigma(self, **kwargs):
        return self.operator('phi_sigma', **kwargs)

    def phi_delta(self, **kwargs):
        return self.operator('phi_delta', **kwargs)

    def phi_s(self, **kwargs):
        return self.operator('phi_s', **kwargs)

    def n_sigma(self, **kwargs):
        return self.operator('n_sigma', **kwargs)

    def n_delta(self, **kwargs):
        return self.operator('n_delta', **kwargs)

    def n_s(self, **kwargs):
        return self.operator('n_s', **kwargs)

    def phi_1(self, **kwargs):
        return self.operator('phi_1', **kwargs)

    def phi_2(self, **kwargs):
        return self.operator('phi_2', **kwargs)

    def phi_3(self, **kwargs):
        return self.operator('phi_3', **kwargs)

    def n_1(self, **kwargs):
        return self.operator('n_1', **kwargs)

    def n_2(self, **kwargs):
        return self.operator('n_2', **kwargs)

    def n_3(self, **kwargs):
        return self.operator('n_3', **kwargs)

    def dH_dphi_ext(self, **kwargs):
        return self.operator('dH_dphi_ext', **kwargs)

    def dH_dtheta_ext(self, **kwargs):
        return self.operator('dH_dtheta_ext', **kwargs)

    def hamiltonian_primitive(self):
        """Return the sparse tensor-basis Hamiltonian of paper Eq. S23."""
        if self._hamiltonian_cache is None:
            sigma, delta, s_mode = self._basis()
            identity = self._identity()
            shifted_s = s_mode.phi - self.phi_ext * s_mode.identity
            cos_shifted_s = (0.5 * (
                (1j * shifted_s).expm()
                + (-1j * shifted_s).expm())).to('csr')

            kinetic = (
                2 * self.E_C * self._promote(sigma=sigma.n ** 2)
                + 2 * self.E_C * self._promote(delta=delta.n ** 2)
                + 4 * self.E_CS * self._promote(s=s_mode.n ** 2))
            josephson = (
                -2 * self.E_J * self._promote(
                    sigma=sigma.cos_phi, delta=delta.cos_phi)
                - self.E_JS * self._promote(s=cos_shifted_s))
            inductive_sigma_s = self.ELprime * (
                self._promote(sigma=sigma.phi ** 2)
                + self._promote(s=s_mode.phi ** 2)
                - 2 * self._promote(sigma=sigma.phi, s=s_mode.phi))
            inductive_delta = self.E_LK * (
                self._promote(delta=delta.phi ** 2)
                - self.theta_ext * self._promote(delta=delta.phi)
                + 0.25 * self.theta_ext ** 2 * identity)
            self._hamiltonian_cache = (
                kinetic + josephson
                + inductive_sigma_s + inductive_delta).to('csr')
        return self._hamiltonian_cache

    def _checked_nlev(self, nlev):
        nlev = self._positive_int('nlev', nlev, minimum=1)
        if nlev >= self.hilbert_dimension:
            raise ValueError('nlev must be smaller than the tensor dimension.')
        return nlev

    def _solve_eigensystem(self, nlev):
        nlev = self._checked_nlev(nlev)
        cached = 0 if self._eigvals is None else len(self._eigvals)
        if cached >= nlev:
            return
        matrix = self.hamiltonian_primitive().data.as_scipy()
        ncv = min(matrix.shape[0], max(2 * nlev + 1, 20))
        values, vectors = eigsh(
            matrix, k=nlev, which='SA', tol=self.eigensolver_tol,
            maxiter=self.eigensolver_maxiter, ncv=ncv)
        order = np.argsort(values)
        self._eigvals = np.real_if_close(values[order])
        self._eigvec_matrix = vectors[:, order]
        self._energy_operator_cache = {}

    def levels(self, nlev=None, eigvecs=False):
        if nlev is None:
            nlev = self.nlev
        nlev = self._checked_nlev(nlev)
        self._solve_eigensystem(nlev)
        values = self._eigvals[:nlev].copy()
        if not eigvecs:
            return values
        return values, self.eigvecs(nlev=nlev)

    def eigvecs(self, nlev=None):
        if nlev is None:
            nlev = self.nlev
        nlev = self._checked_nlev(nlev)
        self._solve_eigensystem(nlev)
        dims = [list(self.truncations), [1]]
        return np.asarray([
            qt.Qobj(self._eigvec_matrix[:, index, None], dims=dims)
            for index in range(nlev)], dtype=object)

    def eigvec(self, level_index):
        if level_index < 0:
            raise ValueError('level_index must be non-negative.')
        return self.eigvecs(nlev=level_index + 1)[level_index]

    def level(self, level_index):
        if level_index < 0:
            raise ValueError('level_index must be non-negative.')
        return self.levels(nlev=level_index + 1)[level_index]

    def H(self, nlev=None):
        """Return the diagonal low-energy Hamiltonian in the energy basis."""
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def freq(self, level1, level2):
        return float(self.level(level2) - self.level(level1))

    def transition_energies(self, lower_level=0, nlev=None):
        if nlev is None:
            nlev = self.nlev
        values = self.levels(nlev=nlev)[lower_level:nlev]
        return values - values[0]

    def matrix_element(self, operator_name, level1, level2):
        count = max(level1, level2) + 1
        operator = self.operator(operator_name, basis='energy', nlev=count)
        return operator[level1, level2]

    def __str__(self):
        return (
            'Symmetric three-mode Gridium (paper Eq. S23) with '
            'E_C={0.E_C:g}, E_J={0.E_J:g}, E_L={0.E_L:g}, '
            'E_LK={0.E_LK:g}, E_JS={0.E_JS:g}, E_CS={0.E_CS:g} '
            '{0.units}; truncations={0.truncations}, numerical centers='
            '{0.phase_centers}.'.format(self))
