# This file is part of QHard: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
###########################################################################
"""Classes for representing transmon qubits.
"""

__all__ = ['TransmonSimple']

import numpy as np

import qutip as qt


class TransmonSimple(object):
    """A class for representing transmons based on Duffing oscillator
    model."""

    def __init__(self, omega_q, alpha, nlev, omega_d=None, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.omega_q = omega_q  # The qubit main transition frequency.
        self.omega_d = omega_d  # Drive frequency for rotating frame stuff.
        self.alpha = alpha  # The qubit anharmonicity (omega_12 - omega_01).
        self.nlev = nlev  # The number of eigenstates in the qubit.
        self.units = units
        self.type = 'qubit'

    def __str__(self):
        s = ('A transmon qubit with omega_q = {} '.format(self.omega_q) + self.units
             + ' and  alpha = {} '.format(self.alpha) + self.units)
        return s

    @property
    def omega_q(self):
        return self._omega_q

    @omega_q.setter
    def omega_q(self, value):
        self._omega_q = value
        self._reset_cache()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value >= 0:
            raise Exception('Anharmonicity must be negative.')
        self._alpha = value
        self._reset_cache()

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('The number of levels must be positive.')
        self._nlev = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def a(self):
        """Annihilation operator."""
        return qt.destroy(self.nlev)

    def adag(self):
        """Creation operator."""
        return qt.create(self.nlev)

    def H(self):
        """Qubit Hamiltonian."""
        omega_q = self.omega_q
        alpha = self.alpha
        nlev = self.nlev
        H_qubit = np.zeros((nlev, nlev))
        for k in range(1, nlev):
            H_qubit[k, k] = k * omega_q + 0.5 * k * (k - 1) * alpha
        return qt.Qobj(H_qubit)

    def H_rotating(self):
        """Qubit Hamiltonian in the rotating frame."""
        a = self.a()
        return self.H() - self.omega_d * a.dag() * a

    def _eigenspectrum(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the LC basis."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H = self.H()
                self._eigvals = H.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H = self.H()
                self._eigvals, self._eigvecs = H.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None):
        """Eigenenergies of the qubit.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        return self._eigenspectrum()[0:nlev]

    def level(self, level_ind):
        """Energy of a single level of the qubit.

        Parameters
        ----------
        level_ind : int
            The qubit level starting from zero.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= self.nlev:
            raise Exception('The level is out of bounds')
        return self._eigenspectrum()[level_ind]

    def freq(self, level1, level2):
        """Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2` defined
            as the difference of energies. Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def eye(self):
        """Identity operator in the qubit eigenbasis.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        return qt.qeye(self.nlev)

    def a_ij(self, level1, level2):
        """The annihilation operator matrix element between two eigenstates.
        Parameters
        ----------
        level1, level2 : int
            The cavity levels.
        Returns
        -------
        complex
            The matrix element of the annihilation operator.
        """
        if (level1 < 0 or level1 > self.nlev or
                level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        return self.a().matrix_element(qt.basis(self.nlev, level1).dag(),
                                       qt.basis(self.nlev, level2))

    def adag_ij(self, level1, level2):
        """The creation operator matrix element between two eigenstates.
        Parameters
        ----------
        level1, level2 : int
            The cavity levels.
        Returns
        -------
        complex
            The matrix element of the annihilation operator.
        """
        if (level1 < 0 or level1 > self.nlev or
                level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        return self.adag().matrix_element(qt.basis(self.nlev, level1).dag(),
                                          qt.basis(self.nlev, level2))
