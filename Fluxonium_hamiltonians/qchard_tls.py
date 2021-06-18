"""The TLS class for representing generic two-level systems.
"""

__all__ = ['TLS']

import numpy as np

import qutip as qt


class TLS(object):
    """A class for representing two-level system."""

    def __init__(self, omega, nlev=20, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.omega = omega  # The tls frequency.
        self.units = units
        self.type = 'tls'

    def __str__(self):
        s = ('A tls with omega/2*pi = {} '.format(self.omega)
             + self.units)
        return s

    def level(self, level_ind):
        """Energy of a single level of the tls.
        Parameters
        ----------
        level_ind : int
            The level index starting from zero.
        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= 2:
            raise Exception('The level is out of bounds')
        return level_ind * self.omega

    def levels(self):
        """Eigenenergies of the tls.
        Parameters
        ----------
        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        energies = [k * self.omega for k in range(2)]
        return np.array(energies)

    def freq(self, level1, level2):
        """Transition energy/frequency
        Parameters
        ----------
        level1, level2 : int
            The indices of cavity levels.
        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2` defined
            as the difference of energies. Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def H(self):
        """The tls Hamiltonian in its eigenbasis.
        Parameters
        ----------
        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels()))

    def eye(self):
        """Identity operator in the tls eigenbasis.
        Parameters
        ----------
        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        return qt.qeye(2)

    def a(self, nlev=None):
        """Cavity annihilation operator.
        Parameters
        ----------
            The number of cavity eigenstates if different from `self.nlev`.
        Returns
        -------
        :class:`qutip.Qobj`
            The annihilation operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        return qt.destroy(nlev)

    def adag(self, nlev=None):
        """Cavity creation operator.
        Parameters
        ----------
            The number of cavity eigenstates if different from `self.nlev`.
        Returns
        -------
        :class:`qutip.Qobj`
            The creation operator.
        """
        return self.a(nlev=nlev).dag()

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