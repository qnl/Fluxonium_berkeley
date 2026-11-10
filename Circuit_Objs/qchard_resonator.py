# This file is part of QHard: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
###########################################################################
"""The Cavity class for representing microwave resonators.
"""

__all__ = ['Cavity']

import numpy as np

import qutip as qt


class Cavity(object):
    """A class for representing microwave resonators."""

    def __init__(self, omega, nlev=20, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.omega = omega  # The resonator frequency.
        self.nlev = nlev  # The number of levels in the resonator.
        self.units = units
        self.type = 'cavity'

    def __str__(self):
        s = ('A resonator with omega/2*pi = {} '.format(self.omega)
             + self.units)
        return s

    def level(self, level_ind):
        """Energy of a single level of the cavity.
        Parameters
        ----------
        level_ind : int
            The level index starting from zero.
        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= self.nlev:
            raise Exception('The level is out of bounds')
        return level_ind * self.omega

    def levels(self, nlev=None):
        """Eigenenergies of the cavity.
        Parameters
        ----------
        nlev : int, optional
            The number of cavity eigenstates if different from `self.nlev`.
        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        energies = [k * self.omega for k in range(nlev)]
        return np.array(energies)

    def freq(self, level1, level2):
        """Transition energy/frequency between two cavity levels.
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

    def H(self, nlev=None):
        """The cavity Hamiltonian in its eigenbasis.
        Parameters
        ----------
        nlev : int, optional
            The number of cavity eigenstates if different from `self.nlev`.
        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def eye(self, nlev=None):
        """Identity operator in the cavity eigenbasis.
        Parameters
        ----------
        nlev : int, optional
            The size of the Hilbert space if different from `self.nlev`.
        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        return qt.qeye(nlev)

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