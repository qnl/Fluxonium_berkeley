# Author: Long Nguyen, 4/2021
###########################################################################
"""The SQUID class for representing a tunable qubit.
"""

__all__ = ['squid']

import numpy as np
import qutip as qt


class Squid(object):
    """A class for representing Superconducting QUantum Interference Devices"""

    def __init__(self, E_C, E_Jsum, d, phi_ext,
                 nlev=5, nlev_charge=20, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.E_C = E_C          # The charging energy.
        self.E_Jsum = E_Jsum    # The total Josephson energy, E_Jsum = E_J1 + E_J2
        self.d = d              # The normalized difference asymmetry, d = |E_J1 - E_J2|/E_Jsum
        self.phi_ext = phi_ext  # The externally induced phase shift
        # E_J = E_Jsum*cos(phi_ext/2)*sqrt(1+d^2tan^2(phi_ext/2))
        self.nlev = nlev        # The number of eigenstates in the qubit.
        self.nlev_charge = nlev_charge  # The number of charge states before diagonalization.
        self.units = units
        self.type = 'qubit'

    def __str__(self):
        s = ('A SQUID with E_C = {} '.format(self.E_C) + self.units
             + ', and E_Jsum = {} '.format(self.E_Jsum) + self.units
             + ', and d = {} '.format(self.d) + self.units
             + '. The external phase shift is phi_ext/pi = {}.'.format(
                    self.phi_ext / np.pi))
        return s

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Charging energy must be positive.')
        self._E_C = value
        self._reset_cache()

    @property
    def E_Jsum(self):
        return self._E_Jsum

    @E_Jsum.setter
    def E_Jsum(self, value):
        if value <= 0:
            print('*** Warning: Total Josephson energy is not positive. ***')
        self._E_Jsum = value
        self._reset_cache()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        if value < 0:
            raise Exception('Asymmetry must be >= 0')
        self._d = value
        self._reset_cache()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value):
        self._phi_ext = value
        self._reset_cache()

    @property
    def nlev_charge(self):
        return self._nlev_charge

    @nlev_charge.setter
    def nlev_charge(self, value):
        if value <= 0:
            raise Exception('The number of levels must be positive.')
        self._nlev_charge = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _n_charge(self):
        """Charge operator in the charge basis."""
        op = np.diag(np.arange(-self.nlev_charge, self.nlev_charge+1))
        return qt.Qobj(op)

    def _phi_charge(self):
        """Flux (phase) operator in the charge basis."""
        coeff = 1.0j/2
        op = coeff*(np.diag(np.ones(self.nlev_charge), 1) - np.diag(np.ones(self.nlev_charge), -1))
        return qt.Qobj(op)

    def _cosphi_charge(self):
        """cos(phase) operator in the charge basis."""
        op = np.diag(np.ones(2 * self.nlev_charge), 1) + np.diag(np.ones(2 * self.nlev_charge), -1)
        return qt.Qobj(op)

    def _hamiltonian_charge(self):
        """Qubit Hamiltonian in the LC basis."""
        E_C = self.E_C
        E_Jsum = self.E_Jsum
        d = self.d
        phi_e = self.phi_ext
        E_J = E_Jsum * np.cos(phi_e/2.0)*np.sqrt(1.0+d**2 * np.tan(phi_e/2.0)**2)
        cosphi = self._cosphi_charge()
        n = self._n_charge()

        return 4 * E_C * n ** 2 - E_J * cosphi

    def _eigenspectrum_charge(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the charge basis."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H_charge = self._hamiltonian_charge()
                self._eigvals = H_charge.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H_charge = self._hamiltonian_charge()
                self._eigvals, self._eigvecs = H_charge.eigenstates()
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
        if nlev < 1 or nlev > self.nlev_charge:
            raise Exception('`nlev` is out of bounds.')
        return self._eigenspectrum_charge()[0:nlev]

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
        if level_ind < 0 or level_ind >= self.nlev_charge:
            raise Exception('The level is out of bounds')
        return self._eigenspectrum_charge()[level_ind]

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

    def H(self, nlev=None):
        """Qubit Hamiltonian in its eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def eye(self, nlev=None):
        """Identity operator in the qubit eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_charge:
            raise Exception('`nlev` is out of bounds.')
        return qt.qeye(nlev)

    def phi(self, nlev=None):
        """Generalized-flux operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The flux operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_charge:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        phi_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                phi_op[ind1, ind2] = self._phi_charge().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(phi_op)

    def n(self, nlev=None):
        """Charge operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The charge operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_charge:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        n_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                n_op[ind1, ind2] = self._n_charge().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(n_op)

    def phi_ij(self, level1, level2):
        """The flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the flux operator.
        """
        if (level1 < 0 or level1 > self.nlev_charge
                or level2 < 0 or level2 > self.nlev_charge):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        return self._phi_charge().matrix_element(
            evecs[level1].dag(), evecs[level2])

    def n_ij(self, level1, level2):
        """The charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the charge operator.
        """
        if (level1 < 0 or level1 > self.nlev_charge
                or level2 < 0 or level2 > self.nlev_charge):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        return self._n_charge().matrix_element(evecs[level1].dag(), evecs[level2])