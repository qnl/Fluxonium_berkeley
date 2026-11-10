# This file is part of QHard: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
# Author: Thomas Ersevim, 2026
###########################################################################
"""Classes for representing transmon qubits."""

# TODO: Implement eigenvector plotting

__all__ = ['TransmonSimple', 'Transmon', 'SCQTransmon', 'std_transmon_params', 'std_transmon_sim_params', 'transmon_creation_from_01']

import numpy as np
import scipy
import dill

import qutip as qt
import scqubits as scq

from Circuit_Objs.qchard_abstractobj import AbstractQubit


std_transmon_params = {
    'E_J': 15,
    'E_C': 0.3,
}

std_transmon_sim_params = {
    'ng': 0,
    'nlev': 5,
    'nlev_lc': 15,
}


def transmon_creation_from_01(linear_freq, EJ_EC_ratio=50):
    '''Creates a transmon with a 01 splitting of the given frequency given also a EJ/EC ratio'''
    h = 1
    E_C = linear_freq*h/(np.sqrt(8*EJ_EC_ratio)-1)
    E_J = EJ_EC_ratio*E_C
    return Transmon(E_C=E_C, E_J=E_J)


class SCQTransmon(object):
    '''A wrapper class for scqubits' form of the transmon'''

    name = 'Transmon'

    def __init__(self, E_C, E_J,
                 ng=0, nlev=5, nlev_lc=15, units='GHz'):
        self._E_C = E_C
        self._E_J = E_J
        self._ng = ng
        self._nlev = nlev
        self._nlev_cutoff = nlev_lc
        self.units = units
        self.type = 'qubit'
        self.scq_transmon = self._recreate_transmon()
    
    def __str__(self):
        s = ('A transmon qubit with E_C = {} '.format(self.E_C) + self.units
             + ', E_J = {} '.format(self.E_J) + self.units)
        return s

    def _save_str(self) -> str:
        s = ('scq_t_'
             + 'EC{}'.format(self.E_C)
             + 'EJ{}'.format(self.E_J)
             + 'nlev{}'.format(self.nlev)
             + 'nlevlc{}'.format(self.nlev_cutoff))
        return s
 
    def _recreate_transmon(self):
        '''
        The place where the SC qubits implementation of the transmon occurs. This gets run every
        time there is a change to any of the internal @property attribtues of the SCQTransmon
        object.
        '''
        self.scq_transmon = scq.Transmon(EC=self.E_C, EJ=self.E_J, ng=self.ng, ncut=self.nlev_cutoff, truncated_dim=self.nlev)
        return self.scq_transmon

    def _scale_E_params(self, scaling):
        self.E_C = self.E_C*scaling
        self.E_J = self.E_J*scaling
        return

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Charging energy must be positive')
        self._E_C = value
        self._recreate_transmon()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        if value <= 0:
            print('*** Warning: Josephson energy is not positive. ***')
        self._E_J = value
        self._recreate_transmon()

    @property
    def ng(self):
        return self._ng

    @ng.setter
    def ng(self, value):
        self._ng = value
        self._recreate_transmon()

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('The number of real levels must be positive.')
        self._nlev = value
        self._recreate_transmon()

    @property
    def nlev_cutoff(self):
        return self._nlev_cutoff

    @nlev_cutoff.setter
    def nlev_cutoff(self, value):
        if value <= 0:
            raise Exception('The number of lc levels must be positive.')
        self._nlev_cutoff = value
        self._recreate_transmon()

    def levels(self, nlev=None, eigvecs=False):
        '''
        The scqubits implementation simply works entirely in the charge basis. It represents the
        hamiltonian in the charge basis and represents the cos(phi) in the charge basis as a sum
        of transition operators between n and n+1 and the H.C.
        '''
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        if eigvecs:
            return_tuple = self.scq_transmon.eigensys(evals_count=self.nlev, return_spectrumdata=False)
            return return_tuple[0], return_tuple[1]
        else:
            eigenvalues = self.scq_transmon.eigenvals(evals_count=self.nlev, return_spectrumdata=False)
            return eigenvalues

    def level(self, level_index, eigvecs=False):
        '''
        Returns the eigenenergies and optionally the eigenvectors for the level_index specified
        '''
        if level_index < 0 or level_index >= self.nlev:
            raise Exception('The level is out of bounds')
        if eigvecs:
            return_tuple = self.scq_transmon.eigensys(evals_count=self.nlev, return_spectrumdata=False)
            return return_tuple[0][level_index], return_tuple[1][level_index]
        else:
            return self.levels()[level_index]

    def eigvec(self, level_index):
        _, evec = self.level(level_index=level_index, eigvecs=True)
        return evec

    def freq(self, level1, level2):
        return self.level(level2) - self.level(level1)
    
    def H(self, nlev=None):
        return qt.Qobj(self.scq_transmon.hamiltonian(energy_esys=True))
    
    def eye(self, nlev=None):
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds')
        return qt.qeye(nlev)
    
    def n(self, nlev=None):
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        n_op = self.scq_transmon.n_operator(energy_esys=True)
        return qt.Qobj(n_op)

    def phi(self, nlev=None): # Kind of doomed I think to this being not a hermitian phi :(
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        exp_iphi_op = self.scq_transmon.exp_i_phi_operator(energy_esys=True)
        exp_iphi_op = np.astype(exp_iphi_op, complex)
        phi_op = scipy.linalg.logm(exp_iphi_op)/(1j)
        return qt.Qobj(phi_op)
    
    def phi_ij(self, level1, level2):
        if (level1 < 0 or level1 > self.nlev or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        evecs = self.eigvec()
        return self.phi().matrix_elements(evecs[level1].dag(), evecs[level2])
    
    def n_ij(self, level1, level2):
        if (level1 < 0 or level1 > self.nlev or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        evecs = self.eigvec()
        return self.n().matrix_element(evecs[level1].dag(), evecs[level2])

    def transition_energies(self, lower_level=0, nlev=None) -> np.ndarray:
        '''From provided lower level, finds the zeroed transition energy to the upper levels'''
        if nlev is None:
            nlev = self.nlev
        eigvals = self.levels()[lower_level:nlev]
        transitions = eigvals - eigvals[0]
        return transitions

class Transmon(object):
    """
    A class for representing superconducting transmon qubits. Follows same construction 
    Architecture as the other qubit classes, though suffers from it being a poor approximation due
    to the compact nature of phi in this circuit while using creation and anhillation operators.
    NOTE: Therefore, the use of this class is discouraged.
    """

    def __init__(self, E_C, E_J,
                 n_g=0, nlev=5, nlev_lc=50, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.E_C = E_C  # The charging energy.
        self.E_J = E_J # The Josephson Energy
        self.n_g = n_g # Gate charge (parity of capacitor)
        self.nlev = nlev # Number of states to show/solve for
        self.nlev_lc = nlev_lc # LC basis cutoff size
        self.units = units
        self.type = 'qubit'

    def __str__(self):
        s = ('A transmon qubit with E_C = {} '.format(self.E_C) + self.units
             + ', E_J = {} '.format(self.E_J) + self.units)
        return s

    def _save_str(self) -> str:
        s = ('EC{}'.format(self.E_C)
             + 'EJ{}'.format(self.E_J)
             + 'nlev{}'.format(self.nlev)
             + 'nlevlc{}'.format(self.nlev_lc))
        return s
 
    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Charging energy must be positive')
        self._E_C = value
        self._reset_cache()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        if value <= 0:
            print('*** Warning: Josephson energy is not positive. ***')
        self._E_J = value
        self._reset_cache()

    @property
    def ng(self):
        return self._ng

    @ng.setter
    def ng(self, value):
        self._ng = value
        self._reset_cache()

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('The number of real levels must be positive.')
        self._nlev = value
        self._reset_cache()

    @property
    def nlev_lc(self):
        return self._nlev_lc

    @nlev_lc.setter
    def nlev_lc(self, value):
        if value <= 0:
            raise Exception('The number of lc levels must be positive.')
        self._nlev_lc = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _b_lc(self):
        """Annihilation operator in the LC basis."""
        return qt.destroy(self.nlev)

    def _phi_lc(self):
        """Flux (phase) operator in the LC basis."""
        return (2 * self.E_C / (self.E_J*2)) ** (0.25) * 2**(0.5) * qt.position(self.nlev_lc) #E_J ~ 1/2*E_L so coefficients get augmented

    def _n_lc(self):
        """Charge operator in the LC basis."""
        return ((self.E_J*2) / (32 * self.E_C)) ** (0.25) * 2**(0.5) * qt.momentum(self.nlev_lc) #E_J ~ 1/2*E_L so coefficients get augmented

    def _hamiltonian_lc(self):
        """Qubit Hamiltonian in the LC basis."""
        E_C = self.E_C
        E_J = self.E_J
        n = self._n_lc()
        phi = self._phi_lc()
        return 4*E_C*(n-self.n_g)**2 - E_J*phi.cosm()

    def _eigenspectrum_lc(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the LC basis."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals = H_lc.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals, self._eigvecs = H_lc.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None, eigvecs = False):
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
        if eigvecs:
            return_tuple = self._eigenspectrum_lc(eigvecs_flag=True)
            return return_tuple[0][:nlev], return_tuple[1][:nlev]
        else:
            return self._eigenspectrum_lc()[:nlev]

    def level(self, level_index, eigvecs=False):
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
        if level_index < 0 or level_index >= self.nlev:
            raise Exception('The level is out of bounds')
        if eigvecs:
            return_tuple = self.levels(eigvecs = True)
            return return_tuple[0][level_index], return_tuple[1][level_index]
        else:
            return self.levels()[level_index]

    def eigvec(self, level_index):
        """A shortcut to get an eigenvector via level(eigvec=True).

        Returns
        -------
        :class:`qutip.Qobj`
            Eigenvector.
        """
        _, evec = self.level(level_index=level_index, eigvecs=True)
        return evec

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
        if nlev < 1 or nlev > self.nlev:
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
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        phi_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                phi_op[ind1, ind2] = self._phi_lc().matrix_element(
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
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        n_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                n_op[ind1, ind2] = self._n_lc().matrix_element(
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
        if (level1 < 0 or level1 > self.nlev
                or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._phi_lc().matrix_element(
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
        if (level1 < 0 or level1 > self.nlev
                or level2 < 0 or level2 > self.nlev):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._n_lc().matrix_element(evecs[level1].dag(), evecs[level2])

    def transition_energies(self, lower_level=0, nlev=None) -> np.ndarray:
        '''From provided lower level, finds the zeroed transition energy to the upper levels'''
        if nlev is None:
            nlev = self.nlev
        eigvals = self._eigenspectrum_lc()[lower_level:nlev]
        transitions = eigvals - eigvals[0]
        return transitions

    def save_obj(self, dir:str):
        with open(dir+self._save_str(), 'wb') as f:
            dill.dump(self, f)

class TransmonSimple(object):
    '''
    A class for representing transmons based on Duffing oscillator model. Original transmon used
    in this package. Legacy.
    '''

    def __init__(self, omega_q, alpha, nlev, nlev_lc, omega_d=None, units='GHz'):
        self.omega_q = omega_q  # The qubit main transition frequency.
        self.omega_d = omega_d  # Drive frequency for rotating frame stuff.
        self.alpha = alpha  # The qubit anharmonicity (omega_12 - omega_01).
        self.nlev = nlev  # The number of eigenstates in the qubit.
        self.nlev_lc = nlev_lc # The number of harmonic oscilator states
        self.units = units
        self.type = 'qubit'
        self.E_C = -self.alpha
        self.E_J = (self.omega_q*self.E_C)**2/(8*self.E_C)

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

    @property
    def nlev_lc(self):
        return self._nlev_lc
    
    @nlev_lc.setter
    def nlev_lc(self, value):
        if value < self.nlev:
            raise Exception('The number of harmonic oscillator states must be > nlev')
        self._nlev_lc = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _phi_lc(self):
        """Flux (phase) operator in the LC basis."""
        return (2 * self.E_C / (self.E_J*2)) ** (0.25) * 2**(0.5) * qt.position(self.nlev_lc) #E_J ~ 1/2*E_L so coefficients get augmented

    def _n_lc(self):
        """Charge operator in the LC basis."""
        return ((self.E_J*2) / (32 * self.E_C)) ** (0.25) * 2**(0.5) * qt.momentum(self.nlev_lc) #E_J ~ 1/2*E_L so coefficients get augmented

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

    def _eigenspectrum_lc(self):
        pass

    _eigenspectrum_lc = _eigenspectrum

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
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        phi_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                phi_op[ind1, ind2] = self._phi_lc().matrix_element(
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
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        n_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                n_op[ind1, ind2] = self._n_lc().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(n_op)

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