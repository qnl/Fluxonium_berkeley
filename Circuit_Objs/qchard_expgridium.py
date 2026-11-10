# Author: Thomas Ersevim, 2026
#########################################################################

'''Establishes the ExpGridium class for modeling the gridium qubit and for its use in qchard.

Is important that ultimately the relevantly coupled/readout variables are labeled n and phi exactly

'''

__all__ = ['ExpGridium', 'soft_ExpGridium_params', 'hard_ExpGridium_params', 'std_ExpGridium_operation_params', 'std_ExpGridium_sim_params']

import numpy as np
import qutip as qt

# Need some way to define phi1, phi2, phi3 as "the" coupled phi which is later used in the simulation. 

# Defines intrinsic circuit parameters for later ease of use
soft_ExpGridium_params = { # Like regime 'a' in Table S1 in https://arxiv.org/pdf/2509.14656
    'E_C': 0.5,
    'E_J': 5,
    'E_Lk': 1,
    'E_L': 1,
    'E_Js': 4,
    'E_Cs': 8,
}
hard_ExpGridium_params = { # Like regime 'd' in Table S1 in https://arxiv.org/pdf/2509.14656
    'E_C': 0.5,
    'E_J': 10,
    'E_Lk': 0.2,
    'E_L': 0.2,
    'E_Js': 4,
    'E_Cs': 8,
}
# Defines standard environmental operating conditions of the ExpGridium
std_ExpGridium_operation_params = {
    'ng': 0,
    'phi_ext': 0,
    'theta_ext': np.pi,
}
# Defines the standard simulation conditions of the ExpGridium
std_ExpGridium_sim_params = {
    'nlev': 7,
    'nlev_lc': 500,
    'units': 'GHz',
}

class ExpGridium(object):
    '''
    A class for representing the experimentally achievable, floating, gridium qubit

    Parameter regieme of interest: E_Cs > E_Js >> E_L and E_J >> E_C, E_Lk
    
    This is a fully detailed experimetal model, and is therefore the most expensive to run

    Should be a three mode model.
    '''

    def __init__(self, E_C, E_J, E_Lk, E_L, E_Js, E_Cs,
                 coupled_var:str='phi1',
                 ng=0, phi_ext=0, theta_ext=np.pi,
                 nlev_lc_kite = 100, nlev_lc_phi1 = 300, units='GHz'):
        self.E_C = E_C, # The globally confining capacitive energy (patterned pads in real circuit)
        self.E_J = E_J, # The individual EJ of each junction in the KITE (no asymmetry).
        self.E_Lk = E_Lk, # The inductance in each leg of the KITE
        self.E_L = E_L, # The globally confining inductance 
        self.E_Js = E_Js, # The josephson junction acting as a phase slip element
        self.E_Cs = E_Cs, # The capacitance inherent to the phase slip josephson junction
        self.coupled_var = coupled_var
        self.ng = ng, # The offset gate charge
        self.phi_ext = phi_ext, # The threaded flux (not through the KITE)
        self.theta_ext = theta_ext # The threaded flux through the KITE
        self.nlev_lc_kite = nlev_lc_kite # The number of lc states to apply to each kite's phi which should be symmetric (hence 1 variable)
        self.nlev_lc_phi1 = nlev_lc_phi1 # The number of lc states to apply to global phi
        self.units = units,
        self.type = 'qubit'
        # Could add any number of extra paracitic capacitance terms if so inclined

    def __str__(self):
        s = ('A multi-mode gridium qubit with E_C = {} '.format(self.E_C) + self.units
            + ', E_J = {} '.format(self.E_J) + self.units
            + ', E_Lk = {} '.format(self.E_Lk) + self.units
            + ', E_L = {} '.format(self.E_L) + self.units
            + ', E_Js = {} '.format(self.E_Js) + self.units
            + ', E_Cs = {} '.format(self.E_Cs) + self.units
            + '. The external phase shift is phi_ext/pi = {}. '.format(self.phi_ext/np.pi)
            + 'The phase shift through the KITE is theta_ext/pi = {}. '.format(self.theta_ext/np.pi)
        )
        return s

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Capacitive energy (C) must be greater than zero.')
        self._E_C = value
        self._reset_cache()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        if value <= 0:
            raise Warning('WARNING: Josephson energy (J) set to be negative')
        self._E_J = value
        self._reset_cache()

    @property
    def E_Lk(self):
        return self._E_Lk

    @E_Lk.setter
    def E_Lk(self, value):
        if value <= 0:
            raise Exception('Inductive energy (Lk) must be greater than zero.')
        self._E_Lk = value
        self._reset_cache()

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, value):
        if value <= 0:
            raise Exception('Inductive energy (L) must be greater than zero.')
        self._E_L = value
        self._reset_cache()

    @property
    def E_Js(self):
        return self._E_Js

    @E_Js.setter
    def E_Js(self, value):
        if value <= 0:
            raise Warning('WARNING: Josephson energy (Js) set to be negative')
        self._E_Js = value
        self._reset_cache()

    @property
    def E_Cs(self):
        return self._E_Cs

    @E_Cs.setter
    def E_Cs(self, value):
        if value <= 0:
            raise Exception('Capacitive energy (Cs) must be greater than zero.')
        self._E_Cs = value
        self._reset_cache()
        
    @property
    def coupled_var(self):
        return self._coupled_var
    
    @coupled_var.setter
    def coupled_var(self, value):
        if value not in ['phi1', 'phi2', 'phi3', 'n1', 'n2', 'n3']:
            raise Exception('Not a valid coupling variable')
        self._coupled_var = value
        self._reset_cache()

    @property
    def ng(self):
        return self._ng

    @ng.setter
    def ng(self, value):
        self._ng = value
        self._reset_cache()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value):
        self._phi_ext = value
        self._reset_cache()

    @property
    def theta_ext(self):
        return self._theta_ext

    @theta_ext.setter
    def theta_ext(self, value):
        self._theta_ext = value
        self._reset_cache()

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('Diagonalized energy level number must be greater than zero.')
        self._nlev = value
        self._reset_cache()

    @property
    def nlev_lc_kite(self):
        return self.nlev_lc_kite

    @nlev_lc_kite.setter
    def nlev_lc_kite(self, value):
        if value <= 0:
            raise Exception('LC energy level number must be greater than zero.')
        self.nlev_lc_kite = value
        self._reset_cache()

    @property
    def nlev_lc_phi1(self):
        return self._nlev_lc_phi1

    @nlev_lc_phi1.setter
    def nlev_lc_phi1(self, value):
        if value <= 0:
            raise Exception('LC energy level number must be greater than zero.')
        self._nlev_lc_phi1 = value
        self._reset_cache()

    def _reset_cache(self):
        '''Resets cached data that has already been calculated.'''
        self._eigvals = None
        self._eigvecs = None

    def _phi_lc(self):
        if self.coupled_var == 'phi1':
            return self._phi1_lc()
        elif self.coupled_var == 'phi2':
            return self._phi2_lc()
        elif self.coupled_var == 'phi3':
            return self._phi3_lc()
        else:
            return Warning('The specified coupling variable is not recognized')
        
    def _n_lc(self):
        if self.coupled_var == 'n1':
            return self._n1_lc()
        if self.coupled_var == 'n2':
            return self._n2_lc()
        if self.coupled_var == 'n3':
            return self._n3_lc()
        else:
            return Warning('The specified coupling variable is not recognized')

    # TODO: add the "boilerplate" for both n and phi variables which takes 
    # TODO: levels and eigen defs and such need to be hard coded to have the phi1 being the analysis variable, since I have redefined the .n and .phi
    
    def _hamiltonian_lc(self, form='explicit'):
        E_C = self.E_C
        E_J = self.E_J
        E_Lk = self.E_Lk
        E_L = self.E_L
        E_Js = self.E_Js
        E_Cs = self.E_Cs
        ng = self.ng
        phi_ext = self.phi_ext
        theta_ext = self.theta_ext
        phi1 = self._phi1_lc()
        phi2 = self._phi2_lc()
        phi3 = self._phi3_lc()
        n1 = self._n1_lc()
        n2 = self._n2_lc()
        n3 = self._n3_lc()

        if form=='explicit':
            return (
                4*E_Cs*(n1**2+n2**2) +
                4*E_CJ
            )

##### Copied from IdealGridium
'''
    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _b_lc(self):
        """Annihilation operator in the LC basis."""
        return qt.destroy(self.nlev_lc)

    def _phi_lc(self):
        """Flux (phase) operator in the LC basis."""
        return (8 * self.E_C / self.E_L) ** (0.25) * qt.position(self.nlev_lc) # Just a start for the bare resonator mode which is not close to diagonal here
    
    def _n_lc(self):
        """Charge operator in the LC basis."""
        return (self.E_L / (8 * self.E_C)) ** (0.25) * qt.momentum(self.nlev_lc) # Just a start for the bare resonator mode which is not close to diagonal here

    def cos_2pi_n(self):
        """A cos(2*pi*n) operator for use primarily in the hamiltonian"""
        n = self._n_lc()
        cos_2pi_n = 0.5*((-2j*np.pi*n).expm() + (2j*np.pi*n).expm()) # Exp form of cos(2pi*n)
        return cos_2pi_n

    def cos_2phi(self):
        """A cos(2phi) operator for use primarily in the hamiltonian"""
        phi = self._phi_lc()
        cos_2phi = phi.cosm()*phi.cosm() - phi.sinm()*phi.sinm() # Double angle formula
        return cos_2phi

    def _hamiltonian_lc(self):
        """Qubit Hamiltonian in the LC basis."""
        E_C = self.E_C
        E_L = self.E_L
        E_s = self.E_s
        E_2J = self.E_2J
        ng = self.ng
        phi = self._phi_lc()
        n = self._n_lc()
        net_phi = phi + self.phi_ext
        return E_C*(n+ng)**2 + 0.5*E_L*net_phi**2 - E_s*self.cos_2pi_n() + E_2J*self.cos_2phi() # Eq 2 from https://arxiv.org/pdf/2509.14656 for coefficients
    

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
        if nlev < 1 or nlev > self.nlev_lc:
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
        if level_index < 0 or level_index >= self.nlev_lc:
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
        if nlev < 1 or nlev > self.nlev_lc:
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
        if (level1 < 0 or level1 > self.nlev_lc
                or level2 < 0 or level2 > self.nlev_lc):
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
        if (level1 < 0 or level1 > self.nlev_lc
                or level2 < 0 or level2 > self.nlev_lc):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._n_lc().matrix_element(evecs[level1].dag(), evecs[level2])

    def transition_energies(self, lower_level=0, nlev=None) -> np.ndarray:
        """From provided lower level, finds the zeroed transition energy to the upper levels"""
        if nlev is None:
            nlev = self.nlev
        eigvals = self._eigenspectrum_lc()[lower_level:nlev]
        transitions = eigvals - eigvals[0]
        return transitions
'''

    # TODO: Implement eigenvector plotting
    # TODO: Implement spectrum plotting
    # TODO: Implement global plotting standard
    # TODO: Implement .gnd_transitions() method