# This file is part of QHard: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
###########################################################################
"""The CoupledObjects class for representing a system of coupled
superconducting qubits and resonators.
"""

__all__ = ['CoupledObjects']

import numpy as np
import itertools

import qutip as qt


class CoupledObjects(object):
    """A class for representing a system of coupled qubits and resonators.

    Unless specified, work and all returns are in the basis that is
    the tensor product of eigenbases of individual objects.

    Parameters
    ----------
    *args
        Individual quantum objects of qubit and cavity types and couplings
        between them. A coupling between two objects obj1 and obj2 is
        represented by a list [obj1, obj2, E_int, coupling_type], where
        E_int is the coupling constant and coupling_type is the
        string description of the coupling ('flux', 'charge' between qubits)
        or two operators of two objects (not implemented yet).
    Examples
    --------
    A system of two inductively coupled fluxonium qubits.

    >>> qubit1 = fluxonium.Fluxonium(E_L1, E_C1, E_J1)
    >>> qubit2 = fluxonium.Fluxonium(E_L2, E_C2, E_J2)
    >>> system = coupobj.CoupledObjects(
    ...         qubit1, qubit2, [qubit1, qubit2, E_int, 'flux'])

    A system of two qubits with both capacitive and inductive couplings.

    >>> system = coupobj.CoupledObjects(qubit1, qubit2,
    ...                                 [qubit1, qubit2, E_int_C, 'charge'],
    ...                                 [qubit1, qubit2, E_int_L, 'flux'])

    A microwave resonator coupled to a qubit with g*(a + a^+)*n interaciton.

    >>> resonator = resonator = cavity.Cavity(omega=omega_c, nlev=nlev_cav)
    >>> qubit = fluxonium.Fluxonium(E_L, E_C, E_J)
    >>> system = coupobj.CoupledObjects(resonator, qubit,
    ...                                 [resonator, qubit, g, 'JC-charge'])
    """

    def __init__(self, *args):
        self._objects = []
        self._couplings = []
        for arg in args:
            if isinstance(arg, list):
                self._couplings.append(arg)
            else:
                self._check_obj(arg)
                self._objects.append(arg)
        self._nobj = len(self._objects)
        self._ncoupl = len(self._couplings)
        self._reset_cache()

    def _check_obj(self, obj):
        if (not hasattr(obj, 'type') or obj.type != 'qubit'
                and obj.type != 'cavity'):
            raise Exception('The object parameter is unrecognized.')

    def _reset_cache(self):
        """Resets cached data."""
        self._eigvals = None
        self._eigvecs = None
        self._eigvals_nonint = None
        self._eigvecs_nonint = None
        self._state_labels = None
        self._nlev = np.prod([obj.nlev for obj in self._objects])

    def reset(self):
        """Manual "reset" of the cached data in a class instance.

        It is necessary to call this method after a change in any
        parameter that is directly or indirectly used in the class instance.
        For example, after any of the attributes of an underlying qubit
        object has been changed, certain cached data such as eigenvalues
        have to be re-calculated.

        May be depreciated in the future if a proper
        """
        self._reset_cache()

    def promote_op(self, obj, operator):
        """Rewrites the operator in the tensor-product Hilbert space.

        Rewrites the operator `operator` written in the Hilbert space
        of an individual quantum object `obj` in the tensor-product space
        of the composite system. The value of `obj` can be the actual
        object (qubit or cavity) or the sequential number of the object
        in the class instance initialization.
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        if obj not in self._objects:
            raise Exception('The object parameter is unrecognized.')
        if operator.dims != obj.H().dims:
            raise Exception(
                'The operator does not agree with its underlying object.')
        obj_index = self._objects.index(obj)
        # Add identity operators from the left and from the right.
        for k in range(obj_index - 1, -1, -1):
            operator = qt.tensor(qt.qeye(self._objects[k].nlev), operator)
        for k in range(obj_index + 1, self._nobj):
            operator = qt.tensor(operator, qt.qeye(self._objects[k].nlev))
        return operator

    def phi(self, obj):
        """The flux operator for a qubit in the tensor-product space."""
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.phi())

    def n(self, obj):
        """The charge operator for a qubit in the tensor-product space."""
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.n())

    def a(self, obj):
        """The object annihilation operator in the tensor-product space."""
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.a())

    def H_0(self):
        """The Hamiltonian of uncoupled quantum objects."""
        H_0 = 0
        for obj in self._objects:
            H_0 += self.promote_op(obj, obj.H())
        return H_0

    def V(self):
        """The coupling part of the Hamiltonian."""
        V = 0
        for coupling_term in self._couplings:
            obj1 = coupling_term[0]
            obj2 = coupling_term[1]
            E_int = coupling_term[2]
            if isinstance(coupling_term[3], str):
                if coupling_term[3] == 'flux':
                    # Qubit-qubit flux coupling.
                    op1 = self.promote_op(obj1, obj1.phi())
                    op2 = self.promote_op(obj2, obj2.phi())
                elif coupling_term[3] == 'charge':
                    # Qubit-qubit charge coupling.
                    op1 = self.promote_op(obj1, obj1.n())
                    op2 = self.promote_op(obj2, obj2.n())
                elif coupling_term[3] == 'JC-charge':
                    # Qubit-resonator coupling via charge.
                    if obj1.type == 'cavity':
                        op1 = obj1.a() + obj1.adag()
                        op2 = obj2.n()
                    else:
                        op1 = obj1.n()
                        op2 = obj2.a() + obj2.adag()
                    op1 = self.promote_op(obj1, op1)
                    op2 = self.promote_op(obj2, op2)
                elif coupling_term[3] == 'JC-rwa':
                    # Qubit-resonator coupling in the RWA.
                    op1 = self.promote_op(obj1, obj1.a())
                    op2 = self.promote_op(obj2, obj2.a())
                elif coupling_term[3] == 'transmon':
                    # Qubit-resonator coupling in the RWA.
                    op1 = self.promote_op(obj1, obj1.a()+obj1.a().dag())
                    op2 = self.promote_op(obj2, obj2.a()+obj2.a().dag())
                elif coupling_term[3] == 'coupler-charge':
                    #Qubit coupled to a coupler
                    op1 = self.promote_op(obj1, obj1.n())
                    op2 = self.promote_op(obj2, obj2.n())
                else:
                    raise Exception(
                        'This type of coupling is not implemented yet.')
                if coupling_term[3] == 'JC-rwa':
                    V += E_int * (op1 * op2.dag() + op1.dag() * op2)
                elif coupling_term[3] == 'flux':
                    V -= E_int * op1 * op2
                else:
                    V += E_int * op1 * op2
            else:
                raise Exception(
                    'Not a string')
        return V

    def H(self):
        """The Hamiltonian of the coupled system."""
        return self.H_0() + self.V()

    def _eigenspectrum(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates of the coupled system."""
        sparse = False
        if not eigvecs_flag:
            if self._eigvals is None:
                self._eigvals = self.H().eigenenergies(sparse=sparse)
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                self._eigvals, self._eigvecs = self.H().eigenstates(
                    sparse=sparse)
            return self._eigvals, self._eigvecs

    def _spectrum_nonint(self, labels_flag=False, eigvecs_flag=False):
        """Spectrum in the absence of coupling.

        Returns
        -------
        1d array
            Energies in ascending order.
        2d array if `labels_flag` is True
            State labels in the same order as energies.
            1st column - tuple-like labels such as (0, 1)
                for the state '01' of a two-qubit system
            2nd column - string labels such as '01'
        1d array of :class:`qutip.Qobj` if `eigvecs_flag` is True
            Eigenvectors.
        """
        objects = self._objects
        if self._eigvals_nonint is None:
            nobj = self._nobj
            states = np.empty((self._nlev, 4), dtype=object)
            # To make the loop over all possible combinations of indices.
            iterable = [range(obj.nlev) for obj in objects]
            ind = 0
            for state_tuple in itertools.product(*iterable):
                # The sum of energies.
                states[ind, 0] = sum(
                    objects[k].level(state_tuple[k]) for k in range(nobj))
                states[ind, 1] = state_tuple
                # The string label such as '01' for a two-qubit system.
                states[ind, 2] = ''.join(str(k) for k in state_tuple)
                eigenvector = qt.basis(objects[0].nlev, state_tuple[0])
                for k in range(1, nobj):
                    eigenvector = qt.tensor(eigenvector, qt.basis(
                        objects[k].nlev, state_tuple[k]))
                states[ind, 3] = eigenvector
                ind += 1
            states = states[np.argsort(states[:, 0])]
            self._eigvals_nonint = states[:, 0]
            self._state_labels = states[:, 1:3]
            self._eigvecs_nonint = states[:, 3]
        return_objects = [self._eigvals_nonint]
        if labels_flag:
            return_objects.append(self._state_labels)
        if eigvecs_flag:
            return_objects.append(self._eigvecs_nonint)
        if len(return_objects) > 1:
            return tuple(return_objects)
        else:
            return return_objects[0]

    def level_label(self, label, label_format='int'):
        """Converts a label of an energy level into a different format.

        Possible formats of a label:
            int - the sequential index of the level in ascending order
                of energies
            tuple - description of the corresponding noninteracting
                state consisting of indices describing the states of
                underlying objects such as (0, 1) for the state '01'
                of a two-qubit system
            str - string description of the corresponding noninteracting
                state such as '01' for a two-qubit system
        For 'tuple' and 'str' formats, the corresponding noninteracting
        states are chosen assuming the same order of energies, i.e.,
        ignoring the level crossings.

        Parameters
        ----------
        label : int, tuple, or str
        label_format : str (optional)
            Format of the return label ('int', 'tuple', or 'str')

        Returns
        -------
        int, tuple, or str

        Example
        -------
        The ground state of the system with three objects.

        >>> level_label('000')  # returns 0
        >>> level_label(0, label_format='str') # returns '000'
        """
        _, labels = self._spectrum_nonint(labels_flag=True)
        for k in range(len(labels)):
            if label == labels[k, 0] or label == labels[k, 1] or label == k:
                index = k
                break
            if k == len(labels) - 1:
                raise Exception('Unrecognized state label.')
        if label_format == 'int':
            return index
        elif label_format == 'tuple':
            return labels[index, 0]
        elif label_format == 'str':
            return labels[index, 1]
        else:
            raise Exception('Unrecognized format for the return label.')
            return None

    def levels(self, nlev=None, interaction='on', eigvecs=False):
        """Spectrum of the system.

        Parameters
        ----------
        nlev : int, optional
            The number of levels to return. Default is all the levels.
        interaction : 'on' or 'off', optional
            Return energy levels with or without coupling.
        eigvecs : bool, optional
            If True, return eigenvectors in addition to eigenvalues.
        Returns
        -------
        1d ndarray
            Energies in ascending order.
        1d array of :class:`qutip.Qobj`
            Eigenvectors corresponding to `eigenvalues` (if `eigvecs` is True).
        """
        if interaction == 'on':
            spectrum_func = self._eigenspectrum
        elif interaction == 'off':
            spectrum_func = self._spectrum_nonint
        else:
            raise Exception('Unrecognized interaction option.')
        if nlev is None:
            return spectrum_func(eigvecs_flag=eigvecs)
        else:
            if eigvecs:
                return_tuple = spectrum_func(eigvecs_flag=True)
                return return_tuple[0][:nlev], return_tuple[1][:nlev]
            else:
                return spectrum_func()[:nlev]

    def level(self, label, interaction='on', eigvec=False):
        """Energy and eigenvector of a single level.

        Parameters
        ----------
        label : int, tuple, str
            Label of the level: sequential index of the level or its tuple
            or string description in terms of states of uncoupled objects
            such as (0, 1) or '01' for a two-qubit system.
        interaction : 'on' or 'off', optional
            Return eigenstate with or without coupling.
        eigvec : bool, optional
            If True, return eigenvector in addition to eigenvalue.

        Returns
        -------
        float
            Energy of the level.
        :class:`qutip.Qobj`
            Eigenvector if `eigvec` is True.

        Examples
        --------
        >> level('01')  # Energy of the level labeled as '01'.
        """
        level_index = self.level_label(label)
        if eigvec:
            return_tuple = self.levels(interaction=interaction, eigvecs=True)
            return return_tuple[0][level_index], return_tuple[1][level_index]
        else:
            return self.levels(interaction=interaction)[level_index]

    def levels_nonint(self, nlev=None, eigvecs=False):
        """A shortcut for levels(interaction='off')."""
        return self.levels(nlev=nlev, interaction='off', eigvecs=eigvecs)

    def level_nonint(self, label, eigvec=False):
        """A shortcut for level(label, interaction='off')."""
        return self.level(label, interaction='off', eigvec=eigvec)

    def freq(self, level1, level2, interaction='on'):
        """Transition frequency defined as the energy difference."""
        return (self.level(level2, interaction=interaction)
                - self.level(level1, interaction=interaction))

    def freq_nonint(self, level1, level2):
        """A shortcut for freq(interaction='off')."""
        return self.freq(level1, level2, interaction='off')

    def eigvecs(self, nlev=None, interaction='on'):
        """A shortcut to get eigenvectors via levels(eigvecs=True).

        Returns
        -------
        1d array of :class:`qutip.Qobj`
            Eigenvectors.
        """
        _, evecs = self.levels(
            nlev=nlev, interaction=interaction, eigvecs=True)
        return evecs

    def eigvec(self, label, interaction='on'):
        """A shortcut to get an eigenvector via level(eigvec=True).

        Returns
        -------
        :class:`qutip.Qobj`
            Eigenvector.
        """
        _, evec = self.level(label, interaction=interaction, eigvec=True)
        return evec

    def eigvecs_nonint(self, nlev=None):
        """A shortcut for eigvecs(interaction='off')."""
        return self.eigvecs(nlev=nlev, interaction='off')

    def eigvec_nonint(self, label):
        """A shortcut for eigvec(label, interaction='off')."""
        return self.eigvec(label, interaction='off')

    def matr_el(self, obj, operator, level1, level2, interaction='on'):
        """Matrix element of an operator for a specific object."""
        operator = self.promote_op(obj, operator)
        evec1 = self.eigvec(level1, interaction=interaction)
        evec2 = self.eigvec(level2, interaction=interaction)
        return operator.matrix_element(evec1.dag(), evec2)

    def matr_el_nonint(self, obj, operator, level1, level2):
        """A shortcut for matr_el(interaction='off')."""
        return self.matr_el(obj, operator, level1, level2, interaction='off')

    def phi_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the flux operator."""
        return self.matr_el(
            obj, obj.phi(), level1, level2, interaction=interaction)

    def phi_ij_nonint(self, obj, level1, level2):
        """A shortcut for phi_ij(interaction='off')."""
        return self.phi_ij(obj, level1, level2, interaction='off')

    def n_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the charge operator."""
        return self.matr_el(
            obj, obj.n(), level1, level2, interaction=interaction)

    def n_ij_nonint(self, obj, level1, level2):
        """A shortcut for n_ij(interaction='off')."""
        return self.n_ij(obj, level1, level2, interaciton='off')

    def a_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the HO annihilation operator."""
        return self.matr_el(
            obj, obj.a(), level1, level2, interaction=interaction)

    def adag_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the HO creation operator."""
        return self.matr_el(
            obj, obj.adag(), level1, level2, interaction=interaction)

    def projection(self, level_label, interaction='on'):
        """Projection operator for a system level."""
        psi = self.eigvec(level_label, interaction=interaction)
        return psi * psi.dag()

    def projection_nonint(self, level_label):
        """A shortcut for projection(interaction='off')."""
        return self.projection(level_label, interaction='off')