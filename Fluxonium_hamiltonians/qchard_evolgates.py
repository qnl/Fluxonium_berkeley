import sys

sys.dont_write_bytecode = True

import numpy as np
import scipy.special
import qutip as qt


def H_drive_coeff_gate(t, args):
    """
    The time-dependent coefficient of the drive term for the qutip
    representation of time-dependent Hamiltonians.

    Example: H = [H_nodrive, [H_drive, H_drive_coeff_gate]]

    H_drive_coeff_gate = xi_x(t) cos(wt) + xi_y(t) sin(wt)
    Normalization: \int xi(t') dt'= 2\pi for 0 < t' < T_gate
    If DRAG == True: xi_y(t) = alpha * d xi_x / dt,
    else: xi_y = 0
    If SYMM == True: xi_x(t) -> xi_x(t) + beta * d^2 xi_x / dt^2
    """
    nu_d = args['omega_d']
    two_pi_t = 2 * np.pi * t
    T_gate = args['T_gate']
    if 'DRAG' in args and args['DRAG']:
        alpha = args['DRAG_coefficient']
    else:
        alpha = 0
    if 'SYMM' in args and args['SYMM']:
        beta = args['SYMM_coefficient']
    else:
        beta = 0
    if 'shape' not in args or args['shape'] == 'square':
        xi_x = 2 * np.pi / T_gate
        xi_y = 0
    elif args['shape'] == 'cos':
        xi_x = (2 * np.pi / T_gate) * (1 - np.cos(two_pi_t / T_gate))
        xi_x += beta * (2 * np.pi / T_gate) ** 3 * np.cos(two_pi_t / T_gate)
        xi_y = 4 * alpha * np.pi ** 2 / T_gate ** 2 * np.sin(two_pi_t / T_gate)
    elif args['shape'] == 'gauss':
        sigma = args['sigma'] * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 2 * np.pi / integral_value
        xi_x = coeff * (np.exp(- 0.5 * ((t - T_gate / 2) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x += (beta * coeff * (-1 / sigma ** 2 + ((t - T_gate / 2) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_gate / 2) / sigma) ** 2))
        xi_y = (alpha * coeff * (- (t - T_gate / 2) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_gate / 2) / sigma) ** 2))
    else:
        raise Exception('Urecognized shape.')
    return (xi_x * np.cos(two_pi_t * nu_d)
            + xi_y * np.sin(two_pi_t * nu_d))


def evolution_operator_microwave(
        system, H_drive, t_points=None, parallel=False, **kwargs):
    """
    Calculates the evolution operator for the gate activated by
    a microwave drive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system supporting system.H() method
        for the Hamiltonian.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
    parallel : True or False
        Run the qutip propagator function in parallel mode
    **kwargs:
        Contains gate parameters such as pulse shape and gate time.

    Returns
    -------
    U_t : *array* of :class:`qutip.Qobj`
        The evolution operator at time(s) defined in `t_points` written in
        the basis used by `system`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    U_t = qt.propagator(H, t_points, [], args=kwargs, parallel=parallel)

    return U_t

def H_drive_coeff_gate_long(t, args):
    """
    The time-dependent coefficient of the drive term for the qutip
    representation of time-dependent Hamiltonians.

    Example: H = [H_nodrive, [H_drive, H_drive_coeff_gate]]

    H_drive_coeff_gate = xi_x(t) cos(wt) + xi_y(t) sin(wt)
    Normalization: \int xi(t') dt'= 2\pi for 0 < t' < T_gate
    If DRAG == True: xi_y(t) = alpha * d xi_x / dt,
    else: xi_y = 0
    If SYMM == True: xi_x(t) -> xi_x(t) + beta * d^2 xi_x / dt^2
    """
    nu_d = args['omega_d']
    two_pi_t = 2 * np.pi * t
    T_gate = args['T_gate']
    if 'DRAG' in args and args['DRAG']:
        alpha = args['DRAG_coefficient']
    else:
        alpha = 0
    if 'SYMM' in args and args['SYMM']:
        beta = args['SYMM_coefficient']
    else:
        beta = 0
    if 'shape' not in args or args['shape'] == 'square':
        xi_x = 1
        xi_y = 0
    elif args['shape'] == 'cos':
        xi_x = (2 * np.pi / T_gate) * (1 - np.cos(two_pi_t / T_gate))
        xi_x += beta * (2 * np.pi / T_gate) ** 3 * np.cos(two_pi_t / T_gate)
        xi_y = 4 * alpha * np.pi ** 2 / T_gate ** 2 * np.sin(two_pi_t / T_gate)
    elif args['shape'] == 'gauss':
        sigma = args['sigma']
        width = sigma * T_gate
        t_0 = sigma ** -1 * 0.5 * width
        if t <= T_gate:
            xi_x = np.exp(-0.5 * (t - t_0) ** 2 / width ** 2)
        else:
            xi_x = 0
        xi_y = alpha*np.gradient(xi_x)
    elif args['shape'] == 'gauss_flat':
        sigma = args['sigma']
        T_edge = args['T_edge']
        width = sigma * T_edge
        T_flat = T_gate - T_edge
        t_0 = sigma ** -1 * 0.5 * width
        if t <= t_0:
            xi_x = np.exp(-0.5 * (t - t_0) ** 2 / width ** 2)
            xi_y = -np.exp(-0.5 * (t - t_0) ** 2 / width ** 2)*(t-t_0)/width**2/3
        elif (t>t_0) and (t<T_gate - t_0):
            xi_x = 1
            xi_y = 0
        elif (t>=T_gate - t_0) and (t<=T_gate):
            xi_x = np.exp(-0.5 * (t - (t_0+T_flat)) ** 2 / width ** 2)
            xi_y = -np.exp(-0.5 * (t - (t_0+T_flat)) ** 2 / width ** 2) *(t - (t_0+T_flat))/width**2/3
        else:
            xi_x = 0
            xi_y = 0
        xi_y = alpha*xi_y
    elif args['shape'] == 'gauss_flat_haonan':
        width = args['width']
        T_flat = args['T_flat']
        sigma = width / np.sqrt(2 * np.pi)
        y_offset = np.exp(-(-width) ** 2 / (2 * sigma ** 2))
        rescale_factor = 1.0 / (1.0 - y_offset)

        if t <= width:
            xi_x = (np.exp(-0.5 * (t - width) ** 2 / sigma ** 2)-y_offset)*rescale_factor
            xi_y = (-np.exp(-0.5 * (t - width) ** 2 / sigma ** 2)*(t-width)/sigma**2) * rescale_factor
        elif (t>width) and (t<width+T_flat):
            xi_x = 1
            xi_y = 0
        elif (t>=width+T_flat) and (t<=2*width+T_flat):
            xi_x = (np.exp(-0.5 * (t - width-T_flat) ** 2 / sigma ** 2)-y_offset)*rescale_factor
            xi_y = (-np.exp(-0.5 * (t - width-T_flat) ** 2 / sigma ** 2)*(t-width-T_flat)/sigma**2) * rescale_factor
        else:
            xi_x = 0
            xi_y = 0
        xi_y = alpha*xi_y
    else:
        raise Exception('Urecognized shape.')
    return (xi_x * np.cos(two_pi_t * nu_d)
            + xi_y * np.sin(two_pi_t * nu_d))

def evolution_operator_microwave_long(
        system, H_drive, t_points=None, parallel=False, **kwargs):
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate_long]]
    U_t = qt.propagator(H, t_points, [], args=kwargs, parallel=parallel)

    return U_t

def evolution_compspace_microwave(
        system, H_drive, comp_space=['00', '01', '10', '11'],
        interaction='on', t_points=None, **kwargs):
    """
    Partially calculates the evolution operator valid for computational
    states for the gate activated by a microwave drive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system supporting system.H() method
        for the Hamiltonian.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    comp_space : *list* of int, tuple, or str
        Four labels of computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
    **kwargs:
        Contains gate parameters such as pulse shape and gate time.

    Returns
    -------
    U_t : *array* of :class:`qutip.Qobj`
        The evolution operator at time(s) defined in `t_points` written in
        the basis used by `system`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    psi_t = {}
    psi_t0 = {}
    for state in comp_space:
        psi_t0[state] = system.eigvec(state, interaction=interaction)
        result = qt.sesolve(H, psi_t0[state], t_points, [], args=kwargs,
                            options=qt.Options(nsteps=25000))
        psi_t[state] = result.states
    U_t = np.empty_like(t_points, dtype=object)
    for ind_t in range(len(t_points)):
        U = 0
        for state in comp_space:
            U += psi_t[state][ind_t] * psi_t0[state].dag()
        U_t[ind_t] = U
    return U_t

def evolution_compspace_microwave_long(
        system, H_drive, comp_space=['00', '01', '10', '11'],
        interaction='on', t_points=None, **kwargs):
    """
    Partially calculates the evolution operator valid for computational
    states for the gate activated by a microwave drive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system supporting system.H() method
        for the Hamiltonian.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    comp_space : *list* of int, tuple, or str
        Four labels of computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
    **kwargs:
        Contains gate parameters such as pulse shape and gate time.

    Returns
    -------
    U_t : *array* of :class:`qutip.Qobj`
        The evolution operator at time(s) defined in `t_points` written in
        the basis used by `system`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate_long]]
    psi_t = {}
    psi_t0 = {}
    for state in comp_space:
        psi_t0[state] = system.eigvec(state, interaction=interaction)
        result = qt.sesolve(H, psi_t0[state], t_points, [], args=kwargs,
                            options=qt.Options(nsteps=25000))
        psi_t[state] = result.states
    U_t = np.empty_like(t_points, dtype=object)
    for ind_t in range(len(t_points)):
        U = 0
        for state in comp_space:
            U += psi_t[state][ind_t] * psi_t0[state].dag()
        U_t[ind_t] = U
    return U_t

def evolution_psi_microwave(
        system, H_drive, initial_state, t_points=None, **kwargs):
    """
    Calculates the evolution of a specific starting state for
    the gate activated by a microwave drive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system supporting system.H() method
        for the Hamiltonian.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    initial_state : :class:`qutip.Qobj`
        Initial state of the system.
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
    **kwargs:
        Contains gate parameters such as pulse shape and gate time.

    Returns
    -------
    *array* of :class:`qutip.Qobj`
        The evolving state at time(s) defined in `t_points`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]

    result = qt.sesolve(H, initial_state, t_points, [], args=kwargs,
                        options=qt.Options(nsteps=25000))

    return result.states


def evolution_mcsolve_microwave(
        system, H_drive, initial_state, c_ops, e_ops, num_cpus=0,
        nsteps=2000, ntraj=1000, t_points=None, **kwargs):
    """
    Calculates the expectation values vs time for a dissipative system
    for the gate activated by a microwave drive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system supporting system.H() method
        for the Hamiltonian.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    initial_state : :class:`qutip.Qobj`
        Initial state of the system.
    c_ops : *list* of :class:`qutip.Qobj`
        The list of collaps operators for MC solver.
    e_ops : *list* of :class:`qutip.Qobj`
        The list of operators to calculate expectation values.
    num_cpus, nsteps, ntraj : int
        Parameters for MC solver
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
    **kwargs:
        Contains gate parameters such as pulse shape and gate time.

    Returns
    -------
    *array*
        result.expect of mcsolve (time-dependent expectation values)
        result.expect[0] for the expectation value of the first operator
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    H_nodrive = system.H()
    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]

    options = qt.Options(num_cpus=num_cpus, nsteps=nsteps)
    result = qt.mcsolve(H, initial_state, t_points, c_ops=c_ops,
                        e_ops=e_ops, args=kwargs, ntraj=ntraj,
                        options=options)
    return result.expect


def prob_transition(
        system, U_t, initial_state, final_state, interaction='on'):
    """The probability to find the system in a certain state.

    Calculates the transition probability between certain initial and
    final states of the system.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    initial_state, final_state : int, tuple, or str
        The labels of the initial and final states according to the
        rules of `coupobj.CoupledObjects` class.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    float or *array* of float
        Transition probability.
    """
    psi_t0 = system.eigvec(initial_state, interaction=interaction)
    psi_t = U_t * psi_t0
    P_final = system.projection(final_state, interaction=interaction)
    return qt.expect(P_final, psi_t)


def prob_subspace(system, U_t, initial_state,
                  subspace=['00', '01', '10', '11'], interaction='on'):
    """The probability to find a system in a certain subspace.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    initial_state : int, tuple, or str
        The label of the initial state according to the rules
        of `coupobj.CoupledObjects` class.
    subspace : *list* of int, tuple, or str
        The labels of final states defining the subspace.
        Example: ['000', '010', '001', '011'] for the computational subspace
        of a two-qubit system + resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    float or *array* of float
        The probability of finding the system in the subspace
        given the definite starting initial state.
    """
    P_t = 0
    for final_state in subspace:
        P_t += prob_transition(system, U_t, initial_state,
                               final_state, interaction=interaction)
    return P_t


def prob_subspace_average(
        system, U_t, subspace=['00', '01', '10', '11'], interaction='on'):
    """The average probability of staying in the same subspace.

    Example: the probability of remaining in the computational subspace
    averaged over all initial computational states.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    subspace : *list* of int, tuple, or str
        The labels of states spanning the subspace.
        Example: ['000', '010', '001', '011'] for the computational subspace
        of a two-qubit system + resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    float or *array* of float
        The probability of staying within the subspace
        averaged over initial states.
    """
    P_t = 0
    for initial_state in subspace:
        P_t += prob_subspace(system, U_t, initial_state,
                             subspace=subspace, interaction=interaction)
    return P_t / len(subspace)


def gate_ideal_cz(
        system, comp_space=['00', '01', '10', '11'], interaction='on'):
    """The operator of the ideal control-z gate for two qubits.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    comp_space : *list* of int, tuple, or str
        Four labels of the computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    :class:`qutip.Qobj`
        The gate operator.
    """
    U_ideal = 0
    for state in comp_space[:3]:
        U_ideal += system.projection(state, interaction=interaction)
    U_ideal -= system.projection(comp_space[3], interaction=interaction)
    return U_ideal


def projection_subspace(
        system, subspace=['00', '01', '10', '11'], interaction='on'):
    """Operator of projection into a subspace defined by several states.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    subspace : *list* of int, tuple, or str
        The labels of states spanning the subspace.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    :class:`qutip.Qobj`
        The projection operator.
    """
    P = 0
    for state in subspace:
        P += system.projection(state, interaction=interaction)
    return P


def change_operator_single_qub_z(
        system, U, comp_space=['00', '01', '10', '11'], interaction='on'):
    """Adjusts phases in the operator via single-qubit z-rotations.

    Tunes phases of the first three diagonal matrix elements in the
    computational space to make those elements real and positive.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U : :class:`qutip.Qobj`
        Operator to modify.
    comp_space : *list* of int, tuple, or str
        Four labels of computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    :class:`qutip.Qobj`
        Modified operator.
    """
    phases = []
    single_qubit_gate = 0
    for state in comp_space:
        vec = system.eigvec(state, interaction=interaction)
        phase = np.angle(U.matrix_element(vec.dag(), vec))
        if state != comp_space[3]:
            phases.append(phase)
            single_qubit_gate += np.exp(-1j * phase) * vec * vec.dag()
        else:
            single_qubit_gate += (np.exp(1j * (phases[0] - phases[1]
                                               - phases[2])) * vec * vec.dag())
    return single_qubit_gate * U


def change_operator_proj_subspace(
        system, U, subspace=['00', '01', '10', '11'], interaction='on'):
    """Projects an operator into a subspace.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U : :class:`qutip.Qobj` or *array* of such
        Operator to project.
    subspace : *list* of int, tuple, or str
        The labels of states spanning the subspace.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.

    Returns
    -------
    :class:`qutip.Qobj` or *array* of such
        Modified operator.
    """
    P = projection_subspace(
        system, subspace=subspace, interaction=interaction)
    return P * U * P


def fidelity_general(U_ideal, U_real):
    """A general expression for fidelity of a two-qubit gate.

    Parameters
    ----------
    U_ideal, U_real : :class:`qutip.Qobj`
        The ideal and real evolution operators projected into
        the computational space.

    Returns
    -------
    float
        Fidelity.
    """
    op1 = U_real.dag() * U_real
    op2 = U_real * U_ideal.dag()
    return (op1.tr() + (abs(op2.tr())) ** 2) / 20.0


def fidelity_cz_gate_point(
        system, U, comp_space=['00', '01', '10', '11'],
        interaction='on', single_gates='z'):
    """Fidelity of the control-z gate for a specific operator.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U : :class:`qutip.Qobj`
        The 'real' evolution operator previously calculated in
        `evolution_operator()`.
    comp_space : *list* of int, tuple, or str
        Four labels of computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.
    single_gates : str, optional
        Determines single-qubit gates to adjust two-qubit evolution
        operator to compare it with the ideal operator.

    Returns
    -------
    float
        Gate fidelity.
    """
    U_ideal = gate_ideal_cz(
        system, comp_space=comp_space, interaction=interaction)
    U_real = change_operator_proj_subspace(
        system, U, subspace=comp_space, interaction=interaction)
    if single_gates == 'z':
        U_real = change_operator_single_qub_z(
            system, U_real, comp_space=comp_space,
            interaction=interaction)
    elif single_gates == 'no':
        pass
    else:
        raise Exception('Unrecognized option for `single_gates`.')
    return fidelity_general(U_ideal, U_real)


def fidelity_cz_gate(system, U_t, comp_space=['00', '01', '10', '11'],
                     interaction='on', single_gates='z'):
    """Fidelity of the control-z gate.

    Parameters
    ----------
    system : :class:`coupobj.CoupledObjects` or similar
        An object of a quantum system.
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator previously calculated in
        `evolution_operator()`.
    comp_space : *list* of int, tuple, or str
        Four labels of computational subspace of the two qubits with
        the last label being for the '11' state.
        Example: ['000', '010', '001', '011'] for the system
        two qubits + resonator. The first index is the resonator.
    interaction : 'on' or 'off', optional
        Determines whether we work in the interacting or noninteracting
        basis.
    single_gates : str, optional
        Determines single-qubit gates to adjust two-qubit evolution
        operator to compare it with the ideal operator.

    Returns
    -------
    float or *array* of float
        Gate fidelity.
    """
    if isinstance(U_t, np.ndarray):
        fidelity = np.empty_like(U_t)
        for indU, U in enumerate(U_t):
            fidelity[indU] = fidelity_cz_gate_point(
                system, U, comp_space=comp_space,
                interaction=interaction,
                single_gates=single_gates)
    else:
        fidelity = fidelity_cz_gate_point(
            system, U_t, comp_space=comp_space, interaction=interaction,
            single_gates=single_gates)
    return fidelity