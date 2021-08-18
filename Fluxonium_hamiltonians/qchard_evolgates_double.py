"""
Time evolution during gate operations.
@author: Konstantin Nesterov
"""

import sys

sys.dont_write_bytecode = True

import numpy as np
import scipy.special
import qutip as qt


def H_drive_coeff_gate(t, args):
    """
    The time-dependent coefficient of the microwave-drive term for the qutip
    representation of time-dependent Hamiltonians.

    Example: H = [H_nodrive, [H_drive, H_drive_coeff_gate]]

    H_drive_coeff_gate = xi_x(t) cos(wt + phi) + xi_y(t) sin(wt + phi)
    T_start < t < T_start + T_gate
    Normalization: \int xi_x(t') dt'= \theta (Bloch-sphere rotation angle),
               so H_drive ~ \sigma_x for a resonance drive in an ideal case

    If DRAG == True: xi_y(t) = alpha * d xi_x / dt,
        else: xi_y = 0
    If SYMM == True: xi_x(t) -> xi_x(t) + beta * d^2 xi_x / dt^2

    Default values:
        T_start = 0
        DRAG and SYMM = False
        theta = 2 * pi  (full 2-pi rotation)
        phi = 0  (no phase delay)
        shape = square
    """
    if 'T_start' in args:
        T_start = args['T_start']
    else:
        T_start = 0

    if 'shape' in args:
        shape = args['shape']
    else:
        shape = 'square'

    if 'sigma' in args:
        sigma = args['sigma']
    else:
        sigma = 0.25

    if 'theta' in args:
        theta = args['theta']
    else:
        theta = 2 * np.pi

    if 'phi' in args:
        phi = args['phi']
    else:
        phi = 0

    if 'DRAG' in args and args['DRAG']:
        alpha = args['DRAG_coefficient']

    else:
        alpha = 0
    if 'SYMM' in args and args['SYMM']:
        beta = args['SYMM_coefficient']
    else:
        beta = 0

    if 'drive_ratio' in args:
        eta_amp = args['drive_ratio']
    else:
        eta_amp = 0

    if 'drag_coeff_ratio' in args:
        eta_drag = args['drag_coeff_ratio']
    else:
        eta_drag = 0


    nu_d_1 = args['omega_d_1']
    nu_d_2 = args['omega_d_2']
    two_pi_t1 = 2 * np.pi * t
    two_pi_t2 = 2 * np.pi * (t - T_start)
    T_gate = args['T_gate']

    if shape == 'gaussflattop' and T_gate < 2 * args['T_rise']:
        shape = 'gauss'
        sigma = sigma * args['T_rise'] / T_gate

    # Here xi_x and xi_y are normalized assuming theta = 2 * pi.
    if shape == 'square':
        xi_x_1 = 2 * np.pi / T_gate
        xi_y_1 = 0
    elif shape == 'gaussflattop':
        T_rise = args['T_rise']
        sigma = sigma * T_rise
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x_1 = np.exp(- 0.5 * ((t - T_left) / sigma) ** 2)
            xi_y_1 = (alpha * (- (t - T_left) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_left) / sigma) ** 2))
        elif t > T_right:
            xi_x_1 = np.exp(- 0.5 * ((t - T_right) / sigma) ** 2)
            xi_y_1 = (alpha * (- (t - T_right) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_right) / sigma) ** 2))
        else:
            xi_x_1 = 1
            xi_y_1 = 0
        # Shift to ensure that we start and end at zero.
        xi_x_1 -= np.exp(-0.5 * (T_rise / sigma) ** 2)
        # Normalization.
        if 'normalization_flat' in args and args['normalization_flat']:
            # Normalize as if it were a square pulse.
            coeff = 2 * np.pi / (T_gate * (1 - np.exp(-0.5 * (T_rise / sigma) ** 2)))
            xi_x_1 *= coeff
            xi_y_1 *= coeff
        else:
            # Normalize ``the usual way''.
            integral_value = (np.sqrt(2 * np.pi) * sigma
                              * scipy.special.erf(
                        T_rise / (np.sqrt(2) * sigma))
                              + T_gate - 2 * T_rise
                              - T_gate * np.exp(-0.5 * (T_rise / sigma) ** 2))
            xi_x_1 *= 2 * np.pi / integral_value
            xi_y_1 *= 2 * np.pi / integral_value
    elif shape == 'cos':
        xi_x_1 = (2 * np.pi / T_gate) * (1 - np.cos(two_pi_t2 / T_gate))
        xi_x_1 += beta * (2 * np.pi / T_gate) ** 3 * np.cos(two_pi_t2 / T_gate)
        xi_y_1 = 4 * alpha * np.pi ** 2 / T_gate ** 2 * np.sin(two_pi_t2 / T_gate)

    elif shape == 'gauss':
        sigma = sigma * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 2 * np.pi / integral_value
        T_mid = T_start + T_gate / 2
        xi_x_1 = coeff * (np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x_1 += (beta * coeff * (-1 / sigma ** 2 + ((t - T_mid) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
        xi_y_1 = (alpha * coeff * (- (t - T_mid) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))

    elif shape == 'two_phot_sin':
        xi_x_1 = np.sqrt(2) * (2 * np.pi / T_gate) * np.sin(two_pi_t2 / T_gate)
        xi_y_1 = 0
    else:
        raise Exception('Urecognized pulse shape.')

    H_drive_coeff = (xi_x_1 * np.cos(two_pi_t1 * nu_d_1 + phi) + xi_y_1 * np.sin(two_pi_t1 * nu_d_1 + phi)) * theta / (2 * np.pi) \
                    + (eta_amp*xi_x_1 * np.cos(two_pi_t1 * nu_d_2 + phi) + eta_drag*xi_y_1 * np.sin(two_pi_t1 * nu_d_2 + phi)) * theta / (2 * np.pi)

    return H_drive_coeff

def evolution_psi_microwave(
        H_nodrive, H_drive, psi0, t_points=None, **kwargs):
    """
    Calculates the unitary evolution of a specific starting state for
    a gate activated by a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    psi0 : :class:`qutip.Qobj`
        Initial state of the system (ket or density matrix).
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

    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    result = qt.mesolve(H, psi0, t_points, [], args=kwargs,
                        options=qt.Options(nsteps=25000))

    return result.states

def evolution_psi_microwave_diss(
        H_nodrive, H_drive, psi0, c_ops = [], t_points=None, **kwargs):
    """
    Calculates the unitary evolution of a specific starting state for
    a gate activated by a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    psi0 : :class:`qutip.Qobj`
        Initial state of the system (ket or density matrix).
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

    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    result = qt.mesolve(H, psi0, t_points, c_ops, args=kwargs,
                        options=qt.Options(nsteps=25000))

    return result.states

def evolution_operator_microwave(
        H_nodrive, H_drive, t_points=None, parallel=False, **kwargs):
    """
    Calculates the unitary evolution operator for a gate activated by
    a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
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
        The evolution operator at time(s) defined in `t_points`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)

    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    U_t = qt.propagator(H, t_points, [], args=kwargs, parallel=parallel)

    return U_t

def evolution_operator_microwave_diss(
        H_nodrive, H_drive, t_points=None, c_ops = [], parallel=False, **kwargs):
    """
    Calculates the unitary evolution operator for a gate activated by
    a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
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
        The evolution operator at time(s) defined in `t_points`.
    """
    if t_points is None:
        T_gate = kwargs['T_gate']
        t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)

    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]
    U_t = qt.propagator(H, t_points, c_ops_list=c_ops, args=kwargs, parallel=parallel)

    return U_t