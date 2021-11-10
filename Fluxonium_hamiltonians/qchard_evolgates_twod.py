#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time evolution during gate operations.
@author: Long Nguyen & Yosep Kim
@original code: Konstantin Nesterov
"""

import sys

sys.dont_write_bytecode = True

import numpy as np
import scipy.special
import qutip as qt


def H_drive_coeff_gate1(t, args):
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

    nu_d = args['omega_d1']
    two_pi_t1 = 2 * np.pi * t
    two_pi_t2 = 2 * np.pi * (t - T_start)
    T_gate = args['T_gate']

    if shape == 'gaussflattop' and T_gate < 2 * args['T_rise']:
        shape = 'gauss'
        sigma = sigma * args['T_rise'] / T_gate

    # Here xi_x and xi_y are normalized assuming theta = 2 * pi.    
    if shape == 'square':
        xi_x = 2 * np.pi / T_gate
        xi_y = 0
    elif shape == 'gaussflattop':
        T_rise = args['T_rise']
        sigma = sigma * T_rise
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = np.exp(- 0.5 * ((t - T_left) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_left) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_left) / sigma) ** 2))
        elif t > T_right:
            xi_x = np.exp(- 0.5 * ((t - T_right) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_right) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_right) / sigma) ** 2))
        else:
            xi_x = 1
            xi_y = 0
        # Shift to ensure that we start and end at zero.
        xi_x -= np.exp(-0.5 * (T_rise / sigma) ** 2)
        # Normalization.
        if 'normalization_flat' in args and args['normalization_flat']:
            # Normalize as if it were a square pulse.
            coeff = 2 * np.pi / (T_gate * (1 - np.exp(-0.5 * (T_rise / sigma) ** 2)))
            xi_x *= coeff
            xi_y *= coeff
        else:
            # Normalize ``the usual way''.
            integral_value = (np.sqrt(2 * np.pi) * sigma
                              * scipy.special.erf(
                        T_rise / (np.sqrt(2) * sigma))
                              + T_gate - 2 * T_rise
                              - T_gate * np.exp(-0.5 * (T_rise / sigma) ** 2))
            xi_x *= 2 * np.pi / integral_value
            xi_y *= 2 * np.pi / integral_value

    elif shape == 'cos':
        xi_x = (2 * np.pi / T_gate) * (1 - np.cos(two_pi_t2 / T_gate))
        xi_x += beta * (2 * np.pi / T_gate) ** 3 * np.cos(two_pi_t2 / T_gate)
        xi_y = 4 * alpha * np.pi ** 2 / T_gate ** 2 * np.sin(two_pi_t2 / T_gate)
    elif shape == 'gauss':
        sigma = sigma * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 2 * np.pi / integral_value
        T_mid = T_start + T_gate / 2
        xi_x = coeff * (np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x += (beta * coeff * (-1 / sigma ** 2 + ((t - T_mid) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
        xi_y = (alpha * coeff * (- (t - T_mid) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
    elif shape == 'two_phot_sin':
        xi_x = np.sqrt(2) * (2 * np.pi / T_gate) * np.sin(two_pi_t2 / T_gate)
        xi_y = 0
    else:
        raise Exception('Urecognized pulse shape.')

    return (xi_x * np.cos(two_pi_t1 * nu_d + phi)
            + xi_y * np.sin(two_pi_t1 * nu_d + phi)) * theta / (2 * np.pi)


def H_drive_coeff_gate2(t, args):
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

    nu_d = args['omega_d2']
    two_pi_t1 = 2 * np.pi * t
    two_pi_t2 = 2 * np.pi * (t - T_start)
    T_gate = args['T_gate']

    if shape == 'gaussflattop' and T_gate < 2 * args['T_rise']:
        shape = 'gauss'
        sigma = sigma * args['T_rise'] / T_gate

    # Here xi_x and xi_y are normalized assuming theta = 2 * pi.
    if shape == 'square':
        xi_x = 2 * np.pi / T_gate
        xi_y = 0
    elif shape == 'gaussflattop':
        T_rise = args['T_rise']
        sigma = sigma * T_rise
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = np.exp(- 0.5 * ((t - T_left) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_left) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_left) / sigma) ** 2))
        elif t > T_right:
            xi_x = np.exp(- 0.5 * ((t - T_right) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_right) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_right) / sigma) ** 2))
        else:
            xi_x = 1
            xi_y = 0
        # Shift to ensure that we start and end at zero.
        xi_x -= np.exp(-0.5 * (T_rise / sigma) ** 2)
        # Normalization.
        if 'normalization_flat' in args and args['normalization_flat']:
            # Normalize as if it were a square pulse.
            coeff = 2 * np.pi / (T_gate * (1 - np.exp(-0.5 * (T_rise / sigma) ** 2)))
            xi_x *= coeff
            xi_y *= coeff
        else:
            # Normalize ``the usual way''.
            integral_value = (np.sqrt(2 * np.pi) * sigma
                              * scipy.special.erf(
                        T_rise / (np.sqrt(2) * sigma))
                              + T_gate - 2 * T_rise
                              - T_gate * np.exp(-0.5 * (T_rise / sigma) ** 2))
            xi_x *= 2 * np.pi / integral_value
            xi_y *= 2 * np.pi / integral_value

    elif shape == 'cos':
        xi_x = (2 * np.pi / T_gate) * (1 - np.cos(two_pi_t2 / T_gate))
        xi_x += beta * (2 * np.pi / T_gate) ** 3 * np.cos(two_pi_t2 / T_gate)
        xi_y = 4 * alpha * np.pi ** 2 / T_gate ** 2 * np.sin(two_pi_t2 / T_gate)
    elif shape == 'gauss':
        sigma = sigma * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 2 * np.pi / integral_value
        T_mid = T_start + T_gate / 2
        xi_x = coeff * (np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x += (beta * coeff * (-1 / sigma ** 2 + ((t - T_mid) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
        xi_y = (alpha * coeff * (- (t - T_mid) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
    elif shape == 'two_phot_sin':
        xi_x = np.sqrt(2) * (2 * np.pi / T_gate) * np.sin(two_pi_t2 / T_gate)
        xi_y = 0
    else:
        raise Exception('Urecognized pulse shape.')

    return (xi_x * np.cos(two_pi_t1 * nu_d + phi)
            + xi_y * np.sin(two_pi_t1 * nu_d + phi)) * theta / (2 * np.pi)

def H_drive_coeff_gate1_nonorm(t, args):
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

    nu_d = args['omega_d1']
    two_pi_t1 = 2 * np.pi * t
    two_pi_t2 = 2 * np.pi * (t - T_start)
    T_gate = args['T_gate']

    if shape == 'gaussflattop' and T_gate < 2 * args['T_rise']:
        shape = 'gauss'
        sigma = sigma * args['T_rise'] / T_gate

    # Here xi_x and xi_y are normalized assuming theta = 2 * pi.
    if shape == 'square':
        xi_x = 1
        xi_y = 0
    elif shape == 'gaussflattop':
        T_rise = args['T_rise']
        sigma = sigma * T_rise
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = np.exp(- 0.5 * ((t - T_left) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_left) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_left) / sigma) ** 2))
        elif t > T_right:
            xi_x = np.exp(- 0.5 * ((t - T_right) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_right) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_right) / sigma) ** 2))
        else:
            xi_x = 1
            xi_y = 0
        # Shift to ensure that we start and end at zero.
        xi_x -= np.exp(-0.5 * (T_rise / sigma) ** 2)

    elif shape == 'cosflattop':
        T_rise = args['T_rise']
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = (1+np.cos(2 * np.pi * (t - T_left) / T_rise / 2))/2
            xi_y = alpha * np.sin(2 * np.pi * (t - T_left) / T_rise / 2)
        elif t > T_right:
            xi_x = (1+np.cos(2 * np.pi * (t - T_right) / T_rise / 2))/2
            xi_y = -alpha * np.sin(2 * np.pi * (t - T_right) / T_rise / 2)
        else:
            xi_x = 1
            xi_y = 0

    elif shape == 'cos':
        xi_x =  (1 - np.cos(two_pi_t2 / T_gate))
        xi_x += beta * np.cos(two_pi_t2 / T_gate)
        xi_y = alpha  * np.sin(two_pi_t2 / T_gate)
    elif shape == 'gauss':
        sigma = sigma * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 2 * np.pi / integral_value
        T_mid = T_start + T_gate / 2
        xi_x =  (np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x += (beta * (-1 / sigma ** 2 + ((t - T_mid) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
        xi_y = (alpha * (- (t - T_mid) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
    elif shape == 'two_phot_sin':
        xi_x = np.sqrt(2) * (2 * np.pi / T_gate) * np.sin(two_pi_t2 / T_gate)
        xi_y = 0
    else:
        raise Exception('Urecognized pulse shape.')

    return (xi_x * np.cos(two_pi_t1 * nu_d + phi)
            + xi_y * np.sin(two_pi_t1 * nu_d + phi)) * theta / (2 * np.pi)

def H_drive_coeff_gate2_nonorm(t, args):
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

    nu_d = args['omega_d2']
    two_pi_t1 = 2 * np.pi * t
    two_pi_t2 = 2 * np.pi * (t - T_start)
    T_gate = args['T_gate']

    if shape == 'gaussflattop' and T_gate < 2 * args['T_rise']:
        shape = 'gauss'
        sigma = sigma * args['T_rise'] / T_gate

    # Here xi_x and xi_y are normalized assuming theta = 2 * pi.
    if shape == 'square':
        xi_x = 1
        xi_y = 0
    elif shape == 'gaussflattop':
        T_rise = args['T_rise']
        sigma = sigma * T_rise
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = np.exp(- 0.5 * ((t - T_left) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_left) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_left) / sigma) ** 2))
        elif t > T_right:
            xi_x = np.exp(- 0.5 * ((t - T_right) / sigma) ** 2)
            xi_y = (alpha * (- (t - T_right) / sigma ** 2)
                    * np.exp(- 0.5 * ((t - T_right) / sigma) ** 2))
        else:
            xi_x = 1
            xi_y = 0
        # Shift to ensure that we start and end at zero.
        xi_x -= np.exp(-0.5 * (T_rise / sigma) ** 2)

    elif shape == 'cosflattop':
        T_rise = args['T_rise']
        T_left = T_start + T_rise
        T_right = T_start + T_gate - T_rise
        # Without shift and normalization.
        if t < T_left:
            xi_x = (1+np.cos(2 * np.pi * (t - T_left) / T_rise / 2))/2
            xi_y = alpha * np.sin(2 * np.pi * (t - T_left) / T_rise / 2)
        elif t > T_right:
            xi_x = (1+np.cos(2 * np.pi * (t - T_right) / T_rise / 2))/2
            xi_y = -alpha * np.sin(2 * np.pi * (t - T_right) / T_rise / 2)
        else:
            xi_x = 1
            xi_y = 0

    elif shape == 'cos':
        xi_x = (1 - np.cos(two_pi_t2 / T_gate))
        xi_x += beta * np.cos(two_pi_t2 / T_gate)
        xi_y = alpha * np.sin(two_pi_t2 / T_gate)
    elif shape == 'gauss':
        sigma = sigma * T_gate
        integral_value = (np.sqrt(2 * np.pi) * sigma
                          * scipy.special.erf(
                    T_gate / (2 * np.sqrt(2) * sigma))
                          - T_gate * np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        coeff = 1
        T_mid = T_start + T_gate / 2
        xi_x = coeff * (np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2)
                        - np.exp(-0.5 * (0.5 * T_gate / sigma) ** 2))
        xi_x += (beta * coeff * (-1 / sigma ** 2 + ((t - T_mid) / sigma ** 2) ** 2)
                 * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
        xi_y = (alpha * coeff * (- (t - T_mid) / sigma ** 2)
                * np.exp(- 0.5 * ((t - T_mid) / sigma) ** 2))
    elif shape == 'two_phot_sin':
        xi_x = np.sqrt(2) * (2 * np.pi / T_gate) * np.sin(two_pi_t2 / T_gate)
        xi_y = 0
    else:
        raise Exception('Urecognized pulse shape.')

    return (xi_x * np.cos(two_pi_t1 * nu_d + phi)
            + xi_y * np.sin(two_pi_t1 * nu_d + phi)) * theta / (2 * np.pi)

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
        H_nodrive, H_drive1, H_drive2, psi0, c_ops = [], t_points=None, **kwargs):
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

    H = [2 * np.pi * H_nodrive, [H_drive1, H_drive_coeff_gate1], [H_drive2, H_drive_coeff_gate2]]
    result = qt.mesolve(H, psi0, t_points, c_ops, args=kwargs,
                        options=qt.Options(nsteps=100000, atol=1e-12, rtol=1e-10))

    return result.states

def evolution_operator_microwave(
        H_nodrive, H_drive1, H_drive2, t_points=None, parallel=False, **kwargs):
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
    # print(kwargs)
    H = [2 * np.pi * H_nodrive, [H_drive1, H_drive_coeff_gate1], [H_drive2, H_drive_coeff_gate2]]
    U_t = qt.propagator(H, t_points, [], args=kwargs, parallel=parallel)

    return U_t

def evolution_operator_microwave_nonorm(
        H_nodrive, H_drive1, H_drive2, t_points=None, parallel=False, **kwargs):
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
    # print(kwargs)
    H = [2 * np.pi * H_nodrive, [H_drive1, H_drive_coeff_gate1_nonorm], [H_drive2, H_drive_coeff_gate2_nonorm]]
    U_t = qt.propagator(H, t_points, [], args=kwargs, parallel=parallel)

    return U_t

def evolution_operator_microwave_diss(
        H_nodrive, H_drive1, H_drive2, t_points=None, c_ops = [], parallel=False, **kwargs):
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

def evolution_subspace_microwave(
        H_nodrive, H_drive, subspace_states, t_points=None, **kwargs):
    """
    (faster than `evolution_operator_microwave`)
    Calculates the evolution operator valid only for states in
    a specific subspace for a gate activated by a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    supspace_states : *list* of :class:`qutip.Qobj`
        Vectors defining the subspace. For example, four vectors
        |00>, |01>, |10>, and |11> for a computational subspace of 
        a two-qubit system.
    t_points : *array* of float (optional)
        Times at which the evolution operator is returned.
        If None, it is generated from `kwargs['T_gate']`.
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

    psi_t = np.empty_like(subspace_states, dtype=object)
    for ind, psi_0 in enumerate(subspace_states):
        psi_t[ind] = evolution_psi_microwave(
            H_nodrive, H_drive, psi_0, t_points, **kwargs)

    U_t = np.empty_like(t_points, dtype=object)
    for ind_t in range(len(t_points)):
        U = 0
        for ind, psi_0 in enumerate(subspace_states):
            U += psi_t[ind][ind_t] * psi_0.dag()
        U_t[ind_t] = U

    return U_t


def evolution_mcsolve_microwave(
        H_nodrive, H_drive, psi0, c_ops, e_ops, num_cpus=0,
        nsteps=2000, ntraj=1000, t_points=None, **kwargs):
    """
    Calculates the expectation values vs time for a dissipative system 
    for the gate activated by a microwave drive.

    Parameters
    ----------
    H_nodrive : :class:`qutip.Qobj`
        The Hamiltonian without the drive term.
    H_drive : :class:`qutip.Qobj`
        The time-independent part of the driving term.
        Example: f * (a + a.dag()) or f * qubit.n()
        Normalization: see `H_drive_coeff_gate` function.
    psi0 : :class:`qutip.Qobj`
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

    H = [2 * np.pi * H_nodrive, [H_drive, H_drive_coeff_gate]]

    options = qt.Options(num_cpus=num_cpus, nsteps=nsteps)
    result = qt.mcsolve(H, psi0, t_points, c_ops=c_ops,
                        e_ops=e_ops, args=kwargs, ntraj=ntraj,
                        options=options)
    return result.expect


def concurrence(psi, comp_space, comp_space_nonint):
    """Calculates concurrence of a two-qubit state.

    Parameters
    ----------
    psi : :class:`qutip.Qobj`
        A two-qubit state.
    comp_space : *list* of :class:`qutip.Qobj`
        The list of four vectors composing the computational subspace
        in the order [00, 01, 10, 11].
    comp_space_nonint: *list* of :class:`qutip.Qobj`
        The same, but for noninteracting states. Necessary to choose 
        correct signs / phases of interacting states.

    Returns
    -------
    float
        Concurrence.
    """
    if len(comp_space) != 4:
        raise Exception('Wrong length of comp_space')
    if psi.type != 'ket':
        raise Exception('At this point, `psi` must be a ket vector')
    vec00 = comp_space[0] * np.exp(1j * np.angle(
        comp_space_nonint[0].overlap(comp_space[0])))
    vec01 = comp_space[1] * np.exp(1j * np.angle(
        comp_space_nonint[1].overlap(comp_space[1])))
    vec10 = comp_space[2] * np.exp(1j * np.angle(
        comp_space_nonint[2].overlap(comp_space[2])))
    vec11 = comp_space[3] * np.exp(1j * np.angle(
        comp_space_nonint[3].overlap(comp_space[3])))

    alpha00 = vec00.overlap(psi)
    alpha01 = vec01.overlap(psi)
    alpha10 = vec10.overlap(psi)
    alpha11 = vec11.overlap(psi)

    return 2 * np.abs(alpha00 * alpha11 - alpha01 * alpha10)


def bell_fidelity(psi, state1, state2, best_phase=True):
    """Fidelity to a Bell state.

    Parameters
    ----------
    psi : :class:`qutip.Qobj`
        A two-qubit state.
    state1, state2 : :class:`qutip.Qobj`
        Two states defining a Bell state (|state1> + e^{i phi}|state2>)/sqrt(2)
    best_phase : True or False (optional)
        If True, the Bell-state phase `phi` is chosen to maximize fidelity.
        If False, phi = 0 (a specific Bell state is assumed)

    Returns
    -------
    fidelity : float
        Fidelity defined as |<psi | B >|^2.
    phase : float (only if best_phase = True)
        The phase of the Bell state.
    """
    alpha1 = state1.overlap(psi)
    alpha2 = state2.overlap(psi)
    if best_phase:
        phi = np.angle(alpha2 / alpha1)
        bell_state = (state1 + np.exp(1j * phi) * state2) / np.sqrt(2)
        return np.abs(psi.overlap(bell_state)) ** 2, phi
    else:
        bell_state = (state1 + state2) / np.sqrt(2)
        return np.abs(psi.overlap(bell_state)) ** 2


def fidelity_twoq_general(U_ideal, U_real, comp_space=None):
    """A general expression for fidelity of a two-qubit gate.

    Parameters
    ----------
    U_ideal, U_real : :class:`qutip.Qobj`
        The ideal and real evolution operators projected into
        the computational space.
    comp_space : *list* or *array* of :class:`qutip.Qobj` (optional)
        The list of four vectors composing the computational subspace
        in the order [00, 01, 10, 11]. Necessary when the gate operator
        is not retricted to computational subspace.

    Returns
    -------
    float
        Fidelity.
    """
    if comp_space != None:
        # Do a projection onto the computational space.
        P = 0
        for state in comp_space:
            P += state * state.dag()
        U_ideal = P * U_ideal * P
        U_real = P * U_real * P
    op1 = U_real.dag() * U_real
    op2 = U_real * U_ideal.dag()
    return (op1.tr() + (abs(op2.tr())) ** 2) / 20.0

def fidelity_singleq_general(U_ideal, U_real, comp_space=None):
    """A general expression for fidelity of a single-qubit gate.

    Parameters
    ----------
    U_ideal, U_real : :class:`qutip.Qobj`
        The ideal and real evolution operators projected into
        the computational space.
    comp_space : *list* or *array* of :class:`qutip.Qobj` (optional)
        The list of four vectors composing the computational subspace
        in the order [00, 01, 10, 11]. Necessary when the gate operator
        is not retricted to computational subspace.

    Returns
    -------
    float
        Fidelity.
    """
    if comp_space != None:
        # Do a projection onto the computational space.
        P = 0
        for state in comp_space:
            P += state * state.dag()
        U_ideal = P * U_ideal * P
        U_real = P * U_real * P
    op1 = U_real.dag() * U_real
    op2 = U_real * U_ideal.dag()
    return (op1.tr() + (abs(op2.tr())) ** 2) / 6.0

def prob_transition(U_t, initial_state, final_state):
    """The probability to find the system in a certain state.

    Calculates the transition probability between certain initial and
    final states of the system.

    Parameters
    ----------
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    initial_state, final_state :  :class:`qutip.Qobj`
        Ket vectors representing initial and final states.

    Returns
    -------
    float or *array* of float
        Transition probability.
    """
    psi_t = U_t * initial_state
    P_final = final_state * final_state.dag()
    return qt.expect(P_final, psi_t)


def prob_subspace(U_t, initial_state, subspace):
    """The probability to find a system in a certain subspace.

    Parameters
    ----------
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    initial_state :  :class:`qutip.Qobj`
        Ket vector representing initial state.
    supspace : *list* of :class:`qutip.Qobj`
        Vectors defining the subspace. For example, four vectors
        |00>, |01>, |10>, and |11> for a computational subspace of 
        a two-qubit system.

    Returns
    -------
    float or *array* of float
        The probability of finding the system in the subspace
        given the definite starting initial state.
    """
    P_t = 0
    for final_state in subspace:
        P_t += prob_transition(U_t, initial_state, final_state)
    return P_t


def prob_subspace_average(U_t, subspace):
    """The average probability of staying in the same subspace.

    Example: the probability of remaining in the computational subspace
    averaged over all initial computational states.

    Parameters
    ----------
    U_t : :class:`qutip.Qobj` or *array* of such
        The evolution operator.
    supspace : *list* of :class:`qutip.Qobj`
        Vectors defining the subspace. For example, four vectors
        |00>, |01>, |10>, and |11> for a computational subspace of 
        a two-qubit system.

    Returns
    -------
    float or *array* of float
        The probability of staying within the subspace
        averaged over initial states.
    """
    P_t = 0
    for initial_state in subspace:
        P_t += prob_subspace(U_t, initial_state, subspace=subspace)
    return P_t / len(subspace)

##########################################################
### Some useful functions from evolgates_old     #########
##########################################################
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

def operator_single_qub_z(
        system, U, comp_space=['00', '01', '10', '11'], interaction='on'):
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
    return single_qubit_gate


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
    return fidelity_twoq_general(U_ideal, U_real)


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
