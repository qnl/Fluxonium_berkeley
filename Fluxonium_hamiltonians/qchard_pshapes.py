# This file is part of QHard: quantum hardware modelling package.
#
# Author: Konstantin Nesterov, 2017 and later.
###########################################################################
"""Functions defining shapes of microwave and other pulses."""

__all__ = ['envelope_square', 'derenv_square', 'envelope_gauss',
           'derenv_gauss', 'envelope_cos', 'derenv_cos',
           'envelope_flattop_gauss', 'derenv_flattop_gauss',
           'envelope_flattop_cos', 'derenv_flattop_cos',
           'envelope_generic', 'derenv_generic']

import numpy as np
import scipy.special


def envelope_square(t, t_pulse, t_start=0, **not_needed_kwargs):
    """
    Envelope for a square pulse (1 or 0).
    Parameters
    ----------
    t : float
        Time at which the value of the envelope is to be returned.
    t_pulse : float
        The total duration of the pulse.
    t_start : float, default=0
        Time at which the pulse starts.
    Returns
    -------
    float
        The value of the envelope at a given time (1 or 0).
    See Also
    --------
    derenv_square
    """
    if t < t_start or t > t_start + t_pulse:
        return 0
    else:
        return 1


def derenv_square(t, t_pulse, t_start=0, **not_needed_kwargs):
    """
    Time derivative of the envelope for a square pulse.
    Trivial, for consistency with other derivative functions only.
    See Also
    --------
    envelope_square
    """
    return 0


def envelope_gauss(t, t_pulse, sigma, t_start=0, remove_discontinuities=True,
                   normalize=True, **not_needed_kwargs):
    """
    Envelope for a Gaussian pulse without a flat-top part.
    Parameters
    ----------
    t : float
        Time at which the value of the envelope is to be returned.
    t_pulse : float
        The total duration of the pulse.
    sigma : float
        The standard deviation of the Gaussian.
    t_start : float, default=0
        Time at which the pulse starts.
    remove_discontinuities: bool, default=True
        If True, shift the envelope to start and end at zero
        and renormalize accordingly.
    normalize : bool, default=True
        If True, normalize the envelope xi(t) such that <xi(t)> = 1.
        If False, normalize such that max[xi(t)] = 1.
    Returns
    -------
    float
        The value of the envelope at a given time.
    See Also
    --------
    derenv_gauss, envelope_flattop_gauss
    """
    return envelope_flattop_gauss(
            t, t_pulse, t_rise=t_pulse/2, sigma=sigma, t_start=t_start,
            remove_discontinuities=remove_discontinuities,
            normalize=normalize)


def derenv_gauss(t, t_pulse, sigma, t_start=0, remove_discontinuities=True,
                 normalize=True, **not_needed_kwargs):
    """
    Time derivative of the envelope for a Gaussian pulse.
    Takes the same arguments as `envelope_gauss`, which describe
    the corresponding envelope whose derivative is to be calculated.
    See Also
    --------
    envelope_gauss, derenv_flattop_gauss
    """
    return derenv_flattop_gauss(
            t, t_pulse, t_rise=t_pulse/2, sigma=sigma, t_start=t_start,
            remove_discontinuities=remove_discontinuities,
            normalize=normalize)


def envelope_cos(t, t_pulse, t_start=0, normalize=True, **not_needed_kwargs):
    """
    Envelope for a cosine-shaped pulse without a flat-top part.
    Proportional to ``1 - np.cos(2 * np.pi * t / t_pulse) `` or
    ``np.sin(np.pi * t / t_pulse)**2`` if starts at 0.
    Does not require a standard-deviation parameter.
    No discontinuities in both envelope and its derivative.
    Parameters
    ----------
    t : float
        Time at which the value of the envelope is to be returned.
    t_pulse : float
        The total duration of the pulse.
    t_start : float, default=0
        Time at which the pulse starts.
    normalize : bool, default=True
        If True, normalize the envelope xi(t) such that <xi(t)> = 1.
        If False, normalize such that max[xi(t)] = 1.
    Returns
    -------
    float
        The value of the envelope at a given time.
    See Also
    --------
    derenv_cos, envelope_flattop_cos
    """
    return envelope_flattop_cos(t, t_pulse, t_rise=t_pulse/2, t_start=t_start,
                                normalize=normalize)


def derenv_cos(t, t_pulse, t_start=0, normalize=True, **not_needed_kwargs):
    """
    Time derivative of the envelope for a cosine-shaped pulse.
    Takes the same arguments as `envelope_cos`, which describe
    the corresponding envelope whose derivative is to be calculated.
    See Also
    --------
    envelope_cos, derenv_flattop_cos
    """
    return derenv_flattop_cos(t, t_pulse, t_rise=t_pulse/2, t_start=t_start,
                              normalize=normalize)


def envelope_flattop_gauss(
        t, t_pulse, t_rise, sigma, t_start=0, remove_discontinuities=True,
        normalize=True, **not_needed_kwargs):
    """
    Envelope for a flattopped pulse with Gaussian edges.
    Parameters
    ----------
    t : float
        Time at which the value of the envelope is to be returned.
    t_pulse : float
        The total duration of the pulse.
    t_rise : float
        The durations of the rising and lowering edges of the pulse.
        Replaced by ``t_pulse/2`` when ``t_rise > t_pulse/2``.
    sigma : float
        The standard deviation in the Gaussian edges.
    t_start : float, default=0
        Time at which the pulse starts.
    remove_discontinuities: bool, default=True
        If True, shift the envelope to start and end at zero
        and renormalize accordingly.
    normalize : bool, default=True
        If True, normalize the envelope xi(t) such that <xi(t)> = 1.
        If False, normalize such that max[xi(t)] = 1.
    Returns
    -------
    float
        The value of the envelope at a given time.
    See Also
    --------
    derenv_flattop_gauss
    """
    if t_rise > t_pulse/2:
        t_rise = t_pulse/2

    if t < t_start or t > t_start + t_pulse:
        return 0

    else:
        t_left = t_start + t_rise
        t_right = t_start + t_pulse - t_rise
        t_flat = t_right - t_left
        # Without shift and normalization.
        if t < t_left:
            xi = np.exp(-0.5 * ((t-t_left)/sigma)**2)
        elif t > t_right:
            xi = np.exp(-0.5 * ((t-t_right)/sigma)**2)
        else:
            xi = 1

        if remove_discontinuities:
            # Make xi(t) continuous at t = t_start and t = t_start + t_pulse.
            xi -= np.exp(-0.5 * (t_rise/sigma)**2)
            if normalize:
                # Integrate xi(t) over the pulse duration.
                integral_xi_dt = (
                        np.sqrt(2*np.pi) * sigma
                        * scipy.special.erf(t_rise / (np.sqrt(2)*sigma))
                        + t_flat
                        - t_pulse * np.exp(-0.5 * (t_rise/sigma)**2))
                xi *= t_pulse / integral_xi_dt
            else:
                # Ensure that xi = 1 in the flat part.
                xi /= (1 - np.exp(-0.5 * (t_rise/sigma)**2))

        elif normalize:
            # Integrate xi(t) over the pulse duration.
            integral_xi_dt = (
                    np.sqrt(2*np.pi) * sigma
                    * scipy.special.erf(t_rise / (np.sqrt(2)*sigma))
                    + t_flat)
            xi *= t_pulse / integral_xi_dt
        else:
            # Already xi = 1 in the flat part.
            pass

        return xi


def derenv_flattop_gauss(
        t, t_pulse, t_rise, sigma, t_start=0, remove_discontinuities=True,
        normalize=True, **not_needed_kwargs):
    """
    Time derivative of a flattopped pulse with Gaussian edges.
    Takes the same arguments as `envelope_flattop_gauss`, which describe
    the corresponding envelope whose derivative is to be calculated.
    See Also
    --------
    envelope_flattop_gauss
    """
    if t_rise > t_pulse/2:
        t_rise = t_pulse/2

    if t < t_start or t > t_start + t_pulse:
        return 0

    else:
        t_left = t_start + t_rise
        t_right = t_start + t_pulse - t_rise
        t_flat = t_right - t_left
        # Without shift and normalization.
        if t < t_left:
            dxi_dt = ((-(t-t_left) / sigma**2)
                      * np.exp(-0.5 * ((t-t_left)/sigma)**2))
        elif t > t_right:
            dxi_dt = ((-(t-t_right) / sigma**2)
                      * np.exp(-0.5 * ((t-t_right)/sigma)**2))
        else:
            dxi_dt = 0

        if remove_discontinuities:
            if normalize:
                # Integrate xi(t) over the pulse duration.
                integral_xi_dt = (
                        np.sqrt(2*np.pi) * sigma
                        * scipy.special.erf(t_rise / (np.sqrt(2)*sigma))
                        + t_flat
                        - t_pulse * np.exp(-0.5 * (t_rise/sigma)**2))
                dxi_dt *= t_pulse / integral_xi_dt
            else:
                # Ensure that xi = 1 in the flat part.
                dxi_dt /= (1 - np.exp(-0.5 * (t_rise/sigma)**2))

        elif normalize:
            # Integrate xi(t) over the pulse duration.
            integral_xi_dt = (
                    np.sqrt(2*np.pi) * sigma
                    * scipy.special.erf(t_rise / (np.sqrt(2)*sigma))
                    + t_flat)
            dxi_dt *= t_pulse / integral_xi_dt
        else:
            # Already xi = 1 in the flat part.
            pass

        return dxi_dt


def envelope_flattop_cos(t, t_pulse, t_rise, t_start=0,
                         normalize=True, **not_needed_kwargs):
    """
    Envelope for a flattopped pulse with cosine-shaped edges.
    Parameters
    ----------
    t : float
        Time at which the value of the envelope is to be returned.
    t_pulse : float
        The total duration of the pulse.
    t_rise : float
        The durations of the rising and lowering edges of the pulse.
        Replaced by ``t_pulse/2`` when ``t_rise > t_pulse/2``.
    t_start : float, default=0
        Time at which the pulse starts.
    normalize : bool, default=True
        If True, normalize the envelope xi(t) such that <xi(t)> = 1.
        If False, normalize such that max[xi(t)] = 1.
    Returns
    -------
    float
        The value of the envelope at a given time.
    See Also
    --------
    envelope_cos, derenv_flattop_gauss
    """
    if t_rise > t_pulse/2:
        t_rise = t_pulse/2

    if t < t_start or t > t_start + t_pulse:
        return 0

    else:
        t_left = t_start + t_rise
        t_right = t_start + t_pulse - t_rise
        t_flat = t_right - t_left
        # Without shift and normalization.
        if t < t_left:
            xi = np.sin(0.5 * np.pi * (t-t_start) / t_rise)**2
        elif t > t_right:
            xi = np.sin(0.5 * np.pi * (t-t_start-t_pulse) / t_rise)**2
        else:
            xi = 1

        if normalize:
            # Make <xi> = 1.
            return xi * t_pulse / (t_flat+t_rise)
        else:
            return xi


def derenv_flattop_cos(t, t_pulse, t_rise, t_start=0,
                       normalize=True, **not_needed_kwargs):
    """
    Time derivative of a flattopped pulse with cosine-shaped edges.
    Takes the same arguments as `envelope_flattop_cos`, which describe
    the corresponding envelope whose derivative is to be calculated.
    See Also
    --------
    envelope_flattop_cos
    """
    if t_rise > t_pulse/2:
        t_rise = t_pulse/2

    elif t < t_start or t > t_start + t_pulse:
        return 0

    else:
        t_left = t_start + t_rise
        t_right = t_start + t_pulse - t_rise
        t_flat = t_right - t_left
        # Without shift and normalization.
        if t < t_left:
            dxi_dt = (0.5*np.pi/t_rise) * np.sin(np.pi * (t-t_start) / t_rise)
        elif t > t_right:
            dxi_dt = (0.5*np.pi/t_rise) * np.sin(
                    np.pi * (t-t_start-t_pulse) / t_rise)
        else:
            dxi_dt = 0

        if normalize:
            # Make <xi> = 1.
            return dxi_dt * t_pulse / (t_flat+t_rise)
        else:
            return dxi_dt


def envelope_generic(pshape, *args, **kwargs):
    """
    Envelope for a given pulse shape at a given time or times.
    Parameters
    ----------
    pshape : str or function object
        Pulse shape type or user-provided function.
        Allowed string values are 'square', 'gauss', 'cos',
        'flattop_gauss', 'flattop_cos', 'user'.
    *args and **kwargs
        Positional and keyword arguments to pass on to a pulse shaping
        function.
    See Also
    --------
    derenv_generic
    """
    if callable(pshape):
        return pshape(*args, **kwargs)
    elif pshape == 'square':
        return envelope_square(*args, **kwargs)
    elif pshape == 'gauss':
        return envelope_gauss(*args, **kwargs)
    elif pshape == 'cos':
        return envelope_cos(*args, **kwargs)
    elif pshape == 'flattop_gauss':
        return envelope_flattop_gauss(*args, **kwargs)
    elif pshape == 'flattop_cos':
        return envelope_flattop_cos(*args, **kwargs)
    else:
        raise ValueError(
                '`pshape` must be a functin object or one of ' +
                '"square", "gauss", "cos", "flattop_gauss", "flattop_cos"')
        return None


def derenv_generic(pshape, *args, **kwargs):
    """
    Time derivative of the envelope for a given pulse shape.
    Takes the same arguments as `envelope_generic`, which describe
    the corresponding envelope whose derivative is to be calculated.
    See Also
    --------
    envelope_generic
    """
    if callable(pshape):
        return pshape(*args, **kwargs)
    elif pshape == 'square':
        return derenv_square(*args, **kwargs)
    elif pshape == 'gauss':
        return derenv_gauss(*args, **kwargs)
    elif pshape == 'cos':
        return derenv_cos(*args, **kwargs)
    elif pshape == 'flattop_gauss':
        return derenv_flattop_gauss(*args, **kwargs)
    elif pshape == 'flattop_cos':
        return derenv_flattop_cos(*args, **kwargs)
    else:
        raise ValueError(
                '`pshape` must be a functin object or one of ' +
                '"square", "gauss", "cos", "flattop_gauss", "flattop_cos"')
        return None