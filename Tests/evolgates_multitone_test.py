"""Focused tests for independent multitone microwave evolution."""

import numpy as np
import pytest
import qutip as qt

from Circuit_Objs import qchard_evolgates as gates


def _assert_propagators_close(actual, expected, atol=1e-6):
    # Separate adaptive integrations need only agree within QuTiP's default
    # solver precision; the Hamiltonian structure is tested exactly below.
    assert len(actual) == len(expected)
    for actual_t, expected_t in zip(actual, expected):
        assert np.allclose(actual_t.full(), expected_t.full(),
                           rtol=0, atol=atol)


def _tone(operator, omega_d, amplitude=1.0, **overrides):
    tone = {
        'operator': operator,
        'omega_d': omega_d,
        'amplitude': amplitude,
        'T_gate': 2.0,
        'T_start': 0.0,
        'shape': 'cos',
        'sigma': 0.25,
        'theta': 1.3,
        'phi': 0.2,
        'DRAG': False,
        'SYMM': False,
    }
    tone.update(overrides)
    return tone


def test_single_tone_matches_existing_propagator():
    H_nodrive = 0.37 * qt.sigmaz()
    H_drive = 0.23 * qt.sigmax()
    t_points = np.linspace(0, 2.0, 17)
    tone = _tone(H_drive, omega_d=0.71)
    kwargs = {key: value for key, value in tone.items()
              if key not in {'operator', 'amplitude'}}

    expected = gates.evolution_operator_microwave(
        H_nodrive, H_drive, t_points=t_points, **kwargs)
    actual = gates.evolution_operator_multitone_microwave(
        H_nodrive, [tone], t_points=t_points)

    _assert_propagators_close(actual, expected)


@pytest.mark.parametrize('tone_count', [2, 3])
def test_independent_tones_have_no_cross_terms(tone_count):
    H_nodrive = 0.19 * qt.sigmaz()
    operators = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    tones = [
        _tone(
            operators[index],
            omega_d=0.31 + 0.17 * index,
            amplitude=0.4 + 0.2 * index,
            T_start=0.1 * index,
            T_gate=1.5 + 0.2 * index,
            phi=0.15 * index,
            shape='square' if index == 0 else 'cos',
        )
        for index in range(tone_count)
    ]
    t = 0.8

    H_terms = gates._multitone_hamiltonian(H_nodrive, tones)
    actual = H_terms[0]
    for operator, coefficient in H_terms[1:]:
        actual = actual + operator * coefficient(t, {})

    expected = 2 * np.pi * H_nodrive
    for tone in tones:
        tone_args = {key: value for key, value in tone.items()
                     if key not in {'operator', 'amplitude'}}
        expected = expected + (
            tone['amplitude'] * tone['operator']
            * gates.H_drive_coeff_gate(t, tone_args))

    assert len(H_terms) == tone_count + 1
    assert np.allclose(actual.full(), expected.full(), rtol=0, atol=1e-13)

    delayed_coefficient = H_terms[-1][1]
    assert delayed_coefficient(tones[-1]['T_start'] - 0.01, {}) == 0.0
    assert delayed_coefficient(
        tones[-1]['T_start'] + tones[-1]['T_gate'] + 0.01, {}) == 0.0


def test_zero_amplitude_tone_does_not_change_propagator():
    H_nodrive = 0.29 * qt.sigmaz()
    active_tone = _tone(0.21 * qt.sigmax(), omega_d=0.63)
    zero_tone = _tone(
        0.47 * qt.sigmay(), omega_d=1.17, amplitude=0.0,
        T_start=0.25, T_gate=1.25, phi=-0.4, shape='gauss')
    t_points = np.linspace(0, 2.0, 17)

    expected = gates.evolution_operator_multitone_microwave(
        H_nodrive, [active_tone], t_points=t_points)
    actual = gates.evolution_operator_multitone_microwave(
        H_nodrive, [active_tone, zero_tone], t_points=t_points)

    _assert_propagators_close(actual, expected)
