"""Narrow execution tests for the IdealGridium Rabi multitone workflow."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from Circuit_Objs.qchard_idealgridium import IdealGridium


@pytest.fixture(scope='module')
def workflow():
    path = (
        Path(__file__).parents[1]
        / 'Simulations' / 'Rabi_3Photon' / 'workflow_funcs.py')
    spec = importlib.util.spec_from_file_location(
        'rabi_3photon_workflow_test_module', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def idealgridium(workflow):
    """IdealGridium interface fixture with no numerical eigensolve."""
    class IdealGridiumStub(IdealGridium):
        def __init__(self):
            self.nlev = 4
            self._levels = np.array([0.0, 0.5, 1.2, 2.1])
            self._phi = workflow.qt.Qobj(np.array([
                [0.0, 0.8, 0.0, 0.0],
                [0.8, 0.0, 0.6, 0.0],
                [0.0, 0.6, 0.0, 0.4],
                [0.0, 0.0, 0.4, 0.0],
            ]))

        def levels(self, nlev=None):
            return self._levels[:nlev]

        def freq(self, initial, final):
            return self._levels[final] - self._levels[initial]

        def phi(self):
            return self._phi

        def H(self):
            return workflow.qt.Qobj(np.diag(self._levels))

    return IdealGridiumStub()


def _pulse_configs(workflow, tone_count):
    transitions = ((0, 1), (1, 2), (2, 3))
    shapes = ('square', 'cos', 'gauss')
    return [
        workflow.PulseConfig(
            T_gate=1.0 + 0.1 * index,
            T_start=0.1 * index,
            pulse_shape=shapes[index],
            pulse_sigma=0.2,
            targeted_drive=transitions[index],
            drive_amplitude_factor=0.05 * (index + 1),
            carrier_phase=0.2 * index,
            drive_detuning=0.01 * (index + 1),
            drive_type='flux')
        for index in range(tone_count)
    ]


@pytest.mark.parametrize('tone_count', [1, 2, 3])
def test_rabi_workflow_executes_independent_flux_tones(
        workflow, idealgridium, tone_count):
    configs = _pulse_configs(workflow, tone_count)

    t_points, propagators = workflow.solve(
        idealgridium, configs, mute=True)

    assert len(propagators) == len(t_points)
    assert propagators[-1].shape == (idealgridium.nlev, idealgridium.nlev)
    np.testing.assert_allclose(
        (propagators[-1].dag() * propagators[-1]).full(),
        np.eye(idealgridium.nlev), atol=3e-4)


def test_tone_configuration_maps_to_existing_drive_arguments(
        workflow, idealgridium):
    config = workflow.PulseConfig(
        T_gate=3.0,
        T_start=1.25,
        T_rise=0.4,
        pulse_shape='cos',
        pulse_sigma=0.3,
        targeted_drive=(0, 1),
        drive_amplitude_factor=0.7,
        carrier_phase=0.6,
        drive_detuning=0.125,
        DRAG=True,
        DRAG_coeff=0.2,
        drive_type='flux')

    phi_operator = idealgridium.phi()
    term = workflow._pulse_to_drive_term(
        idealgridium, config, phi_operator)

    assert term['operator'] == phi_operator / abs(phi_operator[0, 1])
    assert term['amplitude'] == config.drive_amplitude_factor
    assert term['omega_d'] == pytest.approx(
        abs(idealgridium.freq(0, 1)) + config.drive_detuning)
    assert term['phi'] == config.carrier_phase
    assert term['shape'] == config.pulse_shape
    assert term['sigma'] == config.pulse_sigma
    assert term['T_start'] == config.T_start
    assert term['T_gate'] == config.T_gate
    assert term['T_rise'] == config.T_rise
    assert term['DRAG'] is True
    assert term['DRAG_coefficient'] == config.DRAG_coeff


def test_explicit_frequency_is_detuned_and_nonflux_drive_is_rejected(
        workflow, idealgridium):
    config = workflow.PulseConfig(
        T_gate=1.0,
        pulse_shape='square',
        targeted_drive=(0, 1),
        drive_frequency=2.0,
        drive_detuning=-0.25)
    assert workflow._carrier_frequency(idealgridium, config) == 1.75

    config.drive_type = 'charge'
    with pytest.raises(ValueError, match='only drive_type'):
        workflow._pulse_to_drive_term(
            idealgridium, config, idealgridium.phi())


def test_three_tone_yaml_wrapper_executes(
        workflow, idealgridium, tmp_path):
    paths = []
    for index, config in enumerate(_pulse_configs(workflow, 3)):
        path = tmp_path / 'tone_{}.yaml'.format(index + 1)
        path.write_text(
            '\n'.join([
                'T_gate: {}'.format(config.T_gate),
                'T_start: {}'.format(config.T_start),
                'pulse_shape: {}'.format(config.pulse_shape),
                'pulse_sigma: {}'.format(config.pulse_sigma),
                'targeted_drive: [{}, {}]'.format(*config.targeted_drive),
                'drive_amplitude_factor: {}'.format(
                    config.drive_amplitude_factor),
                'carrier_phase: {}'.format(config.carrier_phase),
                'drive_detuning: {}'.format(config.drive_detuning),
                'drive_type: flux',
            ]))
        paths.append(path)

    t_points, propagators = workflow.solve_three_tone_drive(
        idealgridium, *paths, mute=True)

    assert len(propagators) == len(t_points)
    assert propagators[-1].shape == (idealgridium.nlev, idealgridium.nlev)

    figure = workflow.solve_two_photon_drive(
        idealgridium, paths[0], paths[1], n_shown_states=2)
    assert len(figure.axes) == 2


@pytest.fixture(scope='module')
def four_mode_gridium(workflow):
    """Minimal four-mode interface without a circuit eigensolve."""
    class FourModeStub:
        nlev = 5

        def __init__(self):
            self._levels = np.array([0.0, 0.4, 1.1, 1.9, 2.8])
            matrix = np.array([
                [0.0, 0.7, 0.0, 0.3, 0.0],
                [0.7, 0.0, 0.2, 0.0, 0.4],
                [0.0, 0.2, 0.0, 0.5, 0.0],
                [0.3, 0.0, 0.5, 0.0, 0.6],
                [0.0, 0.4, 0.0, 0.6, 0.0],
            ])
            self._grid_phi = workflow.qt.Qobj(matrix)

        def levels(self, nlev=None):
            return self._levels[:nlev]

        def grid_phi(self):
            return self._grid_phi

    return FourModeStub()


def test_four_mode_selects_grid_phi_and_propagates_three_tones(
        workflow, four_mode_gridium):
    operator = four_mode_gridium.grid_phi()
    assert workflow._global_phase_operator(four_mode_gridium) == operator
    config = workflow.PulseConfig(
        T_gate=1.0, pulse_shape='square', targeted_drive=(0, 1),
        drive_amplitude_factor=0.05, drive_type='flux')
    term = workflow._pulse_to_drive_term(
        four_mode_gridium, config, operator)
    assert term['operator'] == operator / abs(operator[0, 1])

    transitions = ((0, 1), (0, 3), (1, 4))
    configs = [workflow.PulseConfig(
        T_gate=1.0, T_start=0.0, pulse_shape='square',
        targeted_drive=transition, drive_amplitude_factor=0.03,
        drive_type='flux') for transition in transitions]
    t_points, propagators = workflow.solve(
        four_mode_gridium, configs, mute=True)
    assert len(propagators) == len(t_points)
    assert propagators[-1].shape == (four_mode_gridium.nlev,
                                     four_mode_gridium.nlev)
    np.testing.assert_allclose(
        (propagators[-1].dag() * propagators[-1]).full(),
        np.eye(four_mode_gridium.nlev), atol=3e-4)
