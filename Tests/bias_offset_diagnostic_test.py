"""Narrow tests for the one-mode IdealGridium bias-offset diagnostic."""

import numpy as np
import pytest

from Circuit_Objs.qchard_idealgridium import IdealGridium
from Simulations.Rabi_3Photon import diagnose_bias_offset as diagnostic
from Simulations.Rabi_3Photon import workflow_funcs as workflow


@pytest.fixture(scope='module')
def experiment():
    return diagnostic.comparison._load_config(
        diagnostic.DEFAULT_CONFIGS['X90'])


@pytest.fixture(scope='module')
def qubits(experiment):
    model = dict(experiment['model'])
    model.update(nlev=8, nlev_lc=40)
    model['phi_ext'] = np.pi
    reference = IdealGridium(**model)
    model['phi_ext'] = np.pi + 0.0025
    shifted = IdealGridium(**model)
    return reference, shifted


def test_frozen_pulse_preserves_carrier_and_flux_operator_scale(
        experiment, qubits):
    reference, shifted = qubits
    tracking = diagnostic._track(reference, shifted)
    parameters = diagnostic._candidate_vector(experiment)
    configs = diagnostic._pulse_configs(
        experiment, parameters, reference, shifted, tracking,
        follow_transitions=False)

    reference_phi = reference.phi()
    shifted_phi = shifted.phi()
    for index, (name, config) in enumerate(zip(
            diagnostic.TRANSITION_NAMES, configs)):
        reference_transition = tuple(experiment['transitions'][name])
        shifted_transition = diagnostic._tracked_transition(
            reference_transition, tracking)
        term = workflow._pulse_to_drive_term(shifted, config, shifted_phi)
        assert term['omega_d'] == pytest.approx(
            abs(reference.freq(*reference_transition)) + parameters[6 + index])
        expected_scale = parameters[index] / abs(
            reference_phi[reference_transition[0], reference_transition[1]])
        np.testing.assert_allclose(
            term['amplitude'] * term['operator'].full(),
            expected_scale * shifted_phi.full())
        assert config.targeted_drive == shifted_transition


def test_carrier_following_changes_only_bare_transition_frequency(
        experiment, qubits):
    reference, shifted = qubits
    tracking = diagnostic._track(reference, shifted)
    parameters = diagnostic._candidate_vector(experiment)
    frozen = diagnostic._pulse_configs(
        experiment, parameters, reference, shifted, tracking,
        follow_transitions=False)
    followed = diagnostic._pulse_configs(
        experiment, parameters, reference, shifted, tracking,
        follow_transitions=True)

    for index, (frozen_config, followed_config) in enumerate(
            zip(frozen, followed)):
        transition = followed_config.targeted_drive
        assert workflow._carrier_frequency(
            shifted, followed_config) == pytest.approx(
                abs(shifted.freq(*transition)) + parameters[6 + index])
        assert followed_config.drive_amplitude_factor == pytest.approx(
            frozen_config.drive_amplitude_factor)
        assert followed_config.carrier_phase == pytest.approx(
            frozen_config.carrier_phase)
        assert followed_config.T_gate == pytest.approx(frozen_config.T_gate)


def test_tracking_summary_is_limited_to_pathway_states(experiment, qubits):
    reference, shifted = qubits
    tracking = diagnostic._track(reference, shifted)
    summary = diagnostic._tracking_summary(tracking)

    assert all(
        set(item.get(
            'reference_states', (item.get('reference_state'),)))
        & set(range(6))
        for item in summary)
