"""Bounded tests for frozen Rabi validation artifacts."""

from pathlib import Path

import numpy as np
import qutip as qt

from Circuit_Objs.qchard_idealgridium import IdealGridium
from Simulations.Rabi_3Photon import generate_validation_artifacts as artifacts
from Simulations.Rabi_3Photon import workflow_funcs as workflow


def _unitary_trajectory(dimension=6):
    propagators = []
    for angle in (0.0, 0.2, 0.35):
        matrix = np.eye(dimension, dtype=complex)
        matrix[0, 0] = np.cos(angle)
        matrix[0, 2] = -np.sin(angle)
        matrix[2, 0] = np.sin(angle)
        matrix[2, 2] = np.cos(angle)
        propagators.append(qt.Qobj(matrix))
    return np.asarray(propagators)


def _qubit():
    return IdealGridium(
        E_L=0.5, E_C=0.5, E_s=4, E_2J=12,
        ng=0, phi_ext=np.pi, nlev=6, nlev_lc=40, units='GHz')


def _pulse_configs():
    return [
        workflow.PulseConfig(
            T_gate=1.0, pulse_shape='cos', targeted_drive=(0, 1),
            drive_amplitude_factor=0.1 + 0.02 * index,
            drive_detuning=0.001 * index, carrier_phase=0.2 * index)
        for index in range(3)
    ]


def _synthetic_validation():
    qubit = _qubit()
    times = np.array([0.0, 0.5, 1.0])
    propagators = _unitary_trajectory(qubit.nlev)
    metrics = workflow.evaluate_logical_gate(
        propagators, 'X90', t_points=times)
    case = {
        'qubit': qubit,
        't_points': times,
        'propagators': propagators,
        'metrics': metrics,
        'unitarity_residual': 1e-14,
    }
    experiment = {
        'target': 'X90',
        'evaluation_frame': 'bare_energy_interaction',
        'model': {'nlev': 6, 'nlev_lc': 40},
        'pulse': {'shape': 'cos', 'sigma': 0.25},
        'transitions': {
            'outer_05': [0, 1],
            'middle_54': [0, 1],
            'outer_41': [0, 1],
        },
    }
    parameters = np.array([
        0.10, 0.12, 0.14, 0.20, 0.40, 1.0,
        0.0, 0.001, 0.002,
    ])
    return {
        'experiment': experiment,
        'parameters': parameters,
        'solver_cases': {'bounded': case},
        'retained_level_cases': {6: case},
        'lc_cutoff_cases': {40: case},
        'reference': case,
    }


def test_state_populations_are_normalized_and_match_leakage_metrics():
    propagators = _unitary_trajectory()
    times = np.array([0.0, 0.5, 1.0])
    populations = workflow.state_population_trajectories(propagators)
    metrics = workflow.evaluate_logical_gate(
        propagators, 'X180', t_points=times)

    for initial in (0, 1):
        np.testing.assert_allclose(
            np.sum(populations[initial], axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(
            np.sum(populations[initial][:, 2:], axis=1),
            metrics['state_diagnostics'][initial]['leakage'], atol=1e-14)
    np.testing.assert_allclose(
        0.5 * (
            np.sum(populations[0][:, 2:], axis=1)
            + np.sum(populations[1][:, 2:], axis=1)),
        metrics['leakage_by_time'], atol=1e-14)


def test_drive_trace_helper_returns_three_finite_actual_coefficients():
    times = np.linspace(0.0, 1.0, 21)
    traces = workflow.drive_coefficient_traces(
        _qubit(), _pulse_configs(), times)

    assert len(traces) == 3
    for trace in traces:
        assert trace['values'].shape == times.shape
        assert np.all(np.isfinite(trace['values']))
        assert np.max(np.abs(trace['values'])) > 0.0


def test_artifact_generation_writes_nonempty_figures_and_tables(tmp_path):
    config_path = tmp_path / 'synthetic_frozen_candidate.yaml'
    config_path.write_text('target: X90\n')
    generated = artifacts.write_validation_artifacts(
        _synthetic_validation(), config_path, tmp_path)

    assert generated['summary']['target_gate'] == 'X90'
    assert generated['summary']['control_operator'].startswith(
        'IdealGridium.phi()')
    assert 'intermediate_populations' in generated['paths']
    for path in generated['paths'].values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
