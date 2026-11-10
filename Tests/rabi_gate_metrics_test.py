"""Analytic tests for pathway-independent logical gate metrics."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import qutip as qt


@pytest.fixture(scope='module')
def workflow():
    path = (
        Path(__file__).parents[1]
        / 'Simulations' / 'Rabi_3Photon' / 'workflow_funcs.py')
    spec = importlib.util.spec_from_file_location(
        'rabi_gate_metrics_test_module', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _embedded_logical_gate(logical_gate, dimension=3, global_phase=0.0):
    matrix = np.eye(dimension, dtype=complex)
    matrix[:2, :2] = np.exp(1j * global_phase) * logical_gate
    return qt.Qobj(matrix)


def test_perfect_x180_up_to_global_phase(workflow):
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    propagator = _embedded_logical_gate(
        pauli_x, dimension=3, global_phase=0.37)

    metrics = workflow.evaluate_logical_gate(propagator, 'X180')

    assert metrics['logical_process_fidelity'] == pytest.approx(1.0)
    assert metrics['logical_gate_fidelity'] == pytest.approx(1.0)
    assert metrics['final_leakage'] == pytest.approx(0.0)
    assert metrics['peak_leakage'] == pytest.approx(0.0)
    np.testing.assert_allclose(
        metrics['state_diagnostics'][0]['logical_populations'][-1],
        [0.0, 1.0])
    np.testing.assert_allclose(
        metrics['state_diagnostics'][1]['logical_populations'][-1],
        [1.0, 0.0])


def test_perfect_x90_up_to_global_phase(workflow):
    x90 = workflow.logical_gate_target('X90').full()
    propagator = _embedded_logical_gate(
        x90, dimension=4, global_phase=-0.81)

    metrics = workflow.evaluate_logical_gate(propagator, 'x90')

    assert metrics['logical_process_fidelity'] == pytest.approx(1.0)
    assert metrics['logical_gate_fidelity'] == pytest.approx(1.0)
    assert metrics['final_leakage'] == pytest.approx(0.0)
    for initial in (0, 1):
        np.testing.assert_allclose(
            metrics['state_diagnostics'][initial]
            ['logical_populations'][-1],
            [0.5, 0.5])


def test_known_final_and_transient_leakage(workflow):
    identity = np.eye(3, dtype=complex)
    full_rotation = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=complex)
    angle = np.pi / 6
    partial_rotation = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)],
    ], dtype=complex)

    metrics = workflow.evaluate_logical_gate(
        [qt.Qobj(identity), qt.Qobj(full_rotation),
         qt.Qobj(partial_rotation)],
        'X180', t_points=[0.0, 1.0, 2.0])

    # Input |0> has final leakage sin(pi/6)^2 = 1/4; input |1>
    # never leaks. Aggregate leakage is their arithmetic mean.
    assert metrics['state_diagnostics'][0]['final_leakage'] == pytest.approx(.25)
    assert metrics['state_diagnostics'][1]['final_leakage'] == pytest.approx(0.0)
    assert metrics['final_leakage'] == pytest.approx(.125)
    assert metrics['state_diagnostics'][0]['peak_leakage'] == pytest.approx(1.0)
    assert metrics['peak_leakage'] == pytest.approx(.5)
    assert metrics['peak_leakage_index'] == 1
    assert metrics['peak_leakage_time'] == pytest.approx(1.0)


def test_zero_leakage_trajectory(workflow):
    identity = qt.qeye(4)
    x180 = _embedded_logical_gate(
        workflow.logical_gate_target('X180').full(), dimension=4)

    metrics = workflow.evaluate_logical_gate(
        [identity, x180], 'X180', t_points=[0.0, 1.0])

    np.testing.assert_allclose(metrics['leakage_by_time'], [0.0, 0.0])
    assert metrics['final_leakage'] == pytest.approx(0.0)
    assert metrics['peak_leakage'] == pytest.approx(0.0)
    for initial in (0, 1):
        np.testing.assert_allclose(
            metrics['state_diagnostics'][initial]['leakage'], [0.0, 0.0])
