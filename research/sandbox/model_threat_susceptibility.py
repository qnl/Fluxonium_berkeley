"""First-order model-parameter susceptibilities from the saved N=71 checkpoint.

No modified large Stage-1 eigensystem is solved.  Derivative operators are
assembled in the raw basis, projected into the authenticated saved k=180
Stage-1 subspace, and analyzed in the resulting 180 x 8 Stage-2 space.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la

from Circuit_Objs.qchard_gridium_netlist import (
    _assemble_nonlinear_sector,
    _coeffs,
)
from research.sandbox.state_0178_checkpoint_diagnostics import (
    N71_DIR,
    ROOT,
    _load_n71,
)


OUTPUT_JSON = ROOT / "research/sandbox/model_threat_susceptibility.json"
OUTPUT_CSV = ROOT / "research/sandbox/model_threat_path_susceptibility.csv"
PATHS = (
    (0, 7, 8, 1),
    (0, 3, 8, 1),
    (0, 7, 2, 1),
    (0, 3, 6, 1),
)
PARAMETERS = (
    "EJ", "EC", "EL", "ELK", "EJS", "ECS", "eC", "eP",
    "eps_J", "eps_LK", "ng", "phi_ext", "theta_ext",
)


def _coefficients(parameters):
    return _coeffs(
        parameters["EJ"], parameters["EC"], parameters["EL"],
        parameters["ELK"], parameters["EJS"], parameters["ECS"],
        parameters["eps_J"], parameters["eps_LK"],
        parameters["eC"], parameters["eP"],
    )


def _assemble_stage1(parameters, spatial):
    ecm, stiffness, ej1, ej2 = _coefficients(parameters)
    hamiltonian, _ = _assemble_nonlinear_sector(
        ecm[:3, :3], stiffness[:2, :2], ej1, ej2,
        parameters["EJS"], parameters["ng"], parameters["phi_ext"],
        parameters["theta_ext"], spatial["n1max"], spatial["N2"],
        spatial["L2"], spatial["N3"], spatial["L3"],
    )
    return hamiltonian


def _difference_step(name, value):
    if name in {"eps_J", "eps_LK", "ng", "phi_ext", "theta_ext"}:
        return 1e-5
    return 1e-5 * max(abs(value), 1.0)


def _report_step(name, value):
    if name in {"eps_J", "eps_LK"}:
        return 0.01, "+0.01 absolute asymmetry"
    if name == "ng":
        return 0.01, "+0.01 Cooper-pair offset charge"
    if name in {"phi_ext", "theta_ext"}:
        return 0.01, "+0.01 radian bias displacement"
    return 0.01 * value, "+1% parameter change"


def _baseline_oscillator(ecm, stiffness, count):
    a = np.diag(np.sqrt(np.arange(1, count)), 1)
    a4 = 4 * ecm[3, 3]
    b4 = 0.5 * stiffness[2, 2]
    f4 = (a4 / (4 * b4)) ** 0.25
    g4 = 1.0 / (2 * f4)
    x4 = f4 * (a + a.T)
    n4 = 1j * g4 * (a.T - a)
    return x4, n4


def _stage2_hamiltonian(stage1_values, operators, parameters, n4_count):
    ecm, stiffness, _, _ = _coefficients(parameters)
    x4, n4 = _baseline_oscillator(ecm, stiffness, n4_count)
    a4 = 4 * ecm[3, 3]
    b4 = 0.5 * stiffness[2, 2]
    w4 = 2 * np.sqrt(a4 * b4) * (np.arange(n4_count) + 0.5)
    identity1 = np.eye(len(stage1_values))
    identity4 = np.eye(n4_count)
    hamiltonian = np.kron(
        np.diag(stage1_values), identity4
    ).astype(complex)
    hamiltonian += np.kron(identity1, np.diag(w4))
    for index, name in enumerate(("n1", "n2", "n3")):
        hamiltonian += 8 * ecm[index, 3] * np.kron(operators[name], n4)
    for index, name in enumerate(("x2", "x3")):
        hamiltonian += stiffness[index, 2] * np.kron(operators[name], x4)
    return 0.5 * (hamiltonian + hamiltonian.conj().T), x4, n4


def _derivative_operator(name, parameters, spatial, stage1_vectors,
                         operators, baseline_x4, baseline_n4,
                         step_scale=1.0):
    step = step_scale * _difference_step(name, parameters[name])
    plus = dict(parameters)
    minus = dict(parameters)
    plus[name] += step
    minus[name] -= step

    raw_derivative = (
        _assemble_stage1(plus, spatial) - _assemble_stage1(minus, spatial)
    ) * (0.5 / step)
    projected = stage1_vectors.conj().T @ (raw_derivative @ stage1_vectors)
    projected = 0.5 * (projected + projected.conj().T)

    ecm_plus, stiffness_plus, _, _ = _coefficients(plus)
    ecm_minus, stiffness_minus, _, _ = _coefficients(minus)
    decm = (ecm_plus - ecm_minus) * (0.5 / step)
    dstiffness = (stiffness_plus - stiffness_minus) * (0.5 / step)

    count1 = projected.shape[0]
    count4 = baseline_x4.shape[0]
    identity1 = np.eye(count1)
    identity4 = np.eye(count4)
    derivative = np.kron(projected, identity4)
    derivative += np.kron(
        identity1,
        4 * decm[3, 3] * (baseline_n4 @ baseline_n4)
        + 0.5 * dstiffness[2, 2] * (baseline_x4 @ baseline_x4),
    )
    for index, operator_name in enumerate(("n1", "n2", "n3")):
        derivative += 8 * decm[index, 3] * np.kron(
            operators[operator_name], baseline_n4
        )
    for index, operator_name in enumerate(("x2", "x3")):
        derivative += dstiffness[index, 2] * np.kron(
            operators[operator_name], baseline_x4
        )
    return 0.5 * (derivative + derivative.conj().T), step


def _state_response(derivative_eigenbasis, values, state, report_step):
    denominators = values[state] - values
    coefficients = np.zeros(len(values), dtype=complex)
    mask = np.arange(len(values)) != state
    coefficients[mask] = derivative_eigenbasis[mask, state] / denominators[mask]
    order = np.argsort(np.abs(coefficients))[::-1][:6]
    return coefficients, {
        "first_order_state_derivative_norm_per_parameter_unit": float(
            np.linalg.norm(coefficients)
        ),
        "linearized_state_distance_at_report_step": float(
            abs(report_step) * np.linalg.norm(coefficients)
        ),
        "dominant_mixing_channels_at_report_step": [
            {
                "state": int(other),
                "amplitude": float(abs(report_step * coefficients[other])),
                "energy_denominator_GHz": float(denominators[other]),
            }
            for other in order if other != state
        ],
    }


def _path_response(path, values, energy_derivative, grid_matrix,
                   grid_derivative, report_step):
    pairs = list(zip(path[:-1], path[1:]))
    edge_magnitudes = [float(abs(grid_matrix[a, b])) for a, b in pairs]
    edge_derivatives = []
    for (initial, final), magnitude in zip(pairs, edge_magnitudes):
        element = grid_matrix[initial, final]
        derivative = grid_derivative[initial, final]
        edge_derivatives.append(float(
            np.real(np.conj(element) * derivative) / magnitude
        ))
    carrier_derivatives = []
    for initial, final in pairs:
        sign = np.sign(values[final] - values[initial])
        carrier_derivatives.append(float(
            sign * (energy_derivative[final] - energy_derivative[initial])
        ))
    logarithmic_score_derivative = float(sum(
        derivative / magnitude
        for derivative, magnitude in zip(edge_derivatives, edge_magnitudes)
    ))
    return {
        "path": list(path),
        "edge_magnitude_derivatives_per_parameter_unit": edge_derivatives,
        "edge_relative_changes_percent_at_report_step": [
            float(100 * report_step * derivative / magnitude)
            for derivative, magnitude in zip(edge_derivatives, edge_magnitudes)
        ],
        "carrier_derivatives_GHz_per_parameter_unit": carrier_derivatives,
        "carrier_changes_MHz_at_report_step": [
            float(1e3 * report_step * derivative)
            for derivative in carrier_derivatives
        ],
        "score_logarithmic_derivative_per_parameter_unit": (
            logarithmic_score_derivative
        ),
        "score_relative_change_percent_at_report_step": float(
            100 * report_step * logarithmic_score_derivative
        ),
    }


def _node_capacitance_matrix(parameters):
    matrix = np.zeros((4, 4))
    matrix[0, 0] += 1.0 / parameters["EC"]
    matrix[1, 1] += 1.0 / parameters["EC"]
    matrix[2, 2] += 1.0 / parameters["ECS"]
    vector = np.array([1.0, -1.0, 0.0, 0.0])
    matrix += np.outer(vector, vector) / parameters["eC"]
    matrix[3, 3] += 1.0 / parameters["eP"]
    return matrix


def main():
    metadata, stage1_values, stage1_vectors, operators = _load_n71()
    parameters = dict(metadata["configuration"]["physical_parameters"])
    spatial = metadata["configuration"]["spatial_cutoffs"]
    n4_count = spatial["N4"]
    ecm, stiffness, ej1, ej2 = _coefficients(parameters)
    hamiltonian, x4, n4 = _stage2_hamiltonian(
        stage1_values, operators, parameters, n4_count
    )
    values, vectors = la.eigh(hamiltonian, check_finite=False)
    grid_operator = np.kron(
        0.5 * operators["x2"] + operators["x3"], np.eye(n4_count)
    )
    grid_eigenbasis = vectors.conj().T @ grid_operator @ vectors

    saved_analysis = json.loads((N71_DIR / "analysis.json").read_text())
    saved_values = np.asarray(
        saved_analysis["n71_reference_k180"]["energies_GHz"]
    )
    saved_grid = np.asarray(
        saved_analysis["n71_reference_k180"]["grid_phi_abs"]
    )

    results = {}
    for name in PARAMETERS:
        derivative, difference_step = _derivative_operator(
            name, parameters, spatial, stage1_vectors, operators, x4, n4
        )
        derivative_eigenbasis = vectors.conj().T @ derivative @ vectors
        energy_derivative = np.real(np.diag(derivative_eigenbasis))
        report_step, report_step_description = _report_step(
            name, parameters[name]
        )
        derivative_step_check = None
        if name in {"eC", "eP", "eps_J", "eps_LK"}:
            half_step_derivative, half_step = _derivative_operator(
                name, parameters, spatial, stage1_vectors, operators, x4, n4,
                step_scale=0.5,
            )
            derivative_step_check = {
                "half_finite_difference_step": half_step,
                "relative_Frobenius_change_full_vs_half_step": float(
                    np.linalg.norm(derivative - half_step_derivative)
                    / np.linalg.norm(half_step_derivative)
                ),
            }

        response_coefficients = {}
        state_response = {}
        for state in range(11):
            coefficients, response = _state_response(
                derivative_eigenbasis, values, state, report_step
            )
            response_coefficients[state] = coefficients
            state_response[str(state)] = response

        coefficients = np.column_stack([
            response_coefficients[state] for state in range(11)
        ])
        grid_derivative = (
            coefficients.conj().T @ grid_eigenbasis[:, :11]
            + grid_eigenbasis[:11, :] @ coefficients
        )
        f78_derivative = float(energy_derivative[8] - energy_derivative[7])
        results[name] = {
            "baseline_value": parameters[name],
            "finite_difference_step_for_operator": difference_step,
            "finite_difference_step_check": derivative_step_check,
            "report_step": report_step,
            "report_step_description": report_step_description,
            "low11_energy_derivatives_GHz_per_parameter_unit": (
                energy_derivative[:11].tolist()
            ),
            "low11_energy_changes_MHz_at_report_step": (
                1e3 * report_step * energy_derivative[:11]
            ).tolist(),
            "f78_derivative_GHz_per_parameter_unit": f78_derivative,
            "f78_change_MHz_at_report_step": float(
                1e3 * report_step * f78_derivative
            ),
            "state_response": state_response,
            "paths": {
                "-".join(map(str, path)): _path_response(
                    path, values, energy_derivative, grid_eigenbasis,
                    grid_derivative, report_step,
                )
                for path in PATHS
            },
        }

    transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    output = {
        "scope": (
            "first-order fixed-k180-subspace response from the authenticated "
            "N71 checkpoint; no modified Stage-1 eigensolve"
        ),
        "authenticated_input": {
            "directory": str(N71_DIR.relative_to(ROOT)),
            "configuration_sha256": metadata["configuration_sha256"],
            "file_sha256": metadata["file_sha256"],
        },
        "baseline_parameters": parameters,
        "baseline_node_capacitance_matrix_in_inverse_energy_units": (
            _node_capacitance_matrix(parameters).tolist()
        ),
        "coordinate_transform_M": transform.tolist(),
        "baseline_transformed_inverse_capacitance_matrix_ECmat": ecm.tolist(),
        "baseline_inductive_stiffness_matrix_K": stiffness.tolist(),
        "baseline_KITE_junction_energies": {"EJ1": ej1, "EJ2": ej2},
        "reproduction_checks": {
            "maximum_low11_energy_difference_GHz": float(np.max(np.abs(
                values[:11] - saved_values
            ))),
            "maximum_low11_abs_grid_phi_difference": float(np.max(np.abs(
                np.abs(grid_eigenbasis[:11, :11]) - saved_grid
            ))),
        },
        "method": {
            "energy_derivative": "Hellmann-Feynman <n|dH/dp|n>",
            "state_derivative": (
                "sum over all other eigenvectors in the retained 1440-state "
                "Stage-2 space: |k><k|dH/dp|n>/(E_n-E_k)"
            ),
            "grid_phi_derivative": (
                "first-order bra and ket response; grid_phi coordinate held "
                "fixed in the netlist coordinate basis"
            ),
            "important_caveat": (
                "responses omit Stage-1 states above saved rank 179 and are "
                "linearized local derivatives, not finite-sweep eigensolves"
            ),
        },
        "parameter_susceptibilities": results,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    with OUTPUT_CSV.open("w", newline="") as handle:
        fields = [
            "parameter", "baseline", "report_step", "path",
            "score_change_percent", "edge_changes_percent",
            "carrier_changes_MHz", "f78_change_MHz",
            "state7_distance", "state8_distance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            for path_name, path_result in result["paths"].items():
                writer.writerow({
                    "parameter": name,
                    "baseline": result["baseline_value"],
                    "report_step": result["report_step"],
                    "path": path_name,
                    "score_change_percent": path_result[
                        "score_relative_change_percent_at_report_step"
                    ],
                    "edge_changes_percent": ";".join(map(
                        str, path_result[
                            "edge_relative_changes_percent_at_report_step"
                        ]
                    )),
                    "carrier_changes_MHz": ";".join(map(
                        str, path_result["carrier_changes_MHz_at_report_step"]
                    )),
                    "f78_change_MHz": result[
                        "f78_change_MHz_at_report_step"
                    ],
                    "state7_distance": result["state_response"]["7"][
                        "linearized_state_distance_at_report_step"
                    ],
                    "state8_distance": result["state_response"]["8"][
                        "linearized_state_distance_at_report_step"
                    ],
                })
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
