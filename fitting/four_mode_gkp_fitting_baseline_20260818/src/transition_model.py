
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TransitionPrediction",
    "assign_bright_transition_points",
    "assign_bright_transition_track",
    "predict_fixed_transitions",
    "simulate_transition_prediction",
]


@dataclass(frozen=True)
class TransitionPrediction:
    currents_uA: np.ndarray
    theta_ext: np.ndarray
    phi_ext: np.ndarray
    energies_GHz: np.ndarray
    initial_states: np.ndarray
    final_states: np.ndarray
    frequencies_GHz: np.ndarray
    thermal_populations: np.ndarray
    operator_matrices: dict[str, np.ndarray]
    matrix_elements: dict[str, np.ndarray]
    visibility: dict[str, np.ndarray]

    def branch_label(self, branch_index: int) -> str:
        return f"{self.initial_states[branch_index]}->{self.final_states[branch_index]}"


def _plain_json(value):
    if isinstance(value, dict):
        return {
            str(key): _plain_json(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain_json(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _prediction_cache_key(payload: dict) -> str:
    encoded = json.dumps(
        _plain_json(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _solve_hamiltonian_point(task):
    constructor_kwargs, drive_operators = task
    from Circuit_Objs.qchard_gridium_netlist import Gridium4Mode

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context = None
    else:
        context = threadpool_limits(limits=1)

    if context is None:
        model = Gridium4Mode(**constructor_kwargs)
        operator_values = {
            name: np.asarray(
                getattr(model, name)(constructor_kwargs["nlev"]).full()
            )
            for name in drive_operators
        }
        energies = np.asarray(
            model.levels(constructor_kwargs["nlev"]), dtype=float
        )
    else:
        with context:
            model = Gridium4Mode(**constructor_kwargs)
            operator_values = {
                name: np.asarray(
                    getattr(model, name)(constructor_kwargs["nlev"]).full()
                )
                for name in drive_operators
            }
            energies = np.asarray(
                model.levels(constructor_kwargs["nlev"]), dtype=float
            )
    return energies, operator_values


def _thermal_populations(
    energies_GHz: np.ndarray, temperature_K: float
) -> np.ndarray:
    if temperature_K <= 0.0:
        populations = np.zeros_like(energies_GHz)
        populations[:, 0] = 1.0
        return populations
    kB_over_h_GHz_per_K = 20.836619123
    thermal_frequency_GHz = kB_over_h_GHz_per_K * temperature_K
    shifted = energies_GHz - energies_GHz[:, :1]
    exponent = -shifted / thermal_frequency_GHz
    exponent -= np.max(exponent, axis=1, keepdims=True)
    weights = np.exp(exponent)
    return weights / np.sum(weights, axis=1, keepdims=True)


def simulate_transition_prediction(
    *,
    currents_uA: Sequence[float],
    theta_ext: Sequence[float],
    phi_ext: Sequence[float],
    circuit_parameters: dict,
    cutoffs: dict,
    nlevels: int = 15,
    initial_states: Sequence[int] = (0, 1, 2),
    drive_operators: Sequence[str] = ("d_theta",),
    temperature_K: float = 0.020,
    origin_population_floor: dict[int, float] | None = None,
    num_cpus: int = 1,
    executor_backend: str = "process",
    cache_directory: str | Path | None = None,
) -> TransitionPrediction:
    #calculate transition families and their drive visibility
    
    currents = np.asarray(currents_uA, dtype=float)
    theta = np.asarray(theta_ext, dtype=float)
    phi = np.asarray(phi_ext, dtype=float)
    if not (currents.shape == theta.shape == phi.shape):
        raise ValueError("currents_uA, theta_ext, and phi_ext must have one shape.")
    if nlevels < 2:
        raise ValueError("nlevels must be at least two.")
    origins = tuple(sorted({int(state) for state in initial_states}))
    if any(state < 0 or state >= nlevels - 1 for state in origins):
        raise ValueError("Every initial state must be between 0 and nlevels - 2.")

    if executor_backend not in {"process", "thread"}:
        raise ValueError("executor_backend must be 'process' or 'thread'.")

    payload = {
        "currents_uA": currents,
        "theta_ext": theta,
        "phi_ext": phi,
        "circuit_parameters": circuit_parameters,
        "cutoffs": cutoffs,
        "nlevels": nlevels,
        "drive_operators": tuple(drive_operators),
    }
    cache_path = None
    cached = None
    if cache_directory is not None:
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path = cache_directory / (
            f"transition_prediction_{_prediction_cache_key(payload)}.npz"
        )
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                cached = {
                    "energies": saved["energies"],
                    **{
                        name: saved[f"operator_{name}"]
                        for name in drive_operators
                    },
                }

    if cached is None:
        tasks = []
        for theta_value, phi_value in zip(theta, phi):
            kwargs = dict(circuit_parameters)
            kwargs.update(cutoffs)
            kwargs.update(
                theta_ext=float(theta_value),
                phi_ext=float(phi_value),
                nlev=int(nlevels),
            )
            tasks.append((kwargs, tuple(drive_operators)))

        worker_count = min(max(int(num_cpus), 1), max(len(tasks), 1))
        if worker_count == 1:
            rows = [_solve_hamiltonian_point(task) for task in tasks]
        elif executor_backend == "thread":
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                rows = list(executor.map(_solve_hamiltonian_point, tasks))
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context
            ) as executor:
                rows = list(executor.map(_solve_hamiltonian_point, tasks))

        energies = np.asarray([row[0] for row in rows])
        operator_arrays = {
            name: np.asarray([row[1][name] for row in rows])
            for name in drive_operators
        }
        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                energies=energies,
                **{
                    f"operator_{name}": values
                    for name, values in operator_arrays.items()
                },
            )
    else:
        energies = np.asarray(cached["energies"])
        operator_arrays = {
            name: np.asarray(cached[name]) for name in drive_operators
        }

    populations = _thermal_populations(energies, temperature_K)
    if origin_population_floor:
        for state, floor in origin_population_floor.items():
            if state < populations.shape[1]:
                populations[:, state] = np.maximum(
                    populations[:, state], float(floor)
                )
        populations /= np.sum(populations, axis=1, keepdims=True)

    initial_indices = []
    final_indices = []
    for initial_state in origins:
        for final_state in range(initial_state + 1, nlevels):
            initial_indices.append(initial_state)
            final_indices.append(final_state)
    initial_indices = np.asarray(initial_indices, dtype=int)
    final_indices = np.asarray(final_indices, dtype=int)
    frequencies = (
        energies[:, final_indices] - energies[:, initial_indices]
    )

    matrix_elements = {}
    visibility = {}
    for name, operator in operator_arrays.items():
        elements = np.abs(
            operator[:, initial_indices, final_indices]
        )
        matrix_elements[name] = elements
        visibility[name] = (
            populations[:, initial_indices] * elements**2
        )

    return TransitionPrediction(
        currents_uA=currents,
        theta_ext=theta,
        phi_ext=phi,
        energies_GHz=energies,
        initial_states=initial_indices,
        final_states=final_indices,
        frequencies_GHz=frequencies,
        thermal_populations=populations,
        operator_matrices=operator_arrays,
        matrix_elements=matrix_elements,
        visibility=visibility,
    )




def _soft_rms_MHz(residual_GHz: np.ndarray, scale_GHz: float = 0.05) -> float:
    normalized = np.asarray(residual_GHz, dtype=float) / scale_GHz
    soft_square = 2.0 * (np.sqrt(1.0 + normalized**2) - 1.0)
    return float(1e3 * scale_GHz * np.sqrt(np.mean(soft_square)))


def assign_bright_transition_points(
    prediction,
    currents_uA: np.ndarray,
    observed_frequency_GHz: np.ndarray,
    *,
    operator_names=("d_theta", "grid_phi"),
    allowed_initial_states=(0, 1),
    maximum_state: int = 8,
    matrix_penalty_MHz: float = 25.0,
) -> dict:
    currents = np.asarray(currents_uA, dtype=float)
    observed = np.asarray(observed_frequency_GHz, dtype=float)
    energies = np.asarray(
        [
            np.interp(
                currents,
                prediction.currents_uA,
                prediction.energies_GHz[:, state],
            )
            for state in range(maximum_state + 1)
        ]
    ).T
    populations = np.asarray(
        [
            np.interp(
                currents,
                prediction.currents_uA,
                prediction.thermal_populations[:, state],
            )
            for state in range(maximum_state + 1)
        ]
    ).T

    operator_values = {}
    for name in operator_names:
        source = np.abs(prediction.operator_matrices[name])
        values = np.empty(
            (currents.size, maximum_state + 1, maximum_state + 1)
        )
        for row in range(maximum_state + 1):
            for column in range(maximum_state + 1):
                values[:, row, column] = np.interp(
                    currents,
                    prediction.currents_uA,
                    source[:, row, column],
                )
        operator_values[name] = values

    selected_initial = []
    selected_final = []
    selected_frequency = []
    selected_relative = []
    selected_visibility = []
    selected_population = []
    selected_score = []
    for point_index, target in enumerate(observed):
        scales = {}
        visibility_scales = {}
        for name, values in operator_values.items():
            candidates = [
                values[point_index, initial, final]
                for initial in allowed_initial_states
                for final in range(initial + 1, maximum_state + 1)
            ]
            scales[name] = max(float(np.max(candidates)), 1e-15)
            visibility_candidates = [
                populations[point_index, initial]
                * values[point_index, initial, final] ** 2
                for initial in allowed_initial_states
                for final in range(initial + 1, maximum_state + 1)
            ]
            visibility_scales[name] = max(
                float(np.max(visibility_candidates)), 1e-30
            )

        candidates = []
        for initial in allowed_initial_states:
            for final in range(initial + 1, maximum_state + 1):
                frequency = (
                    energies[point_index, final]
                    - energies[point_index, initial]
                )
                relative = max(
                    operator_values[name][point_index, initial, final]
                    / scales[name]
                    for name in operator_names
                )
                relative_visibility = max(
                    populations[point_index, initial]
                    * operator_values[name][
                        point_index, initial, final
                    ]
                    ** 2
                    / visibility_scales[name]
                    for name in operator_names
                )
                frequency_error_MHz = 1e3 * abs(frequency - target)
                score = (
                    frequency_error_MHz
                    + matrix_penalty_MHz * -np.log(max(relative, 1e-12))
                )
                candidates.append(
                    (
                        score,
                        frequency_error_MHz,
                        -relative,
                        -relative_visibility,
                        initial,
                        final,
                        frequency,
                    )
                )
        (
            score,
            _error,
            negative_relative,
            negative_visibility,
            initial,
            final,
            frequency,
        ) = min(candidates)
        selected_initial.append(initial)
        selected_final.append(final)
        selected_frequency.append(frequency)
        selected_relative.append(-negative_relative)
        selected_visibility.append(-negative_visibility)
        selected_population.append(populations[point_index, initial])
        selected_score.append(score)

    selected_frequency = np.asarray(selected_frequency)
    return {
        "current_uA": currents,
        "observed_GHz": observed,
        "predicted_GHz": selected_frequency,
        "residual_GHz": selected_frequency - observed,
        "initial_states": np.asarray(selected_initial, dtype=int),
        "final_states": np.asarray(selected_final, dtype=int),
        "relative_drive_strength": np.asarray(selected_relative),
        "relative_thermal_visibility": np.asarray(selected_visibility),
        "origin_population": np.asarray(selected_population),
        "total_score_MHz": np.asarray(selected_score),
    }


def assign_bright_transition_track(
    prediction,
    currents_uA: np.ndarray,
    observed_frequency_GHz: np.ndarray,
    *,
    operator_names=("d_theta", "grid_phi"),
    allowed_initial_states=(0, 1),
    maximum_state: int = 14,
    frequency_scale_MHz: float = 60.0,
    slope_scale_MHz: float = 80.0,
    matrix_weight: float = 0.25,
    label_change_cost: float = 0.15,
) -> dict:
 
    currents = np.asarray(currents_uA, dtype=float)
    observed = np.asarray(observed_frequency_GHz, dtype=float)
    order = np.argsort(currents)
    currents = currents[order]
    observed = observed[order]
    maximum_state = min(
        maximum_state, prediction.energies_GHz.shape[1] - 1
    )
    labels = [
        (initial, final)
        for initial in allowed_initial_states
        for final in range(initial + 1, maximum_state + 1)
    ]

    energies = np.asarray(
        [
            np.interp(
                currents,
                prediction.currents_uA,
                prediction.energies_GHz[:, state],
            )
            for state in range(maximum_state + 1)
        ]
    ).T
    populations = np.asarray(
        [
            np.interp(
                currents,
                prediction.currents_uA,
                prediction.thermal_populations[:, state],
            )
            for state in range(maximum_state + 1)
        ]
    ).T
    operator_values = {}
    for name in operator_names:
        source = np.abs(prediction.operator_matrices[name])
        values = np.empty(
            (currents.size, maximum_state + 1, maximum_state + 1)
        )
        for row in range(maximum_state + 1):
            for column in range(maximum_state + 1):
                values[:, row, column] = np.interp(
                    currents,
                    prediction.currents_uA,
                    source[:, row, column],
                )
        operator_values[name] = values

    candidate_frequency = np.empty((currents.size, len(labels)))
    candidate_strength = np.empty_like(candidate_frequency)
    candidate_visibility = np.empty_like(candidate_frequency)
    for point in range(currents.size):
        matrix_scales = {}
        visibility_scales = {}
        for name, values in operator_values.items():
            amplitudes = np.asarray(
                [
                    values[point, initial, final]
                    for initial, final in labels
                ]
            )
            visibilities = np.asarray(
                [
                    populations[point, initial]
                    * values[point, initial, final] ** 2
                    for initial, final in labels
                ]
            )
            matrix_scales[name] = max(float(np.max(amplitudes)), 1e-15)
            visibility_scales[name] = max(
                float(np.max(visibilities)), 1e-30
            )
        for candidate, (initial, final) in enumerate(labels):
            candidate_frequency[point, candidate] = (
                energies[point, final] - energies[point, initial]
            )
            candidate_strength[point, candidate] = max(
                operator_values[name][point, initial, final]
                / matrix_scales[name]
                for name in operator_names
            )
            candidate_visibility[point, candidate] = max(
                populations[point, initial]
                * operator_values[name][point, initial, final] ** 2
                / visibility_scales[name]
                for name in operator_names
            )

    local_cost = (
        1e3 * (candidate_frequency - observed[:, None])
        / frequency_scale_MHz
    ) ** 2
    local_cost += matrix_weight * -np.log(
        np.maximum(candidate_strength, 1e-12)
    )
    total_cost = np.full_like(local_cost, np.inf)
    predecessor = np.full(local_cost.shape, -1, dtype=int)
    total_cost[0] = local_cost[0]
    for point in range(1, currents.size):
        observed_step = observed[point] - observed[point - 1]
        for candidate, label in enumerate(labels):
            predicted_step = (
                candidate_frequency[point, candidate]
                - candidate_frequency[point - 1, :]
            )
            slope_cost = (
                1e3 * (predicted_step - observed_step) / slope_scale_MHz
            ) ** 2
            change_cost = np.asarray(
                [
                    0.0 if previous_label == label else label_change_cost
                    for previous_label in labels
                ]
            )
            candidate_total = (
                total_cost[point - 1] + slope_cost + change_cost
            )
            best_previous = int(np.argmin(candidate_total))
            predecessor[point, candidate] = best_previous
            total_cost[point, candidate] = (
                local_cost[point, candidate]
                + candidate_total[best_previous]
            )

    path = np.empty(currents.size, dtype=int)
    path[-1] = int(np.argmin(total_cost[-1]))
    for point in range(currents.size - 1, 0, -1):
        path[point - 1] = predecessor[point, path[point]]
    selected_initial = np.asarray([labels[index][0] for index in path])
    selected_final = np.asarray([labels[index][1] for index in path])
    selected_frequency = candidate_frequency[np.arange(path.size), path]
    selected_strength = candidate_strength[np.arange(path.size), path]
    selected_visibility = candidate_visibility[np.arange(path.size), path]
    selected_population = populations[
        np.arange(path.size), selected_initial
    ]
    return {
        "current_uA": currents,
        "observed_GHz": observed,
        "predicted_GHz": selected_frequency,
        "residual_GHz": selected_frequency - observed,
        "initial_states": selected_initial,
        "final_states": selected_final,
        "relative_drive_strength": selected_strength,
        "relative_thermal_visibility": selected_visibility,
        "origin_population": selected_population,
        "path_cost": float(np.min(total_cost[-1])),
    }


def predict_fixed_transitions(
    prediction,
    currents_uA: np.ndarray,
    initial_states: np.ndarray,
    final_states: np.ndarray,
) -> np.ndarray:

    currents = np.asarray(currents_uA, dtype=float)
    initial = np.asarray(initial_states, dtype=int)
    final = np.asarray(final_states, dtype=int)
    if not (currents.shape == initial.shape == final.shape):
        raise ValueError("Current and state-label arrays must have equal shapes.")
    if np.any(initial < 0) or np.any(final <= initial):
        raise ValueError("Every fixed transition must satisfy 0 <= i < j.")
    if np.any(final >= prediction.energies_GHz.shape[1]):
        raise ValueError("A fixed transition exceeds the simulated level count.")

    predicted = np.empty(currents.size, dtype=float)
    for point, (current, state_i, state_j) in enumerate(
        zip(currents, initial, final)
    ):
        energy_i = np.interp(
            current,
            prediction.currents_uA,
            prediction.energies_GHz[:, state_i],
        )
        energy_j = np.interp(
            current,
            prediction.currents_uA,
            prediction.energies_GHz[:, state_j],
        )
        predicted[point] = energy_j - energy_i
    return predicted

