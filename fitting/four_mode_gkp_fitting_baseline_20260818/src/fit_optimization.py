#numerical stages for the active joint two-cut fit

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import numpy as np

from .fit_config import (
    INTERMEDIATE_CUTOFFS,
    JOINT_COARSE_CUTOFFS,
    JOINT_MEDIUM_CUTOFFS,
    JOINT_PARAMETER_BOUNDS,
    JOINT_PARAMETER_STEPS,
    JOINT_PARAMETER_TRUST,
    JOINT_STARTING_PARAMETERS,
    PREVIEW_CUTOFFS,
    TWO_CUT_RESULTS_DIRECTORY,
    TWO_CUT_SUMMARY_PATH,
    two_cut_optimizer_cutoffs,
)
from .fit_observations import (
    FluxCutExperiment,
    JointTrackObservation,
    _branch_balanced_score,
    _joint_parameter_update,
    assess_two_cut_fit_quality,
    assign_joint_positive_tracks,
    assign_positive_tracks_for_cut,
    augment_with_lower_v_observations,
    evaluate_joint_track_observations,
    flux_cut_summary,
    joint_residual_metrics,
    per_track_residual_metrics,
    prepare_two_cut_experiments,
    residual_metrics_for_role,
    simulate_flux_cut,
)
from .transition_model import (
    TransitionPrediction,
    predict_fixed_transitions,
)

__all__ = [
    "corrected_preview_parameter_update",
    "evaluate_observations_from_predictions",
    "refresh_two_cut_final_validation",
    "run_joint_linearized_stage",
    "run_two_cut_branch_balance_refinement",
    "run_two_cut_confirmation_refinement",
    "run_two_cut_hierarchical_fit",
    "run_two_cut_lower_v_refinement",
    "run_two_cut_multifidelity_refinement",
]


def _robust_joint_score(raw_residual_GHz: np.ndarray) -> float:
    scale = 0.10
    normalized = np.asarray(raw_residual_GHz, dtype=float) / scale
    values = 2.0 * (np.sqrt(1.0 + normalized**2) - 1.0)
    return float(scale * np.sqrt(np.mean(values)))


def run_joint_linearized_stage(
    *,
    stage_name: str,
    cuts: Sequence[FluxCutExperiment],
    parameters: dict,
    active_parameters: Sequence[str],
    cutoffs: dict,
    workers: int,
    assignment_points: int,
    simulation_points: int,
    maximum_assignment_rms_MHz: float,
    iterations: int = 1,
    include_lower_v: bool = False,
    branch_balanced: bool = False,
) -> tuple[dict, dict]:
    """Run regularized Gauss-Newton updates with branch labels held fixed."""
    parameters = dict(parameters)
    history = []
    for iteration in range(iterations):
        observations, assignment_diagnostics = assign_joint_positive_tracks(
            cuts=cuts,
            parameters=parameters,
            cutoffs=cutoffs,
            workers=workers,
            simulation_points=assignment_points,
            maximum_assignment_rms_MHz=maximum_assignment_rms_MHz,
            include_lower_v=include_lower_v,
        )
        baseline_raw, baseline_weighted, baseline_rows = (
            evaluate_joint_track_observations(
                cuts=cuts,
                observations=observations,
                parameters=parameters,
                cutoffs=cutoffs,
                workers=workers,
                simulation_points=simulation_points,
            )
        )
        if branch_balanced:
            track_rms = np.asarray(
                [
                    np.sqrt(
                        np.mean(
                            np.asarray(row["residual_MHz"], dtype=float) ** 2
                        )
                    )
                    for row in baseline_rows
                ]
            )
            median_track_rms = max(float(np.median(track_rms)), 1e-12)
            branch_boost = []
            for row, rms_MHz in zip(baseline_rows, track_rms):
                boost = np.clip(
                    (rms_MHz / median_track_rms) ** 0.75, 0.75, 2.50
                )
                branch_boost.extend([boost] * len(row["residual_MHz"]))
            robust_weight = np.asarray(branch_boost, dtype=float)
        else:
            robust_weight = 1.0 / np.sqrt(
                1.0 + (baseline_raw / 0.14) ** 2
            )
        point_weight = np.divide(
            baseline_weighted,
            baseline_raw,
            out=np.ones_like(baseline_weighted),
            where=np.abs(baseline_raw) > 1e-15,
        )
        point_weight = np.nan_to_num(
            point_weight, nan=1.0, posinf=0.0, neginf=0.0
        )
        point_weight = np.clip(point_weight, 0.0, 10.0)
        fixed_weight = np.nan_to_num(
            point_weight * robust_weight,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        fixed_weight = np.clip(fixed_weight, 0.0, 10.0)
        baseline_fit_residual = np.nan_to_num(
            baseline_raw * fixed_weight,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        jacobian_columns = []
        sensitivities = {}
        for name in active_parameters:
            step = JOINT_PARAMETER_STEPS[name]
            lower, upper = JOINT_PARAMETER_BOUNDS[name]
            trial_value = min(parameters[name] + step, upper)
            if trial_value == parameters[name]:
                trial_value = max(parameters[name] - step, lower)
            actual_step = trial_value - parameters[name]
            trial_parameters = _joint_parameter_update(
                parameters, {name: trial_value}
            )
            trial_raw, _weighted, _rows = evaluate_joint_track_observations(
                cuts=cuts,
                observations=observations,
                parameters=trial_parameters,
                cutoffs=cutoffs,
                workers=workers,
                simulation_points=simulation_points,
            )
            derivative = (trial_raw - baseline_raw) / actual_step
            derivative = np.nan_to_num(
                derivative, nan=0.0, posinf=0.0, neginf=0.0
            )
            derivative = np.clip(derivative, -1e3, 1e3)
            trust = JOINT_PARAMETER_TRUST[name]
            jacobian_columns.append(derivative * fixed_weight * trust)
            sensitivities[name] = {
                "finite_difference_step": float(actual_step),
                "trust_scale": trust,
                "rms_shift_for_trust_scale_MHz": float(
                    1e3 * trust * np.sqrt(np.mean(derivative**2))
                ),
            }

        jacobian = np.asarray(
            np.column_stack(jacobian_columns), dtype=np.float64
        )
        jacobian = np.nan_to_num(
            jacobian, nan=0.0, posinf=0.0, neginf=0.0
        )
        jacobian = np.clip(jacobian, -1e4, 1e4)
        ridge = 0.08
        augmented_jacobian = np.vstack(
            (
                jacobian,
                np.sqrt(ridge) * np.eye(jacobian.shape[1]),
            )
        )
        augmented_target = np.concatenate(
            (-baseline_fit_residual, np.zeros(jacobian.shape[1]))
        )
        if not np.all(np.isfinite(augmented_jacobian)) or not np.all(
            np.isfinite(augmented_target)
        ):
            raise FloatingPointError(
                "The branch-fit least-squares system contains non-finite values."
            )
        scaled_update = np.linalg.lstsq(
            augmented_jacobian,
            augmented_target,
            rcond=1e-12,
        )[0]
        scaled_update = np.nan_to_num(
            scaled_update, nan=0.0, posinf=0.0, neginf=0.0
        )
        scaled_update = np.clip(scaled_update, -1.0, 1.0)
        full_update = {
            name: float(
                scaled_update[index] * JOINT_PARAMETER_TRUST[name]
            )
            for index, name in enumerate(active_parameters)
        }

        baseline_score = (
            _branch_balanced_score(baseline_rows)
            if branch_balanced
            else _robust_joint_score(baseline_raw)
        )
        baseline_metrics = joint_residual_metrics(baseline_rows)
        baseline_track_metrics = per_track_residual_metrics(baseline_rows)
        baseline_lower_v = residual_metrics_for_role(
            baseline_rows, "lower_v_"
        )
        accepted = None
        line_search = []
        sparse_fractions = () if branch_balanced else (1.0, 0.5, 0.25, 0.125)
        for fraction in sparse_fractions:
            candidate = _joint_parameter_update(
                parameters,
                {
                    name: parameters[name] + fraction * delta
                    for name, delta in full_update.items()
                },
            )
            candidate_raw, _candidate_weighted, candidate_rows = (
                evaluate_joint_track_observations(
                    cuts=cuts,
                    observations=observations,
                    parameters=candidate,
                    cutoffs=cutoffs,
                    workers=workers,
                    simulation_points=simulation_points,
                )
            )
            score = (
                _branch_balanced_score(candidate_rows)
                if branch_balanced
                else _robust_joint_score(candidate_raw)
            )
            metrics = joint_residual_metrics(candidate_rows)
            track_metrics = per_track_residual_metrics(candidate_rows)
            cut_guard = all(
                metrics[cut.name]["rms_MHz"]
                <= 1.08 * baseline_metrics[cut.name]["rms_MHz"] + 5.0
                for cut in cuts
            )
            candidate_lower_v = residual_metrics_for_role(
                candidate_rows, "lower_v_"
            )
            lower_v_guard = (
                not include_lower_v
                or candidate_lower_v["rms_MHz"]
                <= 1.05 * baseline_lower_v["rms_MHz"] + 2.0
            )
            track_guard = (
                not branch_balanced
                or track_metrics["maximum_rms_MHz"]
                < baseline_track_metrics["maximum_rms_MHz"]
            )
            record = {
                "fraction": fraction,
                "robust_score_MHz": 1e3 * score,
                "metrics": metrics,
                "passes_two_cut_guard": bool(cut_guard),
                "lower_v_metrics": candidate_lower_v,
                "passes_lower_v_guard": bool(lower_v_guard),
                "per_track_metrics": track_metrics,
                "passes_worst_track_guard": bool(track_guard),
            }
            line_search.append(record)
            if (
                cut_guard
                and lower_v_guard
                and track_guard
                and score < baseline_score
            ):
                accepted = (candidate, candidate_rows, record)
                break

        iteration_record = {
            "iteration": iteration + 1,
            "active_parameters": list(active_parameters),
            "assigned_track_count": len(observations),
            "assigned_tracks_by_cut": {
                cut.name: sum(
                    observation.cut_name == cut.name
                    for observation in observations
                )
                for cut in cuts
            },
            "baseline_parameters": {
                key: float(value) for key, value in parameters.items()
            },
            "baseline_metrics": baseline_metrics,
            "baseline_lower_v_metrics": baseline_lower_v,
            "baseline_per_track_metrics": baseline_track_metrics,
            "baseline_robust_score_MHz": 1e3 * baseline_score,
            "sensitivities": sensitivities,
            "proposed_update": full_update,
            "line_search": line_search,
            "assignment_diagnostics": assignment_diagnostics,
            "accepted": accepted is not None,
        }
        if accepted is None:
            iteration_record["accepted_parameters"] = dict(parameters)
            history.append(iteration_record)
            break
        parameters, accepted_rows, accepted_record = accepted
        iteration_record["accepted_parameters"] = dict(parameters)
        iteration_record["accepted_metrics"] = joint_residual_metrics(
            accepted_rows
        )
        iteration_record["accepted_fraction"] = accepted_record["fraction"]
        history.append(iteration_record)

    return parameters, {
        "name": stage_name,
        "cutoffs": dict(cutoffs),
        "assignment_points": assignment_points,
        "simulation_points": simulation_points,
        "maximum_assignment_rms_MHz": maximum_assignment_rms_MHz,
        "includes_lower_v": include_lower_v,
        "branch_balanced": branch_balanced,
        "iterations": history,
    }




def run_two_cut_hierarchical_fit(
    workers: int = 4, *, optimizer_basis: str = "preview"
) -> dict:
    """Fit shared four-mode parameters in progressively finer stages."""
    cuts = prepare_two_cut_experiments()
    optimizer_cutoffs = two_cut_optimizer_cutoffs(optimizer_basis)
    parameters = dict(JOINT_STARTING_PARAMETERS)
    stages = []
    started = time.time()
    stage_specs = (
        (
            "coarse_inductive_and_global",
            ("EJ", "EL", "ELK", "eP"),
            JOINT_COARSE_CUTOFFS,
            21,
            13,
            260.0,
            2,
        ),
        (
            "coarse_nonlinear_and_charging",
            ("EC", "EJS", "ECS", "eC"),
            JOINT_COARSE_CUTOFFS,
            21,
            13,
            230.0,
            2,
        ),
        (
            "coarse_kite_asymmetry",
            ("eps_J", "eps_LK"),
            JOINT_COARSE_CUTOFFS,
            21,
            13,
            210.0,
            1,
        ),
        (
            "medium_joint_refinement",
            tuple(JOINT_PARAMETER_BOUNDS),
            JOINT_MEDIUM_CUTOFFS,
            21,
            13,
            180.0,
            1,
        ),
        (
            "production_joint_refinement",
            tuple(JOINT_PARAMETER_BOUNDS),
            optimizer_cutoffs,
            17,
            13,
            160.0,
            1,
        ),
    )
    for (
        name,
        active,
        cutoffs,
        assignment_points,
        simulation_points,
        maximum_rms,
        iterations,
    ) in stage_specs:
        print(f"\n=== {name} ===", flush=True)
        parameters, stage = run_joint_linearized_stage(
            stage_name=name,
            cuts=cuts,
            parameters=parameters,
            active_parameters=active,
            cutoffs=cutoffs,
            workers=workers,
            assignment_points=assignment_points,
            simulation_points=simulation_points,
            maximum_assignment_rms_MHz=maximum_rms,
            iterations=iterations,
        )
        stages.append(stage)
        print(
            "accepted parameters:",
            json.dumps(parameters, sort_keys=True),
            flush=True,
        )

    final_observations, final_assignment_diagnostics = (
        assign_joint_positive_tracks(
            cuts=cuts,
            parameters=parameters,
            cutoffs=optimizer_cutoffs,
            workers=workers,
            simulation_points=31,
            maximum_assignment_rms_MHz=160.0,
            maximum_points_per_track=10_000,
        )
    )
    _raw, _weighted, final_rows = evaluate_joint_track_observations(
        cuts=cuts,
        observations=final_observations,
        parameters=parameters,
        cutoffs=optimizer_cutoffs,
        workers=workers,
        simulation_points=31,
    )
    summary = {
        "workflow": "joint hierarchical fit of phi=0 and phi=pi theta sweeps",
        "physics_constraints": {
            "shared_hamiltonian_parameters": True,
            "fixed_phi_by_cut": {cut.name: cut.phi_ext for cut in cuts},
            "theta_center_maps_to_pi": True,
            "positive_signal_branches_only": True,
            "allowed_thermal_origins": [0, 1],
            "EL_and_ELK_fitted_independently": True,
        },
        "cuts": [flux_cut_summary(cut) for cut in cuts],
        "starting_parameters_GHz": dict(JOINT_STARTING_PARAMETERS),
        "final_parameters_GHz": dict(parameters),
        "parameter_bounds": JOINT_PARAMETER_BOUNDS,
        "stages": stages,
        "final_metrics": joint_residual_metrics(final_rows),
        "final_assignments": final_rows,
        "final_assignment_diagnostics": final_assignment_diagnostics,
        "optimizer_basis": optimizer_basis,
        "optimizer_cutoffs": dict(optimizer_cutoffs),
        "fit_cutoffs": dict(optimizer_cutoffs),
        "runtime_seconds": time.time() - started,
    }
    summary["quality_gate"] = assess_two_cut_fit_quality(
        summary["final_metrics"], summary["final_assignments"]
    )
    TWO_CUT_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)
    print(f"saved {TWO_CUT_SUMMARY_PATH}", flush=True)
    return summary


def run_two_cut_confirmation_refinement(workers: int = 4) -> dict:
    """Run one additional pass at the saved optimizer basis."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
    )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    optimizer_cutoffs = dict(
        summary.get("optimizer_cutoffs", PREVIEW_CUTOFFS)
    )
    cuts = prepare_two_cut_experiments()
    parameters, stage = run_joint_linearized_stage(
        stage_name="production_confirmation_refinement",
        cuts=cuts,
        parameters=summary["final_parameters_GHz"],
        active_parameters=(
            "EJ", "EC", "EL", "ELK", "ECS", "eC", "eP"
        ),
        cutoffs=optimizer_cutoffs,
        workers=workers,
        assignment_points=17,
        simulation_points=13,
        maximum_assignment_rms_MHz=160.0,
        iterations=1,
    )
    summary["stages"] = [
        saved
        for saved in summary["stages"]
        if saved.get("name") != stage["name"]
    ]
    summary["stages"].append(stage)
    summary["final_parameters_GHz"] = dict(parameters)
    summary["final_validation_uses_all_track_points"] = False
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    summary = refresh_two_cut_final_validation(workers)
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)
    print(f"saved {TWO_CUT_SUMMARY_PATH}", flush=True)
    return summary


def run_two_cut_lower_v_refinement(workers: int = 4) -> dict:
    """Add both polarity-changing lower-V arms as guarded fit constraints."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
    )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    optimizer_cutoffs = dict(
        summary.get("optimizer_cutoffs", PREVIEW_CUTOFFS)
    )
    cuts = prepare_two_cut_experiments()
    parameters, stage = run_joint_linearized_stage(
        stage_name="lower_v_guarded_refinement",
        cuts=cuts,
        parameters=summary["final_parameters_GHz"],
        active_parameters=("EJ", "EC", "EL", "ELK", "eC", "eP"),
        cutoffs=optimizer_cutoffs,
        workers=workers,
        assignment_points=21,
        simulation_points=13,
        maximum_assignment_rms_MHz=140.0,
        iterations=2,
        include_lower_v=True,
    )
    summary["stages"] = [
        saved
        for saved in summary["stages"]
        if saved.get("name") != stage["name"]
    ]
    summary["stages"].append(stage)
    summary["final_parameters_GHz"] = dict(parameters)
    summary["include_lower_v_branches"] = True
    summary["physics_constraints"]["positive_signal_branches_only"] = False
    summary["physics_constraints"]["ridge_polarities_used"] = [
        "positive",
        "negative segments of the reflected lower V",
    ]
    summary["final_validation_uses_all_track_points"] = False
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    summary = refresh_two_cut_final_validation(workers)
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)
    print(
        json.dumps(summary["quality_gate"]["lower_v_metrics"], indent=2),
        flush=True,
    )
    return summary


def _rows_with_residual_vector(
    rows: Sequence[dict], residual_GHz: np.ndarray
) -> list[dict]:
    """Replace row residuals while preserving the observation layout."""
    residual = np.asarray(residual_GHz, dtype=float)
    updated = []
    offset = 0
    for row in rows:
        count = len(row["observed_GHz"])
        values = residual[offset : offset + count]
        if values.size != count:
            raise ValueError("Residual vector does not match assignment rows.")
        copied = dict(row)
        observed = np.asarray(row["observed_GHz"], dtype=float)
        copied["predicted_GHz"] = (observed + values).tolist()
        copied["residual_MHz"] = (1e3 * values).tolist()
        updated.append(copied)
        offset += count
    if offset != residual.size:
        raise ValueError("Residual vector has unused values.")
    return updated


def evaluate_observations_from_predictions(
    *,
    observations: Sequence[JointTrackObservation],
    predictions: dict[str, TransitionPrediction],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Evaluate observations using already computed per-cut predictions."""
    raw = []
    weighted = []
    rows = []
    for observation in observations:
        prediction = predictions[observation.cut_name]
        predicted = predict_fixed_transitions(
            prediction,
            observation.currents_uA,
            observation.initial_states,
            observation.final_states,
        )
        residual = predicted - observation.observed_GHz
        raw.extend(residual)
        weighted.extend(residual * observation.point_weights)
        rows.append(
            {
                "cut_name": observation.cut_name,
                "track_id": observation.track_id,
                "current_uA": observation.currents_uA.tolist(),
                "observed_GHz": observation.observed_GHz.tolist(),
                "predicted_GHz": predicted.tolist(),
                "residual_MHz": (1e3 * residual).tolist(),
                "initial_states": observation.initial_states.tolist(),
                "final_states": observation.final_states.tolist(),
                "assignment_rms_MHz": observation.assignment_rms_MHz,
                "median_drive_strength": observation.median_drive_strength,
                "label_changes": observation.label_changes,
                "track_role": observation.track_role,
            }
        )
    return np.asarray(raw), np.asarray(weighted), rows


def corrected_preview_parameter_update(
    *,
    cuts: Sequence[FluxCutExperiment],
    observations: Sequence[JointTrackObservation],
    parameters: dict,
    active_parameters: Sequence[str],
    correction_GHz: np.ndarray,
    workers: int,
    simulation_points: int,
) -> tuple[dict, dict]:
    """Optimize the preview model with a fixed intermediate-basis correction."""
    baseline_raw, baseline_weighted, baseline_rows = (
        evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=parameters,
            cutoffs=PREVIEW_CUTOFFS,
            workers=workers,
            simulation_points=simulation_points,
        )
    )
    correction = np.asarray(correction_GHz, dtype=float)
    if correction.shape != baseline_raw.shape:
        raise ValueError("Multifidelity correction has the wrong shape.")
    corrected_baseline = baseline_raw + correction
    point_weight = np.divide(
        baseline_weighted,
        baseline_raw,
        out=np.ones_like(baseline_weighted),
        where=np.abs(baseline_raw) > 1e-15,
    )
    robust_weight = 1.0 / np.sqrt(
        1.0 + (corrected_baseline / 0.14) ** 2
    )
    fixed_weight = point_weight * robust_weight

    columns = []
    sensitivities = {}
    for name in active_parameters:
        step = JOINT_PARAMETER_STEPS[name]
        lower, upper = JOINT_PARAMETER_BOUNDS[name]
        value = min(parameters[name] + step, upper)
        if value == parameters[name]:
            value = max(parameters[name] - step, lower)
        actual_step = value - parameters[name]
        trial = _joint_parameter_update(parameters, {name: value})
        trial_raw, _weighted, _rows = evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=trial,
            cutoffs=PREVIEW_CUTOFFS,
            workers=workers,
            simulation_points=simulation_points,
        )
        derivative = np.nan_to_num(
            (trial_raw - baseline_raw) / actual_step,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        derivative = np.clip(derivative, -1e3, 1e3)
        trust = JOINT_PARAMETER_TRUST[name]
        columns.append(derivative * fixed_weight * trust)
        sensitivities[name] = {
            "finite_difference_step": float(actual_step),
            "trust_scale": trust,
            "rms_shift_for_trust_scale_MHz": float(
                1e3 * trust * np.sqrt(np.mean(derivative**2))
            ),
        }

    jacobian = np.column_stack(columns)
    target = corrected_baseline * fixed_weight
    normal = jacobian.T @ jacobian + 0.08 * np.eye(jacobian.shape[1])
    scaled_update = -np.linalg.solve(normal, jacobian.T @ target)
    scaled_update = np.clip(scaled_update, -1.0, 1.0)
    update = {
        name: float(scaled_update[index] * JOINT_PARAMETER_TRUST[name])
        for index, name in enumerate(active_parameters)
    }

    baseline_rows_corrected = _rows_with_residual_vector(
        baseline_rows, corrected_baseline
    )
    baseline_metrics = joint_residual_metrics(baseline_rows_corrected)
    baseline_lower_v = residual_metrics_for_role(
        baseline_rows_corrected, "lower_v_"
    )
    baseline_score = _robust_joint_score(corrected_baseline)
    line_search = []
    accepted = None
    for fraction in (1.0, 0.5, 0.25, 0.125):
        candidate = _joint_parameter_update(
            parameters,
            {
                name: parameters[name] + fraction * delta
                for name, delta in update.items()
            },
        )
        raw, _weighted, rows = evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=candidate,
            cutoffs=PREVIEW_CUTOFFS,
            workers=workers,
            simulation_points=simulation_points,
        )
        corrected = raw + correction
        corrected_rows = _rows_with_residual_vector(rows, corrected)
        metrics = joint_residual_metrics(corrected_rows)
        lower_v_metrics = residual_metrics_for_role(
            corrected_rows, "lower_v_"
        )
        score = _robust_joint_score(corrected)
        guard = all(
            metrics[cut.name]["rms_MHz"]
            <= 1.08 * baseline_metrics[cut.name]["rms_MHz"] + 5.0
            for cut in cuts
        )
        lower_v_guard = (
            lower_v_metrics["rms_MHz"]
            <= 1.03 * baseline_lower_v["rms_MHz"] + 2.0
        )
        record = {
            "fraction": fraction,
            "robust_score_MHz": 1e3 * score,
            "metrics": metrics,
            "passes_two_cut_guard": bool(guard),
            "lower_v_metrics": lower_v_metrics,
            "passes_lower_v_guard": bool(lower_v_guard),
        }
        line_search.append(record)
        if guard and lower_v_guard and score < baseline_score:
            accepted = (candidate, record)
            break

    if accepted is None:
        result_parameters = dict(parameters)
        accepted_metrics = baseline_metrics
        accepted_fraction = 0.0
    else:
        result_parameters, record = accepted
        accepted_metrics = record["metrics"]
        accepted_fraction = record["fraction"]
    return result_parameters, {
        "active_parameters": list(active_parameters),
        "baseline_metrics": baseline_metrics,
        "baseline_lower_v_metrics": baseline_lower_v,
        "baseline_robust_score_MHz": 1e3 * baseline_score,
        "correction_rms_MHz": float(
            1e3 * np.sqrt(np.mean(correction**2))
        ),
        "correction_maximum_MHz": float(1e3 * np.max(np.abs(correction))),
        "sensitivities": sensitivities,
        "proposed_update": update,
        "line_search": line_search,
        "accepted": accepted is not None,
        "accepted_fraction": accepted_fraction,
        "accepted_metrics": accepted_metrics,
        "accepted_parameters": dict(result_parameters),
    }


def run_two_cut_multifidelity_refinement(workers: int = 4) -> dict:
    """Correct the fast joint fit and validate directly at the larger basis."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
        )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    cuts = prepare_two_cut_experiments()
    parameters = summary["final_parameters_GHz"]
    started = time.time()

    observations, assignment_diagnostics = assign_joint_positive_tracks(
        cuts=cuts,
        parameters=parameters,
        cutoffs=PREVIEW_CUTOFFS,
        workers=workers,
        simulation_points=17,
        maximum_assignment_rms_MHz=170.0,
        maximum_points_per_track=10,
        include_lower_v=True,
    )
    preview_raw, _preview_weighted, _preview_rows = (
        evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=parameters,
            cutoffs=PREVIEW_CUTOFFS,
            workers=workers,
            simulation_points=9,
        )
    )
    intermediate_raw, _intermediate_weighted, intermediate_rows = (
        evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=parameters,
            cutoffs=INTERMEDIATE_CUTOFFS,
            workers=workers,
            simulation_points=9,
        )
    )
    correction = intermediate_raw - preview_raw
    parameters, update = corrected_preview_parameter_update(
        cuts=cuts,
        observations=observations,
        parameters=parameters,
        active_parameters=(
            "EJ", "EC", "EL", "ELK", "ECS", "eC", "eP"
        ),
        correction_GHz=correction,
        workers=workers,
        simulation_points=9,
    )

    # Reassign every retained ridge with the accepted Hamiltonian at the
    # intermediate basis.  These all-point metrics are the reported result.
    final_observations = []
    final_diagnostics = {}
    final_predictions = {}
    for cut in cuts:
        prediction = simulate_flux_cut(
            cut=cut,
            parameters=parameters,
            cutoffs=INTERMEDIATE_CUTOFFS,
            point_count=21,
            workers=workers,
            drive_operators=("n1", "grid_n", "d_theta", "grid_phi"),
        )
        selected, diagnostics = assign_positive_tracks_for_cut(
            cut=cut,
            prediction=prediction,
            maximum_assignment_rms_MHz=180.0,
            maximum_points_per_track=10_000,
        )
        selected, diagnostics = augment_with_lower_v_observations(
            cut=cut,
            prediction=prediction,
            selected=selected,
            diagnostics=diagnostics,
            maximum_assignment_rms_MHz=180.0,
            maximum_points_per_arm=10_000,
        )
        final_observations.extend(selected)
        final_diagnostics[cut.name] = diagnostics
        final_predictions[cut.name] = prediction
    _raw, _weighted, final_rows = evaluate_observations_from_predictions(
        observations=final_observations,
        predictions=final_predictions,
    )
    stage = {
        "name": "two_cut_multifidelity_correction",
        "optimizer_cutoffs": dict(PREVIEW_CUTOFFS),
        "correction_and_validation_cutoffs": dict(INTERMEDIATE_CUTOFFS),
        "assignment_diagnostics": assignment_diagnostics,
        "baseline_intermediate_metrics": joint_residual_metrics(
            intermediate_rows
        ),
        "baseline_intermediate_lower_v_metrics": residual_metrics_for_role(
            intermediate_rows, "lower_v_"
        ),
        "update": update,
        "runtime_seconds": time.time() - started,
    }
    summary["stages"] = [
        saved
        for saved in summary["stages"]
        if saved.get("name") != stage["name"]
    ]
    summary["stages"].append(stage)
    summary["final_parameters_GHz"] = dict(parameters)
    summary["final_metrics"] = joint_residual_metrics(final_rows)
    summary["final_assignments"] = final_rows
    summary["final_assignment_diagnostics"] = final_diagnostics
    summary["fit_cutoffs"] = dict(INTERMEDIATE_CUTOFFS)
    summary["optimizer_cutoffs"] = dict(PREVIEW_CUTOFFS)
    summary["final_validation_uses_all_track_points"] = True
    summary["multifidelity_correction_applied"] = True
    summary["include_lower_v_branches"] = True
    summary["physics_constraints"]["positive_signal_branches_only"] = False
    summary["physics_constraints"]["ridge_polarities_used"] = [
        "positive",
        "negative segments of the reflected lower V",
    ]
    summary["quality_gate"] = assess_two_cut_fit_quality(
        summary["final_metrics"], summary["final_assignments"]
    )
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)
    print(f"saved {TWO_CUT_SUMMARY_PATH}", flush=True)
    return summary


def _evaluate_intermediate_two_cut_fit(
    parameters: dict, workers: int
) -> dict:
    """Assign and evaluate all retained points at the authoritative basis."""
    observations = []
    diagnostics = {}
    predictions = {}
    for cut in prepare_two_cut_experiments():
        prediction = simulate_flux_cut(
            cut=cut,
            parameters=parameters,
            cutoffs=INTERMEDIATE_CUTOFFS,
            point_count=21,
            workers=workers,
            drive_operators=("n1", "grid_n", "d_theta", "grid_phi"),
        )
        selected, rows_for_cut = assign_positive_tracks_for_cut(
            cut=cut,
            prediction=prediction,
            maximum_assignment_rms_MHz=180.0,
            maximum_points_per_track=10_000,
        )
        selected, rows_for_cut = augment_with_lower_v_observations(
            cut=cut,
            prediction=prediction,
            selected=selected,
            diagnostics=rows_for_cut,
            maximum_assignment_rms_MHz=180.0,
            maximum_points_per_arm=10_000,
        )
        observations.extend(selected)
        diagnostics[cut.name] = rows_for_cut
        predictions[cut.name] = prediction
    _raw, _weighted, rows = evaluate_observations_from_predictions(
        observations=observations,
        predictions=predictions,
    )
    metrics = joint_residual_metrics(rows)
    per_track = per_track_residual_metrics(rows)
    lower_v = residual_metrics_for_role(rows, "lower_v_")
    return {
        "metrics": metrics,
        "per_track_metrics": per_track,
        "lower_v_metrics": lower_v,
        "assignments": rows,
        "assignment_diagnostics": diagnostics,
    }


def _authoritative_branch_balance_score(payload: dict) -> float:
    """Combine global, worst-branch, and lower-V errors in MHz."""
    return float(
        payload["metrics"]["combined"]["rms_MHz"]
        + 0.30 * payload["per_track_metrics"]["maximum_rms_MHz"]
        + 0.20 * payload["lower_v_metrics"]["rms_MHz"]
    )


def run_two_cut_branch_balance_refinement(workers: int = 4) -> dict:
    """Directly reduce the worst retained branch at the final fit basis."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
        )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    baseline_parameters = dict(summary["final_parameters_GHz"])
    baseline_payload = _evaluate_intermediate_two_cut_fit(
        baseline_parameters, workers
    )
    _sparse_parameters, stage = run_joint_linearized_stage(
        stage_name="intermediate_branch_balance_refinement",
        cuts=prepare_two_cut_experiments(),
        parameters=baseline_parameters,
        active_parameters=(
            "EJ", "EC", "EL", "ELK", "EJS", "ECS", "eC", "eP",
            "eps_J", "eps_LK",
        ),
        cutoffs=INTERMEDIATE_CUTOFFS,
        workers=workers,
        assignment_points=21,
        simulation_points=9,
        maximum_assignment_rms_MHz=180.0,
        iterations=1,
        include_lower_v=True,
        branch_balanced=True,
    )
    iteration = stage["iterations"][-1]
    proposed_update = iteration["proposed_update"]
    baseline_score = _authoritative_branch_balance_score(baseline_payload)
    accepted = None
    authoritative_line_search = []
    for fraction in (0.50, 0.25, 0.125, -0.125):
        candidate_parameters = _joint_parameter_update(
            baseline_parameters,
            {
                name: baseline_parameters[name] + fraction * delta
                for name, delta in proposed_update.items()
            },
        )
        payload = _evaluate_intermediate_two_cut_fit(
            candidate_parameters, workers
        )
        score = _authoritative_branch_balance_score(payload)
        cut_guard = all(
            payload["metrics"][name]["rms_MHz"]
            <= 1.05 * baseline_payload["metrics"][name]["rms_MHz"] + 2.0
            for name in ("phi0", "phipi")
        )
        lower_v_guard = (
            payload["lower_v_metrics"]["rms_MHz"]
            <= 1.15 * baseline_payload["lower_v_metrics"]["rms_MHz"] + 2.0
        )
        worst_track_guard = (
            payload["per_track_metrics"]["maximum_rms_MHz"]
            < baseline_payload["per_track_metrics"]["maximum_rms_MHz"]
        )
        record = {
            "fraction": fraction,
            "parameters": candidate_parameters,
            "score_MHz": score,
            "metrics": payload["metrics"],
            "per_track_metrics": payload["per_track_metrics"],
            "lower_v_metrics": payload["lower_v_metrics"],
            "passes_two_cut_guard": bool(cut_guard),
            "passes_lower_v_guard": bool(lower_v_guard),
            "passes_worst_track_guard": bool(worst_track_guard),
            "accepted": bool(
                cut_guard
                and lower_v_guard
                and worst_track_guard
                and score < baseline_score
            ),
        }
        authoritative_line_search.append(record)
        if record["accepted"] and (
            accepted is None or score < accepted[0]
        ):
            accepted = (score, candidate_parameters, payload, fraction)

    if accepted is None:
        parameters = baseline_parameters
        payload = baseline_payload
        accepted_fraction = 0.0
    else:
        _score, parameters, payload, accepted_fraction = accepted
    stage["authoritative_baseline"] = {
        "score_MHz": baseline_score,
        "metrics": baseline_payload["metrics"],
        "per_track_metrics": baseline_payload["per_track_metrics"],
        "lower_v_metrics": baseline_payload["lower_v_metrics"],
    }
    stage["authoritative_line_search"] = authoritative_line_search
    stage["authoritative_accepted"] = accepted is not None
    stage["authoritative_accepted_fraction"] = accepted_fraction
    stage["guarded_converged"] = accepted is None
    summary["stages"] = [
        saved
        for saved in summary["stages"]
        if saved.get("name") != stage["name"]
    ]
    summary["stages"].append(stage)
    summary["final_parameters_GHz"] = dict(parameters)
    summary["final_metrics"] = payload["metrics"]
    summary["final_assignments"] = payload["assignments"]
    summary["final_assignment_diagnostics"] = payload[
        "assignment_diagnostics"
    ]
    summary["fit_cutoffs"] = dict(INTERMEDIATE_CUTOFFS)
    summary["include_lower_v_branches"] = True
    summary["final_validation_uses_all_track_points"] = True
    summary["quality_gate"] = assess_two_cut_fit_quality(
        summary["final_metrics"], summary["final_assignments"]
    )
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)
    compact_track_metrics = {
        key: value
        for key, value in summary["quality_gate"]["per_track_metrics"].items()
        if key != "records"
    }
    print(
        json.dumps(compact_track_metrics, indent=2),
        flush=True,
    )
    print(
        json.dumps(
            {
                "authoritative_update_accepted": accepted is not None,
                "guarded_converged": accepted is None,
            },
            indent=2,
        ),
        flush=True,
    )
    return summary




def refresh_two_cut_final_validation(workers: int = 4) -> dict:
    """Re-evaluate every point of each retained measured branch."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
        )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    parameters = summary["final_parameters_GHz"]
    include_lower_v = bool(summary.get("include_lower_v_branches", False))
    validation_cutoffs = summary.get("fit_cutoffs", PREVIEW_CUTOFFS)
    assignment_limit = (
        180.0 if validation_cutoffs == INTERMEDIATE_CUTOFFS else 160.0
    )
    cuts = prepare_two_cut_experiments()
    if validation_cutoffs == INTERMEDIATE_CUTOFFS:
        observations = []
        diagnostics = {}
        predictions = {}
        for cut in cuts:
            prediction = simulate_flux_cut(
                cut=cut,
                parameters=parameters,
                cutoffs=validation_cutoffs,
                point_count=21,
                workers=workers,
                drive_operators=("n1", "grid_n", "d_theta", "grid_phi"),
            )
            selected, rows_for_cut = assign_positive_tracks_for_cut(
                cut=cut,
                prediction=prediction,
                maximum_assignment_rms_MHz=assignment_limit,
                maximum_points_per_track=10_000,
            )
            if include_lower_v:
                selected, rows_for_cut = augment_with_lower_v_observations(
                    cut=cut,
                    prediction=prediction,
                    selected=selected,
                    diagnostics=rows_for_cut,
                    maximum_assignment_rms_MHz=assignment_limit,
                    maximum_points_per_arm=10_000,
                )
            observations.extend(selected)
            diagnostics[cut.name] = rows_for_cut
            predictions[cut.name] = prediction
        _raw, _weighted, rows = evaluate_observations_from_predictions(
            observations=observations,
            predictions=predictions,
        )
    else:
        observations, diagnostics = assign_joint_positive_tracks(
            cuts=cuts,
            parameters=parameters,
            cutoffs=validation_cutoffs,
            workers=workers,
            simulation_points=31,
            maximum_assignment_rms_MHz=assignment_limit,
            maximum_points_per_track=10_000,
            include_lower_v=include_lower_v,
        )
        _raw, _weighted, rows = evaluate_joint_track_observations(
            cuts=cuts,
            observations=observations,
            parameters=parameters,
            cutoffs=validation_cutoffs,
            workers=workers,
            simulation_points=31,
        )
    summary["final_metrics"] = joint_residual_metrics(rows)
    summary["final_assignments"] = rows
    summary["final_assignment_diagnostics"] = diagnostics
    summary["final_validation_uses_all_track_points"] = True
    summary["quality_gate"] = assess_two_cut_fit_quality(
        summary["final_metrics"], summary["final_assignments"]
    )
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


