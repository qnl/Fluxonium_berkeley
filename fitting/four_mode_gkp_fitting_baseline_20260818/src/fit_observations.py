#measured-cut preparation, ridge observations, and fit metrics."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fit_config import (
    AUTHOR_DEVICE_PARAMETERS,
    JOINT_PARAMETER_BOUNDS,
    PHI0_DATA_PATH,
    PHIPI_DATA_PATH,
    TWO_CUT_CACHE_DIRECTORY,
    TWO_CUT_QUALITY_LIMITS,
)
from .spectrum_processing import (
    FluxCutExperiment,
    RidgeConfig,
    RidgeTrack,
    TrackConfig,
    estimate_symmetry_center,
    extract_ridge_detections,
    find_reflection_matches,
    link_ridge_tracks,
    load_two_tone,
    sanitize_spectrum_map,
    symmetric_crop,
    track_table,
)
from .transition_model import (
    TransitionPrediction,
    assign_bright_transition_track,
    predict_fixed_transitions,
    simulate_transition_prediction,
)

__all__ = [
    "JointTrackObservation",
    "assess_two_cut_fit_quality",
    "assign_joint_positive_tracks",
    "assign_lower_v_observations",
    "assign_positive_tracks_for_cut",
    "augment_with_lower_v_observations",
    "current_to_theta",
    "evaluate_joint_track_observations",
    "extract_lower_v_branches",
    "flux_cut_summary",
    "joint_residual_metrics",
    "per_track_residual_metrics",
    "physical_parameters",
    "prepare_flux_cut_experiment",
    "prepare_two_cut_experiments",
    "residual_metrics_for_role",
    "simulate_flux_cut",
]


@dataclass(frozen=True)
class JointTrackObservation:
    """A measured ridge with model transition labels held fixed."""

    cut_name: str
    track_id: int
    currents_uA: np.ndarray
    observed_GHz: np.ndarray
    initial_states: np.ndarray
    final_states: np.ndarray
    point_weights: np.ndarray
    assignment_rms_MHz: float
    median_drive_strength: float
    label_changes: int
    track_role: str = "positive_ridge"


def current_to_theta(
    current_uA: np.ndarray,
    spectrum_current_uA: np.ndarray,
    *,
    symmetry_center_uA: float | None = None,
    half_period_uA: float | None = None,
) -> np.ndarray:
    minimum = float(np.min(spectrum_current_uA))
    maximum = float(np.max(spectrum_current_uA))
    if symmetry_center_uA is None:
        symmetry_center_uA = 0.5 * (minimum + maximum)
    if half_period_uA is None:
        half_period_uA = 0.5 * (maximum - minimum)
    if half_period_uA <= 0.0:
        raise ValueError("half_period_uA must be positive.")
    return np.pi + np.pi * (
        np.asarray(current_uA, dtype=float) - symmetry_center_uA
    ) / half_period_uA


def physical_parameters(
    *,
    EJ: float | None = None,
    EL: float | None = None,
    ELK: float | None = None,
    EJS: float | None = None,
    ECS: float | None = None,
    eC: float | None = None,
    eP: float | None = None,
    eps_J: float = 0.0,
    eps_LK: float = 0.0,
) -> dict:
    parameters = dict(AUTHOR_DEVICE_PARAMETERS)
    if EJ is not None:
        parameters["EJ"] = float(EJ)
    if ELK is not None:
        parameters["ELK"] = float(ELK)
        if EL is None:
            parameters["EL"] = float(ELK)
    if EL is not None:
        parameters["EL"] = float(EL)
    if EJS is not None:
        parameters["EJS"] = float(EJS)
    if ECS is not None:
        parameters["ECS"] = float(ECS)
    if eC is not None:
        parameters["eC"] = float(eC)
    if eP is not None:
        parameters["eP"] = float(eP)
    parameters["eps_J"] = float(eps_J)
    parameters["eps_LK"] = float(eps_LK)
    return parameters




def prepare_flux_cut_experiment(
    *,
    name: str,
    path: str | Path,
    phi_ext: float,
    ridge_config: RidgeConfig | None = None,
    track_config: TrackConfig | None = None,
) -> FluxCutExperiment:
    """Load, flux-calibrate, and extract positive ridges from one map."""
    source_path = Path(path)
    spectrum = sanitize_spectrum_map(load_two_tone(source_path))
    lower, upper = np.quantile(spectrum.current_uA, (0.30, 0.70))
    center, _scores = estimate_symmetry_center(
        spectrum, np.linspace(lower, upper, 801)
    )
    spectrum, center = symmetric_crop(spectrum, center)
    half_period = min(
        center - float(spectrum.current_uA.min()),
        float(spectrum.current_uA.max()) - center,
    )

    ridge_config = ridge_config or RidgeConfig(
        height_sigma=3.5,
        prominence_sigma=2.5,
    )
    track_config = track_config or TrackConfig(
        min_points=8,
        min_current_span_fraction=0.08,
    )
    detections, processed = extract_ridge_detections(
        spectrum, ridge_config
    )
    positive_detections = [
        [point for point in points if point.polarity > 0]
        for points in detections
    ]
    tracks = link_ridge_tracks(positive_detections, track_config)
    return FluxCutExperiment(
        name=str(name),
        phi_ext=float(phi_ext),
        source_path=source_path,
        spectrum=spectrum,
        symmetry_center_uA=float(center),
        half_period_uA=float(half_period),
        processed_signal_uV=processed,
        tracks=tuple(tracks),
    )


def prepare_two_cut_experiments() -> tuple[FluxCutExperiment, ...]:
    """Prepare the measured phi=0 and phi=pi theta sweeps."""
    return (
        prepare_flux_cut_experiment(
            name="phi0", path=PHI0_DATA_PATH, phi_ext=0.0
        ),
        prepare_flux_cut_experiment(
            name="phipi", path=PHIPI_DATA_PATH, phi_ext=np.pi
        ),
    )


def extract_lower_v_branches(
    cut: FluxCutExperiment,
) -> tuple[tuple[RidgeTrack, RidgeTrack], dict]:
    """Reconstruct the two reflected lower-V arms across signal sign flips."""
    if not np.isclose(cut.phi_ext, np.pi):
        raise ValueError("The measured lower V belongs to the phi=pi cut.")

    detections, _processed = extract_ridge_detections(
        cut.spectrum,
        RidgeConfig(height_sigma=3.5, prominence_sigma=2.5),
    )
    all_polarity_tracks = link_ridge_tracks(
        detections,
        TrackConfig(
            min_points=5,
            min_current_span_fraction=0.04,
            polarity_mismatch_cost=0.0,
            max_gap_steps=4,
            base_gate_MHz=65.0,
        ),
    )
    center = cut.symmetry_center_uA
    left_candidates = []
    for track in all_polarity_tracks:
        if track.currents_uA.max() >= center:
            continue
        slope = np.polyfit(track.currents_uA, track.frequencies_GHz, 1)[0]
        if (
            slope < -0.015
            and track.frequencies_GHz.min() < 1.20
            and track.frequencies_GHz.max() > 3.80
            and np.ptp(track.currents_uA) > 80.0
        ):
            score = np.ptp(track.frequencies_GHz) * np.ptp(track.currents_uA)
            left_candidates.append((score, track))
    if not left_candidates:
        raise RuntimeError("Could not reconstruct the left lower-V arm.")
    source_left = max(left_candidates, key=lambda item: item[0])[1]

    reflection_matches = find_reflection_matches(
        all_polarity_tracks,
        center,
        min_overlap_points=3,
        max_rms_MHz=60.0,
    )
    track_by_id = {track.track_id: track for track in all_polarity_tracks}
    partner_matches = [
        match
        for match in reflection_matches
        if match.left_track_id == source_left.track_id
        and np.polyfit(
            track_by_id[match.right_track_id].currents_uA,
            track_by_id[match.right_track_id].frequencies_GHz,
            1,
        )[0]
        > 0.015
    ]
    if len(partner_matches) < 2:
        raise RuntimeError("Could not reconstruct both lower-V right fragments.")

    right_points_by_index = {}
    for match in partner_matches:
        for point in track_by_id[match.right_track_id].detections:
            previous = right_points_by_index.get(point.current_index)
            if previous is None or point.prominence_uV > previous.prominence_uV:
                right_points_by_index[point.current_index] = point

    left = RidgeTrack(track_id=-101, detections=source_left.detections)
    right = RidgeTrack(
        track_id=-102,
        detections=tuple(
            right_points_by_index[index]
            for index in sorted(right_points_by_index)
        ),
    )
    if len(left.detections) < 20 or len(right.detections) < 20:
        raise RuntimeError("The reconstructed lower-V arms are too sparse.")

    diagnostics = {
        "source_left_track_id": source_left.track_id,
        "source_right_track_ids": [
            match.right_track_id for match in partner_matches
        ],
        "reflection_rms_MHz": [
            match.rms_MHz for match in partner_matches
        ],
        "left_points": len(left.detections),
        "right_points": len(right.detections),
        "left_polarities": sorted(
            {point.polarity for point in left.detections}
        ),
        "right_polarities": sorted(
            {point.polarity for point in right.detections}
        ),
    }
    return (left, right), diagnostics


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flux_cut_summary(cut: FluxCutExperiment) -> dict:
    """Return compact extraction metadata suitable for JSON output."""
    return {
        "name": cut.name,
        "fixed_phi_ext_radians": cut.phi_ext,
        "source_path": str(cut.source_path),
        "source_sha256": _file_sha256(cut.source_path),
        "current_points": int(cut.spectrum.current_uA.size),
        "frequency_points": int(cut.spectrum.frequency_GHz.size),
        "current_range_uA": [
            float(cut.spectrum.current_uA.min()),
            float(cut.spectrum.current_uA.max()),
        ],
        "frequency_range_GHz": [
            float(cut.spectrum.frequency_GHz.min()),
            float(cut.spectrum.frequency_GHz.max()),
        ],
        "symmetry_center_uA": cut.symmetry_center_uA,
        "half_period_uA": cut.half_period_uA,
        "positive_track_count": len(cut.tracks),
        "positive_track_table": track_table(list(cut.tracks)).tolist(),
    }


def _joint_parameter_update(parameters: dict, updates: dict) -> dict:
    result = dict(parameters)
    for name, value in updates.items():
        lower, upper = JOINT_PARAMETER_BOUNDS[name]
        result[name] = float(np.clip(value, lower, upper))
    result["ng"] = 0.0
    return result


def _joint_simulation_grid(
    cut: FluxCutExperiment, point_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if point_count < 3 or point_count % 2 == 0:
        raise ValueError("Joint simulation grids require an odd count >= 3.")
    currents = np.linspace(
        cut.spectrum.current_uA.min(),
        cut.spectrum.current_uA.max(),
        point_count,
    )
    theta = current_to_theta(
        currents,
        cut.spectrum.current_uA,
        symmetry_center_uA=cut.symmetry_center_uA,
        half_period_uA=cut.half_period_uA,
    )
    return currents, theta


def simulate_flux_cut(
    *,
    cut: FluxCutExperiment,
    parameters: dict,
    cutoffs: dict,
    point_count: int,
    workers: int,
    drive_operators: Sequence[str] = (),
    nlevels: int = 15,
) -> TransitionPrediction:
    """Simulate one calibrated sweep while keeping its phi value fixed."""
    currents, theta = _joint_simulation_grid(cut, point_count)
    return simulate_transition_prediction(
        currents_uA=currents,
        theta_ext=theta,
        phi_ext=np.full_like(theta, cut.phi_ext),
        circuit_parameters=parameters,
        cutoffs=cutoffs,
        nlevels=nlevels,
        initial_states=(0, 1),
        drive_operators=tuple(drive_operators),
        temperature_K=0.020,
        num_cpus=workers,
        executor_backend="thread",
        cache_directory=TWO_CUT_CACHE_DIRECTORY,
    )


def _sample_indices(size: int, maximum_points: int) -> np.ndarray:
    if size <= maximum_points:
        return np.arange(size, dtype=int)
    return np.unique(
        np.rint(np.linspace(0, size - 1, maximum_points)).astype(int)
    )


def assign_positive_tracks_for_cut(
    *,
    cut: FluxCutExperiment,
    prediction: TransitionPrediction,
    maximum_assignment_rms_MHz: float,
    maximum_tracks: int = 20,
    maximum_points_per_track: int = 10,
    frequency_range_GHz: tuple[float, float] = (0.45, 6.85),
) -> tuple[list[JointTrackObservation], list[dict]]:
    """Label high-quality positive ridges with continuous model paths."""
    operator_names = tuple(prediction.operator_matrices)
    if not operator_names:
        raise ValueError("Track assignment requires drive operators.")

    candidates: list[tuple[float, JointTrackObservation, dict]] = []
    diagnostics = []
    full_span = float(np.ptp(cut.spectrum.current_uA))
    for track in cut.tracks:
        in_window = (
            (track.frequencies_GHz >= frequency_range_GHz[0])
            & (track.frequencies_GHz <= frequency_range_GHz[1])
        )
        if np.count_nonzero(in_window) < 8:
            continue
        currents = track.currents_uA[in_window]
        observed = track.frequencies_GHz[in_window]
        prominence = track.prominence_uV[in_window]
        assignment = assign_bright_transition_track(
            prediction,
            currents,
            observed,
            operator_names=operator_names,
            allowed_initial_states=(0, 1),
            maximum_state=14,
            frequency_scale_MHz=70.0,
            slope_scale_MHz=100.0,
            matrix_weight=0.22,
            label_change_cost=0.30,
        )
        residual = np.asarray(assignment["residual_GHz"], dtype=float)
        rms_MHz = float(1e3 * np.sqrt(np.mean(residual**2)))
        drive_strength = np.asarray(
            assignment["relative_drive_strength"], dtype=float
        )
        median_strength = float(np.median(drive_strength))
        initial = np.asarray(assignment["initial_states"], dtype=int)
        final = np.asarray(assignment["final_states"], dtype=int)
        label_changes = int(
            np.count_nonzero(
                (np.diff(initial) != 0) | (np.diff(final) != 0)
            )
        )
        span_fraction = float(np.ptp(currents) / full_span)
        accepted = bool(
            rms_MHz <= maximum_assignment_rms_MHz
            and median_strength >= 5e-4
            and span_fraction >= 0.075
        )

        sample = _sample_indices(currents.size, maximum_points_per_track)
        sampled_prominence = prominence[sample]
        prominence_scale = max(
            float(np.median(sampled_prominence)), np.finfo(float).eps
        )
        point_weights = np.sqrt(sampled_prominence / prominence_scale)
        point_weights = np.clip(point_weights, 0.65, 1.55)
        persistence_weight = np.clip(
            (currents.size / 12.0) ** 0.20
            * (prominence_scale / 50.0) ** 0.15,
            0.70,
            1.60,
        )
        point_weights *= persistence_weight
        point_weights /= np.sqrt(point_weights.size)
        observation = JointTrackObservation(
            cut_name=cut.name,
            track_id=track.track_id,
            currents_uA=currents[sample],
            observed_GHz=observed[sample],
            initial_states=initial[sample],
            final_states=final[sample],
            point_weights=point_weights,
            assignment_rms_MHz=rms_MHz,
            median_drive_strength=median_strength,
            label_changes=label_changes,
        )
        frequency_median = float(np.median(observed))
        if frequency_median < 3.3:
            frequency_band = "low"
        elif frequency_median < 5.3:
            frequency_band = "middle"
        elif frequency_median < 6.05:
            frequency_band = "high_lower"
        else:
            frequency_band = "high_upper"
        quality_score = (
            rms_MHz
            + 16.0 * max(-np.log10(max(median_strength, 1e-12)), 0.0)
            - 18.0 * min(span_fraction, 0.5)
        )
        record = {
            "track_id": track.track_id,
            "points": int(currents.size),
            "current_span_fraction": span_fraction,
            "median_frequency_GHz": frequency_median,
            "frequency_band": frequency_band,
            "assignment_rms_MHz": rms_MHz,
            "median_drive_strength": median_strength,
            "label_changes": label_changes,
            "accepted_by_threshold": accepted,
            "quality_score": float(quality_score),
        }
        diagnostics.append(record)
        if accepted:
            candidates.append((quality_score, observation, record))

    # Preserve information from low, middle, and high branches instead of
    # allowing one dense horizontal family to consume the entire fit.
    quotas = {"low": 4, "middle": 7, "high_lower": 2, "high_upper": 6}
    selected = []
    for band, quota in quotas.items():
        rows = sorted(
            (
                row
                for row in candidates
                if row[2]["frequency_band"] == band
            ),
            key=lambda row: row[0],
        )
        selected.extend(row[1] for row in rows[:quota])
    selected = sorted(selected, key=lambda item: item.track_id)
    if len(selected) > maximum_tracks:
        selected = selected[:maximum_tracks]
    selected_ids = {item.track_id for item in selected}
    for record in diagnostics:
        record["selected_for_fit"] = record["track_id"] in selected_ids
    return selected, diagnostics


def assign_lower_v_observations(
    *,
    cut: FluxCutExperiment,
    prediction: TransitionPrediction,
    maximum_assignment_rms_MHz: float,
    maximum_points_per_arm: int,
) -> tuple[list[JointTrackObservation], list[dict]]:
    """Assign both measured lower-V arms to continuous bright model paths."""
    tracks, extraction = extract_lower_v_branches(cut)
    observations = []
    diagnostics = []
    for side, track in zip(("left", "right"), tracks):
        assignment = assign_bright_transition_track(
            prediction,
            track.currents_uA,
            track.frequencies_GHz,
            operator_names=tuple(prediction.operator_matrices),
            allowed_initial_states=(0, 1),
            maximum_state=14,
            frequency_scale_MHz=90.0,
            slope_scale_MHz=110.0,
            matrix_weight=0.12,
            label_change_cost=0.45,
        )
        residual = np.asarray(assignment["residual_GHz"], dtype=float)
        rms_MHz = float(1e3 * np.sqrt(np.mean(residual**2)))
        if rms_MHz > maximum_assignment_rms_MHz:
            raise RuntimeError(
                f"The {side} lower-V assignment is too far from the model: "
                f"{rms_MHz:.1f} MHz."
            )

        sample = _sample_indices(
            len(track.detections), maximum_points_per_arm
        )
        prominence = track.prominence_uV[sample]
        scale = max(float(np.median(prominence)), np.finfo(float).eps)
        point_weights = np.sqrt(prominence / scale)
        point_weights = np.clip(point_weights, 0.65, 1.55)
        point_weights *= 1.8 / np.sqrt(point_weights.size)
        initial = np.asarray(assignment["initial_states"], dtype=int)
        final = np.asarray(assignment["final_states"], dtype=int)
        strength = np.asarray(
            assignment["relative_drive_strength"], dtype=float
        )
        label_changes = int(
            np.count_nonzero(
                (np.diff(initial) != 0) | (np.diff(final) != 0)
            )
        )
        observations.append(
            JointTrackObservation(
                cut_name=cut.name,
                track_id=track.track_id,
                currents_uA=track.currents_uA[sample],
                observed_GHz=track.frequencies_GHz[sample],
                initial_states=initial[sample],
                final_states=final[sample],
                point_weights=point_weights,
                assignment_rms_MHz=rms_MHz,
                median_drive_strength=float(np.median(strength)),
                label_changes=label_changes,
                track_role=f"lower_v_{side}",
            )
        )
        diagnostics.append(
            {
                "track_id": track.track_id,
                "track_role": f"lower_v_{side}",
                "points": len(track.detections),
                "frequency_band": "lower_v",
                "median_frequency_GHz": float(
                    np.median(track.frequencies_GHz)
                ),
                "assignment_rms_MHz": rms_MHz,
                "median_drive_strength": float(np.median(strength)),
                "label_changes": label_changes,
                "selected_for_fit": True,
                "extraction": extraction,
            }
        )
    return observations, diagnostics


def _observation_overlap_fraction(
    observation: JointTrackObservation,
    reference: JointTrackObservation,
    tolerance_MHz: float = 60.0,
) -> float:
    within = (
        (observation.currents_uA >= reference.currents_uA.min())
        & (observation.currents_uA <= reference.currents_uA.max())
    )
    if not np.any(within):
        return 0.0
    interpolated = np.interp(
        observation.currents_uA[within],
        reference.currents_uA,
        reference.observed_GHz,
    )
    matches = (
        np.abs(observation.observed_GHz[within] - interpolated)
        <= 1e-3 * tolerance_MHz
    )
    return float(np.count_nonzero(matches) / observation.observed_GHz.size)


def augment_with_lower_v_observations(
    *,
    cut: FluxCutExperiment,
    prediction: TransitionPrediction,
    selected: Sequence[JointTrackObservation],
    diagnostics: Sequence[dict],
    maximum_assignment_rms_MHz: float,
    maximum_points_per_arm: int,
) -> tuple[list[JointTrackObservation], list[dict]]:
    """Add the V arms and remove shorter duplicate positive fragments."""
    selected = list(selected)
    diagnostics = [dict(record) for record in diagnostics]
    if not np.isclose(cut.phi_ext, np.pi):
        return selected, diagnostics

    lower_v, lower_v_diagnostics = assign_lower_v_observations(
        cut=cut,
        prediction=prediction,
        maximum_assignment_rms_MHz=maximum_assignment_rms_MHz,
        maximum_points_per_arm=maximum_points_per_arm,
    )
    duplicate_ids = {
        observation.track_id
        for observation in selected
        if any(
            _observation_overlap_fraction(observation, arm) >= 0.60
            for arm in lower_v
        )
    }
    selected = [
        observation
        for observation in selected
        if observation.track_id not in duplicate_ids
    ]
    for record in diagnostics:
        if record["track_id"] in duplicate_ids:
            record["selected_for_fit"] = False
            record["excluded_for_lower_v_overlap"] = True
    return selected + lower_v, diagnostics + lower_v_diagnostics


def assign_joint_positive_tracks(
    *,
    cuts: Sequence[FluxCutExperiment],
    parameters: dict,
    cutoffs: dict,
    workers: int,
    simulation_points: int,
    maximum_assignment_rms_MHz: float,
    maximum_points_per_track: int = 10,
    include_lower_v: bool = False,
) -> tuple[list[JointTrackObservation], dict[str, list[dict]]]:
    observations = []
    diagnostics = {}
    for cut in cuts:
        prediction = simulate_flux_cut(
            cut=cut,
            parameters=parameters,
            cutoffs=cutoffs,
            point_count=simulation_points,
            workers=workers,
            drive_operators=("n1", "grid_n", "d_theta", "grid_phi"),
        )
        selected, rows = assign_positive_tracks_for_cut(
            cut=cut,
            prediction=prediction,
            maximum_assignment_rms_MHz=maximum_assignment_rms_MHz,
            maximum_points_per_track=maximum_points_per_track,
        )
        if include_lower_v:
            selected, rows = augment_with_lower_v_observations(
                cut=cut,
                prediction=prediction,
                selected=selected,
                diagnostics=rows,
                maximum_assignment_rms_MHz=maximum_assignment_rms_MHz,
                maximum_points_per_arm=maximum_points_per_track,
            )
        observations.extend(selected)
        diagnostics[cut.name] = rows
    if not observations:
        raise RuntimeError("No positive branches could be assigned to the model.")
    return observations, diagnostics


def evaluate_joint_track_observations(
    *,
    cuts: Sequence[FluxCutExperiment],
    observations: Sequence[JointTrackObservation],
    parameters: dict,
    cutoffs: dict,
    workers: int,
    simulation_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Evaluate fixed physical labels for every selected measured branch."""
    cut_lookup = {cut.name: cut for cut in cuts}
    prediction_by_cut = {
        cut.name: simulate_flux_cut(
            cut=cut,
            parameters=parameters,
            cutoffs=cutoffs,
            point_count=simulation_points,
            workers=workers,
            drive_operators=(),
        )
        for cut in cuts
    }
    raw_residuals = []
    weighted_residuals = []
    rows = []
    for observation in observations:
        if observation.cut_name not in cut_lookup:
            raise KeyError(f"Unknown cut {observation.cut_name!r}.")
        prediction = prediction_by_cut[observation.cut_name]
        predicted = predict_fixed_transitions(
            prediction,
            observation.currents_uA,
            observation.initial_states,
            observation.final_states,
        )
        residual = predicted - observation.observed_GHz
        raw_residuals.extend(residual)
        weighted_residuals.extend(residual * observation.point_weights)
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
    return (
        np.asarray(raw_residuals, dtype=float),
        np.asarray(weighted_residuals, dtype=float),
        rows,
    )


def joint_residual_metrics(rows: Sequence[dict]) -> dict:
    """Summarize unweighted branch residuals for each cut and jointly."""
    result = {}
    for cut_name in sorted({row["cut_name"] for row in rows}):
        values = np.concatenate(
            [
                np.asarray(row["residual_MHz"], dtype=float)
                for row in rows
                if row["cut_name"] == cut_name
            ]
        )
        result[cut_name] = {
            "points": int(values.size),
            "rms_MHz": float(np.sqrt(np.mean(values**2))),
            "median_absolute_MHz": float(np.median(np.abs(values))),
            "p90_absolute_MHz": float(np.quantile(np.abs(values), 0.90)),
            "fraction_within_100_MHz": float(np.mean(np.abs(values) <= 100.0)),
        }
    all_values = np.concatenate(
        [np.asarray(row["residual_MHz"], dtype=float) for row in rows]
    )
    result["combined"] = {
        "points": int(all_values.size),
        "rms_MHz": float(np.sqrt(np.mean(all_values**2))),
        "median_absolute_MHz": float(np.median(np.abs(all_values))),
        "p90_absolute_MHz": float(np.quantile(np.abs(all_values), 0.90)),
        "fraction_within_100_MHz": float(
            np.mean(np.abs(all_values) <= 100.0)
        ),
    }
    return result


def residual_metrics_for_role(
    rows: Sequence[dict], role_prefix: str
) -> dict:
    """Summarize residuals for a named physical branch family."""
    selected = [
        row
        for row in rows
        if row.get("track_role", "").startswith(role_prefix)
    ]
    if not selected:
        return {
            "points": 0,
            "tracks": 0,
            "rms_MHz": float("inf"),
            "median_absolute_MHz": float("inf"),
            "fraction_within_100_MHz": 0.0,
        }
    values = np.concatenate(
        [np.asarray(row["residual_MHz"], dtype=float) for row in selected]
    )
    return {
        "points": int(values.size),
        "tracks": len(selected),
        "rms_MHz": float(np.sqrt(np.mean(values**2))),
        "median_absolute_MHz": float(np.median(np.abs(values))),
        "fraction_within_100_MHz": float(np.mean(np.abs(values) <= 100.0)),
    }


def per_track_residual_metrics(rows: Sequence[dict]) -> dict:
    """Report branch-balanced residual statistics for the retained tracks."""
    records = []
    for row in rows:
        residual = np.asarray(row["residual_MHz"], dtype=float)
        records.append(
            {
                "cut_name": row["cut_name"],
                "track_id": row["track_id"],
                "track_role": row.get("track_role", "positive_ridge"),
                "points": int(residual.size),
                "rms_MHz": float(np.sqrt(np.mean(residual**2))),
                "maximum_absolute_MHz": float(np.max(np.abs(residual))),
                "fraction_within_100_MHz": float(
                    np.mean(np.abs(residual) <= 100.0)
                ),
            }
        )
    if not records:
        return {
            "tracks": 0,
            "maximum_rms_MHz": float("inf"),
            "p90_rms_MHz": float("inf"),
            "tracks_above_100_MHz_rms": 0,
            "records": [],
        }
    rms = np.asarray([record["rms_MHz"] for record in records])
    return {
        "tracks": len(records),
        "maximum_rms_MHz": float(np.max(rms)),
        "p90_rms_MHz": float(np.quantile(rms, 0.90)),
        "tracks_above_100_MHz_rms": int(np.count_nonzero(rms > 100.0)),
        "records": records,
    }


def _branch_balanced_score(rows: Sequence[dict]) -> float:
    """Score every branch equally while retaining pressure on the worst one."""
    track_rms_GHz = np.asarray(
        [
            1e-3
            * np.sqrt(
                np.mean(np.asarray(row["residual_MHz"], dtype=float) ** 2)
            )
            for row in rows
        ]
    )
    return float(
        np.sqrt(np.mean(track_rms_GHz**2))
        + 0.25 * np.max(track_rms_GHz)
    )


def assess_two_cut_fit_quality(
    metrics: dict, assignments: Sequence[dict]
) -> dict:
    """Apply the declared final agreement requirements to both flux cuts."""
    cuts = ("phi0", "phipi")
    track_counts = {
        name: len(
            {
                row["track_id"]
                for row in assignments
                if row["cut_name"] == name
            }
        )
        for name in cuts
    }
    lower_v = residual_metrics_for_role(assignments, "lower_v_")
    per_track = per_track_residual_metrics(assignments)
    checks = {
        "rms_below_limit_for_each_cut": all(
            metrics[name]["rms_MHz"]
            <= TWO_CUT_QUALITY_LIMITS["maximum_rms_MHz_per_cut"]
            for name in cuts
        ),
        "coverage_above_limit_for_each_cut": all(
            metrics[name]["fraction_within_100_MHz"]
            >= TWO_CUT_QUALITY_LIMITS[
                "minimum_fraction_within_100_MHz_per_cut"
            ]
            for name in cuts
        ),
        "enough_retained_tracks_for_each_cut": all(
            track_counts[name]
            >= TWO_CUT_QUALITY_LIMITS["minimum_retained_tracks_per_cut"]
            for name in cuts
        ),
        "every_track_rms_below_limit": (
            per_track["maximum_rms_MHz"]
            <= TWO_CUT_QUALITY_LIMITS["maximum_rms_MHz_per_track"]
        ),
        "lower_v_rms_below_limit": (
            lower_v["rms_MHz"]
            <= TWO_CUT_QUALITY_LIMITS["maximum_lower_v_rms_MHz"]
        ),
        "both_lower_v_arms_retained": (
            lower_v["tracks"]
            >= TWO_CUT_QUALITY_LIMITS["required_lower_v_arms"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "limits": dict(TWO_CUT_QUALITY_LIMITS),
        "checks": checks,
        "retained_tracks_by_cut": track_counts,
        "lower_v_metrics": lower_v,
        "per_track_metrics": per_track,
    }


