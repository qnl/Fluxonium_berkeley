#measured-spectrum loading, ridge extraction, and branch tracking

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, savgol_filter

__all__ = [
    "FluxCutExperiment",
    "ReflectionMatch",
    "RidgeConfig",
    "RidgeDetection",
    "RidgeTrack",
    "SpectrumMap",
    "TrackConfig",
    "estimate_symmetry_center",
    "extract_ridge_detections",
    "find_reflection_matches",
    "link_ridge_tracks",
    "load_two_tone",
    "sanitize_spectrum_map",
    "symmetric_crop",
    "track_table",
]


@dataclass(frozen=True)
class SpectrumMap:
    current_uA: np.ndarray
    frequency_GHz: np.ndarray
    signal_uV: np.ndarray


@dataclass(frozen=True)
class RidgeConfig:
    frequency_min_GHz: float = 0.2
    frequency_max_GHz: float = 6.9
    background_bins: int = 101
    smooth_bins: int = 9
    height_sigma: float = 4.0
    prominence_sigma: float = 3.0
    minimum_spacing_MHz: float = 60.0
    merge_MHz: float = 45.0


@dataclass(frozen=True)
class RidgeDetection:
    current_index: int
    current_uA: float
    frequency_GHz: float
    prominence_uV: float
    polarity: int
    width_GHz: float
    snr: float


@dataclass(frozen=True)
class TrackConfig:
    max_gap_steps: int = 3
    base_gate_MHz: float = 45.0
    max_slope_GHz_per_uA: float = 0.045
    polarity_mismatch_cost: float = 0.12
    unmatched_cost: float = 1.0
    min_points: int = 12
    min_current_span_fraction: float = 0.12


@dataclass(frozen=True)
class RidgeTrack:
    track_id: int
    detections: tuple[RidgeDetection, ...]

    @property
    def current_indices(self) -> np.ndarray:
        return np.asarray([point.current_index for point in self.detections], dtype=int)

    @property
    def currents_uA(self) -> np.ndarray:
        return np.asarray([point.current_uA for point in self.detections])

    @property
    def frequencies_GHz(self) -> np.ndarray:
        return np.asarray([point.frequency_GHz for point in self.detections])

    @property
    def prominence_uV(self) -> np.ndarray:
        return np.asarray([point.prominence_uV for point in self.detections])

    @property
    def polarity(self) -> int:
        signed_weight = sum(
            point.polarity * point.prominence_uV for point in self.detections
        )
        return 1 if signed_weight >= 0.0 else -1


@dataclass(frozen=True)
class ReflectionMatch:
    left_track_id: int
    right_track_id: int
    overlap_points: int
    rms_MHz: float
    median_error_MHz: float


@dataclass(frozen=True)
class FluxCutExperiment:
    """One measured theta sweep at a fixed main-loop flux."""

    name: str
    phi_ext: float
    source_path: Path
    spectrum: SpectrumMap
    symmetry_center_uA: float
    half_period_uA: float
    processed_signal_uV: np.ndarray
    tracks: tuple[RidgeTrack, ...]


def _odd_window(requested: int, available: int, minimum: int = 3) -> int:
    window = min(int(requested) | 1, available if available % 2 else available - 1)
    return max(minimum, window)


def load_two_tone(path: str | Path) -> SpectrumMap:
    """Load the Rohde & Schwarz tab-separated two-tone export."""
    path = Path(path)
    with path.open() as handle:
        lines = handle.read().splitlines()

    marker = next(
        index for index, line in enumerate(lines) if line.startswith("First row")
    )

    def parse(line: str) -> np.ndarray:
        return np.asarray(
            [float(value) for value in line.split("\t") if value.strip()]
        )

    current_A = parse(lines[marker - 1])
    frequency_Hz = parse(lines[marker + 1])
    signal_V = np.asarray(
        [parse(line) for line in lines[marker + 2 :] if line.strip()]
    )
    expected_shape = (current_A.size, frequency_Hz.size)
    if signal_V.shape != expected_shape:
        raise ValueError(
            f"Signal shape {signal_V.shape} does not match axes {expected_shape}."
        )

    current_order = np.argsort(current_A)
    frequency_order = np.argsort(frequency_Hz)
    return SpectrumMap(
        current_uA=current_A[current_order] * 1e6,
        frequency_GHz=frequency_Hz[frequency_order] / 1e9,
        signal_uV=signal_V[current_order, :][:, frequency_order] * 1e6,
    )


def sanitize_spectrum_map(spectrum: SpectrumMap) -> SpectrumMap:
    """Drop empty traces and interpolate isolated missing frequency samples."""
    finite_rows = np.any(np.isfinite(spectrum.signal_uV), axis=1)
    if not np.any(finite_rows):
        raise ValueError("The spectrum contains no finite signal samples.")

    current = np.asarray(spectrum.current_uA[finite_rows], dtype=float)
    frequency = np.asarray(spectrum.frequency_GHz, dtype=float)
    signal = np.asarray(spectrum.signal_uV[finite_rows], dtype=float).copy()
    for row in range(signal.shape[0]):
        finite = np.isfinite(signal[row])
        if not np.any(finite):
            raise ValueError("An empty trace remained after spectrum cleanup.")
        if not np.all(finite):
            signal[row, ~finite] = np.interp(
                frequency[~finite], frequency[finite], signal[row, finite]
            )
    return SpectrumMap(
        current_uA=current,
        frequency_GHz=frequency.copy(),
        signal_uV=signal,
    )


def estimate_symmetry_center(
    spectrum: SpectrumMap,
    center_candidates_uA: Sequence[float],
    frequency_range_GHz: tuple[float, float] = (3.2, 6.8),
) -> tuple[float, np.ndarray]:
    """Find the current about which the flux-dependent map is most symmetric."""
    residual = spectrum.signal_uV - median_filter(
        spectrum.signal_uV, size=(1, 101), mode="nearest"
    )
    residual = gaussian_filter(residual, sigma=(0.6, 1.2))
    frequency_mask = (
        (spectrum.frequency_GHz >= frequency_range_GHz[0])
        & (spectrum.frequency_GHz <= frequency_range_GHz[1])
    )
    traces = residual[:, frequency_mask].copy()
    traces -= np.median(traces, axis=1, keepdims=True)
    trace_rms = np.sqrt(np.mean(traces**2, axis=1, keepdims=True))
    traces /= np.maximum(trace_rms, np.finfo(float).eps)

    interpolate_trace = interp1d(
        spectrum.current_uA,
        traces,
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
    )
    candidates = np.asarray(center_candidates_uA, dtype=float)
    scores = np.full(candidates.size, np.nan)
    for index, center_uA in enumerate(candidates):
        reflected = interpolate_trace(2.0 * center_uA - spectrum.current_uA)
        valid = (
            np.isfinite(reflected[:, 0])
            & (np.abs(spectrum.current_uA - center_uA) > 6.0)
        )
        scores[index] = np.mean(traces[valid] * reflected[valid])
    return float(candidates[np.nanargmax(scores)]), scores


def symmetric_crop(
    spectrum: SpectrumMap, symmetry_center_uA: float
) -> tuple[SpectrumMap, float]:
    """Retain the largest measured interval paired around a symmetry point."""
    current = spectrum.current_uA
    snapped_center = float(current[np.argmin(np.abs(current - symmetry_center_uA))])
    current_step = float(np.median(np.diff(current)))
    available_half_range = min(
        snapped_center - current.min(), current.max() - snapped_center
    )
    half_steps = int(np.floor(available_half_range / current_step + 1e-9))
    half_range = half_steps * current_step
    keep = (
        (current >= snapped_center - half_range - 1e-9)
        & (current <= snapped_center + half_range + 1e-9)
    )
    cropped = SpectrumMap(
        current_uA=current[keep],
        frequency_GHz=spectrum.frequency_GHz.copy(),
        signal_uV=spectrum.signal_uV[keep],
    )
    left = np.count_nonzero(cropped.current_uA < snapped_center)
    right = np.count_nonzero(cropped.current_uA > snapped_center)
    if left != right:
        raise RuntimeError(
            f"Symmetric crop is unbalanced: {left} left and {right} right."
        )
    return cropped, snapped_center


def _merged_detection(
    group: Sequence[tuple[float, float, int, float]],
    *,
    current_index: int,
    current_uA: float,
    noise: float,
) -> RidgeDetection:
    weights = np.asarray([point[1] for point in group])
    signed = sum(point[1] * point[2] for point in group)
    return RidgeDetection(
        current_index=current_index,
        current_uA=current_uA,
        frequency_GHz=float(
            np.average([point[0] for point in group], weights=weights)
        ),
        prominence_uV=float(np.sum(weights)),
        polarity=1 if signed >= 0.0 else -1,
        width_GHz=float(
            np.average([point[3] for point in group], weights=weights)
        ),
        snr=float(np.sum(weights) / noise),
    )


def extract_ridge_detections(
    spectrum: SpectrumMap, config: RidgeConfig | None = None
) -> tuple[list[list[RidgeDetection]], np.ndarray]:
    """Detect both signal polarities and retain line-quality metadata."""
    config = config or RidgeConfig()
    frequencies = spectrum.frequency_GHz
    valid_frequency = (
        (frequencies >= config.frequency_min_GHz)
        & (frequencies <= config.frequency_max_GHz)
    )
    background_bins = _odd_window(config.background_bins, frequencies.size)
    smooth_bins = _odd_window(config.smooth_bins, frequencies.size, minimum=5)
    frequency_step_GHz = float(np.median(np.diff(frequencies)))
    distance_bins = max(
        1,
        round(1e-3 * config.minimum_spacing_MHz / frequency_step_GHz),
    )
    merge_GHz = 1e-3 * config.merge_MHz

    detections: list[list[RidgeDetection]] = []
    processed = np.empty_like(spectrum.signal_uV)
    for current_index, trace in enumerate(spectrum.signal_uV):
        background = median_filter(trace, size=background_bins, mode="nearest")
        residual = savgol_filter(
            trace - background, smooth_bins, polyorder=2, mode="interp"
        )
        processed[current_index] = residual
        centered = residual[valid_frequency] - np.median(
            residual[valid_frequency]
        )
        noise = 1.4826 * np.median(np.abs(centered))
        noise = max(float(noise), np.finfo(float).eps)

        raw_points: list[tuple[float, float, int, float]] = []
        for polarity in (1, -1):
            score = np.maximum(polarity * residual, 0.0)
            peaks, properties = find_peaks(
                score,
                height=config.height_sigma * noise,
                prominence=config.prominence_sigma * noise,
                distance=distance_bins,
            )
            keep = valid_frequency[peaks]
            peaks = peaks[keep]
            prominences = properties["prominences"][keep]
            if peaks.size == 0:
                continue
            widths = (
                peak_widths(score, peaks, rel_height=0.5)[0]
                * frequency_step_GHz
            )
            raw_points.extend(
                (
                    float(frequencies[peak]),
                    float(prominence),
                    polarity,
                    float(width),
                )
                for peak, prominence, width in zip(peaks, prominences, widths)
            )

        raw_points.sort(key=lambda point: point[0])
        merged_points: list[RidgeDetection] = []
        group: list[tuple[float, float, int, float]] = []

        for point in raw_points:
            if group and point[0] - group[-1][0] > merge_GHz:
                merged_points.append(
                    _merged_detection(
                        group,
                        current_index=current_index,
                        current_uA=float(spectrum.current_uA[current_index]),
                        noise=noise,
                    )
                )
                group = []
            group.append(point)
        if group:
            merged_points.append(
                _merged_detection(
                    group,
                    current_index=current_index,
                    current_uA=float(spectrum.current_uA[current_index]),
                    noise=noise,
                )
            )
        detections.append(merged_points)

    return detections, processed


def _predict_track_frequency(
    points: Sequence[RidgeDetection], target_current_uA: float
) -> float:
    recent = points[-4:]
    if len(recent) < 2:
        return recent[-1].frequency_GHz
    currents = np.asarray([point.current_uA for point in recent])
    frequencies = np.asarray([point.frequency_GHz for point in recent])
    slope, intercept = np.polyfit(currents, frequencies, 1)
    return float(slope * target_current_uA + intercept)


def link_ridge_tracks(
    detections: Sequence[Sequence[RidgeDetection]],
    config: TrackConfig | None = None,
) -> list[RidgeTrack]:
    """Link point detections using gated constant-slope assignments."""
    config = config or TrackConfig()
    mutable_tracks: list[list[RidgeDetection]] = []
    total_current_span = (
        detections[-1][0].current_uA - detections[0][0].current_uA
        if detections and detections[0] and detections[-1]
        else 0.0
    )

    for current_index, points_at_current in enumerate(detections):
        points = list(points_at_current)
        active_indices = [
            index
            for index, track in enumerate(mutable_tracks)
            if current_index - track[-1].current_index <= config.max_gap_steps + 1
        ]
        assigned_point_indices: set[int] = set()

        if active_indices and points:
            cost = np.full(
                (len(active_indices), len(points) + len(active_indices)),
                config.unmatched_cost,
            )
            for row, track_index in enumerate(active_indices):
                track = mutable_tracks[track_index]
                last = track[-1]
                prediction = _predict_track_frequency(
                    track, points[0].current_uA
                )
                current_gap = points[0].current_uA - last.current_uA
                gate_GHz = (
                    1e-3 * config.base_gate_MHz
                    + config.max_slope_GHz_per_uA * abs(current_gap)
                )
                for column, point in enumerate(points):
                    normalized_distance = (
                        abs(point.frequency_GHz - prediction) / gate_GHz
                    )
                    polarity_cost = (
                        config.polarity_mismatch_cost
                        if point.polarity != last.polarity
                        else 0.0
                    )
                    candidate_cost = normalized_distance + polarity_cost
                    if normalized_distance <= 1.0:
                        cost[row, column] = candidate_cost

            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if (
                    column < len(points)
                    and cost[row, column] < config.unmatched_cost
                ):
                    mutable_tracks[active_indices[row]].append(points[column])
                    assigned_point_indices.add(column)

        for point_index, point in enumerate(points):
            if point_index not in assigned_point_indices:
                mutable_tracks.append([point])

    retained: list[RidgeTrack] = []
    for points in mutable_tracks:
        span = points[-1].current_uA - points[0].current_uA
        if len(points) < config.min_points:
            continue
        if total_current_span and (
            span / total_current_span < config.min_current_span_fraction
        ):
            continue
        retained.append(
            RidgeTrack(track_id=len(retained), detections=tuple(points))
        )
    return retained




def find_reflection_matches(
    tracks: Sequence[RidgeTrack],
    symmetry_current_uA: float,
    *,
    min_overlap_points: int = 6,
    max_rms_MHz: float = 100.0,
    require_opposite_slopes: bool = True,
) -> list[ReflectionMatch]:
    """Pair left/right track fragments related by flux reflection symmetry."""
    left_tracks = [
        track
        for track in tracks
        if np.median(track.currents_uA) < symmetry_current_uA
    ]
    right_tracks = [
        track
        for track in tracks
        if np.median(track.currents_uA) > symmetry_current_uA
    ]
    matches = []
    for left in left_tracks:
        left_slope = np.polyfit(
            left.currents_uA, left.frequencies_GHz, 1
        )[0]
        for right in right_tracks:
            right_slope = np.polyfit(
                right.currents_uA, right.frequencies_GHz, 1
            )[0]
            if require_opposite_slopes and left_slope * right_slope >= 0.0:
                continue

            mirrored_currents = 2.0 * symmetry_current_uA - right.currents_uA
            order = np.argsort(mirrored_currents)
            mirrored_currents = mirrored_currents[order]
            mirrored_frequencies = right.frequencies_GHz[order]
            overlap = (
                (left.currents_uA >= mirrored_currents.min())
                & (left.currents_uA <= mirrored_currents.max())
            )
            if np.count_nonzero(overlap) < min_overlap_points:
                continue
            reflected_frequency = np.interp(
                left.currents_uA[overlap],
                mirrored_currents,
                mirrored_frequencies,
            )
            error_GHz = (
                left.frequencies_GHz[overlap] - reflected_frequency
            )
            rms_MHz = float(1e3 * np.sqrt(np.mean(error_GHz**2)))
            if rms_MHz > max_rms_MHz:
                continue
            matches.append(
                ReflectionMatch(
                    left_track_id=left.track_id,
                    right_track_id=right.track_id,
                    overlap_points=int(np.count_nonzero(overlap)),
                    rms_MHz=rms_MHz,
                    median_error_MHz=float(1e3 * np.median(error_GHz)),
                )
            )
    matches.sort(key=lambda match: (match.rms_MHz, -match.overlap_points))
    return matches


def track_table(tracks: Iterable[RidgeTrack]) -> np.ndarray:
    """Return a compact numeric summary for diagnostics and notebook display."""
    rows = []
    for track in tracks:
        rows.append(
            (
                track.track_id,
                len(track.detections),
                track.currents_uA.min(),
                track.currents_uA.max(),
                track.frequencies_GHz.min(),
                track.frequencies_GHz.max(),
                np.median(track.prominence_uV),
                track.polarity,
            )
        )
    return np.asarray(rows, dtype=float)

