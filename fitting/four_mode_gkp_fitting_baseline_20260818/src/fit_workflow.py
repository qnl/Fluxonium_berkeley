"""Top-level profiles, overlays, and CLI for the two-cut fit."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .fit_config import (
    DEFAULT_FIT_PROFILE,
    FIT_PROFILES,
    PREVIEW_CUTOFFS,
    TWO_CUT_FIGURE_PATH,
    TWO_CUT_OVERLAY_PATH,
    TWO_CUT_SUMMARY_PATH,
    get_fit_profile,
)
from .fit_observations import (
    flux_cut_summary,
    prepare_two_cut_experiments,
    simulate_flux_cut,
)
from .fit_optimization import (
    refresh_two_cut_final_validation,
    run_two_cut_branch_balance_refinement,
    run_two_cut_confirmation_refinement,
    run_two_cut_hierarchical_fit,
    run_two_cut_lower_v_refinement,
    run_two_cut_multifidelity_refinement,
)
from .transition_model import TransitionPrediction

__all__ = [
    "run_two_cut_extract",
    "run_two_cut_full_workflow",
    "run_two_cut_overlay",
]


def run_two_cut_extract() -> dict:
    cuts = prepare_two_cut_experiments()
    summary = {
        "workflow": "positive-ridge extraction for two fixed-phi theta sweeps",
        "cuts": [flux_cut_summary(cut) for cut in cuts],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary




def _combined_relative_matrix_element(
    prediction: TransitionPrediction,
) -> np.ndarray:
    combined = np.zeros_like(prediction.frequencies_GHz)
    for values in prediction.matrix_elements.values():
        scale = np.maximum(np.max(values, axis=1, keepdims=True), 1e-15)
        combined = np.maximum(combined, values / scale)
    return combined




def run_two_cut_overlay(workers: int = 4, point_count: int = 21) -> dict:
    """Generate final simulated spectra and overlay them on both maps."""
    if not TWO_CUT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Run the two-cut fit first; {TWO_CUT_SUMMARY_PATH} is missing."
        )
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    if not summary.get("final_validation_uses_all_track_points", False):
        summary = refresh_two_cut_final_validation(workers)
    parameters = summary["final_parameters_GHz"]
    fitted_rows = summary.get("final_assignments", [])
    overlay_cutoffs = summary.get("fit_cutoffs", PREVIEW_CUTOFFS)
    cuts = prepare_two_cut_experiments()
    predictions = {}
    arrays = {}
    for cut in cuts:
        prediction = simulate_flux_cut(
            cut=cut,
            parameters=parameters,
            cutoffs=overlay_cutoffs,
            point_count=point_count,
            workers=workers,
            drive_operators=("n1", "grid_n", "d_theta", "grid_phi"),
        )
        predictions[cut.name] = prediction
        prefix = cut.name
        arrays.update(
            {
                f"{prefix}_measured_current_uA": cut.spectrum.current_uA,
                f"{prefix}_measured_frequency_GHz": (
                    cut.spectrum.frequency_GHz
                ),
                f"{prefix}_measured_signal_uV": cut.spectrum.signal_uV,
                f"{prefix}_simulation_current_uA": prediction.currents_uA,
                f"{prefix}_theta_ext": prediction.theta_ext,
                f"{prefix}_phi_ext": prediction.phi_ext,
                f"{prefix}_energies_GHz": prediction.energies_GHz,
                f"{prefix}_initial_states": prediction.initial_states,
                f"{prefix}_final_states": prediction.final_states,
                f"{prefix}_transition_frequencies_GHz": (
                    prediction.frequencies_GHz
                ),
                f"{prefix}_relative_matrix_element": (
                    _combined_relative_matrix_element(prediction)
                ),
            }
        )
    np.savez_compressed(TWO_CUT_OVERLAY_PATH, **arrays)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1, 2, figsize=(17.0, 6.4), sharey=True, constrained_layout=True
    )
    for axis, cut in zip(axes, cuts):
        prediction = predictions[cut.name]
        color_limit = np.nanpercentile(
            np.abs(cut.spectrum.signal_uV), 99.5
        )
        image = axis.pcolormesh(
            cut.spectrum.current_uA,
            cut.spectrum.frequency_GHz,
            cut.spectrum.signal_uV.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            rasterized=True,
        )
        relative = _combined_relative_matrix_element(prediction)
        for branch in range(prediction.frequencies_GHz.shape[1]):
            frequency = prediction.frequencies_GHz[:, branch]
            visible = (
                (frequency >= 0.2)
                & (frequency <= 6.9)
                & (relative[:, branch] >= 0.012)
            )
            if np.count_nonzero(visible) < 2:
                continue
            color = (
                "#159447"
                if prediction.initial_states[branch] == 0
                else "#e6a100"
            )
            axis.plot(
                prediction.currents_uA[visible],
                frequency[visible],
                color=color,
                linewidth=0.70,
                alpha=0.34,
            )
        for row in fitted_rows:
            if row["cut_name"] != cut.name:
                continue
            current = np.asarray(row["current_uA"], dtype=float)
            observed = np.asarray(row["observed_GHz"], dtype=float)
            fitted = np.asarray(row["predicted_GHz"], dtype=float)
            is_lower_v = row.get("track_role", "").startswith("lower_v_")
            fit_color = "#7b2cbf" if is_lower_v else "#00a86b"
            marker_color = "#b47aea" if is_lower_v else "#00d084"
            axis.plot(
                current,
                fitted,
                color=fit_color,
                linewidth=2.15 if is_lower_v else 1.65,
                alpha=0.95,
                zorder=5,
            )
            axis.scatter(
                current,
                fitted,
                s=18 if is_lower_v else 14,
                facecolor=marker_color,
                edgecolor="black",
                linewidth=0.25,
                zorder=6,
            )
            axis.scatter(
                current,
                observed,
                s=18,
                marker="x",
                color="black",
                linewidth=0.75,
                zorder=7,
            )
        axis.axvline(
            cut.symmetry_center_uA,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        phi_label = "0" if cut.phi_ext == 0.0 else r"\pi"
        axis.set(
            xlabel="KITE-bias current (uA)",
            title=(
                rf"$\varphi_{{\rm ext}}={phi_label}$; "
                rf"sweep $\vartheta_{{\rm ext}}$"
            ),
            ylim=(0.2, 6.9),
        )
        figure.colorbar(image, ax=axis, label="Demodulated signal (uV)")
    axes[0].set_ylabel("Drive frequency (GHz)")
    axes[0].legend(
        handles=(
            Line2D([0], [0], color="#159447", lw=1.4, label="all visible 0 to k"),
            Line2D([0], [0], color="#e6a100", lw=1.4, label="all visible 1 to k"),
            Line2D(
                [0], [0], color="#00a86b", marker="o", lw=1.7,
                label="fitted positive ridges",
            ),
            Line2D(
                [0], [0], color="#7b2cbf", marker="o", lw=2.1,
                label="fitted lower V",
            ),
            Line2D(
                [0], [0], color="black", marker="x", lw=0,
                label="extracted measured ridges",
            ),
        ),
        loc="lower left",
        ncol=2,
        frameon=True,
        framealpha=0.88,
        fontsize=8,
    )
    figure.suptitle(
        "Joint four-mode fit: simulated bright transitions over measured maps",
        y=1.00,
    )
    figure.savefig(TWO_CUT_FIGURE_PATH, dpi=190)
    plt.close(figure)

    overlay_result = {
        "overlay_bundle": str(TWO_CUT_OVERLAY_PATH),
        "overlay_figure": str(TWO_CUT_FIGURE_PATH),
        "point_count_per_cut": point_count,
        "cutoffs": dict(overlay_cutoffs),
    }
    summary["overlay"] = overlay_result
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(overlay_result, indent=2), flush=True)
    return overlay_result


def run_two_cut_full_workflow(
    workers: int = 4,
    overlay_points: int = 21,
    *,
    profile: str = DEFAULT_FIT_PROFILE,
) -> dict:
    """Run the two-cut fit selected by one named runtime/accuracy profile."""
    fit_profile = get_fit_profile(profile)
    run_two_cut_hierarchical_fit(
        workers, optimizer_basis=fit_profile.optimizer_basis
    )

    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    summary["fit_profile"] = {
        "name": fit_profile.name,
        "optimizer_basis": fit_profile.optimizer_basis,
        "validation_basis": fit_profile.validation_basis,
        "use_basis_correction": fit_profile.use_basis_correction,
        "run_branch_balance": fit_profile.run_branch_balance,
    }
    TWO_CUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    run_two_cut_confirmation_refinement(workers)
    run_two_cut_lower_v_refinement(workers)
    if fit_profile.use_basis_correction:
        run_two_cut_multifidelity_refinement(workers)
    else:
        summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
        summary["multifidelity_correction_applied"] = False
        summary["multifidelity_skip_reason"] = (
            "The selected optimizer and validation bases are identical."
        )
        summary["fit_cutoffs"] = fit_profile.validation_cutoffs
        TWO_CUT_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )

    if fit_profile.run_branch_balance:
        run_two_cut_branch_balance_refinement(workers)
    run_two_cut_overlay(workers, overlay_points)
    summary = json.loads(TWO_CUT_SUMMARY_PATH.read_text())
    if not summary["quality_gate"]["passed"]:
        raise RuntimeError(
            "The completed two-cut fit did not pass the declared quality gate."
        )
    return summary


def _parse_two_cut_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "extract", "fit", "refine", "lower-v", "multifidelity",
            "branch-balance", "overlay", "all"
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overlay-points", type=int, default=21)
    parser.add_argument(
        "--profile",
        choices=tuple(FIT_PROFILES),
        default=DEFAULT_FIT_PROFILE,
        help="One switch for optimizer cost and final validation effort.",
    )
    return parser.parse_args()


def _two_cut_main() -> None:
    arguments = _parse_two_cut_args()
    if arguments.stage == "extract":
        run_two_cut_extract()
    elif arguments.stage == "fit":
        fit_profile = get_fit_profile(arguments.profile)
        run_two_cut_hierarchical_fit(
            arguments.workers,
            optimizer_basis=fit_profile.optimizer_basis,
        )
    elif arguments.stage == "refine":
        run_two_cut_confirmation_refinement(arguments.workers)
    elif arguments.stage == "lower-v":
        run_two_cut_lower_v_refinement(arguments.workers)
    elif arguments.stage == "multifidelity":
        run_two_cut_multifidelity_refinement(arguments.workers)
    elif arguments.stage == "branch-balance":
        run_two_cut_branch_balance_refinement(arguments.workers)
    elif arguments.stage == "overlay":
        run_two_cut_overlay(arguments.workers, arguments.overlay_points)
    else:
        run_two_cut_full_workflow(
            workers=arguments.workers,
            overlay_points=arguments.overlay_points,
            profile=arguments.profile,
        )


