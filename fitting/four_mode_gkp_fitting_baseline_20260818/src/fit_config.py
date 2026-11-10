
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUTHOR_DEVICE_PARAMETERS",
    "BASIS_PRESETS",
    "DEFAULT_FIT_PROFILE",
    "FIT_PROFILES",
    "INTERMEDIATE_CUTOFFS",
    "JOINT_COARSE_CUTOFFS",
    "JOINT_MEDIUM_CUTOFFS",
    "JOINT_PARAMETER_BOUNDS",
    "JOINT_PARAMETER_STEPS",
    "JOINT_PARAMETER_TRUST",
    "JOINT_STARTING_PARAMETERS",
    "PHI0_DATA_PATH",
    "PHIPI_DATA_PATH",
    "PREVIEW_CUTOFFS",
    "REFINED_SPOT_CHECK_CUTOFFS",
    "TWO_CUT_CACHE_DIRECTORY",
    "TWO_CUT_FIGURE_PATH",
    "TWO_CUT_OVERLAY_PATH",
    "TWO_CUT_QUALITY_LIMITS",
    "TWO_CUT_RESULTS_DIRECTORY",
    "TWO_CUT_SUMMARY_PATH",
    "FitProfile",
    "get_fit_profile",
    "two_cut_optimizer_cutoffs",
]


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
TWO_CUT_DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

PHI0_DATA_PATH = Path(
    os.environ.get(
        "GRIDIUM_PHI0_DATA_PATH",
        TWO_CUT_DATA_DIRECTORY
        / "Gridium_v4_1_x2y7_GKP1_phi0_theta_sweep.txt",
    )
)
PHIPI_DATA_PATH = Path(
    os.environ.get(
        "GRIDIUM_PHIPI_DATA_PATH",
        TWO_CUT_DATA_DIRECTORY
        / "Gridium_v4_1_x2y7_GKP1_phipi_theta_sweep.txt",
    )
)
TWO_CUT_RESULTS_DIRECTORY = Path(
    os.environ.get("GRIDIUM_TWO_CUT_RESULTS_DIRECTORY", PROJECT_DIRECTORY / "results")
)
TWO_CUT_CACHE_DIRECTORY = Path(
    os.environ.get(
        "GRIDIUM_TWO_CUT_CACHE_DIRECTORY",
        Path(tempfile.gettempdir()) / "four_mode_gkp_two_cut_cache",
    )
)
TWO_CUT_SUMMARY_PATH = TWO_CUT_RESULTS_DIRECTORY / "two_cut_fit_summary.json"
TWO_CUT_OVERLAY_PATH = TWO_CUT_RESULTS_DIRECTORY / "two_cut_fit_overlay.npz"
TWO_CUT_FIGURE_PATH = TWO_CUT_RESULTS_DIRECTORY / "two_cut_fit_overlay.png"


AUTHOR_DEVICE_PARAMETERS = {
    "EJ": 7.212,
    "EC": 0.77,
    "EL": 0.733,
    "ELK": 0.733,
    "EJS": 1.13,
    "ECS": 8.13,
    "eC": 0.70,
    "eP": 1.00,
    "eps_J": 0.0,
    "eps_LK": 0.0,
    "ng": 0.0,
}

JOINT_STARTING_PARAMETERS = {
    **AUTHOR_DEVICE_PARAMETERS,
    "EJ": 8.5027919483608,
    "EL": 0.4880342002652243,
    "ELK": 0.4880342002652243,
    "eP": 1.3592553622021752,
    "eps_LK": 0.05,
}


JOINT_COARSE_CUTOFFS = {
    "n1max": 2,
    "N2": 21,
    "L2": 11.0,
    "N3": 21,
    "L3": 14.0,
    "N4": 5,
    "nkeep": 30,
}
JOINT_MEDIUM_CUTOFFS = {
    "n1max": 3,
    "N2": 25,
    "L2": 11.0,
    "N3": 25,
    "L3": 14.0,
    "N4": 6,
    "nkeep": 40,
}
PREVIEW_CUTOFFS = {
    "n1max": 3,
    "N2": 31,
    "L2": 11.0,
    "N3": 31,
    "L3": 14.0,
    "N4": 8,
    "nkeep": 60,
}
INTERMEDIATE_CUTOFFS = {
    "n1max": 4,
    "N2": 51,
    "L2": 11.0,
    "N3": 51,
    "L3": 14.0,
    "N4": 8,
    "nkeep": 120,
}
REFINED_SPOT_CHECK_CUTOFFS = {
    "n1max": 5,
    "N2": 71,
    "L2": 11.0,
    "N3": 71,
    "L3": 14.0,
    "N4": 8,
    "nkeep": 200,
}

BASIS_PRESETS = {
    "coarse": JOINT_COARSE_CUTOFFS,
    "medium": JOINT_MEDIUM_CUTOFFS,
    "preview": PREVIEW_CUTOFFS,
    "intermediate": INTERMEDIATE_CUTOFFS,
    "refined": REFINED_SPOT_CHECK_CUTOFFS,
}


@dataclass(frozen=True)
class FitProfile:
    """Named balance between runtime and final validation effort."""

    name: str
    optimizer_basis: str
    validation_basis: str
    use_basis_correction: bool
    run_branch_balance: bool

    @property
    def optimizer_cutoffs(self) -> dict:
        return dict(BASIS_PRESETS[self.optimizer_basis])

    @property
    def validation_cutoffs(self) -> dict:
        return dict(BASIS_PRESETS[self.validation_basis])


FIT_PROFILES = {
    #fast optimizer plus the already validated preview-to-intermediate correction
    "preview": FitProfile(
        name="preview",
        optimizer_basis="preview",
        validation_basis="intermediate",
        use_basis_correction=True,
        run_branch_balance=False,
    ),
  
    "production": FitProfile(
        name="production",
        optimizer_basis="preview",
        validation_basis="intermediate",
        use_basis_correction=True,
        run_branch_balance=True,
    ),
}
DEFAULT_FIT_PROFILE = "preview"


def get_fit_profile(profile: str | FitProfile = DEFAULT_FIT_PROFILE) -> FitProfile:
    """Resolve a profile name and provide a clear error for invalid values."""
    if isinstance(profile, FitProfile):
        return profile
    try:
        return FIT_PROFILES[str(profile)]
    except KeyError as error:
        choices = ", ".join(FIT_PROFILES)
        raise ValueError(f"Unknown fit profile {profile!r}; choose {choices}.") from error


def two_cut_optimizer_cutoffs(basis: str) -> dict:
    """Return a copy of a supported optimizer basis."""
    if basis not in ("preview", "intermediate"):
        raise ValueError("optimizer basis must be 'preview' or 'intermediate'.")
    return dict(BASIS_PRESETS[basis])


JOINT_PARAMETER_BOUNDS = {
    "EJ": (5.0, 12.0),
    "EC": (0.35, 1.50),
    "EL": (0.20, 1.20),
    "ELK": (0.20, 1.20),
    "EJS": (0.40, 3.00),
    "ECS": (4.00, 14.0),
    "eC": (0.30, 2.00),
    "eP": (0.50, 2.80),
    "eps_J": (-0.20, 0.20),
    "eps_LK": (-0.20, 0.20),
}
JOINT_PARAMETER_STEPS = {
    "EJ": 0.06,
    "EC": 0.015,
    "EL": 0.012,
    "ELK": 0.012,
    "EJS": 0.025,
    "ECS": 0.12,
    "eC": 0.025,
    "eP": 0.025,
    "eps_J": 0.008,
    "eps_LK": 0.008,
}
JOINT_PARAMETER_TRUST = {
    "EJ": 0.80,
    "EC": 0.16,
    "EL": 0.12,
    "ELK": 0.12,
    "EJS": 0.30,
    "ECS": 1.20,
    "eC": 0.28,
    "eP": 0.30,
    "eps_J": 0.045,
    "eps_LK": 0.045,
}
TWO_CUT_QUALITY_LIMITS = {
    "maximum_rms_MHz_per_cut": 100.0,
    "maximum_rms_MHz_per_track": 150.0,
    "minimum_fraction_within_100_MHz_per_cut": 0.75,
    "minimum_retained_tracks_per_cut": 12,
    "maximum_lower_v_rms_MHz": 100.0,
    "required_lower_v_arms": 2,
}
