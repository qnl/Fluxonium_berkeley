"""Sandbox registration for the frozen epsilon_J=0.11 N71 k380 profile."""
from copy import deepcopy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRITERIA = HERE / "criteria.json"
PROFILE_NAME = "n71-k380-epsj011-convergence"
FINAL_DIRECTORY_NAME = "four-mode-asymmetric-n71-epsj011-k380"
EXPECTED_CONFIGURATION_SHA256 = "8f02dbfc09f6a6c4e6bcd49d938c79d570b063103edf33beaad8b331e085ac5e"
RETAINED_CASES = tuple(
    (f"nested_prefix_k{k}", k) for k in (300, 320, 340, 360, 380)
)


def expected_definition():
    frozen = json.loads(CRITERIA.read_text())
    profile = frozen["profile"]
    if frozen["status"] != "frozen-before-any-k380-stage1-or-new-shell-dressed-result":
        raise ValueError("Criteria status mismatch")
    if profile["name"] != PROFILE_NAME:
        raise ValueError("Profile name mismatch")
    if profile["configuration_sha256"] != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError("Profile fingerprint mismatch")
    return {
        "configuration": deepcopy(profile["configuration"]),
        "retained_cases": RETAINED_CASES,
        "retained_index_mode": "prefix",
        "final_directory_name": FINAL_DIRECTORY_NAME,
    }


def install_profile(runner):
    definition = expected_definition()
    observed = runner.configuration_fingerprint(definition["configuration"])
    if observed != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError("Configuration fingerprint mismatch")
    if PROFILE_NAME in runner.PROFILE_DEFINITIONS:
        current = runner._canonical_json(runner.PROFILE_DEFINITIONS[PROFILE_NAME])
        if current != runner._canonical_json(definition):
            raise ValueError("Conflicting profile registration")
    else:
        runner.PROFILE_DEFINITIONS[PROFILE_NAME] = definition
    runner.CONFIGURATION_SHA256_BY_PROFILE[PROFILE_NAME] = observed
    for rel in (
        "research/sandbox/epsj011_k380_preregistration/criteria.json",
        "research/sandbox/epsj011_k380_preregistration/profile.py",
        "research/sandbox/epsj011_k380_preregistration/selection_hook.py",
        "research/sandbox/epsj011_k380_preregistration/parent_analysis.py",
        "research/sandbox/epsj011_k380_preregistration/analyze_k380.py",
        "research/sandbox/epsj011_k380_preregistration/evaluate.py",
        "research/sandbox/epsj011_k380_preregistration/preregistration_manifest.json",
    ):
        runner.PROVENANCE_SOURCE_FILES = tuple(
            dict.fromkeys((*runner.PROVENANCE_SOURCE_FILES, Path(rel)))
        )
    runner.validate_frozen_configuration(definition["configuration"], PROFILE_NAME)
    return definition
