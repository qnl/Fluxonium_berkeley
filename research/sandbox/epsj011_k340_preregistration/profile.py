"""Sandbox registration for the frozen epsilon_J=0.11 N71 k340 profile."""
from copy import deepcopy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRITERIA = HERE / "criteria.json"
PROFILE_NAME = "n71-k340-epsj011-convergence"
FINAL_DIRECTORY_NAME = "four-mode-asymmetric-n71-epsj011-k340"
EXPECTED_CONFIGURATION_SHA256 = "10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0"
RETAINED_CASES = tuple(
    (f"nested_prefix_k{k}", k) for k in (260, 280, 300, 320, 340)
)


def expected_definition():
    frozen = json.loads(CRITERIA.read_text())
    profile = frozen["profile"]
    if frozen["status"] != "frozen-before-any-k340-stage1-or-new-shell-dressed-result":
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
        "research/sandbox/epsj011_k340_preregistration/criteria.json",
        "research/sandbox/epsj011_k340_preregistration/profile.py",
        "research/sandbox/epsj011_k340_preregistration/selection_hook.py",
        "research/sandbox/epsj011_k340_preregistration/parent_analysis.py",
        "research/sandbox/epsj011_k340_preregistration/analyze_k340.py",
        "research/sandbox/epsj011_k340_preregistration/evaluate.py",
        "research/sandbox/epsj011_k340_preregistration/preregistration_manifest.json",
    ):
        runner.PROVENANCE_SOURCE_FILES = tuple(
            dict.fromkeys((*runner.PROVENANCE_SOURCE_FILES, Path(rel)))
        )
    runner.validate_frozen_configuration(definition["configuration"], PROFILE_NAME)
    return definition
