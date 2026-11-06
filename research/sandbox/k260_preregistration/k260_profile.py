"""Sandbox-only registration of the preregistered N71 k260 profile.

This module changes no production file.  It extends the imported frozen runner
in memory, then reuses its unchanged assembly, solve, analysis, validation and
atomic-publication paths.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


PROFILE_NAME = "n71-k260"
EXPECTED_CONFIGURATION_SHA256 = (
    "990dad04b52deeef9d90908c6c19ce38e917d7f5b7fab32f0e39699c3c83c4bb"
)
FINAL_DIRECTORY_NAME = "four-mode-asymmetric-n71-k260"
RETAINED_CASES = (
    ("nested_prefix_k180", 180),
    ("nested_prefix_k200", 200),
    ("nested_prefix_k220", 220),
    ("nested_prefix_k240", 240),
    ("nested_prefix_k260", 260),
)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CRITERIA_PATH = HERE / "criteria.json"


def expected_definition():
    criteria = json.loads(CRITERIA_PATH.read_text())
    assert criteria["status"] == "frozen-before-any-k260-result"
    profile = criteria["profile"]
    assert profile["name"] == PROFILE_NAME
    assert profile["configuration_sha256"] == EXPECTED_CONFIGURATION_SHA256
    assert profile["retained_cases"] == [180, 200, 220, 240, 260]
    return {
        "configuration": deepcopy(profile["configuration"]),
        "retained_cases": RETAINED_CASES,
        "retained_index_mode": "prefix",
        "final_directory_name": FINAL_DIRECTORY_NAME,
    }


def install_profile(runner):
    """Install the preregistered profile into one imported runner process."""
    definition = expected_definition()
    observed = runner.configuration_fingerprint(definition["configuration"])
    assert observed == EXPECTED_CONFIGURATION_SHA256
    if PROFILE_NAME in runner.PROFILE_DEFINITIONS:
        assert runner._canonical_json(runner.PROFILE_DEFINITIONS[PROFILE_NAME]) == (
            runner._canonical_json(definition)
        )
    else:
        runner.PROFILE_DEFINITIONS[PROFILE_NAME] = definition
    runner.CONFIGURATION_SHA256_BY_PROFILE[PROFILE_NAME] = observed
    provenance_files = (
        Path("research/sandbox/k260_preregistration/criteria.json"),
        Path("research/sandbox/k260_preregistration/k260_profile.py"),
        Path("research/sandbox/k260_preregistration/guarded_k260.py"),
    )
    runner.PROVENANCE_SOURCE_FILES = tuple(dict.fromkeys(
        (*runner.PROVENANCE_SOURCE_FILES, *provenance_files)
    ))
    runner.validate_frozen_configuration(
        definition["configuration"], PROFILE_NAME
    )
    return definition

