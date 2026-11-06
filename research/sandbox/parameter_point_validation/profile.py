"""Sandbox registration for the frozen eps_J=0.11 N71 k300 profile."""
from copy import deepcopy
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
CRITERIA = HERE / 'criteria.json'
PROFILE_NAME = 'n71-k300-epsj011-heldout'
FINAL_DIRECTORY_NAME = 'four-mode-asymmetric-n71-epsj011-k300'
EXPECTED_CONFIGURATION_SHA256 = '8ed1b0c72746a2594e9139d4cc782a86007a3b416394eeab71da071dc8364646'
RETAINED_CASES = tuple((f'nested_prefix_k{k}', k) for k in (180,200,220,240,260,280,300))


def expected_definition():
    frozen = json.loads(CRITERIA.read_text())
    profile = frozen['profile']
    if frozen['status'] != 'preregistered-before-any-epsj011-stage1-or-dressed-result':
        raise ValueError('Criteria status mismatch')
    if profile['name'] != PROFILE_NAME or profile['configuration_sha256'] != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError('Profile identity mismatch')
    return {'configuration': deepcopy(profile['configuration']),
            'retained_cases': RETAINED_CASES, 'retained_index_mode': 'prefix',
            'final_directory_name': FINAL_DIRECTORY_NAME}


def install_profile(runner):
    definition = expected_definition()
    observed = runner.configuration_fingerprint(definition['configuration'])
    if observed != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError('Configuration fingerprint mismatch')
    if PROFILE_NAME in runner.PROFILE_DEFINITIONS:
        if runner._canonical_json(runner.PROFILE_DEFINITIONS[PROFILE_NAME]) != runner._canonical_json(definition):
            raise ValueError('Conflicting profile registration')
    else:
        runner.PROFILE_DEFINITIONS[PROFILE_NAME] = definition
    runner.CONFIGURATION_SHA256_BY_PROFILE[PROFILE_NAME] = observed
    for rel in (
        'research/sandbox/parameter_point_validation/criteria.json',
        'research/sandbox/parameter_point_validation/profile.py',
        'research/sandbox/parameter_point_validation/frozen_selector.py',
        'research/sandbox/parameter_point_validation/selection_hook.py',
        'research/sandbox/parameter_point_validation/parameter_analysis.py',
        'research/sandbox/parameter_point_validation/analyze_parameter_point.py',
        'research/sandbox/parameter_point_validation/evaluate.py',
        'research/sandbox/parameter_point_validation/preregistration_manifest.json'):
        runner.PROVENANCE_SOURCE_FILES = tuple(dict.fromkeys((*runner.PROVENANCE_SOURCE_FILES, Path(rel))))
    runner.validate_frozen_configuration(definition['configuration'], PROFILE_NAME)
    return definition
