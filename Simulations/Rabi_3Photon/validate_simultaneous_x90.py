"""Validate the frozen simultaneous X90 candidate."""

from pathlib import Path

from Simulations.Rabi_3Photon import validate_simultaneous_x180 as validation


DEFAULT_CONFIG = (
    Path(__file__).parent / 'yamls' / 'experiments'
    / 'idealgridium_soft_candidate_x90.yaml')


def main():
    validation._print_results(validation.run_validation(DEFAULT_CONFIG))


if __name__ == '__main__':
    main()
