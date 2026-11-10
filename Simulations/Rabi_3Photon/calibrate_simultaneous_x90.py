"""Run the Pareto calibration for the simultaneous X90 protocol."""

from pathlib import Path

from Simulations.Rabi_3Photon import calibrate_simultaneous_x180 as calibration


DEFAULT_CONFIG = (
    Path(__file__).parent / 'yamls' / 'experiments'
    / 'idealgridium_soft_candidate_x90.yaml')


def main():
    experiment, results, front, correlations = calibration.run_search(
        DEFAULT_CONFIG)
    calibration._print_results(experiment, results, front, correlations)


if __name__ == '__main__':
    main()
