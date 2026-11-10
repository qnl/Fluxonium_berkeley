"""A/B the refactored pipeline against the pre-refactor baseline.

Everything runs at PREVIEW cutoffs so the comparison is cheap; the code paths
exercised are the same ones the expensive INTERMEDIATE run uses.
"""
import json, sys
sys.path.insert(0, '/Users/eric_jin/2q_gridium-main')
import numpy as np

from tmp._baseline_check import fit_observations as OldO, fit_optimization as OldP
from tmp._baseline_check.fit_config import PREVIEW_CUTOFFS
from tmp.four_mode_gkp_fitting import fit_observations as NewO
from tmp.four_mode_gkp_fitting.fit_config import TWO_CUT_SUMMARY_PATH

PARAMS = json.loads(TWO_CUT_SUMMARY_PATH.read_text())["final_parameters_GHz"]
WORKERS = 4
failures = []

def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)

# ---------------------------------------------------------------- experiments
old_cuts = OldO.prepare_two_cut_experiments()
new_cuts = NewO.prepare_two_cut_experiments()
check("cut preparation identical",
      [c.name for c in old_cuts] == [c.name for c in new_cuts]
      and all(np.array_equal(a.spectrum.signal_uV, b.spectrum.signal_uV)
              for a, b in zip(old_cuts, new_cuts)))

# ------------------------------------- 1. assignment + all-point evaluation
# old: the hand-inlined block that appeared three times in fit_optimization
old_obs, old_diag, old_pred = [], {}, {}
for cut in old_cuts:
    prediction = OldO.simulate_flux_cut(
        cut=cut, parameters=PARAMS, cutoffs=PREVIEW_CUTOFFS, point_count=21,
        workers=WORKERS, drive_operators=("n1", "grid_n", "d_theta", "grid_phi"))
    selected, rows = OldO.assign_positive_tracks_for_cut(
        cut=cut, prediction=prediction,
        maximum_assignment_rms_MHz=180.0, maximum_points_per_track=10_000)
    selected, rows = OldO.augment_with_lower_v_observations(
        cut=cut, prediction=prediction, selected=selected, diagnostics=rows,
        maximum_assignment_rms_MHz=180.0, maximum_points_per_arm=10_000)
    old_obs.extend(selected); old_diag[cut.name] = rows; old_pred[cut.name] = prediction
_r, _w, old_rows = OldP.evaluate_observations_from_predictions(
    observations=old_obs, predictions=old_pred)
old_payload = {
    "metrics": OldO.joint_residual_metrics(old_rows),
    "per_track_metrics": OldO.per_track_residual_metrics(old_rows),
    "lower_v_metrics": OldO.residual_metrics_for_role(old_rows, "lower_v_"),
}

new_payload = NewO.evaluate_two_cut_at_basis(
    parameters=PARAMS, cutoffs=PREVIEW_CUTOFFS, workers=WORKERS, cuts=new_cuts)

check("evaluate_two_cut_at_basis: row count",
      len(old_rows) == len(new_payload["assignments"]),
      f"{len(old_rows)} branches")
old_res = np.concatenate([r["residual_MHz"] for r in old_rows])
new_res = np.concatenate([r["residual_MHz"] for r in new_payload["assignments"]])
check("evaluate_two_cut_at_basis: residuals bit-identical",
      old_res.shape == new_res.shape and np.array_equal(old_res, new_res),
      f"{old_res.size} points, max |diff| = "
      f"{np.max(np.abs(old_res - new_res)) if old_res.shape == new_res.shape else 'n/a'}")
check("evaluate_two_cut_at_basis: metrics identical",
      json.dumps(old_payload["metrics"], sort_keys=True)
      == json.dumps(new_payload["metrics"], sort_keys=True),
      f"RMS {new_payload['metrics']['combined']['rms_MHz']:.4f} MHz")
check("evaluate_two_cut_at_basis: per-track + lower-V identical",
      json.dumps(old_payload["per_track_metrics"], sort_keys=True)
      == json.dumps(new_payload["per_track_metrics"], sort_keys=True)
      and json.dumps(old_payload["lower_v_metrics"], sort_keys=True)
      == json.dumps(new_payload["lower_v_metrics"], sort_keys=True))

# ------------------------- 2. evaluate_joint_track_observations delegation
old_raw, old_wt, old_jrows = OldO.evaluate_joint_track_observations(
    cuts=old_cuts, observations=old_obs, parameters=PARAMS,
    cutoffs=PREVIEW_CUTOFFS, workers=WORKERS, simulation_points=13)
new_raw, new_wt, new_jrows = NewO.evaluate_joint_track_observations(
    cuts=new_cuts, observations=old_obs, parameters=PARAMS,
    cutoffs=PREVIEW_CUTOFFS, workers=WORKERS, simulation_points=13)
check("evaluate_joint_track_observations: raw residuals bit-identical",
      np.array_equal(old_raw, new_raw), f"{old_raw.size} points")
check("evaluate_joint_track_observations: weighted residuals bit-identical",
      np.array_equal(old_wt, new_wt))
check("evaluate_joint_track_observations: rows identical",
      json.dumps(old_jrows, sort_keys=True) == json.dumps(new_jrows, sort_keys=True))

# ------------------------------------------ 3. assign_joint_positive_tracks
old_a, old_d = OldO.assign_joint_positive_tracks(
    cuts=old_cuts, parameters=PARAMS, cutoffs=PREVIEW_CUTOFFS, workers=WORKERS,
    simulation_points=17, maximum_assignment_rms_MHz=170.0,
    maximum_points_per_track=10, include_lower_v=True)
new_a, new_d = NewO.assign_joint_positive_tracks(
    cuts=new_cuts, parameters=PARAMS, cutoffs=PREVIEW_CUTOFFS, workers=WORKERS,
    simulation_points=17, maximum_assignment_rms_MHz=170.0,
    maximum_points_per_track=10, include_lower_v=True)
check("assign_joint_positive_tracks: same observations",
      len(old_a) == len(new_a)
      and all(np.array_equal(x.observed_GHz, y.observed_GHz)
              and np.array_equal(x.final_states, y.final_states)
              and x.track_id == y.track_id
              for x, y in zip(old_a, new_a)),
      f"{len(new_a)} tracks")
check("assign_joint_positive_tracks: same diagnostics",
      json.dumps(old_d, sort_keys=True) == json.dumps(new_d, sort_keys=True))

# ------------------------------------------- 4. rows_with_residual_vector
probe = np.linspace(-0.03, 0.05, old_raw.size)
check("rows_with_residual_vector identical",
      json.dumps(OldP._rows_with_residual_vector(old_jrows, probe), sort_keys=True)
      == json.dumps(NewO.rows_with_residual_vector(new_jrows, probe), sort_keys=True))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
