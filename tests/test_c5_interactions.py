"""Self-check for Step C5. Run directly: `python tests/test_c5_interactions.py`

Synthetic agent tracks in the EGO FRAME with analytically known answers, plus
regressions for the defects C5 exists to fix and the ones found while building it.

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groundtruth import (  # noqa: E402
    CUT_IN_LOOKBACK_S, INTERACTION_TAGS, LEAD_BRAKE_MPS2, LEAD_LATERAL_M,
    LEAD_RANGE_M, LEAD_TREND_S, MIN_AGENT_SPEED_MPS, PED_INTERACT_RANGE_M,
    PED_LATERAL_MPS, _lead_of, _state_before, interaction_labels,
)

ROOT = Path(__file__).resolve().parent.parent
DT = 0.5   # nuScenes keyframes are 2 Hz


def _agent(cat="vehicle.car", fwd=10.0, lat=0.0, v_fwd=0.0, v_lat=0.0,
           t_s=0.0, in_frustum=True):
    return {"cat": cat, "fwd_m": fwd, "lat_m": lat, "v_fwd_mps": v_fwd,
            "v_lat_mps": v_lat, "speed_mps": float(np.hypot(v_fwd, v_lat)),
            "in_frustum": in_frustum, "t_s": t_s}


def _scene(*per_frame):
    """Frames from a list of {key: kwargs}; t_s is filled in at 2 Hz."""
    frames = []
    for i, spec in enumerate(per_frame):
        frames.append({k: _agent(t_s=i * DT, **kw) for k, kw in spec.items()})
    return frames


def test_lead_is_the_ego_frame_forward_axis_not_the_global_one():
    """Regression for the defect C5 exists to fix.

    The original test was `abs(dy) > abs(dx)` on GLOBAL coordinates: a wedge pointing
    along global north/south, not along the car. It fires on vehicles BEHIND the ego
    and misses real leads whenever the car is not driving along the global Y axis.
    Here the ego frame says the car is 12 m behind; no global-frame reasoning applies.
    """
    behind = _scene({"a": dict(fwd=-12.0, lat=0.0)})
    assert interaction_labels(behind, 0)["lead_vehicle"] is False
    ahead = _scene({"a": dict(fwd=12.0, lat=0.0)})
    assert interaction_labels(ahead, 0)["lead_vehicle"] is True


def test_lead_must_be_in_the_lane_corridor_and_in_range():
    assert _lead_of(_scene({"a": dict(fwd=10.0, lat=LEAD_LATERAL_M + 0.1)})[0])[0] is None
    assert _lead_of(_scene({"a": dict(fwd=LEAD_RANGE_M + 1, lat=0.0)})[0])[0] is None
    assert _lead_of(_scene({"a": dict(fwd=10.0, lat=LEAD_LATERAL_M - 0.1)})[0])[0] == "a"


def test_lead_is_the_nearest_candidate_not_an_arbitrary_one():
    frames = _scene({"far": dict(fwd=30.0), "near": dict(fwd=8.0),
                     "mid": dict(fwd=15.0)})
    key, lead = _lead_of(frames[0])
    assert key == "near" and lead["fwd_m"] == 8.0


def test_pedestrians_are_never_the_lead_vehicle():
    frames = _scene({"p": dict(cat="human.pedestrian.adult", fwd=6.0, lat=0.0)})
    assert interaction_labels(frames, 0)["lead_vehicle"] is False


def test_lead_braking_uses_the_trend_baseline_not_the_previous_frame():
    """R6. box_velocity is already a centred difference, so re-differencing it at 0.5 s
    flips sign 29.6% of the time; at 1.5 s that falls to 16.8%.

    This track slows steadily. The label must be computed against the state
    LEAD_TREND_S back, so the measured deceleration is the real one.
    """
    n = int(LEAD_TREND_S / DT) + 1
    frames = _scene(*[{"a": dict(fwd=15.0, v_fwd=10.0 - 1.0 * (i * DT))} for i in range(n + 1)])
    row = interaction_labels(frames, n)
    assert row["lead_vehicle"] and row["lead_braking"]
    assert np.isclose(row["lead_accel_mps2"], -1.0, atol=0.05), row["lead_accel_mps2"]


def test_lead_accelerating_is_not_braking():
    n = int(LEAD_TREND_S / DT) + 1
    frames = _scene(*[{"a": dict(fwd=15.0, v_fwd=2.0 + 1.0 * (i * DT))} for i in range(n + 1)])
    row = interaction_labels(frames, n)
    assert row["lead_braking"] is False and row["lead_accel_mps2"] > 0


def test_lead_braking_needs_the_same_instance():
    """A different car arriving in the lane is not the first one braking.

    Tracking is by instance_token precisely so a swap cannot read as a deceleration.
    """
    n = int(LEAD_TREND_S / DT) + 1
    frames = _scene(*[{"old": dict(fwd=15.0, v_fwd=10.0)} for _ in range(n)],
                    {"new": dict(fwd=15.0, v_fwd=1.0)})
    row = interaction_labels(frames, n)
    assert row["lead_vehicle"] and not row["lead_braking"]
    assert np.isnan(row["lead_accel_mps2"]), "no shared history means no estimate"


def test_lead_columns_are_nan_when_there_is_no_lead():
    row = interaction_labels(_scene({"a": dict(fwd=-5.0)}), 0)
    assert row["lead_vehicle"] is False
    assert np.isnan(row["lead_distance_m"]) and np.isnan(row["lead_speed_mps"])


def test_cut_in_requires_entering_the_corridor():
    n = int(CUT_IN_LOOKBACK_S / DT) + 1
    outside = LEAD_LATERAL_M + 2.0
    frames = _scene(*[{"a": dict(fwd=15.0, lat=outside, v_fwd=8.0, v_lat=-1.5)}
                      for _ in range(n)],
                    {"a": dict(fwd=15.0, lat=0.5, v_fwd=8.0, v_lat=-1.5)})
    assert interaction_labels(frames, n)["cut_in"] is True


def test_a_car_already_in_the_lane_is_not_cutting_in():
    n = int(CUT_IN_LOOKBACK_S / DT) + 1
    frames = _scene(*[{"a": dict(fwd=15.0, lat=0.4, v_fwd=8.0, v_lat=-0.2)}
                      for _ in range(n + 1)])
    assert interaction_labels(frames, n)["cut_in"] is False


def test_parked_cars_do_not_cut_in_when_the_ego_turns():
    """Regression for F-035, and the reason three signals must agree.

    The ego frame rotates with the car, so during a turn a STATIONARY kerbside vehicle
    sweeps across the corridor boundary. Its ego-frame lateral position moves inward
    exactly as a real cut-in would; what it does not have is ground speed. Detecting on
    position alone would have manufactured cut-ins out of the ego's own steering.
    """
    n = int(CUT_IN_LOOKBACK_S / DT) + 1
    frames = _scene(*[{"parked": dict(fwd=15.0, lat=LEAD_LATERAL_M + 2.0,
                                      v_fwd=0.0, v_lat=0.0)} for _ in range(n)],
                    {"parked": dict(fwd=15.0, lat=0.5, v_fwd=0.0, v_lat=0.0)})
    row = interaction_labels(frames, n)
    assert row["cut_in"] is False, "a stationary car cannot cut in"
    assert row["n_cut_in"] == 0
    # and the guard is the speed one, at the stated threshold
    assert frames[n]["parked"]["speed_mps"] < MIN_AGENT_SPEED_MPS


def test_cut_in_requires_lateral_velocity_pointing_inward():
    """A car drifting OUT of the lane, caught mid-frame, is not cutting in."""
    n = int(CUT_IN_LOOKBACK_S / DT) + 1
    frames = _scene(*[{"a": dict(fwd=15.0, lat=LEAD_LATERAL_M + 1.0, v_fwd=8.0, v_lat=+1.5)}
                      for _ in range(n)],
                    {"a": dict(fwd=15.0, lat=1.0, v_fwd=8.0, v_lat=+1.5)})
    assert interaction_labels(frames, n)["cut_in"] is False


def test_pedestrian_crossing_needs_inward_motion_and_arrival_in_time():
    near = _scene({"p": dict(cat="human.pedestrian.adult", fwd=10.0, lat=4.0,
                             v_lat=-1.2)})
    assert interaction_labels(near, 0)["pedestrian_crossing_path"] is True
    # same speed, but walking away from the centreline
    away = _scene({"p": dict(cat="human.pedestrian.adult", fwd=10.0, lat=4.0,
                             v_lat=+1.2)})
    assert interaction_labels(away, 0)["pedestrian_crossing_path"] is False
    # inward, but too slow to arrive within PED_TIME_TO_CORRIDOR_S
    slow = _scene({"p": dict(cat="human.pedestrian.adult", fwd=10.0, lat=18.0,
                             v_lat=-PED_LATERAL_MPS - 0.01)})
    assert interaction_labels(slow, 0)["pedestrian_crossing_path"] is False


def test_pedestrian_standing_still_is_not_crossing():
    still = _scene({"p": dict(cat="human.pedestrian.adult", fwd=8.0, lat=3.0, v_lat=0.0)})
    assert interaction_labels(still, 0)["pedestrian_crossing_path"] is False


def test_pedestrian_behind_or_far_does_not_count():
    back = _scene({"p": dict(cat="human.pedestrian.adult", fwd=-8.0, lat=3.0, v_lat=-1.2)})
    assert interaction_labels(back, 0)["pedestrian_crossing_path"] is False
    far = _scene({"p": dict(cat="human.pedestrian.adult",
                            fwd=PED_INTERACT_RANGE_M + 1.0, lat=0.5, v_lat=-1.2)})
    assert interaction_labels(far, 0)["pedestrian_crossing_path"] is False


def test_tags_are_360_degree_facts_and_frustum_is_only_provenance():
    """D-026. An agent the front camera cannot see still counts.

    The `*_in_frustum` counts record how much of each tag the camera could see, so
    Stage F can report field-of-view effects (L-019) without the tag itself depending
    on the camera.
    """
    hidden = _scene({"p": dict(cat="human.pedestrian.adult", fwd=5.0, lat=4.0,
                               v_lat=-1.5, in_frustum=False)})
    row = interaction_labels(hidden, 0)
    assert row["pedestrian_crossing_path"] is True
    assert row["n_pedestrian_crossing_path"] == 1
    assert row["n_pedestrian_crossing_path_in_frustum"] == 0


def test_nan_velocity_never_reads_as_a_stationary_agent():
    """0.2% of annotations appear in one keyframe only, so box_velocity is NaN.

    Zero-filling would make them look parked - and a NaN comparison is False, so every
    motion test must fail closed rather than silently pass.
    """
    n = int(CUT_IN_LOOKBACK_S / DT) + 1
    nan_agent = dict(fwd=15.0, lat=0.5, v_fwd=float("nan"), v_lat=float("nan"))
    frames = _scene(*[{"a": dict(fwd=15.0, lat=LEAD_LATERAL_M + 2.0,
                                 v_fwd=float("nan"), v_lat=float("nan"))}
                      for _ in range(n)], {"a": nan_agent})
    row = interaction_labels(frames, n)
    assert row["cut_in"] is False
    assert row["lead_vehicle"] is True, "position is still known, only velocity is not"
    assert np.isnan(row["lead_speed_mps"])


def test_state_before_walks_by_time_not_by_frame_count():
    frames = _scene(*[{"a": dict(fwd=15.0, v_fwd=float(i))} for i in range(10)])
    past = _state_before(frames, 6, "a", LEAD_TREND_S)
    assert past is not None
    assert frames[6]["a"]["t_s"] - past["t_s"] <= LEAD_TREND_S + 1e-6
    assert _state_before(frames, 0, "a", LEAD_TREND_S) is None, "nothing before the start"


def test_state_before_stops_at_a_track_gap():
    """An agent that disappears and returns must not be differenced across the hole."""
    frames = _scene({"a": dict(v_fwd=10.0)}, {}, {"a": dict(v_fwd=2.0)})
    assert _state_before(frames, 2, "a", LEAD_TREND_S) is None


def test_real_table_matches_the_predicates():
    """Guarded end-to-end check against the cached mini table."""
    import pandas as pd
    p = ROOT / "outputs" / "interactions_mini.parquet"
    if not p.exists():
        print("     (skip: outputs/interactions_mini.parquet not built)")
        return
    df = pd.read_parquet(p)
    assert len(df) == 404 and df.sample_token.is_unique
    assert ((df.n_cut_in > 0) == df.cut_in).all()
    assert ((df.n_pedestrian_crossing_path > 0) == df.pedestrian_crossing_path).all()
    assert (df.n_cut_in_in_frustum <= df.n_cut_in).all()
    assert (~df.lead_braking | df.lead_vehicle).all(), "lead_braking implies lead_vehicle"
    lead = df[df.lead_vehicle]
    assert (lead.lead_distance_m > 0).all() and (lead.lead_distance_m < LEAD_RANGE_M).all()
    assert df.lead_distance_m.isna().equals(~df.lead_vehicle)
    assert (lead.lead_speed_mps.dropna() < 35).all(), "faster than any road vehicle"
    for t in INTERACTION_TAGS:
        assert df[t].dtype == bool, t


def test_thresholds_are_consistent_with_the_rest_of_the_pipeline():
    import src.groundtruth as G
    assert LEAD_TREND_S == G.TREND_S, "same trend width as C2, or say why not"
    assert LEAD_BRAKE_MPS2 == -G.ACCEL_ENTER_MPS2, \
        "'lead braking' and 'ego decelerating' should mean the same deceleration"
    assert 0 < LEAD_LATERAL_M < 3.5, "a corridor wider than a lane is not a lane"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C5 self-checks passed")
