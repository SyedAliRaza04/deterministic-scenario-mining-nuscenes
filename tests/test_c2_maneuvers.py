"""Self-check for Step C2. Run directly: `python tests/test_c2_maneuvers.py`

Synthetic trajectories with known manoeuvres, plus regressions for the two real bugs
found while building C2 (F-015 noisy-signal detection, F-016 tag-name drift).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.groundtruth as G  # noqa: E402
from src.groundtruth import (  # noqa: E402
    LATERAL_LABELS, LONGITUDINAL_LABELS, PoseTrack, _hysteresis_segments,
    _merge_same_sign, lateral_events, longitudinal_events, maneuver_labels,
)

FS = 50.0


def _track(yaw, speed, dur):
    """Synthetic track from yaw(t) and speed(t) callables."""
    t = np.arange(0, dur, 1 / FS)
    y = np.array([yaw(u) for u in t])
    sp = np.array([speed(u) for u in t])
    dt = 1 / FS
    x = np.cumsum(sp * np.cos(y)) * dt
    yy = np.cumsum(sp * np.sin(y)) * dt
    from scipy.signal import savgol_filter
    w = int(round(G.TREND_S * FS)) | 1
    w = max(5, min(w, (len(sp) // 2) * 2 - 1))
    return PoseTrack(
        t_s=t, x=x, y=yy, yaw_rad=y, speed_mps=sp,
        accel_mps2=np.gradient(sp, dt),
        yaw_rate_deg_s=np.degrees(np.gradient(y, dt)),
        accel_trend_mps2=np.gradient(savgol_filter(sp, w, 2), dt),
        t0_us=0, n_raw_poses=len(t), resample_hz=FS,
    )


def test_hysteresis_widens_to_exit_threshold():
    """Boundaries must backtrack to the LOWER threshold, not the confirm threshold."""
    t = np.arange(0, 10, 1 / FS)
    x = np.zeros_like(t)
    x[(t > 3) & (t < 7)] = 0.5      # shoulder, above exit only
    x[(t > 4) & (t < 6)] = 5.0      # peak, above enter
    segs = _hysteresis_segments(x, t, enter=4.0, exit_=0.25, min_dur_s=0.5)
    assert len(segs) == 1
    a, b = segs[0]
    assert t[a] < 3.1 and t[b] > 6.9, "segment must extend across the whole shoulder"


def test_hysteresis_rejects_short_spikes():
    t = np.arange(0, 10, 1 / FS)
    x = np.zeros_like(t)
    x[(t > 5) & (t < 5.1)] = 9.0
    assert _hysteresis_segments(x, t, 4.0, 1.5, min_dur_s=0.5) == []


def test_merge_joins_same_direction_only():
    t = np.arange(0, 10, 1 / FS)
    sig = np.ones_like(t)
    assert len(_merge_same_sign([(0, 100), (110, 200)], t, sig, max_gap_s=0.5)) == 1
    sig2 = np.concatenate([np.ones(105), -np.ones(len(t) - 105)])
    assert len(_merge_same_sign([(0, 100), (110, 200)], t, sig2, max_gap_s=0.5)) == 2


def test_left_turn_sign_convention():
    """nuScenes yaw grows counter-clockwise, so POSITIVE heading change is LEFT.

    Anchored to scene-0061, whose description says "turn left" and which measures
    +98 deg (F-011). Getting this backwards would invert every steering label.
    """
    tr = _track(yaw=lambda u: np.radians(30) * max(0.0, min(1.0, (u - 5) / 4)),
                speed=lambda u: 8.0, dur=20)
    ev = [e for e in lateral_events(tr) if e["kind"] != "stationary"]
    assert len(ev) == 1 and ev[0]["kind"] == "turn_left"
    assert ev[0]["total_heading_deg"] > 0


def test_right_turn():
    tr = _track(yaw=lambda u: -np.radians(40) * max(0.0, min(1.0, (u - 5) / 4)),
                speed=lambda u: 8.0, dur=20)
    ev = lateral_events(tr)
    assert len(ev) == 1 and ev[0]["kind"] == "turn_right"
    assert ev[0]["total_heading_deg"] < 0


def test_u_turn_needs_event_scope():
    """A 180 deg U-turn over 8 s is detectable as an EVENT.

    It is NOT detectable from a 4 s window: the largest window in the whole dataset
    measures 114 deg, below the 150 deg cut (F-012). This is why D-013 chose events.
    """
    tr = _track(yaw=lambda u: np.radians(180) * max(0.0, min(1.0, (u - 4) / 8)),
                speed=lambda u: 4.0, dur=20)
    ev = lateral_events(tr)
    assert len(ev) == 1 and ev[0]["kind"] == "u_turn", ev
    assert ev[0]["duration_s"] > 4.0, "the manoeuvre outlasts the C1 window"


def test_gentle_bend_is_not_a_turn():
    """8 deg of drift is a road bend, below MIN_TURN_DEG."""
    tr = _track(yaw=lambda u: np.radians(8) * min(1.0, u / 20), speed=lambda u: 10.0, dur=20)
    assert lateral_events(tr) == []


def test_deceleration_to_stop_is_detected():
    """Regression for F-015.

    Detecting on the raw second derivative missed this entirely: the signal alternated
    sign several times a second, so each segment was too short to pass the speed-change
    filter and a real 'arrive and stop' produced no event. Detection uses the 1.5 s
    trend acceleration instead.
    """
    tr = _track(yaw=lambda u: 0.0,
                speed=lambda u: max(0.0, 10.0 - 1.2 * max(0.0, u - 2)), dur=20)
    kinds = [e["kind"] for e in longitudinal_events(tr)]
    assert "decelerating" in kinds, kinds
    assert "stationary" in kinds, "it should also register coming to rest"


def test_acceleration_detected():
    tr = _track(yaw=lambda u: 0.0, speed=lambda u: min(12.0, 1.0 * u), dur=20)
    assert "accelerating" in [e["kind"] for e in longitudinal_events(tr)]


def test_stationary_beats_decelerating_when_they_overlap():
    tr = _track(yaw=lambda u: 0.0,
                speed=lambda u: max(0.0, 8.0 - 2.0 * u), dur=20)
    lab = maneuver_labels(15.0, [], longitudinal_events(tr))
    assert lab["longitudinal_label"] == "stationary", "a stopped car is stopped"


def test_tags_are_exhaustive_and_exclusive():
    """Regression for F-016.

    The is_* tags were once hand-written and compared against 'turning_left' while
    events emit 'turn_left', so is_turning_left was False on all 5,500 turning frames.
    Stage F would have scored the VLM at zero recall against an all-False column.
    Tags are now generated from the vocabulary tuples.
    """
    tr = _track(yaw=lambda u: np.radians(60) * max(0.0, min(1.0, (u - 5) / 4)),
                speed=lambda u: 8.0, dur=20)
    lat, lon = lateral_events(tr), longitudinal_events(tr)
    for kf in (1.0, 7.0, 12.0, 19.0):
        lab = maneuver_labels(kf, lat, lon)
        assert sum(lab[f"is_{n}"] for n in LATERAL_LABELS) == 1, f"steering @{kf}"
        assert sum(lab[f"is_{n}"] for n in LONGITUDINAL_LABELS) == 1, f"speed @{kf}"
        assert lab[f"is_{lab['lateral_label']}"] is True, "tag must match its label"
        assert lab[f"is_{lab['longitudinal_label']}"] is True


def test_axes_are_independent():
    """D-014: turning and braking must be able to co-occur."""
    tr = _track(yaw=lambda u: np.radians(60) * max(0.0, min(1.0, (u - 5) / 5)),
                speed=lambda u: max(2.0, 12.0 - 1.2 * max(0.0, u - 5)), dur=20)
    lab = maneuver_labels(8.0, lateral_events(tr), longitudinal_events(tr))
    assert lab["lateral_label"] == "turn_left"
    assert lab["longitudinal_label"] == "decelerating", lab
    # a single-label scheme would have had to throw one of these away


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C2 self-checks passed")
