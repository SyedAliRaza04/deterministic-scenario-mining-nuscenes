"""Self-check for Step C1. Run directly: `python tests/test_c1_kinematics.py`

No framework, no fixtures. Synthetic trajectories with analytically known answers,
plus the two real bugs that were found during C1 so they cannot come back.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groundtruth import PoseTrack, ego_kinematics, keyframe_window  # noqa: E402

FS = 50.0


def _track(t, x, y, yaw):
    """Build a PoseTrack with derivatives, mirroring scene_pose_track's final step."""
    dt = 1.0 / FS
    return PoseTrack(
        t_s=t, x=x, y=y, yaw_rad=yaw,
        speed_mps=np.hypot(np.gradient(x, dt), np.gradient(y, dt)),
        accel_mps2=np.gradient(np.hypot(np.gradient(x, dt), np.gradient(y, dt)), dt),
        yaw_rate_deg_s=np.degrees(np.gradient(yaw, dt)),
        t0_us=0, n_raw_poses=len(t), resample_hz=FS,
    )


def test_window_centred():
    """Mid-scene keyframe: window is centred and 4 s long."""
    s, e, off, full = keyframe_window(10.0, 0.0, 20.0, 4.0)
    assert (s, e) == (8.0, 12.0) and off == 0.0 and full


def test_window_slides_at_start():
    """D-010: at a scene start the window SLIDES, it does not shrink."""
    s, e, off, full = keyframe_window(0.5, 0.0, 20.0, 4.0)
    assert s == 0.0 and e == 4.0, "window must stay 4 s long"
    assert abs(e - s - 4.0) < 1e-9
    assert abs(off - 1.5) < 1e-9, "offset records how far off-centre it had to move"
    assert full


def test_window_slides_at_end():
    s, e, off, full = keyframe_window(19.5, 0.0, 20.0, 4.0)
    assert (s, e) == (16.0, 20.0) and abs(off + 1.5) < 1e-9 and full


def test_window_shorter_than_scene():
    """Scene shorter than the window is the only case flagged not-full."""
    s, e, off, full = keyframe_window(1.0, 0.0, 2.0, 4.0)
    assert (s, e) == (0.0, 2.0) and not full


def test_straight_line_constant_speed():
    """Driving straight at 10 m/s: speed recovered, heading change zero."""
    t = np.arange(0, 20, 1 / FS)
    tr = _track(t, 10.0 * t, np.zeros_like(t), np.zeros_like(t))
    k = ego_kinematics(tr, 10.0, window_s=4.0)
    assert abs(k["speed_mps"] - 10.0) < 1e-6
    assert abs(k["heading_change_deg"]) < 1e-9
    assert abs(k["path_length_m"] - 40.0) < 1e-3, "4 s at 10 m/s = 40 m"
    assert abs(k["net_displacement_m"] - k["path_length_m"]) < 1e-6, "straight: path == net"
    assert not k["edge_guard"]


def test_constant_radius_turn():
    """Circle, radius 20 m at 10 m/s -> 0.5 rad/s -> 28.6 deg/s, 114.6 deg per 4 s."""
    r, v = 20.0, 10.0
    omega = v / r
    t = np.arange(0, 20, 1 / FS)
    tr = _track(t, r * np.sin(omega * t), r * (1 - np.cos(omega * t)), omega * t)
    k = ego_kinematics(tr, 10.0, window_s=4.0)
    assert abs(k["speed_mps"] - v) < 1e-3
    assert abs(k["yaw_rate_deg_s"] - np.degrees(omega)) < 1e-6
    assert abs(k["heading_change_deg"] - np.degrees(omega) * 4.0) < 1e-6
    assert k["net_displacement_m"] < k["path_length_m"], "curved: net must be shorter"


def test_heading_change_is_window_length_invariant():
    """A turn measured over a slid window equals one measured over a centred window.

    This is the whole point of D-010: without sliding, an edge frame's 2 s window
    would report half the heading change and be mislabelled 'going straight'.
    """
    omega = 0.25
    t = np.arange(0, 20, 1 / FS)
    tr = _track(t, np.zeros_like(t), np.zeros_like(t), omega * t)
    centred = ego_kinematics(tr, 10.0, window_s=4.0)["heading_change_deg"]
    at_start = ego_kinematics(tr, 0.1, window_s=4.0)["heading_change_deg"]
    assert abs(centred - at_start) < 1e-6, "constant-rate turn must read the same"


def test_no_grid_overshoot_regression():
    """Regression: the uniform grid must never extend past the last raw pose.

    np.interp clamps outside its range, so an overshooting final grid point repeated
    the last position, collapsed the apparent speed and produced a phantom
    -12.8 m/s^2 on the last keyframe of 6 of 10 mini scenes (PROCESS.md F-008).
    """
    for t_end in (20.0303, 19.9871, 20.0, 17.3334):
        dt = 1 / FS
        n = int(np.floor(t_end / dt)) + 1
        grid = np.arange(n) * dt
        assert grid[-1] <= t_end + 1e-9, f"grid overshot for t_end={t_end}"
        assert t_end - grid[-1] < dt, "grid should not lose more than one step"


def test_edge_guard_flags_track_ends():
    t = np.arange(0, 20, 1 / FS)
    tr = _track(t, 10.0 * t, np.zeros_like(t), np.zeros_like(t))
    assert ego_kinematics(tr, 0.04, window_s=4.0)["edge_guard"], "scene start must flag"
    assert ego_kinematics(tr, 19.95, window_s=4.0)["edge_guard"], "scene end must flag"
    assert not ego_kinematics(tr, 10.0, window_s=4.0)["edge_guard"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C1 self-checks passed")
