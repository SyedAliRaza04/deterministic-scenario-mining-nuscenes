"""Self-check for Step C7. Run directly: `python tests/test_c7_coverage.py`

Synthetic ego tracks with analytically known answers for the new derivations (signed
motion, lane changes), plus regressions for the defects found while building C7.

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.groundtruth as G  # noqa: E402
from src.groundtruth import (  # noqa: E402
    EGO_EVENT_TAGS, HIGHWAY_SPEED_MPS, LC_LATERAL_RATE_MAX_MPS, LC_SEP_MAX_M,
    LC_SEP_MIN_M, REVERSE_MIN_RUN, REVERSE_STEP_M, label_tags, reversing_runs,
    signed_forward_steps,
)

ROOT = Path(__file__).resolve().parent.parent


def _track(fwd_steps, yaw_deg=0.0, name="scene-test"):
    """Ego table from a list of per-keyframe forward steps (metres, signed)."""
    x = np.concatenate([[0.0], np.cumsum(fwd_steps)])
    yaw = np.radians(yaw_deg)
    return pd.DataFrame({
        "scene_name": name,
        "keyframe_index": np.arange(len(x)),
        "ego_x": x * np.cos(yaw),
        "ego_y": x * np.sin(yaw),
        "yaw_deg": yaw_deg,
        "speed_mps": np.abs(np.concatenate([[0.0], fwd_steps])) / 0.5,
    })


def test_signed_steps_recover_the_direction_speed_discards():
    """The whole reason C7 needed a new derivation.

    C1's speed_mps is a magnitude, so a car reversing at 2 m/s and one driving forward
    at 2 m/s are identical in it. Projecting displacement onto the heading restores the
    sign, which is the only way to see a parallel-parking manoeuvre at all.
    """
    fwd = _track([1.0, 1.0, 1.0])
    assert (signed_forward_steps(fwd).dropna() > 0).all()
    back = _track([-1.0, -1.0, -1.0])
    assert (signed_forward_steps(back).dropna() < 0).all()
    # and the magnitudes are identical, which is exactly the information speed loses
    assert np.allclose(fwd.speed_mps.values, back.speed_mps.values)


def test_signed_steps_are_heading_relative_not_axis_relative():
    """Driving 'backwards' along global +x is forwards if the car faces that way.

    Same class of error as F-033: a direction test in the wrong frame.
    """
    facing_west = _track([1.0, 1.0], yaw_deg=180.0)
    assert (signed_forward_steps(facing_west).dropna() > 0).all(), \
        "the car faces west and moves west - that is forward"


def test_reversing_needs_a_sustained_run():
    """A single backward step is localisation noise; only 1 such run exists in 850 scenes."""
    blip = _track([1.0, -0.5, 1.0, 1.0])
    assert reversing_runs(blip) == []
    real = _track([1.0] + [-0.5] * REVERSE_MIN_RUN + [1.0])
    runs = reversing_runs(real)
    assert len(runs) == 1
    assert runs[0]["net_m"] < 0 and runs[0]["kf1"] - runs[0]["kf0"] + 1 >= REVERSE_MIN_RUN


def test_reversing_threshold_is_outside_localisation_noise():
    """REVERSE_STEP_M must be well below the p01 of real forward steps (-0.002 m)."""
    assert REVERSE_STEP_M <= -0.05
    tiny = _track([1.0] + [REVERSE_STEP_M / 2] * 5 + [1.0])
    assert reversing_runs(tiny) == [], "sub-threshold jitter is not reversing"


def test_reversing_runs_prefers_a_precomputed_column():
    """Regression: gt_all drops ego_x/ego_y/yaw_deg but carries fwd_step_m.

    Recomputing from the raw columns raised KeyError on the merged table, and would
    have been a second copy of the same derivation (R4).
    """
    df = pd.DataFrame({"scene_name": "s", "keyframe_index": np.arange(6),
                       "fwd_step_m": [np.nan, 1.0, -0.5, -0.5, -0.5, 1.0]})
    runs = reversing_runs(df)
    assert len(runs) == 1 and runs[0]["kf0"] == 2 and runs[0]["kf1"] == 4


def test_lane_change_is_a_scored_tag_in_the_schema():
    """D-033: the user's decision to reopen the frozen schema for tag 37."""
    tags = label_tags()
    assert "lane_change" in tags and tags["lane_change"] == "ego_maneuver"
    assert len(tags) == 37, f"expected 37 tags after C7, got {len(tags)}"
    assert EGO_EVENT_TAGS == ("lane_change",)
    schema = G.label_schema()
    entry = schema["tags"]["lane_change"]
    assert entry["scope"] == "ego"
    assert entry["parameters"], "a detector with six thresholds must record them"
    assert entry["decisions"] == ["D-033"]
    assert schema["schema_version"] == "C6.3", "reopening the schema must bump it (C6.2 -> C6.3, D-041)"
    assert entry["source_table"].startswith("outputs/lane_change_"), \
        "lane_change lives in its own table, not maneuvers"


def test_separation_bounds_bracket_one_lane_width():
    """A 'lane change' with 0.3 m of separation is a fork; with 12 m it is a map artifact."""
    assert LC_SEP_MIN_M > 1.0 and LC_SEP_MAX_M < 8.0
    assert LC_SEP_MIN_M < 3.5 < LC_SEP_MAX_M, "a nuScenes lane is ~3.5 m wide"


def test_lateral_rate_bound_is_physically_plausible():
    """R2. This one gate removed the sep>5.8 m artifacts: 7 m in 1 s is not a lane change."""
    assert 0.5 <= LC_LATERAL_RATE_MAX_MPS <= 3.0
    # 3.5 m at the bound takes at least this long, which must exceed one keyframe
    assert 3.5 / LC_LATERAL_RATE_MAX_MPS >= 0.5


def test_no_highway_speed_exists_in_the_dataset():
    """F-042, the decisive evidence for L-001.

    A highway merge needs highway speed. If this ever fails, nuScenes changed.
    """
    p = ROOT / "outputs" / "gt_all.parquet"
    if not p.exists():
        print("     (skip: outputs/gt_all.parquet not built)")
        return
    gt = pd.read_parquet(p)
    assert (gt.speed_mps < HIGHWAY_SPEED_MPS).all(), \
        f"something reaches {3.6*gt.speed_mps.max():.1f} km/h"
    assert 3.6 * gt.speed_mps.max() < 70.0


def test_gt_all_carries_every_schema_tag_exactly_once():
    p = ROOT / "outputs" / "gt_all.parquet"
    if not p.exists():
        print("     (skip: outputs/gt_all.parquet not built)")
        return
    gt = pd.read_parquet(p)
    assert len(gt) == 34149 and gt.sample_token.is_unique
    missing = set(label_tags()) - set(gt.columns)
    assert not missing, f"gt_all is missing {sorted(missing)}"
    assert len(gt.columns) == len(set(gt.columns)), "a join duplicated a column"
    for t in label_tags():
        assert gt[t].dtype == bool, t


def test_lane_change_events_agree_with_the_frame_flags():
    """Structural invariant (R3): flagged frames are exactly the frames inside events."""
    ep = ROOT / "outputs" / "lane_change_events_trainval.parquet"
    tp = ROOT / "outputs" / "lane_change_trainval.parquet"
    if not (ep.exists() and tp.exists()):
        print("     (skip: lane-change tables not built)")
        return
    ev, tab = pd.read_parquet(ep), pd.read_parquet(tp)
    inside = set()
    for r in ev.itertuples():
        assert r.kf1 >= r.kf0
        assert LC_SEP_MIN_M <= r.sep_m <= LC_SEP_MAX_M
        for k in range(r.kf0, r.kf1 + 1):
            inside.add((r.scene_name, k))
    flagged = {(r.scene_name, r.keyframe_index) for r in tab.itertuples() if r.lane_change}
    assert flagged == inside, f"{len(flagged)} flagged vs {len(inside)} inside events"
    per_scene = ev.groupby("scene_name").size()
    assert per_scene.max() <= 5, "more than 5 lane changes in a 20 s scene is not credible"


def test_lane_change_windows_are_long_enough_to_be_possible():
    """Regression for the padding fix.

    Detector A fires on a single token flip, but crossing sep metres takes at least
    sep / LC_LATERAL_RATE_MAX_MPS seconds. Unpadded, the mid-change frames either side
    were labelled False - a ground-truth false negative at every event edge.
    """
    ep = ROOT / "outputs" / "lane_change_events_trainval.parquet"
    if not ep.exists():
        print("     (skip: lane-change events not built)")
        return
    ev = pd.read_parquet(ep)
    # intervals, not frames - the same denominator the detector's own gate uses
    dur_s = (ev.kf1 - ev.kf0) * 0.5
    assert (dur_s > 0).all(), "an event spanning no time is not a crossing"
    rate = ev.sep_m / dur_s
    full = ev[~ev.window_clamped]
    assert (rate[full.index] <= LC_LATERAL_RATE_MAX_MPS + 1e-9).all(), \
        f"implied lateral speed up to {rate[full.index].max():.2f} m/s"
    # a clamped window is only legitimate at a scene boundary (D-012 treatment)
    kin = pd.read_parquet(ROOT / "outputs" / "ego_kinematics_trainval.parquet")
    last = kin.groupby("scene_name").keyframe_index.max()
    for r in ev[ev.window_clamped].itertuples():
        assert r.kf0 == 0 or r.kf1 >= last[r.scene_name], \
            f"{r.scene_name} kf{r.kf0}-{r.kf1} is clamped but not at a scene edge"


def test_coverage_report_states_the_brief_answer():
    p = ROOT / "outputs" / "coverage_report.json"
    if not p.exists():
        print("     (skip: coverage_report.json not built)")
        return
    rep = json.loads(p.read_text())
    bm = rep["brief_maneuvers"]
    assert bm["highway_merge"]["present"] is False
    assert bm["highway_merge"]["evidence"]["keyframes_at_or_above_70_kmh"] == 0
    assert bm["parallel_parking"]["present"] is False
    assert bm["roundabout"]["n_traversals"] == 2
    # every claim must carry its evidence, not just a verdict
    for name, entry in bm.items():
        assert "conclusion" in entry and entry["conclusion"], name
        if entry["present"] is False:
            assert entry["evidence"], f"{name} claims absence with no evidence"


# ---------------------------------------------------------------------------
# C7 revisited (F-050): the frozen detector must stay frozen, and the measured
# alternative must be reachable without touching it.
# ---------------------------------------------------------------------------

class _StubCL:
    """CentrelineIndex with hand-placed straight lines, for the geometry checks."""

    def __init__(self, lines):
        self.lines = [np.asarray(L, dtype=float) for L in lines]
        self.tokens = [f"L{i}" for i in range(len(lines))]

    signed_offset = G.CentrelineIndex.signed_offset
    perp_offset = G.CentrelineIndex.perp_offset
    min_dist = G.CentrelineIndex.min_dist


def _two_parallel_lanes(sep=3.5, length=20.0):
    """Two straight centrelines along +x, `sep` apart, both ending at x = length."""
    xs = np.arange(0.0, length + 1e-9, 1.0)
    return _StubCL([np.stack([xs, np.zeros_like(xs)], 1),
                    np.stack([xs, np.full_like(xs, sep)], 1)])


def test_default_lane_change_parameters_are_the_adopted_ones():
    """D-041 adopted both halves of the F-050 fix. Schema C6.2 -> C6.3.

    If this fails, `build_lane_change_table()` no longer reproduces the shipped
    lane_change_*.parquet and gt_all.parquet has silently drifted.

    Both knobs are asserted because NEITHER WORKS ALONE: lateral sep on its own
    finds 14/24 scenes and the gap look-back on its own finds 14/24, the same as
    the C6.2 detector. Together they find 17/24. A future edit that reverts one
    and leaves the other would look harmless and would silently undo the fix.
    """
    assert G.LC_TOKEN_GAP_KF == 4, "adopted detector looks back across unmapped runs"
    assert G.LC_SEP_LATERAL is True, "adopted detector differences two SIGNED offsets"
    import inspect
    sig = inspect.signature(G.lane_change_events).parameters
    assert sig["token_gap_kf"].default == G.LC_TOKEN_GAP_KF
    assert sig["sep_lateral"].default == G.LC_SEP_LATERAL


def test_the_c6_2_operating_point_is_still_reachable():
    """F-050's comparison table must regenerate after adoption (R30).

    The superseded detector is evidence, not dead code: the thesis reports both
    operating points, so `token_gap_kf=0, sep_lateral=False` has to keep working.
    """
    import inspect
    sig = inspect.signature(G.lane_change_events).parameters
    assert "token_gap_kf" in sig and "sep_lateral" in sig, \
        "the C6.2 behaviour must stay reachable by keyword, or F-050 cannot be reproduced"
    cl = _two_parallel_lanes()
    # On CONSECUTIVE frames the two sep measures agree - that is why the defect hid
    # for six detector revisions. Assert the agreement rather than assuming it.
    a = abs(cl.signed_offset(0, 10.0, 0.0) - cl.signed_offset(1, 10.0, 0.0))
    b = cl.min_dist(0, 10.0, 0.0) + cl.min_dist(1, 10.0, 0.0)
    assert abs(a - b) < 0.1, (a, b)


def test_signed_offset_keeps_the_side_perp_offset_throws_away():
    """The sign is the whole point: two offsets can only be DIFFERENCED if they are signed."""
    cl = _two_parallel_lanes()
    left = cl.signed_offset(0, 5.0, +2.0)
    right = cl.signed_offset(0, 5.0, -2.0)
    assert left > 0 > right, "positive must mean LEFT of the directed centreline"
    assert np.isclose(abs(left), 2.0) and np.isclose(abs(right), 2.0)
    # and perp_offset must stay exactly the old quantity (R4: one derivation, one place)
    for x, y in [(5.0, 2.0), (5.0, -2.0), (0.0, 0.9), (18.0, -3.1)]:
        assert np.isclose(cl.perp_offset(0, x, y), abs(cl.signed_offset(0, x, y)))


def test_separation_measure_regression_along_track_contamination():
    """Regression for F-050.

    `min_dist(old) + min_dist(new)` measures distance to the nearest POINT of each
    polyline, so once the ego has driven past the old lane's last vertex the along-track
    travel is counted as separation. Measured over 608 real flip pairs, the median goes
    3.65 m at one keyframe to 18.1 m at four, while the true lateral gap is 0.03-0.11 m -
    which is why 99.6% of cross-gap pairs were rejected by `sep <= LC_SEP_MAX_M`.
    """
    cl = _two_parallel_lanes(sep=3.5, length=20.0)
    x, y = 32.0, 3.5                       # 12 m past the end of both lanes, in lane 1
    mindist = cl.min_dist(0, x, y) + cl.min_dist(1, x, y)
    lateral = abs(cl.signed_offset(0, x, y) - cl.signed_offset(1, x, y))
    assert np.isclose(lateral, 3.5), f"the lanes are 3.5 m apart wherever you stand, got {lateral}"
    assert mindist > G.LC_SEP_MAX_M, "the point of the regression: the old measure blows the gate"
    assert mindist / lateral > 3, f"contamination factor only {mindist / lateral:.1f}"


def test_token_gap_lookback_never_steps_over_a_mapped_frame():
    """The look-back bridges frames with NO lane polygon; it must not skip a mapped one.

    Skipping a mapped frame would compare two lanes several seconds apart and turn
    ordinary progression along a road into a fake crossing.
    """
    toks = ["a", None, None, "b", "c", None, "d"]

    def pairs(gap):
        out = []
        for i in range(1, len(toks)):
            if not isinstance(toks[i], str):
                continue
            for back in range(1, gap + 2):
                if i - back < 0:
                    break
                if isinstance(toks[i - back], str):
                    out.append((i - back, i))
                    break
        return out

    assert pairs(0) == [(3, 4)], "gap=0 must be consecutive-only, as frozen"
    assert pairs(2) == [(0, 3), (3, 4), (4, 6)]
    for a, b in pairs(4):
        assert all(toks[k] is None for k in range(a + 1, b)), \
            "only unmapped frames may be bridged"


def test_lane_change_comparison_reports_both_operating_points():
    """R30: the L-022 recall number must be recomputable, not quoted from a transcript."""
    p = ROOT / "outputs" / "lane_change_variants.json"
    if not p.exists():
        print("     (skip: outputs/lane_change_variants.json not built)")
        return
    rep = json.loads(p.read_text())
    frozen = rep["variants"]["frozen (D-033)"]
    assert frozen["parameters"] == {"token_gap_kf": 0, "sep_lateral": False}
    adopted = rep["variants"]["sep_lateral + gap"]
    assert adopted["parameters"] == {"token_gap_kf": 4, "sep_lateral": True}
    ep = ROOT / "outputs" / "lane_change_events_trainval.parquet"
    if ep.exists():
        # After D-041 the SHIPPED table is the adopted variant, not the frozen one.
        assert adopted["events"] == len(pd.read_parquet(ep)), \
            "the adopted row must match the shipped event table, or one of them is stale"
    for name, v in rep["variants"].items():
        assert v["parameters"] and v["recall_events"], name
        assert v["description_hit_rate_pct"] is not None, name
    # the reference is weak in a specific, documented way (L-007) and must say so
    assert set(rep["reference"]["excluded_non_ego"]) == set(G.LC_DESCRIPTION_NON_EGO)
    assert len(rep["reference"]["scenes"]) > len(rep["reference"]["ego_only"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C7 self-checks passed")
