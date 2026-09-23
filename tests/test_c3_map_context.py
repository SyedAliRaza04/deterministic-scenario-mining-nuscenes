"""Self-check for Step C3. Run directly: `python tests/test_c3_map_context.py`

Uses the real boston-seaport map (small, already downloaded). Includes regressions for
the three defects found while building C3 — two of which returned wrong answers silently
rather than raising.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groundtruth import (  # noqa: E402
    AHEAD_RANGE_M, CORRIDOR_HALFWIDTH_M, POLYGON_LAYERS, MapIndex, map_context,
)

DATAROOT = str(Path(__file__).resolve().parent.parent / "data" / "nuscenes")
_IDX = None


def idx():
    global _IDX
    if _IDX is None:
        _IDX = MapIndex(DATAROOT, "boston-seaport")
    return _IDX


def test_drivable_area_uses_plural_key():
    """Regression: `drivable_area` records carry `polygon_tokens`, not `polygon_token`.

    Filtering on the singular key silently produced ZERO drivable-area polygons, and
    therefore a 0% hit rate, with no error raised.
    """
    tree, recs, arr = idx().trees["drivable_area"]
    assert tree is not None and len(recs) > 0, "drivable_area must not be empty"


def test_strtree_predicate_direction():
    """Regression: shapely evaluates `input.predicate(tree_geom)`, not the reverse.

    So to find the polygon CONTAINING a point, the point is the input and the predicate
    is `within`. Using `contains` asks whether the point contains the polygon — never
    true — and returns an empty result silently. This produced a 0% hit rate on every
    layer before it was caught.
    """
    from shapely.geometry import Point
    tree, recs, arr = idx().trees["road_segment"]
    # a point guaranteed inside a known polygon
    pt = arr[0].representative_point()
    assert len(tree.query(pt, predicate="within")) >= 1, "'within' must find it"
    assert len(tree.query(pt, predicate="contains")) == 0, (
        "'contains' asks the reverse question and must find nothing — "
        "this is the trap, kept as documentation")


def test_road_block_is_excluded():
    """Regression: `road_block` is degenerate on singapore-queenstown.

    All 676 records there point to one 7,748-vertex polygon spanning the whole map, so
    containment is True everywhere and each query cost 894 ms.
    """
    assert "road_block" not in POLYGON_LAYERS


def test_corridor_points_forward_not_backward():
    c = idx()._corridor(0.0, 0.0, 0.0, length=30.0, halfwidth=4.0)
    minx, miny, maxx, maxy = c.bounds
    assert minx >= -1e-9 and maxx > 25, "corridor must extend along +x at yaw=0"
    assert abs(miny + 4.0) < 1e-9 and abs(maxy - 4.0) < 1e-9


def test_corridor_rotates_with_heading():
    c = idx()._corridor(0.0, 0.0, np.pi / 2, length=30.0, halfwidth=4.0)
    minx, miny, maxx, maxy = c.bounds
    assert maxy > 25 and abs(maxx - 4.0) < 1e-6, "at yaw=90 deg the corridor points +y"


def test_ahead_distance_cannot_exceed_corner_distance():
    """The far CORNERS of the corridor are sqrt(L^2 + W^2) away, not L.

    A naive `distance <= L` assertion fails legitimately at 30.25 m; the bound is 30.27.
    """
    bound = float(np.hypot(AHEAD_RANGE_M, CORRIDOR_HALFWIDTH_M))
    assert bound > AHEAD_RANGE_M
    tree, recs, arr = idx().trees["road_segment"]
    pt = arr[0].representative_point()
    d, rec = idx().ahead(pt.x, pt.y, 0.0, "road_segment")
    if d == d:
        assert d <= bound + 1e-6


def test_map_context_shape_and_types():
    tree, recs, arr = idx().trees["lane"]
    pt = arr[0].representative_point()
    ctx = map_context(idx(), pt.x, pt.y, 0.0)
    for k in ("on_drivable_area", "at_intersection", "on_lane", "on_walkway",
              "intersection_ahead", "ped_crossing_ahead"):
        assert isinstance(ctx[k], bool), f"{k} must be a plain bool"
    assert ctx["map_location"] == "boston-seaport"
    # binary flags must agree with their distance columns
    assert ctx["ped_crossing_ahead"] == (ctx["ped_crossing_ahead_m"] == ctx["ped_crossing_ahead_m"])


def test_point_on_lane_is_on_drivable_area():
    """Physical invariant: every lane lies inside the drivable area."""
    tree, recs, arr = idx().trees["lane"]
    for g in arr[:25]:
        p = g.representative_point()
        ctx = map_context(idx(), p.x, p.y, 0.0)
        assert ctx["on_lane"] and ctx["on_drivable_area"]


def test_walkway_is_not_drivable_in_practice():
    """Sanity anchor for the QC check that ego is never on a walkway (measured 0.00%)."""
    tree, recs, arr = idx().trees["walkway"]
    assert tree is not None and len(recs) > 0, "boston has walkways to test against"




# ---------------------------------------------------------------------------
# Topological roundabout detection (F-026). Synthetic maps with known answers:
# a donut of drivable area with an OVAL hole is, by construction, a roundabout.
# ---------------------------------------------------------------------------

class _StubIndex:
    """Minimal stand-in exposing only what map_islands/roundabout_traversals use."""

    def __init__(self, drivable):
        from shapely import STRtree
        geoms = list(drivable.geoms) if drivable.geom_type == "MultiPolygon" else [drivable]
        self.trees = {"drivable_area": (STRtree(geoms), [{}] * len(geoms),
                                        np.array(geoms, dtype=object))}


def _donut(cx=0.0, cy=0.0, rx=6.0, ry=6.0, road=9.0):
    """Drivable annulus around an island with semi-axes rx, ry (oval when rx != ry)."""
    from shapely.affinity import scale
    from shapely.geometry import Point
    outer = Point(cx, cy).buffer(max(rx, ry) + road)
    island = scale(Point(cx, cy).buffer(1.0), rx, ry)
    return outer.difference(island)


def _circle_path(cx, cy, r, deg_from, deg_to, n=40):
    a = np.radians(np.linspace(deg_from, deg_to, n))
    x, y = cx + r * np.cos(a), cy + r * np.sin(a)
    sgn = 1.0 if deg_to > deg_from else -1.0
    yaw = np.degrees(a + sgn * np.pi / 2)
    return x, y, yaw


def test_map_islands_finds_an_oval_island():
    """R26/F-026: detection must be shape-agnostic. A 2:1 oval is still an island.

    The superseded v2 detector fitted a circle and rejected exactly this case.
    """
    from src.groundtruth import map_islands
    idx = _StubIndex(_donut(rx=8.0, ry=4.0))          # deliberately oval
    isl = map_islands(idx)
    assert len(isl) == 1, f"expected one enclosed island, got {len(isl)}"
    assert 80 < isl[0].area < 120, isl[0].area


def test_circulating_an_oval_island_is_detected():
    from src.groundtruth import map_islands, roundabout_traversals
    idx = _StubIndex(_donut(rx=8.0, ry=4.0))
    x, y, yaw = _circle_path(0, 0, 11.0, 0, 200)
    hits = roundabout_traversals(idx, x, y, yaw, islands=map_islands(idx))
    assert len(hits) == 1, hits
    assert hits[0]["sweep_ratio"] > 0.8


def test_driving_past_a_refuge_is_not_a_traversal():
    """Regression: a straight pass sweeps a big angle AROUND a small island while the
    vehicle's own heading barely changes. Measured in real data: 161 deg around vs
    11.8 deg heading. Without the ratio test this is a false positive.
    """
    from src.groundtruth import map_islands, roundabout_traversals
    idx = _StubIndex(_donut(rx=3.0, ry=3.0, road=14.0))
    x = np.linspace(-25.0, 25.0, 40)
    y = np.full_like(x, 5.0)
    yaw = np.zeros_like(x)                              # heading never changes
    hits = roundabout_traversals(idx, x, y, yaw, islands=map_islands(idx))
    assert hits == [], f"passing a refuge must not count as a traversal: {hits}"


def test_opposite_turn_direction_is_rejected():
    """Regression for scene-0020: swept +123 deg around while heading turned -84 deg.

    You cannot circulate an island one way while turning the other.
    """
    from src.groundtruth import map_islands, roundabout_traversals
    idx = _StubIndex(_donut(rx=6.0, ry=6.0))
    x, y, _ = _circle_path(0, 0, 11.0, 0, 200)
    yaw = np.linspace(0.0, -120.0, len(x))              # heading turns the wrong way
    hits = roundabout_traversals(idx, x, y, yaw, islands=map_islands(idx))
    assert hits == [], f"sign disagreement must be rejected: {hits}"


def test_ring_drivable_is_reported_but_does_not_gate():
    """R27/F-026: the strongest real candidate scores only 0.76 because the map's
    drivable polygons are incomplete around small islands. It must still be returned.
    """
    from src.groundtruth import map_islands, roundabout_traversals
    from shapely.geometry import Point
    d = _donut(rx=6.0, ry=6.0).difference(Point(13.0, 0.0).buffer(6.0))  # gap in the ring
    idx = _StubIndex(d)
    x, y, yaw = _circle_path(0, 0, 9.0, 20, 200)
    hits = roundabout_traversals(idx, x, y, yaw, islands=map_islands(idx))
    assert len(hits) == 1, "a gap in the surrounding road must not disqualify it"
    assert hits[0]["ring_drivable_frac"] < 1.0, "the gap should show up in the evidence"



class _StubMap:
    """Fake lane graph: `connectivity` is all circulatory_winding needs."""
    def __init__(self, conn): self.connectivity = conn


def _square_lanes(cx=0.0, cy=0.0, r=10.0, n=24):
    """Four directed lane segments forming an anticlockwise loop about (cx, cy)."""
    quads, conn, cent = ["a", "b", "c", "d"], {}, {}
    for k, name in enumerate(quads):
        a = np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, n)
        cent[name] = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
        conn[name] = {"outgoing": [quads[(k + 1) % 4]], "incoming": [quads[(k - 1) % 4]]}
    return _StubMap(conn), cent


def test_lane_cycle_detects_a_one_way_circuit():
    """F-027: a roundabout's carriageway lets you travel a full lap."""
    from src.groundtruth import circulatory_winding
    m, cent = _square_lanes()
    wind, nlanes = circulatory_winding(m, cent, 0.0, 0.0, island_radius_m=4.0)
    assert wind > 300, f"a full loop must wind ~360 deg, got {wind}"
    assert nlanes >= 4


def test_open_ring_still_counts_as_circulatory():
    """Regression for F-028: requiring a CLOSED cycle rejected a real roundabout.

    nuScenes models the circulatory carriageway as entry->exit paths, so scene-0994's
    ring spans 359 deg but its last lane never links back to its first. Reachable
    winding along a path must still detect it.
    """
    from src.groundtruth import circulatory_winding
    m, cent = _square_lanes()
    m.connectivity["d"] = {"outgoing": [], "incoming": ["c"]}   # break the loop
    wind, _ = circulatory_winding(m, cent, 0.0, 0.0, island_radius_m=4.0)
    assert wind > 240, f"an open 3/4-plus ring must still register, got {wind}"


def test_median_divider_has_no_circuit():
    """The criterion that answers the traffic-flow objection.

    Two antiparallel carriageways either side of a median enclose it topologically, but
    no legal loop exists. Measured on real data: 4 of 5 topological candidates had a
    winding of exactly 0 deg despite 9-17 directed lane edges nearby.
    """
    from src.groundtruth import circulatory_winding
    up = np.stack([np.full(20, -6.0), np.linspace(-20, 20, 20)], axis=1)
    down = np.stack([np.full(20, 6.0), np.linspace(20, -20, 20)], axis=1)
    m = _StubMap({"up": {"outgoing": [], "incoming": []},
                  "down": {"outgoing": [], "incoming": []}})
    wind, _ = circulatory_winding(m, {"up": up, "down": down}, 0.0, 0.0, 4.0)
    # A single straight lane passing an island sweeps up to ~180 deg of bearing, which is
    # real and expected. What matters is that you cannot get ROUND: no path chains the two
    # antiparallel carriageways, so the reachable winding stays far below the 300 deg cut.
    assert wind < 300.0, f"a median divider must not reach roundabout winding, got {wind}"
    assert wind <= 185.0, f"one straight pass should cap near 180 deg, got {wind}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C3 self-checks passed")
