"""Self-check for Step C4. Run directly: `python tests/test_c4_objects.py`

Synthetic box records with analytically known answers for the tag vocabulary, plus
regressions for the defects found while building C4.

The predicates in OBJECT_TAGS operate on plain dicts, so most of this needs no devkit
and no dataset. The two checks that DO need real data are guarded and skip cleanly.

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.groundtruth as G  # noqa: E402
from src.groundtruth import (  # noqa: E402
    CLEAR_VISIBILITY, MID_RANGE_M, NEAR_RANGE_M, OBJECT_TAGS,
)

ROOT = Path(__file__).resolve().parent.parent


def _box(cat, dist=20.0, attrs=(), vis="v80-100", npts=10):
    return {"cat": cat, "attrs": set(attrs), "dist_m": dist,
            "vis_level": vis, "n_lidar_pts": npts}


def _tags(boxes):
    """Which tags fire on this box list."""
    return {t for t, p in OBJECT_TAGS.items() if any(p(b) for b in boxes)}


def test_parked_bicycle_is_not_a_cyclist():
    """The distinction the original pipeline collapsed.

    nuScenes annotates the bicycle, and the `cycle.with_rider` attribute says whether
    anyone is on it. A bicycle in a rack is street furniture; a cyclist is a road user
    the ego has to yield to. Conflating them is a 6.4% vs 8.1% error across the dataset,
    and it is wrong in both directions.
    """
    racked = _box("vehicle.bicycle", attrs=["cycle.without_rider"])
    ridden = _box("vehicle.bicycle", attrs=["cycle.with_rider"])
    assert _tags([racked]) & {"cyclist", "parked_bicycle"} == {"parked_bicycle"}
    assert _tags([ridden]) & {"cyclist", "parked_bicycle"} == {"cyclist"}
    # a motorcycle with a rider is a cyclist too - same attribute, same rule
    assert "cyclist" in _tags([_box("vehicle.motorcycle", attrs=["cycle.with_rider"])])


def test_two_wheelers_count_as_vehicles_but_carry_no_motion_attribute():
    """Regression for a real confusion seen on scene-0916.

    has_vehicle=10 with parked+moving+stopped=3 looks like a bug. It is not: bicycles
    and motorcycles take cycle.* attributes, never vehicle.*, so 7 of those 10 are
    two-wheelers. The three motion tags are a SUBSET of has_vehicle, never a partition.
    """
    bikes = [_box("vehicle.bicycle", attrs=["cycle.without_rider"]) for _ in range(7)]
    cars = [_box("vehicle.car", attrs=["vehicle.parked"]) for _ in range(3)]
    fired = _tags(bikes + cars)
    assert "has_vehicle" in fired and "parked_vehicle" in fired
    assert "moving_vehicle" not in fired and "stopped_vehicle" not in fired
    n_motion = sum(1 for b in bikes + cars
                   if b["attrs"] & {"vehicle.parked", "vehicle.moving", "vehicle.stopped"})
    assert n_motion == 3 < len(bikes + cars), "motion tags must not have to cover all vehicles"


def test_presence_tags_have_no_range_limit():
    """D-022. A car at 60 m is plainly visible, so has_vehicle is True.

    Range-gating presence would score a VLM as a false positive for correctly naming
    something it can see - the same mistake D-017 avoided for map context.
    """
    far = _box("vehicle.car", dist=60.0, attrs=["vehicle.moving"])
    fired = _tags([far])
    assert "has_vehicle" in fired
    assert "vehicle_near" not in fired, "but it is NOT near"


def test_near_tags_use_the_stated_radius():
    inside = _box("human.pedestrian.adult", dist=NEAR_RANGE_M - 0.1)
    outside = _box("human.pedestrian.adult", dist=NEAR_RANGE_M + 0.1)
    assert "pedestrian_near" in _tags([inside])
    assert "pedestrian_near" not in _tags([outside])
    assert "has_pedestrian" in _tags([outside]), "still present, just not near"


def test_occlusion_never_filters_a_tag():
    """D-023 / R27. A barely-visible pedestrian is still a pedestrian.

    Two gates in this project have already rejected true positives (F-026 circle-ring,
    F-028 closed cycle). Visibility is recorded as a count, never as a filter - and
    nuScenes' visibility is summed over all SIX cameras, so gating on it would import
    evidence from cameras the VLM is never shown (L-016).
    """
    hidden = _box("human.pedestrian.adult", dist=10.0, vis="v0-40", npts=0)
    assert "has_pedestrian" in _tags([hidden]) and "pedestrian_near" in _tags([hidden])


def test_construction_object_is_exactly_cone_barrier_debris():
    """And a shopping trolley is NOT a construction object.

    movable_object.pushable_pullable is 2.65% of visible boxes; it was drawn the same
    blue as a traffic cone in the first C4 figure, showing a tag the table never had.
    """
    for cat in ("movable_object.trafficcone", "movable_object.barrier",
                "movable_object.debris"):
        assert "construction_object" in _tags([_box(cat)]), cat
    trolley = _tags([_box("movable_object.pushable_pullable")])
    assert trolley == set(), f"trolley must carry NO tag, got {trolley}"
    for cat in ("static_object.bicycle_rack", "animal"):
        assert _tags([_box(cat)]) == set(), cat


def test_cone_and_barrier_are_subsets_of_construction_object():
    boxes = [_box("movable_object.trafficcone"), _box("movable_object.barrier")]
    fired = _tags(boxes)
    assert {"traffic_cone", "barrier", "construction_object"} <= fired


def test_large_vehicle_membership():
    for cat in ("vehicle.truck", "vehicle.bus.rigid", "vehicle.bus.bendy",
                "vehicle.trailer", "vehicle.construction"):
        assert "large_vehicle" in _tags([_box(cat)]), cat
    for cat in ("vehicle.car", "vehicle.motorcycle", "vehicle.emergency.police"):
        assert "large_vehicle" not in _tags([_box(cat)]), cat


def test_every_tag_emits_three_columns_generated_from_the_vocabulary():
    """Regression for the F-016 failure mode, in its C4 form.

    The bool, the count and the clear-count must all come from OBJECT_TAGS, so a tag
    renamed in one place cannot leave a permanently-False column behind.
    """
    boxes = [_box("human.pedestrian.adult", dist=5.0, vis="v80-100"),
             _box("human.pedestrian.adult", dist=5.0, vis="v0-40"),
             _box("vehicle.car", dist=50.0, attrs=["vehicle.moving"])]
    row = {}   # mirrors the body of visible_objects(), minus the devkit lookup
    for tag, pred in OBJECT_TAGS.items():
        hits = [b for b in boxes if pred(b)]
        row[tag] = len(hits) > 0
        row[f"n_{tag}"] = len(hits)
        row[f"n_{tag}_clear"] = sum(b["vis_level"] == CLEAR_VISIBILITY for b in hits)
    for tag in OBJECT_TAGS:
        assert row[tag] == (row[f"n_{tag}"] > 0), tag
        assert row[f"n_{tag}_clear"] <= row[f"n_{tag}"], tag
    assert row["n_has_pedestrian"] == 2 and row["n_has_pedestrian_clear"] == 1
    assert row["n_pedestrian_near"] == 2, "both are inside 15 m"
    assert row["n_vehicle_near"] == 0, "the car is at 50 m"


def test_range_bands_partition_the_boxes():
    """near + mid + far must equal the total, with no box counted twice."""
    dists = [1.0, NEAR_RANGE_M, NEAR_RANGE_M + 0.01, MID_RANGE_M, MID_RANGE_M + 0.01, 200.0]
    boxes = [_box("vehicle.car", dist=d) for d in dists]
    near = sum(b["dist_m"] <= NEAR_RANGE_M for b in boxes)
    mid = sum(NEAR_RANGE_M < b["dist_m"] <= MID_RANGE_M for b in boxes)
    far = sum(b["dist_m"] > MID_RANGE_M for b in boxes)
    assert (near, mid, far) == (2, 2, 2)
    assert near + mid + far == len(boxes)


def test_empty_frame_produces_no_tags():
    """The whole point of C4: nothing ahead means nothing tagged.

    Measured on the real data, 4.24% of keyframes have zero boxes in the CAM_FRONT
    frustum while the 360 deg annotation set is non-empty - scene-0916 has one with 37
    annotations, all of them behind or beside the car. Under the old approach every one
    of those frames read has_vehicle = True.
    """
    assert _tags([]) == set()


def test_table_matches_the_predicates_on_real_data():
    """Guarded end-to-end check against the cached mini table."""
    import pandas as pd
    p = ROOT / "outputs" / "objects_mini.parquet"
    if not p.exists():
        print("     (skip: outputs/objects_mini.parquet not built)")
        return
    df = pd.read_parquet(p)
    assert len(df) == 404, f"mini has 404 keyframes, table has {len(df)}"
    assert df.sample_token.is_unique
    for tag in OBJECT_TAGS:
        assert ((df[f"n_{tag}"] > 0) == df[tag]).all(), tag
        assert (df[f"n_{tag}_clear"] <= df[f"n_{tag}"]).all(), tag
        assert (df[f"n_{tag}"] <= df.n_boxes_visible).all(), tag
    assert (df.n_boxes_visible <= df.n_annotations_360).all(), \
        "more boxes visible than exist - frustum filter is not a filter"
    assert (df.n_boxes_near + df.n_boxes_mid + df.n_boxes_far == df.n_boxes_visible).all()
    assert (df.n_pedestrian_near <= df.n_has_pedestrian).all()
    assert (df.n_traffic_cone + df.n_barrier <= df.n_construction_object).all()


def test_frustum_filter_actually_discards_on_real_data():
    """Regression against the defect C4 exists to fix.

    If someone swaps BoxVisibility.ANY for BoxVisibility.NONE, every assertion above
    still passes - the counts just silently become the 360 deg set again. This is the
    check that would fail.
    """
    import pandas as pd
    p = ROOT / "outputs" / "objects_mini.parquet"
    if not p.exists():
        print("     (skip: outputs/objects_mini.parquet not built)")
        return
    df = pd.read_parquet(p)
    ratio = df.n_annotations_360.sum() / df.n_boxes_visible.sum()
    assert ratio > 2.0, f"expected a large over-count factor, got {ratio:.2f}x"
    assert (df.n_boxes_visible == 0).any(), \
        "no frame is empty ahead - the frustum filter is probably not applied"


def test_parameters_are_physically_stated():
    assert 0 < NEAR_RANGE_M < MID_RANGE_M
    assert MID_RANGE_M == G.AHEAD_RANGE_M, \
        "the mid-range band should match C3's ahead corridor (D-020) or say why not"
    assert CLEAR_VISIBILITY == "v80-100"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C4 self-checks passed")
