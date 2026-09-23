"""Self-check for Step C6. Run directly: `python tests/test_c6_schema.py`

The schema's whole job is to stop the label vocabulary drifting from what the tables
actually emit, so most of these tests are drift guards. They are also the reason the
schema is *generated* from the vocabulary constants rather than typed out (R4/F-016).

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.groundtruth as G  # noqa: E402
from src.groundtruth import (  # noqa: E402
    INTERACTION_TAGS, LABEL_IMPLICATIONS, LATERAL_LABELS, LONGITUDINAL_LABELS,
    MAP_AHEAD_TAGS, MAP_CONTAINMENT_TAGS, MIN_SCOREABLE_POSITIVES, OBJECT_TAGS,
    label_schema, label_tags,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("maneuvers", "map_context", "objects", "interactions", "lane_change")


def test_schema_is_generated_from_the_vocabulary_constants():
    """Regression for the F-016 failure mode at schema level.

    Every tag name must come from a constant. If someone renames a tag in OBJECT_TAGS
    and hand-edits the schema, or vice versa, these sets stop matching.
    """
    tags = label_tags()
    expected = (
        {f"is_{n}" for n in LATERAL_LABELS + LONGITUDINAL_LABELS}
        | set(G.EGO_EVENT_TAGS)
        | set(MAP_CONTAINMENT_TAGS) | set(MAP_AHEAD_TAGS)
        | set(OBJECT_TAGS) | set(INTERACTION_TAGS)
    )
    assert set(tags) == expected
    # 36 at the C6 freeze; C7 reopened it for lane_change under D-033
    assert len(tags) == 37, f"expected 37 scored tags, got {len(tags)}"
    assert set(label_schema()["tags"]) == expected


def test_every_tag_has_a_family_scope_and_derivation():
    for name, entry in label_schema()["tags"].items():
        assert entry["family"] in ("ego_maneuver", "map_context",
                                   "visible_objects", "interaction"), name
        assert entry["scope"], name
        assert entry["derivation"] and len(entry["derivation"]) > 10, name
        assert entry["dtype"] == "bool", name


def test_scope_separates_the_two_incompatible_object_families():
    """The reason this schema exists.

    C4 asks what CAM_FRONT can see (D-025); C5 asks what is interacting with the car,
    visible or not (D-026). `has_pedestrian` and `pedestrian_crossing_path` are not
    commensurable, and without an explicit scope field nothing says so (L-019).
    """
    tags = label_schema()["tags"]
    assert tags["has_pedestrian"]["scope"] != tags["pedestrian_crossing_path"]["scope"]
    assert all(tags[t]["scope"] == "camera_frustum_CAM_FRONT" for t in OBJECT_TAGS)
    assert all(tags[t]["scope"] == "world_360" for t in INTERACTION_TAGS)


def test_every_tuned_parameter_is_recorded_with_its_tag():
    """The EU AI Act traceability claim, made testable.

    A label whose threshold is not written down is not auditable. Any tag whose
    derivation depends on a number must carry that number.
    """
    tags = label_schema()["tags"]
    for name in ("lead_vehicle", "lead_braking", "cut_in", "pedestrian_crossing_path",
                 "pedestrian_near", "vehicle_near", *MAP_AHEAD_TAGS,
                 *[f"is_{n}" for n in LATERAL_LABELS + LONGITUDINAL_LABELS]):
        assert tags[name]["parameters"], f"{name} has a threshold but records none"
    assert tags["lead_vehicle"]["parameters"]["LEAD_RANGE_M"] == G.LEAD_RANGE_M
    assert tags["pedestrian_near"]["parameters"]["NEAR_RANGE_M"] == G.NEAR_RANGE_M
    # pure category membership genuinely has no tuned number - that is not a gap
    assert tags["traffic_cone"]["parameters"] == {}


def test_every_tag_cites_at_least_one_decision():
    for name, entry in label_schema()["tags"].items():
        assert entry["decisions"], name
        assert all(d.startswith("D-") for d in entry["decisions"]), name


def test_implications_are_structural_not_merely_observed():
    """Only relations that follow from the DEFINITIONS belong in the schema (R37).

    `is_turn_left => on_drivable_area` holds on nuScenes and is not here: it is an
    accident of the data, not a definition, and asserting it would break on a dataset
    with worse map registration. Same for `on_carpark => has_pedestrian`, which rests
    on 23 frames.
    """
    pairs = {(a, b) for a, b, _ in LABEL_IMPLICATIONS}
    assert ("traffic_cone", "construction_object") in pairs
    assert ("cut_in", "lead_vehicle") in pairs
    assert ("is_turn_left", "on_drivable_area") not in pairs
    assert ("on_carpark", "has_pedestrian") not in pairs
    for a, b, why in LABEL_IMPLICATIONS:
        assert why, f"{a}=>{b} has no stated reason"
        assert a in label_tags() and b in label_tags()


def test_subset_tags_never_exceed_their_parent_by_construction():
    """The implications, checked on synthetic rows rather than the cached table.

    Catches a broken predicate even when the parquet is stale or absent.
    """
    from src.groundtruth import CLEAR_VISIBILITY  # noqa: F401  (import guard)

    def tags_for(box):
        return {t for t, p in OBJECT_TAGS.items() if p(box)}

    cone = {"cat": "movable_object.trafficcone", "attrs": set(), "dist_m": 5.0,
            "vis_level": "v80-100", "n_lidar_pts": 9}
    fired = tags_for(cone)
    assert "traffic_cone" in fired and "construction_object" in fired
    bike = {"cat": "vehicle.bicycle", "attrs": {"cycle.with_rider"}, "dist_m": 5.0,
            "vis_level": "v80-100", "n_lidar_pts": 9}
    fired = tags_for(bike)
    assert "cyclist" in fired and "has_vehicle" in fired


def test_walkway_is_marked_unscoreable_not_silently_scored():
    """Regression for the defect this step found.

    `on_walkway` has ZERO positives in 34,149 keyframes. It is a real C3 QC invariant -
    the ego is never on a pavement - but its recall is undefined, and quoting an F1 for
    it would put a meaningless number in a results table.
    """
    p = ROOT / "outputs" / "label_schema.json"
    if not p.exists():
        print("     (skip: outputs/label_schema.json not built)")
        return
    schema = json.loads(p.read_text())
    walkway = schema["tags"]["on_walkway"]
    assert walkway["n_positive"] == 0
    assert walkway["scoreable"] is False
    assert walkway["role"] == "qc_invariant"
    scored = [t for t, e in schema["tags"].items() if e["scoreable"]]
    assert len(scored) == 35, f"expected 35 scoreable tags, got {len(scored)}"
    for t, e in schema["tags"].items():
        assert e["scoreable"] == (e["n_positive"] >= MIN_SCOREABLE_POSITIVES), t


def test_prevalence_never_appears_without_its_provenance():
    """A baseline quoted without the split it came from is not auditable."""
    bare = label_schema()
    assert bare["prevalence_measured_on"] is None
    assert "prevalence_pct" not in next(iter(bare["tags"].values()))
    withp = label_schema({"is_u_turn": 0.24}, split="trainval", n_frames=34149)
    assert withp["prevalence_measured_on"] == {"split": "trainval", "n_frames": 34149}
    assert withp["tags"]["is_u_turn"]["prevalence_pct"] == 0.24


def test_map_context_still_emits_exactly_the_declared_map_vocabulary():
    """Drift guard. map_context() writes its keys literally; the schema reads them
    from constants. If the two diverge, a map tag silently vanishes from the schema.
    """
    import inspect
    import re
    src = inspect.getsource(G.map_context)
    emitted = set(re.findall(r'"([a-z_]+)":\s', src))
    declared = set(MAP_CONTAINMENT_TAGS) | set(MAP_AHEAD_TAGS)
    assert declared <= emitted, f"declared but never emitted: {sorted(declared - emitted)}"


def test_schema_matches_the_real_tables_both_directions():
    import pandas as pd
    paths = {k: ROOT / "outputs" / f"{k}_trainval.parquet" for k in SPLITS}
    if not all(p.exists() for p in paths.values()):
        print("     (skip: Stage C trainval tables not built)")
        return
    tags = label_tags()
    emitted = set()
    for k, p in paths.items():
        df = pd.read_parquet(p)
        emitted |= {c for c in df.columns if df[c].dtype == bool}
    missing = set(tags) - emitted
    assert not missing, f"schema names tags no table emits: {sorted(missing)}"


def test_implications_hold_on_every_real_frame():
    """The assertions the schema makes, checked against all 34,149 rows."""
    import pandas as pd
    paths = {k: ROOT / "outputs" / f"{k}_trainval.parquet" for k in SPLITS}
    if not all(p.exists() for p in paths.values()):
        print("     (skip: Stage C trainval tables not built)")
        return
    tags = label_tags()
    df = None
    for k, p in paths.items():
        f = pd.read_parquet(p)
        cols = ["sample_token"] + [c for c in f.columns if c in tags]
        df = f[cols] if df is None else df.merge(f[cols], on="sample_token", validate="1:1")
    assert len(df) == 34149
    for a, b, why in LABEL_IMPLICATIONS:
        n = int((df[a] & ~df[b]).sum())
        assert n == 0, f"{a} => {b} violated on {n} frames ({why})"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C6 self-checks passed")
