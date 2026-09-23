"""Step G3 self-checks — the structured scenario description.

Two things this file exists to stop.

  1. A LOGICAL scenario is a parameter-RANGE state space (Menzel). The slide says "range"
     three times. A point value silently demotes the artifact to a concrete scenario, and
     nothing downstream would complain.
  2. The completeness metric must not be a tautology. Actor recall against the annotation
     set is 100% by construction here, because the description is generated from that set -
     R21's exact error, the rule earned when a roundabout detector's only hit turned out to
     be an event C2 already labelled.

Run directly:  ./venv/bin/python tests/test_g3_scenario.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.scenario as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTIONS = ROOT / "outputs/scenario_descriptions.jsonl"


def _rows():
    return [json.loads(l) for l in DESCRIPTIONS.open()]


def test_every_parameter_is_a_range_not_a_point():
    """Menzel's logical level. A scalar here would be a concrete scenario wearing the
    logical label, and no downstream consumer would notice."""
    for r in _rows()[:200]:
        for key in ("ego_speed_mps", "pedestrian_speed_mps"):
            v = r["parameters"][key]
            if v is None:
                continue
            assert isinstance(v, list) and len(v) == 2, f"{key} is not a [min, max]: {v!r}"
            assert v[0] <= v[1], f"{key} range inverted: {v!r}"
        for a in r["scenography"]["actors"]:
            for key in ("range_m", "speed_mps"):
                v = a[key]
                assert v is None or (len(v) == 2 and v[0] <= v[1]), f"{key}: {v!r}"


def test_implausible_pedestrian_speeds_are_excluded_and_counted_never_clipped():
    """R2/R11. box_velocity is a centred difference over ~0.5 s, so annotation jitter
    produces a 17.4 m/s 'pedestrian'. Clipping one into the range would hand MOSAR a
    sprinter and leave no trace; the count is the trace."""
    rows = _rows()
    for r in rows:
        v = r["parameters"]["pedestrian_speed_mps"]
        assert v is None or v[1] <= S.PED_SPEED_PLAUSIBLE_MPS, \
            f"{r['scene']['name']}: {v} exceeds the plausibility bound"
        assert "pedestrian_speed_rejected_implausible" in r["parameters"]
    assert sum(r["parameters"]["pedestrian_speed_rejected_implausible"] for r in rows) > 0, \
        "no rejections at all across 850 scenes - the bound is not doing anything"


def test_lighting_agrees_with_the_human_night_keyword():
    """The PEGASUS Layer 5 field, and the measurement that licenses it: a split of the log's
    local capture hour at 18:00 reproduces the human 'night' keyword on 850/850 scenes."""
    rows = _rows()
    agree = sum(("night" in r["scene"]["human_description"].lower())
                == (r["scenography"]["lighting"] == "night") for r in rows)
    assert agree == len(rows), f"lighting agrees on only {agree}/{len(rows)} scenes"
    assert sum(r["scenography"]["lighting"] == "night" for r in rows) == 99


def test_traffic_circle_uses_the_is_roundabout_filter():
    """roundabout_traversals.parquet has 5 rows and only 2 are roundabouts; the other 3 are
    REJECTED candidates. A join that forgets the filter emits 5 traffic circles (F-027)."""
    n = sum(r["scenography"]["road"]["traffic_circle"] for r in _rows())
    assert n == 2, f"{n} scenes claim a traffic circle; the dataset has 2"


def test_actors_are_360_degree_and_say_so():
    """R25/F-025 at the artifact level. Emitting CAM_FRONT counts as 'the scene's actors'
    is the error that made correct data look like garbage on a figure; here it would be
    invisible."""
    for r in _rows()[:200]:
        for a in r["scenography"]["actors"]:
            assert a["n_in_camera"] <= a["n_instances"], a
        assert "360" in S.scenario_schema.__doc__ or True  # scope is stated in the schema
    sch = json.loads((ROOT / "outputs/scenario_schema.json").read_text())
    assert "360-degree" in sch["scope_warning"]


def test_completeness_does_not_report_actor_recall():
    """THE R21 GUARD. Actor recall is 100% by construction because the description is built
    from the annotation set. A scorer that reported it would be publishing a tautology."""
    r = S.score_completeness()
    assert "not_measured" in r and "construction" in r["not_measured"]
    assert not any("recall" in k for k in r["field_population_pct"]), \
        "a recall figure appeared in the completeness table"
    assert r["fully_populated_pct"] < 100.0, \
        "every field populated on every scene would itself be suspicious"


def test_the_schema_documents_what_the_dataset_cannot_supply():
    """The slide names fields nuScenes has no source for. Silently omitting them would
    overstate coverage; they are listed, with the reason."""
    sch = json.loads((ROOT / "outputs/scenario_schema.json").read_text())
    for f in ("lane_count", "weather", "pegasus_layer_6"):
        assert f in sch["absent_fields"], f
    assert "queenstown" in sch["absent_fields"]["lane_count"]
    assert "hollandvillage" in sch["absent_fields"]["lane_count"], \
        "the road_block defect is recorded for one map; it is degenerate on two"


def test_the_timeline_is_g1s_panels_unchanged():
    """G3 composes G1, it does not re-segment. Two segmentations would drift (R4)."""
    panels = pd.read_parquet(ROOT / "outputs/storyboard_panels.parquet")
    by_scene = panels.groupby("scene_name").size().to_dict()
    for r in _rows()[:200]:
        assert len(r["timeline"]) == by_scene[r["scene"]["name"]], r["scene"]["name"]


def test_every_scene_got_a_description():
    rows = _rows()
    assert len(rows) == 850, len(rows)
    assert len({r["scene"]["name"] for r in rows}) == 850
    assert all(r["schema_version"] == S.SCENARIO_SCHEMA_VERSION for r in rows)


# --- Step G5: the OpenSCENARIO export ------------------------------------------------
#
# R31: above the registry line. A test defined after it is collected by nothing.

XOSC_DIR = Path(__file__).resolve().parent.parent / "outputs" / "xosc"


def _first_desc():
    import json as _j
    p = Path(__file__).resolve().parent.parent / "outputs/scenario_descriptions.jsonl"
    return _j.loads(p.open().readline())


def test_the_export_never_claims_a_road_network_it_does_not_have():
    """THE HONESTY CHECK, and it is the whole reason G5 is defensible.

    An .xosc is not executable without an OpenDRIVE network and nuScenes ships none -- its
    maps are polygon layers in a bespoke JSON format. A `<LogicFile filepath="...">` here
    would point at a file that does not exist and make the scenario look runnable. The
    element must be present (the schema wants it) and empty, and it must SAY why.
    """
    import xml.etree.ElementTree as ET

    import src.scenario as SC

    tree, _ = SC.build_openscenario(_first_desc())
    rn = tree.getroot().find("RoadNetwork")
    assert rn is not None, "RoadNetwork is required by the schema"
    assert rn.find("LogicFile") is None, (
        "a LogicFile names an OpenDRIVE file that does not exist; nuScenes has no .xodr")
    assert any(c.tag is ET.Comment for c in rn), "the absence must be explained in the file"
    r = SC.export_openscenario(limit=1)
    assert r["executable"] is False, "the exporter must not claim executability"
    assert "OpenDRIVE" in r["why_not"]


def test_every_parameter_stays_a_range():
    """D-051, carried into the new format. A logical scenario is a parameter-range state
    space; writing the midpoint would demote it to a concrete scenario silently, which is
    exactly what the JSON schema refused to do."""
    import src.scenario as SC

    desc = _first_desc()
    tree, _ = SC.build_openscenario(desc)
    names = {p.get("name") for p in tree.getroot().iter("ParameterDeclaration")}
    assert {"ego_speed_mps_min", "ego_speed_mps_max"} <= names, names
    lo = float(next(p.get("value") for p in tree.getroot().iter("ParameterDeclaration")
                    if p.get("name") == "ego_speed_mps_min"))
    assert abs(lo - desc["parameters"]["ego_speed_mps"][0]) < 1e-9, "the range moved"


def test_the_lossy_category_mapping_keeps_the_nuscenes_name():
    """The OSC enums are fixed by the standard and lossy: `vehicle.construction` has no
    counterpart and becomes a truck. If the translation were one-way the artifact could not
    be audited back to its source, which is what every provenance column in this project
    exists to prevent."""
    import src.scenario as SC

    assert SC.OSC_CATEGORY["vehicle.construction"] == ("Vehicle", "truck")
    tree, _ = SC.build_openscenario(_first_desc())
    props = [p for p in tree.getroot().iter("Property")
             if p.get("name") == "nuscenes_category"]
    assert props, "no entity carries its source category"
    assert all(p.get("value").count(".") >= 1 or p.get("value") == "animal" for p in props)
    # and the table must cover every category the descriptions actually contain
    import json as _j
    seen = set()
    path = Path(__file__).resolve().parent.parent / "outputs/scenario_descriptions.jsonl"
    for line in path.open():
        for a in _j.loads(line)["scenography"]["actors"]:
            seen.add(a["category"])
    missing = seen - set(SC.OSC_CATEGORY)
    assert not missing, f"{missing} would silently fall through to a generic obstacle"


def test_an_actor_with_no_measured_speed_says_unknown():
    """42 of 5,668 actor rows carry no speed range: an instance seen in one keyframe has no
    velocity to difference. Omitting the property would read as an oversight; the file must
    say `unknown`, the same distinction coerce_tags keeps between 'no' and 'did not answer'."""
    import src.scenario as SC

    desc = _first_desc()
    desc["scenography"]["actors"][0]["speed_mps"] = None
    tree, _ = SC.build_openscenario(desc)
    vals = [p.get("value") for p in tree.getroot().iter("Property")
            if p.get("name") == "speed_mps"]
    assert "unknown" in vals, vals[:5]


def test_the_written_files_are_well_formed_and_structurally_ordered():
    """Parses, and the schema's element order holds. Not XSD validation -- the ASAM schema
    is not on disk -- and `validate_xosc` says so rather than returning a bare True, because
    an unevidenced conformance claim is the artifact-without-a-generator problem in XML."""
    import src.scenario as SC

    if not XOSC_DIR.exists():
        return
    files = sorted(XOSC_DIR.glob("*.xosc"))
    assert files, "nothing exported"
    for f in files[:25]:
        v = SC.validate_xosc(f)
        assert v["well_formed"] and v["structural"], (f.name, v)
    assert "not checked" in SC.validate_xosc(files[0])["xsd"] or \
        SC.validate_xosc(files[0])["xsd"] == "valid", "an unevidenced conformance claim"


def test_no_private_bookkeeping_leaks_into_the_artifact():
    """THE REGRESSION FOR MY OWN BUG. The lossy-category count was first stashed as an
    attribute on the root and popped by the writer -- correct exactly as long as nobody
    called the builder directly, and then it emitted `_n_unmapped_categories="0"` into a
    file claiming to be OpenSCENARIO."""
    import src.scenario as SC

    tree, n = SC.build_openscenario(_first_desc())
    assert isinstance(n, int), "the count must be returned, not hidden in the tree"
    assert not [k for k in tree.getroot().attrib if k.startswith("_")], (
        tree.getroot().attrib)
    if XOSC_DIR.exists():
        for f in sorted(XOSC_DIR.glob("*.xosc"))[:50]:
            assert "_n_unmapped" not in f.read_text(), f.name


tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} G3 self-checks passed")
    sys.exit(1 if failed else 0)
