"""Stage D self-checks — the two BEV arms.

The load-bearing check here is `test_description_never_states_a_label`. RQ3a compares
a rendered BEV against a symbolic one at FIXED ground truth; if the text arm names a
tag, it hands over the answer and the comparison measures nothing. The raster cannot
leak that way, so nothing but this test protects the design.

The second theme is that both arms must render THE SAME CONTENT. If one could see
something the other could not, the experiment would measure content rather than form.

Run:  ./venv/bin/python tests/test_bev.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.bev as B  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def _agent(cat="vehicle.car", x=10.0, y=0.0, speed=5.0, in_cam=True, L=4.5, W=1.9):
    return {"category": cat, "style": B._style_of(cat), "x_fwd_m": x, "y_left_m": y,
            "range_m": math.hypot(x, y),
            "bearing_deg": math.degrees(math.atan2(y, x)),
            "yaw_rel_deg": 0.0, "length_m": L, "width_m": W, "speed_mps": speed,
            "v_fwd_mps": speed, "v_left_mps": 0.0, "in_camera": in_cam}


def _content(agents, radius=B.BEV_RADIUS_M, layers=None):
    return {"sample_token": "t0", "scene_name": "scene-0000", "radius_m": radius,
            "agents": agents, "map_layers": layers or {"drivable_area": []},
            "n_in_camera": sum(a["in_camera"] for a in agents)}


# --- the design-critical one -----------------------------------------------------------

@check("test_description_never_states_a_label")
def _():
    """D-039/D2: the symbolic arm may state GEOMETRY, never a derived tag.

    Checked against the real frozen schema, so a tag added later is covered automatically
    rather than needing this list updated by hand (R4).
    """
    schema = json.loads((ROOT / "outputs" / "label_schema.json").read_text())
    agents = [_agent("vehicle.car", 8, 2.0), _agent("human.pedestrian.adult", 12, -3),
              _agent("movable_object.trafficcone", 5, 1, speed=0.0),
              _agent("vehicle.bicycle", 20, 6)]
    text = B.describe_bev(_content(agents, layers={
        "drivable_area": [], "lane": [], "ped_crossing": [], "stop_line": []})).lower()
    leaked = [t for t in schema["tags"] if t.lower() in text]
    assert not leaked, (
        f"describe_bev states the tag name(s) {leaked} — that hands the answer to the "
        "symbolic arm and voids the RQ3a comparison")


@check("test_description_does_not_leak_via_the_map_layer_names")
def _():
    """`ped_crossing` is both a map layer and a tag stem. The layer must be reworded.

    The raster shows a crossing as a coloured polygon, so the INFORMATION is legitimately
    in both arms — what must not appear is the tag's own identifier.
    """
    import numpy as _np

    square = [(_np.array([[8.0, -4.0], [14.0, -4.0], [14.0, 4.0], [8.0, 4.0], [8.0, -4.0]]),
               [])]
    text = B.describe_bev(_content([], layers={"ped_crossing": square,
                                               "stop_line": square}))
    assert "ped_crossing" not in text and "stop_line" not in text, text
    assert "ped crossing" in text and "stop line" in text, \
        "the layer must still be NAMED, or the two arms stop showing the same content"
    # A layer key carrying no drawable ring is NOT named, because the raster draws nothing
    # for it either. Naming it would make the text arm the more informative of the two,
    # which is the same asymmetry in the opposite direction (D-039).
    empty = B.describe_bev(_content([], layers={"ped_crossing": [], "stop_line": []}))
    assert "ped crossing" not in empty and "no mapped road surface" in empty, empty


# --- both arms, same content -----------------------------------------------------------

@check("test_both_arms_are_driven_by_one_content_dict")
def _():
    """Structural guarantee: neither renderer may take its own data source."""
    import inspect
    for fn in (B.render_bev, B.describe_bev):
        params = list(inspect.signature(fn).parameters)
        assert params[0] == "content", f"{fn.__name__} takes {params[0]}, not content"


@check("test_every_agent_in_the_content_reaches_the_description")
def _():
    agents = [_agent("vehicle.car", 10, 0), _agent("human.pedestrian.adult", 20, 5),
              _agent("movable_object.barrier", 3, -2, speed=0.0)]
    text = B.describe_bev(_content(agents))
    assert "OBJECTS (3" in text, text
    for a in agents:
        assert a["category"] in text, f"{a['category']} missing from the description"


@check("test_the_camera_field_of_view_is_stated_identically_in_both_arms")
def _():
    """The raster draws the wedge; the text states the same bearings. One constant."""
    text = B.describe_bev(_content([]))
    half = B.CAM_FRONT_HFOV_DEG / 2
    assert f"-{half:.0f} to +{half:.0f} degrees" in text, text


# --- geometry --------------------------------------------------------------------------

@check("test_bearing_sign_is_positive_left")
def _():
    """D-027's frame: x forward, y left. A car on the left must read positive."""
    left = B.describe_bev(_content([_agent("vehicle.car", 10, 10)]))
    right = B.describe_bev(_content([_agent("vehicle.car", 10, -10)]))
    assert "bearing +45" in left, left
    assert "bearing -45" in right, right


@check("test_an_agent_directly_behind_reads_as_180_degrees_and_outside_the_camera")
def _():
    text = B.describe_bev(_content([_agent("vehicle.car", -20, 0, in_cam=False)]))
    assert "bearing +180" in text or "bearing -180" in text, text
    assert "outside camera view" in text, text


@check("test_stationary_agents_are_described_as_not_moving")
def _():
    """MOVING_MPS reuses C2's STATIONARY_MPS so 'moving' means one thing project-wide."""
    from src import groundtruth as G
    assert B.MOVING_MPS == G.STATIONARY_MPS, "the two definitions of moving have drifted"
    text = B.describe_bev(_content([_agent("vehicle.car", 10, 0, speed=0.1)]))
    assert "not moving" in text, text


@check("test_a_nan_velocity_is_not_reported_as_stationary")
def _():
    """0.2% of annotations appear in one keyframe only and have no velocity.

    Reporting those as 'not moving' would write a false fact into the scene description,
    which is the C5 lesson (velocities are propagated as NaN, never zero-filled).
    """
    a = _agent("vehicle.car", 10, 0, speed=float("nan"))
    text = B.describe_bev(_content([a]))
    assert "speed unknown" in text, text
    assert "not moving" not in text, text


@check("test_small_classes_are_drawn_as_markers_not_inflated_boxes")
def _():
    """At 0.26 m/px a 0.7 m pedestrian is 2.7 px. A to-scale box is invisible; an
    inflated box would misstate its size. The glyph changes instead, and the raster
    must not silently redraw a pedestrian at car scale."""
    src = (ROOT / "src" / "bev.py").read_text()
    assert 'marker="o"' in src, "the fixed-size marker path for small classes is gone"
    for style in ("pedestrian", "cyclist", "traffic_cone", "barrier"):
        assert f'"{style}"' in src


@check("test_every_legend_colour_is_used_and_every_used_colour_is_in_the_legend")
def _():
    """PROMPT_V3_BEV must describe exactly BEV_COLORS. A style drawn but not listed
    would make the prompt silently untrue (R36)."""
    styles = {k for _, k in B._CATEGORY_STYLE} | {"other"}
    missing = styles - set(B.BEV_COLORS)
    assert not missing, f"drawn styles absent from the legend: {missing}"
    for layer in B._MAP_LAYERS:
        assert layer in B.BEV_COLORS, f"map layer {layer} has no legend colour"


@check("test_road_block_is_not_drawn")
def _():
    """D-019 excludes road_block from the pipeline, so no label is ever derived from it.
    Drawing it would show the model a layer the ground truth knows nothing about (R36)."""
    assert "road_block" not in B._MAP_LAYERS
    assert "road_block" not in B.BEV_COLORS


@check("test_radius_default_matches_the_measured_coverage_decision")
def _():
    """D-042. 50 m would crop 19.7% of the boxes the ground truth counts, handicapping
    the BEV arm by construction — the F-029 failure class."""
    assert B.BEV_RADIUS_M == 100.0, "the measured 99.7%-coverage radius has changed"
    assert B.BEV_PIXELS == 768, "pixels must scale with radius to hold legibility"


@check("test_render_writes_a_png_of_the_expected_size")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = B.render_bev(_content([_agent()]), Path(d) / "a.png", pixels=256)
        assert p.exists() and p.stat().st_size > 0
        from PIL import Image
        assert Image.open(p).size == (256, 256), Image.open(p).size


@check("test_description_is_deterministic")
def _():
    """The same content must yield byte-identical text, or results cannot be reproduced."""
    c = _content([_agent("vehicle.car", 9, 1), _agent("human.pedestrian.adult", 4, -2)])
    assert B.describe_bev(c) == B.describe_bev(c)


@check("test_unknown_categories_fall_back_to_other_rather_than_a_wrong_tag")
def _():
    """R36/L-018: 2.9% of visible objects carry no C4 tag (trolleys, bicycle racks,
    animals). They must read as unlabelled, not be silently coloured as something else."""
    assert B._style_of("movable_object.pushable_pullable") == "other"
    assert B._style_of("animal") == "other"
    assert B._style_of("static_object.bicycle_rack") == "other"
    # ...while the real classes still resolve
    assert B._style_of("vehicle.car") == "vehicle"
    assert B._style_of("vehicle.bus.rigid") == "large_vehicle"
    assert B._style_of("human.pedestrian.child") == "pedestrian"
    assert B._style_of("vehicle.bicycle") == "cyclist"


@check("test_invalid_map_polygons_are_repaired_not_fatal")
def _():
    """Regression 2026-09-06: the D3 batch died at frame 144 of 1,800.

    nuScenes ships 7 invalid (self-intersecting) map polygons out of 11,496 across the
    four locations - 2 road_segment on singapore-onenorth, 5 on boston-seaport.
    `.intersection()` raises GEOSException on those rather than returning something
    wrong, so the failure is loud but fatal.
    """
    from shapely.errors import GEOSException
    from shapely.geometry import Polygon, box as shapely_box
    from src import groundtruth as G

    # A bowtie: the canonical self-intersection, invalid exactly as the map polygons are.
    bowtie = Polygon([(0, 0), (4, 4), (4, 0), (0, 4)])
    assert not bowtie.is_valid
    patch = shapely_box(-1, -1, 5, 5)
    try:
        bowtie.intersection(patch)
    except GEOSException:
        pass          # what the real polygons do
    out = G.clip_to_patch(bowtie, patch)
    assert out.area > 0, "clip_to_patch must return usable geometry, not an empty one"


@check("test_no_caller_clips_map_geometry_without_the_repair")
def _():
    """Fix the shared function, not the path the ticket named.

    `figures._draw_map_patch` carried the IDENTICAL latent bug - it simply had not yet
    been asked to draw a patch containing one of the seven. A future caller that goes
    straight to `.intersection()` re-opens it, silently, until some frame far into a
    batch happens to hit one.
    """
    for name in ("bev.py", "figures.py"):
        src = (ROOT / "src" / name).read_text()
        for line in src.splitlines():
            if ".intersection(patch)" in line and "clip_to_patch" not in line:
                raise AssertionError(f"{name} clips a map polygon without the repair: "
                                     f"{line.strip()}")


@check("test_bev_agrees_with_c4_and_c5_on_the_rendered_subset")
def _():
    """R9: cross-validate Stage D against an INDEPENDENT derivation, not itself.

    Two invariants over all 1,800 rendered frames, both measured 2026-09-06:

    1. BEV's in-camera count is never GREATER than C4's frustum count. Greater would
       mean the BEV invented a visible object, which is the direction that is a bug.
       It is LOWER on 154 frames, fully explained: mostly frustum boxes beyond the
       100 m radius (C4 has no range limit, D-022), plus 6 boxes within 1.4 m of the
       boundary that the two distance measures classify differently - C4 uses 3D
       distance in the CAMERA frame, BEV uses 2D distance in the EGO frame.
    2. BEV's 360 count never exceeds C5's, because the BEV clips at 100 m. 99.3% of
       agents survive the clip, which corroborates the 99.7% coverage D-042 was
       chosen on.

    Skipped when the batch has not been rendered, so the suite stays runnable.
    """
    import pandas as pd
    jl = ROOT / "outputs" / "bev_symbolic.jsonl"
    if not jl.exists():
        print("     (skip: outputs/bev_symbolic.jsonl not built)")
        return
    rows = [json.loads(l) for l in jl.open()]
    d = pd.DataFrame([{"sample_token": r["sample_token"], "cam": r["n_in_camera"],
                       "all": r["n_agents"]} for r in rows])
    o = pd.read_parquet(ROOT / "outputs" / "objects_trainval.parquet")[
        ["sample_token", "n_boxes_visible"]]
    i = pd.read_parquet(ROOT / "outputs" / "interactions_trainval.parquet")[
        ["sample_token", "n_agents"]]
    m = d.merge(o, on="sample_token").merge(i, on="sample_token")

    over = (m.cam > m.n_boxes_visible).sum()
    assert over == 0, (
        f"{over} frames report MORE objects in camera than C4's frustum filter finds - "
        "the BEV has invented visible objects")
    over360 = (m["all"] > m.n_agents).sum()
    assert over360 == 0, f"{over360} frames exceed C5's 360-degree agent count"
    kept = m["all"].sum() / m.n_agents.sum()
    assert kept > 0.98, (
        f"only {kept:.1%} of agents survive the {B.BEV_RADIUS_M:.0f} m clip; D-042 was "
        "chosen on 99.7% coverage, so the radius or the clip has changed")


# --- the repair (F-063c, F-064b) --------------------------------------------------------
#
# The old guard checked the CONTENT DICT: both arms were driven by one dict, so both were
# said to see the same scene. They were not. The raster drew map polygons and the text
# wrote their names, and neither drew the traffic-light layer at all. These check the
# RENDERED OUTPUT, which is the layer the experiment actually reads.

def _repair_content():
    """One square of every drawn layer, plus a fixture, in the ego frame."""
    import numpy as _np

    def square(cx, cy, half=6.0):
        e = _np.array([[cx - half, cy - half], [cx + half, cy - half],
                       [cx + half, cy + half], [cx - half, cy + half],
                       [cx - half, cy - half]], dtype=float)
        return [(e, [])]

    layers = {layer: square(0.0 if i == 0 else 20.0 + 4 * i, 0.0)
              for i, layer in enumerate(B._MAP_LAYERS)}
    c = _content([_agent()], layers=layers)
    c["traffic_lights"] = [{"x_fwd_m": 18.0, "y_left_m": -3.0,
                            "range_m": math.hypot(18.0, 3.0),
                            "bearing_deg": math.degrees(math.atan2(-3.0, 18.0))}]
    return c


@check("test_the_traffic_light_layer_reaches_both_rendered_arms")
def _():
    """F-063c: `traffic_light_ahead` scored 0.67 -> 0.00 in the BEV arms because the layer
    the tag derives from was drawn by neither of them. The tag was unanswerable by
    construction, which is not a result about BEV."""
    from PIL import Image

    assert "traffic_light" in B.DRAWN_LAYERS
    c = _repair_content()
    text = B.describe_bev(c)
    assert "fixture" in text.lower(), "the text arm does not state the fixture"
    assert "17" in text or "18" in text, f"the fixture has no distance: {text}"

    out = Path(tempfile.gettempdir()) / "bev_repair_test.png"
    B.render_bev(c, out, pixels=256)
    px = set(Image.open(out).convert("RGB").getdata())
    want = tuple(int(B.BEV_COLORS["traffic_light"][i:i + 2], 16) for i in (1, 3, 5))
    assert any(sum(abs(a - b) for a, b in zip(want, got)) < 30 for got in px), \
        "the fixture colour is nowhere in the rendered raster"
    out.unlink(missing_ok=True)


@check("test_the_text_arm_states_where_a_surface_is_not_only_that_it_exists")
def _():
    """F-064b: the description used to write the layers as a bare list of names, so
    at_intersection, intersection_ahead and lane_change scored 0.00 in the text arm while
    the raster showed exactly where the junction was."""
    text = B.describe_bev(_repair_content())
    assert "present out to" in text, "no extent: the text arm is back to a list of names"
    for layer in B._MAP_LAYERS:
        assert layer.replace("_", " ") in text, f"{layer} missing from the description"
    body = text.split("ROAD SURFACES", 1)[1].split("TRAFFIC LIGHT", 1)[0]
    assert "the vehicle is standing on it" in body, "containment is never stated"
    assert "bearing" in body, "a surface the vehicle is not on has no bearing"


@check("test_neither_arm_claims_to_know_which_light_is_lit")
def _():
    """The map's `items` field is the fixture's bulb inventory, never the live state
    (F-004/F-076). A description mentioning a colour would invent ground truth the dataset
    does not have, and Stage H exists precisely because it does not."""
    text = B.describe_bev(_repair_content()).lower()
    block = text.split("traffic light", 1)[1]
    for word in ("red", "green", "amber", "yellow", "lit", "stop", "go"):
        assert word not in block.split("objects")[0], \
            f"the fixture block states {word!r}, which the map cannot support"


@check("test_the_repaired_arms_read_the_repaired_inputs_and_keep_the_published_ones")
def _():
    """R29 in advance: the published rasters and descriptions ARE the evidence for the
    v3_bev and v3b_bev_symbolic numbers. Re-rendering over them would leave no way to show
    what the repair changed, which is the whole reason the re-run is being paid for."""
    import inspect

    src = inspect.getsource(B.build_bev_batch)
    assert "BEV_R2_DIR" in src and "BEV_R2_JSONL" in src, \
        "the batch builder no longer defaults to the repaired destinations"
    assert B.BEV_R2_DIR != "data/bev"
    assert B.BEV_R2_JSONL != "outputs/bev_symbolic.jsonl"

    import src.prompts as P

    assert P.REGISTRY["v3_bev"] != P.REGISTRY["v3_bev_r2"], \
        "the repaired raster arm reuses the prompt that describes the old picture"
    assert "teal" in P.REGISTRY["v3_bev_r2"].lower(), "the new glyph is not in the legend"
    assert "teal" not in P.REGISTRY["v3_bev"].lower(), \
        "the PUBLISHED prompt changed; its results are no longer reproducible from it"
    assert "present out to" not in P.REGISTRY["v3b_bev_symbolic"], \
        "the published text prompt changed"


@check("test_the_renderer_and_the_packer_agree_on_where_the_repaired_inputs_live")
def _():
    """The destinations are literals inside `build_bev_batch`, because the appendix ledger
    credits a generator by the string in its write call and cannot follow a constant
    (F-087). The constants are what `src/data.py` packs and `scripts/preflight.py` checks.
    Two spellings of one fact need a drift guard, which is this (R4)."""
    import inspect

    src = inspect.getsource(B.build_bev_batch)
    assert f'"{B.BEV_R2_DIR}"' in src, "the writer no longer names BEV_R2_DIR"
    assert f'"{B.BEV_R2_JSONL}"' in src, "the writer no longer names BEV_R2_JSONL"
    assert '"data/bev"' in src and '"outputs/bev_symbolic.jsonl"' in src, \
        "the PUBLISHED destinations lost their writer when the repair was added"

    import src.data as D

    assert "BEV_R2_DIR" in inspect.getsource(D.build_colab_bundle)
    assert "B.BEV_R2_JSONL" in inspect.getsource(D.build_colab_bundle)


@check("test_the_map_geometry_measurements_are_physically_sane")
def _():
    """R2: bounds, not eyeballing. Nothing may be farther than the patch, a surface the
    vehicle stands on is at range 0, and an extent is never negative."""
    c = _repair_content()
    geom = B.map_geometry(c)
    R = c["radius_m"]
    for layer, g in geom.items():
        assert 0.0 <= g["range_m"] <= R * math.sqrt(2), (layer, g["range_m"])
        assert -180.0 <= g["bearing_deg"] <= 180.0, (layer, g["bearing_deg"])
        assert (g["range_m"] == 0.0) == g["on_it"], (layer, g)
        for bearing, reach in g["extent_m"].items():
            assert 0.0 <= reach <= R + 1e-6, (layer, bearing, reach)
    first = B._MAP_LAYERS[0]
    assert geom[first]["on_it"], "the layer drawn under the vehicle reads as not under it"


if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} BEV self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
