"""Bird's-eye view, rendered and symbolic — Stage D, answering RQ3a.

The SystemX brief asks for "vue conducteur OU vue drone / BEV", so the
front-vs-BEV comparison (E5-E7, F4) is the thesis differentiator.

WHY THERE ARE TWO RENDERERS (D-039)
-----------------------------------
Neither of the two closest published works feeds a raster BEV to a model:
Talk2BEV (REFERENCES.md R-20) converts BEV objects to *text* for an LLM, and
BEV-InMLLM (R-21) injects *learned features*. Separately, MLLMs are measured to
read abstract non-photographic images - diagrams, charts, maps - poorly (R-22),
and a hand-drawn BEV raster is exactly such an image.

So the honest prediction, written down before the experiment: the raster arm may
well lose to the front camera. Alone that result is uninterpretable - it cannot
separate "BEV carries no extra information" from "the VLM cannot parse our
renderer". `describe_bev` is the control that separates them.

The two renderers therefore take THE SAME `bev_content` dict. That is structural,
not a promise: if one arm could see something the other could not, the comparison
would measure content rather than form, and the whole of RQ3a would be void.

MEASURED PARAMETERS
-------------------
Everything below was measured on the 1,800-frame evaluation subset before being
chosen (R1). See F-051.

NOT PERCEIVED, RENDERED. This BEV is drawn from the ground-truth 3D boxes and the
HD map, not inferred from cameras as in BEVFormer-style work (R-33). It is an
upper bound on what a perception stack could supply, and the report must say so -
an AD reader assumes the learned kind by default.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# --- Step D parameters ---------------------------------------------------------------
BEV_RADIUS_M = 100.0          # half-extent of the patch, metres. AUTHOR'S DECISION (D-042)
                              # taken on the measurement: over 2,258 frustum boxes in 250
                              # subset frames, a patch of radius R contains
                              #   30 m 49.6% | 50 m 80.3% | 75 m 96.1% | 100 m 99.7%
                              # of the boxes the ground truth counts. D-022 deliberately
                              # put NO range limit on the presence tags - a car at 60 m is
                              # plainly visible and `has_vehicle` is True there - so a BEV
                              # cropped at the originally planned 50 m would be blind to
                              # 19.7% of objects the CAMERA can see, handicapping the BEV
                              # arm by construction and making the RQ3a headline partly an
                              # artefact of cropping. Same failure class as F-029.
BEV_PIXELS = 768              # raster size. At radius 100 m this is 0.26 m/px, so a 4.5 m
                              # car is ~17 px long - small, but a shape rather than a dot.
                              # Scaled up from 512 with the radius to hold that legibility.
CAM_FRONT_HFOV_DEG = 64.0     # full horizontal field of view of CAM_FRONT. L-019 measured
                              # the half-FOV at about 32 deg. Drawn on the raster and stated
                              # in the text so BOTH arms know which region the
                              # CAM_FRONT-scoped tags (D-025) refer to, and a model is not
                              # scored wrong for correctly reporting a car behind the ego.

# Fixed legend, identical on every frame. Consistency matters more than beauty: the model
# has to learn this from PROMPT_V3_BEV, which MUST describe exactly these colours, so they
# can never drift between frames or the prompt silently stops being true.
BEV_COLORS: dict[str, str] = {
    "drivable_area": "#ffffff",
    "lane":          "#eaf2fa",   # was near-white in the C3 figures, where the map is the
                                  # subject. Here it must READ as lane structure against
                                  # drivable_area at a glance, so the contrast is raised.
    "road_segment":  "#c3dcf0",
    "carpark_area":  "#f6dde6",
    "walkway":       "#dff0dc",
    "ped_crossing":  "#cfe0f2",
    "stop_line":     "#f8e3b4",
    "background":    "#e6e6e6",   # anything NOT drivable, islands included
    "ego":           "#e53935",
    "camera_fov":    "#ffd54f",
    "vehicle":       "#1e88e5",
    "large_vehicle": "#3949ab",
    "pedestrian":    "#43a047",
    "cyclist":       "#8e24aa",
    "barrier":       "#fb8c00",
    "traffic_cone":  "#f4511e",
    "traffic_light": "#00897b",   # F-063c: the layer traffic_light_ahead derives from was
                                  # drawn by NEITHER arm, so the tag was unanswerable by
                                  # construction. Teal reads against every road colour
                                  # above and is used by nothing else.
    "other":         "#9e9e9e",   # R36: anything the pipeline emits NO tag for is drawn
}                                 # grey, so a coverage gap is visible instead of disguised

# Category -> legend key. Read from nuScenes' own category names (D-024), never inferred.
_CATEGORY_STYLE = (
    ("human.pedestrian",             "pedestrian"),
    ("vehicle.bicycle",              "cyclist"),
    ("vehicle.motorcycle",           "cyclist"),
    ("vehicle.bus",                  "large_vehicle"),
    ("vehicle.truck",                "large_vehicle"),
    ("vehicle.construction",         "large_vehicle"),
    ("vehicle.trailer",              "large_vehicle"),
    ("vehicle.emergency",            "large_vehicle"),
    ("vehicle.",                     "vehicle"),
    ("movable_object.barrier",       "barrier"),
    ("movable_object.trafficcone",   "traffic_cone"),
)

# Map layers drawn, in paint order. Exactly the layers MapIndex indexes, minus road_block,
# which D-019 excludes from the pipeline - the picture must not show a layer no label is
# ever derived from (R36).
_MAP_LAYERS = ("drivable_area", "lane", "road_segment", "carpark_area",
               "walkway", "ped_crossing", "stop_line")

# Everything the two arms actually show, polygons plus the traffic-light points. ONE
# definition (R4): `scripts/preflight.py` gates a GPU session on it and `tests/test_bev.py`
# asserts against it, and when the drawn set lived in three places the traffic-light layer
# was missing from the renderer while the preflight still reported the input as able to
# answer the tag (F-063c).
DRAWN_LAYERS = _MAP_LAYERS + ("traffic_light",)

# Traffic-light fixtures are POINTS (the map layer is lines; MapIndex indexes the node
# mean), so they are drawn as a marker rather than to scale. Deliberately LARGER than the
# pedestrian dot and a different shape: F-063b measured that isolated single dots are
# invisible to the model (pedestrian tags 0.67 -> 0.01) while larger, clustered or to-scale
# glyphs survive, so repeating the 5 pt dot here would close the content gap and leave the
# tag unanswerable anyway.
TRAFFIC_LIGHT_MARKER_PT = 9.0

# Bearings along which the text arm reports how far the road surface continues. The raster
# shows the shape of the surface directly; without these the text said only that a layer
# existed somewhere within 100 m, which is F-064b. Four bearings, not a sector sweep: ahead,
# both sides and behind are what separate a junction from a straight road and a lane
# position from a lane change, and each extra ray costs 1,800 renders.
ROAD_EXTENT_BEARINGS_DEG = (0.0, 90.0, -90.0, 180.0)

MOVING_MPS = 0.5              # at or below this an agent is drawn and described as still.
                              # Reuses C2's STATIONARY_MPS so "moving" means the same thing
                              # in the scene description as it does in the labels.


def _style_of(category: str) -> str:
    """Legend key for a nuScenes category name. Unknown categories are 'other'."""
    for prefix, key in _CATEGORY_STYLE:
        if category.startswith(prefix):
            return key
    return "other"


def bev_content(nusc: Any, sample_token: str, index: Any,
                radius_m: float = BEV_RADIUS_M) -> dict[str, Any]:
    """The scene content BOTH arms render. One extractor, two renderers.

    Returns agents in the EGO FRAME (x forward, y left, per D-027) plus the map polygons
    already clipped and rotated into that frame, so neither renderer does geometry of its
    own and the two cannot drift apart.

    Every vector crossing the frame boundary is rotated, positions and velocities alike -
    R40 and F-033, where rotating the position but not the velocity reproduced the original
    defect one level down and silently, because the speeds stayed plausible while pointing
    the wrong way.
    """
    from nuscenes.utils.geometry_utils import BoxVisibility
    from pyquaternion import Quaternion
    from shapely.geometry import box as shapely_box

    from . import groundtruth as G

    sample = nusc.get("sample", sample_token)
    # LIDAR_TOP's ego_pose is exactly the keyframe timestamp; CAM_FRONT's is 35.7 ms
    # earlier. C5 uses LIDAR_TOP for the same reason, so the two agree.
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    rot = Quaternion(ego["rotation"]).inverse
    origin = np.array(ego["translation"])
    yaw = Quaternion(ego["rotation"]).yaw_pitch_roll[0]

    # Same call C5's `scene_agent_tracks` uses, so "in the camera" means exactly what it
    # means in the C4/C5 tables. `frustum_boxes` cannot serve here: it returns category and
    # distance but not the annotation token, so its records cannot be matched to agents.
    _, cam_boxes, _ = nusc.get_sample_data(sample["data"]["CAM_FRONT"],
                                           box_vis_level=BoxVisibility.ANY)
    in_frustum = {b.token for b in cam_boxes}

    agents = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        rel = rot.rotate(np.array(ann["translation"]) - origin)
        if math.hypot(rel[0], rel[1]) > radius_m:
            continue
        vel = nusc.box_velocity(ann_token)
        # NaN whenever the agent appears in only one keyframe. Propagated, never
        # zero-filled, so a missing velocity cannot masquerade as a stationary agent.
        v = rot.rotate(vel) if np.isfinite(vel).all() else np.full(3, np.nan)
        speed = float(np.hypot(v[0], v[1]))
        size = ann["size"]                      # [width, length, height]
        box_yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0] - yaw
        agents.append({
            "category": ann["category_name"],
            "style": _style_of(ann["category_name"]),
            "x_fwd_m": float(rel[0]),
            "y_left_m": float(rel[1]),
            "range_m": float(math.hypot(rel[0], rel[1])),
            "bearing_deg": float(math.degrees(math.atan2(rel[1], rel[0]))),
            "yaw_rel_deg": float(math.degrees(box_yaw)),
            "length_m": float(size[1]),
            "width_m": float(size[0]),
            "speed_mps": speed,
            "v_fwd_mps": float(v[0]),
            "v_left_mps": float(v[1]),
            "in_camera": ann_token in in_frustum,
        })
    agents.sort(key=lambda a: a["range_m"])

    # Map polygons, clipped to the patch then rotated into the ego frame.
    cx, cy = float(origin[0]), float(origin[1])
    patch = shapely_box(cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    c, s = math.cos(-yaw), math.sin(-yaw)
    layers: dict[str, list] = {}
    for layer in _MAP_LAYERS:
        tree, _, arr = index.trees.get(layer, (None, None, None))
        if tree is None:
            continue
        rings = []
        for i in tree.query(patch, predicate="intersects"):
            clipped = G.clip_to_patch(arr[int(i)], patch)
            if clipped.is_empty:
                continue
            parts = clipped.geoms if clipped.geom_type.startswith("Multi") else [clipped]
            for p in parts:
                if p.geom_type != "Polygon":
                    continue
                def to_ego(coords):
                    a = np.asarray(coords, dtype=float)
                    dx, dy = a[:, 0] - cx, a[:, 1] - cy
                    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=1)
                rings.append((to_ego(p.exterior.coords),
                              [to_ego(r.coords) for r in p.interiors]))
        if rings:
            layers[layer] = rings

    # Traffic-light FIXTURES. A line layer indexed as points (the node mean), so they are
    # carried as points and never as polygons. Position only: the map's `items` field is
    # the fixture's bulb inventory, not what is lit (F-004/F-076), so neither arm may say
    # anything about state. Both renderers read this one list, so they cannot drift (D-039).
    lights = []
    if getattr(index, "tl_tree", None) is not None:
        for i in index.tl_tree.query(patch, predicate="intersects"):
            pt = index._tl_arr[int(i)]
            dx, dy = float(pt.x) - cx, float(pt.y) - cy
            fx, fy = dx * c - dy * s, dx * s + dy * c
            if math.hypot(fx, fy) > radius_m:
                continue
            lights.append({"x_fwd_m": fx, "y_left_m": fy,
                           "range_m": float(math.hypot(fx, fy)),
                           "bearing_deg": float(math.degrees(math.atan2(fy, fx)))})
    lights.sort(key=lambda t_: t_["range_m"])

    return {
        "sample_token": sample_token,
        "scene_name": nusc.get("scene", sample["scene_token"])["name"],
        "radius_m": radius_m,
        "agents": agents,
        "map_layers": layers,
        "traffic_lights": lights,
        "n_in_camera": sum(a["in_camera"] for a in agents),
    }


def render_bev(content: dict[str, Any], out_path: Path | str,
               pixels: int = BEV_PIXELS) -> Path:
    """Draw one top-down frame: map layers + boxes + ego, heading always UP. Step D1.

    Ego at centre, patch rotated by ego yaw so "up" is always where the car is going -
    otherwise the model would have to recover heading from the map before reading
    anything, which is a second task on top of the one being measured.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Wedge

    R = content["radius_m"]
    fig = plt.figure(figsize=(pixels / 100, pixels / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(BEV_COLORS["background"])

    # Ego frame is x-forward / y-left; the image is heading-up, so plot (-y_left, x_fwd).
    for layer in _MAP_LAYERS:
        for ext, holes in content["map_layers"].get(layer, []):
            ax.add_patch(MplPolygon(np.stack([-ext[:, 1], ext[:, 0]], 1), closed=True,
                                    fc=BEV_COLORS[layer], ec="#c3c8cc", lw=0.4, zorder=1))
            for h in holes:   # punch holes back out or an island vanishes under its road
                ax.add_patch(MplPolygon(np.stack([-h[:, 1], h[:, 0]], 1), closed=True,
                                        fc=BEV_COLORS["background"], ec="#c3c8cc",
                                        lw=0.4, zorder=2))

    # The camera's field of view, so both arms know which region the CAM_FRONT-scoped
    # tags (D-025) are about. Without it a model correctly naming a car behind the ego
    # would be scored wrong, confounding field of view with representation.
    # Drawn as two boundary rays plus a very light fill: at alpha 0.16 the wedge washed
    # out every map layer inside it, which cost more legibility than the cue was worth.
    half = CAM_FRONT_HFOV_DEG / 2.0
    ax.add_patch(Wedge((0, 0), R, 90 - half, 90 + half, fc=BEV_COLORS["camera_fov"],
                       alpha=0.06, ec="none", zorder=3))
    for sign in (-1, 1):
        a_rad = math.radians(90 + sign * half)
        ax.plot([0, R * math.cos(a_rad)], [0, R * math.sin(a_rad)],
                color="#d99a00", lw=1.1, ls="--", alpha=0.9, zorder=4)

    for a in content["agents"]:
        # Vehicles are drawn TO SCALE. Pedestrians, cones and barriers are drawn as
        # fixed-size markers instead: at 0.26 m/px a 0.7 m pedestrian is 2.7 px and a cone
        # is ~1 px, so a to-scale box makes them invisible - "in the patch" but unreadable,
        # which loses them exactly as surely as cropping them out (the R27 failure mode in
        # a different costume). Inflating their BOXES would instead misstate their size, so
        # the glyph differs rather than the scale, and PROMPT_V3_BEV says so.
        if a["style"] in ("pedestrian", "cyclist", "traffic_cone", "barrier", "other"):
            ax.plot(-a["y_left_m"], a["x_fwd_m"], marker="o", ms=5.0,
                    mfc=BEV_COLORS[a["style"]], mec="#222222", mew=0.6, zorder=5)
        else:
            L, W = max(a["length_m"], 0.4), max(a["width_m"], 0.4)
            th = a["yaw_rel_deg"]
            corners = np.array([[L / 2, W / 2], [L / 2, -W / 2],
                                [-L / 2, -W / 2], [-L / 2, W / 2]])
            cs, sn = math.cos(math.radians(th)), math.sin(math.radians(th))
            rc = corners @ np.array([[cs, sn], [-sn, cs]])
            px = -(rc[:, 1] + a["y_left_m"]); py = rc[:, 0] + a["x_fwd_m"]
            ax.add_patch(MplPolygon(np.stack([px, py], 1), closed=True,
                                    fc=BEV_COLORS[a["style"]], ec="#222222", lw=0.5,
                                    zorder=5))
        # Velocity arrow, so the raster carries the motion the text states as speed.
        if a["speed_mps"] > MOVING_MPS:
            ax.arrow(-a["y_left_m"], a["x_fwd_m"], -a["v_left_mps"], a["v_fwd_mps"],
                     head_width=1.6, head_length=2.0, fc="#111111", ec="#111111",
                     lw=0.7, length_includes_head=True, zorder=6)

    # Traffic-light fixtures, above the map and below the ego. A square, not another dot:
    # the shape distinguishes a fixture from a pedestrian at a glance, and the size is the
    # F-063b lesson applied rather than restated.
    for tl in content.get("traffic_lights", []):
        ax.plot(-tl["y_left_m"], tl["x_fwd_m"], marker="s",
                ms=TRAFFIC_LIGHT_MARKER_PT, mfc=BEV_COLORS["traffic_light"],
                mec="#00332e", mew=0.9, zorder=6)

    ax.add_patch(MplPolygon(np.array([[-0.9, -2.3], [0.9, -2.3], [0.9, 2.3], [0, 3.1],
                                      [-0.9, 2.3]]), closed=True,
                            fc=BEV_COLORS["ego"], ec="#7f0000", lw=0.8, zorder=7))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BEV_COLORS["background"])
    plt.close(fig)
    return out_path


def map_geometry(content: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Where each map layer is, measured off the SAME polygons the raster draws.

    F-064b: `describe_bev` used to write the layers as a bare list of names, so the picture
    showed where a junction was while the words said only that one existed somewhere within
    100 m. Three tags scored 0.00 in the text arm for that reason alone. D-039's guarantee
    held for the content dict and not for the rendered output, which is the layer the
    comparison actually reads.

    Per layer: whether the vehicle is standing on it, the nearest point of it (range and
    bearing), and how far along each of ROAD_EXTENT_BEARINGS_DEG the surface is still
    present (the farthest point of it on that ray; a gap along the way is not reported,
    because the raster does not report one either). Nothing here
    is a label: a junction is not named, it is left visible as road surface extending to
    both sides, exactly as the raster leaves it.
    """
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union

    R = float(content["radius_m"])
    here = Point(0.0, 0.0)
    out: dict[str, dict[str, Any]] = {}
    for layer, rings in content["map_layers"].items():
        polys = []
        for ext, holes in rings:
            if len(ext) < 4:
                continue
            try:
                poly = Polygon(ext, [h for h in holes if len(h) >= 4])
                polys.append(poly if poly.is_valid else poly.buffer(0))
            except Exception:                    # noqa: BLE001 - a degenerate ring is not a layer
                continue
        if not polys:
            continue
        shape = unary_union(polys)
        on_it = bool(shape.contains(here))
        if on_it:
            rng, brg = 0.0, 0.0
        else:
            from shapely.ops import nearest_points
            q = nearest_points(shape, here)[0]
            rng = float(here.distance(shape))
            brg = float(math.degrees(math.atan2(q.y, q.x)))
        extents = {}
        for bearing in ROAD_EXTENT_BEARINGS_DEG:
            a = math.radians(bearing)
            ray = LineString([(0.0, 0.0), (R * math.cos(a), R * math.sin(a))])
            hit = ray.intersection(shape)
            # ONE definition, for every layer: how far along this bearing the surface is
            # still present. Not the contiguous run from the vehicle -- that reads as 0 in
            # every direction for a layer the vehicle is merely next to, which is the
            # uninformative sentence F-064b is about. A gap is not reported, and the
            # docstring says so, because the raster does not report one either.
            reach = 0.0
            for part in (hit.geoms if hit.geom_type.startswith("Multi") else [hit]):
                if part.is_empty or part.geom_type != "LineString":
                    continue
                reach = max(reach, max(here.distance(Point(c)) for c in part.coords))
            extents[bearing] = round(reach, 1)
        out[layer] = {"on_it": on_it, "range_m": round(rng, 1),
                      "bearing_deg": round(brg, 1), "extent_m": extents}
    return out


def describe_bev(content: dict[str, Any]) -> str:
    """The same scene content as `render_bev`, as compact text. Step D2 (D-039).

    The control that makes the raster arm interpretable. If the symbolic arm scores far
    above the raster, the BEV *information* helps and our *rendering* is the bottleneck,
    which is R-22's prediction; if the two are close, the raster is being read fine; if
    both lose to the camera, BEV genuinely adds nothing for these tags.

    IT MUST NOT STATE A LABEL. Only geometry, category and motion - the things the raster
    also shows. `describe_bev` naming a tag would hand the answer to one arm and measure
    nothing. `tests/test_bev.py` asserts no schema tag name appears in the output.
    """
    lines = [
        f"Bird's-eye view, {content['radius_m']:.0f} m radius around the vehicle.",
        "Coordinates: distance in metres, bearing in degrees "
        "(0 = straight ahead, + = left, - = right).",
        f"The front camera covers bearings -{CAM_FRONT_HFOV_DEG / 2:.0f} to "
        f"+{CAM_FRONT_HFOV_DEG / 2:.0f} degrees.",
        "",
    ]

    geom = map_geometry(content)
    present = [k for k in _MAP_LAYERS if k in geom]
    if not present:
        lines.append("ROAD: no mapped road surface in range")
    else:
        lines.append("ROAD SURFACES (where each one is, and how far it reaches):")
        for layer in present:
            g = geom[layer]
            where = ("the vehicle is standing on it" if g["on_it"]
                     else f"nearest {g['range_m']:.0f} m, bearing {g['bearing_deg']:+.0f} deg")
            e = g["extent_m"]
            lines.append(
                f"  - {layer.replace('_', ' ')}: {where}; present out to "
                f"{e[0.0]:.0f} m ahead, {e[90.0]:.0f} m to the left, "
                f"{e[-90.0]:.0f} m to the right, {e[180.0]:.0f} m behind")
    lines.append("")

    # Fixtures, stated exactly as the raster draws them: a position and nothing else. The
    # map records which bulbs a fixture has, never which is lit (F-004), so the text says
    # nothing about state and neither does the picture.
    tls = content.get("traffic_lights", [])
    if tls:
        lines.append(f"TRAFFIC LIGHT FIXTURES ({len(tls)}, position only, nearest first):")
        for tl in tls:
            lines.append(f"  - fixture: {tl['range_m']:.0f} m, "
                         f"bearing {tl['bearing_deg']:+.0f} deg")
    else:
        lines.append("TRAFFIC LIGHT FIXTURES: none within range.")
    lines.append("")

    if not content["agents"]:
        lines.append("OBJECTS: none within range.")
    else:
        lines.append(f"OBJECTS ({len(content['agents'])}, nearest first):")
        for a in content["agents"]:
            motion = (f"moving {a['speed_mps']:.1f} m/s" if a["speed_mps"] > MOVING_MPS
                      else "not moving" if np.isfinite(a["speed_mps"]) else "speed unknown")
            cam = "in camera view" if a["in_camera"] else "outside camera view"
            lines.append(f"  - {a['category']}: {a['range_m']:.0f} m, "
                         f"bearing {a['bearing_deg']:+.0f} deg, {motion}, {cam}")
    return "\n".join(lines)


# Where the REPAIRED arms' renders live. Separate from data/bev and
# outputs/bev_symbolic.jsonl, which are the inputs that produced the published v3_bev and
# v3b_bev_symbolic numbers: overwriting them would leave no way to show what the repair
# changed, which is the entire point of re-running (F-063c, F-064b).
BEV_R2_DIR = "data/bev_r2"
BEV_R2_JSONL = "outputs/bev_symbolic_r2.jsonl"


def build_bev_batch(nusc: Any, sample_tokens: list[str],
                    out_dir: Path | str | None = None,
                    jsonl_path: Path | str | None = None,
                    variant: str = "repaired",
                    dataroot: str = "data/nuscenes",
                    radius_m: float = BEV_RADIUS_M,
                    pixels: int = BEV_PIXELS,
                    progress: bool = True) -> dict[str, int]:
    """Render both arms for a list of keyframes. CPU-only, runs on the Mac. Step D3.

    Grouped by scene because `MapIndex` costs seconds to build per location and the
    subset's 1,800 tokens span only 4 maps (R20: push the cost onto the cached path).
    Resumable: an existing PNG is skipped, and the JSONL is rewritten from scratch each
    run so it cannot end up half-stale against the images.
    """
    import json

    from . import data as D
    from . import groundtruth as G

    # Both destinations are named HERE, as literals, because this is the only function
    # that writes either and the appendix ledger credits a generator by the string inside
    # its write call (F-087). `tests/test_bev.py` asserts they still match BEV_R2_DIR and
    # BEV_R2_JSONL, which is what the packer and the preflight read.
    if variant == "published":
        out_dir = out_dir or "data/bev"
        jsonl_path = jsonl_path or "outputs/bev_symbolic.jsonl"
    elif variant == "repaired":
        out_dir = out_dir or "data/bev_r2"
        jsonl_path = jsonl_path or "outputs/bev_symbolic_r2.jsonl"
    else:
        raise ValueError(f"variant must be 'published' or 'repaired', not {variant!r}")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(sample_tokens)
    by_scene: dict[str, list[str]] = {}
    for t in sample_tokens:
        by_scene.setdefault(nusc.get("sample", t)["scene_token"], []).append(t)

    indexes: dict[str, Any] = {}
    rows, n_rendered = [], 0
    for si, (scene_token, toks) in enumerate(by_scene.items()):
        scene = nusc.get("scene", scene_token)
        loc = D.map_name_for_scene(nusc, scene)
        if loc not in indexes:
            indexes[loc] = G.MapIndex(dataroot, loc)
        if progress and si % 20 == 0:
            print(f"  scene {si}/{len(by_scene)} ({loc})", flush=True)
        for t in toks:
            content = bev_content(nusc, t, indexes[loc], radius_m=radius_m)
            png = out_dir / f"{t}.png"
            if not png.exists():
                render_bev(content, png, pixels=pixels)
                n_rendered += 1
            rows.append({"sample_token": t, "scene_name": content["scene_name"],
                         "n_agents": len(content["agents"]),
                         "n_in_camera": content["n_in_camera"],
                         "bev_png": str(png), "description": describe_bev(content)})

    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"n_tokens": len(wanted), "n_rows": len(rows), "n_rendered": n_rendered,
            "n_skipped": len(rows) - n_rendered}
