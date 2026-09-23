"""Structured scenario description — Step G3, serving RQ5.

The slides' TERMINAL artifact. Their pipeline is
`Storyboard -> Sequencage video -> Analyse de scenes -> Description de scenario`, feeding
MOSAR, and the description has two named parts:
    Scenography  road type (lanes, intersection, traffic circle), object instance types
                 and locations
    Parameters   speed ranges, road extent, pedestrian speed ranges

WHAT ONE SCENARIO IS. The slide's own storyboard strip is five panels of ONE accident
scenario, with a single MOSAR logo under the whole strip rather than under any panel, and its
scene-analysis page is one environment paragraph followed by a numbered "Suite d'evenements".
So a scenario is ONE SCENE, and G1's panels are its timeline. (PROCESS.md's RQ5 wording says
"each event", which would make a scenario a panel; the slide settles it and the disagreement
is recorded as a fork rather than silently resolved.)

WHY EVERY PARAMETER IS A RANGE. The slide writes "range" three times - speed range, road range
size, pedestrian speed range - and that is not incidental phrasing: a LOGICAL scenario in
Menzel's functional/logical/concrete hierarchy is precisely a parameter-range state space over
a functional description. PROCESS.md section 2.1 already claims this artifact sits at the
logical level, so the schema carries [min, max] per parameter and never a point value.

SCOPE, stated because it is the error this project keeps making (R25, F-025). Actors here are
**360-degree** facts drawn from the annotation set, not CAM_FRONT facts: only 28.46% of
nuScenes annotations fall inside the front camera, and 444 of 850 scenes contain at least one
object class that never enters it at all. Emitting C4's CAM_FRONT counts as "the scene's
actors" would repeat F-025's error at the artifact level, where it is far harder to spot than
on a figure. Every actor entry therefore carries `n_in_camera` beside its count, so a reader
can see which part of the scenario the front camera could ever have supported.

Build:  ./venv/bin/python -c "import src.scenario as S; S.build_scenario_descriptions()"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# --- parameters (one place only, each with the measurement that chose it) ---------------

SCENARIO_SCHEMA_VERSION = "G3.1"

# A plausibility bound on pedestrian speed, applied because the raw numbers are not
# plausible. `box_velocity` is a centred difference across ~0.5 s keyframe gaps, so
# sub-metre annotation jitter becomes multi-m/s error: the measured maximum over the
# dataset is 17.4 m/s and 21 of 777 scenes exceed 4 m/s, all `human.pedestrian.adult`.
# Measured p99 is 2.171 m/s, and 3.0 m/s is a fast run. Values above it are FLAGGED and
# excluded from the emitted range, never silently clipped into it - a MOSAR scenario handed
# a sprinting pedestrian is worse than one told the sample was noisy (R2, R11).
PED_SPEED_PLAUSIBLE_MPS = 3.0

# Lighting, the one PEGASUS Layer 5 field this dataset can support deterministically.
# nuScenes has no weather or lighting annotation, but `log.logfile` embeds the local
# capture time (e.g. n015-2018-08-02-17-16-37+0800). Measured over all 850 scenes: a split
# at 18:00 reproduces the human `description` keyword "night" on **850/850 scenes, 100.0%**,
# with 99 night scenes by both routes and ZERO disagreements in either direction. Capture
# hours run 10:00-19:00, so no pre-dawn case exists to test the lower bound - it is written
# for completeness, not validated. Weather (19.4% of descriptions say rain) has no
# deterministic source and stays absent; Layer 5 is therefore PARTLY covered, not covered.
NIGHT_FROM_HOUR = 18
NIGHT_TO_HOUR = 6

_LOGFILE_TIME = re.compile(r"-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")

# nuScenes category prefixes -> the actor kinds a scenario description talks about. Read
# from the category names themselves rather than from our 37-tag vocabulary on purpose: the
# tag schema collapses 23 nuScenes categories into 8 signatures (bus, truck, trailer and
# construction vehicle share one, 12.38% of annotations), so describing actors through it
# would lose instance TYPE, which is exactly what the slide asks for by name.
_PEDESTRIAN = "human.pedestrian"


def _capture_hour(nusc: Any, scene: dict) -> int | None:
    m = _LOGFILE_TIME.search(nusc.get("log", scene["log_token"])["logfile"])
    return int(m.group(4)) if m else None


def lighting_of(nusc: Any, scene: dict) -> str:
    """"day" or "night", from the log's local capture time. 100% agreement with the human
    keyword over all 850 scenes; see NIGHT_FROM_HOUR."""
    h = _capture_hour(nusc, scene)
    if h is None:
        return "unknown"
    return "night" if (h >= NIGHT_FROM_HOUR or h < NIGHT_TO_HOUR) else "day"


def _rng(values: list[float]) -> list[float] | None:
    """[min, max] rounded, or None when nothing was observed. Never a point value."""
    v = [x for x in values if x == x]                     # drop NaN
    return [round(min(v), 2), round(max(v), 2)] if v else None


def scene_scenario(nusc: Any, scene_name: str) -> dict[str, Any]:
    """The structured scenario description for one scene. Step G3."""
    import numpy as np
    import pandas as pd

    from . import groundtruth as G

    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    kin = pd.read_parquet(OUT / "ego_kinematics_trainval.parquet")
    kin = kin[kin.scene_name == scene_name]
    mc = pd.read_parquet(OUT / "map_context_trainval.parquet")
    mc = mc[mc.scene_name == scene_name]
    panels = pd.read_parquet(OUT / "storyboard_panels.parquet")
    panels = panels[panels.scene_name == scene_name]
    rb = pd.read_parquet(OUT / "roundabout_traversals.parquet")
    # Three of the five rows are REJECTED candidates. A join that forgets the filter emits
    # five traffic circles where the dataset has two (F-027).
    circle = bool(len(rb[(rb.scene_name == scene_name) & rb.is_roundabout])) if "is_roundabout" in rb else False

    # 360-degree actors, with locations and velocities, in the ego frame (D-027/R40).
    tracks = G.scene_agent_tracks(nusc, scene)
    per_cat: dict[str, dict] = {}
    ped_speeds, ped_rejected = [], 0
    for frame in tracks:
        for inst, st in frame.items():
            d = per_cat.setdefault(st["cat"], {"inst": set(), "in_cam": set(),
                                               "rng": [], "spd": []})
            d["inst"].add(inst)
            if st["in_frustum"]:
                d["in_cam"].add(inst)
            d["rng"].append(float(np.hypot(st["fwd_m"], st["lat_m"])))
            d["spd"].append(st["speed_mps"])
            if st["cat"].startswith(_PEDESTRIAN):
                s = st["speed_mps"]
                if s != s:                      # NaN: agent seen in one keyframe only
                    pass
                elif s <= PED_SPEED_PLAUSIBLE_MPS:
                    ped_speeds.append(s)
                else:
                    ped_rejected += 1           # counted, not clipped into the range

    actors = [{
        "category": cat,
        "n_instances": len(d["inst"]),
        "n_in_camera": len(d["in_cam"]),          # scope provenance, never a filter
        "range_m": _rng(d["rng"]),
        "speed_mps": _rng(d["spd"]),
    } for cat, d in sorted(per_cat.items(), key=lambda kv: -len(kv[1]["inst"]))]

    ego_x, ego_y = kin.ego_x.to_numpy(), kin.ego_y.to_numpy()
    extent = float(np.hypot(ego_x.max() - ego_x.min(), ego_y.max() - ego_y.min()))

    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "scene": {
            "name": scene_name,
            "location": str(kin.location.iloc[0]),
            "duration_s": round(float(kin.t_rel_s.max() - kin.t_rel_s.min()), 2),
            "n_keyframes": int(len(kin)),
            "human_description": scene["description"],   # WHOLE-SCENE, L-007
        },
        "scenography": {
            "lighting": lighting_of(nusc, scene),        # PEGASUS L5, partial
            "road": {
                # Both where the ego IS and what is AHEAD, per D-017. Containment alone
                # would call scene-0553 "not at an intersection" while its human description
                # reads "Wait at intersection" - the ego is stopped at the stop line, which
                # is outside the junction polygon. Both facts are true of different things.
                "intersection_fraction": round(float(mc.at_intersection.mean()), 3),
                "intersection_ahead_fraction": round(float(mc.intersection_ahead.mean()), 3),
                "on_lane_fraction": round(float(mc.on_lane.mean()), 3),
                "has_ped_crossing": bool(mc.ped_crossing_ahead.any() or mc.on_ped_crossing.any()),
                "has_stop_line": bool(mc.stop_line_ahead.any() or mc.on_stop_line.any()),
                "traffic_circle": circle,
                "on_carpark": bool(mc.on_carpark.any()),
            },
            "actors": actors,
        },
        "parameters": {
            "ego_speed_mps": _rng(kin.speed_mps.tolist()),
            "road_extent_m": round(extent, 1),
            "pedestrian_speed_mps": _rng(ped_speeds),
            "pedestrian_speed_rejected_implausible": ped_rejected,
        },
        "timeline": [{
            "panel": int(r.panel_index),
            "t_s": [round(float(r.t_start_s), 2), round(float(r.t_end_s), 2)],
            "steering": r.lateral_label,
            "speed": r.longitudinal_label,
            "lane_change": bool(r.lane_change),
            "description": r.description,
        } for r in panels.itertuples()],
    }


def scenario_schema(out_path: str = "outputs/scenario_schema.json") -> dict:
    """The documented schema, emitted as an artifact. G3's first verify criterion.

    Documented rather than inferred: a reader must be able to see which fields exist, what
    each is derived from, and - the part that matters - which of the slide's named fields
    this dataset CANNOT support.
    """
    schema = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "abstraction_level": "logical (Menzel) - every parameter is a [min, max] range",
        "scope_warning": (
            "actors are 360-degree annotation facts, NOT CAM_FRONT facts. Only 28.46% of "
            "nuScenes annotations fall in the front camera and 444 of 850 scenes contain a "
            "class it never sees. `n_in_camera` is provided per actor so the front-camera "
            "subset stays visible (R25/F-025)."),
        "fields": {
            "scenography.lighting": {
                "source": "log.logfile local capture hour, split at 18:00",
                "pegasus_layer": 5,
                "validation": "100.0% agreement with the human 'night' keyword on 850/850 scenes"},
            "scenography.road.*": {"source": "map_context_trainval.parquet (C3)",
                                   "pegasus_layer": 1},
            "scenography.actors[]": {
                "source": "groundtruth.scene_agent_tracks - the raw 360-degree annotation set",
                "pegasus_layer": 4,
                "fields": ["category", "n_instances", "n_in_camera", "range_m", "speed_mps"]},
            "parameters.ego_speed_mps": {"source": "ego_kinematics_trainval.parquet (C1)"},
            "parameters.road_extent_m": {
                "source": "bounding-box diagonal of the ego trajectory",
                "caveat": "a PROXY for the slide's 'road range size'. It measures how far the "
                          "ego travelled, not how large the road is, and collapses to ~0 on "
                          "the 65 scenes where the ego never moves."},
            "parameters.pedestrian_speed_mps": {
                "source": "scene_agent_tracks, human.pedestrian.* only",
                "bound": f"values above {PED_SPEED_PLAUSIBLE_MPS} m/s excluded and counted, "
                         "not clipped (R2/R11)"},
            "timeline[]": {"source": "storyboard_panels.parquet (G1)"},
        },
        "absent_fields": {
            "lane_count": (
                "The slide names '3 lines [lanes] road'. No cached table carries a lane "
                "COUNT. It is derivable from the map's lane STRtree, but `road_block` - the "
                "layer that would carry it - is DEGENERATE ON TWO MAPS: all 676 "
                "singapore-queenstown records and all 387 singapore-hollandvillage records "
                "point at a single whole-map polygon (7,748 and 4,851 nodes). That is 200 of "
                "850 scenes, 23.5%. D-019/F-020 recorded only queenstown."),
            "weather": (
                "19.4% of human descriptions say rain; nuScenes has no weather annotation "
                "and no deterministic proxy was found. PEGASUS Layer 5 is therefore PARTLY "
                "covered by `lighting`, not covered."),
            "pegasus_layer_6": "V2X / digital information is absent from the dataset entirely.",
        },
    }
    (ROOT / out_path).write_text(json.dumps(schema, indent=2))
    return schema


def build_scenario_descriptions(nusc: Any, scenes: list[str] | None = None,
                                out_path: str = "outputs/scenario_descriptions.jsonl") -> int:
    """Emit one scenario description per scene. Step G3."""
    names = scenes or sorted(s["name"] for s in nusc.scene)
    with (ROOT / out_path).open("w") as f:
        for n in names:
            f.write(json.dumps(scene_scenario(nusc, n)) + "\n")
    return len(names)


def score_completeness(path: str = "outputs/scenario_descriptions.jsonl") -> dict:
    """FIELD-POPULATION completeness. G3's second verify criterion.

    WHAT THIS DELIBERATELY DOES NOT MEASURE. The obvious reading of "are the actors of the
    real scene present in the description?" is actor RECALL against the nuScenes annotation
    set - and that is 100% by construction here, because `scene_agent_tracks` reads
    `sample["anns"]` directly. The description IS the annotation set, reshaped. Reporting
    100% would be R21's exact error: a number that agrees perfectly with its own source has
    measured nothing. (A related tautology was reproduced for G1's panel captions: scored
    against the CAM_FRONT tables that generated them they give 315/315 = 100.0%, while
    against the raw annotations they give 72.6% - because THOSE captions are front-scoped
    and the annotations are not.)

    What IS a real question is how often the schema can be FILLED: a field that is absent on
    a third of scenes is a limit of the dataset, and the slide named fields this dataset
    cannot supply at all. That is what this reports, per field, with the absent ones listed
    separately rather than scored as misses.
    """
    rows = [json.loads(l) for l in (ROOT / path).open()]
    n = len(rows)

    def populated(r, path_):
        cur = r
        for k in path_.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                return False
        return cur != [] and cur != ""

    fields = ["scenography.lighting", "scenography.road", "scenography.actors",
              "parameters.ego_speed_mps", "parameters.road_extent_m",
              "parameters.pedestrian_speed_mps", "timeline"]
    out = {f: round(100 * sum(populated(r, f) for r in rows) / n, 1) for f in fields}
    return {
        "n_scenes": n,
        "field_population_pct": out,
        "fully_populated_pct": round(
            100 * sum(all(populated(r, f) for f in fields) for r in rows) / n, 1),
        "not_measured": "actor recall against the annotation set: 100% by construction (R21)",
        "absent_by_dataset": ["lane_count", "weather", "pegasus_layer_6"],
    }


# --- Step G5: OpenSCENARIO export -----------------------------------------------------
#
# PLAN G5, the bonus: "European homologation workflows speak OpenX, which fits the brief's
# Catena-X standardisation framing." The G3 description is already Menzel's LOGICAL level --
# every parameter a [min, max] range (D-051) -- and OpenSCENARIO expresses exactly that as a
# parameterised .xosc whose ranges a ParameterValueDistribution later concretises. So this
# is a format change, not a new derivation, and nothing here invents a number.
#
# WHAT THIS FILE CANNOT BE, SAID IN THE FILE ITSELF. An .xosc is not executable without an
# OpenDRIVE road network, and **nuScenes ships none** -- its maps are polygon layers in a
# custom JSON format, not `.xodr`. `<RoadNetwork>` is therefore emitted EMPTY with a comment
# saying so, rather than pointing at a plausible-looking path that does not exist. Two more
# gaps follow from the same place: actors are declared but never PLACED, because G3 holds
# per-category ranges rather than per-instance trajectories, and `lane_count` is absent
# because `road_block` is degenerate on two maps (F-020). An export that quietly filled any
# of the three would be a file that claims conformance it has not got, which is the R30
# problem wearing an ASAM logo.
#
# ponytail: written with stdlib ElementTree and checked for well-formedness and structure,
# NOT against the official ASAM XSD, which is not on disk. `pip install scenariogeneration`
# brings the schema; `validate_xosc` below uses it automatically when it is importable and
# says plainly when it is not, so the upgrade is one install and no code change.

XOSC_REV = (1, 2)          # ASAM OpenSCENARIO 1.2
XOSC_DIR = "outputs/xosc"

# nuScenes category -> (OpenSCENARIO element, its category enum value). The enums are fixed
# by the standard, so this is a translation table and not a taxonomy: `vehicle.construction`
# has no OSC counterpart and becomes a truck, `personal_mobility` and `stroller` are not
# pedestrian categories in OSC and become the generic `pedestrian`. Every lossy mapping is
# recorded in the emitted file as an attribute, so the nuScenes name survives the round trip
# (R4: the vocabulary is written once, here).
OSC_CATEGORY = {
    "vehicle.car": ("Vehicle", "car"),
    "vehicle.truck": ("Vehicle", "truck"),
    "vehicle.construction": ("Vehicle", "truck"),
    "vehicle.trailer": ("Vehicle", "trailer"),
    "vehicle.bus.bendy": ("Vehicle", "bus"),
    "vehicle.bus.rigid": ("Vehicle", "bus"),
    "vehicle.emergency.ambulance": ("Vehicle", "van"),
    "vehicle.emergency.police": ("Vehicle", "car"),
    "vehicle.motorcycle": ("Vehicle", "motorbike"),
    "vehicle.bicycle": ("Vehicle", "bicycle"),
    "human.pedestrian.adult": ("Pedestrian", "pedestrian"),
    "human.pedestrian.child": ("Pedestrian", "pedestrian"),
    "human.pedestrian.construction_worker": ("Pedestrian", "pedestrian"),
    "human.pedestrian.police_officer": ("Pedestrian", "pedestrian"),
    "human.pedestrian.personal_mobility": ("Pedestrian", "pedestrian"),
    "human.pedestrian.stroller": ("Pedestrian", "pedestrian"),
    "human.pedestrian.wheelchair": ("Pedestrian", "wheelchair"),
    "animal": ("Pedestrian", "animal"),
    "movable_object.barrier": ("MiscObject", "barrier"),
    "movable_object.trafficcone": ("MiscObject", "obstacle"),
    "movable_object.debris": ("MiscObject", "obstacle"),
    "movable_object.pushable_pullable": ("MiscObject", "obstacle"),
    "static_object.bicycle_rack": ("MiscObject", "obstacle"),
}

# The ego manoeuvre vocabulary -> what an OSC Act is told to do. Only the LONGITUDINAL axis
# maps onto a standard action without inventing geometry: a lane change needs a target lane
# and a turn needs a road to turn on, and neither exists without an OpenDRIVE network. Those
# panels emit a `UserDefinedAction` carrying the manoeuvre name instead of a fabricated
# `LaneChangeAction` -- a reader can see what was known and what was not.
_LON_ACTION = {"accelerating": "increase", "decelerating": "decrease"}


def _speed_action(parent: Any, value: str) -> None:
    """A SpeedAction driving toward the scenario's ego-speed parameter."""
    import xml.etree.ElementTree as ET

    pa = ET.SubElement(parent, "PrivateAction")
    la = ET.SubElement(pa, "LongitudinalAction")
    sa = ET.SubElement(la, "SpeedAction")
    ET.SubElement(sa, "SpeedActionDynamics", dynamicsShape="linear",
                  value="2.0", dynamicsDimension="rate")
    tgt = ET.SubElement(sa, "SpeedActionTarget")
    ET.SubElement(tgt, "AbsoluteTargetSpeed", value=value)


def build_openscenario(desc: dict) -> tuple[Any, int]:
    """One G3 description -> (OpenSCENARIO 1.2 ElementTree, count of lossy categories).

    The count is RETURNED, never stashed on the root element. The first version set it as
    an attribute and popped it in the writer, which works exactly as long as nobody calls
    this function directly -- and then emits `_n_unmapped_categories="0"` into a file that
    claims to be OpenSCENARIO. A private channel through the artifact itself is the kind of
    thing that survives review and fails in someone else's hands.

    The ranges become `ParameterDeclarations`, which is what makes the file LOGICAL in
    ASAM's sense rather than a single concrete run: a downstream
    `ParameterValueDistribution` samples them. Writing the midpoint instead would silently
    demote the artifact to a concrete scenario, the same mistake D-051 rejected for the
    JSON.
    """
    import xml.etree.ElementTree as ET

    sc, sg, pr = desc["scene"], desc["scenography"], desc["parameters"]
    root = ET.Element("OpenSCENARIO")
    ET.SubElement(root, "FileHeader", revMajor=str(XOSC_REV[0]), revMinor=str(XOSC_REV[1]),
                  date="2026-09-20T00:00:00", author="nuScenes auto-labeling pipeline (G5)",
                  description=(f"{sc['name']} @ {sc['location']}, {sc['duration_s']} s, "
                               f"lighting={sg['lighting']}. LOGICAL scenario: every "
                               f"parameter is a range. Derived from sensor data with no "
                               f"human annotation (schema {desc['schema_version']})."))

    params = ET.SubElement(root, "ParameterDeclarations")
    def _p(name, value, ptype="double"):
        ET.SubElement(params, "ParameterDeclaration", name=name,
                      parameterType=ptype, value=str(value))
    for key in ("ego_speed_mps", "pedestrian_speed_mps"):
        rng = pr.get(key)
        if rng:
            _p(f"{key}_min", rng[0]); _p(f"{key}_max", rng[1])
    _p("road_extent_m", pr.get("road_extent_m", 0.0))
    _p("lighting", sg["lighting"], "string")
    _p("scene_duration_s", sc["duration_s"])

    ET.SubElement(root, "CatalogLocations")
    rn = ET.SubElement(root, "RoadNetwork")
    # DELIBERATELY EMPTY. nuScenes has no OpenDRIVE; a LogicFile here would name a file
    # that does not exist and make the scenario look executable when it is not.
    rn.append(ET.Comment(
        " No LogicFile: nuScenes ships polygon map layers in its own JSON format, not "
        "OpenDRIVE. This scenario is NOT executable until a .xodr for "
        f"'{sc['location']}' is supplied. "))

    ents = ET.SubElement(root, "Entities")
    ego = ET.SubElement(ents, "ScenarioObject", name="Ego")
    veh = ET.SubElement(ego, "Vehicle", name="ego_vehicle", vehicleCategory="car")
    ET.SubElement(veh, "BoundingBox").append(ET.Comment(" nuScenes ego: Renault Zoe "))
    ET.SubElement(veh, "Performance", maxSpeed="30.0", maxDeceleration="9.0",
                  maxAcceleration="5.0")
    ax = ET.SubElement(veh, "Axles")
    for tag in ("FrontAxle", "RearAxle"):
        ET.SubElement(ax, tag, maxSteering="0.5", wheelDiameter="0.6", trackWidth="1.8",
                      positionX="1.4" if tag == "FrontAxle" else "0.0", positionZ="0.3")

    n_unmapped = 0
    for actor in sg["actors"]:
        kind, osc_cat = OSC_CATEGORY.get(actor["category"], ("MiscObject", "obstacle"))
        if actor["category"] not in OSC_CATEGORY:
            n_unmapped += 1
        for i in range(actor["n_instances"]):
            name = f"{actor['category'].replace('.', '_')}_{i}"
            obj = ET.SubElement(ents, "ScenarioObject", name=name)
            attrs = {"name": name}
            if kind == "Vehicle":
                attrs["vehicleCategory"] = osc_cat
            elif kind == "Pedestrian":
                attrs.update(pedestrianCategory=osc_cat, mass="80.0", model="")
            else:
                attrs.update(miscObjectCategory=osc_cat, mass="10.0")
            el = ET.SubElement(obj, kind, **attrs)
            ET.SubElement(el, "BoundingBox")
            if kind == "Vehicle":
                ET.SubElement(el, "Performance", maxSpeed="30.0", maxDeceleration="9.0",
                              maxAcceleration="5.0")
                a2 = ET.SubElement(el, "Axles")
                for tag in ("FrontAxle", "RearAxle"):
                    ET.SubElement(a2, tag, maxSteering="0.5", wheelDiameter="0.6",
                                  trackWidth="1.8", positionX="1.4", positionZ="0.3")
            # PROVENANCE, on every entity. The OSC enum is lossy -- `vehicle.construction`
            # becomes a truck -- so the nuScenes name and the observed ranges ride along as
            # properties and the translation stays auditable (R16's habit: keep the source
            # key visible).
            props = ET.SubElement(el, "Properties")
            ET.SubElement(props, "Property", name="nuscenes_category", value=actor["category"])
            # 42 of 5,668 actor rows (0.74%) carry NO speed range: an instance seen in a
            # single keyframe has no velocity to difference. Written as "unknown" rather
            # than omitted, because an absent property reads as an oversight while an
            # explicit unknown is a measurement -- the same reason `coerce_tags` keeps
            # "did not answer" distinct from "no" (F-062).
            for key in ("range_m", "speed_mps"):
                rng = actor.get(key)
                ET.SubElement(props, "Property", name=key,
                              value=f"{rng[0]}..{rng[1]}" if rng else "unknown")
            ET.SubElement(props, "Property", name="observed_in_front_camera",
                          value=str(i < actor["n_in_camera"]).lower())

    sb = ET.SubElement(root, "Storyboard")
    init = ET.SubElement(sb, "Init")
    acts0 = ET.SubElement(init, "Actions")
    priv = ET.SubElement(acts0, "Private", entityRef="Ego")
    _speed_action(priv, "$ego_speed_mps_min" if pr.get("ego_speed_mps") else "0.0")
    priv.append(ET.Comment(
        " No TeleportAction: placing the ego needs a road network (see RoadNetwork). "))

    story = ET.SubElement(sb, "Story", name=f"{sc['name']}_storyboard")
    act = ET.SubElement(story, "Act", name="ego_manoeuvres")
    for panel in desc["timeline"]:
        mg = ET.SubElement(act, "ManeuverGroup", maximumExecutionCount="1",
                           name=f"panel_{panel['panel']}")
        actors_el = ET.SubElement(mg, "Actors", selectTriggeringEntities="false")
        ET.SubElement(actors_el, "EntityRef", entityRef="Ego")
        man = ET.SubElement(mg, "Maneuver", name=f"panel_{panel['panel']}_maneuver")
        evt = ET.SubElement(man, "Event", name=f"panel_{panel['panel']}_event",
                            priority="parallel")
        a = ET.SubElement(evt, "Action", name=f"panel_{panel['panel']}_action")
        lon = _LON_ACTION.get(panel["speed"])
        if lon:
            _speed_action(a, "$ego_speed_mps_max" if lon == "increase"
                          else "$ego_speed_mps_min")
        else:
            # A turn or a lane change needs a target lane, and there is no road network to
            # name one. The manoeuvre is carried as a user-defined action rather than
            # fabricated as a LaneChangeAction against a lane that does not exist.
            uda = ET.SubElement(a, "UserDefinedAction")
            ET.SubElement(uda, "CustomCommandAction", type="nuscenes_manoeuvre").text = (
                f"steering={panel['steering']} speed={panel['speed']} "
                f"lane_change={str(panel['lane_change']).lower()}")
        st = ET.SubElement(evt, "StartTrigger")
        cg = ET.SubElement(st, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", name=f"panel_{panel['panel']}_start",
                             delay="0", conditionEdge="rising")
        bvc = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(bvc, "SimulationTimeCondition",
                      value=str(panel["t_s"][0]), rule="greaterThan")
    # THE ACT NEEDS ITS OWN START TRIGGER. Without one an Act may never be entered, so
    # every panel below it would be unreachable and the story would play nothing -- a file
    # that parses, validates and does nothing, which is the worst of the three outcomes.
    ast = ET.SubElement(act, "StartTrigger")
    acg = ET.SubElement(ast, "ConditionGroup")
    acond = ET.SubElement(acg, "Condition", name="act_start", delay="0",
                          conditionEdge="rising")
    abvc = ET.SubElement(acond, "ByValueCondition")
    ET.SubElement(abvc, "SimulationTimeCondition", value="0", rule="greaterThan")

    stt = ET.SubElement(sb, "StopTrigger")
    cg = ET.SubElement(stt, "ConditionGroup")
    cond = ET.SubElement(cg, "Condition", name="scene_end", delay="0",
                         conditionEdge="rising")
    bvc = ET.SubElement(cond, "ByValueCondition")
    ET.SubElement(bvc, "SimulationTimeCondition", value=str(sc["duration_s"]),
                  rule="greaterThan")

    return ET.ElementTree(root), n_unmapped


def export_openscenario(path: str = "outputs/scenario_descriptions.jsonl",
                        out_dir: str = XOSC_DIR, limit: int | None = None) -> dict:
    """Write one .xosc per scenario description. Step G5.

    Into a SUBDIRECTORY, not `outputs/` itself: the R30 guards glob `outputs/*.png` and
    `outputs/*.csv`, and 850 new files beside them would drown the artifacts a reader is
    meant to find.
    """
    import xml.etree.ElementTree as ET

    d = ROOT / out_dir
    d.mkdir(parents=True, exist_ok=True)
    written, unmapped = 0, 0
    for line in (ROOT / path).open():
        desc = json.loads(line)
        tree, n_unmapped = build_openscenario(desc)
        unmapped += n_unmapped
        ET.indent(tree, space="  ")
        tree.write(d / f"{desc['scene']['name']}.xosc",
                   encoding="UTF-8", xml_declaration=True)
        written += 1
        if limit and written >= limit:
            break
    return {"dir": str(d), "n_written": written,
            "n_actor_categories_without_an_osc_enum": unmapped,
            "executable": False,
            "why_not": ("nuScenes ships no OpenDRIVE road network, so <RoadNetwork> is "
                        "empty and no entity is placed. The file is a valid OpenSCENARIO "
                        "PARAMETER and STORY description, not a runnable scenario."),
            "validated_against_xsd": validate_xosc(d / f"{desc['scene']['name']}.xosc")}


def validate_xosc(path: Path | str) -> dict:
    """Check one .xosc. Well-formedness always; the ASAM XSD only if it is installed.

    A claim of OpenSCENARIO conformance with nothing behind it is exactly the kind of
    unevidenced artifact R30 exists to stop, so this reports WHICH check ran rather than
    returning a bare True.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    required = ["FileHeader", "Entities", "Storyboard"]
    present = [c.tag for c in root]
    structural = (root.tag == "OpenSCENARIO" and all(r in present for r in required)
                  and present.index("Entities") < present.index("Storyboard"))
    out = {"well_formed": True, "structural": structural,
           "xsd": "not checked: pip install scenariogeneration brings the ASAM schema"}
    try:                                    # ponytail: optional, one install upgrades it
        import xmlschema  # noqa: F401
        from scenariogeneration import xosc  # noqa: F401
        from pathlib import Path as _P
        xsd = _P(xosc.__file__).parent / "xsd" / "OpenSCENARIO.xsd"
        if xsd.exists():
            import xmlschema as _xs
            out["xsd"] = "valid" if _xs.XMLSchema(str(xsd)).is_valid(str(path)) else "INVALID"
    except ImportError:
        pass
    return out
