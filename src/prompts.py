"""Every prompt, versioned, in one place — Step E3.

Why this module exists: in `docs/nuscens.py`, STRUCTURED_PROMPT is defined twice
(lines 215 and 869) with different text, and the second silently overwrites the
first. It is therefore ambiguous which prompt produced which published number.
Never define a prompt anywhere else, and always log `prompt_version` on every
result row.

THE TAG LIST IS GENERATED, NEVER TYPED. `build_prompt()` fills the `{tags}` slot
from `outputs/label_schema.json`, so the prompt cannot drift from the ground truth
it is scored against. R4, and the reason F-016 happened: a vocabulary written by
hand in two places diverged, leaving `is_turning_left` False on all 5,500 turning
frames. A prompt that asks for a tag the schema no longer has is the same defect
pointed at the model instead of the table.

Each constant below is the STATIC instruction text. The tag block is appended at
build time, so `REGISTRY` still maps a version name to a stable string and
`prompt_version` remains a meaningful column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "label_schema.json"

# The instruction every condition shares, so differences between conditions are only
# ever about the INPUT, never about how the answer is requested. If the ask differed
# between arms, RQ3a would be measuring prompt wording.
_ANSWER_RULES = """Answer with ONE JSON object and nothing else. Every field takes true or false.

Rules:
- Answer every field. If you are unsure, give your best judgement rather than omitting it;
  a missing field is recorded as unanswered, not as false.
- The fields are grouped by WHAT THEY ARE ABOUT. The groups ask about different regions of
  space, and a fact that is true for one group can be false for another. Read the group
  descriptions before answering.
- Keep the group structure of the JSON exactly as shown."""

PROMPT_V1_STRUCTURED = """You are labelling one frame of a driving scene, recorded from a camera
mounted on the front of a car ("the ego vehicle"). You see what its driver sees.

""" + _ANSWER_RULES

# E8's reasoning arm needs its OWN answer rules. The shared block ends with "Answer with
# ONE JSON object and nothing else", which directly contradicts "think first" - and the
# model obeyed the prohibition: measured over 628 frames, 0 produced any text before the
# JSON (F-065). The condition tested a prompt the model ignored, exactly like the original
# query_multiframe claiming five images while sending one.
_ANSWER_RULES_COT = """Write your reasoning first, then the answer.

Rules:
- Begin with a SHORT paragraph of reasoning. Do not skip it.
- After the paragraph, give ONE JSON object as the LAST thing in your reply.
- Every field takes true or false. Answer every field; if you are unsure, give your best
  judgement rather than omitting it, because a missing field is recorded as unanswered,
  not as false.
- The fields are grouped by WHAT THEY ARE ABOUT. The groups ask about different regions of
  space, and a fact that is true for one group can be false for another. Read the group
  descriptions before answering.
- Keep the group structure of the JSON exactly as shown."""

PROMPT_V2_COT = """You are labelling one frame of a driving scene, recorded from a camera
mounted on the front of a car ("the ego vehicle"). You see what its driver sees.

First think step by step in a short paragraph: what kind of road is this, what is the ego
doing, what road users are present and what are they doing. Then give the answer.

""" + _ANSWER_RULES_COT

# The colour legend MUST match src.bev.BEV_COLORS exactly. `tests/test_prompts.py` asserts
# it, because a legend that drifts from the renderer makes this prompt silently untrue and
# every BEV number meaningless.
PROMPT_V3_BEV = """You are labelling one frame of a driving scene from a BIRD'S-EYE VIEW drawn
from above. This is a diagram, not a photograph.

How to read it:
- The ego vehicle is the RED arrow-shaped marker at the exact centre. It always points UP:
  the top of the image is straight ahead, the bottom is behind, left is left.
- The image covers 100 metres in every direction, so it is 200 m across.
- Two dashed ORANGE lines mark the front camera's field of view (64 degrees wide). Objects
  outside those lines are real, but a forward-facing camera could not see them.
- Road surface: WHITE = drivable area, PALE BLUE = marked lane, MID BLUE = road segment,
  PALE YELLOW = stop line, LIGHT BLUE = pedestrian crossing, GREEN = walkway,
  PINK = car park. FLAT GREY is not road at all - pavement, buildings, or a traffic island.
- Objects, drawn as filled shapes: BLUE box = car or van, DARK BLUE box = bus, truck or
  other large vehicle. Boxes are drawn to scale and pointed the way the object faces.
- Small objects are drawn as fixed-size DOTS, not to scale, or they would be a single pixel:
  GREEN dot = pedestrian, PURPLE dot = cyclist or motorcyclist, ORANGE dot = barrier,
  RED-ORANGE dot = traffic cone, GREY dot = an object of some other kind.
- A short BLACK ARROW from an object means it is moving, and points where it is going.
  No arrow means it is stationary.

""" + _ANSWER_RULES

# The REPAIRED BEV arms (F-063c, F-064b). New names rather than edits, because the old
# prompts produced published numbers and a prompt that changes under a name that does not
# is unauditable; `outputs/results/v3_bev.jsonl` and `v3b_bev_symbolic.jsonl` stay exactly
# what they were, and the two runs can be compared.
#
# The addition is one legend line. The raster now draws the traffic-light fixture layer the
# tag derives from, so the legend has to name its glyph, and nothing else about the picture
# changed.
_BEV_TRAFFIC_LIGHT_LEGEND = """- TEAL SQUARES mark traffic-light FIXTURES: where a traffic light stands. They are larger
  than the dots above so they are not mistaken for one. The diagram shows only WHERE a
  fixture is; it never shows which colour is lit.
"""

PROMPT_V3_BEV_R2 = PROMPT_V3_BEV.replace(
    "  No arrow means it is stationary.\n",
    "  No arrow means it is stationary.\n" + _BEV_TRAFFIC_LIGHT_LEGEND)

PROMPT_V3B_BEV_SYMBOLIC = """You are labelling one frame of a driving scene. Instead of a picture
you are given a written description of the same scene, listing every object around the ego
vehicle with its distance, bearing and motion.

Bearings are degrees from straight ahead: 0 is directly in front, positive is to the LEFT,
negative is to the RIGHT, and +/-180 is directly behind. Distances are in metres.

""" + _ANSWER_RULES

# The repaired text arm. The description now states WHERE each road surface is and how far
# it reaches, and lists traffic-light fixtures, so the reader has to be told how to read
# those two blocks. Before this, the description named the surfaces and gave no geometry at
# all, which is why three tags scored 0.00 in this arm (F-064b).
PROMPT_V3B_BEV_SYMBOLIC_R2 = """You are labelling one frame of a driving scene. Instead of a picture
you are given a written description of the same scene: every object around the ego vehicle
with its distance, bearing and motion, the road surfaces around it, and any traffic-light
fixtures.

Bearings are degrees from straight ahead: 0 is directly in front, positive is to the LEFT,
negative is to the RIGHT, and +/-180 is directly behind. Distances are in metres.

The ROAD SURFACES block says, for each kind of surface, whether the vehicle is standing on
it or how far away the nearest piece of it is, and how far that surface is still present
ahead, to each side and behind. A surface reaching far to the left and right as well as
ahead is a different shape of road from one that only reaches ahead.

The TRAFFIC LIGHT FIXTURES block gives the position of each fixture. It says only where a
fixture stands, never which colour is lit.

""" + _ANSWER_RULES

PROMPT_V4_BOTH = """You are labelling one frame of a driving scene. You are given TWO images of
the same instant:

1. The FRONT CAMERA view - what the driver sees.
2. A BIRD'S-EYE VIEW drawn from above, covering 100 m in every direction.

Read the bird's-eye view as follows:
- The ego vehicle is the RED arrow at the centre, always pointing UP.
- Two dashed ORANGE lines mark what the front camera can see.
- WHITE = drivable area, PALE BLUE = lane, MID BLUE = road segment, PALE YELLOW = stop line,
  LIGHT BLUE = pedestrian crossing, GREEN = walkway, PINK = car park, FLAT GREY = not road.
- BLUE box = car or van, DARK BLUE box = large vehicle, drawn to scale.
- GREEN dot = pedestrian, PURPLE dot = cyclist, ORANGE dot = barrier, RED-ORANGE dot = cone,
  GREY dot = other. Dots are fixed-size, not to scale.
- A BLACK ARROW means the object is moving, and points where it is going.

Use the camera for appearance and the bird's-eye view for geometry and for what lies outside
the camera's field of view.

""" + _ANSWER_RULES

# Step E9. The old query_multiframe (docs/nuscens.py:649) sent ONE image while the prompt
# claimed five, so the model never saw a sequence - which is why multi-frame "did not help".
PROMPT_V6_TEMPORAL = """You are labelling a driving scene from a SEQUENCE of {n_frames} front-camera
frames, in order, half a second apart. The LAST frame is the moment being labelled; the
earlier frames are there so you can see how the situation is changing.

Use the sequence to judge anything that involves motion or change over time - whether the ego
is speeding up, slowing down, turning or changing lane, and whether another road user is
moving into the ego's path or braking. Judge everything else from the last frame.

""" + _ANSWER_RULES

# Step E8, the agentic decomposition. One focused question per tag rather than one large
# JSON. Filled from the schema by `build_yesno_prompts()`.
PROMPT_V5_YESNO: dict[str, str] = {}

_YESNO_TEMPLATE = """Look at this driving scene, recorded from a camera on the front of a car.

Question: {question}

{scope_note}

Answer with exactly one word: yes or no."""

REGISTRY: dict[str, str | dict[str, str]] = {
    "v1_structured":     PROMPT_V1_STRUCTURED,
    "v2_cot":            PROMPT_V2_COT,
    "v3_bev":            PROMPT_V3_BEV,
    "v3b_bev_symbolic":  PROMPT_V3B_BEV_SYMBOLIC,
    "v3_bev_r2":         PROMPT_V3_BEV_R2,
    "v3b_bev_symbolic_r2": PROMPT_V3B_BEV_SYMBOLIC_R2,
    "v4_both":           PROMPT_V4_BOTH,
    "v5_yesno":          PROMPT_V5_YESNO,
    "v6_temporal":       PROMPT_V6_TEMPORAL,
}


# --- What each tag means, in words a model can act on ------------------------------------
# The schema's `derivation` field says how the label is COMPUTED - "a ped_crossing polygon
# intersects the forward corridor". That is written for auditability (D-001) and is the
# right text for the thesis, but as a prompt it would measure whether the model can parse
# our jargon rather than whether it can see a crossing. A perceptual phrasing cannot be
# derived mechanically from a computational one, so these are written once, here.
#
# R4 still applies, and is enforced rather than trusted: `tests/test_prompts.py` asserts
# these keys are EXACTLY the schema's scoreable tags, so adding tag 38 or renaming one
# fails loudly instead of silently dropping a question.
# The JSON group keys the model is asked to emit. The schema's own scope names are
# internal ("camera_frustum_CAM_FRONT"), and a model should not have to know what a
# frustum is to answer. `vlm.coerce_tags` flattens the groups before matching, so these
# keys are presentation only and can be readable without affecting scoring.
SCOPE_JSON_KEYS: dict[str, str] = {
    "ego": "ego_motion",
    "map_containment": "where_the_car_is",
    "map_corridor_30m": "ahead_within_30m",
    "camera_frustum_CAM_FRONT": "visible_in_camera",
    "world_360": "around_the_car",
}

SCOPE_DESCRIPTIONS: dict[str, str] = {
    # The schema's own `scopes` text is written for the audit reader and uses the
    # pipeline's vocabulary - "only what the CAM_FRONT frustum contains". Same problem as
    # the derivations: correct for the thesis, unusable as a prompt. Written once here,
    # with a test asserting these keys match the schema's exactly.
    "ego": "how the car carrying the camera is moving",
    "map_containment": "what the car is physically on at this moment",
    "map_corridor_30m": "what lies within about 30 metres ahead, along the car's path",
    "camera_frustum_CAM_FRONT": "what is inside the front camera's field of view",
    "world_360": "what is around the car in every direction, whether or not a camera "
                 "can see it",
}

TAG_QUESTIONS: dict[str, str] = {
    # ego - how the car carrying the camera is moving
    "is_going_straight": "the car is driving straight ahead rather than turning",
    "is_cruising": "the car is holding a roughly steady speed, neither speeding up nor slowing down",
    "is_stationary": "the car is stopped, or creeping no faster than walking pace",
    "is_decelerating": "the car is slowing down",
    "is_accelerating": "the car is speeding up",
    "is_turn_right": "the car is turning right",
    "is_turn_left": "the car is turning left",
    "lane_change": "the car is moving out of its lane into an adjacent one",
    "is_u_turn": "the car is making a U-turn",
    # map_containment - what the car is physically on RIGHT NOW
    "on_drivable_area": "the car is on a surface vehicles are meant to drive on",
    "on_lane": "the car is inside a marked traffic lane",
    "at_intersection": "the car is inside a junction at this moment, not merely approaching one",
    "on_stop_line": "the car is standing on a stop line",
    "on_ped_crossing": "the car is standing on a pedestrian crossing",
    # map_corridor_30m - what lies within 30 m ahead, along the car's path
    "intersection_ahead": "a junction lies within about 30 metres ahead",
    "stop_line_ahead": "a stop line lies within about 30 metres ahead",
    "ped_crossing_ahead": "a pedestrian crossing lies within about 30 metres ahead",
    "traffic_light_ahead": "a traffic light lies within about 30 metres ahead",
    # camera_frustum_CAM_FRONT - what is in the front camera's field of view
    "has_vehicle": "at least one vehicle is visible",
    "moving_vehicle": "at least one visible vehicle is moving",
    "parked_vehicle": "at least one visible vehicle is parked",
    "large_vehicle": "at least one visible vehicle is a bus, truck, or other large vehicle",
    "has_pedestrian": "at least one pedestrian is visible",
    "vehicle_near": "a vehicle is visible within about 15 metres",
    "stopped_vehicle": "at least one visible vehicle is halted but not parked, such as waiting in traffic",
    "construction_object": "roadworks equipment is visible, such as cones or barriers",
    "traffic_cone": "at least one traffic cone is visible",
    "barrier": "at least one barrier is visible",
    "pedestrian_near": "a pedestrian is visible within about 15 metres",
    "parked_bicycle": "a bicycle with nobody riding it is visible",
    "cyclist": "somebody is riding a bicycle or motorcycle",
    # world_360 - facts about the surroundings, whether or not a camera sees them
    "lead_vehicle": "there is a vehicle directly ahead of the car, in its own lane",
    "pedestrian_crossing_path": "a pedestrian is walking into the car's path",
    "cut_in": "another vehicle is moving sideways into the car's lane ahead of it",
    "lead_braking": "the vehicle directly ahead of the car is braking",
}


def load_schema(path: Path | str = SCHEMA_PATH) -> dict[str, Any]:
    """The frozen label schema. The single source of the tag vocabulary."""
    return json.loads(Path(path).read_text())


def scoreable_tags(schema: dict[str, Any]) -> list[str]:
    """The 35 tags Stage F scores. `qc_invariant` tags are excluded per D-032 —
    asking a model for a tag with zero positives yields an undefined recall."""
    return [t for t, v in schema["tags"].items() if v["role"] == "scored"]


def tag_block(schema: dict[str, Any]) -> str:
    """The JSON skeleton and the group descriptions, GENERATED from the schema (R4).

    Grouped by scope rather than by family, because scope is what a model can get wrong
    while being right: `on_ped_crossing` is true on 2.9% of frames and `ped_crossing_ahead`
    on 37.9%, and a model reporting the crossing it can plainly see 20 m away is correct
    about the world and wrong about the question unless the prompt says which is being
    asked (D-017, D-030).
    """
    by_scope: dict[str, list[str]] = {}
    for tag in scoreable_tags(schema):
        by_scope.setdefault(schema["tags"][tag]["scope"], []).append(tag)

    lines = ["", "The groups:"]
    for scope in by_scope:
        lines.append(f"  {SCOPE_JSON_KEYS[scope]}: {SCOPE_DESCRIPTIONS[scope]}")
    lines += ["", "Answer in exactly this shape:", "{"]
    scopes = list(by_scope)
    for si, scope in enumerate(scopes):
        lines.append(f'  "{SCOPE_JSON_KEYS[scope]}": {{')
        tags = by_scope[scope]
        for ti, tag in enumerate(tags):
            comma = "," if ti < len(tags) - 1 else ""
            lines.append(f'    "{tag}": true|false{comma}'
                         f'   // {TAG_QUESTIONS[tag]}')
        lines.append("  }" + ("," if si < len(scopes) - 1 else ""))
    lines.append("}")
    return "\n".join(lines)


def build_prompt(version: str, schema: dict[str, Any] | None = None, **fmt: Any) -> str:
    """The full prompt text for one condition: static instructions + generated tags.

    `version` becomes the `prompt_version` column on every result row, so the string
    that produced a number is always recoverable — the ambiguity this module exists to
    remove.
    """
    if version not in REGISTRY:
        raise KeyError(f"unknown prompt version {version!r}; have {sorted(REGISTRY)}")
    if version == "v5_yesno":
        raise ValueError("v5_yesno is per-tag; call build_yesno_prompts() instead")
    schema = schema if schema is not None else load_schema()
    head = REGISTRY[version]
    assert isinstance(head, str)
    return (head.format(**fmt) if fmt else head) + "\n" + tag_block(schema)


def build_yesno_prompts(schema: dict[str, Any] | None = None) -> dict[str, str]:
    """One focused yes/no question per tag. Step E8.

    DriveLM (REFERENCES.md R-17) is the evidence for trying this at all: multi-step
    decomposition beat single-round VQA there. Here it costs 35 model calls per frame
    against one, so E8 runs it on a 300-frame slice to protect the T4 quota.
    """
    schema = schema if schema is not None else load_schema()
    out = {}
    for tag in scoreable_tags(schema):
        spec = schema["tags"][tag]
        out[tag] = _YESNO_TEMPLATE.format(
            question=f"Is it true that {TAG_QUESTIONS[tag]}?",
            scope_note=f"This question is about {SCOPE_DESCRIPTIONS[spec['scope']]}.")
    PROMPT_V5_YESNO.clear()
    PROMPT_V5_YESNO.update(out)
    return out
