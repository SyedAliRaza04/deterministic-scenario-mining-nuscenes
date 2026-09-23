"""Signage — Step H2, serving RQ4.

WHAT NUSCENES CAN AND CANNOT SUPPORT. It has no traffic-sign or traffic-light ANNOTATION:
signage exists only as a map layer, 634 records across the four maps, each carrying a pose.
So "a traffic light is in view" is derivable by projecting those poses into the camera;
"the light is red" is not derivable at all (F-004).

THE TRAP THAT MAKES THAT WORTH SAYING TWICE. Those map records DO carry an `items[].color`
field with values RED, YELLOW and GREEN, which reads exactly like signal state. It is not:
626 of the 634 records carry all three at once, which no lit signal can be, and no record has
a time dimension. The field enumerates the LAMPS ON THE FIXTURE. Ground truth built from it
would look principled and be meaningless (F-076).

WHY THIS IS NOT C3's TAG UNDER A NEW NAME (R21, measured before building). C3 already has
`traffic_light_ahead`: a light within a 30 m corridor along the ego heading. Sampled over
4,000 keyframes, the frustum test and the corridor test agree on only **89.65% of frames with
a Jaccard of 0.549** at comparable range — 179 frames have a corridor light with nothing in
the image, 235 have a light in the image with nothing in the corridor. They measure different
things: "is a light ahead of me" versus "would a light appear in this photograph". The second
is the one a VLM scored on a camera frame can actually answer.

Build:  ./venv/bin/python -c "import src.signage as S; S.build_traffic_light_table()"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MAPS = ROOT / "data/nuscenes/maps/expansion"

# --- parameters (one place only, each with the measurement that chose it) ---------------

# Range beyond which a projected light is in the frustum but not perceivable.
#
# THIS CAP IS NOT OPTIONAL, and the uncapped version is the reason. A map traffic_light is a
# POINT: it has no extent to filter on and the map has no occlusion, so an uncapped frustum
# test fires on 66.38% of keyframes with a median "visible" distance of 115 m, a p90 of 520 m
# and a maximum of 3,351 m. A light 3.3 km away is behind buildings and is a fraction of a
# pixel. That is the inverse of F-030, where the question was whether far 3D BOXES were too
# small to see and the measured answer was no: boxes carry an extent, map points do not.
#
# The cap is derived, not chosen. The map states its own fixture geometry - `items[].rel_pos`
# gives a lamp stack of median 0.61 m - and CAM_FRONT's focal length is 1266.4 px, so
# apparent height is 1266.4 * 0.61 / d:
#
#     distance   15 m   30 m   50 m   75 m   115 m   250 m
#     pixels     51.5   25.7   15.4   10.3    6.7     3.1
#
# F-030 measured that only 0.3% of frustum-visible boxes fall under 15 px tall and none under
# 64 px^2, so 15 px is this project's own perceptibility floor rather than a fresh number.
# 0.61 m subtends 15 px at 51 m; 50 m is that bound rounded to the measurement's precision.
TRAFFIC_LIGHT_MAX_M = 50.0

# Behind the camera. Not 0.0: a light exactly on the image plane projects to infinity.
_MIN_DEPTH_M = 0.1


def traffic_light_poses(location: str) -> Any:
    """(N, 3) global positions of every mapped traffic light on one map.

    X AND Y COME FROM THE LINE GEOMETRY, NOT FROM `pose`. The obvious field is `pose`, which
    carries tx/ty/tz and reads as the fixture's position — and on TWO of the four maps its
    tx and ty are **zero-filled**: all 119 singapore-hollandvillage records and all 81
    singapore-queenstown records sit at (0, 0), 200 of 634 records (31.5%). Using it put
    every light on those maps ~2 km from the ego and produced a silent **0.00%** hit rate on
    exactly the maps where C3's corridor tag fires 19.70% and 6.42%.

    That is R16 and F-019's failure class again — the plausible key is the wrong key, and it
    fails by returning nothing rather than by raising. `line_token` resolves through `node`
    on all four maps (307/127/119/81, all with sane coordinates).

    `pose.tz` IS valid everywhere (2.20-8.61 m), so only the horizontal position is
    corrupted and the height needs no substitution.
    """
    import numpy as np

    m = json.loads((MAPS / f"{location}.json").read_text())
    recs = m.get("traffic_light", [])
    if not recs:
        return np.zeros((0, 3))
    nodes = {n["token"]: (n["x"], n["y"]) for n in m["node"]}
    lines = {ln["token"]: ln for ln in m.get("line", [])}

    out = []
    for r in recs:
        ln = lines.get(r.get("line_token"))
        xy = [nodes[t] for t in ln["node_tokens"] if t in nodes] if ln else []
        if not xy:
            continue                      # no geometry at all: dropped, not placed at (0,0)
        cx, cy = np.mean(xy, axis=0)
        out.append([float(cx), float(cy), float(r["pose"]["tz"])])
    return np.array(out) if out else np.zeros((0, 3))


def project_lights(nusc: Any, sample_token: str, poses: Any) -> tuple:
    """(uv, distances) for every mapped light that falls on this keyframe's image.

    Separated from the tag so the FIGURE draws exactly the quantity the tag is computed
    from. A verification picture that recomputes its own geometry can agree with a broken
    detector, which is how a 0.00% hit rate survived inspection once already (F-019).
    """
    import numpy as np
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    if not len(poses):
        return np.zeros((2, 0)), np.zeros(0)
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    p = poses.T - np.array(ego["translation"]).reshape(3, 1)
    p = Quaternion(ego["rotation"]).inverse.rotation_matrix @ p
    p = p - np.array(cs["translation"]).reshape(3, 1)
    p = Quaternion(cs["rotation"]).inverse.rotation_matrix @ p
    front = p[2, :] > _MIN_DEPTH_M
    if not front.any():
        return np.zeros((2, 0)), np.zeros(0)
    uv = view_points(p[:, front], np.array(cs["camera_intrinsic"]), normalize=True)
    on = (uv[0] >= 0) & (uv[0] < sd["width"]) & (uv[1] >= 0) & (uv[1] < sd["height"])
    idx = np.where(front)[0][on]
    return uv[:2, on], np.linalg.norm(p[:, idx], axis=0)


def lights_in_view(nusc: Any, sample_token: str, poses: Any,
                   max_m: float = TRAFFIC_LIGHT_MAX_M) -> dict[str, Any]:
    """Which mapped lights project into this keyframe's CAM_FRONT image. Step H2."""
    import numpy as np
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    if not len(poses):
        return {"n_in_view": 0, "nearest_m": float("nan"), "n_in_frustum_uncapped": 0}

    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    # Global -> ego -> camera. Both rotations are inverses: we are moving the POINT into
    # the sensor frame, not the sensor into the world (R40's bookkeeping, one frame up).
    p = poses.T - np.array(ego["translation"]).reshape(3, 1)
    p = Quaternion(ego["rotation"]).inverse.rotation_matrix @ p
    p = p - np.array(cs["translation"]).reshape(3, 1)
    p = Quaternion(cs["rotation"]).inverse.rotation_matrix @ p

    front = p[2, :] > _MIN_DEPTH_M
    if not front.any():
        return {"n_in_view": 0, "nearest_m": float("nan"), "n_in_frustum_uncapped": 0}

    uv = view_points(p[:, front], np.array(cs["camera_intrinsic"]), normalize=True)
    on_image = (uv[0] >= 0) & (uv[0] < sd["width"]) & (uv[1] >= 0) & (uv[1] < sd["height"])
    idx = np.where(front)[0][on_image]
    if not len(idx):
        return {"n_in_view": 0, "nearest_m": float("nan"), "n_in_frustum_uncapped": 0}

    dist = np.linalg.norm(p[:, idx], axis=0)
    near = dist <= max_m
    return {
        "n_in_view": int(near.sum()),
        "nearest_m": float(dist.min()),
        # Provenance, not a label: how many were geometrically in the frustum before the
        # perceptibility cap. Keeping it means the cap's effect is auditable per frame
        # rather than being a number in a docstring.
        "n_in_frustum_uncapped": int(len(idx)),
    }


def build_traffic_light_table(dataroot: str = "data/nuscenes",
                              out_path: str = "outputs/h2_traffic_light.parquet") -> Any:
    """`traffic_light_in_view` for every keyframe. Step H2.

    NOT added to `outputs/label_schema.json`. The frozen schema has been reopened exactly
    once, for `lane_change`, and that was an explicit decision with the measurements in hand
    (D-033). This table stands on its own as the answer to "what signage IS verifiable in
    nuScenes"; promoting it to a scored tag is a separate call.
    """
    import pandas as pd

    from . import data as D

    nusc = D.load_nusc("v1.0-trainval", dataroot)
    gt = pd.read_parquet(OUT / "gt_all.parquet",
                         columns=["sample_token", "scene_name", "location", "keyframe_index"])
    cache = {loc: traffic_light_poses(loc) for loc in gt.location.unique()}

    rows = []
    for r in gt.itertuples():
        res = lights_in_view(nusc, r.sample_token, cache[r.location])
        rows.append({"sample_token": r.sample_token, "scene_name": r.scene_name,
                     "location": r.location, "keyframe_index": r.keyframe_index,
                     "traffic_light_in_view": res["n_in_view"] > 0, **res})
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / out_path, index=False)
    return df
