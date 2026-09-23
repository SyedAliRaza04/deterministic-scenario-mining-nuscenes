"""Step G4 -- can a scenario description from this pipeline be instantiated in a simulator?

PLAN G4 asks for evidence, not a thesis claim: load a generated description into
MetaDrive via ScenarioNet and see whether it instantiates. F-086 had recorded the step as
blocked; that diagnosis was wrong. Under Python 3.12 the resolver can only see
metadrive-simulator 0.2.6.0 (every later release declares `requires_python <3.12`), and
0.2.6.0 depends on the abandoned `gym`. Under Python 3.11, 0.4.3 installs from a pure
Python wheel and depends on `gymnasium`. The one real obstacle is that ScenarioNet still
imports `pkg_resources`, which setuptools removed at 81, so `./venv-sim` pins
setuptools 80.9.0.

RUNS UNDER ./venv-sim, never ./venv: MetaDrive pulls its own numpy and the main venv's
`numpy<2` pin is load-bearing for the devkit. Every simulator import is inside a function
so ./venv can still import this module (the appendix ledger parses it).

WHAT THE DESCRIPTION IS, AND WHY THAT DECIDES THE DESIGN. G3's description is an ABSTRACT
scenario: the road as statistics, the actors as categories with counts and ranges, and the
ego's behaviour as a timeline of manoeuvre labels. ScenarioNet needs a CONCRETE one: a
10 Hz state for every actor and the map as polylines. So instantiation cannot be "load the
description"; it has to be "derive the concrete scenario the description determines, and
measure how much of it that is". The ego's track is synthesised from the timeline and the
speed range alone. Everything the description does not carry -- the starting pose, the
map, every other actor -- is taken from ScenarioNet's own conversion of the same log, and
`build_g4_field_mapping` says so field by field rather than letting a replay of the log
pass for an instantiation of the description.
"""
from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SIM_DB = "data/sim/db/nuscenes_mini"          # ScenarioNet's conversion of v1.0-mini
SIM_DB_DESCRIBED = "data/sim/db/g4_described"  # the same, with the ego from OUR description
DESCRIPTIONS = "outputs/scenario_descriptions.jsonl"

# ScenarioNet's nuScenes converter writes metadata.sample_rate = 0.1, i.e. 10 Hz.
SIM_DT_S = 0.1

# The description records THAT a turn happened, never by how much: the timeline carries a
# steering label per panel and no angle. A nominal right-angle turn is therefore an
# assumption the description forces, not a parameter it supplies, and the heading error it
# causes is part of what G4 measures.
TURN_DEG = 90.0
U_TURN_DEG = 180.0

# A U-turn crosses the opposing carriageway, so its direction is set by which side traffic
# drives on (R32: the domain, not the geometry). Boston drives on the right, Singapore on
# the left. The description names the location, so this needs nothing it does not carry.
DRIVES_ON_RIGHT = {"boston-seaport": True, "singapore-onenorth": False,
                   "singapore-hollandvillage": False, "singapore-queenstown": False}

# The converted map gives street lanes as POLYGONS but junction lanes only as centrelines
# (LANE_SURFACE_UNSTRUCTURE carries a polyline and no polygon), so "on the road" means
# inside a street-lane polygon or within half a lane of a junction centreline. The width
# is checked, not trusted: the LOG ego is on the road by definition, so the log arm's
# on-road fraction is the ceiling this test can reach, and it is reported beside the
# described arm's.
JUNCTION_HALF_WIDTH_M = 2.0


def _descriptions() -> dict[str, dict]:
    return {r["scene"]["name"]: r
            for r in map(json.loads, (ROOT / DESCRIPTIONS).read_text().splitlines())}


def _scenario_files(db: str = SIM_DB) -> dict[str, Path]:
    """scene name -> the converted scenario pickle."""
    mapping = pickle.loads((ROOT / db / "dataset_mapping.pkl").read_bytes())
    out = {}
    for fname, sub in mapping.items():
        scene = fname.rsplit("_", 1)[-1].removesuffix(".pkl")
        out[scene] = ROOT / db / sub / fname
    return out


def described_ego_track(desc: dict, n_steps: int, x0: float, y0: float,
                        heading0: float, dt: float = SIM_DT_S) -> dict[str, Any]:
    """The ego track the description determines, integrated at the simulator's rate.

    Only the timeline and the speed range are read. The starting pose is an argument
    because the description does not carry one (see `build_g4_field_mapping`).

    Speed: `stationary` is 0; `accelerating` ramps linearly to the top of the range over
    the panel, `decelerating` to the bottom, `cruising` holds. The first panel's label sets
    the starting speed the same way. Heading: consecutive panels with the same steering
    label are ONE manoeuvre (a turn is often split across two panels by a speed change),
    and each turn turns by the nominal angle spread evenly over its span.
    """
    import numpy as np

    vmin, vmax = desc["parameters"]["ego_speed_mps"]
    panels = desc["timeline"]
    t = np.arange(n_steps) * dt

    first = panels[0]["speed"]
    v0 = {"stationary": 0.0, "accelerating": vmin, "decelerating": vmax}.get(
        first, (vmin + vmax) / 2)
    speed = np.full(n_steps, v0)
    cur = v0
    for p in panels:
        a, b = p["t_s"]
        tgt = {"stationary": 0.0, "accelerating": vmax, "decelerating": vmin}.get(
            p["speed"], cur)
        inside = (t >= a) & (t <= b)
        if inside.any() and b > a:
            speed[inside] = cur + (tgt - cur) * (t[inside] - a) / (b - a)
        speed[t > b] = tgt
        cur = tgt

    right_hand = DRIVES_ON_RIGHT.get(desc["scene"]["location"], True)
    events: list[list[Any]] = []
    for p in panels:
        if events and events[-1][0] == p["steering"]:
            events[-1][2] = p["t_s"][1]
        else:
            events.append([p["steering"], p["t_s"][0], p["t_s"][1]])
    yaw_rate = np.zeros(n_steps)
    for label, a, b in events:
        total = {"turn_left": TURN_DEG, "turn_right": -TURN_DEG,
                 "u_turn": U_TURN_DEG if right_hand else -U_TURN_DEG}.get(label, 0.0)
        if total and b > a:
            yaw_rate[(t >= a) & (t < b)] = math.radians(total) / (b - a)

    heading = heading0 + np.concatenate([[0.0], np.cumsum(yaw_rate[:-1] * dt)])
    vx, vy = speed * np.cos(heading), speed * np.sin(heading)
    x = x0 + np.concatenate([[0.0], np.cumsum(vx[:-1] * dt)])
    y = y0 + np.concatenate([[0.0], np.cumsum(vy[:-1] * dt)])
    return {"position": np.stack([x, y], 1), "heading": heading,
            "velocity": np.stack([vx, vy], 1), "speed": speed}


def _road_surface(sd: dict) -> Any:
    """The drivable surface of one converted scenario, in the scenario's own frame."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    parts = []
    for f in sd["map_features"].values():
        if f["type"] == "LANE_SURFACE_STREET" and "polygon" in f and len(f["polygon"]) >= 3:
            parts.append(Polygon(f["polygon"][:, :2]).buffer(0))
        elif f["type"] == "LANE_SURFACE_UNSTRUCTURE" and len(f.get("polyline", [])) >= 2:
            parts.append(LineString(f["polyline"][:, :2]).buffer(JUNCTION_HALF_WIDTH_M))
    return unary_union(parts)


def _on_road_fraction(surface: Any, xy: Any) -> float:
    from shapely import contains_xy

    return float(contains_xy(surface, xy[:, 0], xy[:, 1]).mean())


def _write_described_db(scenes: list[str]) -> dict[str, dict]:
    """A copy of the converted database with each ego track replaced by the described one.

    Returns scene -> {"log": ..., "described": ...} ego tracks, for the comparison.
    """
    import numpy as np

    src, dst = ROOT / SIM_DB, ROOT / SIM_DB_DESCRIBED
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    descs = _descriptions()
    files = _scenario_files(SIM_DB_DESCRIBED)
    tracks = {}
    for scene in scenes:
        path = files[scene]
        sd = pickle.loads(path.read_bytes())
        ego = sd["tracks"][sd["metadata"]["sdc_id"]]["state"]
        n = len(ego["heading"])
        log = {"position": np.array(ego["position"][:, :2], dtype=float),
               "heading": np.array(ego["heading"], dtype=float)}
        d = described_ego_track(descs[scene], n, log["position"][0, 0],
                                log["position"][0, 1], log["heading"][0])
        ego["position"][:, 0] = d["position"][:, 0]
        ego["position"][:, 1] = d["position"][:, 1]
        ego["heading"][:] = d["heading"]
        ego["velocity"][:, :2] = d["velocity"]
        ego["valid"][:] = True
        path.write_bytes(pickle.dumps(sd))
        surface = _road_surface(sd)
        tracks[scene] = {"log": log, "described": d,
                         "log_on_road": _on_road_fraction(surface, log["position"]),
                         "described_on_road": _on_road_fraction(surface, d["position"])}
    return tracks


def _replay(db: str, scenes: list[str]) -> dict[str, dict]:
    """Instantiate each scenario in MetaDrive and drive the ego along its stored track.

    Records what the SIMULATOR reports, not what was put in: the ego's simulated position
    and MetaDrive's own out-of-road and collision flags, which are the executability
    evidence a replay can give.
    """
    import numpy as np
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    files = _scenario_files(db)
    order = sorted(files)
    env = ScenarioEnv({"data_directory": str(ROOT / db), "num_scenarios": len(order),
                       "agent_policy": ReplayEgoCarPolicy, "use_render": False,
                       "reactive_traffic": False, "log_level": 50})
    out = {}
    try:
        for scene in scenes:
            env.reset(seed=order.index(scene))
            assert env.engine.data_manager.current_scenario["metadata"][
                "scenario_id"] == scene, "seed does not select the scene it was meant to"
            pos, crash = [], 0
            length = env.engine.data_manager.current_scenario_length
            # MetaDrive re-centres a scenario on load (the ego starts at the origin), so the
            # replay is checked against the track AS LOADED, never the raw pickle. Against
            # the raw one the "deviation" was 700 to 3,100 m of pure frame offset.
            cur = env.engine.data_manager.current_scenario
            loaded = np.asarray(cur["tracks"][cur["metadata"]["sdc_id"]]["state"]["position"],
                                dtype=float)[:, :2]
            for i in range(length - 1):
                _, _, term, trunc, info = env.step([0.0, 0.0])
                pos.append(np.array(env.agent.position, dtype=float))
                crash += bool(info.get("crash_vehicle") or info.get("crash_object"))
                if term or trunc:
                    break
            sim = np.array(pos)
            k = min(len(sim), len(loaded) - 1)
            follow = float(np.linalg.norm(sim[:k] - loaded[1:k + 1], axis=1).max()) if k else 0.0
            out[scene] = {"steps": len(pos), "steps_in_collision": crash,
                          "replay_max_deviation_m": follow}
    finally:
        env.close()
    return out


def build_g4_instantiation(out_csv: str = "outputs/g4_instantiation.csv") -> Any:
    """Instantiate every mini scene twice -- from the log and from our description -- and
    measure how far apart the two concrete scenarios are.

    Rows are scenes. The log arm is ScenarioNet's own conversion replayed as recorded; it
    establishes that the tool chain runs on this scene, which is all it establishes. The
    described arm replaces the ego's track with the one the description determines. The
    gap between the two is the part of the scene the description does not pin down.
    """
    import numpy as np
    import pandas as pd

    scenes = sorted(_scenario_files(SIM_DB))
    tracks = _write_described_db(scenes)
    log_run = _replay(SIM_DB, scenes)
    desc_run = _replay(SIM_DB_DESCRIBED, scenes)

    rows = []
    for scene in scenes:
        L, D = tracks[scene]["log"], tracks[scene]["described"]
        n = len(L["heading"])
        err = np.linalg.norm(L["position"] - D["position"][:n], axis=1)
        dh = (D["heading"][n - 1] - L["heading"][n - 1] + math.pi) % (2 * math.pi) - math.pi
        path = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())  # noqa: E731
        rows.append({
            "scene_name": scene,
            "location": _descriptions()[scene]["scene"]["location"],
            "duration_s": round(n * SIM_DT_S, 1),
            "log_path_m": round(path(L["position"]), 1),
            "described_path_m": round(path(D["position"]), 1),
            "mean_position_error_m": round(float(err.mean()), 1),
            "final_position_error_m": round(float(err[-1]), 1),
            "final_heading_error_deg": round(math.degrees(abs(dh)), 1),
            # The ceiling first: the log ego is on the road by construction, so anything
            # it scores below 1.0 is map coverage, not driving.
            "log_on_road_fraction": round(tracks[scene]["log_on_road"], 3),
            "described_on_road_fraction": round(tracks[scene]["described_on_road"], 3),
            "log_steps_in_collision": log_run[scene]["steps_in_collision"],
            "described_steps_in_collision": desc_run[scene]["steps_in_collision"],
            # R2: the simulator must drive the track it was given, or every number above
            # is a property of the replay rather than of the scenario.
            "replay_max_deviation_m": round(max(log_run[scene]["replay_max_deviation_m"],
                                                desc_run[scene]["replay_max_deviation_m"]), 3),
            "steps_replayed": desc_run[scene]["steps"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / out_csv, index=False)
    return df


def g4_tracks(scene: str) -> dict[str, Any]:
    """The two ego tracks for one scene, WITHOUT the simulator: the log's, read from the
    converted scenario, and the described one, synthesised exactly as
    `_write_described_db` does. `src/figures.py` draws from this in ./venv, so the
    figure's builder lives with every other figure's and needs no MetaDrive."""
    import numpy as np

    sd = pickle.loads(_scenario_files(SIM_DB)[scene].read_bytes())
    ego = sd["tracks"][sd["metadata"]["sdc_id"]]["state"]
    log = np.array(ego["position"][:, :2], dtype=float)
    d = described_ego_track(_descriptions()[scene], len(log), log[0, 0], log[0, 1],
                            float(ego["heading"][0]))
    return {"log": log, "described": d["position"], "surface": _road_surface(sd)}


def build_g4_field_mapping(out_csv: str = "outputs/g4_field_mapping.csv") -> Any:
    """Which ScenarioNet fields the description supplies, partly supplies, or cannot.

    Checked against the converted scenario's actual keys rather than typed from the
    ScenarioNet docs, so a field this table names is a field the loader really reads.
    """
    import pandas as pd

    sd = pickle.loads(next(iter(_scenario_files(SIM_DB).values())).read_bytes())
    ego_state = set(sd["tracks"][sd["metadata"]["sdc_id"]]["state"])
    assert {"position", "heading", "velocity", "valid"} <= ego_state, ego_state
    assert {"tracks", "map_features", "dynamic_map_states", "metadata"} <= set(sd)
    rows = [
        ("metadata.scenario_id", "yes", "scene.name"),
        ("metadata.map", "yes", "scene.location"),
        ("length / metadata.ts", "yes", "scene.duration_s at the simulator's 10 Hz"),
        ("tracks.ego.state.valid", "yes", "every step of the scene's duration"),
        ("tracks.ego.state.velocity", "partly",
         "speed from the timeline's speed labels within parameters.ego_speed_mps"),
        ("tracks.ego.state.heading", "partly",
         "from the steering labels, at a nominal turn angle the description does not carry"),
        ("tracks.ego.state.position", "partly",
         "integrated from speed and heading; the STARTING pose is not in the description"),
        ("tracks.ego.state.length/width/height", "no", "taken from the log conversion"),
        ("tracks.<other road users>", "no",
         "the description gives categories, counts, and range and speed intervals, "
         "never a position or a track"),
        ("map_features (lanes, lines, edges, crossings)", "no",
         "the description gives road statistics (intersection and lane fractions), "
         "not geometry; nuScenes ships no OpenDRIVE"),
        ("dynamic_map_states (signal states)", "no",
         "nuScenes records no signal state at all, so neither does the log conversion"),
    ]
    df = pd.DataFrame(rows, columns=["scenario_field", "supplied_by_description", "source"])
    df.to_csv(ROOT / out_csv, index=False)
    return df


if __name__ == "__main__":
    import pandas as _pd

    _pd.set_option("display.width", 200)
    print(build_g4_field_mapping().to_string(index=False))
    print(build_g4_instantiation().to_string(index=False))
