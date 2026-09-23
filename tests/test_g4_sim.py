"""Step G4 self-checks — instantiating the pipeline's scenario description in MetaDrive.

The simulator itself runs under ./venv-sim (Python 3.11). Everything checked here needs
only numpy and the written artifacts, so this file runs in the main ./venv with the rest
of the suite; `src/sim.py` keeps every simulator import inside its functions for exactly
that reason.

Run:  ./venv/bin/python tests/test_g4_sim.py
      (the artifacts come from  ./venv-sim/bin/python -m src.sim)
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.sim as S  # noqa: E402


def _desc(panels, vrange=(4.0, 6.0), location="boston-seaport"):
    return {"scene": {"name": "scene-test", "location": location},
            "parameters": {"ego_speed_mps": list(vrange)}, "timeline": panels}


def _panel(a, b, steering="going_straight", speed="cruising"):
    return {"t_s": [a, b], "steering": steering, "speed": speed, "lane_change": False}


def test_a_cruising_straight_description_drives_the_analytic_distance():
    """Cruising starts at the middle of the range (5 m/s) and holds: 10 s is 50 m, no turn."""
    d = S.described_ego_track(_desc([_panel(0.0, 10.0)]), n_steps=101, x0=0.0, y0=0.0,
                              heading0=0.0)
    assert abs(d["position"][-1, 0] - 50.0) < 1e-6, d["position"][-1]
    assert abs(d["position"][-1, 1]) < 1e-9
    assert np.allclose(d["heading"], 0.0)


def test_one_turn_split_across_two_panels_is_one_turn():
    """A speed change splits a turn into two timeline panels. Counting each panel as a
    turn doubled the heading change to 180 degrees; consecutive panels with one steering
    label are one manoeuvre."""
    panels = [_panel(0.0, 4.0, "turn_left", "decelerating"),
              _panel(4.5, 9.0, "turn_left", "cruising")]
    d = S.described_ego_track(_desc(panels), n_steps=101, x0=0.0, y0=0.0, heading0=0.0)
    turned = math.degrees(d["heading"][-1] - d["heading"][0])
    assert abs(turned - S.TURN_DEG) < 1.0, f"turned {turned:.1f} deg, expected one turn"


def test_a_u_turn_goes_across_the_opposing_carriageway():
    """R32: the direction is the traffic side's, not a coin toss. Boston drives on the
    right, so a U-turn is to the left (+); Singapore drives on the left, so to the right."""
    for loc, sign in (("boston-seaport", 1), ("singapore-onenorth", -1)):
        d = S.described_ego_track(_desc([_panel(0.0, 6.0, "u_turn")], location=loc),
                                  n_steps=61, x0=0.0, y0=0.0, heading0=0.0)
        turned = math.degrees(d["heading"][-1] - d["heading"][0])
        assert abs(turned - sign * S.U_TURN_DEG) < 4.0, (loc, turned)


def test_the_described_speed_never_leaves_the_range_the_description_states():
    """R2 on the synthesis: the description bounds the speed, so the track must respect it."""
    panels = [_panel(0.0, 3.0, speed="accelerating"), _panel(3.5, 7.0, speed="decelerating"),
              _panel(7.5, 9.0, speed="stationary")]
    d = S.described_ego_track(_desc(panels, (2.0, 9.0)), n_steps=101, x0=0.0, y0=0.0,
                              heading0=0.0)
    assert d["speed"].min() >= 0.0 and d["speed"].max() <= 9.0 + 1e-9, d["speed"]
    assert d["speed"][-1] == 0.0, "a scene ending stationary must end at rest"


def test_the_instantiation_artifact_is_sound_where_it_can_be_checked():
    """The log arm is the control: MetaDrive must replay every track it was given exactly,
    and the log ego -- on the road by construction -- must read as on the road, or the
    on-road test is measuring map coverage rather than driving."""
    import pandas as pd

    p = ROOT / "outputs/g4_instantiation.csv"
    if not p.exists():
        print("     (skip: run ./venv-sim/bin/python -m src.sim first)")
        return
    df = pd.read_csv(p)
    assert len(df) == 10, "every nuScenes mini scene is instantiated"
    assert (df.replay_max_deviation_m < 0.01).all(), \
        "the simulator drifted from the track it was given"
    assert (df.log_on_road_fraction == 1.0).all(), \
        "the log ego reads as off the road, so the on-road test is miscalibrated"
    assert (df.log_steps_in_collision == 0).all(), "the recorded drive collides in replay"
    assert ((df.described_on_road_fraction >= 0) & (df.described_on_road_fraction <= 1)).all()
    assert (df.duration_s.between(15, 25)).all(), "nuScenes scenes are ~20 s"


def test_the_field_mapping_does_not_claim_what_the_description_lacks():
    """R24 for a table: a replay of the log must never pass for an instantiation of the
    description, so the fields that come from the log are said to come from the log."""
    import pandas as pd

    p = ROOT / "outputs/g4_field_mapping.csv"
    if not p.exists():
        print("     (skip: run ./venv-sim/bin/python -m src.sim first)")
        return
    m = pd.read_csv(p).set_index("scenario_field").supplied_by_description
    assert m["tracks.<other road users>"] == "no"
    assert m["map_features (lanes, lines, edges, crossings)"] == "no"
    assert m["tracks.ego.state.position"] == "partly", "the starting pose is borrowed"


def test_the_simulator_module_imports_without_the_simulator():
    """The main venv must be able to import src/sim.py: the appendix ledger parses it and
    this file tests it. A top-level metadrive import would break both, silently for the
    ledger."""
    r = subprocess.run([str(ROOT / "venv/bin/python"), "-c",
                        "import src.sim; import sys; "
                        "sys.exit('metadrive' in sys.modules)"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:] or "src.sim imported metadrive at load time"


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
    print(f"\n{len(tests) - failed}/{len(tests)} G4 self-checks passed")
    sys.exit(1 if failed else 0)
