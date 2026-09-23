"""Step H2 self-checks — signage presence in nuScenes.

The defect this file exists to prevent produced a SILENT ZERO. The obvious positional field
on a `traffic_light` record is `pose`, and on two of the four maps its tx/ty are zero-filled:
all 119 hollandvillage and all 81 queenstown records sit at (0, 0). Reading it put every
light on those maps ~2 km away and the tag fired on 0.00% of their keyframes -- on exactly
the maps where C3's corridor tag fires 19.70% and 6.42%. Nothing raised. That is R16 and
F-019's class: the plausible key is the wrong key and it fails by returning nothing.

Run directly:  ./venv/bin/python tests/test_h2_signage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.signage as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "outputs/h2_traffic_light.parquet"
MAPS = ("boston-seaport", "singapore-onenorth",
        "singapore-hollandvillage", "singapore-queenstown")


def test_every_map_yields_lights_with_real_coordinates():
    """THE regression. Two maps' `pose` is zero-filled; none of the four may return (0,0)."""
    for loc in MAPS:
        p = S.traffic_light_poses(loc)
        assert len(p) > 0, f"{loc}: no lights resolved"
        assert not np.allclose(p[:, :2], 0.0), f"{loc}: positions are zero-filled — reading `pose`?"
        assert p[:, 0].ptp() > 100 and p[:, 1].ptp() > 100, f"{loc}: positions do not span a map"
        assert (p[:, 2] >= 2.0).all() and (p[:, 2] <= 10.0).all(), f"{loc}: implausible mount height"


def test_the_zero_filled_pose_field_is_not_the_source():
    """Guards the fix itself: if someone 'simplifies' back to `pose`, two maps go silent."""
    src = (ROOT / "src" / "signage.py").read_text()
    assert 'r["pose"]["tx"]' not in src, "tx is read from `pose`, which is zero on two maps"
    assert "line_token" in src and "node" in src, "positions must come from the line geometry"


def test_no_map_is_silently_empty_in_the_built_table():
    """The symptom the bug actually produced: a map firing on exactly 0.00% of keyframes."""
    df = pd.read_parquet(TABLE)
    rates = df.groupby("location").traffic_light_in_view.mean()
    assert set(rates.index) == set(MAPS), rates.index.tolist()
    assert (rates > 0.01).all(), f"a map is (near) silent: {rates.to_dict()}"


def test_the_perceptibility_cap_actually_binds():
    """Without it the tag fires on 66% of keyframes on lights up to 3.3 km away."""
    df = pd.read_parquet(TABLE)
    capped = df.traffic_light_in_view.mean()
    uncapped = (df.n_in_frustum_uncapped > 0).mean()
    assert uncapped > capped + 0.20, \
        f"the cap removes only {100*(uncapped-capped):.1f} points; it is nearly inert"
    assert (df.n_in_view <= df.n_in_frustum_uncapped).all(), "capped exceeds uncapped"


def test_it_is_not_c3s_corridor_tag_under_a_new_name():
    """R21, and the check that licensed building this at all. A detector agreeing with an
    existing label has found nothing; measured Jaccard is 0.526."""
    df = pd.read_parquet(TABLE)
    gt = pd.read_parquet(ROOT / "outputs/gt_all.parquet",
                         columns=["sample_token", "traffic_light_ahead"])
    m = df.merge(gt, on="sample_token")
    both = (m.traffic_light_in_view & m.traffic_light_ahead).sum()
    union = (m.traffic_light_in_view | m.traffic_light_ahead).sum()
    assert both / union < 0.80, "the two tags are effectively the same detector"
    assert (m.traffic_light_in_view & ~m.traffic_light_ahead).sum() > 500, \
        "the frustum test adds nothing the corridor test does not already have"


def test_the_module_never_reads_a_colour(  ):
    """F-004/F-076. The map's items[].color enumerates the fixture's LAMPS — 626 of 634
    records carry RED and YELLOW and GREEN at once — so it is not signal state. A tag built
    from it would look principled and be meaningless."""
    src = (ROOT / "src" / "signage.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#") and '"""' not in l)
    assert '"color"' not in code and "['color']" not in code, \
        "signage.py reads a colour field; nuScenes has no signal state"


def test_projections_land_where_a_mounted_fixture_must():
    """Geometry sanity, because the first version's numbers looked plausible while being
    wrong on half the maps. A 2.2-8.6 m fixture at ~30 m cannot project low in the frame."""
    df = pd.read_parquet(TABLE)
    sub = df[df.traffic_light_in_view]
    assert len(sub) > 1000
    assert sub.nearest_m.max() <= S.TRAFFIC_LIGHT_MAX_M + 1e-6
    assert sub.nearest_m.median() < S.TRAFFIC_LIGHT_MAX_M


def test_the_figure_has_a_builder_with_the_guarded_signature():
    """R30/T2: the guard regex is literal."""
    src = (ROOT / "src" / "figures.py").read_text()
    assert 'out_path: str = "outputs/h2_traffic_light.png"' in src


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
    print(f"\n{len(tests) - failed}/{len(tests)} H2 self-checks passed")
    sys.exit(1 if failed else 0)
