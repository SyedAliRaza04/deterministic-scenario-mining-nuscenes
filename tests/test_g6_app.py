"""Step G6 self-checks — the demonstrator.

A Streamlit script cannot be unit-tested meaningfully, so these guard the things that would
actually break it in front of a supervisor: a missing artifact, an accidental devkit load
(~48 s per interaction, which would make it unusable), and drift between what the app claims
to show and what exists.

Run directly:  ./venv/bin/python tests/test_g6_app.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def test_the_app_parses_and_declares_its_scope():
    src = APP.read_text()
    ast.parse(src)
    assert "628" in src or "execution subset" in src, \
        "the app must say WHICH scenes carry VLM results; 139 of 850 do"


def test_the_app_never_loads_the_devkit():
    """R20. load_nusc costs ~48 s; in an interactive app that is fatal, and
    data.image_for_sample_token exists precisely so it is not needed."""
    src = APP.read_text()
    assert "load_nusc" not in src and "NuScenes(" not in src, \
        "the app loads the devkit; use data.image_for_sample_token instead"
    assert "image_for_sample_token" in src


def test_every_artifact_the_app_reads_exists():
    """The failure mode is a FileNotFoundError during a demo."""
    for rel in ("outputs/gt_all.parquet", "outputs/storyboard_panels.parquet",
                "outputs/h2_traffic_light.parquet", "outputs/scenario_descriptions.jsonl",
                "outputs/vlm_subset_tokens.json", "outputs/label_schema.json"):
        assert (ROOT / rel).exists(), f"missing {rel}"


def test_the_demo_scenes_really_have_everything_it_renders():
    """Every scene the picker offers must have a storyboard, a scenario description, a BEV
    raster and a camera image — or a tab renders empty in front of an audience."""
    import src.data as D

    tokens = json.loads((ROOT / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = pd.read_parquet(ROOT / "outputs/gt_all.parquet",
                         columns=["sample_token", "scene_name"]).set_index("sample_token")
    scenes = {gt.loc[t, "scene_name"] for t in tokens}
    assert len(scenes) == 139, len(scenes)

    panels = set(pd.read_parquet(ROOT / "outputs/storyboard_panels.parquet").scene_name)
    scen = {json.loads(l)["scene"]["name"]
            for l in (ROOT / "outputs/scenario_descriptions.jsonl").open()}
    assert scenes <= panels, "a demo scene has no storyboard"
    assert scenes <= scen, "a demo scene has no scenario description"

    for t in tokens[:40]:
        assert (ROOT / f"data/bev/{t}.png").exists(), f"no BEV raster for {t}"
        assert D.image_for_sample_token(t).exists(), f"no camera image for {t}"


def test_the_scope_caveats_the_brief_cares_about_are_present():
    """The app is the easiest place in the project to mislead someone: a BEV rendered FROM
    ground truth reads as perceived, and a whole-scene description reads as a frame label."""
    src = APP.read_text()
    assert "not perceived from cameras" in src, "the BEV's provenance must be stated"
    assert "WHOLE-SCENE" in src, "the human description must be marked whole-scene (L-007)"
    assert "no signal STATE" in src or "F-004" in src, "signage state absence must be stated"


tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in tests:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {fn.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} G6 self-checks passed")
    sys.exit(1 if failed else 0)
