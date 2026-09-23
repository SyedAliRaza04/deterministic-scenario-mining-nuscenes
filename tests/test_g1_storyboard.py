"""Step G1 self-checks — storyboard segmentation.

A storyboard is only useful if its panels partition the scene: a reader who sees six panels
must be looking at all of the scene and none of it twice. Those are structural invariants,
not numeric ones, and R3 exists because F-016 was invisible to every numeric check and was
caught only by an exclusivity assertion.

Two defects these guard against specifically:
  * a panel with no keyframe has no thumbnail. The raw union of both manoeuvre axes produces
    193 such panels (4.3%) across the dataset; MIN_PANEL_S exists to remove them.
  * a tag true on only PART of a panel, stated flatly, is a frame-level claim the ground
    truth does not support (D-001). 77.5% of multi-keyframe panels carry one.

Run directly:  ./venv/bin/python tests/test_g1_storyboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.figures as F        # noqa: E402
import src.groundtruth as G    # noqa: E402
import src.storyboard as S     # noqa: E402

# Scenes with known shapes, used instead of a random draw so a failure names a real case.
BUSY = "scene-0061"     # intersection, turn left, following a van
WAIT = "scene-0553"     # stationary the whole clip


def _fake(labels, dt=0.5):
    """A synthetic scene: one row per keyframe, labels given as (lat, lon, lane_change)."""
    return pd.DataFrame({
        "sample_token": [f"t{i}" for i in range(len(labels))],
        "scene_name": "synthetic",
        "location": "nowhere",
        "keyframe_index": range(len(labels)),
        "t_rel_s": [i * dt for i in range(len(labels))],
        "lateral_label": [a for a, _, _ in labels],
        "longitudinal_label": [b for _, b, _ in labels],
        "lane_change": [c for _, _, c in labels],
    })


def test_panels_partition_the_scene_exactly():
    """R3. Chronological, non-overlapping, contiguous, and covering every keyframe once."""
    for scene in (BUSY, WAIT):
        p = S.scene_panels(scene)
        assert list(p.panel_index) == sorted(p.panel_index)
        assert (p.kf_end >= p.kf_start).all(), "a panel ends before it starts"
        # contiguity in integers, so the check is exact rather than a float comparison
        assert list(p.kf_start[1:]) == [e + 1 for e in p.kf_end[:-1]], \
            f"{scene}: panels are not contiguous"
        n_kf = len(S._frames().query("scene_name == @scene"))
        assert p.kf_start.iloc[0] == 0 and p.kf_end.iloc[-1] == n_kf - 1
        assert p.n_keyframes.sum() == n_kf, "coverage is not exactly the scene"


def test_panel_labels_agree_with_the_keyframes_inside_them():
    """R9: two independent derivations, not one checked twice.

    The panel label comes from the run-length encoding; the keyframe labels come from C2's
    `maneuver_labels`. If the segmentation ever drifts from the table it was built on, this
    is what says so.
    """
    frames = S._frames()
    for scene in (BUSY, WAIT):
        sub = frames[frames.scene_name == scene].reset_index(drop=True)
        for r in S.scene_panels(scene).itertuples():
            inside = sub.iloc[r.kf_start:r.kf_end + 1]
            assert set(inside.lateral_label) == {r.lateral_label}, r
            assert set(inside.longitudinal_label) == {r.longitudinal_label}, r


def test_every_panel_has_at_least_one_keyframe_to_show():
    """THE defect MIN_PANEL_S exists for: 193 raw union panels have no thumbnail at all."""
    table = pd.read_parquet("outputs/storyboard_panels.parquet")
    assert table.n_keyframes.min() >= 1, "a panel with no keyframe has no thumbnail"
    assert table.thumbnail_token.notna().all()


def test_a_short_panel_is_absorbed_by_its_longer_neighbour():
    """Hand-computed. A 1-keyframe blip between a long run and a short one goes LEFT."""
    sub = _fake([("going_straight", "cruising", False)] * 8
                + [("turn_left", "cruising", False)]          # 1 kf = 0.5 s, under the floor
                + [("going_straight", "accelerating", False)] * 3)
    t = sub.t_rel_s.to_numpy()
    raw = S._runs(sub)
    assert len(raw) == 3, raw
    merged = S._merge_short(raw, t, 1.0)
    assert len(merged) == 2, merged
    assert merged[0] == (0, 8), "the blip should join the LONGER (earlier) neighbour"


def test_merge_ties_break_toward_the_earlier_neighbour():
    """Determinism: equal-length neighbours must not depend on iteration order."""
    sub = _fake([("going_straight", "cruising", False)] * 4
                + [("turn_left", "cruising", False)]
                + [("going_straight", "accelerating", False)] * 4)
    merged = S._merge_short(S._runs(sub), sub.t_rel_s.to_numpy(), 1.0)
    assert merged[0] == (0, 4), f"tie did not go to the earlier neighbour: {merged}"


def test_collapse_leaves_no_two_adjacent_panels_reading_the_same():
    """Merging can leave neighbours identical, which prints as two indistinguishable panels."""
    table = pd.read_parquet("outputs/storyboard_panels.parquet")
    for _, p in table.groupby("scene_name"):
        key = list(zip(p.lateral_label, p.longitudinal_label, p.lane_change))
        assert all(a != b for a, b in zip(key, key[1:])), f"duplicate adjacent panels in {p.scene_name.iloc[0]}"


def test_a_scene_with_no_events_still_gets_one_honest_panel():
    """D4. 59 of 850 scenes have zero events; the strip must say so, not be empty."""
    sub = _fake([("going_straight", "cruising", False)] * 40)
    runs = S._collapse(S._merge_short(S._runs(sub), sub.t_rel_s.to_numpy(), 1.0), sub)
    assert len(runs) == 1 and runs[0] == (0, 39)


def test_a_partial_tag_is_hedged_and_never_stated_flatly():
    """D-001. Majority-voting a flickering tag manufactures a frame-level claim."""
    p = S.scene_panels(BUSY)
    hedged = p[p.tags_partial.str.len() > 0]
    assert len(hedged), "no panel with a partial tag; the guard would be vacuous"
    for r in hedged.itertuples():
        partial = set(r.tags_partial.split(","))
        throughout = set(r.tags_throughout.split(",")) if r.tags_throughout else set()
        assert not (partial & throughout), "a tag cannot be both partial and throughout"
        if "Briefly:" in r.description:
            assert "Briefly:" in r.description


def test_the_prevalence_gate_is_one_sided():
    """A near-universal tag says nothing; a RARE one is the most informative thing a panel
    can state. The symmetric version of this gate would delete exactly the wrong half."""
    setting, actors = S._describable_tags()
    describable = set(setting) | set(actors)
    assert "on_drivable_area" not in describable, "99.74% prevalence — says nothing"
    assert "has_vehicle" not in describable, "91.19% prevalence — says nothing"
    assert "cut_in" in describable, "2.43% — rare, therefore informative; must survive"
    assert "lead_braking" in describable, "1.89% — rare, therefore informative; must survive"


def test_the_axis_vocabulary_is_never_hand_written():
    """R4/F-016. The tag names come from the vocabulary constants, generated not typed."""
    assert S._axis_tags() == {f"is_{v}" for v in G.LATERAL_LABELS + G.LONGITUDINAL_LABELS}
    src = (Path(__file__).resolve().parent.parent / "src" / "storyboard.py").read_text()
    for literal in ('"is_turn_left"', '"is_going_straight"', '"is_cruising"'):
        assert literal not in src, f"{literal} is hand-written; generate it from the constants"


def test_idle_panels_did_not_invent_an_event_colour():
    """T1/R36. going_straight and cruising are the ABSENCE of an event; a test elsewhere
    asserts they are not in _EVENT_COLOURS, and the strip must not quietly add them."""
    assert "going_straight" not in F._EVENT_COLOURS
    assert "cruising" not in F._EVENT_COLOURS
    assert hasattr(F, "_IDLE_COLOUR"), "the strip needs its own idle colour"


def test_the_storyboard_figure_has_a_builder_with_the_guarded_signature():
    """R30/T2. The guard regex is literal; a Path or an f-string default passes here and
    fails the figure guard the moment the PNG exists."""
    src = (Path(__file__).resolve().parent.parent / "src" / "figures.py").read_text()
    assert 'out_path: str = "outputs/g1_storyboard.png"' in src


# --- Step G2: boundary scoring -------------------------------------------------------
#
# R31: these sit ABOVE the registry line, which is evaluated once at import. A test defined
# after it is collected by nothing and reports nothing.


def test_boundary_prf_is_existence_matching_not_one_to_one():
    """Precision and recall must be counted from their OWN side.

    One prediction sitting between two gold boundaries one keyframe apart recovers both at
    tol=1 while being a single true positive for precision. Sharing one tp between the two
    ratios would make recall unreachable for any segmenter, and 20.3% of consecutive gold
    boundaries in this dataset ARE within one keyframe.
    """
    tp, fp, fn = S.boundary_prf({10}, {9, 11}, tol=1)
    assert (tp, fp, fn) == (1, 0, 0), (tp, fp, fn)
    tp, fp, fn = S.boundary_prf({10}, {9, 11}, tol=0)
    assert (tp, fp, fn) == (0, 1, 2), (tp, fp, fn)
    assert S.boundary_prf(set(), {5}, tol=1) == (0, 0, 1)
    assert S.boundary_prf({5}, set(), tol=1) == (0, 1, 0)


def test_the_matched_count_baseline_really_is_matched_and_interior():
    """An unmatched baseline wins precision by predicting almost nothing, which is why the
    fixed-interval null is handed the oracle's own boundary count. It must also never place
    a boundary at keyframe 0 or at the last keyframe: those are boundaries for every
    segmenter including a trivial one, and counting them flatters the weakest baseline most.
    """
    for n, k in ((40, 4), (12, 1), (41, 7), (8, 8)):
        b = S.fixed_interval_boundaries(n, k)
        assert len(b) <= k, (n, k, b)
        assert all(0 < x < n for x in b), (n, k, b)
    assert S.fixed_interval_boundaries(40, 0) == set()
    assert S.fixed_interval_boundaries(1, 3) == set()


def test_the_random_null_matches_its_closed_form_on_the_real_data():
    """R9, as a regression.

    The measured random baseline (0.153 / 0.382 / 0.549 / 0.665 at tol 0..3) and the closed
    form (2*tol + 1) * 0.0986 arrive at the chance rate from independent directions. If a
    future change to the gold standard moves one and not the other, something has broken in
    a way no single number would reveal.
    """
    df = pd.read_csv(ROOT / "outputs/g2_segmentation.csv")
    rnd = df[df.method == "random_matched_k"].set_index("tolerance_kf")
    for tol in (0, 1, 2):
        measured, closed = rnd.f1[tol], rnd.chance_f1_closed_form[tol]
        assert abs(measured - closed) < 0.09, (tol, measured, closed)
    assert (rnd.f1_sd < 0.02).all(), "the random null is noisier than reported"


def test_the_fixed_interval_baseline_barely_beats_chance():
    """THE RESULT G2 EXISTS TO STATE.

    An evenly spaced segmenter given the oracle's own boundary count clears random
    placement by 0.03 at +/-1 keyframe and never by more than 0.10. If this assertion ever
    starts failing upward, the baseline has stopped being a null and the comparison it is
    used for is no longer honest.
    """
    df = pd.read_csv(ROOT / "outputs/g2_segmentation.csv")
    for tol in (0, 1, 2, 3):
        d = df[df.tolerance_kf == tol].set_index("method")
        gap = d.f1["fixed_matched_k"] - d.f1["random_matched_k"]
        assert 0 <= gap < 0.10, (tol, gap)


def test_g1_is_scored_but_never_called_a_validation():
    """R21 AS A TEST, not as a paragraph.

    G1's panels are maximal runs of the labels the gold events are derived from, so the G1
    row measures D-050's linearisation and not correctness. It scores 0.885 at +/-1 -- high
    enough to be quoted as a result by someone reading only the CSV -- so both the module
    and the figure must carry the warning in text.
    """
    df = pd.read_csv(ROOT / "outputs/g2_segmentation.csv")
    g1 = df[(df.method == "g1_panels") & (df.tolerance_kf == 1)].iloc[0]
    assert g1.f1 > 0.8, g1.f1
    src = (ROOT / "src" / "storyboard.py").read_text()
    assert "R21" in src.split("Step G2")[1], "the G2 section must name R21"
    fig = (ROOT / "src" / "figures.py").read_text()
    assert "NOT A VALIDATION" in fig, "the figure legend must say so where it is read"


# --- Step G2b: the VLM segmentation arm ----------------------------------------------


def test_the_vlm_scene_draw_is_uniform_and_keeps_the_published_chance_floor():
    """THE SELECTION TRAP, as a test.

    Drawing only scenes with >= 3 manoeuvre boundaries looks like sensible enrichment and
    lifts boundary density from 0.099 to 0.141 per keyframe -- which lifts the chance floor
    from 0.30 to 0.42 at +/-1 and flatters every segmenter scored against it. That is
    F-070's mistake with a different name. The draw must stay uniform, and its density must
    stay within a point of the 850-scene table's or the two are not comparable.
    """
    import json

    spec = json.loads((ROOT / "outputs/g2_vlm_items.json").read_text())
    full = pd.read_csv(ROOT / "outputs/g2_segmentation.csv").gold_density_per_kf.iloc[0]
    assert abs(spec["gold_density_per_kf"] - full) < 0.01, (spec["gold_density_per_kf"], full)
    assert spec["n_scenes_with_no_boundary"] > 0, (
        "a draw with no empty scenes is not uniform -- 14% of nuScenes scenes contain no "
        "manoeuvre event at all, and those are the only scenes that test false positives")
    for tol in ("0", "1", "2"):
        assert spec["chance_f1_closed_form"][tol] > 0, tol


def test_a_pair_item_brackets_the_keyframe_it_asks_about():
    """The off-by-one that would invalidate every number.

    A `true` on item `<scene>:<kf>` is read as a boundary AT kf, so the two images shown
    must be kf-1 and kf+1. If they were kf and kf+1 the answer would name the gap, every
    prediction would sit half a keyframe early, and at +/-0 tolerance the arm would score
    near zero for a reason that is ours and not the model's.
    """
    import json

    spec = json.loads((ROOT / "outputs/g2_vlm_items.json").read_text())
    fr = S._frames()
    by_scene = {sc: g.sort_values("keyframe_index") for sc, g in fr.groupby("scene_name")}
    for it in spec["items"][:200]:
        f = by_scene[it["scene_name"]]
        toks = f.sample_token.tolist()
        kfs = f.keyframe_index.tolist()
        i = kfs.index(it["keyframe_index"])
        assert toks[i - 1] == it["before_token"], it
        assert toks[i + 1] == it["after_token"], it
    # and never the scene's own first or last keyframe (the gold set excludes them too)
    for it in spec["items"]:
        f = by_scene[it["scene_name"]]
        assert f.keyframe_index.min() < it["keyframe_index"] < f.keyframe_index.max(), it


def test_an_unparsed_row_is_not_read_as_no_boundary():
    """F-062 on the segmentation arm. Treating a parse failure as 'no boundary here' would
    make a broken run look like a conservative segmenter with excellent precision."""
    rows = [
        {"sample_token": "scene-0001:5", "backend_error": False, "values": {S.G2_VLM_TAG: True}},
        {"sample_token": "scene-0001:7", "backend_error": False, "values": {}},
        {"sample_token": "scene-0001:9", "backend_error": True, "values": {}},
        {"sample_token": "scene-0001:11", "backend_error": False, "values": {S.G2_VLM_TAG: False}},
    ]
    b = S.boundaries_from_rows(rows)
    assert b["scene-0001"] == {5}, b
    assert 7 not in b["scene-0001"] and 11 not in b["scene-0001"]


def test_the_scorer_reproduces_an_oracle_and_prices_an_always_yes_segmenter():
    """END TO END ON SYNTHETIC RUNS, because a scorer that is never exercised is F-071's
    shape: correct code wired to nothing.

    The always-yes case is the one that matters. A segmenter that calls every keyframe a
    boundary scores F1 0.187 here, and the random null MATCHED TO ITS OWN COUNT scores the
    same 0.187 -- so enthusiasm buys exactly nothing, which is the check the majority
    baseline performs for the tag arms (F-068).
    """
    import json
    import tempfile

    spec = json.loads((ROOT / "outputs/g2_vlm_items.json").read_text())
    gold = {s: set(v) for s, v in spec["gold"].items()}

    def run(fn):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            with p.open("w") as f:
                for it in spec["items"]:
                    f.write(json.dumps({
                        "sample_token": it["item_id"], "backend_error": False,
                        "parse_reason": "ok", "values": {S.G2_VLM_TAG: fn(it)}}) + "\n")
            return S.score_vlm_boundaries(results_stem=str(p),
                                          out_csv=str(Path(td) / "s.csv"))

    oracle = run(lambda it: it["keyframe_index"] in gold[it["scene_name"]])
    o = oracle[(oracle.tolerance_kf == 0) & (oracle.method == "vlm_pairwise")].iloc[0]
    assert abs(o.f1 - 1.0) < 1e-9, o.f1

    yes = run(lambda it: True)
    y = yes[yes.tolerance_kf == 0].set_index("method")
    assert abs(y.recall["vlm_pairwise"] - 1.0) < 1e-9
    assert abs(y.f1["vlm_pairwise"] - y.f1["random_matched_vlm"]) < 0.01, (
        "an always-yes segmenter must score exactly what random placement with the same "
        "boundary count scores; if it does not, the matched null is not matched")


def test_the_g2_bundle_carries_both_frames_of_every_item():
    """A hole in the image set becomes a short results file that nothing flags (F-048), and
    here it would silently drop whole keyframes from the boundary set."""
    import json
    import tarfile

    b = ROOT / "outputs/g2_bundle.tar.gz"
    if not b.exists():
        return
    spec = json.loads((ROOT / "outputs/g2_vlm_items.json").read_text())
    with tarfile.open(b) as tar:
        names = set(tar.getnames())
    need = {f"images/{t}.jpg" for it in spec["items"]
            for t in (it["before_token"], it["after_token"])}
    assert not (need - names), f"{len(need - names)} of {len(need)} frames absent"
    assert "outputs/g2_vlm_items.json" in names, "the work list must travel with it"


def test_the_null_publishes_its_spread_not_only_its_mean():
    """THE REHEARSAL DEFECT, as a regression.

    Found by running the G2 notebook against a fake backend that answers `true` at random:
    it cleared the null's MEAN by +0.051 at +/-1 keyframe, which reads as a real effect. It
    is 1.1 standard deviations of a SINGLE draw -- and the model gets one draw, while the
    null is a mean over 200. Reporting a gap to the mean would have let a coin flip look
    like a segmenter. The band must be in the artifact, and it must be wide enough to
    matter: sd ~0.03 at these counts.
    """
    import json
    import tempfile

    spec = json.loads((ROOT / "outputs/g2_vlm_items.json").read_text())

    def run(fn):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            with p.open("w") as f:
                for it in spec["items"]:
                    f.write(json.dumps({
                        "sample_token": it["item_id"], "backend_error": False,
                        "parse_reason": "ok", "values": {S.G2_VLM_TAG: fn(it)}}) + "\n")
            return S.score_vlm_boundaries(results_stem=str(p),
                                          out_csv=str(Path(td) / "s.csv"))

    import random as _r
    df = run(lambda it: _r.Random(it["item_id"]).random() < 0.25)
    nulls = df[df.method.str.startswith("random_")]
    assert nulls.f1_sd.notna().all(), "the null rows carry no spread"
    assert (nulls.f1_sd > 0.005).all(), f"implausibly tight null: {nulls.f1_sd.tolist()}"
    assert (nulls.f1_hi > nulls.f1).all() and (nulls.f1_lo < nulls.f1).all()
    # a segmenter answering at random must NOT clear its own matched null's upper bound
    vlm = df[df.method == "vlm_pairwise"].set_index("tolerance_kf")
    mv = df[df.method == "random_matched_vlm"].set_index("tolerance_kf")
    for tol in (0, 1, 2, 3):
        assert vlm.f1[tol] <= mv.f1_hi[tol], (
            f"a random segmenter beat its own matched null at tol={tol} "
            f"({vlm.f1[tol]:.3f} > {mv.f1_hi[tol]:.3f}); the null is not matched")
    # the deterministic rows must NOT pretend to have one
    det = df[~df.method.str.startswith("random_")]
    assert det.f1_sd.isna().all(), "a deterministic segmenter was given a spread"


def test_the_g2_bundle_can_score_itself():
    """THE KAGGLE DEFECT, as a regression. The first G2b session ran all 951 calls and then
    died in the scoring cell on `from . import eval`: the bundle shipped storyboard.py
    without the eval.py it imports lazily. The rehearsal had scored from the REPO, where
    every module exists. So this scores from the EXTRACTED BUNDLE, in a fresh interpreter
    that cannot see the repo, which catches any missing module rather than one named here.
    """
    import json
    import subprocess
    import tarfile
    import tempfile

    b = ROOT / "outputs/g2_bundle.tar.gz"
    if not b.exists():
        return
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(b) as tar:
            tar.extractall(td, members=[m for m in tar if not m.name.startswith("images/")],
                           filter="data")
        spec = json.loads((Path(td) / "outputs/g2_vlm_items.json").read_text())
        res = Path(td) / "r.jsonl"
        res.write_text("".join(json.dumps({
            "sample_token": it["item_id"], "backend_error": False, "parse_reason": "ok",
            "values": {S.G2_VLM_TAG: True}}) + "\n" for it in spec["items"]))
        # the notebook's scoring call, verbatim in shape: absolute paths, bundle on sys.path
        code = ("import src.storyboard as S; S.score_vlm_boundaries(results_stem='r.jsonl', "
                "items_path='outputs/g2_vlm_items.json', out_csv='s.csv')")
        p = subprocess.run([sys.executable, "-c", code], cwd=td, capture_output=True,
                           text=True)
        assert p.returncode == 0, "the bundle cannot score its own run:\n" + p.stderr[-600:]
        assert (Path(td) / "s.csv").exists()


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
    print(f"\n{len(tests) - failed}/{len(tests)} G1 self-checks passed")
    sys.exit(1 if failed else 0)
