r"""Step H3 self-checks — BDD100K signage.

`src/bdd.py` was the only module in `src/` with no self-check file, and it is the one about
to be run on paid GPU time. Everything here guards a defect this project has actually hit,
translated to the new dataset:

  R4   the tag vocabulary is written once. `BDD_TAGS` drove the prompt, the scorer and the
       subset counts; a hand-written sixth name in any of them is F-016 again, and F-016
       left `is_turning_left` False on all 5,500 turning frames.
  F-076 the real encoding is {G, R, Y, NA}, NOT {red, green, yellow, none}. PLAN expected
       the latter, and a scorer written against PLAN's spelling would score every state tag
       False and call it a finding.
  F-004 a field whose NAME says state and whose CONTENT is an inventory. nuScenes'
       `items[].color` cost a silent 0.00%; the guard here is that the three state tags are
       derived from `trafficLightColor` on the DETECTION and from nothing else.
  F-048 a transfer that fails quietly. 29.94% of images carry a light whose colour is NA,
       so `state_gt_uncertain` must be set on exactly those, and the bundle must refuse to
       build around a missing or truncated image.

Run:  ./venv/bin/python tests/test_h3_bdd.py
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.bdd as B  # noqa: E402

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


def _sample(colours=(), sign=False, h=720):
    """A BDD100K sample record with the given light colours."""
    dets = [{"label": "traffic light", "trafficLightColor": c,
             "bounding_box": [0.1, 0.1, 0.02, 0.04]} for c in colours]
    if sign:
        dets.append({"label": "traffic sign", "bounding_box": [0.5, 0.5, 0.03, 0.03]})
    return {"filepath": "data/aaaaaaaa-bbbbbbbb.jpg",
            "metadata": {"height": h, "width": 1280, "size_bytes": 1234},
            "detections": {"detections": dets}}


@check("test_state_comes_from_the_detection_colour_and_the_encoding_is_GRY")
def _():
    """F-076 AND F-004 IN ONE.

    PLAN H1 expected `{red, green, yellow, none}`; the export ships `{G, R, Y, NA}`. A
    scorer written against PLAN's spelling returns False on every state tag and looks like
    a model that cannot see colour. And the colour must come from the DETECTION, never from
    a fixture inventory field — which is exactly what made nuScenes' `traffic_light` layer
    useless for state.
    """
    t = B.tags_for(_sample(colours=("R",)))
    assert t["red_light_visible"] and not t["green_light_visible"], t
    assert t["traffic_light_present"], t
    assert B.tags_for(_sample(colours=("G", "R")))["green_light_visible"]
    assert B.tags_for(_sample(colours=("Y",)))["yellow_light_visible"]
    # the spellings PLAN expected must NOT be accepted as state
    loud = B.tags_for(_sample(colours=("red",)))
    assert not loud["red_light_visible"], (
        "a lowercase 'red' was read as a lit red light; the export's encoding is {G,R,Y,NA} "
        "and accepting both spellings hides which one the data actually uses")
    assert loud["traffic_light_present"], "presence must still hold — the light is there"


@check("test_state_gt_uncertain_marks_NA_and_unlit_but_never_a_clean_image")
def _():
    """The 29.94%. A False on a state tag is only trustworthy where the reference could
    have said otherwise, so the flag has to fire on NA and on a light with no lit colour,
    and must NOT fire on an image with no light at all — that one is a clean, certain
    negative and flagging it would shrink the certain population for nothing."""
    assert B.tags_for(_sample(colours=("NA",)))["state_gt_uncertain"]
    assert B.tags_for(_sample(colours=("R", "NA")))["state_gt_uncertain"]
    assert not B.tags_for(_sample(colours=("R",)))["state_gt_uncertain"]
    assert not B.tags_for(_sample(colours=()))["state_gt_uncertain"], (
        "an image with no traffic light has a CERTAIN negative state, not an uncertain one")
    assert not B.tags_for(_sample(colours=(), sign=True))["state_gt_uncertain"]


@check("test_the_tag_vocabulary_is_written_once")
def _():
    """R4. Five names appear in the tuple, in the question map and in the prompt's JSON
    skeleton. Any divergence is a tag the model is never asked for, or one the scorer never
    finds — both silent, both F-016."""
    assert set(B.BDD_TAGS) == set(B.BDD_QUESTIONS), (
        set(B.BDD_TAGS) ^ set(B.BDD_QUESTIONS))
    prompt = B.build_prompt()
    for t in B.BDD_TAGS:
        assert prompt.count(f'"{t}"') >= 2, f"{t} is not both asked and in the skeleton"
    assert set(B.tags_for(_sample()).keys()) >= set(B.BDD_TAGS)
    # and nothing from the nuScenes schema leaked in
    import src.prompts as P
    assert not (set(B.BDD_TAGS) & set(P.load_schema()["tags"])), (
        "a BDD tag shares a name with a frozen nuScenes tag; the two vocabularies are "
        "scored against different references and must not collide")


@check("test_the_prompt_asks_about_the_image_and_forbids_prose")
def _():
    """F-065's shape, checked on the new prompt: a prompt that asks for reasoning and then
    forbids it tests nothing, and 5.6 h of E8 was spent finding that out once."""
    p = B.build_prompt()
    wants = "think step by step" in p or "reasoning first" in p
    assert not (wants and "nothing else" in p), p[:200]
    assert "nothing else" in p, "the arm expects bare JSON; say so"
    assert "do not guess from context" in p, (
        "the tags are about what is VISIBLE; without this the model answers from priors "
        "about roads, which is the yes-bias F-068 measured")


@check("test_the_subset_keeps_real_prevalence_except_where_it_says_otherwise")
def _():
    """D-047's move, re-applied. The uniform 450 exist so four of five baselines are the
    REAL ones; only `yellow_light_visible` is enriched, and the artifact must say which
    view is which or a reader quotes an enriched baseline as a population figure."""
    spec = json.loads((ROOT / "outputs/h3_bdd_subset.json").read_text())
    assert spec["n_uniform"] == B.N_UNIFORM == 450, spec["n_uniform"]
    assert len(spec["uniform_filepaths"]) == spec["n_uniform"]
    assert spec["n_total"] == len(spec["records"]) >= spec["n_uniform"]
    assert spec["positives_all"]["yellow_light_visible"] >= 30, (
        "the enriched view must clear D-032's 30-positive floor or the tag is unscoreable")
    uni = set(spec["uniform_filepaths"])
    assert uni <= {r["filepath"] for r in spec["records"]}, "the uniform view is not a subset"
    # every top-up image is a yellow positive; a top-up that dragged in negatives would
    # quietly move the other four prevalences too
    recs = {r["filepath"]: r for r in spec["records"]}
    topup = [recs[f] for f in recs if f not in uni]
    assert all(r["yellow_light_visible"] for r in topup), (
        f"{sum(not r['yellow_light_visible'] for r in topup)} top-up images are not yellow "
        "positives, so they change every other tag's prevalence for nothing")


@check("test_the_bundle_refuses_to_build_around_a_missing_image")
def _():
    """F-048. A hole in the image set becomes a short results file that nothing flags. The
    bundle must fail loudly on the laptop rather than after a 32 MB upload."""
    import shutil
    import tempfile

    spec = json.loads((ROOT / "outputs/h3_bdd_subset.json").read_text())
    name = Path(spec["records"][0]["filepath"]).name
    src = B.IMAGES / name
    if not src.exists():
        return                                   # images not fetched here; nothing to prove
    with tempfile.TemporaryDirectory() as td:
        hidden = Path(td) / name
        shutil.move(src, hidden)
        try:
            B.build_bundle(out_path="outputs/_h3_bundle_test.tar.gz")
            raise AssertionError("bundle built with an image missing")
        except AssertionError as e:
            assert "missing or truncated" in str(e), e
        finally:
            shutil.move(hidden, src)
            (ROOT / "outputs/_h3_bundle_test.tar.gz").unlink(missing_ok=True)


@check("test_the_built_bundle_carries_every_record_and_the_labels")
def _():
    b = ROOT / "outputs/h3_bundle.tar.gz"
    if not b.exists():
        return
    spec = json.loads((ROOT / "outputs/h3_bdd_subset.json").read_text())
    with tarfile.open(b) as tar:
        names = set(tar.getnames())
    want = {f"images/{Path(r['filepath']).name}" for r in spec["records"]}
    assert not (want - names), f"{len(want - names)} subset images absent from the bundle"
    assert "outputs/h3_bdd_subset.json" in names, "the ground truth must travel with it"
    assert "src/bdd.py" in names and "src/eval.py" in names, (
        "the notebook scores in-session; without the scorer it cannot")


@check("test_scoring_reads_unanswered_as_missing_not_as_false")
def _():
    """F-062's rule on the new vocabulary. A tag the model did not answer must be excluded
    from that tag's counts, never scored False — the conflation that made the original
    pipeline unscoreable."""
    import src.eval as E

    spec = json.loads((ROOT / "outputs/h3_bdd_subset.json").read_text())
    ids = [B.image_id(r["filepath"]) for r in spec["records"][:4]]
    rows = [{"sample_token": i, "backend_error": False, "parse_reason": "ok",
             "values": {t: True for t in B.BDD_TAGS if t != "green_light_visible"}}
            for i in ids]
    pred = E.predictions_frame(rows, ids, list(B.BDD_TAGS))
    assert pred["green_light_visible"].isna().all(), "an unanswered tag became False"
    assert pred["red_light_visible"].all()


@check("test_a_state_answer_is_three_booleans_and_never_one_colour")
def _():
    """THE DESIGN DECISION, frozen. 34% of lit images show more than one distinct lit
    colour and BDD has no lane association, so a single 'what colour is the light?' would
    score a confoundable geometric guess (R22). If a future edit collapses the three state
    tags into one colour field, this fails."""
    state = [t for t in B.BDD_TAGS if t.endswith("_light_visible")]
    assert len(state) == 3, state
    p = B.build_prompt()
    for bad in ("what colour", "what color", '"colour"', '"color"'):
        assert bad not in p.lower(), f"the prompt asks for a colour: {bad!r}"
    both = B.tags_for(_sample(colours=("R", "G")))
    assert both["red_light_visible"] and both["green_light_visible"], (
        "two lit colours in one image must set two tags; a single-colour answer cannot "
        "represent this and 34.0% of lit images are like it")



# --- H3 results and H4 -----------------------------------------------------------------
#
# These read the REAL run (outputs/results/h3_bdd_signage.txt). They skip cleanly when it
# is absent, because a fresh clone has no results and must not fail for that.

_RUN = ROOT / "outputs/results/h3_bdd_signage.txt"


@check("test_the_local_scorer_reproduces_the_kaggle_session_exactly")
def _():
    """THE LICENCE FOR EVERY H3 NUMBER, same test as the Stage E arms.

    The session scored itself on Kaggle; the laptop re-scores the raw rows. If the two ever
    disagree, one of them is not computing what the thesis says it computes, and neither
    number is quotable. Measured: zero discrepancy across all 90 cells.
    """
    kag = ROOT / "outputs/results/h3_scores.txt"
    if not (_RUN.exists() and kag.exists()):
        return
    fresh, old = B.score(), json.loads(kag.read_text())
    for view in ("all", "uniform_view", "state_gt_certain_only"):
        for t in B.BDD_TAGS:
            for k in ("precision", "recall", "f1", "support", "baseline_f1"):
                assert abs(fresh[view][t][k] - old[view][t][k]) < 1e-9, (view, t, k)


@check("test_enrichment_cannot_change_yellows_false_positives")
def _():
    """THE H3 FINDING, frozen.

    Every top-up image is a yellow positive, so it can add a true positive or a false
    negative and NEVER a false positive. The raw rows bear that out: 96 false positives in
    the uniform 450 and the same 96 in the enriched 586, while F1 goes 0.182 -> 0.540. If
    this ever fails, the top-up has started admitting negatives and the view comparison
    stops being a statement about prevalence alone.
    """
    if not _RUN.exists():
        return
    import pandas as pd

    v = pd.read_csv(ROOT / "outputs/h3_views.csv")
    y = v[v.tag == "yellow_light_visible"].set_index("view")
    assert y.fp["uniform_450"] == y.fp["all_586"], y.fp.to_dict()
    assert y.f1["all_586"] > 2 * y.f1["uniform_450"], (
        "the prevalence effect is the finding; it should be large")
    assert (y.tp["all_586"] - y.tp["uniform_450"]) + \
           (y.fn["all_586"] - y.fn["uniform_450"]) == 136, "top-up is 136 images"


@check("test_size_bands_hold_only_lit_images")
def _():
    """An image with no lit light has no size. Binning it at 0 px would put the whole
    no-light population in the smallest band and make that band a statement about
    something else entirely."""
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/h3_size_decomposition.csv")
    spec = json.loads((ROOT / "outputs/h3_bdd_subset.json").read_text())
    n_lit = sum(1 for r in spec["records"] if r["max_lit_px"] > 0)
    per_band = d.drop_duplicates("size_band_px").n_images.sum()
    assert per_band == n_lit, (per_band, n_lit)


@check("test_h4_baseline_is_majority_class_not_always_yes")
def _():
    """THE REGRESSION FOR THE FIGURE'S FALSE LABEL.

    The first H4 render called every baseline "always-yes". At 22.9% prevalence the
    nuScenes majority is NO, so its baseline predicts no and scores 0.000; an always-yes
    predictor would score 0.373. A label true for one column and false for another is a
    claim about the schema that the data contradicts (R36).
    """
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/h4_cross_dataset.csv")
    for r in d.itertuples():
        if r.prevalence < 0.5:
            assert r.baseline_f1 == 0.0, (r.reference_name, r.baseline_f1)
        else:
            assert r.baseline_f1 > 0.5, (r.reference_name, r.baseline_f1)
    fig = (ROOT / "src/figures.py").read_text().split("def h4_cross_dataset_figure", 1)[1]
    assert '("always-yes baseline"' not in fig, "the false row label is back"
    assert "majority-class baseline" in fig


@check("test_h4_describes_the_scored_tag_not_h2s")
def _():
    """THE REGRESSION FOR MY OWN DEFINITION ERROR (F-088).

    The first draft described nuScenes' `traffic_light_ahead` as a frustum projection
    within 50 m. That is H2's `traffic_light_in_view`. The scored tag is a fixture inside a
    30 m x 8 m forward corridor, and the whole explanation of the gap turns on the
    difference. The table must name the corridor's real parameters, read from the schema,
    not typed from memory.
    """
    if not _RUN.exists():
        return
    import pandas as pd

    import src.prompts as P

    par = P.load_schema()["tags"]["traffic_light_ahead"]["parameters"]
    d = pd.read_csv(ROOT / "outputs/h4_cross_dataset.csv")
    scored = d[d.reference_name.str.startswith("corridor")].iloc[0]
    assert f"{int(par['AHEAD_RANGE_M'])} m" in scored.scope, scored.scope
    assert f"{int(par['CORRIDOR_HALFWIDTH_M'])} m" in scored.scope, scored.scope
    assert "50 m" not in scored.scope and "frustum" not in scored.reference, (
        "the scored tag is being described with H2's definition again")


@check("test_the_reference_swap_holds_predictions_fixed_and_closes_most_of_the_gap")
def _():
    """F3's move, cross-dataset. The corridor row and the view row MUST be the same
    predictions -- if they were not, the swap would be two experiments, not one reference
    change. Same frame count, and the predicted-positive count (tp + fp) identical."""
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/h4_cross_dataset.csv")
    cor = d[d.reference_name.str.startswith("corridor")].iloc[0]
    view = d[d.reference_name.str.startswith("view")].iloc[0]
    assert cor.n == view.n, (cor.n, view.n)
    assert cor.tp + cor.fp == view.tp + view.fp, (
        "the model's positive claims changed between rows; that is not a reference swap")
    assert 0.5 < d.share_of_gap_closed_by_scope.iloc[0] <= 1.0, d.share_of_gap_closed_by_scope


@check("test_h4_carries_its_confounds_as_columns")
def _():
    """Reference, scope, occlusion, prevalence and independence differ between rows, so
    the gap cannot be read without them. They are columns, not a footnote, and the
    precision/recall split that locates the gap must be in the artifact."""
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/h4_cross_dataset.csv")
    for col in ("reference", "models_occlusion", "scope", "prevalence",
                "independent_samples", "delta_precision_vs_corridor",
                "delta_recall_vs_corridor", "share_of_gap_closed_by_scope"):
        assert col in d.columns, col
    bd = d[d.dataset == "BDD100K"].iloc[0]
    assert bd.delta_precision_vs_corridor > 5 * abs(bd.delta_recall_vs_corridor), (
        "the finding is that the gap lives in precision; if recall moved as much, it is not")

if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} H3 self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
