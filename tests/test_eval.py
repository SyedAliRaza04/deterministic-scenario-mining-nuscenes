r"""Stage F self-checks — scoring.

The scoring code is the last place a silent defect can still rewrite every number in
the thesis, and until now it did not exist: the five condition tables were scored three
times by ad-hoc heredocs that no longer exist (R30). Two traps are guarded here:

  F1  a tag the model DID NOT ANSWER must not be scored as False. coerce_tags keeps the
      distinction and the scorer is the place it can still be thrown away. Chain-of-thought
      is the first condition where any cell is unanswered at all (F-066).
  F2  the majority baseline's tie-break is worth 0.667 F1 on `lead_vehicle`, which is
      positive on exactly 314 of 628 frames. A tie broken the convenient way flatters the
      model by 0.02 macro F1.

The strongest check is the last one: the scorer must reproduce the four already-published
arms to the third decimal, or the numbers it produces for the fifth are not comparable.

Run:  ./venv/bin/python tests/test_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.eval as E  # noqa: E402

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


def _frame(values, tags=("a",), index=None):
    idx = index or [f"t{i}" for i in range(len(values))]
    return pd.DataFrame(values, index=pd.Index(idx, name="sample_token"),
                        columns=list(tags), dtype=object)


@check("test_per_tag_metrics_match_a_hand_computed_case")
def _():
    gt = _frame([[True], [True], [False], [False]]).astype(bool)
    pred = _frame([[True], [False], [False], [True]])
    m = E.per_tag_metrics(gt, pred, ["a"])
    assert m.tp["a"] == 1 and m.fp["a"] == 1 and m.fn["a"] == 1, m
    assert abs(m.precision["a"] - 0.5) < 1e-9
    assert abs(m.recall["a"] - 0.5) < 1e-9
    assert abs(m.f1["a"] - 0.5) < 1e-9
    assert m.support["a"] == 2 and m.n_answered["a"] == 4


@check("test_an_unanswered_tag_is_excluded_not_scored_as_false")
def _():
    """THE F1 regression. If None were read as False, frame t1 would become a false
    negative and recall would drop to 0.5 — the model punished for our own parsing."""
    gt = _frame([[True], [True], [False], [False]]).astype(bool)
    pred = _frame([[True], [None], [False], [True]])
    m = E.per_tag_metrics(gt, pred, ["a"])
    assert m.n_answered["a"] == 3, m.n_answered["a"]
    assert m.fn["a"] == 0, "an unanswered tag became a false negative"
    assert abs(m.recall["a"] - 1.0) < 1e-9, m.recall["a"]
    assert m.support["a"] == 2, "support must count ground truth, not answered frames"


@check("test_majority_baseline_breaks_ties_toward_the_positive_class")
def _():
    """F2. 2 of 4 positive is a tie; all-True scores 0.667 and all-False scores 0.000."""
    gt = _frame([[True], [True], [False], [False]]).astype(bool)
    b = E.majority_baseline(gt, ["a"])
    assert abs(b["a"] - 2 / 3) < 1e-9, b["a"]
    lopsided = _frame([[True], [False], [False], [False]]).astype(bool)
    assert E.majority_baseline(lopsided, ["a"])["a"] == 0.0


@check("test_all_true_and_all_false_tags_do_not_divide_by_zero")
def _():
    gt = _frame([[False], [False]]).astype(bool)
    pred = _frame([[False], [False]])
    m = E.per_tag_metrics(gt, pred, ["a"])
    assert m.f1["a"] == 0.0 and m.support["a"] == 0
    assert E.majority_baseline(gt, ["a"])["a"] == 0.0


@check("test_parse_failure_rate_excludes_backend_errors_from_the_denominator")
def _():
    """D-045: a backend error is not a reply, so it cannot be a failure to parse."""
    rows = [{"parse_reason": "ok"}, {"parse_reason": "unbalanced_braces"},
            {"backend_error": True, "parse_reason": "RuntimeError: boom"}]
    assert abs(E.parse_failure_rate(rows) - 0.5) < 1e-9
    assert E.parse_failure_rate([]) == 0.0
    assert E.parse_failure_rate([{"parse_reason": "ok_after_trailing_comma_fix"}]) == 0.0


@check("test_set_iou_is_over_true_tags_and_empty_sets_agree")
def _():
    assert E.set_iou({"a": True, "b": False}, {"a": True, "b": True}) == 0.5
    assert E.set_iou({"a": False}, {"a": False}) == 1.0, "two empty sets agree"
    assert E.set_iou({"a": True}, {"a": True}) == 1.0


@check("test_predictions_frame_cuts_a_larger_run_down_to_the_scored_subset")
def _():
    """`v1_structured` holds 1,004 frames; the scored subset is 628 (D-047). Scoring the
    arms over different frame sets would still produce a number, just not a comparable one."""
    rows = [{"sample_token": "t0", "values": {"a": True}},
            {"sample_token": "t1", "values": {"a": False}},
            {"sample_token": "tX", "values": {"a": True}}]
    p = E.predictions_frame(rows, ["t0", "t1"], ["a"])
    assert list(p.index) == ["t0", "t1"], list(p.index)
    assert p.shape == (2, 1)


@check("test_a_backend_error_row_contributes_no_prediction")
def _():
    rows = [{"sample_token": "t0", "backend_error": True, "values": {"a": True}}]
    p = E.predictions_frame(rows, ["t0"], ["a"])
    assert pd.isna(p["a"]["t0"]), "a row the model never answered became a prediction"


@check("test_the_scorer_reproduces_the_four_already_published_arms")
def _():
    """The real check. F-064 published camera 0.389, BEV picture 0.272, BEV text 0.244,
    both 0.375 and a majority baseline of 0.290, over the 34 tags left after F-063's
    exclusion. Those four arms have not been re-run, so any drift here is this module."""
    import src.prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((E.OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    assert len(tokens) == 628, len(tokens)

    published = {"camera": 0.389, "bev_pic": 0.272, "bev_txt": 0.244, "both": 0.375}
    for name, expected in published.items():
        got = E.score_condition(E.CONDITIONS[name], tokens, tags)["macro_34"]
        assert abs(got - expected) < 0.0015, f"{name}: {got:.4f} != published {expected}"

    gt = E.load_gt(tokens, tags)
    scored = [t for t in tags if t not in E.BEV_BLIND_TAGS]
    base = E.majority_baseline(gt, tags)[scored].mean()
    assert abs(base - 0.290) < 0.0015, f"baseline {base:.4f} != published 0.290"


class _FakeNusc:
    """Minimal devkit stand-in: only the four tables `legacy_tags` touches.

    Synthetic so the legacy reproductions can be checked against analytically known
    answers without a 46 s devkit load.
    """

    def __init__(self, cats, xy=None, ego=(0.0, 0.0)):
        self._cats = list(cats)
        self._xy = list(xy) if xy else [(0.0, 0.0)] * len(self._cats)
        self._ego = ego

    def get(self, table, token):
        if table == "sample":
            return {"anns": [f"a{i}" for i in range(len(self._cats))],
                    "data": {"CAM_FRONT": "cam"}}
        if table == "sample_annotation":
            i = int(token[1:])
            return {"category_name": self._cats[i],
                    "translation": [self._xy[i][0], self._xy[i][1], 0.0]}
        if table == "sample_data":
            return {"ego_pose_token": "ego"}
        if table == "ego_pose":
            return {"translation": [self._ego[0], self._ego[1], 0.0]}
        raise KeyError(table)


@check("test_wilson_interval_stays_in_bounds_and_keeps_width_at_zero")
def _():
    """L-021. At k=0 the normal approximation gives a zero-width interval, which would
    report a rare tag scoring 0.000 as if it were known exactly."""
    lo, hi = E.wilson_interval(0, 36)
    assert lo == 0.0 and 0.0 < hi < 0.15, (lo, hi)
    lo, hi = E.wilson_interval(36, 36)
    assert hi == 1.0 and 0.85 < lo < 1.0, (lo, hi)
    lo, hi = E.wilson_interval(50, 100)
    assert lo < 0.5 < hi and 0.0 <= lo and hi <= 1.0
    assert E.wilson_interval(0, 0) == (0.0, 0.0)


@check("test_mean_tag_iou_drops_unanswered_tags_from_both_sets")
def _():
    """D-005's column must not punish the model for our parsing (the F1 rule again)."""
    gt = _frame([[True, True]], tags=("a", "b")).astype(bool)
    both = E.mean_tag_iou(gt, _frame([[True, True]], tags=("a", "b")), ["a", "b"])
    assert both == 1.0, both
    # 'b' unanswered: the frame reduces to {a} vs {a}, not to a missed positive
    part = E.mean_tag_iou(gt, _frame([[True, None]], tags=("a", "b")), ["a", "b"])
    assert part == 1.0, part
    half = E.mean_tag_iou(gt, _frame([[True, False]], tags=("a", "b")), ["a", "b"])
    assert abs(half - 0.5) < 1e-9, half


@check("test_legacy_v1_hardcodes_the_three_temporal_tags_false")
def _():
    """The defect PLAN.md's Stage C intro is written about: three of six tags were
    hardcoded False with the comment 'cannot derive from static annotations'."""
    n = _FakeNusc(["vehicle.car", "human.pedestrian.adult", "movable_object.barrier"])
    t = E.legacy_tags(n, "tok", "v1_presence")
    assert t["has_parked_vehicle"] and t["has_pedestrian"] and t["has_construction"]
    assert t["requires_slowdown"] is False
    assert t["turn_left"] is False
    assert t["following_vehicle"] is False


@check("test_legacy_v3_following_vehicle_is_presence_not_geometry")
def _():
    """THE F3 regression. v3 is the variant that produced the published IoU, and its
    `following_vehicle` fires on any car/truck/bus ANYWHERE in the 360 set — including
    one 200 m behind the ego. Reproducing it as a lead test would silently make the old
    reference look better than it was and understate what it cost."""
    behind = _FakeNusc(["vehicle.car"], xy=[(-200.0, 0.0)], ego=(0.0, 0.0))
    t = E.legacy_tags(behind, "tok", "v3_fair")
    assert t["following_vehicle"] is True, "v3 must be presence-based, distance-blind"
    assert set(t) == {"has_parked_vehicle", "has_construction", "has_pedestrian",
                      "following_vehicle"}, "v3 scored 4 tags; temporal ones were dropped"
    none = E.legacy_tags(_FakeNusc(["human.pedestrian.adult"]), "tok", "v3_fair")
    assert none["following_vehicle"] is False


@check("test_legacy_v2_reproduces_the_global_frame_lead_bug")
def _():
    """F-033: `abs(dy) > abs(dx)` is two wedges pinned to global north/south, so a car
    directly BEHIND the ego on the global Y axis counts as 'ahead'. Reproduced on
    purpose — F3 measures what the bug cost."""
    behind_on_y = _FakeNusc(["vehicle.car"], xy=[(0.0, -10.0)], ego=(0.0, 0.0))
    assert E.legacy_tags(behind_on_y, "tok", "v2_gated")["following_vehicle"] is True
    beside_on_x = _FakeNusc(["vehicle.car"], xy=[(10.0, 0.0)], ego=(0.0, 0.0))
    assert E.legacy_tags(beside_on_x, "tok", "v2_gated")["following_vehicle"] is False
    far = _FakeNusc(["vehicle.car"], xy=[(0.0, -38.0)], ego=(0.0, 0.0))
    assert far and E.legacy_tags(far, "tok", "v2_gated")["has_parked_vehicle"] is False


@check("test_every_published_csv_has_a_builder")
def _():
    """R30 for TABLES, not just figures.

    `test_every_published_figure_has_a_builder` has guarded `outputs/*.png` since F-049,
    but the identical debt sat unguarded for CSVs and three ad-hoc transcript tables had
    accumulated in `outputs/` with no generator anywhere in `src/` — the exact thing R30
    exists to prevent, escaping only because the existing guard globs `*.png`.
    """
    import re

    # Scans ALL of src/, not just eval.py. The guard was written when every table came
    # from the scorer; `g2_segmentation.csv` is built in `src/storyboard.py` and tripped it
    # while being perfectly reproducible. A guard that fires on a table WITH a builder
    # teaches the next reader to disable it, which is worse than not having it.
    known = set()
    for mod in sorted((ROOT / "src").glob("*.py")):
        txt = mod.read_text()
        known |= set(re.findall(r'OUT / "([^"]+\.csv)"', txt))
        known |= {m.rsplit("/", 1)[-1] for m in re.findall(r'"(outputs/[^"]+\.csv)"', txt)}
    published = {p.name for p in E.OUT.glob("*.csv")}
    orphans = sorted(published - known)
    assert not orphans, (
        f"published with no builder anywhere in src/: {orphans}. "
        "Give each a `build_*` with a default `out_csv`, or delete it (R30).")


@check("test_per_tag_detail_rows_are_real_confusion_matrices")
def _():
    """PLAN F1 asks for a confusion matrix per tag. A row that does not close is not one.

    Also checks the Wilson intervals bracket their own point estimate — an interval that
    excludes the value it describes would be worse than no interval at all (L-021).
    """
    df = E.build_per_tag_detail_table(out_csv=E.OUT / "f1_per_tag_detail.csv")
    assert len(df) == 35 * len(E.conditions_available()), len(df)
    closes = (df.tp + df.fp + df.fn + df.tn) == df.n_answered
    assert closes.all(), f"{(~closes).sum()} rows where tp+fp+fn+tn != n_answered"
    assert (df.precision_lo <= df.precision + 1e-9).all()
    assert (df.precision <= df.precision_hi + 1e-9).all()
    assert (df.recall_lo <= df.recall + 1e-9).all()
    assert (df.recall <= df.recall_hi + 1e-9).all()
    cam = df[df.condition == "camera"].set_index("tag")
    assert abs(cam.f1["has_vehicle"] - 0.969) < 0.0015, cam.f1["has_vehicle"]


@check("test_family_table_reproduces_the_published_family_numbers")
def _():
    """F-066 published the family x condition table. F4 recomputes it from
    score_condition rather than from the rounded per-tag CSV, so it must still agree."""
    fam = E.build_family_table(out_csv=E.OUT / "f4_family_conditions.csv")
    published = {("visible_objects", "camera"): 0.620, ("visible_objects", "cot"): 0.675,
                 ("map_context", "cot"): 0.436, ("map_context", "bev_pic"): 0.349,
                 ("ego_maneuver", "bev_txt"): 0.005, ("interaction", "cot"): 0.298}
    for (f, c), want in published.items():
        assert abs(fam.loc[f, c] - want) < 0.0015, f"{f}/{c}: {fam.loc[f, c]} != {want}"
    assert fam.loc["map_context", "baseline"] > fam.loc["map_context"][
        ["camera", "cot", "bev_pic", "bev_txt", "both"]].max(), \
        "map_context is supposed to lose to its own majority baseline in every arm"


@check("test_the_builders_iterate_conditions_available_not_the_frozen_five")
def _():
    """THE E9 REGRESSION.

    `conditions_available()` correctly adds the temporal arm once its results file
    exists -- but all three builders iterated `CONDITIONS`, the frozen five, while the
    comment above `TEMPORAL_CONDITION` claimed they iterated `conditions_available()`.
    E9's results would have been dropped from every published table with no error and no
    crash, just a missing column, and notebooks/README.md told the reader the opposite.

    Same class as F-016's silent tag drift and F-049's stale figure: a claim the code
    does not honour, invisible in the output.
    """
    import re

    src = (Path(__file__).resolve().parent.parent / "src" / "eval.py").read_text()
    assert "for name, stem in CONDITIONS.items():" not in src, (
        "a builder iterates the frozen CONDITIONS dict, so a condition that HAS been run "
        "is silently dropped from its table. Iterate conditions_available() instead.")
    n = len(re.findall(r"for name, stem in conditions_available\(\)\.items\(\):", src))
    assert n >= 3, f"expected all three builders to iterate conditions_available(), got {n}"



# --- Step F6: paired significance and cluster-robust intervals -----------------------
#
# R31: these sit ABOVE the runner. Tests appended after `if __name__ == "__main__"`
# silently never execute.


@check("test_a_unit_weight_bootstrap_draw_reproduces_the_point_estimate_exactly")
def _():
    """THE IDENTITY THE WHOLE SECTION RESTS ON.

    `_cell_counts` re-expresses `per_tag_metrics`' exclusion rule as arithmetic: an
    unanswered cell contributes 0 to tp, fp and fn instead of being masked out. If the two
    ever disagree, every bootstrap interval is centred on a different quantity than the
    point estimate it is published beside, and nothing in the output would show it.
    """
    import numpy as np

    gt = _frame([[True, True], [True, False], [False, True], [False, False]],
                tags=("a", "b")).astype(bool)
    pred = _frame([[True, None], [False, False], [False, True], [True, True]],
                  tags=("a", "b"))
    tags = ["a", "b"]
    clusters = ["s0", "s0", "s1", "s1"]
    one = np.ones((1, 2), dtype=np.int64)
    boot = E.bootstrap_macro(gt, pred, tags, tags, clusters, weights=one)[0]
    point = E.per_tag_metrics(gt, pred, tags)["f1"].mean()
    assert abs(boot - point) < 1e-12, f"{boot} != {point}"


@check("test_an_arm_against_itself_is_exactly_zero_with_p_of_one")
def _():
    """The pairing sanity check. If the two arms did NOT share a resample, an arm against
    itself would show a non-zero spread from sampling noise alone, and every delta in the
    table would be inflated by the variance the pairing exists to cancel."""
    gt = _frame([[True], [True], [False], [False], [True], [False]]).astype(bool)
    pred = _frame([[True], [False], [False], [True], [True], [False]])
    r = E.paired_delta(gt, pred, pred, ["a"], ["a"], ["s0", "s0", "s1", "s1", "s2", "s2"],
                       n_draws=200)
    assert r["delta"] == 0.0 and r["delta_lo"] == 0.0 and r["delta_hi"] == 0.0, r
    assert r["p"] == 1.0, r["p"]


@check("test_holm_is_step_down_and_monotone")
def _():
    adj = E.holm([0.01, 0.02, 0.04])
    assert [round(a, 4) for a in adj] == [0.03, 0.04, 0.04], adj    # 3x, 2x, then carried
    assert all(a >= b for a, b in zip(adj, [0.01, 0.02, 0.04])), "adjusted below raw"
    assert E.holm([0.6, 0.9]) == [1.0, 1.0], "adjustment must clip at 1"
    assert E.holm([0.001]) == [0.001], "a single comparison needs no correction"


@check("test_clustering_widens_the_interval_on_the_real_subset")
def _():
    """F-082's measurement, as a regression.

    The 628 scored frames are 139 scenes of correlated keyframes (ICC 0.517 on per-frame
    accuracy, mean cluster 4.50, design effect 2.81). If `design_effect` ever returns a
    ratio near 1, either the clustering variable stopped being read or the resample stopped
    using it — and the published intervals would quietly go back to being too narrow.
    """
    import json

    import src.prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    scored = [t for t in tags if t not in E.BEV_BLIND_TAGS]
    tokens = json.loads((ROOT / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = E.load_gt(tokens, tags)
    clusters = E._scene_of(tokens)
    assert clusters.nunique() == 139, clusters.nunique()
    pred = E.predictions_frame(E.load_rows("v1_structured"), tokens, tags)
    de = E.design_effect(gt, pred, tags, scored, clusters, n_draws=500)
    assert de["width_ratio"] > 1.5, de
    assert 1.5 < de["width_ratio"] < 2.5, f"far from sqrt(2.81)=1.68: {de}"


@check("test_the_degenerate_flag_covers_recall_not_only_f1")
def _():
    """THE REGRESSION FOR MY OWN BUG.

    The flag was first written as `f1_lo == f1_hi`, which left 9 (condition, tag) cells
    unflagged where the model answers True on EVERY frame: fn == 0, so recall is exactly
    1.000 in every resample and its interval collapses while F1's does not. A collapsed
    interval published unflagged claims a precision the resample cannot deliver.
    """
    import pandas as pd

    det = pd.read_csv(ROOT / "outputs/f1_per_tag_detail.csv")
    zero = ((det.precision_hi_clust - det.precision_lo_clust == 0)
            | (det.recall_hi_clust - det.recall_lo_clust == 0)
            | (det.f1_hi_clust - det.f1_lo_clust == 0))
    assert (zero == det.boot_degenerate).all(), \
        "a zero-width clustered interval is published without its flag"
    assert det.boot_degenerate.sum() > 0, "no degenerate cells at all is itself suspicious"
    nd = det[~det.boot_degenerate]
    for c in ("precision", "recall", "f1"):
        assert (nd[f"{c}_lo_clust"] <= nd[c] + 1e-4).all(), c
        assert (nd[c] <= nd[f"{c}_hi_clust"] + 1e-4).all(), c


@check("test_the_two_averages_disagree_about_two_arms")
def _():
    """F-083, frozen as a test.

    Mean-of-per-tag and pooled-counts F1 are not the same number and not the same
    ORDERING: `both` and `temporal` lose to the camera under macro and are above it under
    micro. If a refactor ever collapses the two definitions into one, this fails instead
    of silently rewriting two of the thesis's conclusions.
    """
    import pandas as pd

    df = pd.read_csv(ROOT / "outputs/f6_significance.csv").set_index(["average", "condition"])
    for cond in ("both", "temporal"):
        assert df.loc[("macro", cond), "delta"] < 0, cond
        assert bool(df.loc[("macro", cond), "significant_05"]), cond
        assert df.loc[("micro", cond), "delta"] > 0, cond
        assert not bool(df.loc[("micro", cond), "significant_05"]), cond
    for cond in ("cot", "bev_pic", "bev_txt"):
        assert (df.loc[("macro", cond), "delta"] * df.loc[("micro", cond), "delta"]) > 0, cond
        assert bool(df.loc[("macro", cond), "significant_05"]), cond
        assert bool(df.loc[("micro", cond), "significant_05"]), cond


@check("test_the_text_reader_rules_on_known_descriptions")
def _():
    """The reader ceiling is only a ceiling if its rules read the description correctly.
    Synthetic lines with known answers, in the renderer's own format, including the
    corridor edges: a fixture 4 m off-axis is in, one beyond 30 m ahead or behind is out."""
    d = ("\n  - stop line: the vehicle is standing on it; present out to 26 m ahead\n"
         "  - ped crossing: nearest 9 m, bearing +101 deg; present out to 0 m ahead\n"
         "  - fixture: 31 m, bearing +0 deg\n  - fixture: 11 m, bearing -179 deg\n")
    assert "standing on it" in E._reader_line(d, "stop line")
    assert "standing on it" not in E._reader_line(d, "ped crossing")
    assert E._reader_line(d, "walkway") == ""
    assert not E._reader_fixture_in_corridor(d, 30.0, 4.0), "31 m ahead or behind is outside"
    edge = "  - fixture: 20 m, bearing +11 deg\n"      # y = 20 sin(11 deg) = 3.82 m
    assert E._reader_fixture_in_corridor(edge, 30.0, 4.0)
    assert not E._reader_fixture_in_corridor("  - fixture: 20 m, bearing +12 deg\n", 30.0, 4.0)


@check("test_the_repaired_text_states_the_containment_tags_exactly")
def _():
    """The reason the text arm's gain is reported as reading, not perception: a rule that
    only reads the description reproduces the containment and fixture tags. If the renderer
    or the reference changes and this stops holding, the thesis sentence is wrong."""
    if not (E.RESULTS_DIR / "v3b_bev_symbolic_r2.txt").exists() and \
            not (E.RESULTS_DIR / "v3b_bev_symbolic_r2.jsonl").exists():
        return
    df = E.build_reader_ceiling_table(out_csv=None).set_index("tag")
    for tag in ("on_stop_line", "on_ped_crossing", "traffic_light_ahead"):
        assert df.loc[tag, "ceiling"] == "exact"
        assert df.loc[tag, "reader_f1"] >= 0.99, (tag, df.loc[tag, "reader_f1"])
    assert (df.reader_f1 > df.model_f1).all(), "the model beat a reader of its own prompt"


@check("test_the_two_model_slices_stay_separate_tables")
def _():
    """Both slices run the same alternative model on the same 150 frames and differ only in
    what it was shown, so one builder serves both. It must be told which: a hardcoded slice
    silently scored the camera pair under the repaired raster's name."""
    if not E._has_results(E.BEV_MODEL_SLICE[1]):
        return
    cam, _ = E.build_model_comparison_table(out_csv=None)
    bev, meta = E.build_model_comparison_table(out_csv=None, slice_spec=E.BEV_MODEL_SLICE)
    assert meta["n_frames"] == 150 and meta["n_scenes_spanned"] == 30
    assert not cam.equals(bev), "both slices produced the same table"
    # the reason the run was worth the GPU: the glyph the 7B never reported is readable
    assert bev.loc["traffic_light_ahead", "qwen3_vl_8b"] > 0.2, bev.loc["traffic_light_ahead"]
    assert cam.loc["traffic_light_ahead", "qwen2_5_vl_7b"] > 0.5, "camera pair got swapped in"

if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} eval self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
