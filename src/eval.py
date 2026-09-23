"""Evaluation — Stage F.

The headline methodological change: per-frame set-IoU over 3-6 binary tags is far
noisier than the effect being measured (one flipped tag on a 2-true-tag frame swings
IoU by 0.33). Report per-tag precision / recall / F1 / support instead.

Every Stage E table is built here rather than in a notebook cell or a transcript
heredoc (R30). The five condition tables were scored three times ad hoc before this
module existed, which is exactly the artifact-without-a-generator R30 forbids.

Build:  ./venv/bin/python -c "import src.eval as e; e.build_condition_table()"
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RESULTS_DIR = OUT / "results"

# `traffic_light_ahead` is unanswerable from either BEV arm: src/bev.py never draws the
# traffic_light map layer (F-063). It stays in the per-tag table, and is excluded from
# any macro that compares INPUTS, so the exclusion is visible instead of silent.
BEV_BLIND_TAGS = ("traffic_light_ahead",)

# extract_json's two success reasons. Anything else is RQ2b's numerator.
PARSE_OK = ("ok", "ok_after_trailing_comma_fix")

# Condition -> results stem. One place, because a table keyed by a typo'd stem silently
# scores the wrong arm.
CONDITIONS = {
    "camera": "v1_structured",
    "cot": "v2_cot",
    "bev_pic": "v3_bev",
    "bev_txt": "v3b_bev_symbolic",
    "both": "v4_both",
}

# E9's arm, scored only once its results file exists. Kept OUT of `CONDITIONS` until then
# so `build_condition_table()` does not start raising FileNotFoundError on a run that has
# not happened yet; `conditions_available()` is what the builders iterate.
TEMPORAL_CONDITION = ("temporal", "v6_temporal")

# The REPAIRED BEV arms (F-063c, F-064b): same prompt shape, same frames, same model, but
# the raster now draws the traffic-light layer the tag derives from and the description
# states where each road surface is. Kept out of CONDITIONS on the same principle as the
# two above -- an arm that has not run is not a 0.00 row -- and picked up automatically the
# moment its results file lands. The published `bev_pic` and `bev_txt` rows stay, because
# the point of the re-run is the DIFFERENCE between the two.
REPAIRED_CONDITIONS = (("bev_pic_fixed", "v3_bev_r2", "bev_pic"),
                       ("bev_txt_fixed", "v3b_bev_symbolic_r2", "bev_txt"))

# E10's model-sensitivity slice (D-048). An ALTERNATIVE model on the same prompt and the same
# frames as the primary arm, so the only thing that varies is the model. Kept out of
# CONDITIONS because it is not an input condition: mixing it in would put two models in one
# column of a table whose whole point is that the model is held constant.
MODEL_SLICE = ("Qwen3-VL-8B", "v1_structured__Qwen3-VL-8B-Instruct", "v1_structured", 150)

# The same alternative model on the REPAIRED raster, over the identical 150 frames, so the
# two slices differ only in what the model was shown. Asks whether the 7B's inability to
# read the diagram is the model's or the representation's.
BEV_MODEL_SLICE = ("Qwen3-VL-8B", "v3_bev_r2__Qwen3-VL-8B-Instruct", "v3_bev_r2", 150)


def _has_results(stem: str) -> bool:
    return any((RESULTS_DIR / f"{stem}{ext}").exists() for ext in (".jsonl", ".txt"))


def conditions_available(include_repaired: bool = False) -> dict[str, str]:
    """The five measured arms, plus E9's temporal arm once it has been run.

    A condition with no results file is not a zero — it is an experiment that has not
    happened, and scoring it as absent would put a fabricated 0.00 row in a results table.

    The repaired BEV arms are NOT in the main family unless asked for. When they joined it
    automatically, their results landing rewrote the per-tag and family tables under the
    thesis figures, would have re-run Holm over seven comparisons instead of five, and
    crashed the results appendix. They answer a different question (did the repair change
    the arm?), which `build_bev_repair_table` asks of them directly.
    """
    out = dict(CONDITIONS)
    name, stem = TEMPORAL_CONDITION
    if _has_results(stem):
        out[name] = stem
    if include_repaired:
        out.update({n: st for n, st, _pub in REPAIRED_CONDITIONS if _has_results(st)})
    return out


def build_bev_repair_table(out_csv: Path | str = OUT / "bev_repair.csv") -> Any:
    """Published BEV arm vs its repaired re-run, per tag and per family.

    The reason the re-run is worth ten hours of quota: F-082 has both BEV arms losing to
    the camera by 0.10 to 0.15 with Holm-corrected significance, and two of our own defects
    bounded that reading. This table turns "a renderer bug cannot explain the gap" from an
    assertion into a measurement. Both arms saw the same 628 frames and the same reference,
    so the row-wise difference is the repair and nothing else.

    Returns an empty frame with the right columns until the re-run exists, so a caller can
    build it before the GPU session without special-casing (the absent arm is an experiment
    that has not happened, never a zero).
    """
    import pandas as pd

    from . import prompts as P

    cols = ["arm", "scope", "name", "published_f1", "repaired_f1", "delta", "n_positive"]
    tags = P.scoreable_tags(P.load_schema())
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    rows: list[dict[str, Any]] = []
    for _name, stem, published in REPAIRED_CONDITIONS:
        if not _has_results(stem):
            continue
        a = score_condition(CONDITIONS[published], tokens, tags)["per_tag"]
        b = score_condition(stem, tokens, tags)["per_tag"]
        for tag in sorted(tags):
            rows.append({"arm": published, "scope": "tag", "name": tag,
                         "published_f1": round(float(a.f1[tag]), 4),
                         "repaired_f1": round(float(b.f1[tag]), 4),
                         "delta": round(float(b.f1[tag] - a.f1[tag]), 4),
                         "n_positive": int(a.support[tag])})
    df = pd.DataFrame(rows, columns=cols)
    if out_csv is not None:
        df.to_csv(Path(out_csv), index=False)
    return df


def _reader_line(desc: str, layer: str) -> str:
    """The one line the repaired description writes for a map layer, or ''."""
    m = re.search(rf"\n  - {layer}: ([^\n]*)", desc)
    return m.group(1) if m else ""


def _reader_fixture_in_corridor(desc: str, ahead_m: float, halfwidth_m: float) -> bool:
    for r, bearing in re.findall(r"fixture: (\d+) m, bearing ([+-]?\d+) deg", desc):
        th = math.radians(int(bearing))
        x, y = int(r) * math.cos(th), int(r) * math.sin(th)
        if 0 <= x <= ahead_m and abs(y) <= halfwidth_m:
            return True
    return False


def build_reader_ceiling_table(out_csv: Path | str | None = OUT / "bev_txt_reader_ceiling.csv",
                               stem: str = "v3b_bev_symbolic_r2") -> Any:
    """How much of the repaired text arm's gain is reading back an answer the text states.

    The repaired description says, for example, "stop line: the vehicle is standing on it",
    derived from the same polygons as the reference. So each map-point tag is scored twice
    on the same 628 descriptions: by the model, and by a fixed rule that only reads the
    text. The rule's F1 is the ceiling a perfect reader reaches with no perception at all.

    The "ahead" rows for polygon layers are LOWER bounds on that ceiling: the text gives how
    far a layer extends along four rays, not whether it meets the corridor, so the rule
    ("extends any distance ahead") is the best the stated field allows. Fixtures are given
    by range and bearing, so that row applies the schema's own corridor exactly.
    """
    import pandas as pd

    from . import prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = load_gt(tokens, tags)
    pred = predictions_frame(load_rows(stem), tokens, tags)
    desc = {json.loads(l)["sample_token"]: json.loads(l)["description"]
            for l in (OUT / "bev_symbolic_r2.jsonl").open()}
    cp = schema["tags"]["traffic_light_ahead"]["parameters"]
    ahead = lambda s: int((re.search(r"present out to (\d+) m ahead", s) or [0, "0"])[1]) > 0
    rules = {
        "on_stop_line": ("exact", lambda d: "standing on it" in _reader_line(d, "stop line")),
        "on_ped_crossing": ("exact", lambda d: "standing on it" in _reader_line(d, "ped crossing")),
        "traffic_light_ahead": ("exact", lambda d: _reader_fixture_in_corridor(
            d, cp["AHEAD_RANGE_M"], cp["CORRIDOR_HALFWIDTH_M"])),
        "stop_line_ahead": ("lower_bound", lambda d: ahead(_reader_line(d, "stop line"))),
        "ped_crossing_ahead": ("lower_bound", lambda d: ahead(_reader_line(d, "ped crossing"))),
    }
    rows = []
    for tag, (kind, rule) in rules.items():
        reader = pd.DataFrame({tag: [rule(desc[t]) for t in tokens]}, index=gt.index)
        f_reader = float(per_tag_metrics(gt, reader, [tag]).f1[tag])
        f_model = float(per_tag_metrics(gt, pred, [tag]).f1[tag])
        rows.append({"tag": tag, "ceiling": kind, "n_positive": int(gt[tag].sum()),
                     "reader_true": int(reader[tag].sum()), "model_true": int(pred[tag].sum()),
                     "reader_f1": round(f_reader, 4), "model_f1": round(f_model, 4)})
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(Path(out_csv), index=False)
    return df


def load_rows(stem: str | Path) -> list[dict[str, Any]]:
    """Read a results JSONL. Accepts the `.txt` Kaggle writes and the `.jsonl` we keep.

    Rows are returned verbatim: scoring decisions belong to the functions below, not to
    the loader, so a re-score never needs the GPU again (run_batch's contract).
    """
    p = Path(stem)
    if not p.exists():
        for ext in (".jsonl", ".txt"):
            cand = RESULTS_DIR / f"{stem}{ext}"
            if cand.exists():
                p = cand
                break
        else:
            raise FileNotFoundError(f"no results file for {stem!r}")
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def predictions_frame(rows: list[dict], tokens: list[str], tags: list[str]) -> Any:
    """Rows -> DataFrame[token x tag] of True / False / None.

    None is "the model did not answer this tag", never False — the conflation coerce_tags
    exists to prevent. Reindexed onto `tokens` so a run covering MORE frames than the
    scored subset is cut down to it: `v1_structured` holds 1,004 frames while the scored
    subset is 628 (D-047), and scoring the arms over different frame sets would make the
    comparison meaningless while still producing a plausible-looking number.
    """
    import pandas as pd

    by_token = {}
    for r in rows:
        if r.get("backend_error"):
            continue                       # no observation at all (D-045)
        by_token[r["sample_token"]] = r.get("values") or {}
    data = [{t: by_token.get(tok, {}).get(t) for t in tags} for tok in tokens]
    return pd.DataFrame(data, index=pd.Index(tokens, name="sample_token"), dtype=object)


def load_gt(tokens: list[str], tags: list[str], path: Path | str | None = None) -> Any:
    """Ground truth for these tokens, as bools. Step F1."""
    import pandas as pd

    gt = pd.read_parquet(path or OUT / "gt_all.parquet").set_index("sample_token")
    missing = set(tokens) - set(gt.index)
    assert not missing, f"{len(missing)} scored tokens absent from ground truth"
    return gt.loc[tokens, tags].astype(bool)


def per_tag_metrics(gt: Any, pred: Any, tags: list[str]) -> Any:
    """Precision, recall, F1, support per tag, aggregated over the whole subset.

    Returns a DataFrame with one row per tag. Step F1.

    A tag the model did not answer is EXCLUDED from that tag's counts rather than scored
    as False, and `n_answered` records how many frames survived, so "did not answer" can
    never masquerade as "said no". Chain-of-thought is the first condition where this
    matters at all: 3 of 628 rows (F-066).

    Precision and recall carry Wilson 95% intervals (L-021). They are attached HERE rather
    than in one report, because every downstream table calls this function, and a rare
    tag's point estimate is misleading wherever it is read: `is_u_turn` rests on 36
    positives and several conditions score it a flat 0.000.
    """
    import pandas as pd

    out = []
    for t in tags:
        p = pred[t]
        answered = p.notna()
        y = gt[t][answered].astype(bool)
        yhat = p[answered].astype(bool)
        tp = int((y & yhat).sum())
        fp = int((~y & yhat).sum())
        fn = int((y & ~yhat).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        p_lo, p_hi = wilson_interval(tp, tp + fp)
        r_lo, r_hi = wilson_interval(tp, tp + fn)
        out.append({"tag": t, "precision": prec, "recall": rec, "f1": f1,
                    "support": int(gt[t].sum()), "n_answered": int(answered.sum()),
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision_lo": p_lo, "precision_hi": p_hi,
                    "recall_lo": r_lo, "recall_hi": r_hi})
    return pd.DataFrame(out).set_index("tag")


def confusion_by_tag(gt: Any, pred: Any, tags: list[str]) -> dict[str, Any]:
    """2x2 confusion matrix per tag. Step F1."""
    m = per_tag_metrics(gt, pred, tags)
    return {t: {"tp": int(m.tp[t]), "fp": int(m.fp[t]), "fn": int(m.fn[t]),
                "tn": int(m.n_answered[t] - m.tp[t] - m.fp[t] - m.fn[t])} for t in tags}


def majority_baseline(gt: Any, tags: list[str]) -> Any:
    """Predict each tag's most frequent value.

    Without this, a high F1 on an imbalanced tag means nothing. If the VLM does not
    beat this, that IS the result. Step F2.

    Ties break toward the POSITIVE class. `lead_vehicle` is positive on exactly 314 of
    628 frames, and the choice is worth 0.667 F1 on that tag alone (all-True scores
    0.667, all-False scores 0.000) — so the tie-break is not a rounding detail. Breaking
    toward True makes the baseline the STRONGER trivial predictor, i.e. the harder thing
    for the model to beat; the alternative would flatter our own hypothesis.
    """
    import pandas as pd

    n = len(gt)
    out = {}
    for t in tags:
        pos = int(gt[t].sum())
        if pos * 2 >= n:                   # majority True (ties included)
            tp, fp, fn = pos, n - pos, 0
        else:
            tp, fp, fn = 0, 0, pos
        out[t] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return pd.Series(out, name="baseline_f1")


def set_iou(gt_tags: dict[str, bool], pred_tags: dict[str, bool]) -> float:
    """Intersection-over-union of the true-tag sets.

    Kept ONLY for continuity with the earlier results in docs/nuscens.py so the
    before/after comparison in Step F3 is like-for-like. Do not lead with it.
    """
    g = {k for k, v in gt_tags.items() if v}
    p = {k for k, v in pred_tags.items() if v}
    union = g | p
    return 1.0 if not union else len(g & p) / len(union)


def metrics_by_family(gt: Any, pred: Any, schema: dict[str, dict]) -> Any:
    """F1 grouped by label family (ego maneuver / map context / objects).

    This is the shape of the headline result: front vs BEV vs both, per family.
    Expectation to test, not assume: front camera wins on object presence, BEV wins
    on maneuver and geometry. Step F4.
    """
    import pandas as pd

    tags = [t for t in pred.columns]
    fam = pd.Series({t: schema["tags"][t]["family"] for t in tags})
    f1 = per_tag_metrics(gt, pred, tags)["f1"]
    return f1.groupby(fam).mean()


def parse_failure_rate(pred: list[dict]) -> float:
    """Fraction of frames where the model did not emit valid JSON. Step E2 / F1.

    Takes the raw rows, not the frame. Backend errors are excluded from the denominator:
    the model never replied, so there is nothing to have failed to parse (D-045).
    """
    calls = [r for r in pred if not r.get("backend_error")]
    if not calls:
        return 0.0
    bad = sum(1 for r in calls if r.get("parse_reason") not in PARSE_OK)
    return bad / len(calls)


def score_condition(stem: str, tokens: list[str], tags: list[str]) -> dict[str, Any]:
    """Everything measurable about one arm, from its results file."""
    rows = load_rows(stem)
    pred = predictions_frame(rows, tokens, tags)
    gt = load_gt(tokens, tags)
    m = per_tag_metrics(gt, pred, tags)
    scored = [t for t in tags if t not in BEV_BLIND_TAGS]
    return {
        "rows": rows, "pred": pred, "gt": gt, "per_tag": m,
        "macro_35": float(m["f1"].mean()),
        "macro_34": float(m.loc[scored, "f1"].mean()),
        "tag_iou": mean_tag_iou(gt, pred, tags),
        "parse_failure_rate": parse_failure_rate(rows),
        "n_rows": len(rows),
        "n_unanswered_cells": int(pred.isna().sum().sum()),
    }


def build_condition_table(out_csv: Path | str = OUT / "e_all5_f1.csv") -> Any:
    """The five-condition per-tag F1 table, plus support and the majority baseline.

    This regenerates the artifact every Stage E claim is read off (R30).
    """
    import pandas as pd

    from . import prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = load_gt(tokens, tags)

    cols, meta = {}, {}
    for name, stem in conditions_available().items():
        s = score_condition(stem, tokens, tags)
        cols[name] = s["per_tag"]["f1"].round(3)
        meta[name] = {k: s[k] for k in
                      ("macro_34", "macro_35", "tag_iou", "parse_failure_rate",
                       "n_unanswered_cells")}
    df = pd.DataFrame(cols)
    df["family"] = [schema["tags"][t]["family"] for t in df.index]
    df["support"] = [int(gt[t].sum()) for t in df.index]
    df["baseline_f1"] = majority_baseline(gt, tags).round(3)
    df.to_csv(out_csv)
    return df, pd.DataFrame(meta).T


# --- Step F1 / F2 completion: intervals, and the IoU continuity column ---------------

WILSON_Z = 1.96   # 95%. L-021 names the tags that need an interval rather than a point
                  # estimate: in the 628-frame execution subset `is_u_turn` has 36
                  # positives, `lead_braking` 108, `cut_in` 109. At n=36 the normal
                  # approximation is invalid and can leave [0, 1] entirely.


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a proportion. Step F1, required by L-021.

    Wilson rather than the normal approximation because it stays inside [0, 1] and keeps
    a non-zero width at k == 0 and k == n — exactly the regime the rare tags sit in, where
    several conditions score a flat 0.000 (F-062: `cut_in`, `is_u_turn`).

    Applies to PRECISION and RECALL, which are genuine proportions (tp/(tp+fp),
    tp/(tp+fn)). It does NOT apply to F1, a ratio of sums rather than a proportion, so no
    interval is reported for F1 instead of reporting one that does not mean what it says.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_tag_iou(gt: Any, pred: Any, tags: list[str]) -> float:
    """Mean per-frame set-IoU over the true-tag sets. D-005's continuity column.

    The slides name IoU on tags as THE evaluation criterion, so it is reported beside
    per-tag F1 in every table even though F1 leads (D-005). Kept honest in two ways: a
    tag the model did not answer is dropped from BOTH sets for that frame rather than
    read as False (the same rule as `per_tag_metrics`), and two empty sets count as
    agreement — which is the convention `docs/nuscens.py:772` used, so the F3
    before/after comparison is like-for-like rather than differing by a scoring quirk.
    """
    import pandas as pd

    vals = []
    for tok in pred.index:
        g, p = {}, {}
        for t in tags:
            v = pred[t][tok]
            if not pd.isna(v):
                p[t] = bool(v)
                g[t] = bool(gt[t][tok])
        vals.append(set_iou(g, p))
    return float(sum(vals) / len(vals)) if vals else 0.0


# --- Step F3: what the old reference cost -------------------------------------------
#
# `docs/nuscens.py` defines the ground truth THREE times and the three disagree, so "the
# old ground truth" is not one thing. All three are reproduced verbatim here and the
# table reports all three, because picking one would be a preference where the file's own
# call graph is evidence (R5). Which is which, checked against the call sites:
#
#   v1_presence  `extract_gt_tags_from_annotations` (line 286), called by
#                `evaluate_sample` (line 334). Pure 360-degree category presence;
#                `requires_slowdown`, `turn_left` and `following_vehicle` are hardcoded
#                False with the comment "cannot derive from static annotations".
#   v2_gated     `extract_smart_gt_tags` (line 930), live in the agentic arm (lines 979,
#                1190, 1226). Distance-gated, and carries the global-frame
#                `abs(dy) > abs(dx)` lead test F-033 measured at 12.2% precision.
#   v3_fair      `evaluate_fair_tags`, defined TWICE (lines 738 and 986 — the genuine
#                later-wins redefinition PLAN.md flagged) with the two bodies agreeing.
#                THIS is the one that produced the published IoU, so it is the reference
#                the thesis narrative is actually about. Only 4 tags: the temporal ones
#                were dropped rather than scored. Its `following_vehicle` is pure
#                car/truck/bus PRESENCE — weaker even than v2's broken geometry.
#
# Corroboration that the reconstruction is faithful: v3 scored on our frames reproduces a
# mean set-IoU of ~0.545 against the 0.533 the original reported on 50 sequential mini
# keyframes — a different frame set, so agreement to ~0.01 is the expected shape.
LEGACY_VARIANTS = ("v1_presence", "v2_gated", "v3_fair")

# Legacy tag -> the frozen-schema tag whose MODEL ANSWER is the comparable prediction.
# The comparison holds the model output fixed and swaps only the reference, which is what
# makes the delta attributable to the ground truth rather than to the model.
LEGACY_TO_SCHEMA = {
    "has_parked_vehicle": "has_vehicle",
    "has_construction": "construction_object",
    "has_pedestrian": "has_pedestrian",
    "following_vehicle": "lead_vehicle",
    "turn_left": "is_turn_left",
    "requires_slowdown": "is_decelerating",
}

# v2's gates, verbatim from docs/nuscens.py:945-970. Named rather than inlined so the
# reproduction is auditable, but deliberately NOT tuned — these are the old numbers.
_V2_MAX_M, _V2_VEH_M, _V2_AHEAD_M, _V2_CON_M, _V2_PED_M = 40.0, 20.0, 15.0, 25.0, 15.0
_V2_MIN_CONSTRUCTION = 2      # the old code required >= 2 before saying "construction"


def legacy_tags(nusc: Any, sample_token: str, variant: str = "v3_fair") -> dict[str, bool]:
    """One frame of legacy ground truth, reproduced verbatim. Step F3.

    Bugs are reproduced ON PURPOSE — the point of F3 is to measure what they cost, so a
    "fixed" reproduction would measure nothing.
    """
    s = nusc.get("sample", sample_token)
    cats = {nusc.get("sample_annotation", a)["category_name"] for a in s["anns"]}

    if variant == "v1_presence":
        return {
            "has_parked_vehicle": any("vehicle" in c for c in cats),
            "has_construction": any("construction" in c or "barrier" in c
                                    or "trafficcone" in c for c in cats),
            "has_pedestrian": any("pedestrian" in c for c in cats),
            "requires_slowdown": False,   # hardcoded in the original
            "turn_left": False,           # hardcoded in the original
            "following_vehicle": False,   # hardcoded in the original
        }

    if variant == "v3_fair":
        return {
            "has_parked_vehicle": any("vehicle" in c for c in cats),
            "has_construction": any(k in c for c in cats
                                    for k in ("barrier", "trafficcone", "debris")),
            "has_pedestrian": any("pedestrian" in c for c in cats),
            # presence of ANY car/truck/bus anywhere in the 360 set, not a lead test
            "following_vehicle": any(k in c for c in cats
                                     for k in ("vehicle.car", "vehicle.truck",
                                               "vehicle.bus")),
        }

    if variant == "v2_gated":
        import numpy as np

        cam = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        ego = nusc.get("ego_pose", cam["ego_pose_token"])
        ex, ey = ego["translation"][0], ego["translation"][1]
        veh = con = ped = 0
        ahead = False
        for a in s["anns"]:
            ann = nusc.get("sample_annotation", a)
            cat = ann["category_name"]
            ox, oy = ann["translation"][0], ann["translation"][1]
            dist = float(np.hypot(ox - ex, oy - ey))
            if dist > _V2_MAX_M:
                continue
            if "vehicle" in cat and dist < _V2_VEH_M:
                veh += 1
                # F-033, reproduced: a pair of wedges pinned to global north/south, not a
                # cone pointing where the car is going.
                if abs(oy - ey) > abs(ox - ex) and dist < _V2_AHEAD_M:
                    ahead = True
            if any(k in cat for k in ("barrier", "trafficcone", "debris")) and dist < _V2_CON_M:
                con += 1
            if "pedestrian" in cat and dist < _V2_PED_M:
                ped += 1
        return {
            "has_parked_vehicle": veh >= 1,
            "has_construction": con >= _V2_MIN_CONSTRUCTION,
            "has_pedestrian": ped >= 1,
            "requires_slowdown": False,
            "turn_left": False,
            "following_vehicle": ahead,
        }

    raise ValueError(f"unknown legacy variant {variant!r}; expected one of {LEGACY_VARIANTS}")


def build_reference_cost_table(nusc: Any = None, stem: str = "v1_structured",
                               out_csv: Path | str = OUT / "f3_reference_cost.csv") -> Any:
    """Same model, same outputs, three old references and the new one. Step F3.

    This is the measurement PROCESS.md 2.1 calls the thesis's strongest methodological
    result, and until now it existed only as a claim. The model outputs are held fixed
    and only the ground truth is swapped, so every delta is attributable to the reference.
    """
    import pandas as pd

    from . import prompts as P
    from .data import load_nusc

    if nusc is None:
        nusc = load_nusc(version="v1.0-trainval")

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    new = load_gt(tokens, tags)
    pred = predictions_frame(load_rows(stem), tokens, tags)

    rows, summary = [], {}
    for variant in LEGACY_VARIANTS:
        legacy = {t: legacy_tags(nusc, t, variant) for t in tokens}
        names = list(next(iter(legacy.values())))
        old_frame = pd.DataFrame([legacy[t] for t in tokens],
                                 index=pd.Index(tokens, name="sample_token"))
        mapped = [LEGACY_TO_SCHEMA[n] for n in names]
        sub_pred = pred[mapped]

        m_old = per_tag_metrics(old_frame.rename(columns=LEGACY_TO_SCHEMA).astype(bool),
                                sub_pred, mapped)
        m_new = per_tag_metrics(new[mapped], sub_pred, mapped)
        for n, newt in zip(names, mapped):
            rows.append({
                "variant": variant, "legacy_tag": n, "new_tag": newt,
                "prevalence_legacy": round(float(old_frame[n].mean()), 3),
                "prevalence_new": round(float(new[newt].mean()), 3),
                "f1_legacy": round(float(m_old.f1[newt]), 3),
                "f1_new": round(float(m_new.f1[newt]), 3),
                "delta_new_minus_legacy": round(float(m_new.f1[newt] - m_old.f1[newt]), 3),
            })
        summary[variant] = {
            "n_tags": len(names),
            "macro_f1_legacy": round(float(m_old.f1.mean()), 3),
            "macro_f1_new": round(float(m_new.f1.mean()), 3),
            "delta": round(float(m_new.f1.mean() - m_old.f1.mean()), 3),
            "tag_iou_legacy": round(mean_tag_iou(
                old_frame.rename(columns=LEGACY_TO_SCHEMA).astype(bool), sub_pred, mapped), 3),
            "tag_iou_new": round(mean_tag_iou(new[mapped], sub_pred, mapped), 3),
        }

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df, pd.DataFrame(summary).T


# Tags the earlier references could not express at all, as opposed to expressing
# differently. `requires_slowdown` and `turn_left` were hardcoded False in both earlier
# variants, and v1's `following_vehicle` too; v2's `following_vehicle` is the reproduced
# global-frame wedge bug (F-033). Everything else differs only in what the tag counts,
# which is the scope effect the decomposition isolates.
DEFECTIVE_LEGACY_TAGS = {
    ("v1_presence", "requires_slowdown"), ("v1_presence", "turn_left"),
    ("v1_presence", "following_vehicle"),
    ("v2_gated", "requires_slowdown"), ("v2_gated", "turn_left"),
    ("v2_gated", "following_vehicle"),
}


def build_reference_decomposition(
        in_csv: Path | str = OUT / "f3_reference_cost.csv",
        out_csv: Path | str = OUT / "f3_reference_decomposition.csv") -> Any:
    """Split each reference swap into a scope effect and a defect effect. Step F3b.

    A reviewer of the draft objected that calling the earlier references "different, not
    wrong" is not supportable when three of one variant's tags are False on every frame and
    another carries a frame bug. The objection is correct, and the answer is to report the
    two effects separately rather than to argue about the wording: each tag contributes
    (f1_new - f1_legacy) / n_tags to the macro delta, so partitioning the tags partitions
    the delta exactly.
    """
    import pandas as pd

    df = pd.read_csv(in_csv)
    rows = []
    for variant, g in df.groupby("variant", sort=False):
        bad = g[[(variant, t) in DEFECTIVE_LEGACY_TAGS for t in g.legacy_tag]]
        ok = g[[(variant, t) not in DEFECTIVE_LEGACY_TAGS for t in g.legacy_tag]]
        n = len(g)
        contrib = lambda part: (len(part) * (part.f1_new.mean() - part.f1_legacy.mean()) / n
                                if len(part) else 0.0)
        rows.append({
            "variant": variant, "n_tags": n,
            "macro_f1_legacy": round(float(g.f1_legacy.mean()), 3),
            "macro_f1_new": round(float(g.f1_new.mean()), 3),
            "delta_total": round(float(g.f1_new.mean() - g.f1_legacy.mean()), 3),
            "n_scope_tags": len(ok), "delta_from_scope": round(float(contrib(ok)), 3),
            "n_defective_tags": len(bad), "delta_from_defects": round(float(contrib(bad)), 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def build_alternative_baselines(
        in_csv: Path | str = OUT / "f1_per_tag_detail.csv",
        out_csv: Path | str = OUT / "f7_alt_baselines.csv") -> Any:
    """Baselines and metrics that do not collapse on a rare tag. Step F7.

    The majority class baseline answers "no" on a rare tag, so its F1 is 0 and any positive
    prediction clears it. Three alternatives are computed from the same confusion counts:
    the F1 of always answering yes, Matthews' correlation coefficient, and balanced
    accuracy. None of them is free of the prevalence, but none of them is zero by
    construction either.
    """
    import numpy as np
    import pandas as pd

    d = pd.read_csv(in_csv)
    tp, fp, fn, tn = (d[c].astype(float) for c in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    prev = (tp + fn) / n                      # positives among the scored frames
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = pd.DataFrame({
        "condition": d.condition, "tag": d.tag, "family": d.family,
        "prevalence": prev.round(3),
        "f1": d.f1,
        "baseline_f1_majority": d.baseline_f1,
        # Always yes: precision is the prevalence, recall is 1.
        "baseline_f1_always_yes": (2 * prev / (1 + prev)).round(3),
        "mcc": np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0).round(3),
        "balanced_accuracy": (0.5 * (tp / np.maximum(tp + fn, 1)
                                     + tn / np.maximum(tn + fp, 1))).round(3),
    })
    out.to_csv(out_csv, index=False)
    return out


# --- Step F4: the headline view-condition analysis ----------------------------------

def build_family_table(out_csv: Path | str = OUT / "f4_family_conditions.csv") -> Any:
    """F1 per label FAMILY x view condition, with the family baseline beside it. Step F4.

    Computed from `score_condition` rather than by re-reading the rounded per-tag CSV, so
    the family means are not a mean of rounded numbers.

    `traffic_light_ahead` is excluded throughout (F-063): `src/bev.py` never draws the
    traffic_light layer, so the tag is unanswerable from either BEV arm by construction
    and would charge a renderer gap to the representation being tested.
    """
    import pandas as pd

    from . import prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    scored = [t for t in tags if t not in BEV_BLIND_TAGS]
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = load_gt(tokens, tags)
    fam = pd.Series({t: schema["tags"][t]["family"] for t in scored})

    cols = {}
    for name, stem in conditions_available().items():
        f1 = score_condition(stem, tokens, tags)["per_tag"]["f1"]
        cols[name] = f1.loc[scored].groupby(fam).mean()
    out = pd.DataFrame(cols)
    out["baseline"] = majority_baseline(gt, tags)[scored].groupby(fam).mean()
    out["n_tags"] = fam.value_counts()
    out = out.round(3)
    out.to_csv(out_csv)
    return out


def majority_baseline_accuracy(gt: Any, tags: list[str]) -> Any:
    """Accuracy of always predicting each tag's most frequent value. Step F2.

    Reported beside F1 because the two disagree wildly on the near-constant tags and the
    disagreement is the point: `on_drivable_area` scores 1.00 F1 against a baseline that
    also scores 1.00, while a rare tag can post a high accuracy and a zero F1 (F-038).
    """
    import pandas as pd

    n = len(gt)
    out = {}
    for t in tags:
        pos = int(gt[t].sum())
        out[t] = max(pos, n - pos) / n if n else 0.0
    return pd.Series(out, name="baseline_accuracy")


def build_per_tag_detail_table(
        out_csv: Path | str = OUT / "f1_per_tag_detail.csv") -> Any:
    """Per tag, per condition: P / R / F1 / support / confusion / intervals. Step F1.

    PLAN F1 asks for "precision, recall, F1, support, plus a confusion matrix" per tag,
    and its verify criterion is a results table with one row per tag. `e_all5_f1.csv`
    answers the headline question (F1 across the five arms) but carries only F1, support
    and the baseline, so precision, recall, the confusion counts and the Wilson intervals
    existed solely inside `per_tag_metrics` and reached no artifact. Computed-but-never-
    published is the same debt as published-but-never-computed.

    Long format — one row per (condition, tag) — because a wide table with 12 measures x
    5 conditions is 60 columns nobody reads, and long format is what a pivot wants anyway.

    THE PER-TAG INTERVALS ARE DESCRIPTIVE, NOT INFERENTIAL. 210 cells carry a 95% interval
    and roughly ten of them miss at the nominal level by chance alone, so no claim in the
    thesis rests on a single row of this table being significant. The inferential claims
    live in `f6_significance.csv`: five comparisons per average, Holm-corrected (F-082).

    This table also supersedes `outputs/e5_firstlook.csv`, the ad-hoc first-look table
    from before `src/eval.py` existed (F-062 flagged it as a first look, not Stage F):
    every column that file had, including accuracy and the accuracy baseline, is here for
    all five arms rather than one, and generated rather than typed into a transcript.
    """
    import pandas as pd

    from . import prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = load_gt(tokens, tags)
    base_f1 = majority_baseline(gt, tags)
    base_acc = majority_baseline_accuracy(gt, tags)
    clusters = _scene_of(tokens)

    frames = []
    for name, stem in conditions_available().items():
        s = score_condition(stem, tokens, tags)
        m = s["per_tag"].copy()
        # tn is implied by the other three and n_answered; stated explicitly so the row
        # IS a confusion matrix rather than something a reader must reconstruct.
        m["tn"] = m["n_answered"] - m["tp"] - m["fp"] - m["fn"]
        m["accuracy"] = (m["tp"] + m["tn"]) / m["n_answered"].where(m["n_answered"] > 0)
        m["condition"] = name
        m["family"] = [schema["tags"][t]["family"] for t in m.index]
        m["baseline_f1"] = base_f1
        m["baseline_accuracy"] = base_acc
        # Cluster-robust intervals sit beside the Wilson ones rather than replacing them
        # (F-082): Wilson is what the earlier tables published, and the point of the
        # column pair is that a reader can see how much the correction cost.
        m = m.join(bootstrap_per_tag(s["gt"], s["pred"], tags, clusters))
        frames.append(m.reset_index())

    cols = ["condition", "tag", "family", "support", "n_answered",
            "tp", "fp", "fn", "tn",
            "precision", "precision_lo", "precision_hi",
            "precision_lo_clust", "precision_hi_clust",
            "recall", "recall_lo", "recall_hi",
            "recall_lo_clust", "recall_hi_clust",
            "f1", "f1_lo_clust", "f1_hi_clust", "boot_degenerate",
            "baseline_f1", "accuracy", "baseline_accuracy"]
    df = pd.concat(frames, ignore_index=True)[cols].round(4)
    if out_csv is not None:    # None = "give me the frame", for callers that recombine it
        df.to_csv(out_csv, index=False)
    return df


def build_model_comparison_table(
        out_csv: Path | str = OUT / "e10_model_comparison.csv",
        slice_spec: tuple[str, str, str, int] = MODEL_SLICE) -> Any:
    """E10: two models, one prompt, one set of frames. Step E10 / D-048.

    R12 requires a sensitivity analysis for every researcher-chosen parameter, and the MODEL
    is one. This is a PAIRED comparison — both arms see the identical first 150 tokens of the
    execution subset — so a bias in the slice hits both and largely cancels in the delta,
    which is the quantity reported. The slice spans 30 scenes rather than 139, so it supports
    the DIFFERENCE between models and not absolute per-tag claims about either.

    Unanswered cells are excluded per tag, as everywhere else. That could in principle
    flatter a model that omits what it is unsure of, so the alternative is measured rather
    than assumed: scoring Qwen3-VL-8B's 265 unanswered cells as False moves its macro from
    0.466 to 0.459, a -0.007 shift that changes no conclusion.
    """
    import pandas as pd

    from . import prompts as P

    name, alt_stem, base_stem, n = slice_spec
    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"][:n]
    gt = load_gt(tokens, tags)
    scored = [t for t in tags if t not in BEV_BLIND_TAGS]

    base = predictions_frame(load_rows(base_stem), tokens, tags)
    alt = predictions_frame(load_rows(alt_stem), tokens, tags)
    mb, ma = per_tag_metrics(gt, base, tags), per_tag_metrics(gt, alt, tags)

    df = pd.DataFrame({
        "qwen2_5_vl_7b": mb["f1"].round(3),
        f"{name.lower().replace('-', '_').replace('.', '_')}": ma["f1"].round(3),
        "support": mb["support"],
        "family": [schema["tags"][t]["family"] for t in mb.index],
        "baseline_f1": majority_baseline(gt, tags).round(3),
        "n_unanswered_alt": alt.isna().sum(),
    })
    df["delta"] = (df.iloc[:, 1] - df.iloc[:, 0]).round(3)
    df.to_csv(out_csv)
    import pandas as _pd
    _sc = _pd.read_parquet(OUT / "gt_all.parquet", columns=["sample_token", "scene_name"])
    meta = {
        "n_frames": len(tokens),
        "n_scenes_spanned": int(_sc[_sc.sample_token.isin(tokens)].scene_name.nunique()),
        "macro_34_base": round(float(mb.loc[scored, "f1"].mean()), 3),
        "macro_34_alt": round(float(ma.loc[scored, "f1"].mean()), 3),
        "macro_34_alt_unanswered_as_false": round(float(
            per_tag_metrics(gt, alt.fillna(False), tags).loc[scored, "f1"].mean()), 3),
        "baseline": round(float(majority_baseline(gt, tags)[scored].mean()), 3),
        "n_unanswered_cells_alt": int(alt.isna().sum().sum()),
        "n_unanswered_cells_base": int(base.isna().sum().sum()),
    }
    return df, meta



def build_bev_model_slice_table(
        out_csv: Path | str = OUT / "e10b_bev_model_slice.csv") -> Any:
    """The alternative model on the REPAIRED raster, against the primary model on the same.

    The companion to the camera slice: same model, same 150 frames, same prompt shape, and
    the only difference is the representation. It is what separates "this model cannot read
    a diagram" from "a diagram carries less than a photograph".
    """
    return build_model_comparison_table(out_csv, BEV_MODEL_SLICE)

# --- Step F6: paired significance and cluster-robust intervals -----------------------
#
# Every headline in this thesis is an ORDINAL claim over five arms measured on the SAME
# 628 frames -- camera 0.389 beats both 0.375, temporal 0.363 loses to camera -- and until
# this section existed not one of them carried a test (F-081). Two defects, both measured
# before anything here was written:
#
#   (1) The frames are NOT independent. They sit in 139 scenes, keyframes 0.5 s apart,
#       capped at 12 per scene (D-036). Per-frame accuracy has an intra-scene correlation
#       of ICC = 0.517 at a mean cluster size of 4.50, so the design effect is
#       1 + (k0 - 1) * ICC = 2.81: the effective sample is ~224 frames, not 628, and every
#       Wilson interval already published is about sqrt(2.81) = 1.68x too narrow.
#       Resampling SCENES rather than frames is the remedy (F-082).
#
#   (2) The comparisons are PAIRED. Camera and `both` disagree on 1,016 cells of which
#       camera is right on 510 and `both` on 506 -- a dead heat that an unpaired
#       comparison of two macro numbers cannot see. The same resample must score both
#       arms, or the pairing that makes the test powerful is thrown away.
#
# One machine serves both: aggregate tp/fp/fn to the scene, then resample scenes with a
# multinomial and recover any tag's F1 by matrix product. 2,000 draws x 6 arms is ~1 s,
# so there is no reason to approximate.

N_BOOTSTRAP = 2000      # measured F-082: the macro-F1 CI half-width moves by <0.001 from
                        # 2,000 draws up, and by 0.004 between 200 and 2,000.
BOOTSTRAP_SEED = 20260920   # frozen, so every published interval is reproducible byte-wise


def _scene_of(tokens: list[str]) -> Any:
    """The clustering variable: which scene each scored frame came from.

    Read from `gt_all.parquet` rather than the devkit, because C1 already put it there
    and the devkit costs 58 s to answer a question the cached table answers in 0.2 s (R20).
    """
    import pandas as pd

    sc = pd.read_parquet(OUT / "gt_all.parquet",
                         columns=["sample_token", "scene_name"]).set_index("sample_token")
    return sc.loc[tokens, "scene_name"]


def _cell_counts(gt: Any, pred: Any, tags: list[str]) -> tuple[Any, Any, Any]:
    """Per-frame, per-tag tp / fp / fn indicators as int arrays (n_frames x n_tags).

    An UNANSWERED cell contributes zero to all three, which is `per_tag_metrics`'
    exclusion rule expressed as arithmetic rather than as a mask -- the same rule, so a
    bootstrap draw covering every frame once must reproduce the point estimate exactly.
    That identity is asserted in the self-checks, not assumed.
    """
    import numpy as np

    p = pred[tags]
    answered = p.notna().to_numpy()
    yhat = np.equal(p.to_numpy(), True) & answered      # object dtype: None -> False
    y = gt[tags].to_numpy(dtype=bool)
    return ((y & yhat).astype(np.int64),
            (~y & yhat & answered).astype(np.int64),
            (y & ~yhat & answered).astype(np.int64))


def _by_cluster(counts: tuple[Any, Any, Any], clusters: Any) -> tuple[Any, Any, Any]:
    """Sum per-frame counts within each cluster. (n_clusters x n_tags) per component."""
    import numpy as np

    codes, _ = _cluster_codes(clusters)
    n = codes.max() + 1
    out = []
    for c in counts:
        agg = np.zeros((n, c.shape[1]), dtype=np.int64)
        np.add.at(agg, codes, c)
        out.append(agg)
    return tuple(out)


def _cluster_codes(clusters: Any) -> tuple[Any, Any]:
    import numpy as np

    uniq, codes = np.unique(np.asarray(clusters), return_inverse=True)
    return codes, uniq


def _f1_from_counts(tp: Any, fp: Any, fn: Any) -> Any:
    """2tp / (2tp + fp + fn), which is exactly `per_tag_metrics`' F1 including its zeros.

    Both the tp == 0 case (precision and recall are 0, so F1 is 0) and the empty case
    (a tag with no positives and no predictions) land on 0.0 here as they do there.
    """
    import numpy as np

    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros(np.shape(den), dtype=float), where=den > 0)


def _draw_weights(n_clusters: int, n_draws: int, rng: Any) -> Any:
    """Multinomial cluster weights: one row per bootstrap draw, summing to n_clusters.

    Equivalent to sampling n_clusters clusters with replacement, but as weights it turns
    the whole bootstrap into one matrix product instead of 2,000 fancy-index gathers.
    """
    return rng.multinomial(n_clusters, [1.0 / n_clusters] * n_clusters, size=n_draws)


def bootstrap_macro(gt: Any, pred: Any, tags: list[str], scored: list[str],
                    clusters: Any, n_draws: int = N_BOOTSTRAP,
                    seed: int = BOOTSTRAP_SEED, weights: Any = None,
                    average: str = "macro") -> Any:
    """Bootstrap distribution of an averaged F1 over resampled clusters.

    THE THESIS NEVER SAID WHICH AVERAGE IT MEANT (F-081), and the two are not the same
    number -- they are not even the same ORDERING (F-083):

      `macro`  mean of the 34 per-tag F1 scores. Every tag counts once, so a tag with 36
               positives moves it as far as one with 314. This is what every published
               number in this thesis is, at `per_tag_metrics(...)["f1"].mean()`.
      `micro`  F1 of the counts pooled across tags. Frequent tags dominate, so it answers
               "how many individual tag decisions were right", not "how many of the
               vocabulary's concepts does the model handle".

    Both are reported because the choice reverses two conclusions, and a reader who is
    not told which one is in front of them cannot audit either.

    Pass `weights` to share one resample across two arms -- that is what makes the
    comparison PAIRED and is the whole reason the argument exists.
    """
    import numpy as np

    assert average in ("macro", "micro"), f"unknown average {average!r}"
    tp, fp, fn = _by_cluster(_cell_counts(gt, pred, tags), clusters)
    if weights is None:
        weights = _draw_weights(tp.shape[0], n_draws, np.random.default_rng(seed))
    idx = [tags.index(t) for t in scored]
    tp, fp, fn = (weights @ tp)[:, idx], (weights @ fp)[:, idx], (weights @ fn)[:, idx]
    if average == "micro":
        return _f1_from_counts(tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1))
    return _f1_from_counts(tp, fp, fn).mean(axis=1)


def bootstrap_per_tag(gt: Any, pred: Any, tags: list[str], clusters: Any,
                      n_draws: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> Any:
    """Cluster-robust 95% intervals for each tag's precision, recall and F1.

    Two debts, one function. **F1 had no interval at all**: it is a ratio of sums rather
    than a proportion, which is why `wilson_interval` deliberately refuses to cover it
    (L-021), so every published F1 has so far been a bare point estimate. And **the
    published precision and recall intervals are too narrow**: Wilson treats the 628
    frames as independent Bernoulli trials when they are 139 scenes of correlated
    keyframes, measured at a CI width ratio of 1.88 (F-082). A bootstrap over scenes fixes
    both, because it resamples the thing actually being reported at the level the data was
    actually collected.

    The Wilson columns stay in the artifact beside these, deliberately: they are what the
    earlier tables published, and deleting them would hide the correction rather than
    document it.
    """
    import numpy as np
    import pandas as pd

    tp, fp, fn = _by_cluster(_cell_counts(gt, pred, tags), clusters)
    w = _draw_weights(tp.shape[0], n_draws, np.random.default_rng(seed))
    TP, FP, FN = w @ tp, w @ fp, w @ fn
    q = {}
    for name, num, den in (("precision", TP, TP + FP), ("recall", TP, TP + FN)):
        r = np.divide(num, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
        q[f"{name}_lo_clust"], q[f"{name}_hi_clust"] = np.percentile(r, [2.5, 97.5], axis=0)
    f1 = _f1_from_counts(TP, FP, FN)
    q["f1_lo_clust"], q["f1_hi_clust"] = np.percentile(f1, [2.5, 97.5], axis=0)
    # A BOOTSTRAP CANNOT MOVE A BOUNDARY STATISTIC, and reporting [0.000, 0.000] for
    # `is_u_turn` would claim a certainty the data does not support. With tp == 0 no
    # resample of the observed frames can produce a true positive, so every draw returns
    # 0 and the interval collapses; the mirror case (fp == fn == 0, e.g. `on_drivable_area`)
    # collapses at 1. Wilson has no such failure -- it is a formula, not a resample, and
    # keeps a non-zero width at k == 0 by construction (L-021), which is exactly the case
    # it was adopted for. Flagged rather than silently published, so a reader reads the
    # Wilson column on those rows.
    # The flag covers all three statistics, not F1 alone: 9 (condition, tag) cells answer
    # True on every frame, so fn == 0, recall is exactly 1.000 in every resample and its
    # interval collapses while F1's does not.
    q["boot_degenerate"] = np.logical_or.reduce([
        q[f"{k}_lo_clust"] == q[f"{k}_hi_clust"] for k in ("precision", "recall", "f1")])
    return pd.DataFrame(q, index=pd.Index(tags, name="tag"))


def paired_delta(gt: Any, pred_a: Any, pred_b: Any, tags: list[str], scored: list[str],
                 clusters: Any, n_draws: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED, average: str = "macro") -> dict[str, Any]:
    """Macro F1 of B minus macro F1 of A, with a cluster-bootstrap CI and a p-value.

    The SAME cluster resample scores both arms, so a scene that happens to be easy lifts
    both and cancels in the difference. That is the paired case Dietterich (1998) is
    written for, and it is why the interval on a delta is far tighter than the difference
    of two independent intervals would suggest.

    The p-value is the two-sided bootstrap tail, computed with the (k + 1) / (B + 1)
    convention so it can never be reported as exactly 0 from a finite number of draws.
    """
    import numpy as np

    codes, uniq = _cluster_codes(clusters)
    w = _draw_weights(len(uniq), n_draws, np.random.default_rng(seed))
    kw = {"average": average}
    a = bootstrap_macro(gt, pred_a, tags, scored, clusters, weights=w, **kw)
    b = bootstrap_macro(gt, pred_b, tags, scored, clusters, weights=w, **kw)
    d = b - a
    one = np.ones((1, len(uniq)), dtype=np.int64)
    point_a = float(bootstrap_macro(gt, pred_a, tags, scored, clusters, weights=one, **kw)[0])
    point_b = float(bootstrap_macro(gt, pred_b, tags, scored, clusters, weights=one, **kw)[0])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((1 + int((d <= 0).sum())) / (n_draws + 1),
                (1 + int((d >= 0).sum())) / (n_draws + 1))
    return {"average": average, "macro_a": point_a, "macro_b": point_b,
            "delta": point_b - point_a, "delta_lo": float(lo), "delta_hi": float(hi),
            "p": min(1.0, p), "n_clusters": len(uniq), "n_draws": n_draws}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Step F6.

    175 interval cells and 15 pairwise comparisons carried no multiplicity control at all
    (F-081). Holm rather than Bonferroni because it is uniformly more powerful at the same
    family-wise error rate, and rather than Benjamini-Hochberg because the claims here are
    individually asserted in the text ("camera beats both"), not screened as a batch.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def design_effect(gt: Any, pred: Any, tags: list[str], scored: list[str],
                  clusters: Any, n_draws: int = N_BOOTSTRAP,
                  seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """How much wider a scene-clustered interval is than a frame-level one.

    This is the number that says whether clustering matters, and it is reported rather
    than argued: the same bootstrap is run twice, once resampling scenes and once
    resampling frames, and the ratio of the two CI widths is the answer. A ratio near 1
    would have meant the clustering correction was not worth making.
    """
    import numpy as np

    wide = bootstrap_macro(gt, pred, tags, scored, clusters, n_draws, seed)
    flat = bootstrap_macro(gt, pred, tags, scored,
                           list(range(len(clusters))), n_draws, seed)
    w_lo, w_hi = np.percentile(wide, [2.5, 97.5])
    f_lo, f_hi = np.percentile(flat, [2.5, 97.5])
    return {"ci_width_scene": float(w_hi - w_lo), "ci_width_frame": float(f_hi - f_lo),
            "width_ratio": float((w_hi - w_lo) / (f_hi - f_lo))}


def build_significance_table(out_csv: Path | str = OUT / "f6_significance.csv",
                             reference: str = "camera") -> Any:
    """Every arm against the reference arm, paired, cluster-robust, Holm-corrected.

    The reference is the single front camera: it is the cheapest input any of this could
    use and the one every other arm was built to beat, so "does X beat a photograph?" is
    the question each row answers.
    """
    import pandas as pd

    from . import prompts as P

    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    scored = [t for t in tags if t not in BEV_BLIND_TAGS]
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    gt = load_gt(tokens, tags)
    clusters = _scene_of(tokens)

    preds = {name: predictions_frame(load_rows(stem), tokens, tags)
             for name, stem in conditions_available().items()}
    assert reference in preds, f"reference arm {reference!r} has no results file"

    rows = []
    for average in ("macro", "micro"):
        for name, pred in preds.items():
            if name == reference:
                continue
            r = paired_delta(gt, preds[reference], pred, tags, scored, clusters,
                             average=average)
            r["condition"] = name
            r["reference"] = reference
            rows.append(r)
    df = pd.DataFrame(rows)
    # Holm within each average: the two families are two different questions asked of the
    # same data, and pooling them would penalise each for the other's comparisons.
    df["p_holm"] = 0.0
    for average, g in df.groupby("average"):
        df.loc[g.index, "p_holm"] = holm(list(g["p"]))
    df["significant_05"] = df["p_holm"] < 0.05
    df = df.set_index(["average", "condition"])
    de = design_effect(gt, preds[reference], tags, scored, clusters)
    for k, v in de.items():
        df[k] = round(v, 4)
    cols = ["reference", "macro_a", "macro_b", "delta", "delta_lo", "delta_hi",
            "p", "p_holm", "significant_05", "n_clusters", "n_draws",
            "ci_width_scene", "ci_width_frame", "width_ratio"]
    df = df[cols].round(4)
    df.to_csv(out_csv)
    return df
