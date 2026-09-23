"""BDD100K signage — Step H3, serving RQ4.

WHY A SECOND DATASET AT ALL. The brief requires *signalisation*, and nuScenes cannot supply
it: it has no traffic-sign or traffic-light annotation, and its map's `items[].color` field
is the fixture's lamp inventory rather than what is lit — 626 of 634 records carry RED and
YELLOW and GREEN at once (F-004/F-076). H2 established that PRESENCE is verifiable there and
STATE is not. BDD100K has `trafficLightColor`, so state is the thing this step exists to
measure, and it is the one signage question nuScenes structurally cannot answer.

STATE IS ASKED AS THREE BOOLEANS, NOT AS A COLOUR. "What colour is the light?" is not
well-posed on this data: **34.0% of images with a lit light contain more than one DISTINCT
lit colour**, and BDD100K has no lane association from which to decide which light is the
ego's. The best available proxy (largest box = nearest) agrees with the image's majority
colour on only 87.55% of lit images, so a single-answer question would be scoring a
confoundable geometric guess. `red_light_visible` / `yellow_light_visible` /
`green_light_visible` are well-posed on 100% of images, need no new metric, and score through
`eval.per_tag_metrics` unchanged. R22/R26: when a category can be stated without a
confoundable proxy, state it that way.

SIZE IS A DECOMPOSITION AXIS, NEVER A FILTER — and this is where R34 INVERTS. F-030 measured
nuScenes 3D boxes at median 68.5 px with 0.3% under 15 px and concluded a perceptibility gate
would look principled and change nothing. BDD100K traffic lights are **median 25.6 px with
16.5% of lit lights under 15 px**, so the same gate would bite hard — and per D-049(b) a gate
that removes what the model is being asked about is the "front-answerable filter" mistake.
The size is recorded per image so results can be decomposed by it at zero GPU cost.

Build:  ./venv/bin/python -c "import src.bdd as B; B.build_subset()"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BDD = ROOT / "data/bdd100k"
LABELS = BDD / "samples.json"
IMAGES = BDD / "images"

# The five signage tags. Written in ONE place; the prompt's question text is keyed off this
# tuple and a self-check asserts the two agree, which is the same enforcement F-054 settled
# on rather than generating prose from a schema (R4).
BDD_TAGS = ("traffic_light_present", "traffic_sign_present",
            "red_light_visible", "yellow_light_visible", "green_light_visible")

BDD_QUESTIONS = {
    "traffic_light_present": "at least one traffic light is visible",
    "traffic_sign_present": "at least one traffic sign is visible",
    "red_light_visible": "at least one traffic light showing RED is visible",
    "yellow_light_visible": "at least one traffic light showing YELLOW or amber is visible",
    "green_light_visible": "at least one traffic light showing GREEN is visible",
}

# Subset size. 450 drawn uniformly preserves the true prevalence of four of the five tags
# within ~2 points, so their reported baselines ARE the real-world baselines. Only
# `yellow_light_visible` needs protection: at 3.28% a uniform 586 lands ~19 positives, below
# the 30-positive floor D-032 set, and an F1 there would be noise. The top-up to 150 is
# exactly what D-047 did for `is_u_turn`, and it enriches yellow 7.8x — which is why the
# uniform 450 is retained as a separate scoring view (see `subset_views`).
N_UNIFORM = 450
N_YELLOW_TARGET = 150
SUBSET_SEED = 20260918

_LIT = ("R", "G", "Y")


def tags_for(sample: dict) -> dict[str, Any]:
    """The five tags plus provenance, from one BDD100K sample record."""
    dets = (sample.get("detections") or {}).get("detections", [])
    lights = [d for d in dets if d["label"] == "traffic light"]
    cols = {d.get("trafficLightColor") for d in lights}
    lit = [d for d in lights if d.get("trafficLightColor") in _LIT]
    h = sample["metadata"]["height"]
    return {
        "filepath": sample["filepath"],
        "traffic_light_present": bool(lights),
        "traffic_sign_present": any(d["label"] == "traffic sign" for d in dets),
        "red_light_visible": "R" in cols,
        "yellow_light_visible": "Y" in cols,
        "green_light_visible": "G" in cols,
        # PROVENANCE, not a label. 29.94% of images carry a light whose colour is NA, or a
        # light with no lit colour at all — so a False on the three state tags may be
        # silently wrong there. Carried so state metrics can be reported on both
        # populations instead of quoting one number over contaminated ground truth.
        "state_gt_uncertain": ("NA" in cols) or (bool(lights) and not (cols - {"NA"})),
        "n_lights": len(lights),
        "max_lit_px": max((d["bounding_box"][3] * h for d in lit), default=0.0),
        "size_bytes": sample["metadata"]["size_bytes"],
    }


def build_subset(out_path: str = "outputs/h3_bdd_subset.json") -> dict:
    """Select the images H3 runs on. Step H3.

    Two views are emitted, because one number cannot serve both purposes:
      `uniform`  the 450 drawn at random — prevalences and therefore baselines are the real
                 ones, so this is what a presence result should be read against.
      `all`      those 450 plus the yellow top-up — the only view on which
                 `yellow_light_visible` clears the scoreable floor.
    """
    import numpy as np
    import pandas as pd

    samples = json.loads(LABELS.read_text())["samples"]
    df = pd.DataFrame([tags_for(s) for s in samples])

    rng = np.random.default_rng(SUBSET_SEED)
    order = rng.permutation(len(df))
    uniform = df.iloc[order[:N_UNIFORM]]
    need = N_YELLOW_TARGET - int(uniform.yellow_light_visible.sum())
    rest = df.iloc[order[N_UNIFORM:]]
    topup = rest[rest.yellow_light_visible].head(max(need, 0))
    sub = pd.concat([uniform, topup])

    payload = {
        "seed": SUBSET_SEED, "n_uniform": N_UNIFORM, "n_total": int(len(sub)),
        "rule": (f"{N_UNIFORM} uniform at seed {SUBSET_SEED}, then top up "
                 f"yellow_light_visible to {N_YELLOW_TARGET} positives"),
        "download_mb": round(float(sub.size_bytes.sum()) / 2 ** 20, 1),
        "prevalence_full_split_pct": {t: round(100 * float(df[t].mean()), 2) for t in BDD_TAGS},
        "positives_uniform": {t: int(uniform[t].sum()) for t in BDD_TAGS},
        "positives_all": {t: int(sub[t].sum()) for t in BDD_TAGS},
        "state_gt_uncertain_pct": round(100 * float(sub.state_gt_uncertain.mean()), 2),
        "uniform_filepaths": uniform.filepath.tolist(),
        "records": sub.to_dict("records"),
    }
    (ROOT / out_path).write_text(json.dumps(payload))
    return {k: v for k, v in payload.items() if k not in ("records", "uniform_filepaths")}


def image_id(filepath: str) -> str:
    """`data/b1c66a42-6f7d68ca.jpg` -> `b1c66a42-6f7d68ca`. The run's token."""
    return Path(filepath).stem


def build_prompt() -> str:
    """The H3 prompt. Deliberately NOT built from `prompts.TAG_QUESTIONS`.

    `tests/test_prompts.py` asserts that dict equals the FROZEN nuScenes schema exactly —
    the R4 drift guard that exists because a hand-written tag name once left
    `is_turning_left` False on all 5,500 turning frames (F-016). Adding BDD entries to it
    would break that guard for the schema it was written to protect. A separate vocabulary
    with its own drift test keeps both intact.
    """
    lines = "\n".join(f'  "{t}": true or false   — {BDD_QUESTIONS[t]}' for t in BDD_TAGS)
    return (
        "Look at this photograph taken from a car's dashboard camera.\n\n"
        "Answer each question about WHAT IS VISIBLE IN THIS IMAGE. Judge only what you can\n"
        "see; do not guess from context.\n\n"
        f"{lines}\n\n"
        "Reply with exactly one JSON object and nothing else:\n"
        "{\n" + ",\n".join(f'  "{t}": true' for t in BDD_TAGS) + "\n}"
    )


def score(results_stem: str = "h3_bdd_signage",
          subset_path: str = "outputs/h3_bdd_subset.json") -> dict:
    """Per-tag P/R/F1 against the majority baseline, on both views and both state
    populations. Step H3.

    State metrics are reported twice — over all images and over only those where the ground
    truth is not flagged uncertain — because 29.94% of the subset carries a light whose
    colour is NA, and a single number over that would quietly average a known contamination
    into the result.
    """
    import pandas as pd

    from . import eval as E

    spec = json.loads((ROOT / subset_path).read_text())
    recs = {image_id(r["filepath"]): r for r in spec["records"]}
    uniform = {image_id(f) for f in spec["uniform_filepaths"]}
    rows = E.load_rows(results_stem)

    ids = [r["sample_token"] for r in rows if not r.get("backend_error")]
    if not ids:
        return {"n": 0, "note": "no results yet"}
    gt = pd.DataFrame([{t: recs[i][t] for t in BDD_TAGS} for i in ids], index=ids)
    pred = E.predictions_frame(rows, ids, list(BDD_TAGS))

    def block(sel):
        g, p = gt.loc[sel], pred.loc[sel]
        m = E.per_tag_metrics(g, p, list(BDD_TAGS))
        m["baseline_f1"] = E.majority_baseline(g, list(BDD_TAGS))
        return m[["precision", "recall", "f1", "support", "baseline_f1", "n_answered"]].round(3)

    certain = [i for i in ids if not recs[i]["state_gt_uncertain"]]
    return {
        "n_images": len(ids),
        "parse_failure_rate": round(E.parse_failure_rate(rows), 4),
        "all": block(ids).to_dict("index"),
        "uniform_view": block([i for i in ids if i in uniform]).to_dict("index"),
        "state_gt_certain_only": block(certain).to_dict("index"),
        "n_state_gt_certain": len(certain),
    }


HF_BASE = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main"


def fetch_subset_images(subset_path: str = "outputs/h3_bdd_subset.json",
                        dest: Path = IMAGES, max_attempts: int = 6) -> dict:
    """Download only the subset's images. Step H3.

    RETRIES ON 429, AND VERIFIES EVERY FILE. Anonymous HuggingFace fetching returns HTTP 429
    once a rate window is consumed - measured at 14-16 failures in 32 at every parallelism
    from 2 to 8 - and `curl -sfL` swallows those silently. A hole in the image set becomes an
    absent frame in the run and a shorter-than-expected result file that nothing flags, which
    is F-048's lesson: a transfer that fails quietly is worse than one that fails loudly.

    Each file is checked against the `size_bytes` the label file already records, so a
    truncated download or an error page cannot masquerade as an image.
    """
    import time
    import urllib.request

    spec = json.loads((ROOT / subset_path).read_text())
    dest.mkdir(parents=True, exist_ok=True)
    ok, failed, skipped = 0, [], 0

    for i, rec in enumerate(spec["records"], 1):
        name = Path(rec["filepath"]).name
        out = dest / name
        if out.exists() and out.stat().st_size == rec["size_bytes"]:
            skipped += 1
            continue
        url = f"{HF_BASE}/{rec['filepath']}"
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = r.read()
                if len(body) != rec["size_bytes"]:
                    raise ValueError(f"size {len(body)} != expected {rec['size_bytes']}")
                out.write_bytes(body)
                ok += 1
                break
            except Exception as e:                       # noqa: BLE001
                # Widening backoff: a 429 window lasts tens of seconds, so a flat short
                # retry just burns the attempt budget without waiting it out.
                if attempt == max_attempts - 1:
                    failed.append({"file": name, "error": f"{type(e).__name__}: {e}"})
                else:
                    time.sleep(min(2 ** attempt, 30))
        if i % 150 == 0:
            print(f"  {i}/{len(spec['records'])} ok={ok} skip={skipped} fail={len(failed)}",
                  flush=True)

    return {"n_requested": len(spec["records"]), "n_downloaded": ok,
            "n_already_present": skipped, "n_failed": len(failed),
            "failures": failed[:10], "complete": len(failed) == 0}


def build_bundle(out_path: str = "outputs/h3_bundle.tar.gz",
                 subset_path: str = "outputs/h3_bdd_subset.json") -> dict:
    """Pack H3's 586 images, its labels and the code that runs them. Step H3.

    Its own bundle rather than a flag on `data.build_colab_bundle`: that one is 542 MB of
    nuScenes and BDD100K shares nothing with it, so re-uploading it to run a 33 MB job
    would cost an hour of bandwidth for nothing.

    The GROUND TRUTH TRAVELS WITH THE IMAGES, deliberately. It is 586 rows of booleans, the
    model never sees them, and carrying them means the notebook can score its own run in
    the session instead of the numbers waiting on a download — which is how F-062 became a
    'first look' that had to be re-scored later.
    """
    import hashlib
    import tarfile

    spec = json.loads((ROOT / subset_path).read_text())
    out = ROOT / out_path
    missing, n_img, src_hashes = [], 0, {}
    with tarfile.open(out, "w:gz") as tar:
        for name in (subset_path, "src/bdd.py", "src/vlm.py", "src/prompts.py",
                     "src/eval.py"):
            tar.add(ROOT / name, arcname=name)
            src_hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:12]
        for rec in spec["records"]:
            p = IMAGES / Path(rec["filepath"]).name
            # Size-checked, not existence-checked. `fetch_subset_images` verifies against
            # `size_bytes` for the same reason: an HTTP 429 error page is a file that
            # exists and decodes to nothing (F-048).
            if p.exists() and p.stat().st_size == rec["size_bytes"]:
                tar.add(p, arcname=f"images/{p.name}")
                n_img += 1
            else:
                missing.append(p.name)
    assert not missing, (
        f"{len(missing)} of {len(spec['records'])} images missing or truncated, e.g. "
        f"{missing[:3]}. Run `bdd.fetch_subset_images()` first — a hole in the image set "
        "becomes a short results file that nothing flags.")
    size_mb = out.stat().st_size / 1048576
    (ROOT / "outputs/h3_bundle_manifest.json").write_text(json.dumps({
        "sha256_12": src_hashes, "n_images": n_img,
        "n_records": len(spec["records"]), "size_mb": round(size_mb, 1)}, indent=1))
    return {"path": str(out), "n_images": n_img, "size_mb": round(size_mb, 1),
            "n_records": len(spec["records"])}


# --- Step H3 analysis and Step H4: the cross-dataset contrast -------------------------
#
# Three tables, each answering a question the scores JSON cannot answer on its own.
#
# WHY A VIEW TABLE. `yellow_light_visible` scores 0.182 on the uniform 450, 0.540 on the
# enriched 586 and 0.451 on the certain-reference 381. Three numbers, one capability, a 3x
# spread -- and the raw rows show the model made **exactly 96 false positives in both the
# uniform and the enriched view**. The enrichment could not change the error behaviour,
# because every top-up image is a yellow positive and can therefore only add a true
# positive or a false negative. So F1 tripled while the model did not move at all. That is
# F-067's finding (the reference choice moves the score) arriving through a different
# mechanism: not a defective reference, but the PREVALENCE of the evaluation set.
#
# WHY A SIZE TABLE, AND WHY IT IS NOT A FILTER. F-030 measured nuScenes boxes at median
# 68.5 px with 0.3% under 15 px and concluded a perceptibility gate would look principled
# and change nothing (R34). BDD100K inverts that: in this subset lit lights are median
# 29.8 px with 12.0% under 15 px, so the same gate WOULD bite. It is still not applied --
# removing what the model is being asked about is D-049(b)'s "front-answerable filter"
# mistake -- but it is reported, and it earns its keep: yellow collapses to 0.125 below
# 15 px while green is flat at 0.85-0.93 across every band. Yellow's failure is largely a
# SMALL-OBJECT failure; red's is not, because red is mediocre at every size.
#
# H4, AND A DEFINITION I FIRST GOT WRONG (F-088). The same model scores F1 0.672 on
# nuScenes' `traffic_light_ahead` and 0.912 on BDD100K's `traffic_light_present`. The first
# draft of this comparison described the nuScenes tag as "a mapped light projected into the
# camera frustum within 50 m". That is H2's `traffic_light_in_view`, a DIFFERENT tag. The
# scored one is a fixture inside a 30 m x 8 m FORWARD CORRIDOR (AHEAD_RANGE_M = 30,
# CORRIDOR_HALFWIDTH_M = 4) -- far narrower than what a camera sees. Caught by reading the
# schema before writing the finding (R5/R16), and it changed the finding.
#
# The gap is then TESTABLE rather than arguable, because H2 already recorded which frames
# have a mapped light in view. Of the camera arm's 109 "false positives", 82 (75.2%) have a
# mapped light in view within 50 m, against 3.7% of true negatives. So the model is not
# hallucinating lights; it sees real ones outside a metric corridor it cannot resolve from a
# monocular image. Holding the predictions fixed and swapping in the view reference -- F3's
# move -- takes precision 0.540 -> 0.840 and F1 0.672 -> 0.867, closing 0.195 of the 0.240
# gap. About 81% of the "cross-dataset" difference is the REFERENCE'S SCOPE.
#
# What remains (0.045 F1) still has two confounds that cannot be separated here: H2's view
# reference models no occlusion (L-025) where BDD's human box does, and prevalence differs
# (35.4% vs 56.0%).

SIZE_BANDS = ((0, 15), (15, 25), (25, 40), (40, 10 ** 9))
STATE_TAGS = ("red_light_visible", "yellow_light_visible", "green_light_visible")

# The nuScenes side of H4, read from the published per-tag table rather than recomputed,
# so the two halves of the comparison cannot drift apart (R4).
NUSCENES_TAG = "traffic_light_ahead"
NUSCENES_CONDITION = "camera"


def _frames_for(results_stem: str = "h3_bdd_signage",
                subset_path: str = "outputs/h3_bdd_subset.json"):
    """(gt, pred, records, ids) for the scored rows. One loader, three tables."""
    import pandas as pd

    from . import eval as E

    spec = json.loads((ROOT / subset_path).read_text())
    recs = {image_id(r["filepath"]): r for r in spec["records"]}
    rows = E.load_rows(results_stem)
    ids = [r["sample_token"] for r in rows if not r.get("backend_error")]
    gt = pd.DataFrame([{t: recs[i][t] for t in BDD_TAGS} for i in ids], index=ids)
    pred = E.predictions_frame(rows, ids, list(BDD_TAGS))
    return gt, pred, recs, ids, spec


def build_view_table(out_csv: str = "outputs/h3_views.csv",
                     results_stem: str = "h3_bdd_signage") -> Any:
    """Per tag, per view: the metrics AND the confusion counts. Step H3.

    The confusion counts are the point. Precision and recall alone make the yellow story
    invisible; tp/fp/fn make it arithmetic -- 96 false positives in both the uniform and
    the enriched view, with F1 at 0.182 and 0.540.
    """
    import pandas as pd

    from . import eval as E

    gt, pred, recs, ids, spec = _frames_for(results_stem)
    uniform = {image_id(f) for f in spec["uniform_filepaths"]}
    views = {
        "uniform_450": [i for i in ids if i in uniform],
        "all_586": ids,
        "state_gt_certain_381": [i for i in ids if not recs[i]["state_gt_uncertain"]],
    }
    out = []
    for view, sel in views.items():
        g, p = gt.loc[sel], pred.loc[sel]
        m = E.per_tag_metrics(g, p, list(BDD_TAGS))
        base = E.majority_baseline(g, list(BDD_TAGS))
        for t in BDD_TAGS:
            out.append({
                "view": view, "n_images": len(sel), "tag": t,
                "precision": m.precision[t], "recall": m.recall[t], "f1": m.f1[t],
                "tp": int(m.tp[t]), "fp": int(m.fp[t]), "fn": int(m.fn[t]),
                "support": int(m.support[t]),
                "prevalence": round(float(g[t].mean()), 4),
                "model_true_rate": round(float(p[t].mean()), 4),
                "baseline_f1": base[t],
                "beats_baseline": bool(m.f1[t] > base[t]),
                "precision_lo": m.precision_lo[t], "precision_hi": m.precision_hi[t],
                "recall_lo": m.recall_lo[t], "recall_hi": m.recall_hi[t],
            })
    df = pd.DataFrame(out).round(4)
    df.to_csv(ROOT / out_csv, index=False)
    return df


def build_size_decomposition(out_csv: str = "outputs/h3_size_decomposition.csv",
                             results_stem: str = "h3_bdd_signage") -> Any:
    """State F1 by the pixel height of the largest LIT light. Step H3, R34 inverted.

    Lit images only: an image with no lit light has no size, and binning it as 0 would put
    the entire no-light population in the smallest band and make that band's score a
    statement about something else.
    """
    import numpy as np
    import pandas as pd

    from . import eval as E

    gt, pred, recs, ids, _ = _frames_for(results_stem)
    px = {i: recs[i]["max_lit_px"] for i in ids}
    lit = [i for i in ids if px[i] > 0]
    out = []
    for lo, hi in SIZE_BANDS:
        sel = [i for i in lit if lo <= px[i] < hi]
        if not sel:
            continue
        m = E.per_tag_metrics(gt.loc[sel], pred.loc[sel], list(STATE_TAGS))
        for t in STATE_TAGS:
            out.append({"size_band_px": f"{lo}-{'inf' if hi > 10 ** 8 else hi}",
                        "n_images": len(sel), "tag": t,
                        "precision": m.precision[t], "recall": m.recall[t], "f1": m.f1[t],
                        "support": int(m.support[t])})
    df = pd.DataFrame(out).round(4)
    df["median_lit_px_subset"] = round(float(np.median([px[i] for i in lit])), 1)
    df["pct_lit_under_15px"] = round(100 * sum(1 for i in lit if px[i] < 15) / len(lit), 1)
    df.to_csv(ROOT / out_csv, index=False)
    return df


def build_h4_comparison(out_csv: str = "outputs/h4_cross_dataset.csv",
                        results_stem: str = "h3_bdd_signage",
                        detail_csv: str = "outputs/f1_per_tag_detail.csv",
                        nuscenes_stem: str = "v1_structured") -> Any:
    """The same model, a near-identical question, three references. Step H4.

    Three rows, and the MIDDLE one is the evidence. It holds nuScenes' camera-arm
    predictions fixed and scores them against H2's `traffic_light_in_view` -- a reference
    whose scope (a light visible in the camera within 50 m) is close to BDD100K's -- instead
    of the scored 30 m x 8 m corridor. That single swap closes ~81% of the gap, which is what
    licenses calling the gap a property of the REFERENCE rather than of the dataset.

    THE CAVEATS ARE COLUMNS, not a footnote: reference construction, scope, occlusion
    modelling, prevalence and sample independence all differ between rows.
    """
    import pandas as pd

    from . import eval as E
    from . import prompts as P

    gt, pred, recs, ids, spec = _frames_for(results_stem)
    uniform = [i for i in ids if i in {image_id(f) for f in spec["uniform_filepaths"]}]
    t = "traffic_light_present"
    m = E.per_tag_metrics(gt.loc[uniform], pred.loc[uniform], [t])

    det = pd.read_csv(ROOT / detail_csv)
    ns = det[(det.tag == NUSCENES_TAG) & (det.condition == NUSCENES_CONDITION)].iloc[0]

    # The re-scored row, from the SAME predictions the detail table was built from.
    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    npred = E.predictions_frame(E.load_rows(nuscenes_stem), tokens, tags)[[NUSCENES_TAG]]
    h2 = pd.read_parquet(OUT / "h2_traffic_light.parquet").set_index("sample_token")
    view = pd.DataFrame({NUSCENES_TAG: h2.loc[tokens, "traffic_light_in_view"].astype(bool)},
                        index=tokens)
    mv = E.per_tag_metrics(view, npred, [NUSCENES_TAG])
    q = NUSCENES_TAG

    rows = [{
        "dataset": "nuScenes", "reference_name": "corridor (the scored tag)",
        "tag": NUSCENES_TAG, "n": int(ns.n_answered),
        "precision": ns.precision, "recall": ns.recall, "f1": ns.f1,
        "tp": int(ns.tp), "fp": int(ns.fp), "fn": int(ns.fn), "support": int(ns.support),
        "prevalence": round(ns.support / ns.n_answered, 4), "baseline_f1": ns.baseline_f1,
        "reference": "mapped fixture inside a 30 m x 8 m forward corridor",
        "scope": "within 30 m ahead, +/-4 m laterally",
        "models_occlusion": False, "independent_samples": False,
    }, {
        "dataset": "nuScenes", "reference_name": "view (H2, same predictions)",
        "tag": NUSCENES_TAG, "n": len(tokens),
        "precision": mv.precision[q], "recall": mv.recall[q], "f1": mv.f1[q],
        "tp": int(mv.tp[q]), "fp": int(mv.fp[q]), "fn": int(mv.fn[q]),
        "support": int(mv.support[q]),
        "prevalence": round(float(view[q].mean()), 4),
        "baseline_f1": float(E.majority_baseline(view, [q])[q]),
        "reference": "mapped fixture projected into the camera frustum within 50 m",
        "scope": "visible in the camera, within 50 m",
        "models_occlusion": False, "independent_samples": False,
    }, {
        "dataset": "BDD100K", "reference_name": "human box",
        "tag": t, "n": len(uniform),
        "precision": m.precision[t], "recall": m.recall[t], "f1": m.f1[t],
        "tp": int(m.tp[t]), "fp": int(m.fp[t]), "fn": int(m.fn[t]),
        "support": int(m.support[t]),
        "prevalence": round(float(gt.loc[uniform, t].mean()), 4),
        "baseline_f1": float(E.majority_baseline(gt.loc[uniform], [t])[t]),
        "reference": "human bounding box on a light that was visible",
        "scope": "anywhere in the image",
        "models_occlusion": True, "independent_samples": True,
    }]
    df = pd.DataFrame(rows).round(4)
    base = df.iloc[0]
    for k in ("f1", "precision", "recall"):
        df[f"delta_{k}_vs_corridor"] = (df[k] - base[k]).round(4)
    gap = df.f1.iloc[2] - df.f1.iloc[0]
    df["share_of_gap_closed_by_scope"] = round((df.f1.iloc[1] - df.f1.iloc[0]) / gap, 4)
    df.to_csv(ROOT / out_csv, index=False)
    return df
