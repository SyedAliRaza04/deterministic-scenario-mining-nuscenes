"""Pre-flight check: run this BEFORE spending GPU hours on a condition.

Exists because F-065. `v2_cot` ran for 5.6 h and produced a number that looked like a
finding, and only a post-hoc inspection of the raw text revealed the prompt had told the
model to think and then forbidden it from thinking. Nothing in the pipeline objected.

Every check here is one that would have caught a defect we actually hit:
  F-065  a prompt that contradicts itself
  F-063  a tag the input cannot possibly answer
  F-058  a bundle shipping stale or stub code
  F-060  a run started against the wrong subset
  F-055  a resume file that would make the run a silent no-op

Usage:  ./venv/bin/python scripts/preflight.py [condition] [model]
        the model argument is E10's: it selects the suffixed results file
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.prompts as P  # noqa: E402

CONDITIONS = ["v1_structured", "v2_cot", "v3_bev", "v3b_bev_symbolic", "v4_both",
              "v6_temporal", "v3_bev_r2", "v3b_bev_symbolic_r2"]

# The repaired BEV arms read the re-rendered inputs, never the published ones.
BEV_R2 = ("v3_bev_r2", "v3b_bev_symbolic_r2")

# Prompts whose text carries a format placeholder. `build_prompt` calls `.format(**fmt)`
# only when fmt is non-empty, so a missing key here raises KeyError rather than producing
# a subtly wrong prompt -- but it raises at RUN time, after the model has loaded, which is
# why it belongs in the preflight rather than in the notebook alone.
PROMPT_FMT = {"v6_temporal": {"n_frames": 5}}
OK, BAD = "  PASS ", "  FAIL "
fails = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global fails
    print((OK if cond else BAD) + name + ("" if cond else f"\n        {detail}"))
    fails += 0 if cond else 1


def main(only: str | None = None, model: str | None = None) -> int:
    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    conds = [only] if only else CONDITIONS

    print(f"\n=== PROMPTS ({len(conds)} condition(s)) ===")
    for c in conds:
        t = P.build_prompt(c, schema, **PROMPT_FMT.get(c, {}))
        leftover = [k for k in ("{n_frames}",) if k in t]
        check(f"{c}: no unfilled format placeholder", not leftover,
              f"{leftover} reached the model as literal text -- add it to PROMPT_FMT")
        # F-065: asking for prose and forbidding it in the same prompt
        wants_prose = "think step by step" in t or "reasoning first" in t
        forbids = "nothing else" in t
        check(f"{c}: does not both request and forbid reasoning",
              not (wants_prose and forbids),
              "the model will obey the prohibition and the condition tests nothing")
        missing = [x for x in tags if f'"{x}"' not in t]
        check(f"{c}: asks for all {len(tags)} scoreable tags", not missing, f"missing {missing[:4]}")
        qc = [x for x, v in schema["tags"].items() if v["role"] != "scored" and f'"{x}"' in t]
        check(f"{c}: asks for no unscoreable tag", not qc, f"asks for {qc}")
        for jargon in ("polygon", "frustum", "keyframe", "instance token"):
            if jargon in t.lower():
                check(f"{c}: no implementation jargon", False, f"leaks {jargon!r}")
                break
        else:
            check(f"{c}: no implementation jargon", True)

    print("\n=== INPUT CAN ANSWER THE QUESTION (F-063) ===")
    import src.bev as B
    if any("bev" in c for c in conds):
        # Which MAP LAYER each map tag actually derives from. Written out rather than
        # guessed from the tag name: at_intersection and intersection_ahead come from
        # road_segment's is_intersection flag, not from a layer called "intersection",
        # and a stem-matching heuristic flags them as unanswerable when they are fine.
        # A check that cries wolf trains you to ignore it, which is how F-060 happened.
        SOURCE_LAYER = {
            "on_drivable_area": "drivable_area", "on_lane": "lane",
            "on_ped_crossing": "ped_crossing", "on_stop_line": "stop_line",
            "at_intersection": "road_segment", "intersection_ahead": "road_segment",
            "ped_crossing_ahead": "ped_crossing", "stop_line_ahead": "stop_line",
            "traffic_light_ahead": "traffic_light",
        }
        drawn = set(B.DRAWN_LAYERS)   # polygons AND the traffic-light points (F-063c)
        missing_map = [t for t, layer in SOURCE_LAYER.items() if layer not in drawn]
        check("BEV draws every map layer its tags derive from",
              not missing_map,
              f"{missing_map} cannot be answered from ANY BEV input - exclude them from "
              "the camera-vs-BEV comparison (F-063) or add the layer to src/bev.py")
        unmapped = [t for t, v in schema["tags"].items()
                    if v["role"] == "scored" and v["family"] == "map_context"
                    and t not in SOURCE_LAYER]
        check("every map tag has a known source layer", not unmapped,
              f"{unmapped} unaccounted for - update SOURCE_LAYER")

    print("\n=== BUNDLE (F-058) ===")
    b = ROOT / "outputs" / "colab_bundle.tar.gz"
    mani = ROOT / "outputs" / "colab_bundle_manifest.json"
    check("bundle exists", b.exists())
    if b.exists() and mani.exists():
        rec = json.loads(mani.read_text())["sha256_12"]
        stale = [n for n, h in rec.items()
                 if hashlib.sha256((ROOT / n).read_bytes()).hexdigest()[:12] != h]
        check("bundle matches current source", not stale,
              f"{stale} changed since the bundle was built - rebuild and RE-UPLOAD")
        with tarfile.open(b) as tar:
            names = tar.getnames()
        check("bundle carries the 628-token subset (F-060)",
              "outputs/vlm_subset_tokens.json" in names)
        # E9. Checking that the NOTEBOOK mentions the sequences is not the same as the
        # BUNDLE containing them -- the first version of this preflight passed while the
        # bundle held no sequences at all, so "safe to run" would have meant a 419 MB
        # upload followed by an assert. Check the artifact, not the reference to it.
        if only == "v6_temporal" or (not only and "v6_temporal" in conds):
            has_seq = "outputs/temporal_sequences.json" in names
            check("bundle carries the temporal sequences (E9)", has_seq,
                  "rebuild with src.data.build_colab_bundle(temporal=True)")
            if has_seq:
                import src.data as _D
                seqs = json.loads(
                    (ROOT / "outputs" / "temporal_sequences.json").read_text())["sequences"]
                need = {f for s in seqs.values() for f in s}
                have = {n.split("/")[-1][:-4] for n in names if n.startswith("images/")}
                gap = need - have
                check(f"bundle carries every frame of all {len(seqs)} sequences",
                      not gap,
                      f"{len(gap)} of {len(need)} sequence frames absent; the prompt "
                      "promises 5 frames and the model would get fewer -- the original "
                      "query_multiframe defect rebuilt")
                full = sum(1 for v in seqs.values() if len(v) == _D.TEMPORAL_HISTORY + 1)
                print(f"       sequences at full length: {full}/{len(seqs)} "
                      f"(the rest start near a scene boundary and are genuinely shorter)")
        if any(c in BEV_R2 for c in conds):
            # The repair is only real if the SESSION reads the repaired inputs. A bundle
            # carrying the old rasters would re-run the arm against the very picture the
            # re-run exists to replace, and nothing downstream could tell (F-063c).
            import src.bev as _B

            sub = json.loads((ROOT / "outputs" / "vlm_subset_tokens.json").read_text())
            toks = set(sub["sample_tokens"])
            r2_png = {n.split("/")[-1][:-4] for n in names if n.startswith("bev_r2/")}
            check("bundle carries the RE-RENDERED bev_r2 rasters for the subset",
                  toks <= r2_png,
                  f"{len(toks - r2_png)} of {len(toks)} absent; rebuild the bundle after "
                  "src.bev.build_bev_batch()")
            r2_jsonl = Path(_B.BEV_R2_JSONL).name
            check(f"bundle carries outputs/{r2_jsonl}",
                  f"outputs/{r2_jsonl}" in names)
            if (ROOT / _B.BEV_R2_JSONL).exists():
                rows = [json.loads(x) for x in (ROOT / _B.BEV_R2_JSONL).open()]
                described = {r["sample_token"] for r in rows}
                check("the re-rendered description covers the subset", toks <= described,
                      f"{len(toks - described)} tokens undescribed")
                geom = sum(1 for r in rows if "present out to" in r["description"])
                lights = sum(1 for r in rows if "fixture:" in r["description"])
                check("every re-rendered description states map geometry (F-064b)",
                      geom == len(rows), f"{len(rows) - geom} descriptions state none")
                print(f"       {lights}/{len(rows)} descriptions list a traffic-light "
                      "fixture; the rest have none in range")
        res = sorted(Path(n).name for n in names if n.startswith("results/"))
        print(f"       completed results carried for resume: {res}")
        # Only a condition you intend to RUN can be blocked by its own rows. For a
        # finished condition those rows are the point of carrying them, so flagging
        # them as failures is noise - and noise is what makes a real warning invisible.
        if only:
            import src.vlm as _V

            target = _V.result_filename(only, model)
            print(f"       this run would write: results/{target}")
            blocked = f"results/{target}" in names
            check(f"{only}: no completed rows that would make this run a no-op (F-055)",
                  not blocked,
                  f"results/{target} is in the bundle; every frame would be skipped. "
                  "Move it aside before re-running.")
        else:
            print(f"       (pass a condition name to check whether ITS re-run is blocked)")

    print("\n=== NOTEBOOK (F-060) ===")
    nb = json.loads((ROOT / "notebooks" / "stage_e_kaggle.ipynb").read_text())
    txt = "".join("".join(c["source"]) for c in nb["cells"])
    check("notebook asserts the subset instead of warning", "assert _vlm.exists()" in txt
          or "assert len(tokens) == 628" in txt or "== 628" in txt)
    check("notebook asserts GPU compute capability >= 7.5 (F-061)", "(7, 5)" in txt)
    # A substring test is not enough here: `'v6_temporal'` appears in the notebook's own
    # condition table and in a comment, so `f"'{c}'" in txt` passed while `images_for`
    # had no branch for it and the run would have raised KeyError on frame 1. Check the
    # dispatch itself -- either a key in the images_for mapping or an explicit branch.
    body = txt.split("def images_for(tok):", 1)[-1] if "def images_for(tok):" in txt else ""
    for c in conds:
        handled = (f"'{c}': [" in body) or (f"CONDITION == '{c}'" in txt)
        check(f"{c}: handled by images_for", handled,
              "the run would raise KeyError on the first frame")
    if "v6_temporal" in conds:
        check("notebook loads the precomputed temporal sequences",
              "temporal_sequences.json" in txt,
              "rebuild the bundle with build_colab_bundle(temporal=True); without the "
              "sequences the prompt promises five frames and the model gets one")
        check("run cell passes n_frames to build_prompt (F-065's shape)",
              "n_frames" in txt.split("run_batch(", 1)[-1][:600]
              or "'n_frames': 5} if CONDITION == 'v6_temporal'" in txt,
              "PROMPT_V6_TEMPORAL.format would raise KeyError AFTER the model loads")

    print(f"\n{'ALL CHECKS PASSED' if not fails else f'{fails} CHECK(S) FAILED'} — "
          f"{'safe to run' if not fails else 'DO NOT START THE RUN'}\n")
    return 1 if fails else 0


# --- the two runs that are not Stage E conditions -------------------------------------
#
# H3 and G2b have their own bundles, their own vocabularies and their own notebooks, so
# the Stage E checks above do not apply to them. What DOES carry over is the class of
# defect: a stale bundle (F-058), a prompt that contradicts itself (F-065), a hole in the
# image set that shortens the run silently (F-048), and completed rows that would make the
# whole session a no-op (F-055).


def _bundle_checks(tag: str, bundle: Path, manifest: Path, needed: set[str],
                   extras: tuple[str, ...]) -> None:
    check(f"{tag}: bundle exists", bundle.exists(),
          f"build it first; see notebooks/README.md")
    if not bundle.exists():
        return
    if manifest.exists():
        rec = json.loads(manifest.read_text())["sha256_12"]
        stale = [n for n, h in rec.items()
                 if hashlib.sha256((ROOT / n).read_bytes()).hexdigest()[:12] != h]
        check(f"{tag}: bundle matches current source", not stale,
              f"{stale} changed since it was built - rebuild and RE-UPLOAD")
    with tarfile.open(bundle) as tar:
        names = set(tar.getnames())
    for e in extras:
        check(f"{tag}: bundle carries {e}", e in names)
    gap = needed - names
    check(f"{tag}: bundle carries all {len(needed)} images", not gap,
          f"{len(gap)} absent, e.g. {sorted(gap)[:3]} - a hole becomes a short results "
          "file that nothing flags (F-048)")
    size = bundle.stat().st_size / 1048576
    print(f"       {tag}: {size:.1f} MB, {len([n for n in names if n.startswith('images/')])} images")


def h3() -> int:
    """Pre-flight for Step H3 (BDD100K signage)."""
    import src.bdd as B

    print("\n=== H3 PROMPT (F-065 / R4) ===")
    p = B.build_prompt()
    asks = "think step by step" in p or "reasoning first" in p
    check("H3: does not both request and forbid reasoning", not (asks and "nothing else" in p))
    missing = [t for t in B.BDD_TAGS if f'"{t}"' not in p]
    check(f"H3: asks for all {len(B.BDD_TAGS)} tags", not missing, f"missing {missing}")
    check("H3: asks for three state booleans, never one colour",
          all(x not in p.lower() for x in ("what colour", "what color")),
          "34.0% of lit images show more than one distinct lit colour (R22)")

    print("\n=== H3 SUBSET ===")
    spec_p = ROOT / "outputs/h3_bdd_subset.json"
    check("H3: subset built", spec_p.exists(),
          'run  ./venv/bin/python -c "import src.bdd as B; B.build_subset()"')
    if not spec_p.exists():
        return 1 if fails else 0
    spec = json.loads(spec_p.read_text())
    check("H3: yellow clears the 30-positive floor (D-032)",
          spec["positives_all"]["yellow_light_visible"] >= 30,
          f"only {spec['positives_all']['yellow_light_visible']}")
    print(f"       {spec['n_total']} images, {spec['n_uniform']} uniform, "
          f"state GT uncertain on {spec['state_gt_uncertain_pct']}%")

    print("\n=== H3 BUNDLE (F-058 / F-048) ===")
    needed = {f"images/{Path(r['filepath']).name}" for r in spec["records"]}
    _bundle_checks("H3", ROOT / "outputs/h3_bundle.tar.gz",
                   ROOT / "outputs/h3_bundle_manifest.json", needed,
                   ("outputs/h3_bdd_subset.json", "src/bdd.py", "src/eval.py"))

    print("\n=== H3 NOTEBOOK ===")
    nb = json.loads((ROOT / "notebooks/stage_h3_kaggle.ipynb").read_text())
    txt = "".join("".join(c["source"]) for c in nb["cells"])
    check("H3: notebook asserts GPU compute capability >= 7.5", "(7, 5)" in txt)
    check("H3: notebook hard-fails on a missing image", "assert not missing" in txt)
    check("H3: notebook scores in-session", "B.score(" in txt,
          "the ground truth is in the bundle; scoring later made F-062 a 'first look'")
    _no_op(ROOT / "outputs/h3_bundle.tar.gz", "h3_bdd_signage", "H3")
    return 1 if fails else 0


def g2() -> int:
    """Pre-flight for Step G2b (the VLM segmentation arm)."""
    import src.storyboard as S

    print("\n=== G2b PROMPT (F-065 / R4) ===")
    p = S.VLM_PAIR_PROMPT
    asks = "think step by step" in p or "reasoning first" in p
    check("G2b: does not both request and forbid reasoning", not (asks and "nothing else" in p))
    check("G2b: asks for the tag the scorer reads", f'"{S.G2_VLM_TAG}"' in p,
          f"the scorer looks for {S.G2_VLM_TAG!r} and the prompt never names it")
    check("G2b: tells the model which image is earlier", "first image is the earlier" in p,
          "the pair is ordered and the whole question depends on knowing that")

    print("\n=== G2b WORK LIST ===")
    items_p = ROOT / "outputs/g2_vlm_items.json"
    check("G2b: work list built", items_p.exists(),
          'run  ./venv/bin/python -c "import src.storyboard as S; S.vlm_pair_items()"')
    if not items_p.exists():
        return 1 if fails else 0
    spec = json.loads(items_p.read_text())
    published = None
    csv = ROOT / "outputs/g2_segmentation.csv"
    if csv.exists():
        import csv as _csv
        published = float(next(_csv.DictReader(csv.open()))["gold_density_per_kf"])
    check("G2b: the scene draw keeps the published chance floor",
          published is None or abs(spec["gold_density_per_kf"] - published) < 0.01,
          f"density {spec['gold_density_per_kf']} vs the 850-scene {published}; a draw "
          "that enriches boundaries lifts the null and flatters every segmenter (F-070)")
    check("G2b: the draw includes scenes with NO boundary",
          spec["n_scenes_with_no_boundary"] > 0,
          "those are the only scenes that test false positives")
    print(f"       {spec['n_items']} items, {spec['n_scenes']} scenes, "
          f"{spec['n_gold_boundaries']} gold, chance {spec['chance_f1_closed_form']}")
    print(f"       estimate {spec['est_hours_at_12s']}-{spec['est_hours_at_20s']} h")

    print("\n=== G2b BUNDLE (F-058 / F-048) ===")
    needed = {f"images/{t}.jpg" for it in spec["items"]
              for t in (it["before_token"], it["after_token"])}
    _bundle_checks("G2b", ROOT / "outputs/g2_bundle.tar.gz",
                   ROOT / "outputs/g2_bundle_manifest.json", needed,
                   ("outputs/g2_vlm_items.json", "src/storyboard.py", "src/eval.py"))

    print("\n=== G2b NOTEBOOK ===")
    nb = json.loads((ROOT / "notebooks/stage_g2_kaggle.ipynb").read_text())
    txt = "".join("".join(c["source"]) for c in nb["cells"])
    check("G2b: notebook asserts GPU compute capability >= 7.5", "(7, 5)" in txt)
    check("G2b: notebook sends before THEN after", "before_token" in txt
          and txt.index("before_token") < txt.index("after_token"))
    check("G2b: notebook reads the null's BAND, not its mean", "f1_hi" in txt,
          "a fake model answering at random cleared the null's mean by +0.051 at +/-1")
    _no_op(ROOT / "outputs/g2_bundle.tar.gz", "g2_vlm_pairs", "G2b")
    return 1 if fails else 0


def f5() -> int:
    """Pre-flight for Step F5 (NuScenes-QA). It rides in the Stage E bundle and notebook,
    but its prompt is not a registered TAG prompt, so `main()` raises KeyError on it --
    a traceback instead of a verdict, on the one command the run is supposed to gate on."""
    import src.nuqa as Q

    print("\n=== F5 PROMPT (F-065) ===")
    p = Q.ANSWER_PROMPT
    asks = "think step by step" in p or "reasoning first" in p
    check("F5: does not both request and forbid reasoning", not (asks and "nothing else" in p))
    check("F5: carries the {question} placeholder", "{question}" in p,
          "every call would ask the same question")

    print("\n=== F5 QUESTION LIST ===")
    qp = ROOT / "outputs/f5_questions.json"
    check("F5: question list built", qp.exists(),
          'run  ./venv/bin/python -c "import src.nuqa as Q; Q.build_question_list()"')
    if not qp.exists():
        return 1 if fails else 0
    spec = json.loads(qp.read_text())
    qs = spec["questions"]
    check("F5: every question carries a gold answer", all(q.get("answer") for q in qs),
          "scoring is offline only because val ships answers; without them it is not")
    check("F5: question ids are unique", len({q["qid"] for q in qs}) == len(qs),
          "19 question STRINGS repeat on the same frame; the id is what separates them")
    check("F5: the paper's two report axes are present",
          all(q.get("template_type") and q.get("num_hop") is not None for q in qs),
          "per-type accuracy is how NuScenes-QA reports, and the types differ 3.5x")
    print(f"       {spec['n_questions']} questions over {spec['n_frames']} val-split frames")

    print("\n=== F5 BUNDLE / NOTEBOOK ===")
    b = ROOT / "outputs/colab_bundle.tar.gz"
    check("F5: Stage E bundle exists", b.exists())
    if b.exists():
        with tarfile.open(b) as tar:
            names = set(tar.getnames())
        check("F5: bundle carries the questions", "outputs/f5_questions.json" in names)
        check("F5: bundle carries src/nuqa.py", "src/nuqa.py" in names)
        gap = {f"images/{q['sample_token']}.jpg" for q in qs} - names
        check("F5: bundle carries every questioned frame", not gap,
              f"{len(gap)} absent - those questions would raise on the first call")
        _no_op(b, "f5_nuscenes_qa", "F5")
    nb = json.loads((ROOT / "notebooks/stage_e_kaggle.ipynb").read_text())
    txt = "".join("".join(c["source"]) for c in nb["cells"])
    check("F5: notebook routes the question ids, not frame tokens",
          "run_tokens = list(F5Q)" in txt)
    check("F5: notebook asks ONE question per call", "prompt_for=lambda qid" in txt,
          "61.0% of questions contain another question's answer verbatim, so a batched "
          "prompt hands the model a candidate list read off its own input")
    return 1 if fails else 0


def _no_op(bundle: Path, condition: str, tag: str) -> None:
    """F-055: completed rows in the bundle would make the whole session skip every item."""
    if not bundle.exists():
        return
    import src.vlm as _V

    with tarfile.open(bundle) as tar:
        names = set(tar.getnames())
    target = f"results/{_V.result_filename(condition)}"
    check(f"{tag}: no completed rows that would make this run a no-op (F-055)",
          target not in names,
          f"{target} is in the bundle; every item would be skipped")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    # H3 and G2b are not Stage E conditions and share none of its bundle, so they route to
    # their own checks rather than being bolted onto CONDITIONS, where every Stage E
    # assertion would fire against a vocabulary that was never meant to satisfy it.
    if arg in ("h3", "g2", "f5", "f5_nuscenes_qa"):
        rc = {"h3": h3, "g2": g2, "f5": f5, "f5_nuscenes_qa": f5}[arg]()
        print(f"\n{'ALL CHECKS PASSED' if not fails else f'{fails} CHECK(S) FAILED'} — "
              f"{'safe to run' if not fails else 'DO NOT START THE RUN'}\n")
        sys.exit(rc)
    sys.exit(main(arg, sys.argv[2] if len(sys.argv) > 2 else None))
