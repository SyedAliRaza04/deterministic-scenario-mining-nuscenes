"""Inspect NuScenes-QA's val questions and decide whether Stage F5 option C is viable.

THE QUESTION THIS SETTLES. NuScenes-QA asks about all six BEV sectors while this project
holds CAM_FRONT only, so a front-camera pass cannot answer the rear-referencing questions
for reasons that have nothing to do with the model. Option C (run everything, then report
the front subset, the aggregate, and the gap) needs the questions to be PARTITIONABLE into
front- and rear-referencing. Option B (run only the front subset) needs the same partition
BEFORE the run, so it is strictly the more fragile of the two.

Two ways the partition might be available, and this checks both:
  1. a structured field per question (type / hop / relation) -- the paper reports per-type
     accuracy, so something like it almost certainly exists, but its KEY is unverified;
  2. failing that, the relation as literal text. The templates embed it --
     "Are there any <A2><O2>s to the <R> of the <A><O>?" -- and the six relations are
     front / back / front left / front right / back left / back right.

It also reports how many questions land on OUR frames, because only the 111 execution-subset
frames that fall in the official nuScenes val split are comparable to published numbers
(measured: 27 of our 150 scenes, 25 of the 628-frame subset's scenes, are in val).

No GPU, no model, no writes outside outputs/.

Usage:
    ./venv/bin/python scripts/inspect_nuscenes_qa.py path/to/NuScenes_val_questions.json
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The six relations NuScenes-QA's scene graphs use, longest first so "front left" is
# matched before "front". Word-boundary anchored: "front" must not fire inside "in front
# of the car" when the relation is actually something else, and must not match "fronting".
RELATIONS = ["front left", "front right", "back left", "back right", "front", "back"]
REAR = {"back", "back left", "back right"}

# Keys that would give the partition directly. Checked rather than assumed -- the official
# loader only ever touches question / sample_token / answer, so the field name is unverified.
CANDIDATE_FIELDS = ("type", "question_type", "template_type", "qtype", "category",
                    "hop", "num_hop", "relation", "rel", "tag", "layer")


def relation_of(question: str) -> str | None:
    """The relation a question references, read from its text. None = no relation term."""
    q = question.lower()
    for rel in RELATIONS:
        if re.search(rf"\b{rel}\b", q):
            return rel
    return None


def main(path: str) -> int:
    data = json.loads(Path(path).read_text())
    items = data["questions"] if isinstance(data, dict) and "questions" in data else data
    print(f"file: {path}")
    print(f"top-level: {type(data).__name__}"
          + (f", keys={list(data)[:8]}" if isinstance(data, dict) else ""))
    print(f"questions: {len(items):,}\n")

    sample = items[0]
    print("=== one item, verbatim ===")
    print(json.dumps(sample, indent=2)[:700])
    print(f"\nkeys present: {sorted(sample)}\n")

    # 1. is there a structured field?
    print("=== structured partition fields ===")
    found = [k for k in CANDIDATE_FIELDS if k in sample]
    if found:
        for k in found:
            vals = collections.Counter(str(it.get(k)) for it in items)
            shown = dict(vals.most_common(12))
            print(f"  {k!r}: {len(vals)} distinct -> {shown}")
    else:
        print(f"  NONE of {CANDIDATE_FIELDS} present.")
        print("  -> option B (filter before running) cannot use a field; it would have to")
        print("     rely on the text parse below, which is a heuristic over the benchmark.")
    print()

    # 2. the text parse -- the fallback that decides whether C can be decomposed
    print("=== relation recoverable from question text? ===")
    rels = collections.Counter(relation_of(it["question"]) for it in items)
    total = len(items)
    for rel, n in rels.most_common():
        label = rel if rel is not None else "(no relation term)"
        print(f"  {label:<14} {n:>7,}  {100*n/total:5.1f}%")
    rear = sum(n for r, n in rels.items() if r in REAR)
    front_or_none = total - rear
    print(f"\n  rear-referencing : {rear:,} ({100*rear/total:.1f}%)  <- unanswerable from CAM_FRONT")
    print(f"  front / no relation: {front_or_none:,} ({100*front_or_none/total:.1f}%)")

    # 3. how much of it lands on frames we can actually score
    print("\n=== overlap with our execution subset ===")
    try:
        from nuscenes.utils.splits import create_splits_scenes
        import src.data as D

        val_scenes = set(create_splits_scenes()["val"])
        meta = D._load_meta()
        name_of = {s["token"]: s["name"] for s in meta["scene"]}
        scene_of = {s["token"]: s["scene_token"] for s in meta["sample"]}
        ours = json.loads((ROOT / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
        our_val = {t for t in ours if name_of[scene_of[t]] in val_scenes}

        by_token = collections.Counter(it["sample_token"] for it in items)
        hit = {t: by_token.get(t, 0) for t in our_val}
        covered = sum(1 for n in hit.values() if n)
        qs = sum(hit.values())
        print(f"  our 628-frame subset, val-split only: {len(our_val)} frames")
        print(f"  of those, present in this file      : {covered}")
        print(f"  questions on them                   : {qs:,}"
              + (f"  (median {sorted(hit.values())[len(hit)//2]}/frame)" if hit else ""))
        rear_ours = sum(1 for it in items
                        if it["sample_token"] in our_val and relation_of(it["question"]) in REAR)
        if qs:
            print(f"  rear-referencing among them         : {rear_ours:,} ({100*rear_ours/qs:.1f}%)")
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped: {type(e).__name__}: {e})")

    # 4. the verdict this script exists to give
    print("\n=== verdict ===")
    if found:
        print("  DECOMPOSABLE on the paper's own axes: template_type and num_hop are")
        print("  present, which is exactly how NuScenes-QA reports per-type accuracy, so")
        print("  our results slot beside their published table directly.")
    else:
        print("  NOT decomposable on a structured field. Fall back to the relation parse")
        print("  below, and report it as a text-derived split with its limits stated.")

    print("""
  BUT NOT on front-vs-rear. The relation parse above is a WEAK signal and must not be
  used to decide what a front camera can answer. Measured on this file, three separate
  ways it is wrong:

    * MULTI-RELATION. "What is the moving thing that is to the back of the moving car
      and the front of the moving trailer?" references both sectors; relation_of()
      returns whichever matches first and silently mislabels it.
    * OBJECT-RELATIVE vs EGO-RELATIVE. "to the back left of the stopped construction
      vehicle" is relative to an OBJECT -- if that vehicle is ahead of the ego, the
      region behind it is plainly visible in CAM_FRONT. Only "to the back of me" is
      genuinely occluded.
    * ZERO-HOP IS NOT EXEMPT. num_hop == 0 questions carry no relation term at all
      (verified: 27,244/27,244), but "How many moving buses are there?" still needs the
      full 360 degrees to count correctly, so a front camera under-answers them too.

  The field-of-view gap is therefore PERVASIVE, not a filterable subset. That rules out
  running a "front-answerable" subset (option B) -- no clean criterion exists -- and it
  means the honest design is to run everything and decompose by template_type / num_hop,
  reporting the single-camera limitation as a measured property rather than filtering it
  out of sight. Same failure class as F-033: a test that looks right and means something
  else.""")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: pass the path to NuScenes_val_questions.json")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
