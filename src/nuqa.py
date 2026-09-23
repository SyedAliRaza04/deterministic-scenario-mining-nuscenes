"""NuScenes-QA — Step F5, external benchmark validation.

WHY THIS EXISTS. Every number in this thesis so far is scored against ground truth the
thesis built itself. F5 scores the same model against an EXTERNAL benchmark with published
metrics, so the results can be placed beside other people's.

WHAT IT IS. NuScenes-QA (AAAI 2024) generates questions programmatically from nuScenes' 3D
annotations via scene graphs and templates - the nearest published neighbour to this
project's own method. Val ships answers, so scoring is offline. The answer space is a closed
30-word vocabulary and the paper reports top-1 accuracy per `template_type` and `num_hop`.

THE FILE. `data/nuscenes_qa/NuScenes_val_questions.json`, 25,432,339 bytes. NOT
`samples.json` - that name belongs to the BDD100K export in `data/bdd100k/` and reading it
here would silently produce nonsense rather than an error.

THE UNIT OF WORK IS A QUESTION, NOT A FRAME, and that is a measured decision rather than a
convenience. Batching a frame's questions into one call is ~13x cheaper and was rejected:
**61.0% of our 1,471 questions contain another question's answer string verbatim in their own
text**, and for `object` questions - a 9-way choice - **85.0%** have their answer noun named
by a sibling. A batched prompt hands the model a candidate list read straight off its own
input, which changes the information set and makes the number incomparable to the published
ones. Separately, 19 of the 1,471 are exact duplicate question strings within one frame, and
a JSON object keyed by question text silently keeps the last and drops the rest.

Build:  ./venv/bin/python -c "import src.nuqa as Q; Q.build_question_list()"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
QA_JSON = ROOT / "data/nuscenes_qa/NuScenes_val_questions.json"

# The answer must arrive as JSON, not as a bare word. `extract_json("truck")` returns
# (None, "no_json_object") and so does "The answer is truck." -- five such replies in a row
# trip run_batch's `max_consecutive_failures` guard and kill the run. Six extra output tokens
# buy the fail-fast guard back. (`prompts._YESNO_TEMPLATE` still ends "Answer with exactly
# one word", so Step E8 would hit this same trap if it were ever re-run unchanged.)
ANSWER_PROMPT = (
    "Look at this photograph taken from the front camera of a car.\n\n"
    "Answer the question below using ONE word or number from what you can see.\n"
    "Do not explain. Do not add units.\n\n"
    "QUESTION: {question}\n\n"
    'Reply with exactly one JSON object and nothing else: {{"answer": "<your answer>"}}'
)

# Normalisation that cannot change which vocabulary item is meant. Each was checked against
# the 30-answer space: no item begins with an article, and stripping a trailing "s" is only
# applied when the singular IS in the vocabulary, so "buss" -> "bus" while "status" is safe.
_ARTICLES = ("a ", "an ", "the ")


def load_questions(path: Path | str = QA_JSON) -> list[dict]:
    """The val questions, verbatim. Index in this list is the question's identity."""
    return json.loads(Path(path).read_text())["questions"]


def question_id(sample_token: str, index: int) -> str:
    """`<sample_token>:<index into the released array>`.

    run_batch's `tokens` only has to be unique and resolvable by `images_for`, so making the
    QUESTION the unit costs nothing and buys resume-by-question for free. The index comes
    from the released file's own ordering, so it is stable, and the frame stays recoverable
    as `qid.split(":")[0]` -- which matters because 19 questions are exact duplicates of a
    sibling's text and could not be identified by their text alone.
    """
    return f"{sample_token}:{index}"


def build_question_list(out_path: str = "outputs/f5_questions.json") -> dict:
    """The questions F5 will ask: every val question on our val-split execution frames.

    Only the 111 frames of the 628-frame execution subset that fall in the OFFICIAL nuScenes
    val split are used, because the published numbers this is meant to sit beside are val
    numbers. The other 517 are in those benchmarks' train files (D-049/F-070).
    """
    import pandas as pd
    from nuscenes.utils.splits import create_splits_scenes

    from . import data as D

    val_scenes = set(create_splits_scenes()["val"])
    meta = D._load_meta()
    name_of = {s["token"]: s["name"] for s in meta["scene"]}
    scene_of = {s["token"]: s["scene_token"] for s in meta["sample"]}

    ours = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    our_val = {t for t in ours if name_of[scene_of[t]] in val_scenes}

    items = []
    for i, q in enumerate(load_questions()):
        if q["sample_token"] in our_val:
            items.append({"qid": question_id(q["sample_token"], i),
                          "sample_token": q["sample_token"], "question": q["question"],
                          "answer": q["answer"], "template_type": q["template_type"],
                          "num_hop": q["num_hop"]})
    payload = {"n_frames": len(our_val), "n_questions": len(items),
               "source": str(QA_JSON.name), "questions": items}
    (ROOT / out_path).write_text(json.dumps(payload))
    return {k: v for k, v in payload.items() if k != "questions"}


def normalise(ans: str, vocabulary: set[str]) -> str:
    """Casefold, strip punctuation and articles, de-pluralise ONLY into the vocabulary.

    The line this must not cross: anything that could map a WRONG answer onto a right one.
    Stripping "s" unconditionally would turn "cars" into "car" and also "status" into "statu";
    it is applied only when the result is itself a vocabulary item.
    """
    a = ans.strip().casefold().strip(" .!\"'")
    for art in _ARTICLES:
        if a.startswith(art):
            a = a[len(art):]
    a = re.sub(r"\s+", " ", a)
    if a not in vocabulary and a.endswith("s") and a[:-1] in vocabulary:
        a = a[:-1]
    return a


def score(results_stem: str = "f5_nuscenes_qa",
          questions_path: str = "outputs/f5_questions.json") -> dict:
    """Top-1 accuracy overall, per template_type and per num_hop, against the baselines.

    The paper's own reporting axes, so the numbers slot beside its table. Every cell carries
    its majority-answer baseline, because a 57.5% score on `exist` is exactly what answering
    "yes" to everything achieves (F-038/F2).
    """
    import pandas as pd

    from . import eval as E

    spec = json.loads((ROOT / questions_path).read_text())
    by_qid = {q["qid"]: q for q in spec["questions"]}
    vocab = {q["answer"].casefold() for q in spec["questions"]}

    rows = []
    for r in E.load_rows(results_stem):
        q = by_qid.get(r["sample_token"])
        if q is None or r.get("backend_error"):
            continue
        parsed, _ = (r.get("values") or {}), None
        got = parsed.get("answer") if isinstance(parsed, dict) else None
        if got is None:                       # values is tag-shaped; fall back to raw JSON
            try:
                got = json.loads(r["raw"][r["raw"].index("{"):r["raw"].rindex("}") + 1]).get("answer")
            except Exception:                 # noqa: BLE001
                got = None
        rows.append({"qid": q["qid"], "template_type": q["template_type"],
                     "num_hop": q["num_hop"], "gold": q["answer"].casefold(),
                     "pred": None if got is None else normalise(str(got), vocab),
                     "parse_ok": str(r.get("parse_reason", "")).startswith("ok")})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0, "note": "no results yet"}
    df["correct"] = df.pred == df.gold

    def block(g):
        maj = g.gold.value_counts().idxmax()
        return pd.Series({"n": len(g), "accuracy": round(g.correct.mean(), 4),
                          "majority_answer": maj,
                          "baseline": round((g.gold == maj).mean(), 4)})

    return {
        "n_questions": int(len(df)),
        "n_unanswered": int(df.pred.isna().sum()),
        "parse_ok_rate": round(float(df.parse_ok.mean()), 4),
        "overall": block(df).to_dict(),
        "by_template_type": df.groupby("template_type").apply(block).to_dict("index"),
        "by_num_hop": df.groupby("num_hop").apply(block).to_dict("index"),
    }


# --- F-096: what the F5 run says, as tables with builders (R30) ------------------------
#
# THE AGGREGATE IS A TRAP, and D-049 said so before the run: the per-type majority
# baselines differ 3.5x, so no single number means anything. Measured, it is worse than a
# warning. Overall accuracy is 0.320 against a global majority of 0.228 -- apparently a win
# -- while the model loses to or ties the majority answer in EVERY one of the five template
# types. The global baseline answers "yes" to everything, which can never be right on a
# count or object question; a predictor that answers each TYPE's own majority scores 0.354,
# above the model. The apparent win is the model knowing what KIND of answer a question
# wants, not what the answer is.
#
# THE FAILURE HAS A DIRECTION. The model under-counts 269 times and over-counts 14 of 296,
# says "yes" to 21.9% of existence questions against a reference 57.5%, and answers "not
# visible" or "unknown" 45 times. Two explanations predict exactly that and this data cannot
# separate them: NuScenes-QA derives answers from all six cameras while the model sees one,
# and this model retreats to the negative answer under uncertainty elsewhere in the thesis
# (F-073: zero True answers to the motion tags from five frames). The one test that could
# separate them -- ego-front against ego-behind questions -- has 116 and 123 questions and a
# 95% interval 0.26 wide, so it is reported as underpowered rather than as evidence.

# Synonyms that are UNAMBIGUOUSLY the same answer. Deliberately short: "walking" is not
# mapped to "moving" and "van" is not mapped to anything, because a lenient score that
# guesses is a second model, not a sensitivity check. Its only job is to bound how much of
# the gap is formatting (measured: +0.008).
LENIENT_SYNONYMS = {"bike": "bicycle", "bikes": "bicycle",
                    "cone": "traffic cone", "cones": "traffic cone",
                    "in motion": "moving"}
REFUSALS = ("not visible", "unknown", "none", "cannot determine")
EGO_DIRECTION = re.compile(
    r"\b(front left|front right|back left|back right|front|back)\s+of me\b")


def _scored_rows(results_stem: str = "f5_nuscenes_qa",
                 questions_path: str = "outputs/f5_questions.json") -> list[dict]:
    """One dict per question: gold, the raw answer, strict and lenient correctness."""
    from . import eval as E

    spec = json.loads((ROOT / questions_path).read_text())
    by_qid = {q["qid"]: q for q in spec["questions"]}
    vocab = {q["answer"].casefold() for q in spec["questions"]}
    out = []
    for r in E.load_rows(results_stem):
        q = by_qid.get(r["sample_token"])
        if q is None or r.get("backend_error"):
            continue
        raw = r.get("raw") or ""
        try:
            got = str(json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("answer"))
        except Exception:                               # noqa: BLE001
            got = ""
        gold = q["answer"].casefold()
        strict = normalise(got, vocab)
        d = EGO_DIRECTION.search(q["question"].lower())
        out.append({"qid": q["qid"], "frame": q["sample_token"],
                    "template_type": q["template_type"], "num_hop": q["num_hop"],
                    "gold": gold, "raw": got.casefold().strip(), "pred": strict,
                    "correct": strict == gold,
                    "correct_lenient": LENIENT_SYNONYMS.get(got.casefold().strip(), strict) == gold,
                    "in_vocab": strict in vocab,
                    "ego_direction": None if not d else
                    ("behind" if d.group(1).startswith("back") else "front")})
    return out


def _type_aware_baseline(rows: list[dict]) -> float:
    """Accuracy of answering each template type's own most frequent answer."""
    from collections import Counter

    maj = {}
    for t in {r["template_type"] for r in rows}:
        maj[t] = Counter(r["gold"] for r in rows if r["template_type"] == t).most_common(1)[0][0]
    return sum(r["gold"] == maj[r["template_type"]] for r in rows) / len(rows)


def build_f5_tables(results_stem: str = "f5_nuscenes_qa",
                    by_type_csv: str = "outputs/f5_by_type.csv",
                    direction_csv: str = "outputs/f5_answer_direction.csv") -> tuple[Any, Any]:
    """F-096's two tables. `f5_by_type` is the result; `f5_answer_direction` is why.

    Every accuracy sits beside the baseline it must beat, and the overall row carries BOTH
    baselines so the trap is visible in the artifact rather than only in this comment.
    """
    from collections import Counter

    import numpy as np
    import pandas as pd

    rows = _scored_rows(results_stem)
    df = pd.DataFrame(rows)
    out = []
    for t, g in df.groupby("template_type"):
        maj = g.gold.value_counts().idxmax()
        base = float((g.gold == maj).mean())
        out.append({"template_type": t, "n": len(g), "accuracy": g.correct.mean(),
                    "accuracy_lenient": g.correct_lenient.mean(),
                    "majority_answer": maj, "baseline": base,
                    "delta": g.correct.mean() - base,
                    "beats_baseline": bool(g.correct.mean() > base)})
    glob = df.gold.value_counts().idxmax()
    out.append({"template_type": "OVERALL", "n": len(df), "accuracy": df.correct.mean(),
                "accuracy_lenient": df.correct_lenient.mean(),
                "majority_answer": f"'{glob}' everywhere",
                "baseline": float((df.gold == glob).mean()),
                "delta": df.correct.mean() - float((df.gold == glob).mean()),
                "beats_baseline": True})
    ta = _type_aware_baseline(rows)
    out.append({"template_type": "OVERALL vs type-aware", "n": len(df),
                "accuracy": df.correct.mean(), "accuracy_lenient": df.correct_lenient.mean(),
                "majority_answer": "each type's own majority", "baseline": ta,
                "delta": df.correct.mean() - ta,
                "beats_baseline": bool(df.correct.mean() > ta)})
    by_type = pd.DataFrame(out).round(4)
    by_type.to_csv(ROOT / by_type_csv, index=False)

    # --- why: the direction of the failure, and the test that could not decide it -------
    meas = []
    cnt = df[df.template_type == "count"]
    num = cnt[cnt.gold.str.isdigit() & cnt.pred.str.isdigit()]
    meas += [("count: model under-counts", int((num.pred.astype(int) < num.gold.astype(int)).sum()), len(cnt)),
             ("count: model over-counts", int((num.pred.astype(int) > num.gold.astype(int)).sum()), len(cnt)),
             ("count: exact", int((cnt.pred == cnt.gold).sum()), len(cnt)),
             ("count: model answers 0", int((cnt.pred == "0").sum()), len(cnt)),
             ("count: reference answers 0", int((cnt.gold == "0").sum()), len(cnt))]
    ex = df[df.template_type == "exist"]
    meas += [("exist: model says yes", int((ex.pred == "yes").sum()), len(ex)),
             ("exist: reference says yes", int((ex.gold == "yes").sum()), len(ex)),
             ("explicit 'cannot see' answers", int(df.raw.isin(REFUSALS).sum()), len(df)),
             ("answers outside the answer vocabulary", int((~df.in_vocab).sum()), len(df))]
    table = [{"measure": m, "count": k, "of": n, "rate": round(k / n, 4),
              "ci_lo": np.nan, "ci_hi": np.nan} for m, k, n in meas]

    # Ego-front minus ego-behind accuracy, bootstrapped over FRAMES: the ~13 questions on
    # one frame share an image and are not independent (F-082).
    frames = sorted(df.frame.unique())
    byf = {f: g for f, g in df.groupby("frame")}
    rng = np.random.default_rng(20260922)
    diffs = []
    for _ in range(2000):
        pick = pd.concat([byf[frames[i]] for i in rng.integers(0, len(frames), len(frames))])
        f, b = pick[pick.ego_direction == "front"], pick[pick.ego_direction == "behind"]
        if len(f) and len(b):
            diffs.append(f.correct.mean() - b.correct.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = (df[df.ego_direction == "front"].correct.mean()
             - df[df.ego_direction == "behind"].correct.mean())
    table.append({"measure": "accuracy, ego-front minus ego-behind (UNDERPOWERED)",
                  "count": int((df.ego_direction == "front").sum()),
                  "of": int((df.ego_direction == "behind").sum()),
                  "rate": round(float(point), 4),
                  "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
    direction = pd.DataFrame(table)
    direction.to_csv(ROOT / direction_csv, index=False)
    return by_type, direction
