"""Step F5 self-checks — the external NuScenes-QA benchmark run.

Three traps, all measured rather than imagined:

  1. THE WRONG FILE. `data/bdd100k/samples.json` and
     `data/nuscenes_qa/NuScenes_val_questions.json` are different datasets for different
     stages, and the first draft of this step was written against the wrong name. Reading
     BDD100K here produces nonsense rather than an error.
  2. A BARE-WORD ANSWER KILLS THE RUN. `extract_json("truck")` returns no_json_object, and
     five such replies in a row trip run_batch's abort guard.
  3. NORMALISATION THAT CHEATS. Stripping a trailing "s" unconditionally maps wrong answers
     onto right ones; it is only safe when the singular is itself in the vocabulary.

Run directly:  ./venv/bin/python tests/test_f5_nuqa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.nuqa as Q  # noqa: E402
import src.vlm as V   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "outputs/f5_questions.json"


def _spec():
    return json.loads(SPEC.read_text())


def test_the_question_list_comes_from_the_nuscenes_qa_file_not_bdd100k():
    """Trap 1. The two datasets' filenames are easy to confuse and one fails silently."""
    assert Q.QA_JSON.name == "NuScenes_val_questions.json", Q.QA_JSON
    assert "bdd" not in str(Q.QA_JSON).lower()
    assert _spec()["source"] == "NuScenes_val_questions.json"


def test_the_slice_is_the_val_split_frames_only():
    """Only the official val frames are comparable to published numbers; the other 517 of
    our 628 are in those benchmarks' TRAIN files (D-049/F-070)."""
    s = _spec()
    assert s["n_frames"] == 111, s["n_frames"]
    assert s["n_questions"] == 1471, s["n_questions"]


def test_question_ids_are_unique_and_survive_duplicate_question_text():
    """19 of the 1,471 are exact duplicate strings within one frame. Keyed by TEXT they
    would silently collapse; keyed by index they do not."""
    qs = _spec()["questions"]
    assert len({q["qid"] for q in qs}) == len(qs), "question ids collide"
    texts = [(q["sample_token"], q["question"]) for q in qs]
    assert len(set(texts)) < len(texts), "no duplicate text: this guard would be vacuous"
    for q in qs[:50]:
        assert q["qid"].split(":")[0] == q["sample_token"], "frame not recoverable from qid"


def test_the_prompt_demands_json_because_a_bare_word_aborts_the_run():
    """Trap 2, with the failure reproduced rather than asserted from memory."""
    assert V.extract_json("truck")[1] == "no_json_object"
    assert V.extract_json("The answer is truck.")[1] == "no_json_object"
    assert V.extract_json('{"answer": "truck"}')[1] == "ok"
    p = Q.ANSWER_PROMPT.format(question="What is it?")
    assert '"answer"' in p and "JSON" in p
    assert "one word" in p.lower()


def test_normalisation_never_turns_a_wrong_answer_into_a_right_one():
    """Trap 3. De-pluralising is allowed only INTO the vocabulary."""
    vocab = {"truck", "car", "bus", "moving", "parked", "status"}
    assert Q.normalise("  Truck. ", vocab) == "truck"
    assert Q.normalise("a truck", vocab) == "truck"
    assert Q.normalise("trucks", vocab) == "truck"      # singular is in vocab -> safe
    assert Q.normalise("status", vocab) == "status"     # must NOT become "statu"
    assert Q.normalise("bicycles", vocab) == "bicycles", "stripped into a non-vocabulary word"


def test_run_batch_takes_a_per_item_prompt_and_does_not_fall_back_to_the_registry():
    """F5's version is not a registered TAG prompt, so the fallback would raise AFTER the
    model loaded. Caught by rehearsing the runner, not by reading it."""
    import inspect
    src = inspect.getsource(V.run_batch)
    assert "prompt_for" in V.run_batch.__code__.co_varnames
    assert "if prompt_text is None and prompt_for is None:" in src, \
        "the registry fallback is not guarded; F5 would raise on a loaded model"


def test_the_bundle_ships_what_f5_needs():
    """The questions and the module must travel, or the Kaggle session cannot run F5."""
    src = (ROOT / "src" / "data.py").read_text()
    assert '"outputs/f5_questions.json"' in src
    assert '"src/nuqa.py"' in src


# --- F-096: the run's results ---------------------------------------------------------
# R31: above the registry line. Skips cleanly on a clone without the results file.

_RUN = ROOT / "outputs/results/f5_nuscenes_qa.txt"


def test_a_type_aware_baseline_can_never_lose_to_a_global_one():
    """Structural. Answering each type's majority is at least as good as answering one
    global majority everywhere, because the global answer is one of the options each type
    could have picked. If this ever fails, the baseline code is wrong."""
    rows = [{"template_type": t, "gold": g} for t, g in
            [("exist", "yes"), ("exist", "yes"), ("exist", "no"),
             ("count", "3"), ("count", "3"), ("count", "2")]]
    ta = Q._type_aware_baseline(rows)
    glob = sum(r["gold"] == "yes" for r in rows) / len(rows)
    assert ta >= glob, (ta, glob)
    assert abs(ta - 4 / 6) < 1e-9, ta


def test_the_overall_win_is_the_aggregation_trap():
    """F-096 FROZEN. Overall accuracy beats the global majority baseline while losing to
    or tying the majority in every template type under the benchmark's strict metric, and
    losing to the type-aware baseline overall. If a future change makes the overall row
    read as a clean win, the trap D-049 warned about has been reopened."""
    if not _RUN.exists():
        return
    import pandas as pd

    bt = pd.read_csv(ROOT / "outputs/f5_by_type.csv").set_index("template_type")
    per_type = bt.drop(index=["OVERALL", "OVERALL vs type-aware"])
    assert not per_type.beats_baseline.any(), per_type.beats_baseline.to_dict()
    assert bt.loc["OVERALL", "accuracy"] > bt.loc["OVERALL", "baseline"]
    assert bt.loc["OVERALL vs type-aware", "accuracy"] < bt.loc["OVERALL vs type-aware", "baseline"]


def test_lenient_scoring_maps_only_listed_synonyms_and_never_rescues_the_overall():
    """The lenient score bounds formatting; it must not become a second, generous model.
    Only the listed synonyms may map, every target must be a real answer, and the
    measured gain is small enough that the overall verdict against the type-aware baseline
    does not change. It DOES flip the object type (strict 0.221, lenient 0.252 against
    0.227), and that is reported rather than hidden."""
    import json as _j

    vocab = {q["answer"].casefold() for q in
             _j.loads((ROOT / "outputs/f5_questions.json").read_text())["questions"]}
    assert set(Q.LENIENT_SYNONYMS.values()) <= vocab, "a synonym maps to a non-answer"
    assert "walking" not in Q.LENIENT_SYNONYMS and "van" not in Q.LENIENT_SYNONYMS, (
        "ambiguous mappings turn a sensitivity check into a guess")
    if not _RUN.exists():
        return
    import pandas as pd

    bt = pd.read_csv(ROOT / "outputs/f5_by_type.csv").set_index("template_type")
    ov = bt.loc["OVERALL vs type-aware"]
    assert 0 <= ov.accuracy_lenient - ov.accuracy < 0.02, ov
    assert ov.accuracy_lenient < ov.baseline, "lenient scoring rescued the overall verdict"


def test_the_direction_test_is_reported_as_underpowered():
    """The ego-front against ego-behind comparison is the one test that could separate
    field of view from a negative answering habit, and it cannot: 116 and 123 questions,
    clustered by frame. Its interval must stay wide and straddle zero, and the artifact must
    say UNDERPOWERED, so nobody quotes the point estimate as evidence either way."""
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/f5_answer_direction.csv")
    row = d[d.measure.str.contains("ego-front")].iloc[0]
    assert "UNDERPOWERED" in row.measure
    assert row.ci_lo < 0 < row.ci_hi, (row.ci_lo, row.ci_hi)
    assert row.ci_hi - row.ci_lo > 0.2, "if the interval tightened, re-read the finding"


def test_the_count_failure_is_directional():
    """Random miscounting would err both ways about equally. 269 under against 14 over is
    a direction, and a direction is what both candidate explanations predict."""
    if not _RUN.exists():
        return
    import pandas as pd

    d = pd.read_csv(ROOT / "outputs/f5_answer_direction.csv").set_index("measure")
    under = d.loc["count: model under-counts", "count"]
    over = d.loc["count: model over-counts", "count"]
    assert under > 10 * over, (under, over)


tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in tests:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {fn.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} F5 self-checks passed")
    sys.exit(1 if failed else 0)
