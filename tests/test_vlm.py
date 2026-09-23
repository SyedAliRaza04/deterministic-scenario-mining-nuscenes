r"""Stage E self-checks — output parsing and the checkpointed runner.

Two defects from `docs/nuscens.py` are guarded here, because both silently produced
numbers rather than errors:

  E2  `re.search(r"\{[\s\S]+?\}", raw)` is NON-GREEDY. It stops at the first `}`, so
      nested JSON yields a truncated fragment, json.loads throws, a bare except swallows
      it, and EVERY TAG DEFAULTS TO FALSE. A model answering perfectly scored as if it
      had said no to everything.
  E4  results accumulated in a Python list. On free Colab the session dies and the run
      is lost, so it is redone, so quota is spent twice.

Run:  ./venv/bin/python tests/test_vlm.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.vlm as V  # noqa: E402

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


@check("test_nested_json_survives_the_non_greedy_regression")
def _():
    """THE regression. The old regex truncated this to `{"a": {"x": true}` and threw."""
    raw = '{"a": {"x": true}, "b": {"y": false}}'
    d, r = V.extract_json(raw)
    assert d == {"a": {"x": True}, "b": {"y": False}}, d
    assert r == "ok"


@check("test_json_is_found_inside_a_fence_and_after_prose")
def _():
    d, _ = V.extract_json('Here you go:\n```json\n{"a": {"x": true}}\n```\nHope that helps!')
    assert d == {"a": {"x": True}}, d


@check("test_a_brace_inside_a_string_does_not_end_the_object")
def _():
    """Depth counting must ignore braces in strings, which a regex cannot do."""
    d, _ = V.extract_json('{"note": "a } inside text", "x": true}')
    assert d == {"note": "a } inside text", "x": True}, d


@check("test_a_trailing_comma_is_repaired_and_reported_as_such")
def _():
    d, r = V.extract_json('{"a": true,}')
    assert d == {"a": True}, d
    assert r == "ok_after_trailing_comma_fix", r


@check("test_each_failure_mode_reports_a_distinct_reason")
def _():
    """RQ2b needs the rate AND the kind. One catch-all reason answers neither."""
    reasons = {V.extract_json(x)[1] for x in
               ("", "no json here", '{"a": true')}
    assert reasons == {"empty_output", "no_json_object", "unbalanced_braces"}, reasons


@check("test_a_missing_tag_is_reported_missing_and_never_defaults_to_false")
def _():
    """The whole reason the original numbers were meaningless. Stage F must be able to
    tell 'the model said no' from 'the model did not answer'."""
    vals, probs = V.coerce_tags({"g": {"a": True}}, ["a", "b", "c"])
    assert vals == {"a": True}, vals
    assert "missing:b" in probs and "missing:c" in probs, probs
    assert False not in vals.values(), "a missing tag was silently answered False"


@check("test_the_spellings_models_actually_emit_are_accepted")
def _():
    """Scoring a correct answer wrong for its formatting measures the parser."""
    vals, _ = V.coerce_tags(
        {"g": {"a": "yes", "b": "NO", "c": 1, "d": 0, "e": True}},
        ["a", "b", "c", "d", "e"])
    assert vals == {"a": True, "b": False, "c": True, "d": False, "e": True}, vals


@check("test_an_unparseable_value_is_flagged_not_guessed")
def _():
    vals, probs = V.coerce_tags({"g": {"a": "maybe"}}, ["a"])
    assert "a" not in vals
    assert any(p.startswith("unparseable:a") for p in probs), probs


@check("test_grouped_answers_are_flattened_so_group_names_do_not_matter")
def _():
    """Prompts group tags by scope for readability; scoring must not depend on the
    group keys, or renaming a group would silently break every result."""
    vals, _ = V.coerce_tags(
        {"ego_motion": {"is_turn_left": True},
         "visible_in_camera": {"cyclist": False}}, ["is_turn_left", "cyclist"])
    assert vals == {"is_turn_left": True, "cyclist": False}, vals


# --- the checkpointed runner ------------------------------------------------------------

class _Stub:
    """Backend that answers, then dies at `die_at` - a Colab session ending."""

    model_name = "stub"

    def __init__(self, die_at: int | None = None):
        self.calls, self.die_at = 0, die_at

    def generate(self, images, prompt, max_new_tokens=700, text=None):
        self.calls += 1
        if self.die_at is not None and self.calls > self.die_at:
            raise RuntimeError("session died")
        return '{"g": {"a": true, "b": false}}'


@check("test_run_batch_resumes_without_duplicating_rows")
def _():
    """PLAN E4's stated verification: kill the runtime mid-run, restart, confirm it
    resumes and does not duplicate."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        toks = [f"t{i}" for i in range(10)]
        kw = dict(prompt_version="v1_structured", out_jsonl=out,
                  images_for=lambda t: [], prompt_text="P", tags=["a", "b"],
                  progress=False)
        V.run_batch(_Stub(die_at=4), toks, **kw)          # dies partway
        first = [json.loads(l) for l in out.read_text().splitlines()]
        s2 = V.run_batch(_Stub(), toks, **kw)             # restart
        rows = [json.loads(l) for l in out.read_text().splitlines()]

    assert len(first) == 10, "every attempted frame must be recorded, failures included"
    assert s2["n_skipped"] == 4, f"only the 4 the model actually answered are done: {s2}"
    assert s2["n_run"] == 6, f"the 6 the crash ate must be retried: {s2}"
    # The retried frames append, so the 6 error rows are superseded by 6 good ones.
    good = [r for r in rows if not r["backend_error"]]
    assert len({r["sample_token"] for r in good}) == 10, \
        f"after resume every token must have a real answer, got {len(good)}"


@check("test_a_backend_exception_records_a_row_and_does_not_end_the_run")
def _():
    """One frame OOMing must not throw away the other 1,799."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        st = V.run_batch(_Stub(die_at=2), [f"t{i}" for i in range(5)],
                         prompt_version="v1", out_jsonl=out, images_for=lambda t: [],
                         prompt_text="P", tags=["a"], progress=False)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5, len(rows)
    assert st["n_backend_error"] == 3, st
    assert st["n_parse_fail"] == 0, "a backend crash is not a parse failure; RQ2b " \
                                    "counts model output, not infrastructure"
    assert "RuntimeError" in rows[-1]["parse_reason"], rows[-1]["parse_reason"]


@check("test_a_crashed_frame_is_retried_not_counted_as_done")
def _():
    """Regression 2026-09-06. A row written because the BACKEND died is not a result.

    Counting it as complete meant a restart skipped exactly the frames the crash ate -
    the failure mode the checkpointing exists to prevent, reintroduced by the
    checkpointing itself."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        out.write_text(
            json.dumps({"sample_token": "t0", "backend_error": False}) + "\n" +
            json.dumps({"sample_token": "t1", "backend_error": True}) + "\n")
        assert V.completed_tokens(out) == {"t0"}, V.completed_tokens(out)


@check("test_a_torn_final_line_is_redone_not_treated_as_complete")
def _():
    """A killed session leaves a half-written line. Trusting it loses a frame silently."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        out.write_text('{"sample_token": "t0"}\n{"sample_token": "t1", "raw": "unte')
        assert V.completed_tokens(out) == {"t0"}


@check("test_every_row_carries_the_provenance_stage_f_needs")
def _():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        V.run_batch(_Stub(), ["t0"], prompt_version="v1_structured", out_jsonl=out,
                    images_for=lambda t: [], prompt_text="P", tags=["a", "b"],
                    progress=False)
        row = json.loads(out.read_text().splitlines()[0])
    for k in ("sample_token", "prompt_version", "model", "raw", "parse_reason",
              "values", "problems", "n_answered", "latency_s", "cpu_s", "backend_error"):
        assert k in row, f"row is missing {k}"
    assert row["raw"], "the raw output must be kept so Stage F can re-score without a GPU"


@check("test_rows_record_cpu_time_as_well_as_wall_clock")
def _():
    """Regression 2026-09-06 (F-056). Wall clock counts time the machine spent ASLEEP.

    The E1 pilot reported a median 3,657 s per frame - 61 minutes - which looked like a
    catastrophic performance problem and was macOS Maintenance Sleep on a 60-minute
    cycle. Real generation is 27-35 s. Recording CPU time alongside makes the artefact
    detectable: a large wall/cpu ratio means the process was not running, not that the
    model was slow.
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        V.run_batch(_Stub(), ["t0"], prompt_version="v1", out_jsonl=out,
                    images_for=lambda t: [], prompt_text="P", tags=["a"], progress=False)
        row = json.loads(out.read_text().splitlines()[0])
    assert "cpu_s" in row and isinstance(row["cpu_s"], (int, float)), row
    assert row["cpu_s"] >= 0


@check("test_no_backend_is_still_a_stub")
def _():
    """Regression 2026-09-08. The Colab bundle was built BEFORE TransformersBackend was
    written, so it packaged `raise NotImplementedError`. The first T4 session died on it,
    after a 400 MB Drive upload and a GPU allocation.

    R29: after changing logic, re-check every artifact that logic produced. Here the guard
    is cheaper - refuse to have a stub at all once the bundle step exists.
    """
    src = (Path(__file__).resolve().parent.parent / "src" / "vlm.py").read_text()
    for cls in ("MLXBackend", "TransformersBackend"):
        body = src.split(f"class {cls}:", 1)[1].split("\nclass ", 1)[0]
        assert "raise NotImplementedError" not in body, (
            f"{cls} is a stub; build_colab_bundle would ship it and it would fail "
            "on Colab after the upload")


@check("test_the_shipped_bundle_matches_the_current_source")
def _():
    """A bundle is a build artifact and goes stale silently (R29).

    It sits in gitignored outputs/, so nothing else would notice that the .tar.gz on
    Drive predates the fix the user is waiting on. Skipped when no bundle exists.
    """
    import hashlib
    import json as _json

    root = Path(__file__).resolve().parent.parent
    mani = root / "outputs" / "colab_bundle_manifest.json"
    if not (root / "outputs" / "colab_bundle.tar.gz").exists() or not mani.exists():
        print("     (skip: no bundle built)")
        return
    recorded = _json.loads(mani.read_text())["sha256_12"]
    for name, want in recorded.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()[:12]
        assert got == want, (
            f"{name} changed since the bundle was built ({want} -> {got}); "
            "re-run src.data.build_colab_bundle() and re-upload")


@check("test_both_notebooks_refuse_to_run_the_wrong_subset")
def _():
    """Regression 2026-09-08. The Colab notebook fell back to the full 1,800-frame list
    when the token file was absent, printed a warning, and ran for ten hours longer than
    intended before anyone noticed.

    A warning nobody reads is not a guard. Both notebooks must ASSERT the execution
    subset, and neither may silently substitute a different one - this is the same class
    as the bundle shipping a stub (F-058): an artifact that looked fine and was wrong.
    """
    import json as _json
    root = Path(__file__).resolve().parent.parent
    for name in ("stage_e_colab", "stage_e_kaggle"):
        p = root / "notebooks" / f"{name}.ipynb"
        if not p.exists():
            continue
        text = "".join("".join(c["source"]) for c in _json.loads(p.read_text())["cells"])
        assert "assert _vlm.exists()" in text, f"{name} has no hard subset guard"
        assert "== 628" in text, f"{name} does not assert the subset size"
        assert "WARNING" not in text, (
            f"{name} still has a warn-and-continue path; the ten-hour failure was exactly "
            "that")


class _ScriptedBackend:
    """Returns a scripted sequence of raw replies, so the runner's failure handling can
    be exercised without a model. `model_name` is the provenance column run_batch reads."""

    model_name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, images, prompt, max_new_tokens=700, text=None):
        r = self.replies[self.calls]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


_GOOD = '{"a": true}'
_BANG = "!" * 40          # the degenerate all-'!' output, measured 2026-09-15


@check("test_run_batch_aborts_after_consecutive_failures_instead_of_burning_the_run")
def _():
    """Measured: Qwen2.5-VL emits all-'!' when the vision-token count runs too high, at a
    normal latency and with no exception. At 5 images that is EVERY frame, so without
    this the run spends its whole budget producing nothing. F-060's lesson."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        be = _ScriptedBackend([_BANG] * 10)
        try:
            V.run_batch(be, [f"t{i}" for i in range(10)], "v6_temporal", out,
                        images_for=lambda t: [], prompt_text="p", tags=["a"],
                        max_consecutive_failures=5, progress=False)
            raise AssertionError("run_batch did not abort on an all-failing run")
        except RuntimeError as e:
            assert "consecutive" in str(e), e
        assert be.calls == 5, f"aborted after {be.calls} calls, expected 5"


@check("test_the_failing_rows_are_written_before_the_abort")
def _():
    """D-045: an unparseable reply IS an observation and RQ2b's rate is built from it. A
    guard that aborts and loses the evidence would be worse than no guard, and the run
    must still resume from what it already paid for."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        be = _ScriptedBackend([_BANG] * 10)
        try:
            V.run_batch(be, [f"t{i}" for i in range(10)], "v6_temporal", out,
                        images_for=lambda t: [], prompt_text="p", tags=["a"],
                        max_consecutive_failures=5, progress=False)
        except RuntimeError:
            pass
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(rows) == 5, f"{len(rows)} rows survived the abort, expected 5"
        assert all(r["raw"].startswith("!") for r in rows)
        assert V.completed_tokens(out) == {f"t{i}" for i in range(5)}, "resume would redo paid work"


@check("test_scattered_failures_do_not_trip_the_guard")
def _():
    """The measured parse-failure rate is 0.03% over 3,516 calls (F-066), and E8's single
    truncation sat in an otherwise clean run. A guard that fired on isolated failures
    would abort a healthy run -- the false-positive direction R27 warns about."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        replies = [_GOOD, _BANG, _GOOD, _BANG, _GOOD, _BANG, _GOOD, _BANG, _GOOD, _GOOD]
        be = _ScriptedBackend(replies)
        stats = V.run_batch(be, [f"t{i}" for i in range(10)], "v1_structured", out,
                            images_for=lambda t: [], prompt_text="p", tags=["a"],
                            max_consecutive_failures=5, progress=False)
        assert stats["n_run"] == 10, stats
        assert stats["n_parse_fail"] == 4, stats
        assert be.calls == 10


@check("test_a_backend_error_streak_does_NOT_abort_the_run")
def _():
    """The guard counts PARSE failures only, and this is the regression that proves it.

    A first version counted backend errors too and broke
    `test_run_batch_resumes_without_duplicating_rows`, which deliberately simulates a
    session dying partway so every remaining frame writes an error row. That contract is
    required by F-055/D-045: the rows must exist for the restart to know what to retry.
    And the waste the guard exists to prevent does not occur here -- a dead backend
    raises immediately, so a full run of error rows costs seconds, not quota."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        be = _ScriptedBackend([RuntimeError("CUDA error")] * 10)
        st = V.run_batch(be, [f"t{i}" for i in range(10)], "v1_structured", out,
                         images_for=lambda t: [], prompt_text="p", tags=["a"],
                         max_consecutive_failures=3, progress=False)
        assert st["n_backend_error"] == 10, st
        assert st["n_parse_fail"] == 0, "a crash is not a parse failure (RQ2b)"
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(rows) == 10, f"{len(rows)} rows; every attempted frame must be recorded"
        assert V.completed_tokens(out) == set(), "error rows must not count as done (F-055)"


@check("test_the_backend_is_not_hard_wired_to_one_model_family")
def _():
    """E10 regression. TransformersBackend hard-coded Qwen2_5_VLForConditionalGeneration,
    so loading Qwen3-VL-8B for the model-sensitivity slice would have failed on the first
    cell of a metered Kaggle session -- our limitation, not the hardware's, and it would
    have read as 'the alternative model does not run here' (which is the finding PLAN.md
    says to report if it is TRUE).

    The Auto class dispatches on the checkpoint config. The guard that matters is the
    other half: the DEFAULT model must still resolve to the class the five published arms
    were produced under, or those numbers quietly stop being comparable."""
    src = (Path(__file__).resolve().parent.parent / "src" / "vlm.py").read_text()
    assert "Qwen2_5_VLForConditionalGeneration.from_pretrained" not in src, \
        "the loader is hard-wired to one architecture; E10 cannot run"
    assert "AutoModelForImageTextToText" in src
    assert 'got == "Qwen2_5_VLForConditionalGeneration"' in src, \
        "nothing pins the default model to the class the published arms used"
    assert "torch_dtype=torch.float16" in src, "the T4 is Turing and has no bfloat16"


@check("test_the_kaggle_notebook_runs_f5_and_a_tag_arm_from_its_bundle")
def _():
    """F-091. Under CONDITION = 'f5_nuscenes_qa' the smoke cell built its OWN call from the
    tag prompt and raised KeyError once the model had loaded, so F5 could not have run. The
    preflight passed, because it reads the notebook's text and cannot execute it. This runs
    the notebook's cells, parameters as shipped, from the EXTRACTED bundle in a fresh
    interpreter with a fake model (R50). A tag arm is run as well, so the shared call
    configuration cannot fix F5 by breaking every other condition."""
    import subprocess
    import tarfile

    root = Path(__file__).resolve().parent.parent
    b = root / "outputs/colab_bundle.tar.gz"
    if not b.exists():
        return
    nb = json.loads((root / "notebooks/stage_e_kaggle.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    params, cells = code[2], code[3:]          # skip bundle discovery and the pip install
    # The default is the NEXT run owed, never the last one run (F-091). F5 has run, so the
    # default moved to the repaired raster arm, and F5 is exercised with explicit params.
    assert "CONDITION = 'v3_bev_r2'" in params, "the repaired BEV arm is the run owed"
    f5_params = "CONDITION = 'f5_nuscenes_qa'; MODEL = None; SLICE = None; LIMIT = None"
    qs = json.loads((root / "outputs/f5_questions.json").read_text())["questions"]
    first = json.loads((root / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"][:3]
    need = ({f"images/{q['sample_token']}.jpg" for q in qs}
            | {f"images/{t}.jpg" for t in first}
            # the repaired raster arm: ALL 628, because the notebook refuses to start on
            # a partial set before it slices to LIMIT, which is the behaviour being tested
            | {f"bev_r2/{t}.png" for t in json.loads(
                (root / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]})
    fake = (
        "import glob, json, os, pathlib, shutil, sys\n"
        "sys.path.insert(0, '.')\n"
        "import src.vlm as V\n"
        "class Fake:\n"
        "    model_name = 'fake'\n"
        "    def __init__(self, **kw): pass\n"
        "    def generate(self, images, prompt, max_new_tokens=700, text=None):\n"
        "        assert images and all(pathlib.Path(p).exists() for p in images), images\n"
        "        assert '{question}' not in prompt, 'the question was never filled in'\n"
        "        return '{\"answer\": \"yes\"}'\n"
        "V.TransformersBackend = Fake\n"
        "WORK = pathlib.Path('.').resolve(); OUT = WORK / 'working'; OUT.mkdir(exist_ok=True)\n")
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(b) as tar:
            tar.extractall(td, filter="data", members=[
                m for m in tar if m.name.startswith(("src/", "outputs/")) or m.name in need])
        # F5 is the run owed; the two repaired BEV arms are the ones queued behind it, and
        # a tag arm proves the shared call configuration did not break the finished ones.
        for label, p, n_rows in (
                ("F5", f5_params, len(qs)),
                ("v1_structured", "CONDITION = 'v1_structured'; MODEL = None; "
                 "SLICE = None; LIMIT = 3", 3),
                ("v3_bev_r2", params.replace("LIMIT     = None", "LIMIT     = 3"), 3),
                ("v3b_bev_symbolic_r2", "CONDITION = 'v3b_bev_symbolic_r2'; MODEL = None; "
                 "SLICE = None; LIMIT = 3", 3)):
            r = subprocess.run([sys.executable, "-c", "\n".join([fake, p, *cells])],
                               cwd=td, capture_output=True, text=True)
            assert r.returncode == 0, f"{label} fails in the notebook:\n{r.stderr[-800:]}"
            res = [x for x in (Path(td) / "working").glob("*.jsonl") if x.name != "_smoke.jsonl"]
            assert len(res) == 1, [x.name for x in res]
            assert sum(1 for _ in res[0].open()) == n_rows, f"{label}: wrong row count"
            if label == "F5":   # and the LAPTOP can score what comes back; the notebook
                import src.nuqa as Q   # does not score F5 in-session, so the repo is right here
                assert Q.score(results_stem=str(res[0]))["n_questions"] == len(qs)
            res[0].unlink()


@check("test_the_qwen3_bev_notebook_runs_its_pinned_call_from_the_bundle")
def _():
    """D-053 runs from its own notebook so stage_e_kaggle.ipynb is never edited for it. The
    copy is only safe if its cells are stage_e's and its pinned parameters actually execute
    from the extracted bundle (R50), writing the model-suffixed file E10's pairing reads."""
    import subprocess
    import tarfile

    root = Path(__file__).resolve().parent.parent
    b = root / "outputs/colab_bundle.tar.gz"
    if not b.exists():
        return
    code_of = lambda n: ["".join(c["source"]) for c in json.loads(
        (root / f"notebooks/{n}.ipynb").read_text())["cells"] if c["cell_type"] == "code"]
    q, e = code_of("qwen3_bev_kaggle"), code_of("stage_e_kaggle")
    assert q[:2] + q[3:] == e[:2] + e[3:], "only the parameter cell may differ from stage_e"
    params = q[2]
    for want in ("CONDITION = 'v3_bev_r2'", "MODEL     = 'Qwen/Qwen3-VL-8B-Instruct'",
                 "SLICE     = 150"):
        assert want in params, f"pinned parameter missing: {want}"
    tokens = json.loads((root / "outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
    fake = (
        "import json, pathlib, sys\n"
        "sys.path.insert(0, '.')\n"
        "import src.vlm as V\n"
        "class Fake:\n"
        "    def __init__(self, **kw): self.model_name = kw.get('model', 'fake')\n"
        "    def generate(self, images, prompt, max_new_tokens=700, text=None):\n"
        "        assert len(images) == 1 and '/bev_r2/' in str(images[0]), images\n"
        "        assert pathlib.Path(images[0]).exists(), images\n"
        "        return '{}'\n"
        "V.TransformersBackend = Fake\n"
        "WORK = pathlib.Path('.').resolve(); OUT = WORK / 'working'; OUT.mkdir(exist_ok=True)\n")
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(b) as tar:
            tar.extractall(td, filter="data", members=[
                m for m in tar if m.name.startswith(("src/", "outputs/"))
                or m.name in {f"bev_r2/{t}.png" for t in tokens}])
        p = params.replace("LIMIT     = None", "LIMIT     = 3")
        # the smoke asserts a parse, so the fake must answer in the tag shape
        fake = fake.replace("return '{}'", "return json.dumps({t: False for t in "
                            "json.loads(pathlib.Path('outputs/label_schema.json').read_text())"
                            "['tags']})")
        r = subprocess.run([sys.executable, "-c", "\n".join([fake, p, *q[3:]])],
                           cwd=td, capture_output=True, text=True)
        assert r.returncode == 0, f"the Qwen3 BEV notebook fails:\n{r.stderr[-800:]}"
        res = [x.name for x in (Path(td) / "working").glob("*.jsonl") if x.name != "_smoke.jsonl"]
        assert res == ["v3_bev_r2__Qwen3-VL-8B-Instruct.jsonl"], res


if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} VLM self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
