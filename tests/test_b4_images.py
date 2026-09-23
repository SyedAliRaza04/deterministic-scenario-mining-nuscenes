"""Step B4 self-checks — the image manifest and its verification.

The transfer itself (curl | tar over the public S3 mirror) was verified live on
2026-09-01: a range request into v1.0-trainval01_keyframes.tgz piped through
`tar -xz --include=` produced valid 1600x900 JPEGs. What is tested here is the
logic that decides WHICH files are required and whether the download is complete,
because that is where a silent error would survive.

Run:  ./venv/bin/python tests/test_b4_images.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.data as D  # noqa: E402

MANIFEST = Path("outputs/image_manifest.json")
SUBSET = Path("outputs/subset_tokens.json")

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


def _manifest() -> dict:
    assert MANIFEST.exists(), "run src.data.fetch_subset_manifest() first"
    return json.loads(MANIFEST.read_text())


@check("test_manifest_covers_every_subset_token")
def _():
    """The whole point of B4: no subset token may lack an image."""
    m, sub = _manifest(), json.loads(SUBSET.read_text())
    required = set(m["required"])
    missing = set(sub["sample_tokens"]) - required
    assert not missing, f"{len(missing)} subset tokens absent from the manifest"


@check("test_manifest_is_whole_scenes_not_just_subset_tokens")
def _():
    """D-040: we fetch whole scenes so E9 and Stage G do not need another download.

    1,800 subset tokens live in 150 scenes totalling 6,025 keyframes. A manifest that
    had quietly reverted to token-only would show n_required == n_subset_tokens.
    """
    m = _manifest()
    assert m["n_required"] > m["n_subset_tokens"], (
        f"manifest has {m['n_required']} files for {m['n_subset_tokens']} tokens — "
        "this is the pre-D-040 token-only scope")
    assert m["n_required"] == 6025, m["n_required"]
    assert m["n_scenes"] == 150, m["n_scenes"]


@check("test_every_required_file_is_a_cam_front_keyframe_path")
def _():
    m = _manifest()
    bad = [r["filename"] for r in m["required"].values()
           if not r["filename"].startswith("samples/CAM_FRONT/")
           or not r["filename"].endswith(".jpg")]
    assert not bad, bad[:3]


@check("test_required_filenames_are_unique")
def _():
    """Two tokens mapping to one file would silently under-count the download."""
    m = _manifest()
    names = [r["filename"] for r in m["required"].values()]
    assert len(names) == len(set(names)), f"{len(names) - len(set(names))} duplicates"


@check("test_temporal_neighbours_are_present_for_every_subset_token")
def _():
    """RQ3b / E9 needs +/-2 keyframes around each subset token.

    This is the measurement that drove D-040: only 34.2% of those neighbours are
    themselves subset tokens, so a token-only download would have broken E9.
    """
    import pandas as pd
    m = _manifest()
    kin = pd.read_parquet("outputs/ego_kinematics_trainval.parquet")[
        ["sample_token", "scene_name", "keyframe_index"]]
    have = set(m["required"])
    sub = set(json.loads(SUBSET.read_text())["sample_tokens"])
    idx = {(r.scene_name, r.keyframe_index): r.sample_token for r in kin.itertuples()}
    pos = {r.sample_token: (r.scene_name, r.keyframe_index) for r in kin.itertuples()}

    absent = 0
    for t in sub:
        scene, k = pos[t]
        for d in (-2, -1, 1, 2):
            nb = idx.get((scene, k + d))
            if nb is not None and nb not in have:
                absent += 1
    assert absent == 0, f"{absent} temporal neighbours missing from the manifest"


@check("test_verify_reports_incomplete_when_files_are_absent")
def _():
    """A verifier that cannot detect a missing file is worse than none."""
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "m.json"
        fake.write_text(json.dumps({
            "n_required": 2,
            "required": {
                "tokA": {"filename": "samples/CAM_FRONT/absent_a.jpg", "scene_name": "s1"},
                "tokB": {"filename": "samples/CAM_FRONT/absent_b.jpg", "scene_name": "s1"},
            }}))
        out = D.verify_subset_images(fake, dataroot=d)
    assert out["complete"] is False
    assert out["n_missing"] == 2, out
    assert out["n_present"] == 0, out


@check("test_verify_counts_a_zero_byte_file_as_bad_not_present")
def _():
    """A truncated transfer leaves 0-byte files; treating them as present would
    make `complete` True on a broken download."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "samples" / "CAM_FRONT"
        root.mkdir(parents=True)
        (root / "good.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "trunc.jpg").write_bytes(b"")
        fake = Path(d) / "m.json"
        fake.write_text(json.dumps({
            "n_required": 2,
            "required": {
                "a": {"filename": "samples/CAM_FRONT/good.jpg", "scene_name": "s1"},
                "b": {"filename": "samples/CAM_FRONT/trunc.jpg", "scene_name": "s1"},
            }}))
        out = D.verify_subset_images(fake, dataroot=d)
    assert out["n_empty"] == 1, out
    assert out["complete"] is False, out


@check("test_manifest_names_the_keyframes_tarballs_not_the_full_blobs")
def _():
    """The measured reason B4 is affordable: keyframes ~4 GiB vs blobs ~29 GiB per part.

    A regression to *_blobs.tgz would multiply the transfer by seven and nobody would
    notice until it had been running for a day.
    """
    m = _manifest()
    assert len(m["parts"]) == 10, m["parts"]
    for p in m["parts"]:
        assert p.endswith("_keyframes.tgz"), p


@check("test_download_script_targets_the_public_mirror_and_filters_cam_front")
def _():
    """Guards the two things that make the script cheap and correct."""
    sh = Path("scripts/fetch_b4_images.sh").read_text()
    assert "motional-nuscenes.s3" in sh, "not pointed at the public mirror"
    assert "--include='samples/CAM_FRONT/*'" in sh, "extraction filter lost"
    assert "_keyframes.tgz" in sh, "must fetch keyframes, not full blobs"
    assert "-C -" in sh, "byte-level resume lost; a 40 GiB transfer needs it"


@check("test_script_never_uses_curl_internal_retry_with_resume")
def _():
    """Regression, 2026-09-04: curl --retry TRUNCATES the -o file and restarts at 0.

    `-C -` is evaluated once per curl invocation, not per internal retry, so the two
    do not compose. A "Recv failure: Connection reset by peer" at 3.32 GiB of part 05
    silently discarded all 3.32 GiB and began again — visible only as an apparent
    throughput collapse from 4.2 to 1.2 MiB/s.

    The fix is a shell retry loop: a fresh curl per attempt re-reads the on-disk size,
    so `-C -` genuinely resumes.
    """
    sh = Path("scripts/fetch_b4_images.sh").read_text()
    # Comments are allowed to name the trap; only executable lines may not use it.
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert "--retry" not in code, (
        "curl --retry is back; it truncates the partial file and restarts from zero")
    assert "-C -" in sh, "byte-level resume lost"
    assert "for attempt in" in sh, "shell retry loop lost; a dropped connection now aborts the part"
    assert "--speed-time" in sh, "stall guard lost; an open-but-dead socket can hang the run"


# --- Step E9: the temporal windows themselves ---------------------------------------
# These live here rather than in a new file because the E9 neighbour check above already
# does, and because the question is the same one: does the frame this arm needs exist,
# and is it the right frame.

_SEQ: dict = {}


def _seqs() -> dict:
    """Sequences for the 628-frame execution subset, plus the `next` pointer table.

    Cached: `_load_meta` parses ~10 s of JSON, and six checks would otherwise pay it six
    times over.
    """
    if not _SEQ:
        toks = json.loads(Path("outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
        meta = D._load_meta()
        _SEQ["tokens"] = toks
        _SEQ["seqs"] = D.sequence_for_tokens(toks)
        _SEQ["next"] = {s["token"]: s["next"] for s in meta["sample"]}
    return _SEQ


@check("test_every_sequence_ends_at_the_frame_being_labelled")
def _():
    """PROMPT_V6_TEMPORAL says "the LAST frame is the moment being labelled", and every
    C1/C2 window feature is computed on a window ENDING at the keyframe (D-010). If the
    labelled frame moved off the end, the model would be scored against a label derived
    from a different instant."""
    s = _seqs()
    bad = [t for t, v in s["seqs"].items() if v[-1] != t]
    assert not bad, f"{len(bad)} sequences do not end at their own token, e.g. {bad[:2]}"


@check("test_sequences_are_strictly_causal_and_contain_no_future_frame")
def _():
    """THE design regression. A centred +/-2 window is also 5 frames, and was rejected
    deliberately: it shows the model 1 s of future while the ground truth was derived
    from a window ending at t, so a gain could mean the model saw information the label
    never used. Walking `next` forward from the token must reach nothing in its own
    sequence."""
    s = _seqs()
    nxt, offenders = s["next"], []
    for t, v in s["seqs"].items():
        cur, future = t, set()
        for _ in range(4):
            cur = nxt.get(cur) or ""
            if not cur:
                break
            future.add(cur)
        if future & set(v):
            offenders.append(t)
    assert not offenders, f"{len(offenders)} sequences contain a FUTURE frame, e.g. {offenders[:2]}"


@check("test_sequence_frames_are_chronologically_chained")
def _():
    """Ordering must be real rather than incidental: each frame's `next` is the one after
    it. A sequence in the wrong order shows the model a scene running backwards."""
    s = _seqs()
    nxt = s["next"]
    broken = [t for t, v in s["seqs"].items()
              for a, b in zip(v, v[1:]) if nxt.get(a) != b]
    assert not broken, f"{len(broken)} sequences are not chained, e.g. {broken[:2]}"
    dupes = [t for t, v in s["seqs"].items() if len(set(v)) != len(v)]
    assert not dupes, f"{len(dupes)} sequences repeat a frame, e.g. {dupes[:2]}"


@check("test_scene_starts_are_clamped_not_padded")
def _():
    """A token in the first keyframes of a scene genuinely has no history. Repeating its
    first frame to reach five would feed the model a fake stationary period and
    manufacture `is_stationary` evidence, so the sequence is simply shorter (and the
    runner records how short). 12 of the 628 are scene starts with no history at all."""
    s = _seqs()
    lens = sorted({len(v) for v in s["seqs"].values()})
    assert min(lens) >= 1, lens
    assert max(lens) == D.TEMPORAL_HISTORY + 1, lens
    short = {t: v for t, v in s["seqs"].items() if len(v) < D.TEMPORAL_HISTORY + 1}
    assert short, "expected some scene-start tokens with a truncated window"
    for t, v in short.items():
        assert len(set(v)) == len(v), f"{t} was padded with a repeated frame"


@check("test_a_full_window_is_five_frames_not_three")
def _():
    """Regression, 2026-09-15: the first builder walked `prev` only TEMPORAL_HALF_WIDTH=2
    steps and produced THREE frames while PROMPT_V6_TEMPORAL promised five — the original
    query_multiframe defect (docs/nuscens.py:649) rebuilt one level down, and invisible
    because 3 images still generate a plausible answer."""
    s = _seqs()
    assert D.TEMPORAL_HISTORY == 4, D.TEMPORAL_HISTORY
    full = [v for v in s["seqs"].values() if len(v) == 5]
    assert len(full) > 500, f"only {len(full)} of {len(s['seqs'])} windows reach 5 frames"


@check("test_every_sequence_frame_was_downloaded")
def _():
    """The bundle can only carry frames B4 actually fetched. D-040 widened the download to
    whole scenes precisely so these exist; a regression to token-only scope breaks E9."""
    s = _seqs()
    required = set(_manifest()["required"])
    need = {f for v in s["seqs"].values() for f in v}
    assert not (need - required), f"{len(need - required)} sequence frames absent from the manifest"


if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} B4 self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
