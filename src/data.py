"""nuScenes loading, path resolution, and split helpers.

Also Step B4: the CAM_FRONT image manifest and its verification.

B4 note (2026-09-01). nuScenes is mirrored on a PUBLIC S3 bucket, so the signed,
expiring Download-page URLs PLAN.md assumed are not needed at all:
    https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/
Three tarball families exist per trainval part. Measured with HEAD requests:
    v1.0-trainvalNN_blobs.tgz          ~29 GiB   everything, all sweeps
    v1.0-trainvalNN_blobs_camera.tgz   ~16 GiB   camera incl. sweeps
    v1.0-trainvalNN_keyframes.tgz      ~4 GiB    KEYFRAMES ONLY, all channels  <- this one
The 150 subset scenes spread across all ten parts (8 to 23 scenes each), so every part
must be streamed regardless; there is no subset of blobs that would do. A .tgz cannot be
seeked into, so bandwidth is the full ~40 GiB either way -- but only CAM_FRONT is written
to disk. We keep ALL 34,149 CAM_FRONT keyframes (~4.4 GB) rather than only the 6,025 the
subset needs (~765 MB): the bandwidth is identical, and it means no future widening of the
subset ever needs another download.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATAROOT = REPO_ROOT / "data" / "nuscenes"

# Camera we use throughout. nuScenes also has CAM_FRONT_LEFT/RIGHT and three rear
# cameras; the thesis uses CAM_FRONT as the "driver view" per the SystemX brief.
MAIN_CAMERA = "CAM_FRONT"


def load_env() -> None:
    """Load .env into os.environ (Mac). No-op on Colab, where Secrets are used.

    Step A1.
    """
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")


def load_nusc(version: str = "v1.0-mini", dataroot: Path | str = DATAROOT,
              verbose: bool = False) -> Any:
    """Return a NuScenes devkit handle.

    version: 'v1.0-mini' (Step B1) or 'v1.0-trainval' (Step B3, metadata-only —
    loads fine without images as long as no image file is opened).

    Costs ~46 s for trainval (JSON parse). Load once per session and reuse; this is
    the dominant cost of Stage C, not the arithmetic (PROCESS.md F-006).
    """
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)


def iter_samples(nusc: Any, scene_token: str | None = None) -> Iterator[dict]:
    """Yield keyframe `sample` records, optionally restricted to one scene."""
    for s in nusc.sample:
        if scene_token is None or s["scene_token"] == scene_token:
            yield s


def cam_token(nusc: Any, sample: dict, camera: str = MAIN_CAMERA) -> str:
    """sample_data token for the given camera of this keyframe."""
    raise NotImplementedError


def ego_pose_for(nusc: Any, sample: dict, camera: str = MAIN_CAMERA) -> dict:
    """ego_pose record (translation, rotation quaternion, timestamp in MICROseconds)."""
    raise NotImplementedError


def map_name_for_scene(nusc: Any, scene: dict) -> str:
    """Map location for a scene, e.g. 'singapore-onenorth'.

    Comes from nusc.get('log', scene['log_token'])['location']. Needed by Steps C3 and D1.
    """
    return nusc.get("log", scene["log_token"])["location"]


def scene_sample_tokens(nusc: Any, scene: dict) -> list[str]:
    """Keyframe sample tokens for a scene, in chronological order."""
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t)
        t = nusc.get("sample", t)["next"]
    return toks


# --- Step B4: image manifest -------------------------------------------------------

S3_BASE = ("https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com"
           "/public/v1.0")
TRAINVAL_PARTS = [f"v1.0-trainval{n:02d}_keyframes.tgz" for n in range(1, 11)]


def _load_meta(dataroot: Path | str = DATAROOT, version: str = "v1.0-trainval") -> dict:
    """Read the metadata JSONs directly. ~10 s, against ~58 s for the devkit.

    B4 only needs filenames and scene membership, so instantiating NuScenes would pay
    for indexes we never touch (D-021 / R20: push work onto the cheap path).
    """
    import json
    root = Path(dataroot) / version
    return {n: json.loads((root / f"{n}.json").read_text())
            for n in ("scene", "sample", "sample_data")}


def fetch_subset_manifest(subset_path: Path | str = "outputs/subset_tokens.json",
                          out_path: Path | str = "outputs/image_manifest.json",
                          dataroot: Path | str = DATAROOT,
                          camera: str = MAIN_CAMERA) -> dict:
    """Write the exact list of camera files Stage D/E need. Step B4.

    Emitted BEFORE any download so the transfer is auditable and resumable, and so the
    verification in `verify_subset_images` has something to check against (R30: the
    manifest must regenerate from src/, not from a shell history).

    Two lists, because they answer different questions:
      `required`  the 6,025 files for the 150 subset scenes -- Stage D/E cannot run without
                  every one of these.
      `bonus`     the remaining CAM_FRONT keyframes, which the same download yields free.
    """
    import json

    meta = _load_meta(dataroot)
    subset = json.loads(Path(subset_path).read_text())
    want_scene_names = set(subset["scenes"])

    name_to_token = {s["name"]: s["token"] for s in meta["scene"]}
    missing = want_scene_names - set(name_to_token)
    if missing:
        raise ValueError(f"subset names absent from trainval metadata: {sorted(missing)[:5]}")
    want_scene_tokens = {name_to_token[n] for n in want_scene_names}

    scene_of_sample = {s["token"]: s["scene_token"] for s in meta["sample"]}
    token_to_name = {t: n for n, t in name_to_token.items()}

    prefix = f"samples/{camera}/"
    required, bonus = {}, []
    for r in meta["sample_data"]:
        if not (r["is_key_frame"] and r["filename"].startswith(prefix)):
            continue
        sc = scene_of_sample.get(r["sample_token"])
        if sc in want_scene_tokens:
            required[r["sample_token"]] = {
                "filename": r["filename"],
                "scene_name": token_to_name[sc],
            }
        else:
            bonus.append(r["filename"])

    sub_tokens = set(subset["sample_tokens"])
    covered = sub_tokens & set(required)
    if covered != sub_tokens:
        raise ValueError(f"{len(sub_tokens - covered)} subset tokens have no {camera} keyframe")

    manifest = {
        "camera": camera,
        "source": f"{S3_BASE}/<part>  (public bucket, no signed URL needed)",
        "parts": TRAINVAL_PARTS,
        "n_required": len(required),
        "n_subset_tokens": len(sub_tokens),
        "n_scenes": len(want_scene_names),
        "n_bonus": len(bonus),
        "est_required_mb": round(len(required) * 130 / 1024, 1),
        "est_bonus_mb": round(len(bonus) * 130 / 1024, 1),
        "required": required,
    }
    Path(out_path).write_text(json.dumps(manifest, indent=1))
    return manifest


def verify_subset_images(manifest_path: Path | str = "outputs/image_manifest.json",
                         dataroot: Path | str = DATAROOT) -> dict:
    """Check the download against the manifest. Step B4 verification.

    Reports rather than raises, because a partial download is the normal mid-transfer
    state and the caller needs the count to decide whether to resume.
    """
    import json

    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(dataroot)
    missing, empty = [], []
    for tok, rec in manifest["required"].items():
        f = root / rec["filename"]
        if not f.exists():
            missing.append(rec["filename"])
        elif f.stat().st_size == 0:
            empty.append(rec["filename"])

    n = manifest["n_required"]
    scenes_missing = sorted({manifest["required"][t]["scene_name"]
                             for t in manifest["required"]
                             if not (root / manifest["required"][t]["filename"]).exists()})
    return {
        "n_required": n,
        "n_present": n - len(missing),
        "n_missing": len(missing),
        "n_empty": len(empty),
        "complete": not missing and not empty,
        "scenes_incomplete": scenes_missing[:20],
        "missing_examples": missing[:5],
        "empty_examples": empty[:5],
    }


# --- Step E9: temporal windows ------------------------------------------------------

# 4 keyframes of HISTORY at 2 Hz = a 2.0 s window, 5 frames including the labelled one.
# Five because the old query_multiframe claimed five frames while sending one
# (docs/nuscens.py:649), so five is what that experiment was always supposed to test.
# HISTORY, not a centred +/-2 window: a centred window is also 5 frames but shows the
# model 1 s of FUTURE, while every C1/C2 window feature is computed on a window ENDING at
# the keyframe (D-010). Scoring a model that saw t+1, t+2 against a label derived without
# them would credit it for information the label never used. Decided by the author, with
# the measurement in hand: 577 of 628 tokens get the full 5 frames, 51 fewer (scene
# starts), and all 1,861 distinct frames are already on disk.
TEMPORAL_HISTORY = 4


def sequence_for_tokens(tokens: list[str], dataroot: Path | str = DATAROOT,
                        history: int = TEMPORAL_HISTORY,
                        camera: str = MAIN_CAMERA) -> dict[str, list[str]]:
    """token -> the `history` keyframes before it plus itself, chronological.

    Step E9. The ordering comes from the same `prev`/`next` chain `scene_sample_tokens`
    walks, so there is one definition of "the next keyframe" in the repo rather than two
    (R4, the rule F-016 came from).

    Two deliberate choices, both provenance rather than convenience:

    * **Clamped at scene edges, never padded.** A token in the first two keyframes of a
      scene genuinely has no earlier frames; repeating its first frame would feed the
      model a fake stationary period and manufacture `is_stationary` evidence. The
      sequence is simply shorter, and the runner records how long it was.
    * **The labelled frame is LAST.** `PROMPT_V6_TEMPORAL` tells the model "the LAST frame
      is the moment being labelled", and every C1/C2 window feature is computed on a
      window ENDING at the keyframe. A sequence centred on the token would show the model
      the future and score it against a label derived from the past.

    Reads the metadata JSONs directly rather than instantiating the devkit: this is pure
    token chaining, and the devkit costs ~58 s for data already on disk as JSON.
    """
    meta = _load_meta(dataroot)
    prev_of = {s["token"]: s["prev"] for s in meta["sample"]}
    nxt_of = {s["token"]: s["next"] for s in meta["sample"]}
    missing = [t for t in tokens if t not in prev_of]
    if missing:
        raise ValueError(f"{len(missing)} tokens absent from trainval metadata: {missing[:3]}")

    out = {}
    for tok in tokens:
        back = []
        cur = tok
        for _ in range(history):
            cur = prev_of.get(cur) or ""
            if not cur:
                break
            back.append(cur)
        # chronological, ending at the labelled frame: [t-4, t-3, t-2, t-1, t]
        out[tok] = list(reversed(back)) + [tok]
    return out


@lru_cache(maxsize=1)
def _keyframe_image_index(dataroot: str = str(DATAROOT),
                          camera: str = MAIN_CAMERA) -> dict[str, str]:
    """sample_token -> camera filename, straight from the metadata JSON.

    The devkit would answer this too, at ~46-58 s per session for indexes we never touch.
    `_load_meta` reads the three JSONs in ~8 s and covers all 34,149 keyframes, so Stage G
    can iterate without paying the parse (R20, the same move D-021 made for C3).

    Filtering on the filename prefix rather than resolving calibrated_sensor -> sensor is
    deliberate: it needs one table instead of three, and `samples/<CHANNEL>/` is the layout
    Step B4 verified against the live archive (F-047).
    """
    meta = _load_meta(dataroot)
    prefix = f"samples/{camera}/"
    return {sd["sample_token"]: sd["filename"] for sd in meta["sample_data"]
            if sd["is_key_frame"] and sd["filename"].startswith(prefix)}


def image_for_sample_token(sample_token: str, camera: str = MAIN_CAMERA,
                           dataroot: Path | str = DATAROOT) -> Path:
    """Absolute path to a keyframe's camera JPEG, without loading the devkit.

    All 34,149 trainval CAM_FRONT keyframes are on disk (4.8 GB) since Step B4 completed.
    Every other sensor channel of trainval remains metadata only.
    """
    idx = _keyframe_image_index(str(dataroot), camera)
    fn = idx.get(sample_token)
    if fn is None:
        raise KeyError(f"no {camera} keyframe for sample_token {sample_token!r}")
    p = Path(dataroot) / fn
    if not p.exists():
        raise FileNotFoundError(f"{p} not downloaded -- see PLAN.md Step B4")
    return p


def image_path(nusc: Any, cam_data_token: str) -> Path:
    """Absolute path to a camera JPEG. Raises if the image was not downloaded (Step B4)."""
    p = Path(nusc.get_sample_data_path(cam_data_token))
    if not p.exists():
        raise FileNotFoundError(f"{p} not downloaded -- see PLAN.md Step B4")
    return p


def build_colab_bundle(out_path: Path | str = "outputs/colab_bundle.tar.gz",
                       manifest_path: Path | str = "outputs/image_manifest.json",
                       dataroot: Path | str = DATAROOT,
                       subset_only: bool = True,
                       include_results: bool = True,
                       temporal: bool = False) -> dict[str, Any]:
    """Package exactly what Colab needs, and nothing else. Step E4.

    The repo holds 13 GB. Colab needs under 1 GB of it: the 1,800 subset frames' camera
    images, their BEV rasters, the symbolic descriptions, the frozen schema and the token
    list. Uploading the whole tree to Drive would waste hours of the user's bandwidth and
    most of a free Drive quota (15 GB).

    `subset_only=True` takes the 1,800 EVALUATION tokens rather than all 6,025 keyframes
    of those scenes. The extra 4,225 exist for E9's temporal windows and Stage G; they are
    not needed until then, and shipping them triples the upload.
    """
    import hashlib
    import json

    from . import bev as B
    import tarfile

    manifest = json.loads(Path(manifest_path).read_text())
    sub = json.loads(Path("outputs/subset_tokens.json").read_text())
    tokens = sub["sample_tokens"]
    root, out_path = Path(dataroot), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # A backend that still raises NotImplementedError is a stub, and shipping one wastes
    # a Drive upload and a Colab session before failing. Measured the hard way on
    # 2026-09-08: the bundle was built BEFORE TransformersBackend was written, so the
    # first T4 run died on `raise NotImplementedError` (R29 - after changing logic,
    # re-check every artifact that logic produced).
    vlm_src = Path("src/vlm.py").read_text()
    for cls in ("MLXBackend", "TransformersBackend"):
        body = vlm_src.split(f"class {cls}:", 1)[1].split("\nclass ", 1)[0]
        if "raise NotImplementedError" in body:
            raise RuntimeError(
                f"src/vlm.py:{cls} is still a stub; packaging it would fail on Colab "
                "after the upload. Implement it first.")

    n_img = n_bev = n_bev_r2 = 0
    src_hashes = {}
    with tarfile.open(out_path, "w:gz") as tar:
        for name in ("outputs/subset_tokens.json", "outputs/vlm_subset_tokens.json",
                     "outputs/label_schema.json", "outputs/bev_symbolic.jsonl",
                     # The repaired text arm's descriptions (F-064b). Named from src.bev,
                     # never spelled again here, so the packer and the renderer cannot
                     # disagree about which file the session reads (R4).
                     B.BEV_R2_JSONL,
                     # F5: the 1,471 questions and their gold answers. Small (≈300 KB) and
                     # carried unconditionally, so a Kaggle session can run F5 without a
                     # second upload.
                     "outputs/f5_questions.json",
                     "src/prompts.py", "src/vlm.py",
                     "src/nuqa.py"):
            tar.add(name, arcname=name)
            src_hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()[:12]

        # Completed rows travel WITH the data, so a run on a different host resumes
        # instead of repeating work already paid for. `completed_tokens` skips them and
        # `backend_error` rows are retried, so carrying them is always safe.
        n_resume = 0
        if include_results:
            for f in sorted(Path("outputs/results").glob("*.jsonl")) \
                    if Path("outputs/results").exists() else []:
                tar.add(f, arcname=f"results/{f.name}")
                n_resume += sum(1 for _ in f.open())
        wanted = set(tokens) if subset_only else set(manifest["required"])
        # Step E9. The temporal arm reads a 5-frame sequence per token, and those
        # neighbours are NOT in the evaluation subset -- only 34.2% of a subset token's
        # +/-2 neighbours are themselves subset tokens (F-047), which is exactly why D-040
        # widened B4 to whole scenes. Without this the Kaggle run finds one image where
        # the prompt promises five, which is the original query_multiframe defect rebuilt.
        n_seq_extra = 0
        if temporal:
            vlm_tokens = json.loads(
                Path("outputs/vlm_subset_tokens.json").read_text())["sample_tokens"]
            seqs = sequence_for_tokens(vlm_tokens, dataroot=dataroot)
            seq_tokens = {t for s in seqs.values() for t in s}
            n_seq_extra = len(seq_tokens - wanted)
            wanted |= seq_tokens
            # The sequences themselves travel with the bundle: recomputing them on Kaggle
            # would need the trainval metadata, which is 440 MB and not in the bundle.
            seq_path = Path("outputs/temporal_sequences.json")
            seq_path.write_text(json.dumps(
                {"history": TEMPORAL_HISTORY,
                 "n_tokens": len(seqs),
                 "len_histogram": {str(k): sum(1 for v in seqs.values() if len(v) == k)
                                   for k in sorted({len(v) for v in seqs.values()})},
                 "sequences": seqs}, indent=1))
            tar.add(seq_path, arcname="outputs/temporal_sequences.json")
            src_hashes["outputs/temporal_sequences.json"] = hashlib.sha256(
                seq_path.read_bytes()).hexdigest()[:12]
        for tok, rec in manifest["required"].items():
            if tok not in wanted:
                continue
            p = root / rec["filename"]
            if p.exists():
                tar.add(p, arcname=f"images/{tok}.jpg")
                n_img += 1
            bev = Path("data/bev") / f"{tok}.png"
            if bev.exists():
                tar.add(bev, arcname=f"bev/{tok}.png")
                n_bev += 1
            # The REPAIRED rasters travel beside the published ones rather than replacing
            # them, so one bundle serves both the finished arms (for resume) and the
            # re-run, and the session cannot silently read the old picture (F-063c).
            bev_r2 = Path(B.BEV_R2_DIR) / f"{tok}.png"
            if bev_r2.exists():
                tar.add(bev_r2, arcname=f"bev_r2/{tok}.png")
                n_bev_r2 += 1

    size_mb = out_path.stat().st_size / 1048576
    Path("outputs/colab_bundle_manifest.json").write_text(json.dumps({
        "built": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "sha256_12": src_hashes, "n_camera": n_img, "n_bev": n_bev,
        "n_bev_r2": n_bev_r2, "size_mb": round(size_mb, 1)}, indent=1))
    return {"path": str(out_path), "n_camera": n_img, "n_bev": n_bev,
            "n_bev_r2": n_bev_r2,
            "size_mb": round(size_mb, 1), "n_tokens": len(tokens),
            "n_temporal_extra": n_seq_extra,
            "n_resume_rows": n_resume, "src_sha256_12": src_hashes}
