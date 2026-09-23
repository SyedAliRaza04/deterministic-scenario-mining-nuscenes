"""Storyboard segmentation — Step G1, serving RQ5.

A storyboard is a scene cut into a LINEAR sequence of panels, each carrying a thumbnail, a
time range and a description. The slides ask for exactly this ("Storyboard -> Sequencage
video -> Analyse de scenes -> Description de scenario"), and D-006 kept it in scope because
temporal segmentation is part of the assigned task rather than an optional extra.

THE DESIGN PROBLEM, AND WHY IT IS NOT OBVIOUS. D-014 models ego manoeuvres as TWO ORTHOGONAL
AXES — steering and speed — precisely because a car turns and brakes at the same time, and
collapsing them would mark a VLM wrong for reporting the axis we discarded. A strip is
linear; two overlapping axes are not. Measured over all 850 scenes before choosing:

    both axes simultaneously active   1,336 s = 8.02% of all driving, in 319 scenes (37.5%)
    of the 387 scenes with a turn     48.6% of their turning seconds overlap a speed event
    76.0% of the 495 lateral events   overlap a longitudinal one

so linearising is the common case, not an edge case, and the two axes cannot be separated by
ignoring rare overlaps. Three alternatives were measured and rejected:

    lateral boundaries only     463 of 850 scenes have NO lateral event -> 54.6% of the
                                dataset collapses to a single degenerate panel
    longitudinal only           silently deletes all 495 turn boundaries, the rarest and
                                most visually distinctive manoeuvres
    one panel per event         panels nest and reorder wherever the axes overlap, and the
                                59 event-free scenes get no panel at all

WHAT THIS MODULE DOES INSTEAD. A panel is a maximal run of a constant
(lateral_label, longitudinal_label, lane_change) triple over consecutive keyframes. That is
the union of both axes' boundaries, but read off the per-keyframe labels C2 already
resolved — so this module writes NO new detector (D-013: C2 and G1 share one) and inherits
the "stationary outranks accelerating/decelerating" precedence that `maneuver_labels`
applies to the 345 same-axis event overlaps. Verified: replaying the event table through
that precedence reproduces gt_all on 34,149/34,149 keyframes, both axes.

Panels are keyframe-indexed, which makes the structural invariants exact integers rather
than float comparisons: chronological, non-overlapping, contiguous, and covering every
keyframe of the scene exactly once (R3).

Build:  ./venv/bin/python -c "import src.storyboard as S; S.build_storyboard_table()"
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# --- parameters (one place only, each with the measurement that chose it) -------------

# Shortest panel the strip will show.
#
# What this floor is NOT for. The obvious justification — "the raw union of both axes leaves
# 193 panels containing no keyframe, so they have no thumbnail" — is true of a segmentation
# cut at event TIMES, and was the reason this parameter was introduced. It does not apply
# here: panels are maximal runs of KEYFRAMES, so every panel holds at least one keyframe by
# construction. Measured at floor 0.0 the minimum is exactly 1, never 0. The original
# rationale was carried over from the measurement phase and did not survive the
# implementation; it is recorded here because a parameter justified by a defect it does not
# prevent is worse than an unjustified one.
#
# What it IS for: a one-keyframe panel shows a single 0.5 s instant, which is a sliver in a
# strip meant to be read at a glance. Removing those is a READABILITY choice, and it trades
# against FIDELITY — every absorbed panel takes its label with it. Measured over all 850
# scenes by `panel_sensitivity` (the R12 evidence for this number):
#
#     floor   panels   median/scene   max/scene   min kf/panel   label pairs lost
#     0.0 s   4,115         5            14             1              0.00%
#     0.5 s   3,351         4            10             2             10.60%
#     1.0 s   3,051         4             8             3             15.74%
#     1.5 s   2,705         3             7             4             22.18%
#     2.0 s   2,380         3             6             5             29.13%
#     3.0 s   1,841         2             4             7             41.09%
#
# 0.5 s is chosen: it is the smallest floor that removes single-instant panels (min keyframes
# 1 -> 2, so every panel has a pair to pick a thumbnail from), at the lowest fidelity cost of
# any non-zero floor. It is also exactly groundtruth.MIN_EVENT_S — C2's shortest admissible
# event — so the rule reads "a panel must be able to hold an event", reusing a measured
# constant rather than introducing a fresh one (R1).
#
# This parameter BINDS: it is not inert. Moving it 0.5 -> 1.0 costs a further 5.1 points of
# label fidelity for two fewer panels on the widest scene, which is why the sweep is reported
# rather than a single tuned value (R12, F-010's discipline).
MIN_PANEL_S = 0.5

# Tags too common to be worth stating in a description.
#
# F-038 measured 13 of 36 tags at a majority baseline >= 90%: `on_drivable_area` is true on
# 99.74% of keyframes and `has_vehicle` on 91.19%, so a panel that announces them says
# nothing. The gate is ONE-SIDED ON PURPOSE. The obvious symmetric version would also drop
# rare tags, and that is backwards: a tag true on 2% of frames is mentioned on 2% of panels,
# and when it IS mentioned it is the most informative thing the panel can say. F-038's
# low-prevalence caveat is about F1 being noisy when SCORING a rare tag, which is a different
# question from whether to state it in prose.
MAX_DESCRIBED_PREVALENCE_PCT = 90.0

# How many tags a caption may state, per slot. A storyboard panel is read at a glance beside
# its thumbnail; the first build stated every true tag and produced 15-clause run-on
# sentences that no reader would finish (found by READING the output, R8 — no structural
# check would have caught it, since the panels were correct).
#
# Which tags survive the cut is decided by PREVALENCE, rarest first: `on_ped_crossing`
# (2.89%) tells a reader far more than `intersection_ahead` (77.43%), and the same asymmetry
# that makes a common tag worthless to state makes a rare one worth stating first. The full
# lists stay in the `tags_throughout` / `tags_partial` columns, and the caption says how many
# it left out — a summary, never a silent truncation.
MAX_ACTORS_THROUGHOUT = 3
MAX_ACTORS_PARTIAL = 2

# Tags whose truth is already carried by the panel's own axis labels, so repeating them in
# the description would say the same thing twice. Generated from the vocabulary constants
# (R4) rather than listed by hand — F-016 is the defect that rule exists to prevent, where a
# hand-written tag name drifted from the event vocabulary and left `is_turning_left` False on
# all 5,500 turning frames.
def _axis_tags() -> set[str]:
    from .groundtruth import LATERAL_LABELS, LONGITUDINAL_LABELS
    return {f"is_{v}" for v in LATERAL_LABELS + LONGITUDINAL_LABELS}


@lru_cache(maxsize=1)
def _frames() -> Any:
    """One keyframe-indexed table: manoeuvre labels + every scoreable tag + edge_guard.

    Cached because `scene_panels` is called per scene and the three parquets cost ~1 s to
    join; the devkit is never loaded (R20 — these quantities are all already cached).
    """
    import pandas as pd

    man = pd.read_parquet(OUT / "maneuvers_trainval.parquet")
    gt = pd.read_parquet(OUT / "gt_all.parquet")

    # gt_all carries no t_rel_s, so time comes from the manoeuvre table (T3). gt_all already
    # carries edge_guard, so the kinematics table is not needed here at all — joining it
    # would only collide on that column.
    tag_cols = [c for c in gt.columns if c not in man.columns or c == "sample_token"]
    df = man.merge(gt[tag_cols], on="sample_token", validate="1:1")
    return df.sort_values(["scene_name", "keyframe_index"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _describable_tags() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(setting_tags, actor_tags) — the scoreable tags a description may state.

    Split by the schema's own `family` field rather than by a hand-kept list, so a schema
    change propagates instead of drifting (R4).
    """
    schema = json.loads((OUT / "label_schema.json").read_text())
    axis = _axis_tags()
    setting, actors = [], []
    for tag, spec in schema["tags"].items():
        if spec["role"] != "scored" or tag in axis:
            continue
        if float(spec.get("prevalence_pct", 0.0)) > MAX_DESCRIBED_PREVALENCE_PCT:
            continue
        if spec["family"] == "map_context":
            setting.append(tag)
        elif spec["family"] in ("visible_objects", "interaction"):
            actors.append(tag)
    return tuple(setting), tuple(actors)


@lru_cache(maxsize=1)
def _prevalence() -> dict:
    """Per-tag prevalence from the frozen schema — the caption's ranking key."""
    schema = json.loads((OUT / "label_schema.json").read_text())
    return {t: float(v.get("prevalence_pct", 100.0)) for t, v in schema["tags"].items()}


def _rarest(tags: list[str], k: int) -> list[str]:
    """The k most informative tags: rarest first. Ties broken by name for reproducibility."""
    prev = _prevalence()
    return sorted(tags, key=lambda t: (prev.get(t, 100.0), t))[:k]


def _phrase(tag: str) -> str:
    """The perceptual phrase for a tag, from the ONE place they are written.

    `prompts.TAG_QUESTIONS` is hand-written but its keys are test-asserted to equal the
    schema's scoreable tags exactly, so R4 is satisfied by enforcement rather than by
    generation — the same resolution F-054 reached when the schema's `derivation` text
    turned out to be implementation jargon a reader could not parse.
    """
    from .prompts import TAG_QUESTIONS
    return TAG_QUESTIONS[tag]


def _strip_subject(phrase: str) -> str:
    for lead in ("the car is ", "the car ", "there is ", "at least one "):
        if phrase.startswith(lead):
            return phrase[len(lead):]
    return phrase


def _describe(lat: str, lon: str, lane_change: bool,
              throughout: list[str], partial: list[str],
              setting_tags: tuple, actor_tags: tuple) -> str:
    """A deterministic caption for one panel. D-001: auditable, never hand-written prose.

    Structure: what the EGO is doing (always, from the two axis labels), WHERE it is (the
    single most specific setting that holds for the whole panel), and WHO else is there (the
    rarest few actors). Tags true on every keyframe are stated flatly; tags true on only some
    are hedged and never majority-voted into a flat claim — 77.5% of multi-keyframe panels
    carry at least one flickering tag, and asserting one as though it held throughout would
    manufacture a frame-level claim the ground truth does not support.

    Phrases come verbatim from `prompts.TAG_QUESTIONS`, whose keys are test-asserted to equal
    the schema's scoreable tags, so the vocabulary is written in exactly one place (R4).
    """
    ego = _phrase(f"is_{lat}")
    speed = _strip_subject(_phrase(f"is_{lon}"))
    # em-dash, not a comma: "driving straight ahead rather than turning, stopped, or
    # creeping..." garden-paths into a list of three things. Found by reading the output.
    out = f"{ego[0].upper()}{ego[1:]}, {speed}"
    if lane_change:
        out += f", {_strip_subject(_phrase('lane_change'))}"
    out += "."

    setting = [t for t in throughout if t in setting_tags]
    if setting:
        out += " " + _phrase(_rarest(setting, 1)[0]).capitalize() + "."

    actors = [t for t in throughout if t in actor_tags]
    shown = _rarest(actors, MAX_ACTORS_THROUGHOUT)
    if shown:
        out += " Also: " + "; ".join(_phrase(t) for t in shown)
        extra = len(actors) - len(shown)
        out += f" (+{extra} more)." if extra else "."

    part = _rarest([t for t in partial if t in actor_tags], MAX_ACTORS_PARTIAL)
    if part:
        out += " Briefly: " + "; ".join(_phrase(t) for t in part)
        extra = len([t for t in partial if t in actor_tags]) - len(part)
        out += f" (+{extra} more)." if extra else "."
    return out


def _runs(sub: Any) -> list[tuple[int, int]]:
    """Maximal runs of a constant (lateral, longitudinal, lane_change) triple.

    Returns inclusive [start, end] positions into `sub`.
    """
    key = list(zip(sub.lateral_label, sub.longitudinal_label, sub.lane_change))
    bounds, start = [], 0
    for i in range(1, len(key)):
        if key[i] != key[i - 1]:
            bounds.append((start, i - 1))
            start = i
    bounds.append((start, len(key) - 1))
    return bounds


def _merge_short(runs: list[tuple[int, int]], t: Any, min_panel_s: float) -> list[tuple[int, int]]:
    """Absorb any panel shorter than `min_panel_s` into its LONGER neighbour.

    Shortest-first so the decision never depends on iteration order; ties go to the EARLIER
    neighbour, which keeps the rule deterministic and therefore reproducible. Merging is
    exactly C2's own `_merge_same_sign` idea one level up (MERGE_GAP_S = 0.5 s).
    """
    runs = list(runs)
    while len(runs) > 1:
        durs = [t[b] - t[a] for a, b in runs]
        short = [i for i, d in enumerate(durs) if d < min_panel_s]
        if not short:
            break
        i = min(short, key=lambda j: (durs[j], j))
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            before, after = durs[i - 1], durs[i + 1]
            j = i - 1 if before >= after else i + 1      # ties -> earlier
        lo, hi = min(i, j), max(i, j)
        runs[lo] = (runs[lo][0], runs[hi][1])
        del runs[hi]
    return runs


def _collapse(runs: list[tuple[int, int]], sub: Any) -> list[tuple[int, int]]:
    """Lossless: fuse adjacent panels whose labels are now identical.

    Merging can leave two neighbours reading the same, which would print as two consecutive
    panels a reader cannot tell apart. Measured: run AFTER the merge, not before (3,315
    panels with 2 residual duplicates, against 3,388 and 36 the other way round).
    """
    def label(i):
        return (sub.lateral_label.iloc[i], sub.longitudinal_label.iloc[i], sub.lane_change.iloc[i])

    out = [runs[0]]
    for a, b in runs[1:]:
        if label(a) == label(out[-1][0]):
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def scene_panels(scene_name: str, min_panel_s: float = MIN_PANEL_S) -> Any:
    """The storyboard panels for one scene, in time order. Step G1.

    One row per panel. Panels are contiguous in keyframe index and together cover the whole
    scene exactly once — the invariant `tests/test_g1_storyboard.py` asserts.
    """
    import pandas as pd

    df = _frames()
    sub = df[df.scene_name == scene_name].reset_index(drop=True)
    if sub.empty:
        raise KeyError(f"unknown scene {scene_name!r}")

    t = sub.t_rel_s.to_numpy()
    raw = _runs(sub)
    runs = _collapse(_merge_short(raw, t, min_panel_s), sub)
    setting_tags, actor_tags = _describable_tags()

    rows = []
    for idx, (a, b) in enumerate(runs):
        panel = sub.iloc[a:b + 1]
        n = len(panel)
        throughout, partial = [], []
        for tag in setting_tags + actor_tags:
            col = panel[tag]
            if col.all():
                throughout.append(tag)
            elif col.any():
                partial.append(tag)

        # Thumbnail: the keyframe corroborating the most of what the sentence claims, so the
        # picture and the text agree. A blind midpoint often shows none of them — 77.5% of
        # multi-keyframe panels carry a tag that is true on only part of the panel.
        stated = throughout + partial
        score = panel[stated].sum(axis=1).to_numpy() if stated else [0] * n
        mid = (n - 1) / 2.0
        best = max(range(n), key=lambda k: (score[k], -abs(k - mid)))

        rows.append({
            "panel_index": idx,
            "scene_name": scene_name,
            "location": panel.location.iloc[0],
            "t_start_s": float(panel.t_rel_s.iloc[0]),
            "t_end_s": float(panel.t_rel_s.iloc[-1]),
            "kf_start": int(panel.keyframe_index.iloc[0]),
            "kf_end": int(panel.keyframe_index.iloc[-1]),
            "n_keyframes": n,
            "lateral_label": panel.lateral_label.iloc[0],
            "longitudinal_label": panel.longitudinal_label.iloc[0],
            "lane_change": bool(panel.lane_change.iloc[0]),
            "thumbnail_token": panel.sample_token.iloc[best],
            "thumbnail_kf": int(panel.keyframe_index.iloc[best]),
            "description": _describe(panel.lateral_label.iloc[0],
                                     panel.longitudinal_label.iloc[0],
                                     bool(panel.lane_change.iloc[0]),
                                     throughout, partial,
                                     setting_tags, actor_tags),
            "tags_throughout": ",".join(throughout),
            "tags_partial": ",".join(partial),
            # provenance: how many raw label-runs this panel absorbed (1 = untouched), and
            # whether EVERY keyframe in it sits in a window C1 flagged as unreliable —
            # F-009's frozen-pose scenes produce a phantom event exactly there. Flagged,
            # never special-cased by scene name (R11).
            "n_runs_absorbed": sum(1 for ra, _ in raw if a <= ra <= b),
            "phantom_guard": bool(panel.edge_guard.all()),
        })

    out = pd.DataFrame(rows)
    out["merged"] = out.n_runs_absorbed > 1
    return out


def build_storyboard_table(out_path: str = "outputs/storyboard_panels.parquet") -> Any:
    """Panels for all 850 scenes. Step G1."""
    import pandas as pd

    df = _frames()
    parts = [scene_panels(s) for s in sorted(df.scene_name.unique())]
    table = pd.concat(parts, ignore_index=True)
    table.to_parquet(ROOT / out_path, index=False)
    return table


def panel_sensitivity(out_path: str = "outputs/g1_panel_sensitivity.json") -> dict:
    """Sweep MIN_PANEL_S and tabulate the effect. R12 — the evidence for the chosen floor.

    Reports the two axes that actually trade against each other: how readable the strip is
    (panels per scene, panels with too few keyframes to show) versus how much it costs
    (manoeuvre events that no longer get a panel naming them).
    """
    import numpy as np
    import pandas as pd

    df = _frames()
    scenes = sorted(df.scene_name.unique())
    rows = {}
    for floor in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        n_panels, min_kf, kinds_kept, kinds_total = 0, 10 ** 9, 0, 0
        per_scene = []
        for s in scenes:
            sub = df[df.scene_name == s].reset_index(drop=True)
            t = sub.t_rel_s.to_numpy()
            runs = _collapse(_merge_short(_runs(sub), t, floor), sub) if floor else _runs(sub)
            per_scene.append(len(runs))
            n_panels += len(runs)
            min_kf = min(min_kf, min(b - a + 1 for a, b in runs))
            raw = {(sub.lateral_label.iloc[a], sub.longitudinal_label.iloc[a])
                   for a, b in _runs(sub)}
            kept = {(sub.lateral_label.iloc[a], sub.longitudinal_label.iloc[a])
                    for a, b in runs}
            kinds_total += len(raw)
            kinds_kept += len(raw & kept)
        rows[f"{floor:.1f}"] = {
            "min_panel_s": floor,
            "n_panels": int(n_panels),
            "panels_per_scene_median": float(np.median(per_scene)),
            "panels_per_scene_max": int(max(per_scene)),
            "min_keyframes_in_any_panel": int(min_kf),
            "label_pairs_lost_pct": round(100 * (1 - kinds_kept / kinds_total), 2),
        }
    (ROOT / out_path).write_text(json.dumps(rows, indent=2))
    return rows


# --- Step G2: scoring the segmentation ------------------------------------------------
#
# PLAN G2 asks for boundary precision/recall against kinematic ground truth, with a
# tolerance window, versus a fixed-interval baseline.
#
# WHAT G2 CANNOT DO, AND WHY IT IS SAID FIRST (R21). G1's panels are maximal runs of the
# per-keyframe labels; the C2 event table is the same labels expressed as intervals. Scoring
# one against the other measures D-050's LINEARISATION, not a segmenter: it comes out at
# F1 0.895 (+/-1 kf) with the two boundary sets identical on only 7 of the first 60 scenes,
# and that number says how much the merge rules and the axis union move a boundary --
# nothing about whether the boundary is right. A detector scored against its own source is
# the F-024 mistake, and it is reported here as what it is.
#
# WHAT G2 DOES MEASURE. How much of the kinematic event structure a segmenter that knows
# NOTHING about the signal recovers. That is the null every future segmenter -- the VLM arm
# included -- has to beat, and it is the half of G2 that needs no GPU.
#
# THE BASELINE GETS THE ORACLE'S BOUNDARY COUNT. A segmenter emitting one boundary per scene
# would score high precision on nothing, so the fixed-interval baseline is given exactly as
# many boundaries as the gold standard has, evenly spaced. That is the STRONGER null, the
# same reasoning as `majority_baseline`'s tie-break, and it makes precision and recall equal
# by construction so a single number is honest. The period-based variant (a cut every N
# seconds, no oracle knowledge) is reported beside it because that is what a real fixed
# interval segmenter would do.

# Tolerance for calling a predicted boundary a hit, in KEYFRAMES. Swept in
# `build_g2_table`, because it is a researcher-chosen parameter and R12 requires its
# sensitivity to be reported rather than asserted. One keyframe is 0.5 s.
BOUNDARY_TOL_KF = 1

# Period of the knowledge-free fixed-interval baseline, in seconds. 2.0 s is the median
# panel duration G1 produces over all 850 scenes, so the baseline is handed the right
# GRANULARITY and fails only on placement -- the weaker assumption to make about it.
FIXED_PERIOD_S = 2.0

# Draws for the random-placement null. 200 is enough for a mean to two decimals here; the
# quantity is a mean over 850 scenes, not a tail.
N_RANDOM_DRAWS = 200
RANDOM_SEED = 20260920


def boundary_prf(pred: set[int], gold: set[int], tol: int = BOUNDARY_TOL_KF) -> tuple[int, int, int]:
    """tp / fp / fn for one scene, matching within `tol` keyframes.

    Matching is by EXISTENCE, not one-to-one: a predicted boundary counts as a hit if any
    gold boundary is within tolerance, and a gold boundary is recovered if any prediction
    is. At tol=1 and a dense prediction set the two can disagree, which is why tp is
    counted separately for each side rather than shared -- precision and recall then mean
    what their names say.
    """
    hit_p = {p for p in pred if any(abs(p - g) <= tol for g in gold)}
    hit_g = {g for g in gold if any(abs(p - g) <= tol for p in pred)}
    return len(hit_p), len(pred) - len(hit_p), len(gold) - len(hit_g)


def kinematic_boundaries(scene_name: str, events: Any = None, frames: Any = None) -> set[int]:
    """The gold standard: C2 event starts and ends, snapped to the nearest keyframe.

    Interior boundaries only. A scene's first and last keyframe are boundaries for every
    segmenter including a trivial one, so counting them would inflate every score equally
    and flatter the baselines most.
    """
    import numpy as np
    import pandas as pd

    ev = events if events is not None else pd.read_parquet(
        OUT / "maneuver_events_trainval.parquet")
    fr = frames if frames is not None else _frames()
    sub = fr[fr.scene_name == scene_name].sort_values("keyframe_index")
    t, idx = sub.t_rel_s.to_numpy(), sub.keyframe_index.to_numpy()
    b = set()
    for r in ev[ev.scene_name == scene_name].itertuples():
        for s in (r.start_s, r.end_s):
            b.add(int(idx[np.argmin(np.abs(t - s))]))
    return {x for x in b if 0 < x < idx.max()}


def panel_boundaries(scene_name: str, panels: Any = None) -> set[int]:
    """G1's boundaries: the first keyframe of every panel but the first."""
    import pandas as pd

    p = panels if panels is not None else pd.read_parquet(OUT / "storyboard_panels.parquet")
    sub = p[p.scene_name == scene_name].sort_values("panel_index")
    return {int(k) for k in sub.kf_start.to_numpy()[1:]}


def fixed_interval_boundaries(n_kf: int, k: int) -> set[int]:
    """`k` evenly spaced interior boundaries over `n_kf` keyframes. The matched-count null."""
    if k <= 0 or n_kf < 2:
        return set()
    step = n_kf / (k + 1)
    return {min(n_kf - 1, max(1, int(round(step * (i + 1))))) for i in range(k)}


def period_boundaries(times: Any, period_s: float = FIXED_PERIOD_S) -> set[int]:
    """A cut every `period_s` seconds, snapped to keyframes. The knowledge-free null."""
    import numpy as np

    t = np.asarray(times, dtype=float)
    if len(t) < 2:
        return set()
    cuts = np.arange(t[0] + period_s, t[-1], period_s)
    return {int(np.argmin(np.abs(t - c))) for c in cuts if 0 < np.argmin(np.abs(t - c)) < len(t) - 1}


def random_boundaries(n_kf: int, k: int, rng: Any) -> set[int]:
    """`k` interior boundaries placed uniformly at random. The chance floor.

    Without this, "the fixed-interval baseline recovers 40% of boundaries" is unreadable:
    a scene with 8 gold boundaries in 40 keyframes gives a lot away to chance at +/-1.
    """
    if k <= 0 or n_kf < 3:
        return set()
    return set(rng.choice(range(1, n_kf - 1), size=min(k, n_kf - 2), replace=False).tolist())


def build_g2_table(out_csv: str = "outputs/g2_segmentation.csv",
                   tolerances: tuple[int, ...] = (0, 1, 2, 3)) -> Any:
    """Boundary P/R/F1 for every segmenter against the kinematic events, over ALL 850 scenes.

    Micro-averaged over scenes -- tp, fp and fn are pooled before the ratio -- because a
    per-scene mean lets a 3-panel scene outweigh a 12-panel one, and the quantity asked for
    is "of all the boundaries in the dataset, how many were found".
    """
    import numpy as np
    import pandas as pd

    fr = _frames()
    ev = pd.read_parquet(OUT / "maneuver_events_trainval.parquet")
    pan = pd.read_parquet(OUT / "storyboard_panels.parquet")
    scenes = sorted(fr.scene_name.unique())
    rng = np.random.default_rng(RANDOM_SEED)

    per_scene = {}
    for s in scenes:
        sub = fr[fr.scene_name == s].sort_values("keyframe_index")
        n = len(sub)
        gold = kinematic_boundaries(s, ev, fr)
        k = len(gold)
        per_scene[s] = {
            "n_kf": n, "gold": gold,
            "g1_panels": panel_boundaries(s, pan),
            "fixed_matched_k": fixed_interval_boundaries(n, k),
            "fixed_period_2s": period_boundaries(sub.t_rel_s.to_numpy()),
            "random_matched_k": [random_boundaries(n, k, rng) for _ in range(N_RANDOM_DRAWS)],
        }

    rows = []
    for tol in tolerances:
        for method in ("g1_panels", "fixed_matched_k", "fixed_period_2s", "random_matched_k"):
            if method == "random_matched_k":
                f1s, precs, recs = [], [], []
                for d in range(N_RANDOM_DRAWS):
                    tp = fp = fn = 0
                    for s in scenes:
                        a, b, c = boundary_prf(per_scene[s][method][d], per_scene[s]["gold"], tol)
                        tp, fp, fn = tp + a, fp + b, fn + c
                    p_ = tp / (tp + fp) if tp + fp else 0.0
                    r_ = tp / (tp + fn) if tp + fn else 0.0
                    precs.append(p_); recs.append(r_)
                    f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
                rows.append({"tolerance_kf": tol, "method": method,
                             "precision": float(np.mean(precs)), "recall": float(np.mean(recs)),
                             "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                             "n_predicted": int(sum(len(per_scene[s][method][0]) for s in scenes)),
                             "n_gold": int(sum(len(per_scene[s]["gold"]) for s in scenes))})
                continue
            tp = fp = fn = 0
            for s in scenes:
                a, b, c = boundary_prf(per_scene[s][method], per_scene[s]["gold"], tol)
                tp, fp, fn = tp + a, fp + b, fn + c
            p_ = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            rows.append({"tolerance_kf": tol, "method": method, "precision": p_, "recall": r_,
                         "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, "f1_sd": 0.0,
                         "n_predicted": int(sum(len(per_scene[s][method]) for s in scenes)),
                         "n_gold": int(sum(len(per_scene[s]["gold"]) for s in scenes))})

    df = pd.DataFrame(rows)
    # THE NUMBER THAT MAKES THE REST READABLE. Gold boundaries are 0.099 per keyframe, so a
    # guess inside a +/-tol window hits with probability ~(2*tol + 1) * density before
    # collisions -- 0.099, 0.296, 0.493, 0.690 at tol 0..3. That closed form lands on the
    # measured random baseline (0.153, 0.382, 0.549, 0.665) from an independent direction
    # (R9), and it says plainly that a boundary F1 quoted at a loose tolerance is mostly
    # reporting the tolerance. Carried in the artifact so no reader has to re-derive it.
    density = float(sum(len(per_scene[s]["gold"]) for s in scenes)
                    / sum(per_scene[s]["n_kf"] for s in scenes))
    df["gold_density_per_kf"] = round(density, 4)
    df["chance_f1_closed_form"] = [min(1.0, (2 * t + 1) * density) for t in df.tolerance_kf]
    df = df.round(4)
    df.to_csv(ROOT / out_csv, index=False)
    return df


# --- Step G2b: the VLM segmentation arm (CPU half) ------------------------------------
#
# PLAN G2's verify criterion is "VLM segmentation vs fixed-interval baseline". The
# baselines and the chance floor are built above; this is everything the GPU session needs
# except the GPU.
#
# WHY THE QUESTION IS PAIRWISE AND NOT "SEGMENT THIS SCENE". The obvious design -- show the
# model K frames of a 20 s scene and ask which ones start a new manoeuvre -- is not
# answerable at the resolution the metric scores. E9 measured the ceiling at FIVE images
# (at 1280x720 the 4-bit path emits degenerate all-'!' output from four images up, F-069),
# and five frames spread over 40 keyframes places a boundary no better than +/-4 keyframes.
# Scored at +/-1 that fails for reasons that have nothing to do with the model, which is
# F-065's and F-070's shape: an experiment whose answer is decided by its own setup.
#
# So the unit of work is a KEYFRAME, not a scene: show the frame one keyframe BEFORE and
# one AFTER, and ask whether the vehicle's manoeuvre changed between them. Two images at
# full resolution is the `v4_both` configuration, already verified over 628 frames. The
# answer is a single boolean, so `max_new_tokens` drops from 700 to 32 and most of the
# per-call cost disappears with it.
#
# THE SCENES ARE A UNIFORM DRAW, DELIBERATELY. Filtering to scenes with at least three
# manoeuvre boundaries looks like sensible enrichment and is not: it lifts boundary density
# from 0.099 to 0.141 per keyframe, which lifts the chance floor from 0.30 to 0.42 at +/-1
# and flatters every segmenter scored against it. A uniform draw keeps the null identical
# to the 850-scene table (measured 0.0963 against 0.0986), so the two are directly
# comparable -- and the 4 scenes in 25 with NO boundary at all are informative, because
# they can only produce false positives.

# Scenes in the VLM arm. 25 uniform scenes is 957 calls: ~3.2 h at 12 s each, ~5.3 h at
# 20 s, so it fits one Kaggle session with margin. It carries 97 gold boundaries -- enough
# for a recall estimate, and reported with a cluster interval over scenes because 25
# clusters is few (F-082).
G2_VLM_SCENES = 25
G2_VLM_SEED = 20260920

# The tag. ONE name, used by the prompt, the runner and the scorer alike (R4).
G2_VLM_TAG = "manoeuvre_changed"

VLM_PAIR_PROMPT = (
    "These are two photographs from the same car's dashboard camera, taken ONE SECOND\n"
    "apart. The first image is the earlier moment, the second is the later one.\n\n"
    "Question: between these two moments, did the car CHANGE what it was doing?\n\n"
    "A change means the car started or stopped turning, started or stopped accelerating,\n"
    "started or stopped braking, began or finished a lane change, or came to a stop or\n"
    "pulled away. Continuing to do the same thing is NOT a change, however fast the car is\n"
    "moving and however much the view shifts between the two images.\n\n"
    "Reply with exactly one JSON object and nothing else:\n"
    '{\n  "' + G2_VLM_TAG + '": true\n}'
)


def vlm_pair_items(n_scenes: int = G2_VLM_SCENES, seed: int = G2_VLM_SEED,
                   out_path: str = "outputs/g2_vlm_items.json") -> dict:
    """The work list for the VLM segmentation arm: one item per interior keyframe.

    An item is `<scene>:<kf>` and carries the two sample tokens the model is shown. The
    keyframe it asks about is the one BETWEEN them, so a `true` marks a boundary AT `kf`
    and lands in the same coordinate as the gold set -- no snapping, no off-by-one to
    argue about later.

    The first and last keyframe of a scene are skipped: they are boundaries for every
    segmenter including a trivial one, and the gold set excludes them for the same reason.
    """
    import numpy as np
    import pandas as pd

    fr = _frames()
    ev = pd.read_parquet(OUT / "maneuver_events_trainval.parquet")
    subset_scenes = sorted(json.loads(
        (OUT / "subset_tokens.json").read_text())["scenes"])
    rng = np.random.default_rng(seed)
    scenes = sorted(rng.permutation(subset_scenes)[:n_scenes].tolist())

    items, gold, n_kf = [], {}, {}
    for sc in scenes:
        f = fr[fr.scene_name == sc].sort_values("keyframe_index")
        toks = f.sample_token.tolist()
        kfs = f.keyframe_index.tolist()
        n_kf[sc] = len(kfs)
        gold[sc] = sorted(kinematic_boundaries(sc, ev, fr))
        for i in range(1, len(kfs) - 1):
            items.append({"item_id": f"{sc}:{kfs[i]}", "scene_name": sc,
                          "keyframe_index": int(kfs[i]),
                          "before_token": toks[i - 1], "after_token": toks[i + 1]})

    total_gold = sum(len(v) for v in gold.values())
    total_kf = sum(n_kf.values())
    payload = {
        "seed": seed, "n_scenes": len(scenes), "n_items": len(items),
        "scenes": scenes, "n_keyframes": n_kf, "gold": gold,
        "n_gold_boundaries": total_gold,
        # THE CHANCE FLOOR, CARRIED WITH THE WORK LIST. F-084: a boundary F1 read without
        # its null is mostly a statement about boundary density, and the null must be
        # fixed before the run rather than computed afterwards against whatever came back.
        "gold_density_per_kf": round(total_gold / total_kf, 4),
        "chance_f1_closed_form": {str(t): round(min(1.0, (2 * t + 1) * total_gold / total_kf), 4)
                                  for t in (0, 1, 2, 3)},
        "n_scenes_with_no_boundary": sum(1 for v in gold.values() if not v),
        "est_hours_at_12s": round(len(items) * 12 / 3600, 1),
        "est_hours_at_20s": round(len(items) * 20 / 3600, 1),
        "tag": G2_VLM_TAG,
        "items": items,
    }
    (ROOT / out_path).write_text(json.dumps(payload))
    return {k: v for k, v in payload.items()
            if k not in ("items", "gold", "n_keyframes", "scenes")}


def boundaries_from_rows(rows: list[dict]) -> dict[str, set[int]]:
    """Model rows -> the boundary set it predicted, per scene.

    A row that did not parse, or that answered nothing, contributes NO boundary and is NOT
    read as `false`: the same distinction `coerce_tags` exists to keep (F-062). It is
    counted separately by `score_vlm_boundaries`, because silently treating an unparsed
    row as "no boundary here" would let a broken run look like a conservative segmenter.
    """
    out: dict[str, set[int]] = {}
    for r in rows:
        if r.get("backend_error"):
            continue
        scene, _, kf = r["sample_token"].rpartition(":")
        out.setdefault(scene, set())
        if (r.get("values") or {}).get(G2_VLM_TAG) is True:
            out[scene].add(int(kf))
    return out


def score_vlm_boundaries(results_stem: str = "g2_vlm_pairs",
                         items_path: str = "outputs/g2_vlm_items.json",
                         out_csv: str = "outputs/g2_vlm_scores.csv",
                         tolerances: tuple[int, ...] = (0, 1, 2, 3)) -> Any:
    """Score the VLM arm against the same gold, with the baselines re-run on ITS scenes.

    The baselines are recomputed on the 25 scenes the VLM actually saw rather than quoted
    from the 850-scene table, because a comparison across two different scene sets is not
    a comparison. Everything else -- the matching rule, the tolerance sweep, the random
    null -- is the machinery above, unchanged.
    """
    import numpy as np
    import pandas as pd

    from . import eval as E

    spec = json.loads((ROOT / items_path).read_text())
    gold = {s: set(v) for s, v in spec["gold"].items()}
    n_kf = spec["n_keyframes"]
    rows = E.load_rows(results_stem)
    pred = boundaries_from_rows(rows)

    answered = [r for r in rows if not r.get("backend_error")
                and (r.get("values") or {}).get(G2_VLM_TAG) is not None]
    coverage = len(answered) / spec["n_items"]
    rng = np.random.default_rng(RANDOM_SEED)
    scenes = [s for s in spec["scenes"] if s in pred]

    out = []
    for tol in tolerances:
        methods = {
            "vlm_pairwise": {s: pred.get(s, set()) for s in scenes},
            "fixed_matched_k": {s: fixed_interval_boundaries(n_kf[s], len(gold[s]))
                                for s in scenes},
        }
        for name, bysc in methods.items():
            tp = fp = fn = 0
            for s in scenes:
                a, b, c = boundary_prf(bysc[s], gold[s], tol)
                tp, fp, fn = tp + a, fp + b, fn + c
            p_ = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            out.append({"tolerance_kf": tol, "method": name, "precision": p_, "recall": r_,
                        "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
                        "f1_sd": np.nan, "f1_lo": np.nan, "f1_hi": np.nan,
                        "n_predicted": sum(len(bysc[s]) for s in scenes),
                        "n_gold": sum(len(gold[s]) for s in scenes)})
        # The random null is re-drawn per tolerance on these scenes, matched to the VLM's
        # OWN boundary count as well as to the gold's. A model that says true on 60% of
        # keyframes must be compared with a random segmenter that does the same, or its
        # recall is being credited to enthusiasm rather than to timing.
        for label, counts in (("random_matched_gold", {s: len(gold[s]) for s in scenes}),
                              ("random_matched_vlm", {s: len(pred.get(s, set())) for s in scenes})):
            f1s = []
            for _ in range(N_RANDOM_DRAWS):
                tp = fp = fn = 0
                for s in scenes:
                    a, b, c = boundary_prf(random_boundaries(n_kf[s], counts[s], rng),
                                           gold[s], tol)
                    tp, fp, fn = tp + a, fp + b, fn + c
                p_ = tp / (tp + fp) if tp + fp else 0.0
                r_ = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
            # THE SPREAD, NOT JUST THE MEAN. Found in the notebook rehearsal: a fake model
            # answering `true` at random cleared the null's MEAN by +0.051 at +/-1, which
            # reads as a real effect and is 1.1 sd of a single draw. The model gets one
            # draw, so the honest comparison is against the null's 97.5th percentile, and a
            # gap reported without it is F-082's mistake in a different table.
            lo, hi = np.percentile(f1s, [2.5, 97.5])
            out.append({"tolerance_kf": tol, "method": label, "precision": np.nan,
                        "recall": np.nan, "f1": float(np.mean(f1s)),
                        "f1_sd": float(np.std(f1s)), "f1_lo": float(lo), "f1_hi": float(hi),
                        "n_predicted": sum(counts.values()),
                        "n_gold": sum(len(gold[s]) for s in scenes)})

    df = pd.DataFrame(out)
    df["n_scenes"] = len(scenes)
    df["answer_coverage"] = round(coverage, 4)
    df["parse_failure_rate"] = round(E.parse_failure_rate(rows), 4)
    df["chance_f1_closed_form"] = [spec["chance_f1_closed_form"][str(t)]
                                   for t in df.tolerance_kf]
    df = df.round(4)
    df.to_csv(ROOT / out_csv, index=False)
    return df


def build_g2_bundle(out_path: str = "outputs/g2_bundle.tar.gz",
                    items_path: str = "outputs/g2_vlm_items.json") -> dict:
    """Pack the VLM arm's images and work list. ~500 images, far smaller than Stage E's.

    A separate bundle rather than another flag on `build_colab_bundle`: that one is 542 MB
    because it carries the 628-frame subset plus E9's neighbours, and re-uploading it to
    run a 100 MB job would cost an hour of bandwidth for nothing.
    """
    import hashlib
    import tarfile

    from . import data as D

    spec = json.loads((ROOT / items_path).read_text())
    needed = sorted({t for it in spec["items"]
                     for t in (it["before_token"], it["after_token"])})
    out = ROOT / out_path
    missing, n_img = [], 0
    src_hashes = {}
    with tarfile.open(out, "w:gz") as tar:
        for name in ("outputs/g2_vlm_items.json", "src/storyboard.py", "src/eval.py",
                     "src/vlm.py", "src/prompts.py", "outputs/label_schema.json"):
            tar.add(ROOT / name, arcname=name)
            src_hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:12]
        for tok in needed:
            try:
                p = D.image_for_sample_token(tok)
            except Exception:                      # noqa: BLE001
                missing.append(tok)
                continue
            if Path(p).exists():
                tar.add(p, arcname=f"images/{tok}.jpg")
                n_img += 1
            else:
                missing.append(tok)
    # A HOLE IN THE IMAGE SET IS A SHORTER RESULTS FILE THAT NOTHING FLAGS (F-048). The
    # bundle refuses to be built incomplete rather than letting the notebook discover it
    # after the upload.
    assert not missing, f"{len(missing)} images absent from disk, e.g. {missing[:3]}"
    size_mb = out.stat().st_size / 1048576
    (ROOT / "outputs/g2_bundle_manifest.json").write_text(json.dumps({
        "sha256_12": src_hashes, "n_images": n_img, "n_items": spec["n_items"],
        "n_scenes": spec["n_scenes"], "size_mb": round(size_mb, 1)}, indent=1))
    return {"path": str(out), "n_images": n_img, "n_items": spec["n_items"],
            "size_mb": round(size_mb, 1)}
