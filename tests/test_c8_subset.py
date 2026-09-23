"""Self-check for Step C8. Run directly: `python tests/test_c8_subset.py`

The subset is the input to every remaining stage, so most of these are invariants about
sampling rather than about geometry: it must be reproducible, it must not leak, and it
must actually make the rare tags scoreable.

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groundtruth import (  # noqa: E402
    MIN_SCOREABLE_POSITIVES, SUBSET_BUDGET, SUBSET_MAX_PER_SCENE, SUBSET_QUOTA,
    SUBSET_SEED, _scene_quota_cover, build_evaluation_subset, build_vlm_subset,
    label_tags,
)

ROOT = Path(__file__).resolve().parent.parent


def _toy(n_scenes=12, n_kf=40, seed=0):
    """Synthetic gt_all with one deliberately rare tag confined to 2 scenes."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        for k in range(n_kf):
            rows.append({
                "sample_token": f"tok{s:03d}_{k:03d}",
                "scene_name": f"scene-{s:04d}",
                "keyframe_index": k,
                "common": True,
                "mid": bool(rng.random() < 0.4),
                "rare": bool(s < 2 and rng.random() < 0.5),
            })
    return pd.DataFrame(rows)


def _schema(tags, scoreable=None):
    return {"tags": {t: {"scoreable": (scoreable is None or t in scoreable)}
                     for t in tags}}


def test_subset_is_byte_for_byte_reproducible():
    """A subset that cannot be regenerated cannot be audited.

    The seed is frozen in the module and stored in the output, so re-running the
    pipeline on another machine must select the identical frames.
    """
    gt, sch = _toy(), _schema(["common", "mid", "rare"])
    a = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=3, budget=60,
                                max_per_scene=8, seed=SUBSET_SEED, out_path=None)
    b = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=3, budget=60,
                                max_per_scene=8, seed=SUBSET_SEED, out_path=None)
    assert a["sample_tokens"] == b["sample_tokens"]
    c = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=3, budget=60,
                                max_per_scene=8, seed=SUBSET_SEED + 1, out_path=None)
    assert c["sample_tokens"] != a["sample_tokens"], "a different seed must differ"


def test_the_seed_and_rule_are_recorded_with_the_tokens():
    """The sampling rule is a methods-section claim; it has to live with the output."""
    gt, sch = _toy(), _schema(["common", "mid", "rare"])
    r = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=3, budget=60,
                                max_per_scene=8, seed=SUBSET_SEED, out_path=None)
    rule = r["sampling_rule"]
    assert rule["seed"] == SUBSET_SEED
    assert rule["quota"] == 10 and rule["budget"] == 60
    assert all(k in rule for k in ("stage_1", "stage_2", "stage_3"))


def test_greedy_cover_finds_the_scenes_that_hold_the_rare_tag():
    """Stage 1 exists so the rare tags are scoreable at all.

    `rare` lives only in scenes 0000 (21 positives) and 0001 (16). A quota of 30 cannot
    be met by either alone, so a correct cover must contain BOTH - while a random draw
    over 12 scenes would frequently miss them. The quota is deliberately above what one
    scene can supply; at 15 a single scene suffices and picking one is correct, not a bug.
    """
    gt = _toy()
    core, have = _scene_quota_cover(gt, ["common", "mid", "rare"], quota=30,
                                    seed=SUBSET_SEED)
    assert {"scene-0000", "scene-0001"} <= set(core), sorted(core)
    assert have[2] == 37, "the cover should take every available rare positive"


def test_greedy_cover_terminates_when_a_quota_is_unreachable():
    """Regression against an infinite loop: `rare` has fewer positives than the quota."""
    gt = _toy()
    core, have = _scene_quota_cover(gt, ["common", "mid", "rare"], quota=10_000,
                                    seed=SUBSET_SEED)
    assert len(core) <= gt.scene_name.nunique()
    assert have[2] < 10_000, "the quota is genuinely unreachable"


def test_per_scene_cap_is_respected():
    """Consecutive keyframes are 0.5 s apart and near-duplicates (R35).

    Without the cap the draw buys sample size without information.
    """
    gt, sch = _toy(), _schema(["common", "mid", "rare"])
    r = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=6, budget=200,
                                max_per_scene=3, seed=SUBSET_SEED, out_path=None)
    sub = gt[gt.sample_token.isin(r["sample_tokens"])]
    assert sub.groupby("scene_name").size().max() <= 3


def test_only_scoreable_tags_are_stratified_on():
    """D-032: a tag with no positives has no quota to reach.

    Including on_walkway would make the cover loop chase an impossible target.
    """
    gt = _toy()
    gt["dead"] = False
    sch = _schema(["common", "mid", "rare", "dead"], scoreable=["common", "mid", "rare"])
    r = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=3, budget=60,
                                max_per_scene=8, seed=SUBSET_SEED, out_path=None)
    assert "dead" not in r["tag_counts"]
    assert set(r["tag_counts"]) == {"common", "mid", "rare"}


def test_subset_never_exceeds_its_budget_and_has_no_duplicates():
    gt, sch = _toy(), _schema(["common", "mid", "rare"])
    r = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=6, budget=57,
                                max_per_scene=8, seed=SUBSET_SEED, out_path=None)
    toks = r["sample_tokens"]
    assert len(toks) <= 57
    assert len(set(toks)) == len(toks), "a frame was selected twice"
    assert r["n_frames"] == len(toks)


def test_random_scenes_are_added_on_top_of_the_cover():
    """Stage 2 is what stops the subset being a sample of unusually busy driving.

    Greedy set-cover picks the most eventful scenes by construction (F-045).
    """
    gt, sch = _toy(), _schema(["common", "mid", "rare"])
    r = build_evaluation_subset(gt, sch, quota=10, n_random_scenes=5, budget=300,
                                max_per_scene=40, seed=SUBSET_SEED, out_path=None)
    assert r["n_scenes"] > r["n_core_scenes"], "no random scenes were added"
    assert set(r["core_scenes"]) <= set(r["scenes"])


def test_real_subset_makes_every_scoreable_tag_scoreable():
    """The point of the whole step, on the real data."""
    p = ROOT / "outputs" / "subset_tokens.json"
    if not p.exists():
        print("     (skip: outputs/subset_tokens.json not built)")
        return
    r = json.loads(p.read_text())
    assert r["n_frames"] == SUBSET_BUDGET
    assert r["sampling_rule"]["seed"] == SUBSET_SEED
    assert r["sampling_rule"]["quota"] == SUBSET_QUOTA
    short = {t: n for t, n in r["tag_counts"].items() if n < MIN_SCOREABLE_POSITIVES}
    assert not short, f"tags below the scoreable floor in the subset: {short}"
    assert len(set(r["sample_tokens"])) == r["n_frames"]


def test_real_subset_tokens_all_exist_in_gt_all():
    """A token that is not in the ground truth cannot be scored against it."""
    p = ROOT / "outputs" / "subset_tokens.json"
    g = ROOT / "outputs" / "gt_all.parquet"
    if not (p.exists() and g.exists()):
        print("     (skip: subset or gt_all not built)")
        return
    r = json.loads(p.read_text())
    gt = pd.read_parquet(g)
    assert set(r["sample_tokens"]) <= set(gt.sample_token), "subset references unknown frames"
    sub = gt[gt.sample_token.isin(r["sample_tokens"])]
    assert sub.scene_name.nunique() == r["n_scenes"]
    assert sub.groupby("scene_name").size().max() <= SUBSET_MAX_PER_SCENE
    # all four map areas must survive, or the subset is geographically biased
    assert sub.location.nunique() == 4, f"only {sub.location.nunique()} locations"


def test_real_subset_lifts_the_rare_tags_without_erasing_the_common_ones():
    """Stratification must rebalance, not invert.

    is_going_straight should fall from 83.65% but stay dominant - if it collapsed, the
    subset would be a sample of unusual driving and Stage F would not generalise.
    """
    p = ROOT / "outputs" / "subset_tokens.json"
    if not p.exists():
        print("     (skip: subset not built)")
        return
    r = json.loads(p.read_text())
    pv = r["prevalence_full_vs_subset_pct"]
    assert pv["is_u_turn"][1] > pv["is_u_turn"][0], "the rarest tag must be lifted"
    assert pv["lane_change"][1] > pv["lane_change"][0]
    full_straight, sub_straight = pv["is_going_straight"]
    assert sub_straight < full_straight, "stratification should reduce the majority class"
    assert sub_straight > 50.0, \
        f"is_going_straight collapsed to {sub_straight}% - subset is not representative"


def test_subset_is_a_strict_subset_of_the_label_vocabulary():
    p = ROOT / "outputs" / "subset_tokens.json"
    if not p.exists():
        print("     (skip: subset not built)")
        return
    r = json.loads(p.read_text())
    assert set(r["tag_counts"]) <= set(label_tags()), "subset counts an unknown tag"


def test_the_vlm_execution_subset_costs_no_tag_its_scoreability():
    """D-047. Stage E runs 628 of C8's 1,800 frames because the T4 cannot run them all.

    The reduction is only legitimate if it costs nothing. Every frame carrying a rare tag
    is kept WHOLE and only common tags are subsampled: `is_u_turn` has 36 positives in the
    parent, so a proportional cut would halve it below the 30 floor and it would stop being
    scoreable - the exact failure C8 exists to prevent, reintroduced one stage downstream.
    """
    p = ROOT / "outputs" / "vlm_subset_tokens.json"
    if not p.exists():
        return
    v = json.loads(p.read_text())
    parent = json.loads((ROOT / "outputs" / "subset_tokens.json").read_text())

    assert set(v["sample_tokens"]) <= set(parent["sample_tokens"]), (
        "the execution subset must be NESTED in the evaluation subset, or the images and "
        "BEV rasters already rendered do not cover it")
    assert len(set(v["sample_tokens"])) == len(v["sample_tokens"]), "duplicate tokens"
    for tag, n in v["tag_counts"].items():
        assert n >= MIN_SCOREABLE_POSITIVES, f"{tag} fell to {n}, below the floor"
    assert v["min_positives"] == 36, (
        f"min positives moved to {v['min_positives']}; it should be unchanged from the "
        "parent, because every rare-tag frame is kept whole")


def test_the_vlm_subset_is_reproducible_from_its_frozen_seed():
    """A subset that cannot be regenerated cannot be audited (R30, D-036's rule)."""
    p = ROOT / "outputs" / "vlm_subset_tokens.json"
    if not p.exists():
        return
    on_disk = json.loads(p.read_text())
    gt = pd.read_parquet(ROOT / "outputs" / "gt_all.parquet")
    schema = json.loads((ROOT / "outputs" / "label_schema.json").read_text())
    rebuilt = build_vlm_subset(gt, schema,
                              str(ROOT / "outputs" / "subset_tokens.json"),
                              out_path=None)
    assert rebuilt["sample_tokens"] == on_disk["sample_tokens"], \
        "rebuilding from the frozen seed gives different frames"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} C8 self-checks passed")
