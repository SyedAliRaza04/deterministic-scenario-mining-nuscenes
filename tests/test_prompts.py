"""Prompt-registry self-checks — Step E3.

One bug class, guarded twice. `prompt_version` is logged on every result row and
Stage F groups by it, so a prompt constant that is not reachable from REGISTRY is a
condition that can never be run or scored. `v5_yesno` was absent until 2026-09-05
because REGISTRY was typed dict[str, str] and v5 is a dict — the E8 ablation would
have quietly lost one of its three arms.

This is the R4 failure mode (a vocabulary written by hand in two places drifts) that
already produced F-016, where hand-written tag names diverged from event names and
left is_turning_left False on all 5,500 turning frames.

Run:  ./venv/bin/python tests/test_prompts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.prompts as P  # noqa: E402

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


def _constants() -> dict[str, object]:
    return {n: getattr(P, n) for n in dir(P) if n.startswith("PROMPT_")}


@check("test_every_prompt_constant_is_reachable_from_the_registry")
def _():
    """Regression 2026-09-05: v5_yesno existed but was not in REGISTRY."""
    unreachable = [n for n, v in _constants().items() if v not in P.REGISTRY.values()]
    assert not unreachable, (
        f"{unreachable} defined but absent from REGISTRY — these conditions cannot be "
        "run or scored, and nothing else would report it")


@check("test_registry_has_no_entry_without_a_constant")
def _():
    """The reverse drift: a registry key pointing at a literal defined nowhere."""
    # Not a set: PROMPT_V5_YESNO is a dict and dicts are unhashable.
    values = list(_constants().values())
    orphans = [k for k, v in P.REGISTRY.items()
               if not isinstance(v, (str, dict)) or not any(v is x for x in values)]
    assert not orphans, f"REGISTRY keys not backed by a PROMPT_* constant: {orphans}"


@check("test_registry_covers_every_stage_e_condition")
def _():
    """The conditions PLAN.md Stage E actually runs, including E6b from D-039.

    Named explicitly so that deleting a prompt fails here rather than silently
    shrinking the experiment.
    """
    required = {"v1_structured", "v2_cot", "v3_bev", "v3b_bev_symbolic",
                "v4_both", "v5_yesno", "v6_temporal"}
    missing = required - set(P.REGISTRY)
    assert not missing, f"Stage E conditions absent from REGISTRY: {sorted(missing)}"


@check("test_registry_keys_are_unique_and_lowercase_versioned")
def _():
    """Keys become the `prompt_version` column, so they must be stable identifiers."""
    for k in P.REGISTRY:
        assert k == k.lower(), f"{k} is not lowercase"
        assert k.startswith("v"), f"{k} does not carry a version prefix"


@check("test_tag_questions_cover_exactly_the_schema_scoreable_tags")
def _():
    """R4, enforced rather than trusted.

    TAG_QUESTIONS has to be hand-written: the schema's `derivation` says how a label is
    COMPUTED ("a ped_crossing polygon intersects the forward corridor"), which is the
    right text for an audit and the wrong text for a prompt - asking a VLM that measures
    whether it parses our jargon. A perceptual phrasing cannot be generated from a
    computational one.

    Hand-written means it can drift, which is exactly what produced F-016. So the drift
    is made loud: add tag 38, rename one, or drop a question, and this fails.
    """
    schema = P.load_schema()
    scored = set(P.scoreable_tags(schema))
    have = set(P.TAG_QUESTIONS)
    assert have == scored, (
        f"missing questions: {sorted(scored - have)}; "
        f"questions for tags that are not scored: {sorted(have - scored)}")


@check("test_no_prompt_asks_for_an_unscoreable_tag")
def _():
    """D-032: `on_walkway` has zero positives in 34,149 keyframes and `on_carpark` 23.
    Asking for them yields an undefined recall and a meaningless column in Stage F."""
    schema = P.load_schema()
    block = P.tag_block(schema)
    for tag, spec in schema["tags"].items():
        if spec["role"] != "scored":
            assert f'"{tag}"' not in block, f"{tag} is qc_invariant but is being asked for"


@check("test_every_tag_block_entry_carries_a_perceptual_gloss_not_a_derivation")
def _():
    """Regression 2026-09-06: the first build put the schema's derivation in the prompt,
    so the model was asked about 'polygons' and 'corridors' it has no way to see."""
    block = P.tag_block(P.load_schema())
    for jargon in ("polygon", "frustum", "keyframe", "instance token", "trend window"):
        assert jargon not in block.lower(), f"prompt leaks implementation jargon: {jargon!r}"


@check("test_bev_prompt_describes_every_class_the_renderer_can_draw")
def _():
    """PROMPT_V3_BEV must describe exactly what `src.bev` draws. A class added to the
    renderer but not to the legend makes the prompt silently untrue, and every BEV
    number with it (R36)."""
    import src.bev as B
    prompt = P.PROMPT_V3_BEV.lower()
    words = {"vehicle": "vehicle", "large_vehicle": "large vehicle",
             "pedestrian": "pedestrian", "cyclist": "cyclist", "barrier": "barrier",
             "traffic_cone": "traffic cone", "other": "other"}
    for style in {k for _, k in B._CATEGORY_STYLE} | {"other"}:
        assert words[style] in prompt, f"BEV legend does not describe {style!r}"
    layer_words = {"drivable_area": "drivable area", "lane": "lane",
                   "road_segment": "road segment", "carpark_area": "car park",
                   "walkway": "walkway", "ped_crossing": "pedestrian crossing",
                   "stop_line": "stop line"}
    for layer in B._MAP_LAYERS:
        assert layer_words[layer] in prompt, \
            f"BEV legend does not describe map layer {layer!r}"
    assert f"{B.CAM_FRONT_HFOV_DEG:.0f} degrees" in prompt, \
        "the camera wedge width in the prompt has drifted from src.bev"
    assert f"{B.BEV_RADIUS_M:.0f} metres" in prompt, \
        "the patch radius in the prompt has drifted from src.bev"


@check("test_the_repaired_bev_legend_covers_everything_the_repaired_raster_draws")
def _():
    """Same drift guard as above, for the arm that is about to be re-run.

    The repair exists because the raster drew no traffic-light layer (F-063c). A legend
    that does not name the new glyph would leave the tag unanswerable for a second reason
    after ten hours of GPU spent fixing the first.
    """
    import src.bev as B

    prompt = P.REGISTRY["v3_bev_r2"].lower()
    assert "traffic-light" in prompt or "traffic light" in prompt, \
        "the repaired legend never mentions the fixtures the raster now draws"
    assert "teal" in prompt, "the fixture glyph has no colour in the legend"
    assert "never shows which colour is lit" in prompt, \
        "the legend must say the map gives position only (F-004)"
    for layer in B.DRAWN_LAYERS:
        words = {"drivable_area": "drivable area", "lane": "lane",
                 "road_segment": "road segment", "carpark_area": "car park",
                 "walkway": "walkway", "ped_crossing": "pedestrian crossing",
                 "stop_line": "stop line", "traffic_light": "traffic-light"}
        assert words[layer] in prompt, f"repaired legend does not describe {layer!r}"
    # and the published prompt is untouched, or its results stop being reproducible from it
    assert "teal" not in P.REGISTRY["v3_bev"].lower()


@check("test_the_repaired_text_prompt_explains_the_blocks_the_description_now_has")
def _():
    """The description gained two blocks (road geometry, fixtures). A reader who is not
    told how to read them is being scored on a format change (F-064b)."""
    prompt = P.REGISTRY["v3b_bev_symbolic_r2"]
    assert "ROAD SURFACES" in prompt and "TRAFFIC LIGHT FIXTURES" in prompt, \
        "the repaired text prompt does not describe the blocks the description contains"
    assert "never which colour is lit" in prompt
    assert P.REGISTRY["v3b_bev_symbolic"] != prompt


@check("test_scope_descriptions_cover_exactly_the_schema_scopes")
def _():
    """Same drift guard as TAG_QUESTIONS, for the same reason (R4)."""
    schema = P.load_schema()
    assert set(P.SCOPE_JSON_KEYS) == set(schema["scopes"]), (
        "SCOPE_JSON_KEYS has drifted from the schema's scopes")
    assert set(P.SCOPE_DESCRIPTIONS) == set(schema["scopes"]), (
        f"missing: {sorted(set(schema['scopes']) - set(P.SCOPE_DESCRIPTIONS))}; "
        f"extra: {sorted(set(P.SCOPE_DESCRIPTIONS) - set(schema['scopes']))}")


@check("test_every_prompt_builds_and_contains_all_35_tags")
def _():
    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    for version in P.REGISTRY:
        if version == "v5_yesno":
            continue
        kw = {"n_frames": 5} if version == "v6_temporal" else {}
        text = P.build_prompt(version, schema, **kw)
        missing = [t for t in tags if f'"{t}"' not in text]
        assert not missing, f"{version} omits {missing}"
        assert len(text) > 2000, f"{version} is suspiciously short ({len(text)} chars)"
    y = P.build_yesno_prompts(schema)
    assert len(y) == len(tags) == 35, (len(y), len(tags))


@check("test_the_input_comparison_arms_share_one_answer_instruction")
def _():
    """RQ3a compares INPUTS, so those four arms must ask for the answer identically or
    the result is partly a prompt-wording effect.

    v2_cot is deliberately excluded: it is the PROMPT-strategy ablation (E8), so differing
    wording is the thing being measured, not a confound.
    """
    for v in ("v1_structured", "v3_bev", "v3b_bev_symbolic", "v4_both", "v6_temporal"):
        assert P._ANSWER_RULES in P.REGISTRY[v], f"{v} has diverged from the shared rules"


@check("test_the_reasoning_prompt_does_not_also_forbid_reasoning")
def _():
    """Regression 2026-09-12 (F-065). PROMPT_V2_COT asked the model to think step by step
    and then, four lines later, to answer "with ONE JSON object and nothing else".

    The model obeyed the prohibition: 0 of 628 frames produced any text before the JSON,
    so E8 measured a prompt the model had ignored. Same defect as the original
    query_multiframe, which claimed five images while sending one - the prompt described
    an experiment that was not being run.
    """
    cot = P.build_prompt("v2_cot")
    assert "nothing else" not in cot, (
        "the reasoning prompt forbids the reasoning it asks for; the model will skip it")
    assert "reasoning first" in cot or "SHORT paragraph" in cot, \
        "the reasoning instruction is gone"
    # and the plain arm must still forbid prose, or v1 and v2 stop differing
    assert "nothing else" in P.build_prompt("v1_structured"), \
        "v1 must still demand JSON only, or the E8 contrast disappears"


if __name__ == "__main__":
    for name, ok, err in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} prompt self-checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)
