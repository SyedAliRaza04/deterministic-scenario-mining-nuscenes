"""Self-check for the figure builders. Run directly: `python tests/test_figures.py`

Figures are artifacts like any other table, so they get the same treatment: the
SELECTION rules (which frames, which scenes, which seed) are tested on synthetic inputs
with known answers, and each defect that has actually bitten a figure in this project
gets a regression.

What is tested here is the choosing and the contract, not the pixels. A pixel diff would
break on a matplotlib point release and tell us nothing; what can silently go wrong is
picking three frames of the same turn (R35), drifting a colour vocabulary away from the
label vocabulary (R4/R36), leaking global rc state between builders, or publishing a
figure with no builder at all (R30) - which is exactly what had happened to the six C1,
C2 and C3 figures.

NOTE (R31): the runner lives at the END of this file. Tests appended after it never run.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import figures as F  # noqa: E402
from src.groundtruth import (  # noqa: E402
    LATERAL_LABELS, LONGITUDINAL_LABELS, POLYGON_LAYERS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def _toy_kin(n_scenes=3, n_kf=8):
    """Synthetic C1 table. scene-A holds the biggest turn, twice over.

    scene-A's keyframes 0-2 share ONE heading change on purpose: near a scene start the
    observation window slides instead of shrinking (D-010), so several consecutive
    keyframes really do carry identical window-level values, and index 0 of them is
    edge_guard. That tie is what the cross-check tie-break has to resolve.
    """
    rows = []
    for s in range(n_scenes):
        for k in range(n_kf):
            tied = s == 0 and k < 3
            rows.append({
                "sample_token": f"tok-{s}-{k}",
                "scene_name": f"scene-{chr(65 + s)}",
                "keyframe_index": k,
                "t_rel_s": 0.5 * k,
                "heading_change_deg": (-80.0 if tied else
                                       -70.0 if s == 0 else
                                       (30.0 - 10 * s) * (1 if k % 2 else -1)),
                "window_offset_s": (2.0 - k) if tied else 0.0,
                "speed_mps": 5.0,
                "edge_guard": s == 0 and k == 0,
            })
    return pd.DataFrame(rows)


def _toy_events():
    """Synthetic C2 event table: 3 scenes, 4 distinct kinds between them."""
    return pd.DataFrame([
        {"scene_name": "s1", "axis": "lateral", "kind": "turn_left",
         "start_s": 1.0, "end_s": 4.0},
        {"scene_name": "s1", "axis": "longitudinal", "kind": "decelerating",
         "start_s": 2.0, "end_s": 6.0},
        {"scene_name": "s2", "axis": "lateral", "kind": "turn_right",
         "start_s": 0.0, "end_s": 3.0},
        {"scene_name": "s3", "axis": "longitudinal", "kind": "stationary",
         "start_s": 0.0, "end_s": 9.0},
        {"scene_name": "s3", "axis": "longitudinal", "kind": "decelerating",
         "start_s": 0.0, "end_s": 2.0},
    ])


# --- selection rules ----------------------------------------------------------------


def test_turn_crosscheck_takes_one_frame_per_scene():
    """R35 regression: a plain nlargest returns n frames of the SAME turn.

    scene-A carries the four largest heading changes in the toy table. Sorting by
    magnitude alone would fill every panel from it and verify one manoeuvre n times,
    which is what the C4 panel did before F-032.
    """
    picks = F.c1_turn_crosscheck_frames(_toy_kin(), n=3)
    assert len(picks) == 3, f"expected 3 panels, got {len(picks)}"
    assert picks.scene_name.nunique() == 3, \
        f"panels must come from distinct scenes, got {list(picks.scene_name)}"


def test_turn_crosscheck_breaks_ties_toward_the_centred_window():
    """Regression for the D-010 slid window: tied frames are not interchangeable.

    Keyframes 0-2 of scene-A share an identical heading change because the window slid
    to stay 4 s long. Taking the first of them takes the edge_guard frame at index 0,
    whose INSTANTANEOUS columns are the ones D-012 flags as unreliable. The tie-break
    is smallest |window_offset_s|, i.e. the frame the measurement is really about.
    """
    picks = F.c1_turn_crosscheck_frames(_toy_kin(), n=3)
    top = picks.iloc[0]
    assert top.scene_name == "scene-A"
    assert top.keyframe_index == 2, \
        f"tie must resolve to the most centred window, got kf{top.keyframe_index}"
    assert not bool(top.edge_guard), "an edge_guard frame must never win a tie"


def test_turn_crosscheck_is_deterministic():
    kin = _toy_kin()
    a = list(F.c1_turn_crosscheck_frames(kin, n=3).sample_token)
    b = list(F.c1_turn_crosscheck_frames(kin, n=3).sample_token)
    assert a == b, "the same table must always choose the same panels"


def test_timeline_scenes_cover_every_event_kind():
    """R35: the panels are chosen to EXERCISE the vocabulary, not to be the busiest."""
    ev = _toy_events()
    picks = F.c2_timeline_scenes(ev, n=3)
    covered = set(ev[ev.scene_name.isin(picks)].kind)
    assert covered == set(ev.kind), f"kinds missed by the cover: {set(ev.kind) - covered}"


def test_timeline_scenes_fill_up_to_n_and_are_deterministic():
    ev = _toy_events()
    assert len(F.c2_timeline_scenes(ev, n=3)) == 3
    # two scenes already cover all four kinds, so the third is a fill, not a cover pick
    assert F.c2_timeline_scenes(ev, n=3) == F.c2_timeline_scenes(ev, n=3)
    assert len(F.c2_timeline_scenes(ev, n=2)) == 2, "must not exceed or undershoot n"


def test_class_examples_spread_over_scenes_and_top_up():
    """R35 inside one label, plus the fallback that keeps the row full.

    Two consecutive keyframes of the same turn are 0.5 s apart and visually identical,
    so a plain random draw buys sample size without information. But `turn_right` lives
    entirely in one scene in the mini split, so 'one per scene' cannot be a hard rule -
    a probe that silently returns fewer frames is a check that did not run.
    """
    pool = pd.DataFrame({"scene_name": ["a", "a", "a", "b", "c"],
                         "keyframe_index": range(5)})
    got = F._spread_over_scenes(pool, 3, np.random.default_rng(0))
    assert len(got) == 3
    assert got.scene_name.nunique() == 3, f"expected 3 scenes, got {list(got.scene_name)}"

    one_scene = pd.DataFrame({"scene_name": ["z"] * 4, "keyframe_index": range(4)})
    assert len(F._spread_over_scenes(one_scene, 3, np.random.default_rng(0))) == 3, \
        "a single-scene label must still fill the row"


def test_class_example_seed_is_frozen_and_local():
    """The seed must reproduce, and must not read or move the numpy GLOBAL state.

    `np.random.seed` would make the figure depend on whatever ran before it in the same
    process - the same class of irreproducibility the C8 subset seed exists to prevent.
    """
    pool = pd.DataFrame({"scene_name": list("abcdefgh"), "keyframe_index": range(8)})

    def draw():
        return list(F._spread_over_scenes(
            pool, 3, np.random.default_rng(F.CLASS_EXAMPLE_SEED)).keyframe_index)

    np.random.seed(1)
    first = draw()
    np.random.seed(999)                      # move the global state
    assert draw() == first, "selection must not depend on numpy's global RNG"
    assert draw() != list(F._spread_over_scenes(
        pool, 3, np.random.default_rng(F.CLASS_EXAMPLE_SEED + 1)).keyframe_index), \
        "a different seed must give a different draw, or the seed does nothing"


# --- vocabulary and scope guards ----------------------------------------------------


def test_event_colours_cover_the_real_event_vocabulary():
    """R4/R36: a colour table is a second copy of a vocabulary and will drift.

    Anything the detector emits must have a colour here, or the timeline paints a real
    manoeuvre as an unexplained band. F-016 is the same failure one layer down.
    """
    p = OUT / "maneuver_events_trainval.parquet"
    if not p.exists():
        print("     (skip: events table not built)")
        return
    kinds = set(pd.read_parquet(p).kind)
    assert kinds <= set(F._EVENT_COLOURS), \
        f"no colour for {sorted(kinds - set(F._EVENT_COLOURS))}"


def test_event_colours_do_not_invent_labels():
    """The inverse guard: no colour for something that is not an event.

    `going_straight` and `cruising` are the ABSENCE of an event. Giving either a band
    would draw a manoeuvre the C2 event table does not contain.
    """
    vocab = set(LATERAL_LABELS) | set(LONGITUDINAL_LABELS)
    assert set(F._EVENT_COLOURS) <= vocab, \
        f"colours for non-labels: {sorted(set(F._EVENT_COLOURS) - vocab)}"
    for never in ("going_straight", "cruising"):
        assert never not in F._EVENT_COLOURS, \
            f"{never} is not an event and must not be drawn as a band"


def test_map_layers_drawn_are_exactly_the_layers_the_pipeline_indexes():
    """R36 + D-019: the picture must not show a layer no label comes from.

    `road_block` is excluded from MapIndex because it is degenerate on
    singapore-queenstown; drawing it would put a layer on the figure that the map-context
    table never queries.
    """
    drawn = [layer for layer, _ in F._MAP_LAYER_STYLE]
    assert set(drawn) <= set(POLYGON_LAYERS), \
        f"drawn layers absent from MapIndex: {set(drawn) - set(POLYGON_LAYERS)}"
    assert "road_block" not in drawn, "road_block is excluded by D-019"
    assert len(drawn) == len(set(drawn)), "a layer is drawn twice"


# --- geometry -----------------------------------------------------------------------


def test_fit_circle_recovers_a_known_circle():
    """Analytic check: points on a circle of radius 12.5 about (100, -40)."""
    th = np.linspace(0.2, 3.4, 40)
    cx, cy, r = F._fit_circle(100 + 12.5 * np.cos(th), -40 + 12.5 * np.sin(th))
    assert abs(cx - 100) < 1e-6 and abs(cy + 40) < 1e-6, f"centre wrong: {cx}, {cy}"
    assert abs(r - 12.5) < 1e-6, f"radius wrong: {r}"


def test_fit_circle_survives_a_short_arc():
    """A U-turn is a partial arc, not a full circle; the fit must still return a circle.

    R2 in figure form: assert what geometry allows. The algebraic fit biases the radius
    on short arcs, which is stated in the docstring and is harmless for the claim the
    figure makes - but a NaN or a negative radius would not be.
    """
    th = np.linspace(0.0, 1.1, 12)
    _, _, r = F._fit_circle(20 * np.cos(th), 20 * np.sin(th))
    assert np.isfinite(r) and r > 0, f"degenerate radius {r}"
    assert 5.0 < r < 100.0, f"implausible radius {r} for a 20 m arc"


# --- the contract the module documents ----------------------------------------------


def test_no_builder_mutates_global_rcparams():
    """The style contract, tested rather than only asserted in the docstring.

    Global rc state is process-wide, so a builder that sets it makes every LATER figure
    in the same session depend on run order.

    `backend` is the one deliberate exception: every builder calls
    `matplotlib.use("Agg")` so it renders headless, which is a capability switch, not a
    style choice, and it cannot change what any figure looks like.
    """
    allowed = {"backend", "backend_fallback", "interactive"}
    p = OUT / "ego_kinematics_mini.parquet"
    if not p.exists():
        print("     (skip: C1 table not built)")
        return
    import matplotlib
    before = dict(matplotlib.rcParams)
    F.c1_kinematics_figure(pd.read_parquet(p),
                           str(ROOT / "outputs" / ".rcparam_probe.png"))
    (ROOT / "outputs" / ".rcparam_probe.png").unlink(missing_ok=True)
    changed = [k for k, v in matplotlib.rcParams.items()
               if k in before and k not in allowed and repr(before[k]) != repr(v)]
    assert not changed, f"builder leaked global rcParams: {changed[:8]}"


def test_the_figure_actually_renders():
    """R19: a figure that failed to render proves nothing, so render one for real.

    Cheapest full path in the module - cached parquet, no devkit, no images - so the
    matplotlib call chain is exercised on every run rather than only when someone looks.
    """
    p = OUT / "ego_kinematics_mini.parquet"
    if not p.exists():
        print("     (skip: C1 table not built)")
        return
    probe = ROOT / "outputs" / ".render_probe.png"
    out = F.c1_kinematics_figure(pd.read_parquet(p), str(probe))
    assert Path(out).exists() and probe.stat().st_size > 20_000, "figure did not render"
    probe.unlink()


def test_every_published_figure_has_a_builder():
    """R30, as a test rather than a promise.

    This is the defect the C1/C2/C3 builders were written to fix: PROCESS.md claimed
    "every published figure regenerates from src/" while six PNGs in outputs/ had no
    builder anywhere - the code that made them existed only in a chat transcript and was
    gone. A new figure published without a builder now fails here instead of silently
    becoming unreproducible.
    """
    src = (ROOT / "src" / "figures.py").read_text()
    known = set(re.findall(r'out_path: str = "(outputs/[^"]+\.png)"', src))
    published = {f"outputs/{p.name}" for p in OUT.glob("*.png")}
    orphans = sorted(published - known)
    assert not orphans, (
        f"published with no builder in src/figures.py: {orphans}. "
        "Give each a `build_*`/`*_figure` default out_path, or delete it (R30).")


def test_no_thesis_figure_draws_working_vocabulary_into_the_image():
    """Nothing from the working log may be rendered into a published figure (R49).

    `build_thesis_figures` suppressed only `Figure.suptitle`, so three figures whose
    narrative sat in `ax.set_title` printed stage codes, an em dash and a raw tag name
    straight into the PDF, and one still said "all five" after the sixth input condition
    existed. Panel labels are legitimate, so a blanket suppression is wrong.

    Two earlier versions of this check patched the drawing calls and passed while the
    figure was still wrong: reading titles alone missed axis ticks reading
    `visible_objects`, and adding the tick and legend calls still missed a legend built
    from `Patch(label=...)`, where no string is ever passed to `legend`. So nothing is
    patched except `savefig`, and what is inspected is the figure itself: every `Text`
    artist it is about to render, whatever call produced it.
    """
    import tempfile
    from unittest import mock

    from matplotlib.figure import Figure

    seen: list[str] = []
    real_savefig = Figure.savefig

    def capture(self, *a, **k):
        for t in self.findobj(match=lambda o: hasattr(o, "get_text")):
            txt = t.get_text()
            if isinstance(txt, str) and txt.strip():
                seen.append(txt)
        return real_savefig(self, *a, **k)

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(Figure, "savefig", capture):
            F.build_thesis_figures(tmp)

    assert len(seen) > 100, f"only {len(seen)} text artists captured; the hook has moved"
    text = "\n".join(seen)

    # A stage or step code: a capital letter and a digit standing alone, the spelling the
    # working files use (C6, F4, C8, G2b, H3). "F1" is excluded: it is the metric the
    # whole thesis is scored in, not a stage.
    codes = [c for c in re.findall(r"(?<![A-Za-z0-9])[A-H]\d+[ab]?(?![A-Za-z0-9])", text)
             if c != "F1"]
    assert not codes, f"internal stage codes rendered into a thesis figure: {codes}"

    for token in ("F-0", "D-0", "L-0", "R-0", "RQ1", "RQ2", "RQ3", "RQ4", "RQ5",
                  "PLAN.md", "PROCESS.md", "self-check", "docstring", "regression"):
        assert token not in text, f"working vocabulary {token!r} rendered into a figure"

    for dash in ("\u2014", "\u2013"):
        assert dash not in text, f"dash {dash!r} rendered into a thesis figure"

    for word in ("maneuver", "labeling", "behavior"):
        assert word not in text.lower(), f"US spelling {word!r} rendered into a figure"

    # Tag identifiers ARE published vocabulary: the schema appendix lists every tag by its
    # frozen name, so a figure may label a row `is_turn_left`. Family identifiers are not:
    # that appendix writes "Map context" and the chapters write "map context", so the
    # snake_case spelling appears nowhere the reader can see.
    fams = sorted(f for f in ("ego_maneuver", "map_context", "visible_objects")
                  if f in text)
    assert not fams, f"raw family identifiers in a thesis figure: {fams}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} figure self-checks passed")
