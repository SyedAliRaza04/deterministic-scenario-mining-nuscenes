"""Figure builders. Every published figure must be reproducible from here (R30).

Ad-hoc plotting in a transcript is not an artifact - it cannot be regenerated, and it
goes stale silently when the logic behind it changes (R29).

STYLE CONTRACT for every builder here, so figures cannot interfere with each other:
  * no `matplotlib.rcParams` / `plt.style.use` mutation. Style is passed per artist.
    Global rc state is process-wide and leaks into whatever is drawn next, which makes
    "run C1 then C4" produce a different C4 than "run C4 alone" - the exact class of
    irreproducibility R30 exists to prevent. `rc_context` would be the alternative;
    nothing here needs it, so nothing here has it.
  * every random choice draws from a LOCAL `np.random.default_rng(seed)` with a frozen
    seed constant below, never from the `np.random.*` global state, and the seed is
    printed on the figure so the artifact carries its own provenance (as C8 does).
  * every builder returns `out_path` and closes its figure.

DATA SOURCES. Cached `outputs/*.parquet` wherever the quantity is cached (D-021, R20).
The devkit is loaded only for the two things no parquet holds: camera JPEGs, and the
dense 50 Hz pose signals the C2 timeline draws its events on top of.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- figure parameters (one place only) --------------------------------------------

# Frozen so `c2_class_examples_figure` regenerates byte-identically. Same discipline as
# groundtruth.SUBSET_SEED; the value is the date the figure was first published.
CLASS_EXAMPLE_SEED = 20260809
N_CLASS_EXAMPLES = 3          # frames per manoeuvre label. 3 fits the page and is enough
                              # to show a label is not accidentally right once.
N_TURN_PANELS = 3             # C1 cross-check panels, one per scene (R35: never three
                              # frames from the same turn - that verifies one turn thrice).
N_TIMELINE_SCENES = 4         # C2 timeline panels. 2 already cover every event kind in
                              # mini; the other 2 show the detector NOT firing.
ROUNDABOUT_PATCH_HALF_M = 55.0  # half-width of the map patch drawn around an island.
                                # Islands run to 1500 m2 (ISLAND_MAX_AREA_M2, r ~ 22 m),
                                # so 55 m shows the island plus its approaches. Fixed
                                # rather than per-panel, so the five candidates are drawn
                                # at ONE scale and their sizes are comparable by eye.
UTURN_PATCH_MARGIN = 1.7      # the U-turn panels do scale per panel: the three events
UTURN_PATCH_MIN_HALF_M = 25.0 # differ 5x in radius (5 m to 27 m), and one fixed patch
                              # either buries the tightest or crops the widest.

# Event colours for the C2 timeline. Keyed on the event `kind` strings that
# `lateral_events`/`longitudinal_events` emit; the builder asserts every kind present in
# the table has an entry here, so a new label renders as a failure rather than as an
# unexplained grey band (R4/R36). `going_straight` and `cruising` are deliberately
# absent: they are the ABSENCE of an event, not an event.
_EVENT_COLOURS = {
    "turn_left":    "#28c76f",
    "turn_right":   "#ff8c1a",
    "u_turn":       "#7b61ff",
    "accelerating": "#00b0f0",
    "decelerating": "#8c564b",
    "stationary":   "#d62728",
}

# The storyboard needs a colour for the panels where NOTHING is happening — 38.98% of all
# driving is covered by no event. It is deliberately NOT an entry in _EVENT_COLOURS: a test
# asserts `going_straight` and `cruising` are absent from that dict because they are the
# absence of an event, and the strip must not quietly turn them into one (T1/R36).
_IDLE_COLOUR = "#b9c2cc"


# Map layers drawn behind the C3 panels, back to front, with the colour each is drawn
# in. Only the layers `MapIndex` actually indexes appear (road_block is excluded by
# D-019), so the picture cannot show a layer the pipeline never queries.
_MAP_LAYER_STYLE = (
    ("drivable_area", "#ffffff"),
    ("lane",          "#f5f9fc"),   # near-white: lanes tile the whole carriageway, so a
    ("road_segment",  "#d6e5f2"),   # strong colour here erases the road/junction contrast
    ("carpark_area",  "#f6dde6"),
    ("walkway",       "#dff0dc"),
    ("ped_crossing",  "#cfe0f2"),
    ("stop_line",     "#f8e3b4"),
)
_MAP_BACKGROUND = "#e6e6e6"   # anything NOT drivable, islands included
_MAP_EDGE = "#c3c8cc"

# Colour per tag family, so a wrong colour on the picture is a wrong tag in the table.
# Anything C4 emits no tag for is drawn GREY, so the vocabulary's coverage gap is
# visible rather than disguised: 2.9% of visible boxes are pushable_pullable (shopping
# carts, bins), bicycle_rack or animal, and painting a shopping cart the same blue as a
# traffic cone would show a construction tag that the table does not contain (F-031).
_BOX_COLOURS = {
    "human.":                          "#e8452b",   # pedestrians
    "vehicle.bicycle":                 "#ffb200",   # two-wheelers
    "vehicle.motorcycle":              "#ffb200",
    "movable_object.barrier":          "#00b0f0",   # cones / barriers / debris
    "movable_object.trafficcone":      "#00b0f0",
    "movable_object.debris":           "#00b0f0",
    "vehicle.":                        "#28c76f",   # everything else on wheels
}
_UNTAGGED_COLOUR = "#9aa0a6"


def _box_colour(category: str) -> str:
    for prefix, colour in _BOX_COLOURS.items():
        if category.startswith(prefix):
            return colour
    return _UNTAGGED_COLOUR


def c6_schema_figure(schema: dict,
                     out_path: str = "outputs/c6_label_schema.png") -> str:
    """Prevalence and majority baseline for every tag. Step C6 verification.

    This is the sheet Stage F has to be read against. A tag at 99.7% and a tag at 1.9%
    cannot be compared on raw accuracy, and 13 of the 36 have a majority baseline at or
    above 90% - so "the VLM scored 91%" means nothing until you know the baseline was
    91.2% (F-038). The bar shows prevalence; the marker shows what always answering the
    majority class already scores.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_colour = {"ego_maneuver": "#7b61ff", "map_context": "#00b0f0",
                  "visible_objects": "#28c76f", "interaction": "#ff8c1a"}
    order = ["ego_maneuver", "map_context", "visible_objects", "interaction"]
    rows = []
    for fam in order:
        items = [(t, e) for t, e in schema["tags"].items() if e["family"] == fam]
        rows += sorted(items, key=lambda kv: -kv[1]["prevalence_pct"])

    fig, ax = plt.subplots(figsize=(12.5, 0.34 * len(rows) + 3.2))
    y = np.arange(len(rows))[::-1]
    for yy, (tag, e) in zip(y, rows):
        colour = fam_colour[e["family"]]
        unscoreable = not e.get("scoreable", True)
        ax.barh(yy, e["prevalence_pct"], color=colour,
                alpha=0.35 if unscoreable else 0.9, height=0.68)
        ax.plot(e["majority_baseline_pct"], yy, marker="|", ms=13, mew=2.2,
                color="#d62728", zorder=4)
        note = f'  {e["prevalence_pct"]:.2f}%  (n={e["n_positive"]:,})'
        if unscoreable:
            note += "   not scoreable, consistency check only"
        ax.text(max(e["prevalence_pct"], e["majority_baseline_pct"]) + 1.2, yy, note,
                va="center", fontsize=8,
                color="#999" if unscoreable else "#333")

    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in rows], fontsize=9)
    for tick, (_, e) in zip(ax.get_yticklabels(), rows):
        tick.set_color(fam_colour[e["family"]])
    ax.set_xlim(0, 128)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("percent of the 34,149 keyframes", fontsize=10)
    ax.axvline(50, color="#ccc", lw=0.8, ls=":")
    handles = [plt.Line2D([], [], color=c, lw=7, alpha=0.9, label=family_label(f))
               for f, c in fam_colour.items()]
    handles.append(plt.Line2D([], [], color="#d62728", marker="|", ls="none", ms=13,
                              mew=2.2, label="majority baseline (what guessing scores)"))
    # outside the axes: at "lower right" it covered the rarest tags' own counts
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055),
              ncol=5, fontsize=8.5, frameon=False)
    n_nc = sum(e["near_constant"] for _, e in rows)
    if not _THESIS:  # in the thesis the caption carries this
        ax.set_title(
            f"C6 — the frozen label schema: {len(rows)} tags across 4 families\n"
            f"{n_nc} tags ({100*n_nc/len(rows):.0f}%) have a majority baseline ≥ 90%, so raw "
            "accuracy is uninformative for a third of the vocabulary",
            fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def f4_family_conditions_figure(fam: Any,
                                out_path: str = "outputs/f4_family_conditions.png") -> str:
    """F1 per label family x view condition. Step F4 — the thesis's headline chart.

    Drawn per FAMILY rather than as one aggregate because the families answer different
    questions and move in opposite directions: the camera wins objects outright, while
    the BEV picture is the only arm that reads `lane_change` at all (F-063a).

    The baseline marker is the point of the figure. `map_context` has a family majority
    baseline of 0.534 — higher than EVERY condition scores on it, best being
    chain-of-thought at 0.436 — so that whole family loses to a trivial always-yes
    predictor. Plotting F1 without the baseline beside it would show four honest-looking
    bars and hide that (F-038, F2).

    `fam` is the frame from `eval.build_family_table()`: families x conditions, plus a
    `baseline` column.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_colour = {"camera": "#28c76f", "cot": "#7b61ff", "bev_pic": "#00b0f0",
                   "bev_txt": "#ff8c1a", "both": "#8c564b", "temporal": "#e377c2"}
    conds = [c for c in cond_colour if c in fam.columns]
    order = [f for f in ("visible_objects", "map_context", "ego_maneuver", "interaction")
             if f in fam.index]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(len(order), dtype=float)
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * width
        vals = [fam.loc[f, c] for f in order]
        bars = ax.bar(x + off, vals, width * 0.92, color=cond_colour[c], label=COND_LABEL[c])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.2, color="#444")

    for xx, f in zip(x, order):
        b = fam.loc[f, "baseline"]
        ax.plot([xx - 0.44, xx + 0.44], [b, b], color="#d62728", lw=2.1, ls="--",
                zorder=5, solid_capstyle="butt")
        beaten = max(fam.loc[f, c] for c in conds) >= b
        ax.text(xx + 0.46, b, f" baseline {b:.2f}" + ("" if beaten else "  ← nothing beats it"),
                va="center", fontsize=8, color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{family_label(f)}\n({int(fam.loc[f, 'n_tags'])} tags)"
                        for f in order], fontsize=10)
    ax.set_ylabel("macro F1 within the family", fontsize=10)
    ax.set_ylim(0, max(0.78, float(fam[conds].to_numpy().max()) + 0.14))
    ax.grid(axis="y", color="#eee", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(conds), fontsize=9, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.11))
    if not _THESIS:  # in the thesis the caption carries this
        ax.set_title(
            "F4 — input representation by label family, 34 tags, 628 frames\n"
            "chain-of-thought wins every family; map_context loses to always-yes in "
            f"all {len(conds)}",
            fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def c8_subset_figure(subset: dict, out_path: str = "outputs/c8_subset.png") -> str:
    """Full-dataset vs subset prevalence per tag. Step C8 verification.

    The check PLAN asks for: the subset histogram must be visibly flatter than the
    full one, with the gain concentrated on the rare tags. Tags are ordered by full
    prevalence so the lift is read as a rising wedge on the left.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import MIN_SCOREABLE_POSITIVES

    pv = subset["prevalence_full_vs_subset_pct"]
    rows = sorted(pv.items(), key=lambda kv: kv[1][0])
    names = [t for t, _ in rows]
    full = np.array([v[0] for _, v in rows])
    sub = np.array([v[1] for _, v in rows])
    counts = [subset["tag_counts"][t] for t in names]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(16.5, 0.36 * len(rows) + 3.0),
        gridspec_kw={"width_ratios": [2.15, 1]})

    y = np.arange(len(rows))
    ax.barh(y - 0.2, full, height=0.38, color="#b9c4cc", label="full dataset (34,149)")
    ax.barh(y + 0.2, sub, height=0.38, color="#7b61ff",
            label=f"subset ({subset['n_frames']})")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("percent of keyframes", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("prevalence: full dataset vs stratified subset", fontsize=12)

    # the number that actually matters for Stage F: positives available per tag
    ax2.barh(y, counts, height=0.68,
             color=["#28c76f" if c >= MIN_SCOREABLE_POSITIVES else "#d62728"
                    for c in counts])
    ax2.axvline(MIN_SCOREABLE_POSITIVES, color="#d62728", ls="--", lw=1.3)
    # top of the axis: at the bottom it sat on the rarest tags' own count labels
    ax2.text(MIN_SCOREABLE_POSITIVES * 1.15, len(rows) - 1.4,
             f"scoreable floor\n({MIN_SCOREABLE_POSITIVES} positives, D-032)",
             fontsize=8.5, color="#d62728", va="top")
    for yy, c in zip(y, counts):
        ax2.text(c + 14, yy, str(c), va="center", fontsize=7.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xscale("log")
    ax2.set_xlim(8, max(counts) * 3.2)
    ax2.set_xlabel("positives in the subset (log scale)", fontsize=10)
    ax2.set_title("every tag clears the floor", fontsize=12)

    lift = sub[:8].mean() / max(full[:8].mean(), 1e-9)
    fig.suptitle(
        f"C8 — stratified evaluation subset: {subset['n_frames']} keyframes from "
        f"{subset['n_scenes']} scenes ({subset['n_core_scenes']} quota-covering + "
        f"{subset['n_scenes'] - subset['n_core_scenes']} random)\n"
        f"the 8 rarest tags are lifted {lift:.1f}× against the full dataset; "
        "seed and sampling rule are stored in subset_tokens.json",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def c7_coverage_figure(gt: Any, report: dict,
                       out_path: str = "outputs/c7_coverage.png") -> str:
    """The maneuver-coverage answer, in one figure. Step C7 — for the supervisor.

    Three panels, because the claim needs three different kinds of evidence:
      A  what the brief names against what nuScenes contains (counts of scenes)
      B  the ego speed distribution, which is WHY there is no highway driving
      C  class balance over the ego-maneuver tags, which is why C8 must stratify
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import (FAST_SCENE_MEDIAN_MPS, HIGHWAY_SPEED_MPS,
                              LATERAL_LABELS, LONGITUDINAL_LABELS)

    fig = plt.figure(figsize=(16.5, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.42, wspace=0.24)

    # --- A: the brief's maneuvers vs reality ----------------------------------------
    ax = fig.add_subplot(gs[0, :])
    bm = report["brief_maneuvers"]
    n_sc = report["n_scenes"]
    items = [
        ("roundabout\n(named in brief)", bm["roundabout"]["n_traversals"], "#ff8c1a"),
        ("highway merge\n(named in brief)", 0, "#d62728"),
        ("parallel parking / créneau\n(named in brief)", 0, "#d62728"),
        ("lane change\n(present instead)", report["present_instead"]["lane_change"]["n_scenes"], "#28c76f"),
        ("reversing\n(present instead)", report["present_instead"]["reversing"]["n_events"], "#7b61ff"),
        # either direction - counting is_turn_left alone under an "L/R" label
        # understated this by 164 scenes
        ("turns L/R\n(present instead)",
         int((gt.groupby("scene_name").is_turn_left.any()
              | gt.groupby("scene_name").is_turn_right.any()).sum()), "#28c76f"),
    ]
    xs = np.arange(len(items))
    ax.bar(xs, [max(v, 0.25) for _, v, _ in items],
           color=[c for _, _, c in items], width=0.62)
    for x, (_, v, _) in zip(xs, items):
        ax.text(x, max(v, 0.25) * 1.14,
                "NONE" if v == 0 else f"{v}  ({100*v/n_sc:.2g}% of scenes)",
                ha="center", fontsize=10.5,
                weight="bold" if v == 0 else "normal",
                color="#d62728" if v == 0 else "#333")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in items], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.2, n_sc * 2.2)
    ax.axhline(n_sc, color="#999", ls=":", lw=1)
    ax.text(len(items) - 0.4, n_sc * 1.06, f"all {n_sc} scenes",
            fontsize=8.5, color="#777", ha="right")
    ax.set_ylabel("scenes containing the manoeuvre  (log scale)", fontsize=10)
    ax.set_title("A: two of the three manoeuvres the brief names do not occur in nuScenes at all",
                 fontsize=13, pad=12)

    # --- B: why there is no highway merge -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(3.6 * gt.speed_mps, bins=90, color="#7b61ff", alpha=0.85)
    ax.axvline(3.6 * HIGHWAY_SPEED_MPS, color="#d62728", lw=2)
    ax.text(3.6 * HIGHWAY_SPEED_MPS - 1.5, ax.get_ylim()[1] * 0.62,
            "70 km/h, motorway speed\nreached by 0 of 34,149 keyframes",
            color="#d62728", fontsize=9.5, ha="right", weight="bold")
    ax.axvline(3.6 * FAST_SCENE_MEDIAN_MPS, color="#ff8c1a", lw=1.4, ls="--")
    ax.text(3.6 * FAST_SCENE_MEDIAN_MPS - 1.5, ax.get_ylim()[1] * 0.86,
            "50 km/h", color="#ff8c1a", fontsize=9, ha="right")
    mx = 3.6 * float(gt.speed_mps.max())
    ax.annotate(f"fastest keyframe\nin the dataset: {mx:.1f} km/h",
                xy=(mx, 60), xytext=(mx - 12, ax.get_ylim()[1] * 0.30),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666"))
    # headroom past the threshold, so the 70 km/h line reads as a threshold the data
    # fails to reach rather than as the edge of the axis
    ax.set_xlim(-1, 3.6 * HIGHWAY_SPEED_MPS + 12)
    ax.set_xlabel("ego speed, km/h", fontsize=10)
    ax.set_ylabel("keyframes", fontsize=10)
    ax.set_title("B: the ego never reaches motorway speed", fontsize=12)

    # --- C: class balance -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    tags = [f"is_{n}" for n in LATERAL_LABELS] + ["lane_change"] + \
           [f"is_{n}" for n in LONGITUDINAL_LABELS]
    vals = [100 * float(gt[t].mean()) for t in tags]
    order = np.argsort(vals)
    # A neutral ramp, deliberately NOT the family palette. Every tag in this panel is an
    # ego manoeuvre tag, subdivided by axis; borrowing the family colours made
    # `is_cruising` green, which in the schema figure's key means a visible object. A
    # figure's colour vocabulary is a claim about the schema (R36), so this one states its
    # own and cannot be read against another figure's.
    from matplotlib.patches import Patch
    axis_colour = {"steering": "#37474f", "lane change": "#78909c", "speed": "#b0bec5"}
    def axis_of(t):
        if t == "lane_change":
            return "lane change"
        return "steering" if t[3:] in LATERAL_LABELS else "speed"
    cols = [axis_colour[axis_of(t)] for t in tags]
    ax.barh(np.arange(len(tags)), [vals[i] for i in order],
            color=[cols[i] for i in order], height=0.7)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in axis_colour.items()],
              loc="lower right", fontsize=8.5, frameon=False, title="manoeuvre axis",
              title_fontsize=8.5)
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels([tags[i] for i in order], fontsize=9)
    for y, i in enumerate(order):
        ax.text(vals[i] + 1.2, y, f"{vals[i]:.2f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent of the 34,149 keyframes", fontsize=10)
    ax.set_title("C: severe class imbalance: sequential sampling would be\n"
                 "~84% straight-line driving, so the subset must stratify"
                 if _THESIS else
                 "C: severe class imbalance: sequential sampling would be\n"
                 "~84% straight-line driving, so C8 must stratify", fontsize=12)

    fig.suptitle(
        "C7 — maneuver coverage over all 850 scenes / 34,149 keyframes  "
        "(RQ1; the evidence for L-001)",
        fontsize=15, y=0.985)
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out_path


def c7_lane_change_figure(events: Any, kin: Any,
                          out_path: str = "outputs/c7_lane_changes.png",
                          n_panels: int = 9, dataroot: str = "data/nuscenes") -> str:
    """BEV panels of detected lane-change events, in the MAP frame. Step C7.

    R38: the detector's tests live in the map frame (centreline offsets), so that is
    the frame drawn. Panels are stratified across detector type and separation so the
    rare configurations get looked at, not resampled away (R35). Trainval has no
    images, so the map frame is also the only frame available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import MapIndex, lane_centrelines

    ev = events.copy()
    ev["dur_s"] = (ev.kf1 - ev.kf0) * 0.5
    picks = []
    for det in ("token_flip", "settlement", "settlement+token_flip"):
        sub = ev[ev.detector == det].sort_values("sep_m")
        for q in (0.05, 0.5, 0.95):                  # extremes AND the typical case
            if len(sub):
                picks.append(sub.iloc[min(int(q * len(sub)), len(sub) - 1)])
    picks = picks[:n_panels]

    indices: dict[str, MapIndex] = {}
    ncol = 3
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 6.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, picks):
        g = kin[kin.scene_name == r.scene_name].sort_values("keyframe_index")
        loc = g.location.iloc[0]
        if loc not in indices:
            indices[loc] = MapIndex(dataroot, loc)
        cls = lane_centrelines(indices[loc].nusc_map)
        mid = g[g.keyframe_index.between(r.kf0, r.kf1)]
        cx, cy = float(mid.ego_x.mean()), float(mid.ego_y.mean())
        for pts in cls.values():
            P = np.asarray(pts)[:, :2]
            if np.abs(P[:, 0] - cx).min() < 45 and np.abs(P[:, 1] - cy).min() < 45:
                ax.plot(P[:, 0], P[:, 1], "-", color="#b9c4cc", lw=1)
        ax.plot(g.ego_x, g.ego_y, "k-", lw=1)
        w = g[g.keyframe_index.between(r.kf0 - 2, r.kf1 + 2)]
        ax.plot(w.ego_x, w.ego_y, "r-", lw=2.6)
        ax.plot(w.ego_x.iloc[0], w.ego_y.iloc[0], "g^", ms=10)   # entry marker
        ax.set_xlim(cx - 40, cx + 40)
        ax.set_ylim(cy - 40, cy + 40)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r.scene_name} kf{r.kf0}–{r.kf1}  ·  {r.detector}\n"
                     f"sep {r.sep_m:.2f} m over {r.dur_s:.1f} s", fontsize=9)
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(
        "C7 — detected ego lane changes, map frame (grey = lane centrelines, "
        "black = scene track, red = event ±1 s, green = entry)\n"
        "stratified across detector type × separation percentile (p05 / p50 / p95)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def c5_interactions_figure(nusc: Any, picks: list[tuple[str, str]],
                           out_path: str = "outputs/c5_interactions.png") -> str:
    """Camera view beside an ego-frame bird's-eye view, per frame. Step C5 verification.

    The BEV is the point: every C5 test is a statement in the ego frame, and the
    original defect was invisible precisely because nobody drew the frame the test was
    evaluated in. The shaded wedge is the OLD global-frame `abs(dy) > abs(dx)` region,
    rotated into the ego frame - it swings with the ego's heading instead of pointing
    where the car is going, which is the whole bug in one picture (F-033).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Wedge
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from PIL import Image

    from .groundtruth import (
        LEAD_LATERAL_M, LEAD_RANGE_M, PED_INTERACT_RANGE_M, interaction_labels,
        scene_agent_tracks,
    )

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(19, 5.6 * n),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    axes = np.atleast_2d(axes)

    for r, (tok, why) in enumerate(picks):
        sample = nusc.get("sample", tok)
        scene = nusc.get("scene", sample["scene_token"])
        frames = scene_agent_tracks(nusc, scene)
        toks = []
        t = scene["first_sample_token"]
        while t:
            toks.append(t)
            t = nusc.get("sample", t)["next"]
        k = toks.index(tok)
        row = interaction_labels(frames, k, with_keys=True)
        contrib = row["_contributors"]
        states = frames[k]

        # --- left: the camera image, with the interacting agents outlined ----------
        ax = axes[r][0]
        cam = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", cam)
        path, boxes, K = nusc.get_sample_data(cam, box_vis_level=BoxVisibility.ANY)
        ax.imshow(Image.open(path))
        by_instance = {nusc.get("sample_annotation", b.token)["instance_token"]: b
                       for b in boxes}
        for key, s in states.items():
            b = by_instance.get(key)
            if b is None:
                continue
            colour = _interaction_colour(key, contrib)
            if colour is None:
                continue
            c = view_points(b.corners(), K, normalize=True)[:2]
            x0, x1 = max(c[0].min(), 0), min(c[0].max(), sd["width"])
            y0, y1 = max(c[1].min(), 0), min(c[1].max(), sd["height"])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colour, linewidth=2.4))
            ax.text(x0, y0 - 5, f"{s['fwd_m']:.0f} m fwd, {s['lat_m']:+.1f} m lat",
                    color=colour, fontsize=8, weight="bold")
        ax.set_xlim(0, sd["width"]); ax.set_ylim(sd["height"], 0)
        ax.set_xticks([]); ax.set_yticks([])
        on = [t for t in ("lead_vehicle", "lead_braking", "cut_in",
                          "pedestrian_crossing_path") if row[t]]
        ax.set_title(f"{scene['name']} kf{k}  ·  CAM_FRONT  —  probes: {why}\n"
                     f"TAGS: {', '.join(on) if on else '(none)'}", fontsize=10)

        # --- right: the ego frame, where every test is actually evaluated ----------
        ax = axes[r][1]
        lim = LEAD_RANGE_M + 8
        yaw = _ego_yaw(nusc, sample)
        # the OLD test's region, rotated out of the global frame into the ego frame
        for base in (90.0, 270.0):
            ax.add_patch(Wedge((0, 0), lim * 1.8, base - 45 - np.degrees(yaw),
                               base + 45 - np.degrees(yaw), color="#d62728", alpha=0.10))
        ax.add_patch(Rectangle((-LEAD_LATERAL_M, 0), 2 * LEAD_LATERAL_M, LEAD_RANGE_M,
                               color="#28c76f", alpha=0.16))
        ax.plot(0, 0, marker="^", ms=15, color="k", zorder=5)
        for key, s in states.items():
            colour = _interaction_colour(key, contrib) or "#c8ccd0"
            ax.plot(-s["lat_m"], s["fwd_m"], "o", ms=6, color=colour, zorder=4)
            if np.isfinite(s["v_lat_mps"]) and s["speed_mps"] > 0.5:
                ax.arrow(-s["lat_m"], s["fwd_m"], -s["v_lat_mps"], s["v_fwd_mps"],
                         head_width=0.7, color=colour, alpha=0.85, zorder=3,
                         length_includes_head=True)
        circle = plt.Circle((0, 0), PED_INTERACT_RANGE_M, fill=False, ls=":",
                            color="#888", lw=1)
        ax.add_patch(circle)
        ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)   # +x drawn LEFT, matching "left"
        ax.set_aspect("equal")
        ax.set_xlabel("lateral, m  (left of the ego is left of the plot)", fontsize=9)
        ax.set_ylabel("forward, m", fontsize=9)
        ax.axhline(0, color="#bbb", lw=0.7); ax.axvline(0, color="#bbb", lw=0.7)
        # Make the comparison quantitative rather than suggestive: count what each
        # region actually contains on THIS frame.
        old_hits = 0
        for st in states.values():
            if not st["cat"].startswith("vehicle."):
                continue
            if np.hypot(st["fwd_m"], st["lat_m"]) >= 15.0:
                continue
            gdx = st["fwd_m"] * np.cos(yaw) - st["lat_m"] * np.sin(yaw)
            gdy = st["fwd_m"] * np.sin(yaw) + st["lat_m"] * np.cos(yaw)
            old_hits += abs(gdy) > abs(gdx)
        ax.set_title("ego frame — green = lane corridor, "
                     "red shading = the OLD global-frame test's region\n"
                     f"ego heading {np.degrees(yaw):+.0f}°  ·  "
                     f"old test fires on {old_hits} vehicle(s) here, "
                     f"lead corridor holds {int(row['lead_vehicle'])}"
                     "   (arrows are agent ground velocity)", fontsize=9)

    fig.suptitle(
        "C5 — interaction tags, evaluated in the EGO FRAME\n"
        "green = lead vehicle · orange = cut-in · red = pedestrian crossing · "
        "grey = other annotated agent",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=3.0)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)
    return out_path


def _ego_yaw(nusc: Any, sample: dict) -> float:
    from pyquaternion import Quaternion
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    q = Quaternion(nusc.get("ego_pose", sd["ego_pose_token"])["rotation"])
    return float(q.yaw_pitch_roll[0])


def _interaction_colour(key: str, contributors: dict[str, list[str]]) -> str | None:
    """Colour an agent by the tag it contributes to, per the labeller itself.

    Reads `interaction_labels(..., with_keys=True)` rather than re-testing the
    predicates, so the picture cannot disagree with the table (the F-016 lesson).
    """
    for tag, colour in (("lead_braking", "#e8452b"), ("cut_in", "#ff8c1a"),
                        ("pedestrian_crossing_path", "#e8452b"),
                        ("lead_vehicle", "#28c76f")):
        if key in contributors[tag]:
            return colour
    return None


def build_c4_verification_figure(objects_parquet: str = "outputs/objects_mini.parquet",
                                 out_path: str = "outputs/c4_visible_objects.png") -> str:
    """One call, from the cached table to the published figure. Step C4.

    R30: the roundabout tables once existed only because of ad-hoc scripts in a
    transcript, so nothing cited in the thesis is left to glue code again.
    Mini split, because it is the only one that ships images.
    """
    import pandas as pd

    from .data import load_nusc

    picks = c4_verification_frames(pd.read_parquet(objects_parquet))
    return c4_visible_objects_figure(load_nusc("v1.0-mini"),
                                     [t for t, _ in picks], out_path=out_path,
                                     why=[w for _, w in picks])


def c4_verification_frames(objects_df: Any) -> list[tuple[str, str]]:
    """Frames that EXERCISE the tag vocabulary, one per scene. Step C4 verification.

    Deliberately stratified rather than random: a random draw resamples the common case,
    and the tags at risk of being wrong are the rare ones. One frame per scene is
    enforced because the first attempt sorted by box count and returned the same busy
    scene six times over - a panel of near-duplicates verifies almost nothing.
    """
    picks: list[tuple[str, str]] = []
    seen_scenes: set[str] = set()

    taken: set[str] = set()

    def take(mask: Any, why: str) -> None:
        """Prefer an unseen scene; fall back to any unused frame rather than skip.

        A probe that silently returns nothing is a check that did not run (R31's
        lesson in a different guise), so the fallback is deliberate.
        """
        for pool in (objects_df[mask & ~objects_df.scene_name.isin(seen_scenes)],
                     objects_df[mask & ~objects_df.sample_token.isin(taken)]):
            if len(pool):
                row = pool.sort_values("n_boxes_visible", ascending=False).iloc[0]
                seen_scenes.add(row.scene_name)
                taken.add(row.sample_token)
                picks.append((row.sample_token, why))
                return
        print(f"  !! no frame exercises: {why}")

    df = objects_df
    take(df.cyclist, "cyclist — must require cycle.with_rider")
    take(df.parked_bicycle & ~df.cyclist, "parked_bicycle but NOT cyclist")
    take(df.pedestrian_near, f"pedestrian_near")
    take((df.n_traffic_cone > 0) & (df.n_barrier > 0), "cones AND barriers")
    take(df.large_vehicle & df.moving_vehicle, "large + moving vehicle")
    take(df.n_boxes_visible == 0, "nothing visible ahead")
    take(df.n_boxes_visible >= 20, "crowded frame")
    take(df.parked_vehicle & ~df.moving_vehicle, "parked, nothing moving")
    return picks


def c4_visible_objects_figure(nusc: Any, sample_tokens: list[str],
                              out_path: str = "outputs/c4_visible_objects.png",
                              camera: str = "CAM_FRONT",
                              why: list[str] | None = None) -> str:
    """Projected 3D boxes with the C4 tags printed underneath. Step C4 verification.

    The scope is stated on every panel (R24): these boxes and these tags are BOTH
    statements about this one camera at this one instant. The C1 figure once put a
    360 deg / 20 s scene description under a single front-camera frame and made correct
    data look like garbage (F-025); the count of discarded 360 deg annotations is
    printed here precisely so the difference is visible rather than hidden.

    Needs image files, so it runs on the mini split only (trainval is metadata-only).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from PIL import Image

    from .groundtruth import OBJECT_TAGS, visible_objects

    import textwrap

    n = len(sample_tokens)
    ncol, nrow = 2, int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 7.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, tok) in enumerate(zip(axes, sample_tokens)):
        sample = nusc.get("sample", tok)
        cam_tok = sample["data"][camera]
        sd = nusc.get("sample_data", cam_tok)
        W, H = sd["width"], sd["height"]
        path, boxes, K = nusc.get_sample_data(cam_tok, box_vis_level=BoxVisibility.ANY)
        ax.imshow(Image.open(path))

        for b in boxes:
            ann = nusc.get("sample_annotation", b.token)
            c = view_points(b.corners(), K, normalize=True)[:2]
            colour = _box_colour(ann["category_name"])
            # clip to the image: BoxVisibility.ANY admits boxes hanging off the edge
            # (1.9% have under a tenth of their extent inside), and drawing them
            # unclipped stretches the axes and hides the picture.
            x0, x1 = max(c[0].min(), 0), min(c[0].max(), W)
            y0, y1 = max(c[1].min(), 0), min(c[1].max(), H)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colour, linewidth=1.6))
            ax.text(x0, y0 - 4, f"{float(np.linalg.norm(b.center)):.0f}m", color=colour,
                    fontsize=7, weight="bold")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)

        row = visible_objects(nusc, cam_tok, n_annotations_360=len(sample["anns"]))
        on = [t for t in OBJECT_TAGS if row[t]]
        nvis, n360 = row["n_boxes_visible"], row["n_annotations_360"]
        drop = (f"ALL {n360} discarded" if nvis == 0
                else f"{n360 - nvis} discarded, {n360 / nvis:.1f}×")
        scene = nusc.get("scene", sample["scene_token"])["name"]
        probe = f"  —  probes: {why[i]}" if why else ""
        ax.set_title(f"{scene}  ·  {camera} only, this instant{probe}\n"
                     f"{nvis} boxes in frustum, of {n360} annotated over 360°  ({drop})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        body = ("(no tags)" if not on else
                ",  ".join(f"{t}={row['n_' + t]}" for t in on))
        ax.text(0.0, -0.015, "TAGS  " + "\n".join(textwrap.wrap(body, 95)),
                transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "C4 — visible-object tags, from camera-frustum-filtered 3D boxes\n"
        "red = pedestrian · green = vehicle · amber = two-wheeler · blue = cone/barrier · "
        "grey = visible but C4 emits no tag for it"
        "   (number above each box = distance from the lens)",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=5.5)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# =====================================================================================
# C1 - ego kinematics
# =====================================================================================


def c1_kinematics_figure(kin: Any,
                         out_path: str = "outputs/c1_kinematics_overview.png") -> str:
    """Speed, yaw rate and windowed heading change over a whole split. Step C1.

    The QC sheet for C1: every physical bound in `tests/test_c1_kinematics.py` is
    something you can also read off this picture. Scene boundaries are drawn because
    the series is a CONCATENATION of independent 20 s recordings, not one trajectory -
    the vertical steps at those lines are joins, not braking.

    SCOPE (R24): one point per keyframe. `speed`/`yaw rate` are instantaneous at the
    keyframe; `heading change` is a WINDOW_S-second quantity centred on it (D-010), so
    the three panels are not statements about the same instant and the axis labels say
    so.

    Reads the cached C1 parquet only - no devkit (D-021).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import (MIN_TURN_DEG, RESAMPLE_HZ, SMOOTH_S, STATIONARY_MPS,
                              WINDOW_S)

    k = kin.sort_values(["scene_name", "keyframe_index"]).reset_index(drop=True)
    x = np.arange(len(k))

    fig, axes = plt.subplots(3, 1, figsize=(20, 11.2), sharex=True)

    axes[0].plot(x, k.speed_mps, color="#1f77b4", lw=1.4)
    axes[0].axhline(STATIONARY_MPS, color="k", ls="--", lw=1.1)
    axes[0].text(2, STATIONARY_MPS + 0.35, f"stationary {STATIONARY_MPS} m/s", fontsize=10)
    axes[0].set_ylabel("speed (m/s)", fontsize=11)

    axes[1].plot(x, k.yaw_rate_deg_s, color="#d62728", lw=1.4)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].set_ylabel("yaw rate (deg/s)", fontsize=11)

    axes[2].plot(x, k.heading_change_deg, color="#2ca02c", lw=1.6)
    for s in (+MIN_TURN_DEG, -MIN_TURN_DEG):
        axes[2].axhline(s, color="k", ls=":", lw=1.1)
    # Stated in full because the number is easy to misread as a per-frame gate. C2
    # applies MIN_TURN_DEG to EVENTS (D-013); no window in the dataset reaches a U-turn.
    axes[2].text(2, MIN_TURN_DEG + 2.5,
                 f"C2 turn threshold +/-{MIN_TURN_DEG:.0f} deg "
                 "(applied to EVENTS, not to this window)", fontsize=10)
    axes[2].set_ylabel(f"heading change over {WINDOW_S:.0f} s (deg)", fontsize=11)

    # scene boundaries: the series is 10 separate recordings laid end to end
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
    starts = k.groupby("scene_name", sort=False).apply(lambda g: g.index[0])
    top = axes[0].get_ylim()[1]
    for name, i in starts.items():
        for ax in axes:
            ax.axvline(i, color="#888", ls=":", lw=1.0)
        axes[0].text(i + 2, top * 0.94, name.replace("scene-", ""),
                     fontsize=9, color="#666", va="top")

    axes[2].set_xlabel(f"keyframe index (all {len(k)} keyframes of the split, "
                       "scene boundaries dotted)", fontsize=11)
    axes[2].set_xlim(-8, len(k) + 8)
    split = "nuScenes mini" if len(k) < 1000 else "nuScenes trainval"
    fig.suptitle(f"C1 - ego kinematics over {split} "
                 f"({RESAMPLE_HZ:.0f} Hz resampled, {SMOOTH_S:.2f} s smoothing)",
                 fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def c1_turn_crosscheck_frames(kin: Any, n: int = N_TURN_PANELS) -> Any:
    """The n largest windowed heading changes, ONE PER SCENE. Step C1 verification.

    One per scene is the point (R35): a plain nlargest returns four consecutive
    keyframes of the same turn, which verifies one turn four times.

    Ties are broken toward the CENTRED window. Keyframes near a scene start share an
    identical slid window (D-010), so `heading_change_deg` is literally equal across the
    first few; picking the smallest |window_offset_s| among them takes the frame the
    measurement is actually about, and avoids the edge_guard frame at index 0.
    """
    k = kin.assign(_absdh=kin.heading_change_deg.abs(),
                   _off=kin.window_offset_s.abs())
    pick = (k.sort_values(["_absdh", "_off"], ascending=[False, True])
             .groupby("scene_name", sort=False).head(1))
    return pick.sort_values("_absdh", ascending=False).head(n)


def c1_turn_crosscheck_figure(nusc: Any, kin: Any,
                              out_path: str = "outputs/c1_turn_crosscheck.png",
                              n: int = N_TURN_PANELS,
                              camera: str = "CAM_FRONT") -> str:
    """The largest heading changes, each beside the camera view. Step C1 verification.

    SCOPE, and the whole reason this figure was regenerated once already (F-025): the
    panel title states what is FRUSTUM-VISIBLE in this one camera at this one instant,
    and the scene text underneath is marked, in grey italic, as a 360 deg / 20 s
    statement that does NOT apply to the frame. The first version captioned a single
    front-camera frame with the whole-scene description and made correct data look like
    garbage labels - a 12x over-count on scene-0916.

    Needs image files, so it runs on a split whose JPEGs are on disk.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import textwrap
    from collections import Counter

    from .groundtruth import MIN_TURN_DEG, WINDOW_S, frustum_boxes

    picks = c1_turn_crosscheck_frames(kin, n)
    fig, axes = plt.subplots(2, len(picks), figsize=(6.6 * len(picks), 9.6),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    axes = np.atleast_2d(axes)

    for c, r in enumerate(picks.itertuples()):
        sample = nusc.get("sample", r.sample_token)
        cam_tok = sample["data"][camera]
        boxes = frustum_boxes(nusc, cam_tok)
        seen = Counter(b["cat"].split(".")[-1] for b in boxes)
        visible = ", ".join(f"{n_}x {c_}" for c_, n_ in seen.most_common(4)) or "nothing"

        ax = axes[0][c]
        ax.imshow(Image.open(nusc.get_sample_data_path(cam_tok)))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{r.scene_name} kf{r.keyframe_index} - heading {r.heading_change_deg:+.1f} deg"
            f", {r.speed_mps:.1f} m/s\n"
            f"VISIBLE IN THIS VIEW: {visible}   "
            f"({len(sample['anns'])} annotated in 360 deg, {len(boxes)} visible)",
            fontsize=10, weight="bold")
        # R24: the scene text is a different kind of statement and is labelled as one.
        desc = nusc.get("scene", sample["scene_token"])["description"]
        ax.text(0.5, -0.035,
                f"scene-level description (all 6 cameras, whole 20 s - NOT this frame):\n"
                + "\n".join(textwrap.wrap(f'"{desc}"', 74)),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, style="italic", color="#777")

        ax = axes[1][c]
        g = kin[kin.scene_name == r.scene_name].sort_values("t_rel_s")
        ax.plot(g.t_rel_s, g.heading_change_deg, color="#2ca02c", lw=2)
        ax.axvline(r.t_rel_s, color="#d62728", ls="--", lw=2.2)
        for s in (+MIN_TURN_DEG, -MIN_TURN_DEG):
            ax.axhline(s, color="k", ls=":", lw=1.0)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlabel("t (s)", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"heading change over {WINDOW_S:.0f} s (deg)", fontsize=10)

    fig.suptitle(
        "C1 verification - heading change vs. the camera view.  Panel title = what is "
        "ACTUALLY visible\nin this frame (frustum-filtered).  Grey italic = nuScenes' "
        "whole-scene text, which describes\nall six cameras over 20 s and does NOT apply "
        "to a single frame (F-025).\n"
        f"Dotted lines = C2's {MIN_TURN_DEG:.0f} deg turn threshold; red dashed = this "
        "keyframe.",
        fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=7.0)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def build_c1_figures(split: str = "mini", dataroot: str = "data/nuscenes",
                     ) -> tuple[str, str]:
    """Both C1 figures, from the cached parquet. One call, no glue script (R30)."""
    import pandas as pd

    from .data import load_nusc

    kin = pd.read_parquet(f"outputs/ego_kinematics_{split}.parquet")
    a = c1_kinematics_figure(kin)
    b = c1_turn_crosscheck_figure(load_nusc(f"v1.0-{split}", dataroot), kin)
    return a, b


# =====================================================================================
# C2 - manoeuvre events
# =====================================================================================


def c2_timeline_scenes(events: Any, n: int = N_TIMELINE_SCENES) -> list[str]:
    """Scenes for the timeline panels: a greedy cover over EVENT KINDS, then fill.

    R35 again - stratify on the thing being checked. The kinds at risk of being wrong
    are the rare ones, and drawing the four busiest scenes resamples the common case.
    The cover guarantees every kind the detector emits appears at least once; the
    remaining panels are filled by event count, so they also show the detector staying
    quiet where nothing happens.

    Deterministic: the greedy key and the fill key both end in the scene name.
    """
    by_scene = {s: set(g.kind) for s, g in events.groupby("scene_name")}
    n_ev = events.groupby("scene_name").size().to_dict()
    need, picks = set(events.kind), []
    while need and len(picks) < n:
        best = max(by_scene, key=lambda s: (len(by_scene[s] & need), n_ev[s], s))
        if not (by_scene[best] & need):
            break
        picks.append(best)
        need -= by_scene[best]
        by_scene.pop(best)
    rest = sorted(by_scene, key=lambda s: (-n_ev[s], s))
    picks += rest[:n - len(picks)]
    return sorted(picks)


def c2_event_timeline_figure(nusc: Any, kin: Any, events: Any,
                             out_path: str = "outputs/c2_event_timeline.png",
                             n: int = N_TIMELINE_SCENES) -> str:
    """Detected events drawn over the dense 50 Hz signals they were detected on. C2.

    The figure that carries F-015: the right column plots the RAW acceleration in grey
    behind the 1.5 s TREND acceleration in cyan, and the raw trace swings +/-5 m/s2
    around a true mean of -0.4 while flipping sign several times a second. Detecting on
    it found no "arrive at the junction and stop" deceleration at all. The shaded bands
    are the events the trend signal produces.

    SCOPE (R24): black lines are the dense ~50 Hz resampled pose track, red dots are the
    2 Hz keyframes read back from the cached C1 parquet - so a dot off the line is a
    disagreement between the table and its own source, and visible as one (R9).

    The dense signals are the one C2 quantity no parquet caches, so this builder does
    need the devkit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import (ACCEL_ENTER_MPS2, TREND_S, YAW_ENTER_DEG_S,
                              scene_pose_track)

    unknown = set(events.kind) - set(_EVENT_COLOURS)
    assert not unknown, f"no colour for event kind(s) {sorted(unknown)} - see R4/R36"

    names = c2_timeline_scenes(events, n)
    by_name = {s["name"]: s for s in nusc.scene}
    fig, axes = plt.subplots(len(names), 2, figsize=(19.5, 3.9 * len(names)))
    axes = np.atleast_2d(axes)

    def _bands(ax, evs):
        # Labels alternate between two heights: longitudinal events overlap in time
        # (a stop begins while the deceleration that caused it is still running), and
        # two labels on one line render as one unreadable word.
        lo, hi = ax.get_ylim()
        for i, e in enumerate(evs.itertuples()):
            col = _EVENT_COLOURS[e.kind]
            ax.axvspan(e.start_s, e.end_s, color=col, alpha=0.22, lw=0, zorder=0)
            ax.text((e.start_s + e.end_s) / 2, hi - (0.05 + 0.10 * (i % 2)) * (hi - lo),
                    e.kind, ha="center", va="top", fontsize=9, weight="bold", color=col,
                    zorder=6,
                    # the band's midpoint is often exactly the signal's peak, which is
                    # what makes it a manoeuvre - so the label needs its own ground
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
        ax.set_ylim(lo, hi)

    for row, name in enumerate(names):
        track = scene_pose_track(nusc, by_name[name])
        t = track["t_s"]
        ev = events[events.scene_name == name]
        kf = kin[kin.scene_name == name].sort_values("t_rel_s")

        # --- left: STEERING, detected on yaw rate -----------------------------------
        ax = axes[row][0]
        ax.plot(t, track["yaw_rate_deg_s"], "k-", lw=1.2)
        ax.plot(kf.t_rel_s, kf.yaw_rate_deg_s, "o", ms=4, color="#d62728", zorder=4)
        for s in (+YAW_ENTER_DEG_S, -YAW_ENTER_DEG_S):
            ax.axhline(s, color="#555", ls=":", lw=0.9)
        _bands(ax, ev[ev.axis == "lateral"])
        ax.set_ylabel(f"{name}\nyaw rate (deg/s)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        if row == 0:
            ax.set_title("STEERING axis   (dotted = "
                         f"YAW_ENTER_DEG_S = {YAW_ENTER_DEG_S:.0f} deg/s)",
                         fontsize=12, weight="bold")

        # --- right: SPEED, detected on the TREND acceleration -----------------------
        ax = axes[row][1]
        ax.plot(t, track["speed_mps"], "k-", lw=1.2)
        ax.plot(kf.t_rel_s, kf.speed_mps, "o", ms=4, color="#d62728", zorder=4)
        ax.set_ylabel("speed (m/s)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax2 = ax.twinx()
        ax2.plot(t, track["accel_mps2"], "-", lw=0.7, color="#c0c0c0", zorder=1)
        ax2.plot(t, track["accel_trend_mps2"], "-", lw=1.3, color="#17becf", zorder=2)
        for s in (+ACCEL_ENTER_MPS2, -ACCEL_ENTER_MPS2):
            ax2.axhline(s, color="#17becf", ls=":", lw=0.9)
        ax2.set_ylabel("accel (m/s2)", fontsize=9, color="#17becf")
        ax2.tick_params(axis="y", labelcolor="#17becf", labelsize=8)
        # twinx draws entirely above its parent, which would bury the speed trace and
        # the keyframe dots under the noisy raw-accel line. Lift the speed axes back on
        # top and make its background transparent so ax2 still shows through.
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        _bands(ax, ev[ev.axis == "longitudinal"])
        if row == 0:
            ax.set_title(f"SPEED axis   (cyan = {TREND_S} s trend accel, grey = raw "
                         f"accel; dotted = {ACCEL_ENTER_MPS2} m/s2)",
                         fontsize=12, weight="bold")
        if row == len(names) - 1:
            axes[row][0].set_xlabel("t (s)", fontsize=10)
            ax.set_xlabel("t (s)", fontsize=10)

    fig.suptitle(
        "C2 verification - manoeuvre events over the raw 50 Hz signals.  "
        "Black = dense pose track, red dots = the 2 Hz keyframes in the cached C1 table.\n"
        "Scenes chosen as a greedy cover over event kinds, so every kind the detector "
        "emits appears at least once (R35).",
        fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.4)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _spread_over_scenes(pool: Any, n: int, rng: Any) -> Any:
    """`n` rows drawn from `pool`, preferring one per scene, then topping up. R35.

    A plain random draw inside one label happily returns two CONSECUTIVE keyframes of
    the same turn - 0.5 s apart and visually identical - which verifies one manoeuvre
    twice instead of two manoeuvres once. Where a label lives in a single scene
    (`turn_right` is entirely scene-0916 in mini) the top-up keeps the row full rather
    than silently short, for the same reason `c4_verification_frames` falls back rather
    than skipping: a probe that returns nothing is a check that did not run.
    """
    first, rest, seen = [], [], set()
    for i in rng.permutation(len(pool)):
        scene = pool.iloc[int(i)].scene_name
        (rest if scene in seen else first).append(int(i))
        seen.add(scene)
    return pool.iloc[sorted((first + rest)[:n])]


def c2_class_examples_figure(nusc: Any, man: Any,
                             out_path: str = "outputs/c2_class_examples.png",
                             n_per_label: int = N_CLASS_EXAMPLES,
                             seed: int = CLASS_EXAMPLE_SEED,
                             camera: str = "CAM_FRONT") -> str:
    """Random camera frames per manoeuvre label. Step C2 verification.

    SCOPE, printed on the figure (R24): these are EGO-MOTION labels. They describe what
    the car is doing, not what is in the picture, so a `turn_left` frame showing a
    straight road ahead is not a wrong label - the car is mid-turn and the camera points
    where it will be. Reading these rows as image content is the F-025 mistake in a
    different costume.

    Rows are generated from the label vocabulary constants, never hand-written (R4,
    the F-016 lesson), and labels with no frames in the split are named in the caption
    rather than silently omitted (R36: a coverage gap must be visible).

    The draw is random, so it is seeded from a frozen constant and the seed is printed
    on the figure. A figure that cannot be regenerated cannot be audited.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from .groundtruth import LATERAL_LABELS, LONGITUDINAL_LABELS

    rng = np.random.default_rng(seed)
    labels = list(LATERAL_LABELS) + list(LONGITUDINAL_LABELS)
    axis_of = {**{v: "lateral_label" for v in LATERAL_LABELS},
               **{v: "longitudinal_label" for v in LONGITUDINAL_LABELS}}

    rows, absent = [], []
    for lab in labels:
        pool = man[man[axis_of[lab]] == lab]
        if not len(pool):
            absent.append(lab)
            continue
        rows.append((lab, _spread_over_scenes(pool, n_per_label, rng)))

    fig, axes = plt.subplots(len(rows), n_per_label,
                             figsize=(4.9 * n_per_label, 3.05 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (lab, sel) in enumerate(rows):
        for c in range(n_per_label):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(sel):
                ax.axis("off")
                continue
            row = sel.iloc[c]
            cam = nusc.get("sample", row.sample_token)["data"][camera]
            ax.imshow(Image.open(nusc.get_sample_data_path(cam)))
            ax.set_title(f"{row.scene_name} kf{row.keyframe_index}", fontsize=9)
            if c == 0:
                ax.set_ylabel(lab, fontsize=12, weight="bold")

    gap = (f"   No frames in this split for: {', '.join(absent)}." if absent else "")
    fig.suptitle(
        f"C2 verification - {n_per_label} random {camera} frames per manoeuvre label "
        f"(seed {seed}, frozen in src/figures.py)\n"
        "NOTE: these are EGO-MOTION labels. They describe what the car is DOING, not "
        "what is in the image." + gap,
        fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def build_c2_figures(split: str = "mini", dataroot: str = "data/nuscenes",
                     ) -> tuple[str, str]:
    """Both C2 figures, from the cached parquets. One call, no glue script (R30)."""
    import pandas as pd

    from .data import load_nusc

    nusc = load_nusc(f"v1.0-{split}", dataroot)
    kin = pd.read_parquet(f"outputs/ego_kinematics_{split}.parquet")
    events = pd.read_parquet(f"outputs/maneuver_events_{split}.parquet")
    man = pd.read_parquet(f"outputs/maneuvers_{split}.parquet")
    return (c2_event_timeline_figure(nusc, kin, events),
            c2_class_examples_figure(nusc, man))


# =====================================================================================
# C3 - roundabouts on the HD map
# =====================================================================================


def _draw_map_patch(ax: Any, index: Any, cx: float, cy: float, half_m: float) -> None:
    """Paint the HD map layers around (cx, cy) into an existing axes.

    Draws straight from `MapIndex`'s shapely polygons and STRtree, which C3 has already
    built for its own queries, rather than from the devkit's `render_map_patch` - that
    one creates its own Figure and cannot draw into a subplot, and its polygon patches
    go through the unmaintained `descartes` package.

    R36: the layer list is exactly `_MAP_LAYER_STYLE`, i.e. exactly what `MapIndex`
    indexes. `road_block` is absent here because D-019 excludes it from the pipeline, so
    the picture cannot show a layer no label is ever derived from. Non-drivable ground -
    islands included - is left as the axes background, which is what makes a hole in the
    drivable area read as an island.
    """
    from shapely.geometry import box

    from . import groundtruth as _G

    patch = box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    ax.set_facecolor(_MAP_BACKGROUND)
    for layer, colour in _MAP_LAYER_STYLE:
        tree, _, arr = index.trees[layer]
        if tree is None:
            continue
        for i in tree.query(patch, predicate="intersects"):
            clipped = _G.clip_to_patch(arr[int(i)], patch)
            if clipped.is_empty:
                continue
            parts = (clipped.geoms if clipped.geom_type.startswith("Multi")
                     else [clipped])
            for p in parts:
                if p.geom_type != "Polygon":
                    continue
                ax.fill(*p.exterior.xy, fc=colour, ec=_MAP_EDGE, lw=0.4, zorder=1)
                # punch the holes back out, or an island vanishes under its own road
                for ring in p.interiors:
                    ax.fill(*ring.xy, fc=_MAP_BACKGROUND, ec=_MAP_EDGE, lw=0.4, zorder=1.1)
    ax.set_xlim(cx - half_m, cx + half_m)
    ax.set_ylim(cy - half_m, cy + half_m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _draw_lane_arrows(ax: Any, centrelines: dict, cx: float, cy: float, half_m: float,
                      stride: int = 9) -> None:
    """Directed lane centrelines as arrows. The DIRECTION is the whole point (F-027).

    A median strip and a roundabout island are both fully enclosed by road; what
    separates them is that traffic may legally circulate one and not the other, and that
    is only visible if the lanes are drawn with their direction on.
    """
    px, py, dx, dy = [], [], [], []
    for pts in centrelines.values():
        P = np.asarray(pts)[:, :2]
        keep = (np.abs(P[:, 0] - cx) < half_m) & (np.abs(P[:, 1] - cy) < half_m)
        if keep.sum() < 2:
            continue
        ax.plot(P[keep, 0], P[keep, 1], "-", color="#4a7fb5", lw=0.8, alpha=0.9, zorder=2)
        Q = P[keep]
        for j in range(0, len(Q) - 1, stride):
            px.append(Q[j, 0]); py.append(Q[j, 1])
            dx.append(Q[j + 1, 0] - Q[j, 0]); dy.append(Q[j + 1, 1] - Q[j, 1])
    if px:
        ax.quiver(px, py, dx, dy, color="#2f6ba8", angles="xy", scale_units="xy",
                  scale=0.35, width=0.006, headwidth=4.5, zorder=3)


def c3_roundabout_islands_figure(traversals: Any, kin: Any,
                                 out_path: str = "outputs/c3_roundabout_islands.png",
                                 dataroot: str = "data/nuscenes") -> str:
    """Every roundabout CANDIDATE on the HD map, with its lane-circuit verdict. Step C3.

    SCOPE (R24): one panel per candidate TRAVERSAL, not per scene and not per island.
    The red track is the ego's whole scene; the star is the enclosed island it swept
    around. The verdict on each panel is the traffic-rules criterion of F-027/F-028 -
    the maximum winding reachable along the DIRECTED lane graph - and not the
    trajectory shape, which cannot separate a roundabout from a U-turn (F-024).

    R36: panels are coloured by `is_roundabout`, the column the table actually carries.
    A candidate that fails is drawn as a failed candidate, not left unmarked.

    Reads `roundabout_traversals.parquet` and the cached C1 table; the map index is
    built here because no parquet can hold polygons.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import (ROUNDABOUT_MIN_WINDING_DEG, MapIndex, lane_centrelines)

    tr = traversals.sort_values(["is_roundabout", "lane_winding_deg"],
                                ascending=[False, False]).reset_index(drop=True)
    fig, axes = plt.subplots(1, len(tr), figsize=(4.6 * len(tr), 5.9))
    axes = np.atleast_1d(axes)

    cache: dict[str, tuple[Any, dict]] = {}
    for ax, r in zip(axes, tr.itertuples()):
        if r.location not in cache:
            idx = MapIndex(dataroot, r.location)
            cache[r.location] = (idx, lane_centrelines(idx.nusc_map))
        index, cls = cache[r.location]

        _draw_map_patch(ax, index, r.island_x, r.island_y, ROUNDABOUT_PATCH_HALF_M)
        _draw_lane_arrows(ax, cls, r.island_x, r.island_y, ROUNDABOUT_PATCH_HALF_M)
        g = kin[kin.scene_name == r.scene_name].sort_values("keyframe_index")
        ax.plot(g.ego_x, g.ego_y, "-", color="#e8452b", lw=2.6, zorder=5)
        ax.plot(r.island_x, r.island_y, "*", ms=17, color="#7b2fbe", zorder=6)

        ok = bool(r.is_roundabout)
        colour = "#1a7f37" if ok else "#a4243b"
        for sp in ax.spines.values():
            sp.set_color(colour); sp.set_linewidth(2.6)
        ax.set_title(f"{r.scene_name}   island {r.island_area_m2:.0f} m2\n"
                     f"reachable winding {r.lane_winding_deg:.0f} deg over "
                     f"{r.lane_path_lanes} lanes\n"
                     f"{'ROUNDABOUT' if ok else 'not circulatory'}",
                     fontsize=10, weight="bold", color=colour)

    n_ok = int(tr.is_roundabout.sum())
    n_scenes = int(kin.scene_name.nunique())
    scores = ", ".join(f"{v:.0f}" for v in sorted(tr.lane_winding_deg, reverse=True))
    fig.suptitle(
        f"C3 - roundabouts in nuScenes: {n_ok} of {n_scenes} scenes "
        f"({100 * n_ok / n_scenes:.2f}%).  Blue arrows = DIRECTED lanes.\n"
        "Test (F-027/F-028): how far around the island can a vehicle legally travel?  "
        f"Candidates score {scores} deg - "
        f"a genuine gap at the {ROUNDABOUT_MIN_WINDING_DEG:.0f} deg threshold.\n"
        f"Red = the ego's whole scene track, star = the enclosed island;  every panel is "
        f"a {2 * ROUNDABOUT_PATCH_HALF_M:.0f} x {2 * ROUNDABOUT_PATCH_HALF_M:.0f} m map "
        "patch, so island sizes are comparable across panels.",
        fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def c3_superseded_circle_figure(events: Any, kin: Any,
                                out_path: str = "outputs/c3_roundabout_check.png",
                                dataroot: str = "data/nuscenes") -> str:
    """SUPERSEDED v2 method, kept as the picture behind F-024/F-026 and R27.

    What it shows: the three C2 `u_turn` events, each with the PERFECT CIRCLE the v2
    detector assumed at the driving radius. Two findings live in this one picture:
      F-024  a U-turn and a roundabout traversal have the same trajectory, so the v1
             shape detector re-found a U-turn and called it a roundabout;
      F-026  real islands are oval, so a circle at the driving radius cuts across them
             and the v2 ring test became a FALSE-NEGATIVE mechanism - it rejected
             scene-0185, which the current topological method confirms as a roundabout.

    HONEST LIMIT, and the reason this docstring is long. The v2 detector's code was
    deleted from `src/` when it was superseded, so this builder is a RECONSTRUCTION, not
    a rerun: everything except the circle comes from live tables (`maneuver_events_*`
    for the events, the cached C1 table for the track, the map for the background), but
    the circle is refitted here by least squares to the keyframe track. The radii it
    prints therefore differ from the ones on the original 2026-08-09 PNG (17.4 / 5.2 /
    29.8 m), which came from the dense 50 Hz track under code that no longer exists. The
    figure's CLAIM - that a circle does not fit an oval island - is what it evidences,
    and that is visible rather than numeric.

    Current roundabout results are in `c3_roundabout_islands.png`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .groundtruth import MapIndex

    u = events[events.kind == "u_turn"].sort_values("scene_name").reset_index(drop=True)
    fig, axes = plt.subplots(1, len(u), figsize=(6.2 * len(u), 6.6))
    axes = np.atleast_1d(axes)

    cache: dict[str, Any] = {}
    for ax, e in zip(axes, u.itertuples()):
        g = kin[(kin.scene_name == e.scene_name)
                & kin.t_rel_s.between(e.start_s, e.end_s)].sort_values("t_rel_s")
        x, y = g.ego_x.to_numpy(float), g.ego_y.to_numpy(float)
        cx, cy, radius = _fit_circle(x, y)

        if e.location not in cache:
            cache[e.location] = MapIndex(dataroot, e.location)
        half = max(UTURN_PATCH_MIN_HALF_M,
                   UTURN_PATCH_MARGIN * max(np.ptp(x), np.ptp(y)) / 2)
        _draw_map_patch(ax, cache[e.location],
                        float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2),
                        half)

        th = np.linspace(0, 2 * np.pi, 240)
        ax.plot(cx + radius * np.cos(th), cy + radius * np.sin(th), "--",
                color="#7b2fbe", lw=1.8, zorder=4, label="assumed perfect circle")
        ax.plot(cx, cy, "x", ms=13, mew=3, color="#7b2fbe", zorder=5)
        ax.plot(x, y, "-", color="#e8452b", lw=3.0, zorder=5, label="the manoeuvre")
        ax.plot(x[0], y[0], "o", ms=11, color="#2ecc40", mec="k", zorder=6)
        ax.plot(x[-1], y[-1], "s", ms=11, color="#e8452b", mec="k", zorder=6)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        ax.set_title(f"{e.scene_name}   {e.total_heading_deg:+.0f} deg   "
                     f"fitted R = {radius:.1f} m", fontsize=11, weight="bold")

    fig.suptitle(
        "SUPERSEDED METHOD - kept as evidence for F-024 / F-026 (and for R27).\n"
        "These are the three C2 u_turn events. The purple circle is the perfect-circle "
        "assumption the v2 detector used;\nit cuts across OVAL islands and so produced "
        "FALSE NEGATIVES - it rejected scene-0185, a real roundabout.\n"
        "RECONSTRUCTION: the v2 code no longer exists, so the circle is refitted by "
        "least squares and its radius differs from the\noriginal figure's. Current "
        "roundabout results are in c3_roundabout_islands.png (topological method).\n"
        "Green dot = start of the manoeuvre, red square = end;  panels are scaled "
        "INDEPENDENTLY - the three radii differ 5x.",
        fontsize=11.5, weight="bold", color="#8b1a1a")
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return out_path


def _fit_circle(x: Any, y: Any) -> tuple[float, float, float]:
    """Least-squares circle through a track. Returns (cx, cy, radius).

    The algebraic (Kasa) fit: minimising |p|^2 - 2c.p - k is linear in (cx, cy, k), so
    it is one lstsq and has no starting guess to get wrong. It biases the radius upward
    on short arcs, which is irrelevant here - the figure's point is that NO circle fits
    an oval island, and a biased fit only understates that.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    return cx, cy, float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))


def build_c3_figures(dataroot: str = "data/nuscenes") -> tuple[str, str]:
    """Both C3 map figures, from the cached parquets. One call, no glue script (R30)."""
    import pandas as pd

    kin = pd.read_parquet("outputs/ego_kinematics_trainval.parquet")
    tr = pd.read_parquet("outputs/roundabout_traversals.parquet")
    ev = pd.read_parquet("outputs/maneuver_events_trainval.parquet")
    return (c3_roundabout_islands_figure(tr, kin, dataroot=dataroot),
            c3_superseded_circle_figure(ev, kin, dataroot=dataroot))


# ---------------------------------------------------------------------------
# Stage D verification (PLAN.md D2/D3): the camera beside the two BEV arms.
# ---------------------------------------------------------------------------

D_VERIFY_SEED = 20260906       # frozen so the panel regenerates identically (R30)


def d_verification_frames(jsonl_path: str = "outputs/bev_symbolic.jsonl",
                          gt_path: str = "outputs/gt_all.parquet",
                          n: int = 8) -> list[dict]:
    """Pick n frames that each EXERCISE A DIFFERENT SITUATION, one per scene.

    R35: sorting by object count returns the same busy junction eight times, which
    verifies almost nothing. Each slot below targets a distinct condition the BEV has
    to represent, and the rarest are picked first so they cannot be crowded out.
    """
    import json

    import pandas as pd

    rows = {json.loads(l)["sample_token"]: json.loads(l)
            for l in open(jsonl_path)}
    gt = pd.read_parquet(gt_path).set_index("sample_token")
    gt = gt.loc[[t for t in rows if t in gt.index]]

    # (label, predicate) - rarest condition first so it gets a slot at all.
    wants = [
        ("cyclist present",        lambda r: r["cyclist"]),
        ("pedestrian crossing",    lambda r: r["pedestrian_crossing_path"]),
        ("lead vehicle braking",   lambda r: r["lead_braking"]),
        ("turning",                lambda r: r["is_turn_left"] or r["is_turn_right"]),
        ("at intersection",        lambda r: r["at_intersection"]),
        ("construction objects",   lambda r: r["construction_object"]),
        ("stationary in traffic",  lambda r: r["is_stationary"]),
        ("no pedestrians",         lambda r: not r["has_pedestrian"]),
    ]
    rng = np.random.default_rng(D_VERIFY_SEED)
    picked, used_scenes = [], set()
    for label, pred in wants[:n]:
        cand = [t for t, r in gt.iterrows()
                if pred(r) and rows[t]["scene_name"] not in used_scenes]
        if not cand:
            continue
        tok = cand[int(rng.integers(len(cand)))]
        used_scenes.add(rows[tok]["scene_name"])
        picked.append({"sample_token": tok, "why": label, **rows[tok]})
    return picked


def d_verification_figure(nusc: Any, frames: list[dict],
                          out_path: str = "outputs/d_bev_verification.png") -> str:
    """Camera frame beside its BEV raster, for PLAN.md's D2/D3 visual check.

    Scope is stated on every panel (R24): the camera shows a ~64 deg wedge of one
    instant, the BEV shows 360 deg at 100 m. Putting them side by side without saying
    so is what made the C1 figure misleading (F-025).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(frames)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for i, f in enumerate(frames):
        s = nusc.get("sample", f["sample_token"])
        cam = nusc.get_sample_data_path(s["data"]["CAM_FRONT"])
        axes[i][0].imshow(mpimg.imread(cam)); axes[i][0].axis("off")
        axes[i][0].set_title(f"{f['scene_name']}  CAM_FRONT (~64 deg of one instant)\n"
                             f"picked to exercise: {f['why']}",
                             fontsize=8, loc="left")
        axes[i][1].imshow(mpimg.imread(f["bev_png"])); axes[i][1].axis("off")
        axes[i][1].set_title(f"BEV 360 deg, 100 m radius\n"
                             f"{f['n_agents']} objects, {f['n_in_camera']} in camera",
                             fontsize=8, loc="left")
    fig.suptitle("Stage D verification - the same instant as the driver sees it and as the "
                 "BEV draws it.\nThe BEV covers 360 deg, so it legitimately shows objects "
                 "the camera cannot; the dashed rays mark the camera's wedge.",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def g1_storyboard_figure(scene_name: str | None = None,
                         out_path: str = "outputs/g1_storyboard.png") -> str:
    """One scene as a storyboard strip: thumbnail, time range, description. Step G1.

    SCOPE, because this figure is the easiest in the repo to misread (R24, and F-025 is the
    defect it repeats if unlabelled). Each panel shows ONE CAM_FRONT frame — a 70-degree
    view of a single instant — but its caption describes the EGO'S MOTION OVER THE WHOLE TIME
    RANGE, derived from pose, not from that picture. A reader who takes the sentence as a
    description of the thumbnail will think the labels are wrong, which is exactly what
    happened when a 360-degree/20-second scene description was printed under one frame.

    The scene defaults to a greedy cover over event KINDS (R35) rather than the busiest
    scene: only 3 of 850 scenes contain a u_turn, so any count-ranked pick would never show
    one.
    """
    import textwrap

    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    from . import data as D
    from . import storyboard as S

    if scene_name is None:
        events = pd.read_parquet("outputs/maneuver_events_trainval.parquet")
        scene_name = c2_timeline_scenes(events, 1)[0]

    panels = S.scene_panels(scene_name)
    n = len(panels)
    # Height is set by the longest caption, not by a constant: the first render left ~25% of
    # the canvas empty above the strip (found by reading the PNG, R8).
    longest = max(len(textwrap.fill(r.description, 46).splitlines())
                  for r in panels.itertuples())
    # +1.1 in reserves a header band for the three title lines. Fixed-position fig.text and
    # tight_layout fight otherwise: the first attempt overlapped the title with the scope
    # line AND drew the scene description straight through the captions (R8 again).
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4 + 0.16 * longest), squeeze=False)

    for ax, r in zip(axes[0], panels.itertuples()):
        ax.imshow(Image.open(D.image_for_sample_token(r.thumbnail_token)))
        ax.set_xticks([]); ax.set_yticks([])

        kind = (r.lateral_label if r.lateral_label != "going_straight"
                else r.longitudinal_label if r.longitudinal_label != "cruising" else None)
        colour = _EVENT_COLOURS[kind] if kind else _IDLE_COLOUR
        for spine in ax.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(4)

        head = f"{r.t_start_s:.1f}-{r.t_end_s:.1f} s   kf {r.kf_start}-{r.kf_end}"
        if r.phantom_guard:
            head += "   [edge-guarded]"
        ax.set_title(head, fontsize=10, color=colour, fontweight="bold", pad=6)
        ax.set_xlabel(textwrap.fill(r.description, 46), fontsize=7.2,
                      linespacing=1.35, ha="left", x=0.0, labelpad=8)

    header = 1.0 - 1.1 / fig.get_figheight()          # bottom of the reserved band
    if not _THESIS:  # in the thesis the caption carries the scope note
      fig.text(0.5, 1.0 - 0.30 / fig.get_figheight(),
             f"G1 storyboard — {scene_name}   ({n} panels, "
             f"{panels.t_end_s.max() - panels.t_start_s.min():.1f} s)",
             ha="center", fontsize=13, fontweight="bold")
      fig.text(0.5, 1.0 - 0.58 / fig.get_figheight(),
             "Each caption describes the EGO'S MOTION ACROSS ITS TIME RANGE, derived from "
             "pose and map — not the contents of the single frame shown above it.",
             ha="center", fontsize=8.5, style="italic", color="#444")

    # The human description, marked WHOLE-SCENE. It is a 360-degree/20-second aggregate and
    # is false at frame level (L-007, F-025 measured a 12x over-count), so it sits in the
    # header as a cross-check on the ORDER of manoeuvres — a whole-scene property it can
    # speak to — and never as a caption on any single panel.
    try:
        meta = D._load_meta()
        desc = next((sc["description"] for sc in meta["scene"] if sc["name"] == scene_name), "")
    except Exception:
        desc = ""
    if desc and not _THESIS:
        fig.text(0.5, 1.0 - 0.85 / fig.get_figheight(),
                 f'nuScenes WHOLE-SCENE description (all 6 cameras, 20 s): "{desc}"',
                 ha="center", fontsize=8.5, style="italic", color="#666")

    fig.tight_layout(rect=(0, 0, 1, header))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def h2_traffic_light_figure(out_path: str = "outputs/h2_traffic_light.png") -> str:
    """Mapped traffic lights projected into CAM_FRONT. Step H2's verify criterion.

    SCOPE. The circles mark where the MAP says a light fixture is, projected through the
    camera calibration. They are not detections and they carry NO STATE: nuScenes has no
    signal state anywhere, and the map's `items[].color` field is the fixture's lamp
    inventory, not what is lit (F-004/F-076).

    Eight frames, one per scene (R35), spanning all four maps - because the first version of
    this tag read the `pose` field and returned a silent 0.00% on the two maps whose pose is
    zero-filled. A verification panel drawn only from boston-seaport would have looked
    perfect.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from . import data as D
    from . import signage as S

    df = pd.read_parquet("outputs/h2_traffic_light.parquet")
    hits = df[df.traffic_light_in_view]
    rng = np.random.default_rng(CLASS_EXAMPLE_SEED)
    picks = []
    for loc in sorted(df.location.unique()):                # all four maps, 2 frames each
        pool = hits[hits.location == loc]
        for _, r in pool.sample(min(2, len(pool)), random_state=0).iterrows():
            picks.append(r)
    picks = picks[:8]

    nusc = D.load_nusc("v1.0-trainval")
    cache = {loc: S.traffic_light_poses(loc) for loc in {p.location for p in picks}}
    # row titles overlapped the row above at 7.2 in and the legend sat on the image
    # (found by reading the PNG, R8). Taller rows + explicit hspace fixes both.
    fig, axes = plt.subplots(2, 4, figsize=(22, 8.6))
    for ax, r in zip(axes.ravel(), picks):
        ax.imshow(Image.open(D.image_for_sample_token(r.sample_token)))
        uv, dist = S.project_lights(nusc, r.sample_token, cache[r.location])
        near = dist <= S.TRAFFIC_LIGHT_MAX_M
        ax.scatter(uv[0][near], uv[1][near], s=260, facecolors="none",
                   edgecolors="#00e5ff", linewidths=2.4, label=f"<= {S.TRAFFIC_LIGHT_MAX_M:.0f} m")
        ax.scatter(uv[0][~near], uv[1][~near], s=90, facecolors="none",
                   edgecolors="#ff5252", linewidths=1.4, linestyle=":", label="beyond the cap")
        for x, y, d in zip(uv[0][near], uv[1][near], dist[near]):
            ax.annotate(f"{d:.0f} m", (x, y), color="#00e5ff", fontsize=7.5,
                        xytext=(9, 9), textcoords="offset points")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r.scene_name}  {r.location}\n{int(r.n_in_view)} in view "
                     f"({int(r.n_in_frustum_uncapped)} uncapped)", fontsize=9)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                           ncol=2, fontsize=8, frameon=False)
    fig.suptitle("H2 — mapped traffic lights projected into CAM_FRONT.  Circles are MAP "
                 "POSITIONS, not detections, and carry NO state (F-004).",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.30)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def f6_significance_figure(out_path: str = "outputs/f6_significance.png") -> str:
    """Paired deltas against the front camera, under both definitions of "macro F1".

    WHY TWO PANELS AND NOT ONE. The averaging choice is not a presentational detail here:
    it REVERSES two of the thesis's conclusions (F-083). Mean-of-per-tag F1 says `both`
    and `temporal` lose to a single photograph; F1 of the pooled counts says they are
    indistinguishable from it and nominally above. Drawing only the panel that supports
    the existing claim would be the same defect as R24 -- a true figure that misleads
    about its own scope.

    Each bar is a 95% interval from 2,000 bootstrap resamples OF SCENES, not of frames
    (F-082), and both arms of every comparison are scored on the SAME resample, so the
    pairing survives into the interval.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("outputs/f6_significance.csv")
    order = ["cot", "both", "temporal", "bev_pic", "bev_txt"]
    label = {"cot": "chain-of-thought", "both": "camera + BEV picture",
             "temporal": "5-frame sequence", "bev_pic": "BEV picture only",
             "bev_txt": "BEV text only"}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.5), sharey=True)
    for ax, average, title in zip(axes, ("macro", "micro"),
                                  ("macro F1 = mean of the 34 per-tag F1 scores\n"
                                   "(every tag counts once)",
                                   "micro F1 = F1 of the counts pooled over tags\n"
                                   "(frequent tags dominate)")):
        d = df[df.average == average].set_index("condition").loc[order]
        y = range(len(order))
        for i, (cond, r) in enumerate(d.iterrows()):
            sig = bool(r.significant_05)
            colour = ("#2e7d32" if r.delta > 0 else "#c62828") if sig else "#9e9e9e"
            ax.plot([r.delta_lo, r.delta_hi], [i, i], color=colour, lw=3.2,
                    solid_capstyle="butt")
            ax.plot(r.delta, i, "o", color=colour, ms=7, zorder=3)
            ax.annotate(f"{r.delta:+.3f}  p={r.p_holm:.3f}"
                        + ("" if sig else "  n.s."),
                        (r.delta_hi, i), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=colour)
        ax.axvline(0, color="#333", lw=1.1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([label[c] for c in order], fontsize=9.5)
        ax.set_xlabel("F1 difference from the front camera alone", fontsize=9.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.185, 0.115)
        ax.grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()

    n_sc = int(df.n_clusters.iloc[0])
    ratio = float(df.width_ratio.iloc[0])
    fig.suptitle("F6 — every arm against the front camera, PAIRED and cluster-robust. "
                 f"2,000 bootstrap resamples of the {n_sc} SCENES, Holm-corrected.",
                 fontsize=12.5, fontweight="bold", y=1.005)
    if not _THESIS:
      fig.text(0.5, -0.075,
             "Grey = not significant at 0.05 after Holm correction. The two panels "
             "disagree about `camera + BEV` and `5-frame sequence`: both lose under "
             "macro and are indistinguishable from the camera under micro (F-083).\n"
             f"Resampling scenes rather than frames widens the interval by {ratio:.2f}x "
             "— the 628 keyframes are 139 correlated scenes, ICC 0.517, so the "
             "effective sample is ~224 frames (F-082).",
             ha="center", fontsize=8.8, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_path


def g2_segmentation_figure(out_path: str = "outputs/g2_segmentation.png") -> str:
    """Boundary F1 against tolerance, with the chance floor drawn under it. Step G2.

    THE POINT OF THE FIGURE IS THE GREY BAND. Gold boundaries are 0.099 per keyframe, so
    random placement alone scores 0.38 at +/-1 keyframe and 0.67 at +/-3: at a loose
    tolerance this metric is mostly reporting boundary density. An evenly spaced baseline
    given the oracle's own boundary count clears that floor by 0.03-0.09 -- it is barely a
    segmenter at all -- which is the number any future VLM segmentation arm has to beat
    before its score means anything.

    G1's own curve is drawn dashed and labelled NOT A VALIDATION: its panels are maximal
    runs of the labels the gold events are built from, so the curve measures D-050's
    linearisation, not whether a boundary is correct (R21).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("outputs/g2_segmentation.csv")
    style = {
        "g1_panels": ("storyboard panels (same source as the reference, not a validation)",
                      "#1565c0", "--", "o"),
        "fixed_matched_k": ("fixed interval, given the reference boundary count",
                            "#ef6c00", "-", "s"),
        "fixed_period_2s": ("fixed interval, every 2.0 s (no oracle knowledge)",
                            "#6a1b9a", "-", "^"),
        "random_matched_k": ("random placement, same count (chance level)",
                             "#757575", ":", "x"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    chance = df[df.method == "random_matched_k"].sort_values("tolerance_kf")
    ax.fill_between(chance.tolerance_kf, 0, chance.f1, color="#bdbdbd", alpha=0.35,
                    label="_nolegend_", zorder=0)
    for method, (label, colour, ls, marker) in style.items():
        d = df[df.method == method].sort_values("tolerance_kf")
        ax.plot(d.tolerance_kf, d.f1, ls, color=colour, marker=marker, ms=7, lw=2.1,
                label=label, zorder=3)
        if method == "random_matched_k":
            ax.errorbar(d.tolerance_kf, d.f1, yerr=d.f1_sd, fmt="none", ecolor=colour,
                        capsize=4, zorder=3)
    closed = df.drop_duplicates("tolerance_kf").sort_values("tolerance_kf")
    ax.plot(closed.tolerance_kf, closed.chance_f1_closed_form, color="#424242", lw=1.2,
            alpha=0.8, label="chance, closed form  (2·tol + 1) × 0.099 boundaries/keyframe")

    ax.set_xticks(sorted(df.tolerance_kf.unique()))
    ax.set_xlabel("tolerance (keyframes; 1 keyframe = 0.5 s)", fontsize=10)
    ax.set_ylabel("boundary F1, micro-averaged over 850 scenes", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    # Below the axes, not inside them: the first render put the box straight over G1's
    # curve between tolerance 0 and 2, hiding the only line the reader is told to discount
    # (R8/R19).
    ax.legend(fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False)
    if not _THESIS:
      ax.set_title("G2 — event-boundary scoring against C2's 3,368 kinematic boundaries.\n"
                 "The grey area is what random placement already gets.",
                 fontsize=12, fontweight="bold")
    if not _THESIS:
      fig.text(0.5, -0.19,
             "20.3% of consecutive gold boundaries are within 1 keyframe of each other, so "
             "at ±1 a fifth of the reference is not separable by any segmenter. "
             "Read the gap to the grey floor, never the absolute F1.",
             ha="center", fontsize=8.6, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def h3_signage_figure(out_path: str = "outputs/h3_signage.png") -> str:
    """H3's two readable facts: the view instability, and what size explains.

    LEFT. The same model scored on three views of the same 586 images. `yellow_light_visible`
    reads 0.182, 0.540 and 0.451 -- and the raw rows carry **exactly 96 false positives in
    both the uniform and the enriched view**, because every top-up image is a yellow
    positive and can only add a true positive or a false negative. The model did not change;
    the evaluation set's prevalence did. The dashed bars are the MAJORITY-CLASS baseline --
    always-yes where positives are the majority, always-no (so F1 0, and no visible bar)
    where they are not. `traffic_sign_present` is 77-83% positive in every view, so its
    baseline is always-yes and it loses to it in all three (F-068's shape, second dataset).

    RIGHT. State F1 against the pixel height of the largest lit light. Green is flat across
    every band; yellow collapses to 0.125 below 15 px. Yellow's failure is largely a
    small-object failure and red's is not, which is why size is reported and never used as
    a filter (R34 inverted, D-049(b)).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    views = pd.read_csv("outputs/h3_views.csv")
    sizes = pd.read_csv("outputs/h3_size_decomposition.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    order = ["traffic_light_present", "traffic_sign_present",
             "red_light_visible", "yellow_light_visible", "green_light_visible"]
    vorder = ["uniform_450", "all_586", "state_gt_certain_381"]
    vlabel = {"uniform_450": "uniform 450 (real prevalence)",
              "all_586": "all 586 (yellow enriched)",
              "state_gt_certain_381": "381 with a certain reference"}
    colours = ["#1565c0", "#ef6c00", "#2e7d32"]
    x = np.arange(len(order))
    w = 0.26
    for k, (v, c) in enumerate(zip(vorder, colours)):
        d = views[views.view == v].set_index("tag").loc[order]
        ax1.bar(x + (k - 1) * w, d.f1, w, color=c, label=vlabel[v], zorder=3)
        ax1.bar(x + (k - 1) * w, d.baseline_f1, w, facecolor="none", edgecolor="#555",
                linestyle="--", linewidth=1.1, zorder=4,
                label="majority-class baseline" if k == 0 else "_nolegend_")
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace("_", "\n") for t in order], fontsize=8.5)
    ax1.set_ylabel("F1", fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    # Below the axes: inside it, at any corner, the legend covered bar data (R8).
    ax1.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
               frameon=False)
    ax1.set_title("One capability, three views of the same 586 images", fontsize=10.5,
                  fontweight="bold")
    # The annotation has to point at BOTH yellow bars, because the claim is that these two
    # numbers come from the same 96 errors. The first version put one arrow on the enriched
    # bar alone and ran its text through the red group -- which said the opposite of what is
    # meant and was only visible on reading the render (R8/R19).
    yi = order.index("yellow_light_visible")
    if not _THESIS:  # in the thesis the caption states the point
        ax1.text(yi, 0.975, "the SAME 96\nfalse positives", ha="center", va="top",
                 fontsize=8.6, color="#b71c1c", fontweight="bold")
        ax1.text(yi, 0.855, "F1 0.182 vs 0.540:\nthe model did not move,\nthe prevalence did",
                 ha="center", va="top", fontsize=7.6, color="#b71c1c")
        for dx, top in ((-w, 0.182), (0.0, 0.540)):
            ax1.annotate("", xy=(yi + dx, top + 0.015), xytext=(yi + dx * 0.55, 0.715),
                         arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2))
    si = order.index("traffic_sign_present")
    ax1.text(si, 0.955, "loses to always-yes\nin all three views", ha="center",
             fontsize=7.8, color="#444", style="italic")

    smap = {"red_light_visible": ("#c62828", "red"),
            "yellow_light_visible": ("#f9a825", "yellow"),
            "green_light_visible": ("#2e7d32", "green")}
    bands = list(dict.fromkeys(sizes.size_band_px))
    for tag, (c, lab) in smap.items():
        d = sizes[sizes.tag == tag].set_index("size_band_px").loc[bands]
        ax2.plot(range(len(bands)), d.f1, "-o", color=c, lw=2.2, ms=7, label=lab)
    ax2.set_xticks(range(len(bands)))
    ax2.set_xticklabels([b.replace("-", " to ").replace("inf", "max") + " px" for b in bands], fontsize=9)
    ax2.set_xlabel("height of the largest lit light (images with a lit light only)", fontsize=9.5)
    ax2.set_ylabel("F1", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, title="light state", title_fontsize=8.5)
    ax2.set_title(f"Size explains yellow, not red\n"
                  f"(median {sizes.median_lit_px_subset.iloc[0]:.0f} px, "
                  f"{sizes.pct_lit_under_15px.iloc[0]:.0f}% under 15 px)",
                  fontsize=10.5, fontweight="bold")

    fig.suptitle("H3 — signage on BDD100K val, Qwen2.5-VL-7B, 586 images, 0.0% parse failure",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_path


def h4_cross_dataset_figure(out_path: str = "outputs/h4_cross_dataset.png") -> str:
    """H4: the same predictions, three references. Step H4 (F-088).

    THE MIDDLE BARS ARE THE ARGUMENT. They are nuScenes' camera-arm predictions, unchanged,
    scored against a reference whose scope matches what a camera can see instead of the
    30 m x 8 m corridor the tag is defined on. That swap alone closes ~81% of the gap to
    BDD100K, so the gap is a property of the reference and not of the dataset.

    The first version of this figure described the nuScenes tag as a frustum projection
    within 50 m. That is H2's tag, not the scored one; reading the schema before writing the
    finding caught it, and the correction became the result (R5/R16).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv("outputs/h4_cross_dataset.csv")
    colours = ["#6a1b9a", "#ab47bc", "#00838f"]
    labels = ["nuScenes, corridor reference (scored)",
              "nuScenes, same predictions, view reference",
              "BDD100K, human box"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4),
                                  gridspec_kw={"width_ratios": [1, 1.35]})
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    w = 0.27
    for k, (c, lab) in enumerate(zip(colours, labels)):
        r = df.iloc[k]
        vals = [r[m] for m in metrics]
        ax.bar(x + (k - 1) * w, vals, w, color=c, zorder=3, label=lab)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.8, color=c,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["precision", "recall", "F1"], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8.4, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    share = df.share_of_gap_closed_by_scope.iloc[0]
    ax.set_title(f"Swapping ONLY the reference closes {share:.0%} of the gap\n"
                 f"F1 {df.f1.iloc[0]:.3f} -> {df.f1.iloc[1]:.3f}  (BDD100K "
                 f"{df.f1.iloc[2]:.3f})", fontsize=10.5, fontweight="bold")

    ax2.axis("off")
    cols = ["nuScenes\ncorridor", "nuScenes\ncamera view", "BDD100K\nhuman box"]
    rows = [("reference is", *[r.reference for r in df.itertuples()]),
            ("scope", *[r.scope for r in df.itertuples()]),
            ("models occlusion", *["yes" if r.models_occlusion else "no"
                                   for r in df.itertuples()]),
            ("positive prevalence", *[f"{r.prevalence:.1%}" for r in df.itertuples()]),
            # MAJORITY-class, and it says what it predicts: at <50% prevalence the majority
            # is NO and the baseline scores 0 -- calling that "always-yes" is false (R36).
            ("majority-class baseline",
             *[f"{r.baseline_f1:.3f} ({'YES' if r.prevalence >= 0.5 else 'NO'})"
               for r in df.itertuples()]),
            ("false positives", *[str(int(r.fp)) for r in df.itertuples()]),
            ("independent samples",
             *[f"yes, {int(r.n)} videos" if r.independent_samples
               else f"no, {int(r.n)} frames / 139 scenes" for r in df.itertuples()])]
    xs = [0.0, 0.29, 0.54, 0.79]
    # The first row gets extra height: the view reference wraps to four lines and crossed
    # its separator when every row was spaced evenly (R8).
    yy = np.array([0.745] + list(np.linspace(0.575, 0.03, len(rows) - 1)))
    for j, (c, head) in enumerate(zip(colours, cols)):
        ax2.text(xs[j + 1], 0.92, head, fontsize=8.8, fontweight="bold", color=c,
                 va="center")
    for i, row in enumerate(rows):
        ax2.text(xs[0], yy[i], row[0], fontsize=8.6, fontweight="bold", va="center")
        for j, val in enumerate(row[1:]):
            import textwrap
            ax2.text(xs[j + 1], yy[i], textwrap.fill(str(val), 25), fontsize=7.8,
                     va="center", color=colours[j])
        top = 0.855 if i == 0 else (yy[i] + yy[i - 1]) / 2
        ax2.axhline(top, color="#e0e0e0", lw=0.8)
    ax2.set_title("What differs between the three references", fontsize=10.5,
                  fontweight="bold", pad=22)

    fig.suptitle("H4 — the same model, three references. The cross-dataset gap is mostly "
                 "the reference's SCOPE, not the dataset.",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_g4_figure(out_path: str = "outputs/g4_instantiation.png",
                    table_csv: str = "outputs/g4_instantiation.csv") -> str:
    """Step G4: the ego track instantiated from the log and from the pipeline's scenario
    description, over each scene's road surface.

    Drawn in the map frame the on-road test is evaluated in (R38). The first version showed
    MetaDrive's own top-down renders beside the camera; the ego was indistinguishable from
    the other road users there, so two columns up to 184 m apart looked identical (R19).
    The simulator run is `src.sim.build_g4_instantiation` under ./venv-sim; this reads its
    table and recomputes the two tracks with numpy alone, so it builds in ./venv.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from . import sim as S

    table = pd.read_csv(table_csv).set_index("scene_name")
    scenes = list(table.index)
    fig, axes = plt.subplots(2, (len(scenes) + 1) // 2, figsize=(17, 7.6))
    for ax, scene in zip(axes.ravel(), scenes):
        tr = S.g4_tracks(scene)
        L, D = tr["log"], tr["described"]
        pts = np.vstack([L, D])
        (x0, y0), (x1, y1) = pts.min(0) - 25.0, pts.max(0) + 25.0
        for g in getattr(tr["surface"], "geoms", [tr["surface"]]):
            if not g.is_empty:
                ax.fill(*g.exterior.xy, color="#e4e7ea", lw=0)
        ax.plot(L[:, 0], L[:, 1], color="#111111", lw=2.0, label="ego from the log")
        ax.plot(D[:, 0], D[:, 1], color="#d32f2f", lw=1.6, ls="--",
                label="ego from our description")
        ax.plot(*L[0], marker="o", color="#111111", ms=6)
        row = table.loc[scene]
        ax.set_title(f"{scene}\non road: log {row.log_on_road_fraction:.2f}, "
                     f"described {row.described_on_road_fraction:.2f}", fontsize=9)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Ego track instantiated in MetaDrive from the recorded log (black) and from "
                 "the pipeline's scenario description (red), nuScenes mini.\nGrey: drivable "
                 "lane surface of the converted map. Both start at the log's first pose "
                 "(black dot), which the description does not carry.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# Figures placed in the thesis, in order of appearance. The drawn titles are working
# labels (stage codes, notes to self); the LaTeX caption replaces them, so they are
# suppressed here rather than edited in every builder.
COND_LABEL = {"camera": "front camera", "cot": "chain of thought", "bev_pic": "BEV image",
              "bev_txt": "BEV text", "both": "camera and BEV image", "temporal": "five frames"}
FAMILY_ORDER = ("ego_maneuver", "map_context", "visible_objects", "interaction")


def pipeline_overview_figure(out_path: str = "outputs/pipeline_overview.png") -> str:
    """The whole system on one page: what goes in, what is derived, what is measured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    col = {"data": "#e8eef7", "ref": "#e6f4ea", "model": "#fdf0e0", "score": "#f3e8f7",
           "out": "#fff7d6"}

    def box(x, y, w, h, kind, head, body):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=col[kind], ec="#555", lw=1.1))
        ax.text(x + w / 2, y + h - 0.2, head, ha="center", va="top", fontsize=10.5,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6,
                color="#333", linespacing=1.35)

    def arrow(x0, y0, x1, y1, text=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=8,
                    color="#555", style="italic")

    box(0.2, 2.5, 2.6, 1.95, "data", "nuScenes logs",
        "850 scenes, 34,149 keyframes\nego pose at about 150 Hz\nhigh definition map\n"
        "3D object boxes\nfront camera images")
    box(3.6, 2.5, 2.9, 1.95, "ref", "Deterministic reference",
        "37 tags in 4 families\nrules with measured thresholds\nprovenance on every label\n"
        "15 structural implications\nno new human labelling")
    box(0.2, 0.15, 2.6, 1.95, "data", "Model inputs",
        "front camera frame\nrendered BEV image\nBEV as text\ncamera and BEV image\n"
        "five frames, 2 s")
    box(3.6, 0.15, 2.9, 1.95, "model", "Open weight VLM",
        "Qwen2.5-VL 7B, Qwen3-VL 8B\nstructured or chain of\nthought prompt\n"
        "JSON answer for 35 tags\nfree GPU, 628 keyframes")
    box(7.3, 1.2, 2.7, 1.95, "score", "Scoring",
        "per tag precision, recall, F1\nmajority class baseline\ncluster bootstrap\n"
        "over 139 scenes\nreference swap at fixed output")
    box(10.4, 2.5, 2.4, 1.95, "out", "Scenario output",
        "storyboard of timed events\nlogical scenario description\nOpenSCENARIO export\n"
        "simulator replay")
    box(10.4, 0.15, 2.4, 1.95, "data", "BDD100K",
        "human boxes with\ntraffic light state\n586 images\nsignage scored where\n"
        "nuScenes cannot")
    arrow(2.8, 3.5, 3.6, 3.5, "derive")
    arrow(2.8, 1.1, 3.6, 1.1, "prompt")
    arrow(6.5, 2.9, 7.3, 2.6, "reference")
    arrow(6.5, 1.1, 7.3, 1.7, "answers")
    arrow(6.5, 4.1, 10.4, 4.1, "segment and describe")
    arrow(10.4, 1.1, 10.0, 1.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _subset() -> tuple[list[str], list[str]]:
    import json

    from . import prompts as P
    tokens = json.load(open("outputs/vlm_subset_tokens.json"))["sample_tokens"]
    return tokens, P.scoreable_tags(P.load_schema())


def label_example_frames(gt: Any, pred: Any) -> list[tuple[str, str]]:
    """Three frames from three scenes, each chosen to exercise one finding (R35): a turn the
    single frame cannot show, a light outside the corridor, and pedestrians read correctly."""
    from .eval import _scene_of
    scene = _scene_of(list(gt.index))
    rules = [
        ("a turn the frame cannot show",
         (gt.is_turn_left == True) & (pred.is_turn_left == False)),
        ("a light seen but outside the corridor",
         (gt.traffic_light_ahead == False) & (pred.traffic_light_ahead == True)),
        ("pedestrians read correctly",
         (gt.has_pedestrian == True) & (pred.has_pedestrian == True)
         & (gt.pedestrian_near == True) & (pred.pedestrian_near == True)),
    ]
    picks, used = [], set()
    for why, mask in rules:
        tok = next(t for t in gt.index[mask.fillna(False).astype(bool)] if scene[t] not in used)
        used.add(scene[tok]); picks.append((tok, why))
    return picks


def label_examples_figure(out_path: str = "outputs/label_examples.png") -> str:
    """Front camera frames beside the reference and the model's answers, tag by tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data as D
    from .eval import load_gt, load_rows, predictions_frame
    tokens, tags = _subset()
    gt = load_gt(tokens, tags)
    pred = predictions_frame(load_rows("v1_structured"), tokens, tags)
    picks = label_example_frames(gt, pred)

    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 4.3 * len(picks)),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    for (tok, why), (ai, at) in zip(picks, axes):
        ai.imshow(plt.imread(D.image_for_sample_token(tok))); ai.axis("off")
        ai.set_title(why, fontsize=11, loc="left")
        g, p = gt.loc[tok], pred.loc[tok]
        shown = [t for t in tags if g[t] is True or g[t] == True or p[t] == True]
        agree = int(sum(bool(g[t]) == bool(p[t]) for t in tags if p[t] is not None))
        at.axis("off")
        at.text(0, 1.0, f"tag", fontsize=9, fontweight="bold", va="top")
        at.text(0.62, 1.0, "reference", fontsize=9, fontweight="bold", va="top")
        at.text(0.84, 1.0, "model", fontsize=9, fontweight="bold", va="top")
        dy = 0.93 / max(len(shown), 1)
        for i, t in enumerate(shown):
            y = 0.93 - i * dy
            ok = bool(g[t]) == bool(p[t])
            c = "#1b7a3a" if ok else "#c62828"
            at.text(0, y, t, fontsize=8.4, va="top", color=c)
            at.text(0.62, y, "yes" if g[t] else "no", fontsize=8.4, va="top")
            at.text(0.84, y, "yes" if p[t] else "no", fontsize=8.4, va="top", color=c)
        at.text(0, -0.06, f"agrees on {agree} of {len(tags)} tags; tags where both "
                "say no are not listed", fontsize=8.4, style="italic", va="top", color="#444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def input_representations_figure(out_path: str = "outputs/input_representations.png") -> str:
    """One keyframe in every form the model was given it."""
    import json
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from . import data as D
    from .eval import load_gt, load_rows, predictions_frame
    tokens, tags = _subset()
    gt = load_gt(tokens, tags)
    tok = label_example_frames(gt, predictions_frame(load_rows("v1_structured"), tokens,
                                                     tags))[0][0]
    kf = pd.read_parquet("outputs/gt_all.parquet",
                         columns=["sample_token", "scene_name", "keyframe_index"])
    row = kf.set_index("sample_token").loc[tok]
    scene = kf[kf.scene_name == row.scene_name].set_index("keyframe_index").sample_token
    seq = [scene[i] for i in range(row.keyframe_index - 4, row.keyframe_index + 1)]
    bev = next(json.loads(l) for l in open("outputs/bev_symbolic.jsonl")
               if json.loads(l)["sample_token"] == tok)

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 2.1], hspace=0.12, wspace=0.05)
    for i, t in enumerate(seq):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(plt.imread(D.image_for_sample_token(t)))
        ax.axis("off")
        ax.set_title(f"t = {0.5 * (i - 4):+.1f} s" if i < 4 else "t = 0 (labelled)",
                     fontsize=9)
    fig.text(0.5, 0.965, "five frames, half a second apart", ha="center", fontsize=10.5,
             fontweight="bold")
    ax = fig.add_subplot(gs[1, 0:2]); ax.imshow(plt.imread(D.image_for_sample_token(tok)))
    ax.axis("off"); ax.set_title("front camera", fontsize=10.5, fontweight="bold")
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(plt.imread(bev["bev_png"])); ax.axis("off")
    ax.set_title("BEV image", fontsize=10.5, fontweight="bold")
    ax = fig.add_subplot(gs[1, 3:5]); ax.axis("off")
    ax.set_title("BEV text (opening lines)", fontsize=10.5, fontweight="bold")
    lines = [w for l in bev["description"].splitlines()[:16] for w in textwrap.wrap(l, 62) or [""]]
    ax.text(0, 1, "\n".join(lines[:24]), family="monospace", fontsize=7.6, va="top")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def reference_cost_figure(out_path: str = "outputs/reference_cost.png",
                          csv: str = "outputs/f3_reference_cost.csv") -> str:
    """Same 628 model outputs, per tag F1 against each earlier reference and against ours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv(csv)
    groups = [("v1_presence", "annotation presence reference"),
              ("v2_gated", "distance gated reference"),
              ("v3_fair", "reference of the earlier project reports")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for ax, (v, name) in zip(axes, groups):
        g = d[d.variant == v].reset_index(drop=True)
        y = range(len(g))[::-1]
        for yi, r in zip(y, g.itertuples()):
            c = "#1b7a3a" if r.f1_new > r.f1_legacy else "#c62828" if r.f1_new < r.f1_legacy \
                else "#999"
            ax.annotate("", xy=(r.f1_new, yi), xytext=(r.f1_legacy, yi),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
            ax.plot(r.f1_legacy, yi, "o", mfc="white", mec="#555", ms=7)
            ax.plot(r.f1_new, yi, "o", color=c, ms=7)
        ax.set_yticks(list(y)); ax.set_yticklabels(g.new_tag, fontsize=9)
        ax.set_xlim(-0.03, 1.03); ax.grid(axis="x", alpha=0.3)
        ax.set_title(name, fontsize=10)  # macro F1 is quoted unrounded in the caption
        ax.set_xlabel("F1 of the same model outputs")
    axes[0].plot([], [], "o", mfc="white", mec="#555", label="earlier reference")
    axes[0].plot([], [], "o", color="#1b7a3a", label="this reference, higher")
    axes[0].plot([], [], "o", color="#c62828", label="this reference, lower")
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def per_tag_heatmap_figure(out_path: str = "outputs/per_tag_heatmap.png",
                           csv: str = "outputs/f1_per_tag_detail.csv") -> str:
    """Every scored tag under every input condition, with the tag's baseline beside it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from matplotlib.patches import Rectangle

    d = pd.read_csv(csv)
    conds = [c for c in COND_LABEL if c in set(d.condition)]
    f1 = d.pivot(index="tag", columns="condition", values="f1")[conds]
    base = d.drop_duplicates("tag").set_index("tag").baseline_f1
    fam = d.drop_duplicates("tag").set_index("tag").family
    order = [t for f in FAMILY_ORDER for t in fam[fam == f].sort_values().index]
    m = np.column_stack([f1.loc[order].values, base.loc[order].values])

    fig, ax = plt.subplots(figsize=(8.6, 0.3 * len(order) + 1.6))
    ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i, t in enumerate(order):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if v < 0.55 else "black")
            if j < m.shape[1] - 1 and v > base[t]:   # outlined = beats the tag's baseline
                ax.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False,
                                       ec="#ff3d00", lw=1.6))
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([COND_LABEL[c] for c in conds] + ["majority\nbaseline"], fontsize=8.5,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for k, f in enumerate(FAMILY_ORDER):
        n = int((fam == f).sum()); start = sum(int((fam == g).sum()) for g in FAMILY_ORDER[:k])
        if k:
            ax.axhline(start - 0.5, color="white", lw=2.5)
        ax.text(m.shape[1] - 0.35, start + n / 2 - 0.5, family_label(f), rotation=270,
                va="center", fontsize=9, fontweight="bold")
    ax.axvline(m.shape[1] - 1.5, color="white", lw=2.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


FAMILY_COLOUR = {"ego_maneuver": "#7b61ff", "map_context": "#00a0e0",
                 "visible_objects": "#28a745", "interaction": "#ff8c1a"}

# The schema keys are identifiers and their US spelling is frozen with the schema, so the
# rendered name is mapped rather than derived from the key. `key.replace("_", " ")` put
# "ego maneuver" in a dissertation that writes "manoeuvre" everywhere else.
FAMILY_LABEL = {"ego_maneuver": "ego manoeuvre", "map_context": "map context",
                "visible_objects": "visible objects", "interaction": "interaction"}


def family_label(key: str) -> str:
    """The reader-facing name of a label family."""
    return FAMILY_LABEL.get(key, key.replace("_", " "))


def roundabout_camera_figure(out_path: str = "outputs/roundabout_camera.png",
                             n: int = 5, radius_m: float = 30.0) -> str:
    """What the two confirmed roundabouts look like from the car: front frames taken while
    the ego is within `radius_m` of the island, evenly spaced over that stretch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from . import data as D
    tr = pd.read_parquet("outputs/roundabout_traversals.parquet")
    tr = tr[tr.is_roundabout].sort_values("scene_name")
    kin = pd.read_parquet("outputs/ego_kinematics_trainval.parquet",
                          columns=["sample_token", "scene_name", "keyframe_index", "t_rel_s",
                                   "ego_x", "ego_y"])
    fig, axes = plt.subplots(len(tr), n, figsize=(3.1 * n, 2.25 * len(tr)))
    for row_axes, r in zip(np.atleast_2d(axes), tr.itertuples()):
        k = kin[kin.scene_name == r.scene_name].sort_values("keyframe_index")
        near = k[np.hypot(k.ego_x - r.island_x, k.ego_y - r.island_y) < radius_m]
        pick = near.iloc[np.linspace(0, len(near) - 1, n).round().astype(int)]
        for ax, f in zip(row_axes, pick.itertuples()):
            ax.imshow(plt.imread(D.image_for_sample_token(f.sample_token))); ax.axis("off")
            ax.set_title(f"t = {f.t_rel_s:.1f} s", fontsize=9)
        row_axes[0].text(-0.04, 0.5, r.scene_name.replace("scene-", "scene "),
                         transform=row_axes[0].transAxes, rotation=90, ha="right",
                         va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def bev_gallery_frames(gt: Any) -> list[tuple[str, str]]:
    """One frame per situation, each from a different scene (R35)."""
    from .eval import _scene_of
    scene = _scene_of(list(gt.index))
    rules = [("traffic light ahead", gt.traffic_light_ahead == True),
             ("pedestrian crossing the path", gt.pedestrian_crossing_path == True),
             ("turning at a junction", (gt.is_turn_right == True) & (gt.at_intersection == True))]
    picks, used = [], set()
    for why, mask in rules:
        tok = next(t for t in gt.index[mask.fillna(False).astype(bool)] if scene[t] not in used)
        used.add(scene[tok]); picks.append((tok, why))
    return picks


def bev_gallery_figure(out_path: str = "outputs/bev_gallery.png") -> str:
    """The camera frame beside the BEV raster the model was scored on and the repaired one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data as D
    from .eval import load_gt
    tokens, tags = _subset()
    picks = bev_gallery_frames(load_gt(tokens, tags))
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 3.9 * len(picks)),
                             gridspec_kw={"width_ratios": [1.75, 1, 1]})
    heads = ("front camera", "BEV image as scored", "BEV image, repaired")
    for i, ((tok, why), row) in enumerate(zip(picks, axes)):
        for j, (ax, src) in enumerate(zip(row, (D.image_for_sample_token(tok),
                                                f"data/bev/{tok}.png", f"data/bev_r2/{tok}.png"))):
            ax.imshow(plt.imread(src)); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(heads[j], fontsize=11, fontweight="bold")
        row[0].set_ylabel(why, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def model_lift_figure(out_path: str = "outputs/model_lift.png",
                      csv: str = "outputs/f1_per_tag_detail.csv") -> str:
    """What the model adds: F1 minus the tag's majority baseline, plain prompt and chain of
    thought, one row per tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv(csv)
    d = d[d.tag != "traffic_light_ahead"]           # BEV-blind, excluded as everywhere else
    w = d.pivot(index="tag", columns="condition", values="f1")
    base = d.drop_duplicates("tag").set_index("tag")
    lift = pd.DataFrame({"camera": w.camera - base.baseline_f1,
                         "cot": w.cot - base.baseline_f1, "family": base.family})
    lift = lift.sort_values("camera")
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(lift) + 1.4))
    y = range(len(lift))
    ax.barh(list(y), lift.camera, color=[FAMILY_COLOUR[f] for f in lift.family], alpha=0.85)
    ax.scatter(lift.cot, list(y), marker="D", s=26, color="black", zorder=3,
               label="front camera, chain of thought")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(lift.index, fontsize=8.2)
    for lab, f in zip(ax.get_yticklabels(), lift.family):
        lab.set_color(FAMILY_COLOUR[f])
    ax.set_xlabel("F1 minus the tag's majority class baseline")
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h, l = [Patch(fc="white", ec="#333")] + h, ["front camera, structured prompt (bar)"] + l
    h += [Patch(color=c) for c in FAMILY_COLOUR.values()]
    l += [family_label(f) for f in FAMILY_COLOUR]
    ax.legend(h, l, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def model_comparison_figure(out_path: str = "outputs/model_comparison.png",
                            csv: str = "outputs/e10_model_comparison.csv") -> str:
    """Two model generations on the same 150 frames and prompt, tag by tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv(csv, index_col=0)
    d = d[d.index != "traffic_light_ahead"].sort_values("delta")
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(d) + 1.4))
    for yi, (t, r) in enumerate(d.iterrows()):
        c = "#1b7a3a" if r.delta > 0 else "#c62828" if r.delta < 0 else "#aaa"
        ax.plot([r.qwen2_5_vl_7b, r.qwen3_vl_8b], [yi, yi], color=c, lw=2)
        ax.plot(r.qwen2_5_vl_7b, yi, "o", mfc="white", mec="#555", ms=6)
        ax.plot(r.qwen3_vl_8b, yi, "o", color=c, ms=6)
        ax.plot(r.baseline_f1, yi, "|", color="#d32f2f", ms=11, mew=2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=8.2)
    for lab, f in zip(ax.get_yticklabels(), d.family):
        lab.set_color(FAMILY_COLOUR[f])
    ax.set_xlim(-0.03, 1.03); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("F1 on the same 150 frames")
    ax.plot([], [], "o", mfc="white", mec="#555", label="Qwen2.5-VL 7B")
    ax.plot([], [], "o", color="#1b7a3a", label="Qwen3-VL 8B, better")
    ax.plot([], [], "o", color="#c62828", label="Qwen3-VL 8B, worse")
    ax.plot([], [], "|", color="#d32f2f", ms=11, mew=2, label="majority baseline")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def cot_example_figure(out_path: str = "outputs/cot_example.png") -> str:
    """One frame where the reasoning changed the answer: the image, what the model wrote
    before its JSON, and every tag on which the two prompts disagree."""
    import re
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data as D
    from .eval import load_gt, load_rows, predictions_frame
    tokens, tags = _subset()
    gt = load_gt(tokens, tags)
    rows = {r["sample_token"]: r for r in load_rows("v2_cot")}
    plain = predictions_frame(load_rows("v1_structured"), tokens, tags)
    cot = predictions_frame(list(rows.values()), tokens, tags)
    right = lambda p: (p == gt) & p.notna()
    gain = (right(cot).astype(int) - right(plain).astype(int)).sum(axis=1)
    tok = gain.idxmax()                         # the frame the reasoning helped most
    text = re.split(r"json|```", rows[tok]["raw"].split("{")[0], flags=re.I)[0]
    text = re.sub(r"[#*`]+", "", text).strip()
    lines = [w for l in text.splitlines() for w in (textwrap.wrap(l, 70) or [""])][:26]

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 0.9])
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(plt.imread(D.image_for_sample_token(tok)))
    ax.axis("off"); ax.set_title("front camera", fontsize=11, fontweight="bold")
    ax = fig.add_subplot(gs[:, 1]); ax.axis("off")
    ax.set_title("the model's reasoning, before its answer (excerpt)", fontsize=11,
                 fontweight="bold")
    ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.8, va="top")
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    diff = [t for t in tags if plain.loc[tok, t] != cot.loc[tok, t]]
    for x, h in ((0, "tag"), (0.5, "reference"), (0.7, "structured"), (0.88, "reasoning")):
        ax.text(x, 1, h, fontsize=9, fontweight="bold", va="top")
    for i, t in enumerate(diff[:12]):
        y = 0.88 - i * 0.075
        ax.text(0, y, t, fontsize=8.6, va="top")
        ax.text(0.5, y, "yes" if gt.loc[tok, t] else "no", fontsize=8.6, va="top")
        for x, p in ((0.7, plain), (0.88, cot)):
            ok = p.loc[tok, t] == gt.loc[tok, t]
            ax.text(x, y, "yes" if p.loc[tok, t] else "no", fontsize=8.6, va="top",
                    color="#1b7a3a" if ok else "#c62828")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


_THESIS = False  # set only while build_thesis_figures runs
THESIS_FIG_DIR = "thesis/figures"
THESIS_FIGURES = ("pipeline_overview", "input_representations", "c6_label_schema", "c7_coverage", "f4_family_conditions",
                  "f6_significance", "h3_signage", "h4_cross_dataset",
                  "g1_storyboard", "g2_segmentation", "g4_instantiation", "reference_cost",
                  "per_tag_heatmap", "label_examples", "c3_roundabout_islands",
                  "roundabout_camera", "bev_gallery", "model_lift", "model_comparison",
                  "cot_example")


def build_thesis_figures(out_dir: str = THESIS_FIG_DIR) -> list[str]:
    """Render every figure the thesis includes, from the cached tables, without titles."""
    import json
    from pathlib import Path
    from unittest import mock

    import pandas as pd
    from matplotlib.figure import Figure

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    o = lambda name: f"{out_dir}/{name}.png"
    fam = pd.read_csv("outputs/f4_family_conditions.csv", index_col=0)
    global _THESIS
    nothing = lambda *a, **k: None
    _THESIS = True
    try:
      with mock.patch.object(Figure, "suptitle", nothing):
        return [
            pipeline_overview_figure(o("pipeline_overview")),
            input_representations_figure(o("input_representations")),
            reference_cost_figure(o("reference_cost")),
            per_tag_heatmap_figure(o("per_tag_heatmap")),
            label_examples_figure(o("label_examples")),
            roundabout_camera_figure(o("roundabout_camera")),
            bev_gallery_figure(o("bev_gallery")),
            model_lift_figure(o("model_lift")),
            model_comparison_figure(o("model_comparison")),
            cot_example_figure(o("cot_example")),
            c3_roundabout_islands_figure(
                pd.read_parquet("outputs/roundabout_traversals.parquet"),
                pd.read_parquet("outputs/ego_kinematics_trainval.parquet"),
                o("c3_roundabout_islands")),
            c6_schema_figure(json.load(open("outputs/label_schema.json")), o("c6_label_schema")),
            c7_coverage_figure(pd.read_parquet("outputs/gt_all.parquet"),
                               json.load(open("outputs/coverage_report.json")), o("c7_coverage")),
            f4_family_conditions_figure(fam, o("f4_family_conditions")),
            f6_significance_figure(o("f6_significance")),
            h3_signage_figure(o("h3_signage")),
            h4_cross_dataset_figure(o("h4_cross_dataset")),
            g1_storyboard_figure(out_path=o("g1_storyboard")),
            g2_segmentation_figure(o("g2_segmentation")),
            build_g4_figure(o("g4_instantiation")),
        ]
    finally:
        _THESIS = False


if __name__ == "__main__":
    print("\n".join(build_thesis_figures()))
