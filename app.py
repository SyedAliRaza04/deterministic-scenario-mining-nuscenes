"""Pedagogical demonstrator — Step G6, the brief's named deliverable.

The brief asks for "un demonstrateur pedagogique (Streamlit ou Gradio)". This is it: pick a
scene, see what the camera saw, what the BEV showed, what each VLM arm answered, where it was
right and wrong against deterministic ground truth, how the scene segments into a storyboard,
and the structured scenario description that falls out.

NO GPU AND NO DEVKIT. Everything here is read from precomputed artifacts — the five scored
condition files, gt_all.parquet, the storyboard panels, the scenario descriptions and the H2
signage table. The devkit's ~48 s parse would make the app unusable, and `image_for_sample_token`
exists precisely to avoid it (R20).

SCOPE, because this app is the easiest place in the project to mislead someone. The VLM arms
were run on the 628-frame execution subset (139 scenes), so only those scenes can show a
model comparison; the storyboard and scenario description exist for all 850. The app says
which it is showing rather than quietly rendering an empty panel.

Run:  ./venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import src.data as D
import src.eval as E
import src.prompts as P

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="nuScenes auto-labelling — demonstrator", layout="wide")


@st.cache_data(show_spinner=False)
def load_core():
    schema = P.load_schema()
    tags = P.scoreable_tags(schema)
    gt = pd.read_parquet(OUT / "gt_all.parquet").set_index("sample_token")
    panels = pd.read_parquet(OUT / "storyboard_panels.parquet")
    h2 = pd.read_parquet(OUT / "h2_traffic_light.parquet").set_index("sample_token")
    scen = {json.loads(l)["scene"]["name"]: json.loads(l)
            for l in (OUT / "scenario_descriptions.jsonl").open()}
    tokens = json.loads((OUT / "vlm_subset_tokens.json").read_text())["sample_tokens"]
    return schema, tags, gt, panels, h2, scen, tokens


@st.cache_data(show_spinner=False)
def load_predictions(tokens: tuple, tags: tuple):
    """One frame per condition. Cached because each file is ~1.5-2.4 MB of JSONL."""
    out = {}
    for name, stem in E.conditions_available().items():
        try:
            out[name] = E.predictions_frame(E.load_rows(stem), list(tokens), list(tags))
        except FileNotFoundError:
            pass
    return out


@st.cache_data(show_spinner=False)
def load_raw(stem: str):
    return {r["sample_token"]: r for r in E.load_rows(stem)}


schema, tags, gt, panels, h2, scen, subset_tokens = load_core()
preds = load_predictions(tuple(subset_tokens), tuple(tags))
scene_of = dict(zip(gt.index, gt.scene_name))
scenes_with_vlm = sorted({scene_of[t] for t in subset_tokens})

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Scenario auto-labelling")
st.sidebar.caption("IRT SystemX / ECE — VLM auto-labelling on nuScenes")

scene = st.sidebar.selectbox(
    f"Scene ({len(scenes_with_vlm)} with VLM results)", scenes_with_vlm)
scene_tokens = [t for t in subset_tokens if scene_of[t] == scene]
frame_i = st.sidebar.select_slider(
    "Keyframe (only those the VLM was run on)",
    options=list(range(len(scene_tokens))),
    format_func=lambda i: f"{i + 1} of {len(scene_tokens)}")
token = scene_tokens[frame_i]

condition = st.sidebar.selectbox("VLM arm", list(preds), index=0)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Every number here is precomputed. Ground truth is derived deterministically from ego "
    "pose, the HD map and 3D boxes — no human annotation, and every label traceable to "
    "sensor data (D-001)."
)

sc = scen[scene]
st.title(f"{scene} — {sc['scene']['location']}")
st.caption(f"{sc['scene']['duration_s']} s · {sc['scene']['n_keyframes']} keyframes · "
           f"lighting **{sc['scenography']['lighting']}** · "
           f"human description (WHOLE-SCENE, 360°): _{sc['scene']['human_description']}_")

tab_frame, tab_story, tab_scen = st.tabs(
    ["Frame — what the model saw and answered", "Storyboard (G1)", "Scenario description (G3)"])

# ---------------------------------------------------------------- frame
with tab_frame:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CAM_FRONT")
        st.image(str(D.image_for_sample_token(token)), use_container_width=True)
    with c2:
        st.subheader("BEV (rendered from ground truth)")
        bev = ROOT / f"data/bev/{token}.png"
        if bev.exists():
            st.image(str(bev), use_container_width=True)
            st.caption("Rendered FROM the annotations, not perceived from cameras — an AD "
                       "reader assumes the latter (Stage D note).")
        else:
            st.info("No BEV raster for this keyframe.")

    row = h2.loc[token] if token in h2.index else None
    if row is not None:
        st.caption(f"**H2 signage:** {int(row.n_in_view)} mapped traffic light(s) within 50 m "
                   f"project into this frame ({int(row.n_in_frustum_uncapped)} before the "
                   f"perceptibility cap). Position only — nuScenes has no signal STATE (F-004).")

    st.subheader(f"Tags — {condition} vs deterministic ground truth")
    raw = load_raw(E.conditions_available()[condition]).get(token, {})
    pr = preds[condition]
    fam = {t: schema["tags"][t]["family"] for t in tags}
    rows = []
    for t in tags:
        p = pr[t].get(token) if token in pr.index else None
        g = bool(gt.loc[token, t])
        rows.append({
            "tag": t, "family": fam[t], "ground truth": g,
            "model": "—" if p is None else bool(p),
            "verdict": "not answered" if p is None else ("correct" if bool(p) == g else "WRONG"),
        })
    df = pd.DataFrame(rows)
    ok = (df.verdict == "correct").sum()
    na = (df.verdict == "not answered").sum()
    st.markdown(f"**{ok}/{len(df)} correct**, {len(df) - ok - na} wrong, {na} not answered · "
                f"parse status `{raw.get('parse_reason', 'n/a')}` · "
                f"latency {raw.get('latency_s', float('nan')):.1f} s")
    fams = st.multiselect("families", sorted(set(fam.values())), default=sorted(set(fam.values())))
    st.dataframe(df[df.family.isin(fams)], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- storyboard
with tab_story:
    p = panels[panels.scene_name == scene].sort_values("panel_index")
    st.caption("Each caption describes the EGO'S MOTION OVER ITS TIME RANGE, derived from "
               "pose — not the contents of the single frame shown (R24/F-025).")
    for chunk in range(0, len(p), 4):
        cols = st.columns(min(4, len(p) - chunk))
        for col, r in zip(cols, p.iloc[chunk:chunk + 4].itertuples()):
            with col:
                st.image(str(D.image_for_sample_token(r.thumbnail_token)),
                         use_container_width=True)
                st.markdown(f"**{r.t_start_s:.1f}–{r.t_end_s:.1f} s**  ·  kf {r.kf_start}–{r.kf_end}")
                st.caption(r.description)

# ---------------------------------------------------------------- scenario
with tab_scen:
    st.caption("The slides' terminal artifact, feeding MOSAR. A LOGICAL scenario: every "
               "parameter is a [min, max] range, not a point value.")
    a, b = st.columns(2)
    with a:
        st.subheader("Scenography")
        st.json(sc["scenography"], expanded=True)
    with b:
        st.subheader("Parameters")
        st.json(sc["parameters"], expanded=True)
        st.subheader("Timeline")
        st.json(sc["timeline"], expanded=False)
    st.download_button("Download this scenario description (JSON)",
                       json.dumps(sc, indent=2), file_name=f"{scene}_scenario.json",
                       mime="application/json")
