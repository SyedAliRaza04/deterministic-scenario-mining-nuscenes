# Methodology record

The consolidated record behind the pipeline: what was decided, what was measured, and what
remains uncertain. Every number quoted here is reproducible from this repository.

Three identifier series run through the code as well as this document, so that a comment in
`src/` can point at the evidence behind a line:

* **D-nnn** a design decision, in the log in section 4.
* **F-nnn** an empirical finding, indexed in section 5.
* **L-nnn** a limitation or threat to validity, in section 6.

Companion documents: [`README.md`](README.md) for the overview and results,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for how the system fits together,
[`PLAN.md`](PLAN.md) for the stage by stage build, and [`REFERENCES.md`](REFERENCES.md) for
the annotated bibliography.

---

## 1. Project identity

| | |
|---|---|
| **Title** | Deterministic Scenario Mining as an Auditable Reference for Vision Language Model Auto Labelling on nuScenes |
| **Original subject** | Génération de scénarii pour le véhicule autonome |
| **Host** | IRT SystemX, SYNERGIES project, automated vehicle theme |
| **Academic** | ECE Paris, MSc Artificial Intelligence, Data Management and AI |
| **Author** | Ali Raza |
| **Host supervisor** | Yoann Randon, IRT SystemX |
| **Working language** | English |

### 1.1 What the source documents ask for

The brief and the project slides emphasise different things, and the union of the two defines
the scope.

**From the brief.** Manual annotation is the industry bottleneck: costly, slow and error
prone. Build an auto labelling pipeline using large models. Segment and classify
trajectories, signage, and other road users' intentions during specific manoeuvres, named as
highway merges, parallel parking and roundabouts. Support a driver view and a drone or
bird's eye view. The regulatory context is the EU AI Act, which classifies autonomous driving
as high risk and imposes transparency obligations on training data. The deliverable is an
internal exploratory analysis tool or a pedagogical demonstrator.

**From the slides.** Build a scenario database from raw data covering all cases an automated
vehicle may meet. Use models to generate textual descriptions, storyboards, from video, for
automatic generation of new scenarios. The terminal artifact is a structured scenario
description: scenography generation plus parameter generation. The named evaluation criterion
is IoU on tags. The named datasets are nuScenes and BDD100K.

**Four consequences were drawn, and are recorded as decisions D-005, D-006 and D-007.**
Neither document asks for a driving simulator; the endpoint is a structured description.
Video sequencing means temporal event segmentation is part of the assigned task rather than
an optional extra. IoU on tags is a stated criterion and cannot simply be discarded. BDD100K
is explicitly sanctioned as a second dataset.

---

## 2. Research questions

**Overarching question.** Can a pipeline built around a vision language model automatically
label driving scenarios from nuScenes with sufficient reliability to populate a scenario
database, and where are its limits?

The thesis is a **pipeline, evidenced by evaluation** (D-004). The auto labelling pipeline is
the deliverable; the benchmark is the evidence for what it can and cannot do.

### RQ1. How good a reference can deterministic scenario mining provide, and what does a bad reference cost?

Applying deterministic scenario mining to nuScenes sensor metadata, what fraction of a
scenario taxonomy can be labelled with an auditable derivation and no human annotation, where
does the method fail, and by how much does a naive reference distort the measured accuracy of
an auto labeller?

*Restated at D-037.* The original wording asked whether scenario labels can be derived
automatically without human annotation. That question is already settled in the literature
and cannot be claimed: de Gelder et al. (2020) mine scenarios by exactly this two step shape,
nuPlan auto tags 1,282 hours into around 75 scenario types with SQL over atomic primitives,
and Elrofai et al. did scenario identification from traffic data in 2016. What remains open,
and what this work measured, is the quality question and the consequence question.

*Why it matters.* The brief names manual annotation as the bottleneck, so the pipeline needs
a reference it did not pay a human for. But a reference is an instrument, and an instrument
with unknown error makes every downstream number meaningless. The project's own first phase
is the worked example: it hardcoded three of six tags to `False` with the comment that they
could not be derived from static annotations, and marked a parked vehicle present whenever
any vehicle box existed anywhere in the 360 degree set.

*Sub-questions and their answers.*

* **Coverage.** 37 tags, 35 scoreable. `on_walkway` has zero positives and `on_carpark` has
  23, so both are quality control invariants rather than labels (F-039, D-032). Signage is not
  derivable at all (F-004), which is why RQ4 exists.
* **Corroboration.** Derived labels agree with the human free text scene descriptions for ego
  manoeuvres (F-011, F-017). They are **not** usable for object labels, and the mismatch
  reaches a factor of 12 (F-025, L-007).
* **Cost of a bad reference.** The object over-count is **3.51x** (F-029), and the first
  phase's lead vehicle test ran at **12.2% precision** with 56.4% of its hits pointing
  backwards (F-033). On the scoring side, holding model outputs fixed and exchanging only the
  reference moves macro F1 from 0.303 to 0.523 (F-067).
* **Residual researcher freedom.** Deterministic is not the same as objective. Every threshold
  is a choice (L-002, L-011), and one tag, `lane_change`, could not reach the validation bar
  the other 36 cleared (L-022, D-033).

### RQ2. How well do open weight models label driving scenarios from a single camera frame?

Measured per label family and against a majority class baseline. Sub-questions: how much of
performance is attributable to prompting strategy rather than model capability, and how often
does the model fail to emit parseable structured output?

*Answered.* Camera macro F1 **0.389** against a majority baseline of **0.290**; best arm
chain of thought at **0.448** (F-062, F-066, F-068). Parse failure rate **0.03%** over 3,516
model calls, and the single failure is token budget truncation under chain of thought rather
than malformation. Everything temporally defined scores 0.00 from one frame.

### RQ3. Does richer input representation improve labelling, and for which families?

**3a, spatial.** Does a bird's eye view, alone or combined with the front camera, improve
labelling relative to the front camera alone? **3b, temporal.** Does a multi frame sequence
improve labels that are temporally defined?

*Answered, negative in both halves.* Camera 0.389 > both images 0.375 > BEV raster 0.272 >
BEV symbolic 0.244 (F-068). Five real frames leave every motion tag at **0.000 with zero true
positives in 628 frames** (F-073). The bird's eye view wins only where geometry is the whole
question, taking `lane_change` from 0.00 to 0.30.

### RQ4. Can signage labelling be validated at all, given nuScenes' limitations?

nuScenes provides no traffic sign or traffic light annotations and no light state. Can
signage labelling be validated on BDD100K instead, and what does that reveal about single
dataset evaluation of an auto labelling pipeline?

*Answered* (F-088, F-090). Presence reads well on BDD100K; state is uneven and the errors are
colour confusion rather than invention. **81% of the cross dataset gap is the nuScenes
reference's corridor scope**, not model capability.

### RQ5. Can the pipeline produce a storyboard and a structured scenario description?

Can a driving scene be segmented into a timed sequence of events, and can each be expressed
as a structured description complete enough to instantiate the scenario?

*Answered* (F-072, F-075, F-084, F-089, F-095). 3,351 storyboard panels over 850 scenes; 850
structured descriptions; the model arm cannot segment; the description fixes the sequence of
manoeuvres in a simulator but not the trajectory.

---

## 3. Related work, and what is actually new

The field this work sits in is **scenario mining for scenario based validation**, not model
benchmarking. That is the framing the brief implies, since the EU AI Act, Catena-X and MOSAR
are all homologation vocabulary, and it comes with a decade of prior art. Full bibliography
with verified identifiers is in [`REFERENCES.md`](REFERENCES.md).

| Prior work | What it did | Relation |
|---|---|---|
| Elrofai, Worm, Op den Camp (2016) | automatic extraction and classification of scenarios from real traffic data | establishes that the idea is a decade old |
| de Gelder et al., ITSC 2020 | auto tag the logs, then mine scenarios as tag combinations | the direct methodological ancestor of the reference stage; same shape, different purpose |
| nuPlan, ICRA 2024 | 1,282 h auto tagged into ~75 scenario types by SQL over atomic primitives | production scale instance, from the nuScenes authors |
| NuScenes-QA, AAAI 2024 | 460K QA pairs generated programmatically from 3D annotations | nearest neighbour in deriving references from annotations rather than humans; their scene graphs carry no ego motion |
| DriveLM, ECCV 2024 | graph structured perception, prediction and planning VQA on nuScenes | closest published evaluation of this kind, and the evidence that multi step decomposition beats single round VQA |
| Talk2BEV, ICRA 2024 | language enhanced bird's eye view: BEV objects converted to text | the RQ3a reference point. They did **not** feed a raster picture |
| BEV-InMLLM (2024) | BEV features injected into a multimodal model, ~9% gain | evidence that BEV information helps, via learned injection rather than a picture |
| TAD benchmark (2025) | ~6K QA over 7 temporal tasks on nuScenes, with human ego action labels | closest work to RQ3b and the storyboard stage |

**Standards vocabulary.** Menzel's functional, logical and concrete abstraction levels; the
PEGASUS six layer model as consolidated by Scholtes et al. (2021); ISO 34502; ASAM
OpenSCENARIO and OpenLABEL. `label_schema.json` sits at the functional to logical boundary
and the storyboard artifact is a logical scenario. The four families cover layers 1 and 4.
Layer 5 is partially covered, since day and night is derived from capture time and agrees
with the human night keyword on 850 of 850 scenes. The interaction family sits outside the
canonical six layer model by that model's own design, which removed interactions from layer 4
as non physical, and is therefore presented as a deliberate extension rather than mapped
silently.

### What this work claims

1. **The auto labeller is the system under test; the scenario mining is the instrument.**
   Prior work mines scenarios to assemble test suites. Here they are mined to score a model,
   per tag family. That inversion is not in the cited set.
2. **The measured cost of a bad reference**: the 3.51x over-count, the 12.2% precision, and
   the reference swap decomposition that separates scope from defects.
3. **A negative dataset suitability result over all 850 scenes**: two of the brief's three
   named manoeuvres do not occur in the dataset the field benchmarks on.
4. **An input representation ablation at fixed ground truth**: front camera, rendered bird's
   eye view, symbolic bird's eye view, both, and temporal, against one label set.
5. **Auditability as an artifact rather than a claim.**

### What it does not claim

That deterministic derivation of scenario tags from driving logs is new; that programmatic
reference generation on nuScenes is new; or that bird's eye view information helping a
language model is a new finding. Each of these is already published, and the novelty audit
that established it is recorded in `REFERENCES.md`.

---

## 4. Decision log

Every design decision, with the measurement that fixed it. Decisions marked as taken on the
measurements were referred to the author with the numbers in hand rather than resolved by
default.

| ID | Decision | Rationale | Date |
|---|---|---|---|
| **D-001** | Ground truth is **derived automatically** from ego pose + HD map + 3D boxes, not hand-labelled | Manual annotation is the bottleneck the brief names; also gives full auditability for the EU AI Act framing (every label has a deterministic derivation) | 2026-08-02 |
| **D-002** | Object labels use **camera-frustum filtering** (`BoxVisibility.ANY`), not global annotation presence | Prior approach counted objects behind the vehicle. F-003 quantifies the error at ~5× | 2026-08-02 |
| **D-003** | Evaluation subset is **stratified by maneuver class**, not a sequential slice | F-005: sequential sampling is ~76% straight-line driving | 2026-08-02 |
| **D-004** | Thesis framing: **pipeline, evidenced by evaluation** | The brief asks for a pipeline and a demonstrator; the benchmark is supporting evidence, not the headline | 2026-08-05 |
| **D-005** | **Lead with per-tag precision/recall/F1**; report tag-IoU alongside in every table | Per-frame set-IoU over ~4 binary tags swings 0.33 on a single flipped tag — noisier than the effect measured. IoU retained because the slides name it as the criterion, and for continuity with earlier results | 2026-08-05 |
| **D-006** | Final stage = **storyboard + structured scenario description**; MetaDrive replay is opportunistic validation only | Both source documents terminate at a structured description feeding MOSAR; neither asks for a simulator. Replay demonstrates schema completeness without the thesis depending on it | 2026-08-05 |
| **D-007** | Add **BDD100K** as a second dataset, for signage only | nuScenes cannot validate signage (F-004); BDD100K has a `trafficLightColor` attribute and is named in the slides. Scope limited to the **validation split** — we evaluate, we do not train | 2026-08-05 |
| **D-008** | Working language **English** | Matches the literature and existing report filenames; French translation deferred to the submitted report | 2026-08-05 |
| **D-009** | Kinematics use **every ego_pose (~154 Hz)**, not just the 2 Hz keyframes | A derivative across 2 Hz keyframes averages over 0.5 s and cannot locate the instant a turn begins, which Stage G1 storyboard boundaries need. Cost is ~40 s for all 850 scenes | 2026-08-05 |
| **D-010** | Near scene edges the observation window **slides to stay 4 s long** instead of shrinking | A turn measured over 2 s is not comparable to one over 4 s: the short window shrinks apparent heading change and mislabels real turns as "going straight". 21% of keyframes are affected, so truncation would bias 1 frame in 5 | 2026-08-05 |
| **D-011** | Poses are **resampled onto a uniform 50 Hz grid** before differentiating | Forced by F-007: the raw stream is irregular (2 µs to 49 ms gaps). Also makes Savitzky-Golay legitimate, since it assumes uniform spacing | 2026-08-05 |
| **D-012** | Instantaneous derivatives near track ends are **flagged (`edge_guard`), not repaired** | F-009: affects 3 of 34,149 frames. Window-level features stay valid there, so C2 prefers those. Special-casing one scene would be less honest than a flag | 2026-08-05 |
| **D-013** | Manoeuvres are detected as **events over the whole scene**, then keyframes inherit the label of the event containing them — not thresholded per frame | F-012: 68% of turn events outlast the 4 s window, and no window in the dataset exceeds 114 deg while events reach 164 deg, so a U-turn is *structurally undetectable* per-frame. Also yields the event boundaries Stage G1 needs, so C2 and G1 share one detector | 2026-08-06 |
| **D-014** | Two **orthogonal label axes** (steering, speed) rather than one mutually-exclusive class | A car turns and brakes simultaneously — 4.2% of turning frames are also non-cruising. A single label forces an arbitrary priority order and marks the VLM wrong when it reports the discarded one. Separate axes also match Stage F's per-tag scoring | 2026-08-06 |
| **D-015** | Longitudinal events detect on a **1.5 s trend acceleration**, not the instantaneous derivative | F-015: acceleration is a second derivative and swings ±5 m/s² around a true mean of −0.4, flipping sign 6.7×/s. Detecting on it missed genuine "arrive at the junction and stop" decelerations entirely | 2026-08-06 |
| **D-016** | The `is_*` boolean tags are **generated from vocabulary constants**, never hand-written | F-016: hand-written tags silently drifted from the event strings, leaving `is_turning_left` False on all 5,500 turning frames | 2026-08-06 |
| **D-017** | Map context records both **where the ego IS and what is AHEAD** along its heading | A front camera sees forward, not underneath. Ego is physically on a crossing in only 2.9% of frames but one is ahead in 37.9%. Scoring a VLM that correctly reports a visible crosswalk against a containment label would mark it wrong for being right | 2026-08-06 |
| **D-018** | Map queries use a **shapely STRtree index**, not the devkit's `layers_on_point` | F-019: 209 ms/query -> ~2 h for the dataset makes iteration impossible; the index answers in 0.24 ms, a 3785x speedup | 2026-08-06 |
| **D-019** | The **`road_block` layer is excluded** from map context | F-020: degenerate on singapore-queenstown (all 676 records share one 7,748-vertex polygon covering the map), costing 894 ms/query and adding no label that road_segment/lane/drivable_area do not | 2026-08-06 |
| **D-020** | `AHEAD_RANGE_M = 30 m`, on physical *and* informativeness grounds | F-021: at 50 m `intersection_ahead` is True on 86.8% of frames and near-useless as a scored label; 30 m is also ~3 s of headway at urban speed and clearly resolved by the camera. The full sweep is reported rather than one tuned value | 2026-08-06 |
| **D-021** | Ego position (`ego_x`, `ego_y`) is stored in the **C1 table** | Lets C3 run as pure geometry off the cached parquet instead of re-loading the 58 s devkit for coordinates already computed | 2026-08-06 |
| **D-022** | Object **presence** tags carry **no range limit**; proximity is a separate `*_near` tag at 15 m | A car at 60 m is plainly visible in the image, so scoring `has_vehicle` False there would mark a correct VLM answer as a false positive — the same error D-017 avoided for map context. Range-gating at 30 m would manufacture false positives against 26% of genuinely visible boxes. The cost is that `has_vehicle` is True on 91.2% of frames; that is reported as its Stage F baseline (F-030), not hidden by a threshold | 2026-08-10 |
| **D-023** | Occlusion is **recorded as a per-tag count, never used as a filter** | 25.6% of frustum-visible boxes are `v0-40`. Dropping them would write false negatives into the *ground truth* — the exact failure that produced the poor VLM scores originally. Two gates in this project have already rejected true positives (F-026, F-028), so the rule that a rejection threshold is itself a false-negative mechanism applies. Storing `n_<tag>_clear` lets Stage F re-score at any strictness without regenerating the table. Also, visibility is a **6-camera** quantity (L-016) | 2026-08-10 |
| **D-024** | Tag membership is read from nuScenes' **own category and attribute tables**, never inferred | `vehicle.parked` (36.0% of annotations) and `cycle.with_rider` (0.63%) are annotated fields. Deriving "parked" from box velocity or "cyclist" from a bicycle's presence would be a heuristic where a ground truth already exists — the rule that the dataset is checked for the answer before a proxy is invented, third occurrence | 2026-08-10 |
| **D-025** | C4 runs on **CAM_FRONT only** | It is the channel PLAN B4 fetches and the only one the VLM is shown. A label derived from a camera the model cannot see would be unanswerable by construction | 2026-08-10 |
| **D-026** | Interaction tags are **360° facts about the world**, not about the front camera | Taken on the measurements below. An agent outside CAM_FRONT still counts, so the labels stay valid for a real six-camera vehicle and are directly reusable for the Stage G scenario description. The `n_*_in_frustum` columns are kept as provenance so Stage F can separate field-of-view effects from model capability (L-019). Note this differs from C4/D-025 deliberately: C4 answers "what is in the picture", C5 answers "what is interacting with the car" | 2026-08-11 |
| **D-027** | Every position **and every velocity** is rotated into the ego frame before any spatial test | F-033: the original global-frame test had 12.2% precision. `nusc.box_velocity()` returns a **global** vector and needs the identical rotation — the same defect one level down, and silent, because the numbers stay plausible while meaning the wrong thing | 2026-08-11 |
| **D-028** | `cut_in` requires **three agreeing signals**: entered the corridor, is moving, and its lateral velocity points inward | F-035: the ego frame rotates with the ego, so during a turn a stationary kerbside car sweeps across the corridor boundary exactly as a real cut-in would. Position alone cannot tell them apart | 2026-08-11 |
| **D-029** | `lead_braking` is measured over **`LEAD_TREND_S` = 1.5 s**, the same width as C2's trend, and against the same `0.6 m/s²` | R6: `box_velocity` is already a centred difference, so re-differencing at 0.5 s flips sign 29.6% of the time against 16.8% at 1.5 s. Reusing C2's magnitude means "the lead is braking" and "the ego is decelerating" denote the same physical event | 2026-08-11 |
| **D-030** | Every tag carries an explicit **`scope`** field | Four families make statements about four different regions of space — the ego itself, the ego's map position, the 30 m corridor ahead, the CAM_FRONT frustum, and the full 360°. `has_pedestrian` and `pedestrian_crossing_path` sit in one schema and are not commensurable; without the field nothing says so (L-019) | 2026-08-11 |
| **D-031** | The schema asserts **structural implications only**, never merely observed ones | Structural relations hold on any dataset; observed ones do not. `traffic_cone ⇒ construction_object` follows from the definition and must hold on any dataset. `is_turn_left ⇒ on_drivable_area` holds on nuScenes but is an accident of map registration quality; asserting it would break elsewhere. The measured-only relations belong in the C7 coverage report | 2026-08-11 |
| **D-032** | Tags with fewer than **`MIN_SCOREABLE_POSITIVES` = 30** positives are marked `role: qc_invariant`, not scored | F-039: `on_walkway` has zero positives in 34,149 keyframes. It is a genuine C3 invariant — the ego is never on a pavement — but its recall is undefined and an F1 for it would be a meaningless number in a results table | 2026-08-11 |
| **D-033** | The frozen schema is **reopened** to add `lane_change` as tag 37, at a **precision** operating point | Taken on the measurements below: taken with the measurements in hand. nuScenes genuinely contains lane changes and a VLM can plausibly see one, so leaving the maneuver unscored would waste real signal. Six detector designs were measured; none reaches the clean-validation bar the other 36 tags cleared, so the operating point is precision and the measured event recall (~59% of description-mentioned scenes) is recorded as L-022 rather than hidden. Schema version C6.1 → **C6.2** | 2026-08-12 |
| **D-034** | Lane-change windows truncated by the end of a recording are **flagged (`window_clamped`), not repaired or discarded** | 3 of 66 events are still in progress when the scene ends, so their window cannot be padded to the physically required length. Identical situation and identical treatment to C1's `edge_guard` (D-012) | 2026-08-12 |
| **D-035** | The evaluation subset is **150 scenes / 1,800 keyframes**: a 26-scene greedy quota cover **plus 124 random scenes** | Taken on the measurements below. The cover alone guarantees every scoreable tag reaches 80 positives in only 26 scenes, but those are by construction the busiest scenes in the dataset — `is_going_straight` reads 69% there against a true 83.65%, so results would not generalise (F-045). The random scenes restore representativeness at a cost of ~200 MB of images | 2026-08-12 |
| **D-036** | Sampling is stratified on **scoreable tags only**, capped at **12 frames per scene**, with the **seed frozen** in the module and stored in the output | `on_walkway` has no positives to sample, so a quota for it is unreachable by definition (D-032). The cap exists because consecutive keyframes are 0.5 s apart and near-duplicates — an uncapped draw buys sample size without information, since stratifying on a proxy for the thing being checked resamples the common case. A subset that cannot be regenerated cannot be audited | 2026-08-12 |
| **D-037** | **RQ1 is restated.** The question is no longer "can labels be derived automatically" but "how good a reference can deterministic scenario mining provide, and what does a bad reference cost" | The original wording claims novelty the literature already holds: de Gelder et al. (ITSC 2020) published the same auto-tag-then-mine shape, nuPlan auto-tags 1,282 h into ~75 types, Elrofai et al. did scenario identification in 2016 (REFERENCES.md R-01..R-03). Everything Stage C actually measured — coverage, corroboration, the 3.51x over-count, the 12.2% precision — answers the restated question and none of it is lost | 2026-09-01 |
| **D-038** | **Every external work is cited in `REFERENCES.md`**, with what it is used for and where it appears; nothing enters without a checked identifier | Until 2026-09-01 the repo cited nothing across 1,900 lines of PLAN + PROCESS. Half-remembered citations are worse than none in a document that will be examined, so the file records verification status per entry | 2026-09-01 |
| **D-039** | Stage D runs **two BEV conditions**: a rendered raster BEV *and* a symbolic scene description of the same content | Taken on the measurements below. Neither Talk2BEV (R-20, BEV->text) nor BEV-InMLLM (R-21, learned feature injection) fed a raster BEV to a model, and R-22 finds MLLMs read abstract non-photographic images poorly. With the raster arm alone, a loss cannot be separated into "BEV is uninformative" and "the renderer is unreadable to the model". The symbolic arm makes that separation measurable, at the cost of one extra T4 pass | 2026-09-01 |
| **D-040** | **B4 fetches all 6,025 CAM_FRONT keyframes of the 150 subset scenes (765 MB)**, not the 1,800 subset tokens (230 MB) | Taken on the measurements below: taken with the measurement in hand. Only 34.2% of the +/-2 keyframe neighbours of subset tokens are themselves in the subset, so E9 (RQ3b) needs 4,771 images and Stage G storyboards need continuous scenes. Whole scenes cost 159 MB more than the +/-2 window and remove two further download rounds; nuScenes URLs are signed and expire, so each round has a real manual cost | 2026-09-01 |
| **D-041** | **The F-050 fix is adopted.** `LC_SEP_LATERAL=True` and `LC_TOKEN_GAP_KF=4` become the defaults; schema **C6.2 → C6.3**; `lane_change` goes from 66 events / 499 keyframes to **75 / 551** | Taken on the measurements below: taken with the measurements in hand. F-050 showed the ceiling was a geometry defect, not a property of the data: `sep` counted forward travel as lateral separation, so real changes were rejected for having driven onwards. Recall 58.3% → 70.8% on the description reference. Adoption was verified to be **contained**: only the `lane_change` column of `gt_all` changes, the C8 subset re-selects **byte-identically** (same 1,800 tokens, same 150 scenes), so D-035/D-036 stand and the B4 image set still resolves to the same 6,025 files. The C6.2 operating point stays reachable by keyword so F-050's comparison regenerates (the rule that every published artifact is reproducible from `src/`) | 2026-09-06 |
| **D-042** | The BEV patch is **100 m radius at 768 px** (0.26 m/px), not the 50 m / 512 px PLAN.md assumed | Taken on the measurements below: on the measurement. Over 2,258 frustum boxes in 250 subset frames, a patch of radius R contains 49.6% (30 m), 80.3% (50 m), 96.1% (75 m), **99.7% (100 m)** of the boxes the ground truth counts. D-022 deliberately put no range limit on the presence tags, so a 50 m BEV would be blind to 19.7% of objects the CAMERA can see — handicapping the BEV arm by construction and making the RQ3a headline partly an artefact of cropping. Same failure class as F-029 | 2026-09-06 |
| **D-043** | The BEV shows **the full 360°, with the CAM_FRONT wedge drawn on it** | Taken on the measurements below. A bird's-eye view naturally shows 360°, and the 360° field carries 3.64x more agents than the front camera (36.8 vs 10.1 per frame). But object tags are CAM_FRONT-scoped (D-025) while interaction tags are 360° (D-026), so an unmarked 360° BEV would let a model correctly report a car behind the ego and be scored wrong — confounding field of view with representation, the very thing RQ3a is trying to isolate. The wedge costs one polygon and is stated identically in the symbolic arm | 2026-09-06 |
| **D-044** | The prompt's **tag list, group structure and scope descriptions are generated from `outputs/label_schema.json`**, never typed | A string vocabulary is never hand-written twice, which is the rule F-016 produced. A prompt that asks for a tag the schema no longer has, or omits one it gained, is the same drift defect pointed at the model instead of the table — and it would be invisible, since the model would simply never answer that field. The two things that *cannot* be generated (a perceptual gloss per tag, a plain-English gloss per scope) are written once in `src/prompts.py` with a test asserting their keys equal the schema's exactly | 2026-09-06 |
| **D-045** | A row whose **backend** failed is not "done"; a row whose **model answered unparseably** is | F-055. On resume the two must be treated differently: an unparseable reply is a real observation and RQ2b's parse-failure rate depends on keeping it, while a session death produced no observation at all and must be retried or the frames are lost with the quota already spent | 2026-09-06 |
| **D-047** | Stage E runs a **628-frame execution subset**, nested inside C8's 1,800 | Taken on the measurements below: on the measurement. The 7B answers one frame in **29.2 s** on a T4, so the full subset is 14.6 h per condition and ~73 h across five — more free-tier quota than exists. A proportional cut is impossible: `is_u_turn` has 36 positives, so halving the frames halves it below the 30 floor (D-032). Instead every frame carrying a tag with <150 positives is kept **whole** and only common tags (500+ positives each) are subsampled, so **the minimum positive count is unchanged at 36** and no tag loses scoreability. 5.1 h per condition, ~26 h for five. C8's 1,800 remain THE evaluation subset; D-035/D-036 are untouched | 2026-09-08 |
| **D-048** | **Qwen2.5-VL-7B stays the primary model**; an alternative is run on a **150-frame sensitivity slice** (new step E10) rather than replacing it | Taken on the measurements below. The 7B is verified on this exact T4 (F-059: 24/24 parsed, 24/24 complete, first try), is the model PLAN.md and REFERENCES.md R-08 are written around, and matches the generation used by DriveLM and NuScenes-QA-era work, so the numbers are comparable. Qwen3-VL-8B (R-34) is newer and probably stronger, but the T4 is Turing — no bfloat16, no Flash Attention 2 — and Turing support in its backends is an open issue, so switching risks spending quota on dtype debugging instead of results. The real gap the switch would have addressed is that **the model is a researcher-chosen parameter with no sensitivity reported, and every researcher-chosen parameter owes a sweep**; a 150-frame slice at ~1.2 h answers that directly and can be dropped if quota runs out, without endangering the main result | 2026-09-08 |
| **D-049** | **F5 runs against NuScenes-QA, scoring ALL 1,471 questions on the 111 val-split frames, decomposed by `template_type` and `num_hop`** — not filtered to a "front-answerable" subset | Taken on the measurements below: taken with the measurements in hand (F-070). Three parts. **(a) NuScenes-QA over DriveLM**: its val split ships answers so scoring is offline, its metric is a hard 30-way top-1 accuracy rather than 60% free-text similarity, and it is the methodological nearest neighbour — a programmatically derived reference from the same 3D boxes, which is what this thesis does. DriveLM's val has no released GT (submission server only), needs an OpenAI key and Java to score, and published work finds its GPT score barely moves when the image is removed. **(b) No filtering**: the field-of-view gap is pervasive, not a filterable subset — object-relative relations, multi-relation questions and zero-hop counting questions all defeat a front/rear split (F-070), so there is no criterion to filter on. **(c) Decompose on the paper's axes**: `template_type` and `num_hop` are how NuScenes-QA reports per-type accuracy, so our numbers slot beside their published table instead of beside a split we invented. The single-camera limitation is then reported as a measured property — the same move as F-004 (nuScenes cannot validate signage) and F-042 (two of three manoeuvres absent), both of which became contributions | 2026-09-16 |
| **D-050** | **A storyboard panel is a maximal run of a constant `(lateral_label, longitudinal_label, lane_change)` triple over consecutive KEYFRAMES**, with a 0.5 s minimum panel duration | The strip must be linear; D-014's manoeuvres are two orthogonal axes, and they are simultaneously active for 8.02% of all driving in 319 of 850 scenes, so the conflict cannot be waved away (F-072). Segmenting on keyframe runs of the already-resolved per-keyframe labels means G1 writes NO new detector — D-013 requires C2 and G1 to share one — and inherits the precedence that resolves the 345 same-axis event overlaps. It also makes the structural invariants exact integers rather than float comparisons. The three alternatives were measured and each fails on the data: lateral-only collapses 54.6% of scenes to a single panel, longitudinal-only deletes all 495 turn boundaries, and one-panel-per-event nests wherever the axes overlap. `MIN_PANEL_S = 0.5` is `groundtruth.MIN_EVENT_S`, not a fresh number, and its sweep is reported because it binds | 2026-09-16 |
| **D-051** | **One scenario = one SCENE, with G1's panels as its timeline, and every parameter is a `[min, max]` RANGE** | The slide's own storyboard strip is five panels of a single accident scenario under one MOSAR logo, so the scenario unit is the scene and not the panel — PROCESS's RQ5 wording ("each event") disagrees and is superseded by the primary document. The range form is Menzel's logical abstraction level, which section 2.1 already claims for this artifact and which the slide states three times in its own field names; a point value would demote it to a concrete scenario silently. Completeness is scored on FIELD POPULATION, never on actor recall, because G3 reads the annotation set directly and recall against it is 100% by construction (the rule that work is pushed upstream into the cached table) | 2026-09-17 |
| **D-052** | **Inference is a PAIRED CLUSTER BOOTSTRAP over scenes, reported under BOTH averages, Holm-corrected within each** | The three alternatives were measured against the data rather than chosen by convention. A per-tag McNemar is the textbook paired test but is *itself* clustered — its 628 frames are 139 scenes — and it answers a per-cell question when every headline is an aggregate; pooled over tags it gives camera-vs-both 510/506, a dead heat that macro F1 contradicts, because discordant cells weight by prevalence and macro does not. An unpaired comparison of two intervals throws away the pairing that makes the test powerful. A frame-level bootstrap ignores ICC 0.517 and is 1.88x too narrow (F-082). Resampling SCENES with shared weights fixes clustering and pairing at once, needs no distributional assumption, and costs 1.2 s. Both averages are reported because the choice reverses two conclusions (F-083) and reporting only the favourable one would be R24's defect in a table | 2026-09-20 |
| **D-053** | **Pre-registered, before the run: Qwen3-VL-8B on the repaired BEV raster, E10's 150-frame slice (`tokens[:150]`, 30 scenes).** Paired with E10's Qwen3 camera run on identical frames, so it asks whether F-097's diagram-reading failure is the 7B's or the representation's. **Reference numbers on the slice, fixed now:** 7B camera 0.367, 7B repaired BEV 0.249 (gap **0.118**); Qwen3 camera 0.459 (macro 34, unanswered cells scored False as in E10); Qwen3 camera `traffic_light_ahead` 0.632 (33 positives), `stop_line_ahead` 0.260 (113), `ped_crossing_ahead` 0.404 (91). **Decision rule:** *model-limited* if Qwen3's BEV-vs-camera macro gap is at most **0.059** (half the 7B's) AND its BEV `traffic_light_ahead` F1 is at least **0.20**; *representation-limited* if the gap is at least **0.089** (three quarters of the 7B's) AND all three map-point "ahead" tags stay at or below **0.10**; anything else is reported as *mixed*, not rounded to either. Both unanswered-cell conventions reported (E10). Slice caveat stands: paired deltas only, no absolute per-tag claims. **Rejected:** BEV with chain-of-thought on the 7B (~8 h at 45.9 s/frame, and a new reasoning prompt, which is where F-065 happened); re-running every arm on Qwen3 (quota) | 2026-09-22 |
### Superseded

* The original final stage was closed loop scenario generation with simulator replay and
  trajectory divergence metrics as a core claim. Superseded by D-006 after reading the
  project slides closely: their pipeline ends at a scenario description feeding MOSAR, and
  the stated evaluation is tag IoU, not simulation fidelity. The original scope overshot both
  source documents.

---

## 5. Findings index

One line per finding. The identifier is what comments in `src/` and `tests/` cite; the
evidence for each is the artifact it names, all of which are in `outputs/`.

| ID | Finding |
|---|---|
| F-001 | Metadata-only loading works. |
| F-002 | Automatic maneuver labels agree with human scene descriptions. |
| F-003 | Annotation-presence ground truth over-counts by ~5×. |
| F-004 | nuScenes cannot validate signage. |
| F-005 | Severe class imbalance in ego maneuvers. |
| F-006 | Ground-truth generation is compute-trivial. |
| F-007 | The raw ego_pose stream is irregularly sampled, and naive differentiation of it is invalid. |
| F-008 | Grid overshoot produced a phantom 12.8 m/s² deceleration. |
| F-009 | A small number of scenes ship stale localisation at t = 0. |
| F-010 | C1 results are robust to both free parameters. |
| F-011 | C1 output validated against human scene descriptions on all 10 mini scenes. |
| F-012 | Real turns outlast the observation window, so per-frame thresholding cannot see a U-turn. |
| F-013 | There is no natural threshold to discover; the cut is a physical argument. |
| F-014 | Threshold sensitivity (250 scenes), and the two axes are decoupled. |
| F-015 | Acceleration is too noisy to detect manoeuvres on; a trend signal is required. |
| F-016 | A silent tag-name drift would have scored the VLM at zero recall on turns. |
| F-017 | C2 labels validated against human scene descriptions, and cross-validated against C1. |
| F-018 | Final class balance and the Stage F2 baselines. |
| F-019 | The devkit's map query is 3785x too slow, and the fast path has two silent traps. |
| F-020 | The `road_block` layer is degenerate on TWO maps. |
| F-021 | The "ahead" corridor range trades physical realism against informativeness. |
| F-022 | Map context validated by physical invariants. |
| F-023 | nuScenes contains NO roundabouts. (Corrected — see F-024.) |
| F-024 | Trajectory shape cannot distinguish a roundabout from a U-turn, and the first detector did not. |
| F-025 | nuScenes scene descriptions are 360°/20-s aggregates and are false at frame level. Over-count reaches 12x. |
| F-026 | The roundabout count, third and final revision: ~2 confirmed, 5 candidates, in 850 scenes. The circle-based test was wrong. |
| F-027 | Final roundabout answer: 1 in 850 scenes. Topological enclosure is necessary but not sufficient; the traffic criterion is a one-way circuit. |
| F-028 | Fifth and final revision: 2 roundabouts. Requiring a CLOSED cycle was itself a false negative. |
| F-029 | Frustum filtering discards 71.5% of the annotation set, measured over the whole dataset. |
| F-030 | What actually limits a visible-object label is occlusion and range, not resolution. |
| F-031 | A parked bicycle is not a cyclist, and nuScenes already knows the difference. |
| F-032 | The C4 tags verified against the images, including the scene that produced F-025. |
| F-033 | The original `following_vehicle` tag had 12.2% precision. Measured over all 34,149 keyframes. |
| F-034 | C5 parameter sweeps, and the resulting prevalences. |
| F-035 | The ego frame rotates with the ego, which manufactures phantom cut-ins. |
| F-036 | `lead_braking` correlates with the ego decelerating without duplicating it (convention 21 / convention 9). |
| F-037 | The C5 tags verified on the images beside the ego frame. |
| F-038 | A third of the label vocabulary is near-constant, so raw accuracy is uninformative for it. |
| F-039 | `on_walkway` has exactly zero positives, and two tags cannot be scored at all. |
| F-040 | No two tags are duplicates, checked across all four families (convention 21). |
| F-041 | 15 structural implications, all holding on every one of the 34,149 keyframes. |
| F-042 | Two of the three maneuvers the brief names do not occur in nuScenes at all. |
| F-043 | Lane-change detection: six designs, and why the operating point is precision. |
| F-044 | The unified ground-truth table, and what the coverage report answers. |
| F-045 | Stratification is necessary, but for a sharper reason than F-005 assumed, and greedy stratification alone is a trap. |
| F-046 | The final subset. |
| F-047 | nuScenes has a public S3 mirror and a keyframes-only tarball; B4 is 7x cheaper than planned. |
| F-048 | `curl --retry` and `-C -` do not compose; the retry truncates and restarts. |
| F-049 | Six published figures had no builder, and one of them had been annotated with a constant that no longer exists. |
| F-050 | The `lane_change` misses were diagnosed wrongly, and the real cause is a mismeasured separation. Recall 14/24 → 17/24 at unchanged precision, but the frozen detector is NOT replaced. |
| F-051 | Sizing the BEV: coverage, and the legibility it costs. |
| F-052 | nuScenes ships 7 invalid map polygons, and they killed the D3 batch. |
| F-053 | Stage D cross-validated against Stage C: it agrees, and every disagreement is accounted for. |
| F-054 | The schema's `derivation` field is unusable as a prompt, and using it would have measured jargon comprehension. |
| F-055 | The checkpointing reintroduced the exact failure it exists to prevent. |
| F-056 | The laptop was never the bottleneck; wall-clock timing across machine sleep was. |
| F-057 | The 3B renames a tag on 86% of frames. |
| F-058 | The Colab bundle shipped a stub, because it was built before the code it packages. |
| F-059 | The 7B works, and the 3B's tag-renaming failure does not reproduce at scale. |
| F-060 | A warn-and-continue fallback cost ten hours, and the run that exposed it also produced the cleanest numbers so far. *(2026-09-08/09) |
| F-061 | The 4-bit path pins the hardware harder than the VRAM figure suggests. |
| F-062 | E5 complete: the 7B answers perfectly and sees exactly what one photograph contains. |
| F-063 | E6a: the rendered BEV loses overall, exactly as predicted, and the losses say where the fault is. |
| F-064 | E6b and E7 complete: the BEV information was fine, the drawing was not; and giving the model both inputs does not help. |
| F-065 | E8 never tested chain-of-thought: the prompt forbade the reasoning it asked for. |
| F-066 | E8 re-run: chain-of-thought is the largest gain measured so far, and it buys recall on exactly the tags one frame was failing. |
| F-067 | Step F3: what the old reference actually cost, measured against all three of them — and a bad reference does not only penalise, it can flatter. |
| F-068 | Step F4: the headline view-condition table, and a whole family that loses to always-yes. |
| F-069 | E9 prepared, NOT RUN. Five frames at full resolution make the model emit garbage, which is `query_multiframe`'s defect one level down — and I rebuilt it once myself before measuring. |
| F-070 | F5 scoped against NuScenes-QA, and the front/rear filter I proposed is wrong: the field-of-view gap is pervasive, not a filterable subset. |
| F-071 | The builders iterated the frozen condition list, so E9's results would have been dropped from every published table without an error. |
| F-072 | G1: a storyboard is linear and D-014's manoeuvres are not, and the parameter I introduced to fix that prevents a defect this design cannot have. |
| F-073 | E9 answers RQ3b, and the answer is no: five real frames leave every motion tag at exactly zero, with ZERO true positives in 628 frames. |
| F-074 | E10 could not have run at all, for a reason that would have read as a finding about the hardware. |
| F-075 | G3: the scenario description, and a SECOND kind of tautology in its own verify criterion. |
| F-076 | H1: the documented mirror is dead, the expected label vocabulary is wrong, and the nuScenes map has a `color` field that is not a colour. |
| F-077 | H2: what signage IS verifiable in nuScenes, and a silent 0.00% on half the maps from reading the obvious field. |
| F-078 | E10: a newer model is substantially better, and the motion tags stay at exactly zero anyway. |
| F-079 | G6: the demonstrator, the brief's last unbuilt named deliverable. |
| F-080 | F5 built and rehearsed, NOT RUN; batching the questions was rejected on a measurement, and the design prompt I wrote contained four errors. |
| F-081 | The novelty audit: contribution 2 is pre-empted, and R-01 has happened a second time. |
| F-082 | The intervals were too narrow by 1.88x, and the comparisons were never tested at all. |
| F-083 | Which "macro F1" was never stated, and the answer reverses two conclusions. |
| F-084 | G2: the segmentation metric mostly measures boundary density, and the baseline barely beats chance. |
| F-085 | Three GPU runs packaged, and a null that would have called a coin flip a result. |
| F-086 | G5 exports OpenSCENARIO; G4 cannot run, and the reason is dependency rot. |
| F-087 | The appendices are generated, and the generator found that the pipeline cannot rebuild its own foundation. |
| F-089 | G2b: the VLM says "no change" to 950 of 951 keyframe pairs, and the first session could not even score that, because the bundle was missing `src/eval.py`. |
| F-090 | H3: presence is read well, state is read unevenly, and one tag's F1 tripled while the model did not move. |
| F-088 | H4: 81% of the cross-dataset gap is the reference's scope, and I first described the wrong tag. |
| F-091 | The F5 session ran E10 again, and F5 could not have run anyway: the notebook's smoke cell built its own call. |
| F-092 | The Stage C cache has writers now, and rebuilding it reproduces every table bit for bit. |
| F-093 | The lane-change comparison's "frozen" row had silently become the adopted detector. |
| F-094 | The BEV repair is built: both defects fixed, 1,800 frames re-rendered, one bundle serves F5 and both re-runs. |
| F-095 | G4 was never blocked; the description instantiates, and it determines the ego's manoeuvres but not where they happen. |
| F-096 | F5: the model scores below a predictor that knows only each question's type, and it fails in one direction. |
| F-097 | The BEV raster repair changed nothing: the renderer was drawing the fixtures, and the model does not read them. `v3_bev_r2` ran 2026-09-22 on the 7B: 628/628 parsed, 0 backend errors, all 35 answered,... |
| F-098 | The repaired TEXT arm gains +0.070, and the gain is the model reading back an answer the description now states. `v3b_bev_symbolic_r2` ran 2026-09-22 on the 7B: 628/628 parsed, 0 backend errors, all 35... |
| F-099 | D-053 answered, and the pre-registered verdict is MIXED: the newer model reads the glyph the 7B never reported, and the camera gap survives anyway. Qwen3-VL-8B on the repaired raster, E10's 150 frames,... |
| F-100 | A full audit of the written dissertation against the artifacts: thirteen defects, every one of them a number or a claim that was true when it was typed. 2026-09-22. Read every chapter and re-derived every... |
| F-101 | Three guards written after the fact, each because the existing one was scoped to the instance rather than the class. 2026-09-22. |
| F-102 | `scripts/check_citations.py` could not fail partially, so it never ran. 2026-09-22. The 406 from `export.arxiv.org` was diagnosed in an earlier pass as a blocked network. It was not: the network is fine.... |
---

## 6. Limitations and threats to validity

Entries were written when the limitation was identified and carry their resolution inline
where one was later found. Nothing here is removed once the limitation is closed, because the
reason a check exists is part of the method.

**L-001 — Maneuver coverage may not match the brief.**
The brief names highway merges, parallel parking (*créneaux*) and roundabouts. nuScenes
is urban Boston and Singapore. F-005 already shows 0% U-turns in 60 scenes. If the named
maneuvers are absent, either the taxonomy is renegotiated toward what nuScenes contains,
or a second dataset is needed. **Resolved by Step C7; raise with supervisor.**

**L-002 — Threshold choice is a researcher degree of freedom.**
Maneuver classes come from thresholds (25° turn, 0.5 m/s stationary, ±2 s window) chosen
by us. They must be tuned against the C1 plots, then **frozen and reported**, and results
checked for sensitivity to them.

**L-003 — Frustum filtering is a proxy for visibility.**
`BoxVisibility.ANY` means the box projects into the image, not that a human or model could
actually see the object. Occlusion is only partly captured by the `visibility_token`
attribute.

**L-004 — Roundabout detection is heuristic.**
Not a map layer; approximated as sustained one-directional yaw rate inside an
intersection-flagged segment. Must be labelled a heuristic wherever reported.

**L-005 — Compute constrains model scale.**
Free Colab T4 and a 16 GB M4 Air. Qwen2.5-VL-7B-4bit is the largest practical model; the
Mac runs the 3B for iteration only. Conclusions are about *accessible* open-weight VLMs,
not about frontier capability — this must be stated explicitly.

**L-006 — Single primary dataset.**
Findings on nuScenes may not generalise. RQ4's BDD100K arm gives one cross-dataset check,
limited to signage.

**L-008 — Sliding the window makes edge keyframes share identical window features.**
A consequence of **D-010**. In scene-1077, keyframes 0–3 all use the window [0, 4] s and
therefore report exactly the same `heading_change_deg` of 47.997°. Window-level features
are duplicated across the first and last few keyframes of every scene (21% of frames
carry a non-zero `window_offset_s`). The accepted alternative was a short window, which
biases those same frames toward "going straight" — a systematic error rather than a
duplicated one. Any per-frame independence assumption in later statistics must account
for this.

**L-009 — Resampling to 50 Hz discards detail above 25 Hz.**
Vehicle dynamics sit well below 5 Hz, so nothing physical is lost, but the choice is a
parameter, and F-010 shows results are stable across 25–100 Hz.

**L-010 — The `u_turn` class is too small to evaluate and is threshold-fragile.**
3 events, 82 keyframes, 0.24% of the dataset. Raising `U_TURN_DEG` from 150° to 165°
removes it entirely, and one mini event sits at **−149.96°**, 0.04° below the cut, so it
is recorded as `turn_right`. Any per-class F1 for `u_turn` in Stage F will be dominated
by sampling noise. It should either be excluded from headline metrics, merged into the
turn classes, or reported with explicit caveats — not presented as a comparable number.

**L-011 — Manoeuvre labels are more threshold-dependent than the C1 kinematics.**
C1's continuous outputs move < 0.18° across all parameter sweeps (F-010). C2's discrete
classes move by up to 6.4 percentage points (F-014). Discretisation, not measurement, is
where the researcher's choices enter. The frozen values must be reported in the thesis
and results shown to be qualitatively stable across the F-014 range.

**L-012 — Gentle manoeuvres below the detection thresholds are labelled `cruising`.**
`cruising` means "no *detected* acceleration event", not "constant speed". A vehicle
decelerating at 0.3 m/s² is labelled cruising. This is deliberate — the thresholds are
set to capture sustained physical manoeuvres rather than drift — but the class name
overstates what it guarantees.

**L-013 — The roundabout detector has no ground truth, so only its negatives are
trustworthy.**
nuScenes labels no roundabouts, so precision and recall are unmeasurable. The detector now
returns zero candidates, and each rejection was verified by eye (F-024). But a roundabout
the ego merely passes without entering would not be found at all, since detection keys off
the ego's own trajectory. The safe claim is therefore narrow: **no scene contains a
roundabout the ego traverses.** It is not a claim that the map areas contain none.

**L-014 — Map context inherits the map's own registration error.**
0.26% of keyframes fall outside any drivable area, concentrated in boston-seaport. The
nuScenes documentation notes that maps and ego poses do not always register perfectly.
These frames get all-False containment labels, which is a small systematic error rather
than random noise.

**L-015 — `intersection_ahead` remains close to a majority label even at 30 m.**
True on 77% of frames, so its majority baseline is 77% and Stage F must be read with that
in mind. It is retained because it is physically meaningful, but it is a weak
discriminator, and per-tag F1 for it should be interpreted against the baseline, never in
isolation.

**L-016 — nuScenes visibility is a six-camera quantity, used here to describe one camera.**
`visibility_token` is defined as the fraction of the object visible **summed across all
six cameras**. C4 records it per tag (`n_<tag>_clear`) as the best available occlusion
signal, but a box marked `v0-40` could be 40% visible in CAM_FRONT and hidden elsewhere,
or the reverse. It is therefore **evidence, not a gate** (D-023): gating on it would
import information from cameras the VLM is never shown. Any Stage F analysis that
re-scores using the `_clear` counts must state this caveat.

**L-017 — Frustum membership is geometric and knows nothing about occlusion.**
A car entirely behind a bus is inside the frustum and counts as present. 25.6% of visible
boxes are `v0-40` and 23.2% have zero lidar points, which bounds how large this effect can
be. It is deliberate: the alternative (dropping them) writes false negatives into the
ground truth, which is the failure mode C4 exists to remove. The honest statement of what
a C4 tag means is **"an object of this class occupies this camera's field of view"**, not
"a human would see it".

**L-018 — 2.9% of visible objects carry no C4 tag at all.**
`pushable_pullable` (trolleys, bins, 2.65%), `bicycle_rack` (0.17%) and `animal` (0.08%).
A VLM correctly reporting "a shopping trolley" scores as a false positive because the
ground truth has no such tag. The rate is small and the categories are marginal for
driving decisions, but the asymmetry is real and should be stated when reporting Stage F
precision.

**L-019 — 26.4% of `pedestrian_crossing_path` positives are outside the front camera.**
Per D-026 the interaction tags are 360° facts, so a pedestrian the camera cannot see still
counts. CAM_FRONT's horizontal half-FOV is about **32°**: of pedestrians forward within
20 m, 100% are in frustum below 30° off-axis, 6.3% at 30–45°, and **0% beyond 45°**.
Applying the full crossing test, **73.6%** of the tag's positives are in frustum — so the
cost of the 360° choice is much smaller than the raw forward-pedestrian figure (24.9%)
suggests, because requiring the pedestrian to reach the corridor within 4 s already
concentrates them near the ego's axis. `lead_vehicle` is barely affected at 99.0% and
`cut_in` at 99.9%. Stage F must report the in-frustum fraction beside any
`pedestrian_crossing_path` score, or roughly a quarter of the misses will be field of view
rather than model error. The `n_*_in_frustum` columns exist for exactly this.

**L-020 — The lead corridor is a straight rectangle, and 42.2% of leads are stationary.**
Two consequences. On a curve the corridor does not follow the lane, so a genuine lead can
fall outside it and a vehicle in the adjacent lane can fall inside; C3's lane centrelines
could fix this and are not used here. And a car parked in the lane ahead is geometrically
a lead without any following relationship — measured, 42.2% of leads have a ground speed
below 0.5 m/s, which in urban Boston and Singapore is a mixture of stopped queues (a real
interaction) and parked vehicles (not one). `lead_speed_mps` is in the table so downstream
users can separate them; it is deliberately not gated, per D-023's reasoning.

**L-021 — Two tags are carried but cannot be evaluated, and three more are very rare.**
`on_walkway` (0 positives) and `on_carpark` (23) are marked `qc_invariant` and are excluded
from Stage F. `is_u_turn` (82), `lead_braking` (646) and `cut_in` (829) clear the bar but
remain rare enough that their F1 needs a confidence interval rather than a point estimate.
C8's stratified subset must oversample them deliberately or they will be almost absent from
a 3,000-frame draw.

**L-022 — `lane_change` ground truth is still the weakest of the 37 tags.**
*Amended 2026-09-06 (D-041). The original diagnosis was wrong; see F-050.*

D-033 chose a precision operating point, and D-041 has since fixed the geometry defect
that was capping recall. Measured event recall against the human descriptions is now
**17 of 24** scenes that mention a lane change (70.8%), up from 14 of 24 (58.3%).

What remains, and what a Stage F reader must be told:

1. **Recall is still not 100%, and 2 of the misses are unrecoverable.** scene-0441 and
   scene-0854 are described as ego lane changes but their tracks show under 1 m of lateral
   motion against a 3.5 m lane — no trajectory method can find them. Two more (0028, 0969)
   are blocked by C2's `is_going_straight` gate: at 4.6 m/s a 3.5 m shift over 3 s genuinely
   needs ~29° of heading, so a slow lane change and a turn are not separable by heading
   magnitude alone.
2. **Precision did not measurably improve, and is not measurably worse.** Wilson intervals
   on the description hit rate overlap heavily (14.4–35.4% frozen vs 16.5–36.9% adopted).
   The honest claim is a recall gain at no measured precision cost — not a precision gain.
3. **At least one probable false positive exists.** scene-1086, at `sep_m = 1.81`, barely
   over the `LC_SEP_MIN_M = 1.8` floor. It is **not in any C8 subset scene**, so it cannot
   affect Stage F, but the floor is permissive (half a lane width) and was not reopened.
4. **Ground-truth false negatives still inflate the VLM's apparent false-positive rate** — a
   model correctly reporting a change we miss is scored wrong. Fewer than before, not zero.
5. **The reference is unreliable in both directions** (L-007). The detector fires in scenes
   whose description never mentions a lane change, and those are not necessarily errors. Two
   of the 24 reference scenes describe *another vehicle* changing lane, not the ego
   (scene-0107, scene-0803), so the true ego ceiling is 22, not 24.

The honest reading is unchanged in shape: `lane_change` precision is trustworthy and its
recall is a lower bound. The bound is simply higher than it was.

**L-023 — The symbolic BEV arm states geometry more precisely than any picture could.**
`describe_bev` gives ranges and bearings as numbers; the raster requires the model to
estimate them. That is not a leak — `tests/test_bev.py` asserts no schema tag name appears
in the text, and the raster shows the same objects, layers and motion — but it is an
inherent asymmetry of the two representations, and it means a symbolic win is partly a win
for *precision of encoding* and not only for *form*. It must be stated wherever the E6a/E6b
comparison is reported. The reverse asymmetry also holds: the raster conveys spatial
relations (what is beside what, the shape of a junction) that the text only implies.

**L-024 — The temporal arm sees a quarter of the pixels the single-frame arms saw, so
RQ3b is measured against a resolution handicap.**
F-069: at 1280x720 the 4-bit path emits degenerate output from four images up, so E9 runs
at 640x360 while E5/E6a/E7 ran at 1280x720. The comparison is therefore not a clean
single-frame-vs-sequence contrast: **a temporal gain is achieved in spite of a resolution
loss, and a temporal loss could be caused by one.** The direction matters for how the
result may be read — a gain is a lower bound on what sequence buys, while a loss is
uninterpretable without a resolution control.

Three things bound it. The tags E9 targets (`is_accelerating`, `is_decelerating`,
`cut_in`, `lane_change`, `is_u_turn`) are defined by MOTION between frames rather than by
fine detail, and they sit at 0.00 in all five single-instant arms, so there is no
resolution-sensitive baseline to lose. F-030 measured that what limits a visible-object
label is occlusion and range, not resolution — the median frustum box is 68.5 px tall and
only 0.3% fall under 15 px. And F-056 measured 640x360 against 1280x720 on one frame at
27.0 s vs 34.7 s, i.e. image size barely binds on cost.

The honest control would be a single-frame arm re-run at 640x360, at ~5 h of quota. It is
not run, and until it is, **the static-tag half of E9 must be reported as confounded**.
This is L-005's territory: a property of what an accessible open-weight model on free-tier
hardware can be fed, not a property of whether sequence helps.

**L-025 — `traffic_light_in_view` models no occlusion, and that bites unevenly by map.**
The tag is pure projective geometry: a fixture whose map position falls inside the camera
frustum within 50 m counts as in view, whether or not a tree, a bus or a building stands in
front of it. The geometry itself is verified (89.3% of projections land in the upper half of
the frame, 0.1% below v = 600, median 32 m), so this is a modelling gap and not an error —
but it is a one-sided one: the tag can only OVER-report. It is worse on the Singapore maps,
whose tree canopy is heavy and whose fixtures are mounted lower (tz median 2.2 m against
boston-seaport's 5.18 m), so a light is more often behind foliage and lower in the frame. Any
VLM scored against this tag will be marked wrong for failing to see lights that are genuinely
not visible, and that error is not distributed evenly across the four locations.

**L-007 — Human scene descriptions are 360°/20-s aggregates, not frame labels.**
The nuScenes `description` field is free text covering the whole scene across all six
cameras. F-025 quantifies the mismatch: on scene-0916 keyframe 16 the description names a
bus, bicycles and pedestrians while the front camera shows six parked cars, a **12x**
over-count against the 360° annotation set. Usable to corroborate **ego-manoeuvre** labels,
which are whole-scene facts locatable in time; **never** usable for object labels.

**L-026 — The scenario description is logical, not concrete (F-095).** It states the ego's
manoeuvre sequence, a speed range and the road as statistics. It carries no starting pose,
no turn angle, no speed profile, no map geometry and no other road user's position. So it
cannot be instantiated without borrowing those from the log, and the G4 instantiation
measures the ego alone. This is a design property of a description meant for scenario
search and coverage, not a defect, but it bounds any claim of executability: the claim is
"the manoeuvre sequence instantiates", never "the scenario is reproducible from the
description".

---
---

## 7. Research question to method traceability

One row per question. Does the method answer it, with what data, against what baseline.

| RQ | Method | Data | Primary metric | Baseline | Status |
|---|---|---|---|---|---|
| **RQ1** reference quality, and the cost of a bad reference | Reference build, full split run, reference swap | trainval metadata, 34,149 keyframes; mini for visual validation | agreement with human descriptions; over-count ratio against frustum filtering; macro F1 under a swapped reference | annotation presence labelling | **Answered.** 37 tags frozen over all 34,149 keyframes. Over-count **3.51x**. Holding outputs fixed, the naive presence reference scores **0.303 against 0.523**, and a reference with no defects at all still moves the score by **0.095**. The distortion has no fixed sign: one defective tag flatters the model by 0.020 |
| **RQ2** single frame capability | Inference and evaluation stages | 628 frame execution subset, CAM_FRONT | per tag precision, recall, F1, with tag IoU reported alongside | majority class | **Answered.** Camera macro F1 **0.389**, best arm **0.448**, baseline **0.290**. Everything temporally defined scores 0.00 |
| **RQ2a** prompting effect | Prompt ablation on the same frames | same | delta F1 across prompt versions | structured JSON prompt | **Answered.** Chain of thought **+0.059** macro on identical input, best arm in all four families, and not a yes bias. The only arm that beats the camera under either average |
| **RQ2b** parse failures | Structured output handling | same | share of frames with unparseable output | — | **Answered. 0.03%** over 3,516 calls; the one failure is token budget truncation, not malformation |
| **RQ3a** bird's eye view against front camera | Rendering, four inference arms, family analysis | subset, camera and BEV | F1 per label family by view condition | front camera only | **Answered, negative.** Camera 0.389 > both 0.375 > raster 0.272 > symbolic 0.244. Both BEV losses significant under both averages. Camera over both is definition dependent, significant under macro and not under micro |
| **RQ3b** temporal input | Five frame sequences | subset | delta F1 on temporal against static tags | single frame | **Answered, negative.** Five real frames leave every motion tag at **0.000 with zero true positives**; macro 0.363 against 0.389. Static tag losses are confounded by the resolution cap (L-024); the motion zeros are not |
| **RQ4** signage | BDD100K arm and cross dataset comparison | BDD100K validation split, plus the nuScenes map traffic light layer | F1 on presence and on state | majority class | **Answered.** Presence reads well; state uneven, and the errors are colour confusion rather than invention. **81% of the cross dataset gap is reference scope**, measured with the same predictions |
| **RQ5** storyboard and description | Segmentation, description, export, instantiation | all 850 scenes; mini for replay | boundary accuracy against a chance floor; field population completeness; executability | fixed interval segmentation and a matched random baseline | **Answered.** 3,351 panels, 850 descriptions, 850 OpenSCENARIO files. The model arm answers `true` on 1 of 951 pairs and cannot segment. Replay fixes the manoeuvre sequence but not the trajectory |

### Does the method cover the brief?

| Brief requirement | Covered by | Verdict |
|---|---|---|
| Auto labelling pipeline | RQ1 and RQ2 | Yes |
| Trajectories | RQ1, ego manoeuvre family | Yes |
| Other users' intentions | RQ1, interaction family | Yes |
| Signage | RQ4 | Only via a second dataset; nuScenes cannot (F-004) |
| Named manoeuvres: merge, parallel parking, roundabout | RQ1 and the coverage report | Two of three are absent from the dataset (F-042, L-001) |
| Driver view and bird's eye view | RQ3a | Yes |
| Storyboard and video sequencing | RQ5 and RQ3b | Yes, 3,351 panels |
| Structured scenario description | RQ5 | Yes, 850 descriptions |
| IoU evaluation criterion | D-005, reported in every table | Yes |
| Demonstrator | `app.py` | Yes |
| EU AI Act traceability | D-001 and the frozen schema | Yes, as framing |

Two qualified items, both measured rather than asserted. No requirement is unaddressed.

---

## 8. Artifacts

Everything cited above is in `outputs/` and is regenerated by a named function in `src/`.

| Artifact | Contents |
|---|---|
| `label_schema.json` | the frozen contract: 37 tags, family, scope, derivation, parameters, prevalence |
| `gt_all.parquet` | one row per keyframe, every label family merged, 34,149 rows |
| `ego_kinematics_<split>.parquet` | speed, acceleration, yaw rate, heading, window provenance |
| `maneuvers_<split>.parquet`, `maneuver_events_<split>.parquet` | manoeuvre labels and the events behind them |
| `map_context_<split>.parquet` | map containment and forward corridor tags |
| `objects_<split>.parquet` | frustum filtered object tags with occlusion counts |
| `interactions_<split>.parquet` | lead vehicle, lead braking, cut in, crossing path |
| `lane_change_<split>.parquet`, `lane_change_events_<split>.parquet` | the 37th tag and its events |
| `coverage_report.json` | the brief's manoeuvre questions with their evidence |
| `e_all5_f1.csv` | per tag F1 for all six conditions |
| `f3_reference_decomposition.csv` | the reference swap, split into scope and defects |
| `f4_family_conditions.csv` | macro F1 per family per condition |
| `f6_significance.csv` | paired cluster bootstrap, both averages, Holm corrected |
| `f7_alt_baselines.csv` | majority, always yes, MCC, balanced accuracy |
| `e10_model_comparison.csv` | the model sensitivity slice |
| `g2_segmentation.csv` | storyboard boundaries against three baselines and the chance floor |
| `g4_instantiation.csv`, `g4_field_mapping.csv` | simulator replay and what the description does not carry |
| `h3_size_decomposition.csv`, `h3_views.csv`, `h4_cross_dataset.csv` | the signage arm and the cross dataset comparison |
| `results/*.jsonl` | raw model output, one object per frame, for every condition |
| `*.png` | the 21 rendered figures |
