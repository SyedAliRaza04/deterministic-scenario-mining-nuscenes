# Build plan

The pipeline was built in eight stages, A to H. This document records what each stage set out
to do, what it produced, and what it turned out to be worth. Stages are listed in the order
they were built; G and H were resequenced during the work because H answers a question that G
depends on.

Every stage serves a stated research question. Those questions are in `PROCESS.md`.

---

## Stage A. Setup

Project hygiene, before anything else.

* Rotate the two API keys left live in the first phase's script, and move every secret to a
  gitignored `.env` read through the environment. Verify with a grep that returns nothing.
* Replace the single 1,360 line linear script with a package: logic in `src/`, notebooks that
  only import. The original defined its evaluation function three times, with the last
  definition silently winning.
* Build the main environment with `numpy<2`, which the nuScenes devkit requires.

## Stage B. Data

Get the dataset at the lowest possible cost in disk and bandwidth.

| Step | Result |
|---|---|
| B1 | nuScenes mini: 10 scenes, 404 keyframes, every channel. The development set. |
| B2 | Map expansion pack, for the HD map layers Stage C needs. |
| B3 | trainval **metadata only**: 850 scenes, 34,149 keyframes, 1.17 M annotations. |
| B4 | All 34,149 CAM_FRONT keyframe JPEGs, 4.8 GB, whole scenes, from the public mirror. |

Every other sensor channel of trainval remains metadata only, which is what keeps the whole
dataset inside a laptop's disk budget.

## Stage C. The deterministic reference

**The core of the work.** Derive scenario labels from the sensor record alone, with no new
human annotation, and know how good they are.

| Step | What it built | What it cost to get right |
|---|---|---|
| C1 | Ego kinematics on a uniform 50 Hz grid | The grid must be asserted contained within its source; interpolation clamps at its edges and silently repeats the last value, producing a false 12.8 m/s^2 braking spike on the last keyframe of 6 of 10 mini scenes. |
| C2 | Manoeuvre labels from sustained events | Detected on a 1.5 s trend signal, not on instantaneous derivatives. A first derivative of noisy pose data is usable; a second is mostly noise. |
| C3 | Map context from HD map polygons | Two silent zero yield bugs: the spatial index predicate evaluated in the wrong direction, and `drivable_area` keying its polygons under `polygon_tokens` while every sibling layer uses the singular. Also a 894 ms per query pathological case, fixed by profiling one layer on one map rather than optimising everything. |
| C4 | Visible object labels by frustum filtering | The whole point: what one camera sees, not what a 360 degree annotation set contains. Worth 3.51x in object counts. |
| C5 | Interaction labels in the ego frame | The lead vehicle test was 12.2% precise with 56.4% of its hits pointing backwards. The ego frame rotates with the ego, so a cross frame comparison needs the observer's motion removed, and **every** vector crossing the frame boundary must be transformed, velocity included. |
| C6 | Frozen label schema, version `C6.3` | 37 tags, 4 families, 5 scopes, each with derivation, parameters, source table and prevalence. |
| C7 | Full split run and coverage report | All 34,149 keyframes. 35 of 37 tags scoreable; `on_walkway` has zero positives and `on_carpark` 23, so both become quality control invariants rather than labels. |
| C8 | Stratified evaluation subset | 628 frames across 139 scenes, stratified to exercise rare tags rather than resample the common case. |

### What the coverage report found about the brief

The brief named three manoeuvres. Two of them do not occur in nuScenes at all.

* **Highway merge: absent.** Maximum ego speed across 850 scenes is 66.7 km/h and no keyframe
  reaches 70 km/h. The two scenes whose descriptions say "highway" are *waiting* at a highway
  like intersection at 0 km/h.
* **Parallel parking: absent.** No scene description mentions it, and the three sustained
  reversing runs change heading by at most 10.7 degrees.
* **Roundabouts: present**, and they took three detector revisions to get right. Shape does
  not determine topology: a 180 degree roundabout traversal and a U turn have identical
  trajectories and both have a non drivable turn centre. Enclosure does not separate them
  either, because a median strip is enclosed by road exactly as an island is. What separates
  them is the rule of the domain, that traffic may legally circulate one and not the other,
  and nuScenes ships the `connectivity` table and directed lane centrelines that answer it
  directly.

**Present instead:** 75 lane change events across 67 scenes (551 positive frames), and 3
reversing events.

## Stage D. Input rendering

Give the model something other than a front camera frame, and make the comparison fair.

* **D1 to D3.** Bird's eye view raster from the HD map and the 3D boxes, batch rendered for
  the evaluation subset.
* **The symbolic control.** The same content stated as text. This exists so that "the model
  cannot use a bird's eye view" can be told apart from "the model cannot read this picture".
  It is an upper bound on a reader rather than an equivalent input, and its ceiling is
  measured per tag.
* **The repair.** Two renderer defects, a fixture never drawn into the raster and map geometry
  missing from the description, bounded the conclusion. Both were fixed and both arms
  re-rendered and re-run rather than reported as stated limitations, which converts "a bug
  cannot explain this" from an assertion into a measurement.

## Stage E. Inference

* MLX locally for fast prompt iteration; Transformers with 4-bit NF4 on a free T4 for batches.
* Every prompt in one versioned place.
* A checkpointed runner: per row progress, so a dead session costs the in flight frame only,
  and completed rows travel inside the bundle so any host resumes rather than repeats.
* Six conditions run over the 628 frame subset: front camera, chain of thought, BEV raster,
  BEV symbolic, both images, and a five frame sequence.
* A model sensitivity slice over 150 frames, comparing Qwen2.5-VL-7B against Qwen3-VL-8B.

Every run is preceded by `scripts/preflight.py`, which checks the artifact rather than the
reference to it, and by a rehearsal of the notebook **from the extracted bundle** in an
interpreter that cannot see the repository.

## Stage F. Evaluation

| Step | What it established |
|---|---|
| F1 | Per tag precision, recall and F1, replacing set IoU, which hides which tags a model can do. |
| F2 | Majority class **and** always-yes baselines, plus MCC and balanced accuracy. They rank tags differently, and that matters. |
| F3 | **The reference swap.** Hold the model output fixed, exchange only the reference. |
| F4 | The view condition comparison, per label family. |
| F5 | External validation against NuScenes-QA. |
| F6 | Paired tests with a scene level cluster bootstrap and Holm correction. |

### The two results that carry the thesis

**F3, the cost of a bad reference.** Over 6 tags, swapping an annotation presence reference for
the scoped one moves macro F1 from 0.303 to 0.523. Decomposed: 0.067 of that is the cost of
*scoping* a tag differently, with no error involved anywhere, and 0.153 is defects. Restricted
to the 4 tags with no defects at all, the reference still moves the score by 0.095. The
distortion has no fixed sign, and one defective tag flatters the model.

**F6, what is actually significant.** Against the front camera arm at macro F1 0.389, chain of
thought is the only improvement (+0.059, interval 0.043 to 0.075, Holm corrected p = 0.005).
Every other arm is significantly worse: BEV raster −0.118, BEV symbolic −0.145, both images
−0.014, five frames −0.026.

Two methodological requirements came out of F6 and bind every interval in the project:

* **Correlated observations are fewer observations.** Keyframes 0.5 s apart in one scene are
  not independent trials. Measured intraclass correlation 0.517, effective sample about 224
  from 628 frames, intervals computed as independent Bernoulli trials 1.88x too narrow.
* **A resample cannot move a boundary statistic.** With zero true positives no draw can
  produce one, so the interval collapses to a point and claims a certainty the data does not
  have. Degeneracy is detected as zero width, on every statistic rather than the headline one:
  nine cells had a collapsed recall while F1 still varied.

## Stage H. Signage, via BDD100K

nuScenes ships no traffic sign or traffic light annotations, and its map's `traffic_light`
layer gives position but never state. The brief requires signage, so a second dataset enters,
and the dataset suitability finding becomes a result in its own right.

| Step | Result |
|---|---|
| H1 | BDD100K labels acquired. |
| H2 | What *is* verifiable on nuScenes: fixture presence from the map, never state. |
| H3 | 586 BDD100K images, five signage tags. Light presence F1 0.939 against a 0.797 baseline; sign presence 0.904 against 0.909; red 0.682, yellow 0.540, green 0.869. |
| H4 | The cross dataset comparison. |

**H4 is the cleanest evidence in the project for the reference claim.** Traffic light F1 is
0.672 on nuScenes under a forward corridor reference and 0.912 on BDD100K against human boxes.
Re-scoping the nuScenes reference to what the camera can see, with the same predictions, lifts
it to 0.867 and closes **81%** of the gap.

State is carried as three booleans rather than one colour, because 34.0% of lit images show
more than one distinct lit colour and BDD100K has no lane association, so "what colour is the
light" has no well posed answer. The reference is itself uncertain on 29.94% of images, so
state metrics are reported three times: all images, the uniform subset, and the certain ones.

## Stage G. Storyboard, scenario description, demonstrator

| Step | Result |
|---|---|
| G1 | Scene segmented into a timed sequence of events. F1 0.885 at a one keyframe tolerance and 0.931 at two. |
| G2 | Segmentation scored against a chance floor, and the VLM arm scored against the same floor. |
| G3 | Structured scenario description: scenography plus parameters. Day and night derived from capture time, agreeing with the human night keyword on 850 of 850 scenes. |
| G4 | Instantiation check in MetaDrive. |
| G5 | OpenSCENARIO export, 850 files. |
| G6 | Streamlit demonstrator over precomputed artifacts. |

**The chance floor had to be computed before the run, not after.** A boundary matching metric
with a tolerance window is largely a measure of how dense the reference is: gold boundaries
occur 0.0986 times per keyframe, so random placement with a matched count already scores 0.296
at a one keyframe tolerance and 0.493 at two. The comparison is made against the null's 97.5th
percentile, not its mean, because the model gets one draw and a single random draw has a
standard deviation of about 0.03. In rehearsal, a fake backend answering at random cleared the
null's mean by +0.051 and would have read as a real effect.

**The VLM cannot segment.** Over 951 keyframe pairs it answered `true` once, F1 0.020 at a one
keyframe tolerance against 0.495 for fixed interval placement.

**G4 found the schema's real limit.** Replayed in MetaDrive, the description fixes the sequence
of manoeuvres but not the trajectory: the described ego is on the road 0.655 of the time
against the log's 1.000, and position error across ten scenes runs from 0.5 m to 84.1 m. The
description is logical, not concrete, and the missing pieces are named in the field mapping
table: the starting pose, and the turn angle.

## What remains open

* A model sensitivity slice for the repaired bird's eye view arms under Qwen3-VL-8B. The
  notebook is written and pre-flight green; the run is queued behind quota.
* Two tags cannot be evaluated at all and three more are very rare, so their intervals are
  wide by construction.
* `lane_change` did not reach the validation bar the other 36 tags cleared. It is carried at a
  precision operating point with a measured 59% event recall, and reported as the weakest of
  the 37 rather than quietly included.
