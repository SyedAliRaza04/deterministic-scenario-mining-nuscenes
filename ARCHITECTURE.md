# Architecture

How the pipeline is put together, and why it is put together that way.

---

## 1. The shape of the system

Two halves that meet at exactly one artifact.

```
                       nuScenes sensor record
        ego pose  ·  HD map polygons  ·  3D boxes  ·  attributes
                                |
                                v
   +--------------------------------------------------------------+
   |  DETERMINISTIC REFERENCE           src/groundtruth.py         |
   |  no model, no human, no randomness                            |
   |                                                               |
   |  kinematics -> manoeuvres -> map context -> objects           |
   |                                        -> interactions        |
   +--------------------------------------------------------------+
                                |
                     outputs/gt_all.parquet
                  outputs/label_schema.json  (frozen at C6.3)
                                |
              +-----------------+------------------+
              |                                    |
              v                                    v
   +---------------------+            +-------------------------------+
   |  INPUT RENDERING    |            |  EVALUATION      src/eval.py  |
   |  src/bev.py         |            |                               |
   |                     |            |  per tag P/R/F1               |
   |  camera frame       |            |  majority + always-yes        |
   |  BEV raster         |            |  scene cluster bootstrap      |
   |  symbolic BEV text  |            |  Holm corrected paired tests  |
   |  5 frame sequence   |            |  reference swap decomposition |
   +---------------------+            +-------------------------------+
              |                                    ^
              v                                    |
   +---------------------+                         |
   |  INFERENCE          |    JSONL, one row per frame
   |  src/vlm.py         |-------------------------+
   |  src/prompts.py     |
   |  Qwen2.5-VL-7B 4bit |
   +---------------------+
```

The reference is built once and frozen. Everything downstream, including the choice of what
to feed the model, is scored against that same frozen artifact. The single most important
property of this design is that **the reference can be swapped and everything else held
fixed**, which is what makes the cost of a bad reference measurable at all.

## 2. Layer by layer

### 2.1 The deterministic reference

`src/groundtruth.py` is the core of the work and the largest module. It runs in five stages,
each producing a cached parquet table with provenance columns.

| Stage | Output | What it derives |
|---|---|---|
| Kinematics | `ego_kinematics_<split>.parquet` | speed, acceleration, yaw rate, heading, per keyframe window |
| Manoeuvres | `maneuvers_<split>.parquet`, `maneuver_events_<split>.parquet` | lateral and longitudinal event labels, and the events themselves |
| Map context | `map_context_<split>.parquet` | containment on map layers, and a forward corridor query |
| Objects | `objects_<split>.parquet` | what the CAM_FRONT frustum contains, with attributes |
| Interactions | `interactions_<split>.parquet` | lead vehicle, lead braking, cut in, pedestrian crossing path |

Plus two trainval only artifacts: `lane_change_<split>.parquet` and
`roundabout_traversals.parquet`. `build_stage_c()` runs all of them in dependency order from
a single devkit load.

**Kinematics.** Ego poses arrive at roughly 154 Hz but irregularly, with gaps from 2 us to
49 ms. They are resampled onto a uniform 50 Hz grid, which is what makes a derivative
meaningful at all. Interpolation clamps at its edges, so the grid is asserted to be contained
within its source data; letting it run past silently repeats the last value and produces a
false 12.8 m/s^2 braking spike on the last keyframe of a scene.

**Manoeuvres.** Detected as sustained events on a trend signal over a 1.5 s window, not from
instantaneous derivatives. A first derivative of noisy pose data is usable; a second is mostly
noise. Window level features are bounded and robust where instantaneous ones are not, and a
frame's label is then membership of an event rather than a threshold crossing.

**Map context.** HD map polygon layers under a prepared spatial index. Two things dominate
correctness here: the predicate direction (a query for "which polygon contains this point"
must put the *point* in as input with predicate `within`, or it silently returns nothing and
raises no error), and the fact that `drivable_area` keys its polygons under `polygon_tokens`
while every sibling layer uses the singular `polygon_token`.

**Objects.** 3D boxes projected into the CAM_FRONT frustum. The distinction between what a
360 degree annotation set contains and what one camera sees is the single largest source of
reference error in the naive approach, and it is worth 3.51x in object counts.

**Interactions.** Everything here happens in the ego frame, which rotates with the ego. A test
of the form "was this agent outside my lane a moment ago and inside it now" fires on parked
cars whenever the ego turns, because the frame swept over them. Both the position and the
velocity must be transformed; `nusc.box_velocity()` returns a global vector, and rotating only
the position reproduces the bug one level down, silently, with speeds that stay plausible
while pointing the wrong way.

### 2.2 The frozen schema

`outputs/label_schema.json` is the contract between the two halves. It is frozen at version
`C6.3` and is not rebuilt by `build_stage_c`.

For each of the 37 tags it records: family, dtype, **scope**, a prose derivation, the bound
parameters with their values, the source table, the design decisions behind it, measured
prevalence, positive count and majority class baseline.

**Scope is the field that matters most**, and it is why the reference swap experiment is
possible. Five scopes exist:

| Scope | Meaning |
|---|---|
| `ego` | a fact about the ego vehicle itself |
| `map_containment` | the ego's own position on the HD map |
| `map_corridor_30m` | a 30 m x 8 m corridor ahead of the ego, along its heading |
| `camera_frustum_CAM_FRONT` | only what the CAM_FRONT frustum contains |
| `world_360` | every annotated agent, whether or not a camera sees it |

Two tags with near identical names can differ entirely in scope. `traffic_light_ahead` is a
fixture inside the forward corridor; `traffic_light_in_view` is a fixture projected into the
camera frustum within 50 m. That difference alone explains 81% of a cross dataset gap.

Label names are defined **once**, as constants, and every derived column is generated from
them. Hand writing a string vocabulary twice is how `is_turning_left` came to be False on all
5,500 turning frames while the underlying event table was correct.

### 2.3 Input rendering

`src/bev.py` produces two distinct representations from the same content object:

* `render_bev()` draws a top down raster from the HD map and the 3D boxes, as a **picture**.
* `describe_bev()` states the same content as **text**, which is the symbolic control.

Having both is what separates "the model cannot use a bird's eye view" from "the model cannot
read this particular picture". The symbolic arm is not a fairer version of the raster arm: it
states geometry more precisely than any picture could, so it is an upper bound on a reader,
not an equivalent input. `build_reader_ceiling_table()` in `src/eval.py` quantifies that
ceiling per tag.

`src/data.py` assembles the multi frame sequences and packs everything into a self contained
bundle for a GPU session.

### 2.4 Inference

`src/vlm.py` wraps two backends behind one interface: MLX for fast local iteration on Apple
silicon, and Transformers with 4-bit NF4 quantisation for T4 sessions. `src/prompts.py` holds
every prompt, versioned, in one place.

Both modules are deliberately **free of numpy**, so they can be imported from either Python
environment. This is not tidiness: `mlx-vlm` requires `numpy>=2` and the nuScenes devkit
breaks on it, so the two cannot coexist and the shared code must depend on neither.

Runs are checkpointed per row. A session that dies mid condition costs the in flight frame
and nothing more, and completed rows travel inside the bundle so the next run on any host
resumes rather than repeating work.

### 2.5 Evaluation

`src/eval.py` scores predictions against the frozen reference.

* **Per tag precision, recall and F1**, never set IoU alone. A set level IoU hides which tags
  a model can and cannot do.
* **Two baselines**, not one. Majority class and always-yes rank tags differently, and a tag
  whose prevalence is high can hand an always-yes predictor an F1 that a genuinely
  discriminating model does not reach. MCC and balanced accuracy are reported alongside.
* **Scene level cluster bootstrap.** Keyframes 0.5 s apart in one scene are not independent
  trials. The measured intraclass correlation is 0.517, giving an effective sample of about
  224 from 628 frames; resampling scenes rather than frames widens intervals by 1.88x.
* **Degeneracy detection.** With zero true positives no resample can produce one, so a
  bootstrap interval collapses to a point and claims a certainty the data does not have. Zero
  width is flagged on every statistic, not only the headline one.
* **Holm correction** across the family of paired comparisons, with the membership of that
  family decided explicitly rather than by which result files happen to exist.
* **Reference swap.** `build_reference_decomposition()` holds predictions fixed and exchanges
  only the reference, splitting the movement into the part caused by scope and the part
  caused by defects.

### 2.6 Downstream artifacts

* `src/storyboard.py` segments a scene into a timed sequence of events, and scores the
  segmentation against a chance floor computed **before** the run.
* `src/scenario.py` emits the structured scenario description (scenography plus parameters)
  and an OpenSCENARIO export.
* `src/sim.py` replays a description in MetaDrive to test whether the schema is complete
  enough to instantiate. Every simulator import is kept inside a function, so the main
  environment can import the module without the simulator installed.
* `src/bdd.py` handles the BDD100K signage arm and the cross dataset comparison.
* `app.py` is a Streamlit demonstrator reading only precomputed artifacts.

## 3. Cross cutting design rules

**Logic lives in `src/`; notebooks only import.** The project's first phase was a single
1,360 line linear script that defined the same evaluation function three times, with the last
definition silently winning. A notebook is JSON with its source split into escaped one line
strings, so a mistake inside one is invisible until the session opens it. The notebooks here
are generated from `scripts/nb_*.py`, which are the source of truth.

**Parameters at module top, never inline.** One place per module, each constant carrying the
measurement that justifies its value. No threshold, window or rate in this pipeline was picked
by intuition; each one was measured first and each is reported with a sensitivity sweep.

**Provenance columns on every table.** Window extent, sample counts, guard flags. Any label can
be audited back to its source row, which is the whole point of the exercise.

**Cache aggressively, and push work upstream into the cache.** The devkit's 46 to 58 s JSON
parse dominates everything; the arithmetic is seconds. Ego coordinates are written into the
kinematics table so that downstream map work is pure geometry and never reloads the devkit.

**Rare data defects get a flag column, not a special case.** One scene in 785 has stale
localisation. It carries `edge_guard`, not an `if scene_name == ...`.

**Quality control asserts physical and structural bounds.** Cars do not exceed about 35 m/s,
brake past about 6 m/s^2 or yaw past 90 deg/s. Exactly one tag is true per categorical axis;
net displacement does not exceed path length; events are chronological; tokens are unique.
Both of the silent defects found in the first stage were caught by a bound, not by reading
output, and one was invisible to every numeric check and caught only by an exclusivity
assertion.

**Derived categories are a subset or a partition, and the code says which.** `parked + moving
+ stopped` does not equal `has_vehicle`, because bicycles carry `cycle.*` attributes and never
`vehicle.*`. Read off a figure that looks exactly like a counting bug, so the intended relation
is written into the checks as `<=` or `==` deliberately.

## 4. Testing

`tests/test_<step>.py`, plain asserts, runnable with `python tests/x.py`, no framework and no
fixtures. Synthetic inputs with analytically known answers, plus one regression for every
defect ever found. The runner block sits at the **end** of each file: tests appended after it
never run, and one file once reported 9 of 9 passing while five newer checks sat unexecuted
below the block.

Before any GPU session, `scripts/preflight.py` checks the artifact rather than the reference
to it, and the notebook is rehearsed **from the extracted bundle** in a fresh interpreter that
cannot see the repository. A bundle is a subset of the repository and a lazy import fails only
when its function runs; rehearsing with the repository on the path proved nothing, and let 951
GPU calls complete before the scoring cell died on a missing file.

## 5. Environments

| Environment | Python | For | Constraint |
|---|---|---|---|
| `venv` | 3.12 | devkit, parquet, shapely, figures, evaluation | `numpy<2` is mandatory |
| `venv-mlx` | 3.12 | local VLM iteration on Apple silicon | `mlx-vlm` requires `numpy>=2` |
| `venv-sim` | 3.11 | `src/sim.py` only | MetaDrive past 0.2.6.0 requires Python `<3.12` |

Anything touching the devkit, parquet or shapely runs in `venv`. Anything touching a model
runs in `venv-mlx`. `src/vlm.py` and `src/prompts.py` import from neither.
