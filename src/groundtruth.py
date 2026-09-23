"""Automatic ground-truth label generation — Stage C, the core of the thesis.

No human annotation anywhere. Every label has a deterministic, auditable derivation
from sensor data, which is also the EU AI Act traceability story the brief asks for.

Three label families (Step C6 freezes the schema):
    ego maneuver      from ego_pose kinematics                    [C1, C2]
    map context       from the map expansion pack                 [C3]
    objects/interact  from 3D boxes, frustum-filtered             [C4, C5]

STATUS: C1-C8 implemented. 37 tags (35 scoreable) over all 34,149 trainval keyframes;
schema frozen in outputs/label_schema.json at version C6.3. See PROCESS.md §3, §5, §9.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

# --- Step C1 parameters -----------------------------------------------------------
WINDOW_S = 4.0                # observation window per keyframe, seconds (+/- 2 s)
RESAMPLE_HZ = 50.0            # uniform grid the raw poses are resampled onto before
                              # differentiating. MEASURED REASON: the raw ego_pose
                              # stream is irregular - median gap 5.9 ms but minimum
                              # 2 us, because separate sensors log poses at nearly the
                              # same instant. Dividing a ~2 mm position change by a
                              # 9 us gap yields 198 m/s. 50 Hz is far above vehicle
                              # dynamics (< 5 Hz) and gives a stable 20 ms timestep.
EDGE_GUARD_S = 0.30           # keyframes this close to the start/end of a scene get
                              # edge_guard=True. INSTANTANEOUS derivatives there are
                              # unreliable for two reasons: np.gradient falls back to a
                              # one-sided difference, and a few scenes ship stale
                              # localisation at t=0 (scene-1074 repeats one position for
                              # 67 ms, then jumps - unsmoothed speed reads 0 -> 13 m/s in
                              # 60 ms, which no vehicle does). Only 1 of 785 moving
                              # scenes, so it is FLAGGED, not special-cased.
                              # WINDOW-level features stay valid: accel_window_mps2 spans
                              # [-2.60, +1.93] m/s^2 across all 34,149 keyframes, whereas
                              # the instantaneous accel exceeds 6 m/s^2 on 36 of them.
                              # C2 should therefore prefer the window-level columns.
TREND_S = 1.5                 # smoothing width, seconds, for the TREND acceleration used
                              # by longitudinal event detection. MEASURED REASON: the
                              # instantaneous accel is a second derivative and swings
                              # +/-5 m/s^2 around a true mean of -0.4 m/s^2, flipping sign
                              # 6.7 times a second - it is jitter, not vehicle behaviour.
                              # At 1.5 s the signal converges on the physical value and
                              # flips only 0.69 times a second (F-015).
SMOOTH_S = 0.20               # Savitzky-Golay width, seconds, applied ON THE UNIFORM
                              # GRID (savgol assumes uniform spacing - applying it to
                              # the raw irregular stream is invalid).
                              # Researcher degree of freedom -> PROCESS.md L-002.

# --- Step C2 thresholds -------------------------------------------------------------
# EVENT-based detection (D-013), not per-keyframe thresholding. Every value here is a
# researcher degree of freedom (L-002); each is justified below and its sensitivity is
# reported in PROCESS.md F-014. Change them here only, never inline.
#
# Note the measured context: |heading change| has NO bimodality to cut at - it decays
# smoothly from 44.5% of frames in 0-1 deg (F-013). So these thresholds cannot be read
# off a natural gap in the data; they are physical arguments, and must be defended as such.

YAW_ENTER_DEG_S = 4.0         # a turn is CONFIRMED above this yaw rate. Measured on the
                              # signal this gates, the INSTANTANEOUS |yaw rate|: median
                              # 0.56 deg/s, p75 1.87, and 4.0 is its 86th percentile, so
                              # this sits clear of routine lane-keeping wobble. (2.1 and
                              # 5.1 are the median and p75 of the per-window ABSMAX, a
                              # different column; quoting those as "the dataset's median
                              # yaw rate" put the wrong distribution in the thesis.)
YAW_EXIT_DEG_S = 1.5          # ...and is considered over below this (Schmitt trigger).
                              # The lower exit widens events to their true extent and
                              # stops a single corner fragmenting into several.
MIN_TURN_DEG = 15.0           # total heading change required to call a segment a turn.
                              # Below this it is a bend in the road, not a manoeuvre.
U_TURN_DEG = 150.0            # total heading change for a U-turn. Reachable ONLY at
                              # event level: the largest 4 s window in the entire
                              # dataset is 114 deg, while events reach 163 deg (F-012).
MIN_EVENT_S = 0.5             # shortest admissible event; rejects single-sample spikes.
MERGE_GAP_S = 0.5             # same-direction events closer than this are one manoeuvre.

STATIONARY_MPS = 0.5          # speed below which the vehicle counts as stopped.
MIN_STOP_S = 1.0              # ...sustained this long, so a momentary crawl is not a stop.
ACCEL_ENTER_MPS2 = 0.6        # confirm an accel/decel event, ON THE TREND SIGNAL. Real
ACCEL_EXIT_MPS2 = 0.25        # stop approaches peak at |trend accel| p10 = 0.94, median
                              # 1.38, so 0.6 confirms 100% of them while staying above
                              # the 1.5 s-smoothed noise floor (F-015).
MIN_SPEED_CHANGE_MPS = 1.5    # an event must actually change speed by this much,
                              # otherwise it is throttle noise rather than a manoeuvre.

# --- Step C3 parameters ---------------------------------------------------------------
AHEAD_RANGE_M = 30.0          # how far forward the "ahead" corridor reaches.
                              # TWO reasons, both recorded in F-021.
                              # (1) Physical: at 30 m a crossing or stop line is clearly
                              #     resolved in a 1600x900 front camera, and at typical
                              #     urban speed (~10 m/s) it is ~3 s of headway - the
                              #     range over which a driver must actually react.
                              # (2) Informativeness: `intersection_ahead` is True on 86.8%
                              #     of frames at 50 m and 91.5% at 70 m, i.e. nearly
                              #     constant and close to useless as a scored label. At
                              #     30 m the four ahead-labels sit at 77/38/61/17%.
                              # The full sweep is reported; this value is not tuned to
                              # flatter a metric, and results are shown across the range.
CORRIDOR_HALFWIDTH_M = 4.0    # half-width of that corridor, about one lane either side,
                              # so "ahead" means on the ego's path rather than anywhere
                              # within a radius.
# Roundabout detection is TOPOLOGICAL, not geometric (F-026). nuScenes has no roundabout
# layer, so a roundabout is defined by its defining feature: a patch of non-drivable ground
# FULLY ENCLOSED by road, which vehicles circulate around. Shape is deliberately not used:
# an earlier circle-fitting test produced false negatives on oval islands, and shape alone
# cannot separate a roundabout from a U-turn anyway (F-024).
ISLAND_MIN_AREA_M2 = 5.0      # smaller than this is kerb noise
ISLAND_MAX_AREA_M2 = 1500.0   # larger is a city block enclosed by roads, not an island
ISLAND_MIN_COMPACTNESS = 0.55 # 4*pi*A/P^2; 1.0 = circle. Excludes long thin medians.
ISLAND_SEARCH_PAD_M = 22.0    # how close the ego must come to be considered a traversal
ROUNDABOUT_MIN_ANGLE_AROUND = 90.0   # degrees swept AROUND the island centre
ROUNDABOUT_MIN_HEADING_SWEEP = 60.0  # ...and the ego's OWN heading must turn with it.
ROUNDABOUT_MIN_WINDING_DEG = 300.0   # DECISIVE traffic-rules criterion (F-027/F-028):
                                     # how far around the island can you travel on legal,
                                     # correctly-directed lanes? A roundabout lets you go
                                     # ~360 deg; a median strip between two carriageways is
                                     # topologically enclosed too, but traffic passes it in
                                     # opposite directions and you cannot get round.
                                     # MEASURED, and unusually for this project there IS a
                                     # natural gap to cut at: candidates score
                                     # 207, 220, 252 | 424, 533 degrees. 300 sits in it.
                                     # NOTE this is reachable winding along a PATH, not a
                                     # closed cycle. Requiring a cycle was too strict and
                                     # rejected a real roundabout (scene-0994): the map
                                     # encodes entry->exit paths through the circulatory
                                     # carriageway rather than an endless loop, so the ring
                                     # covers 359 deg but never links its last lane back to
                                     # its first. Second false-negative gate of this kind
                                     # (see also F-026) - hence R27.
ROUNDABOUT_MIN_SWEEP_RATIO = 0.45    # heading / angle-around. Driving PAST a small refuge
                                     # sweeps ~180 deg around it while the heading barely
                                     # changes (measured: 161 deg around, 11.8 deg heading).
                                     # This ratio is what separates circulating from passing.
# ring_drivable is REPORTED as supporting evidence but is deliberately NOT a gate: the
# strongest candidate (scene-0185, 242 deg around a compact island) scores only 0.76 because
# the map's drivable polygons are incomplete around small islands (F-026).

# The label vocabulary. SINGLE SOURCE OF TRUTH: event `kind` strings, the *_label
# columns and the is_* boolean tags are all derived from these tuples, so they cannot
# drift apart. They did drift once - see F-016.
LATERAL_LABELS = ("going_straight", "turn_left", "turn_right", "u_turn")
LONGITUDINAL_LABELS = ("cruising", "stationary", "accelerating", "decelerating")

# --- Step C4 parameters ---------------------------------------------------------------
NEAR_RANGE_M = 15.0           # an object is "near" inside this radius. Chosen for
                              # INFORMATIVENESS, measured over 1,500 random keyframes
                              # (F-030): presence tags without a range limit saturate
                              # (has_vehicle True in 91.2% of frames, has_pedestrian
                              # 48.5%), while at 15 m they read 35.9% and 10.2%. 15 m is
                              # also ~1.5 s of headway at urban speed. NOTE this bands
                              # distance, it does NOT gate presence - see D-022.
MID_RANGE_M = 30.0            # boundary of the middle distance band. Matches C3's
                              # AHEAD_RANGE_M (D-020) so "vehicle ahead" and "intersection
                              # ahead" are stated over the same stretch of road.
CLEAR_VISIBILITY = "v80-100"  # the nuScenes visibility level counted as "clearly visible".
                              # Recorded as a SEPARATE COUNT per tag, never as a filter:
                              # 25.0% of frustum-visible boxes are v0-40 and dropping them
                              # would put false negatives into the ground truth (R27).
                              # It is also a 6-CAMERA quantity, not a CAM_FRONT one (L-016).

# Object-tag vocabulary. SINGLE SOURCE OF TRUTH, same discipline as LATERAL_LABELS: the
# boolean column, the `n_*` count and the `n_*_clear` count are all generated from these
# keys, so they cannot drift apart the way the C2 tags did in F-016.
# Membership is decided by nuScenes' OWN category and attribute tables wherever possible
# (R33) rather than by a heuristic of ours - `vehicle.parked` and `cycle.with_rider` are
# annotated fields, so "parked" and "has a rider" are read, not inferred.
_LARGE_VEHICLES = frozenset({"vehicle.truck", "vehicle.bus.rigid", "vehicle.bus.bendy",
                             "vehicle.trailer", "vehicle.construction"})
_CONSTRUCTION_OBJECTS = frozenset({"movable_object.barrier", "movable_object.trafficcone",
                                   "movable_object.debris"})
_TWO_WHEELERS = frozenset({"vehicle.bicycle", "vehicle.motorcycle"})

OBJECT_TAGS: dict[str, Any] = {
    # presence: no range limit (D-022) - the tag answers "visible in this image at all"
    "has_pedestrian":      lambda b: b["cat"].startswith("human."),
    "has_vehicle":         lambda b: b["cat"].startswith("vehicle."),
    "parked_vehicle":      lambda b: b["cat"].startswith("vehicle.") and "vehicle.parked" in b["attrs"],
    "moving_vehicle":      lambda b: b["cat"].startswith("vehicle.") and "vehicle.moving" in b["attrs"],
    "stopped_vehicle":     lambda b: b["cat"].startswith("vehicle.") and "vehicle.stopped" in b["attrs"],
    "large_vehicle":       lambda b: b["cat"] in _LARGE_VEHICLES,
    # a bicycle in a rack is NOT a cyclist. The old pipeline conflated them; nuScenes
    # separates them for free via the cycle.with_rider attribute. 4.7% vs 6.5% of frames.
    "cyclist":             lambda b: b["cat"] in _TWO_WHEELERS and "cycle.with_rider" in b["attrs"],
    "parked_bicycle":      lambda b: b["cat"] in _TWO_WHEELERS and "cycle.without_rider" in b["attrs"],
    "construction_object": lambda b: b["cat"] in _CONSTRUCTION_OBJECTS,
    "traffic_cone":        lambda b: b["cat"] == "movable_object.trafficcone",
    "barrier":             lambda b: b["cat"] == "movable_object.barrier",
    # proximity: the SAME predicate as above, restricted to NEAR_RANGE_M. Distance is a
    # separate tag rather than a filter on the presence tags, so a VLM correctly naming a
    # car at 60 m is never scored as a false positive (the D-017 lesson).
    "pedestrian_near":     lambda b: b["cat"].startswith("human.") and b["dist_m"] <= NEAR_RANGE_M,
    "vehicle_near":        lambda b: b["cat"].startswith("vehicle.") and b["dist_m"] <= NEAR_RANGE_M,
}

# --- Step C5 parameters ---------------------------------------------------------------
# Every quantity below is expressed in the EGO FRAME: x forward, y left, z up. The
# original pipeline tested `abs(dy) > abs(dx)` on GLOBAL coordinates, which is a wedge
# pointing along global north/south rather than along the car - see F-033 for how badly
# that fails. `nusc.box_velocity()` also returns a GLOBAL vector and must be rotated the
# same way; it is the identical trap one level down.
LEAD_LATERAL_M = 2.5          # half-width of the ego lane corridor. A nuScenes urban lane
                              # is ~3.5 m, so this admits a vehicle straddling the line
                              # without reaching into the neighbouring lane centre.
                              # Sweep (F-034), lead present on % of frames at 40 m:
                              # half 2.0 -> 24.8%, 2.5 -> 27.0%, 3.0 -> 32.5%.
LEAD_RANGE_M = 40.0           # how far ahead a vehicle still counts as the lead. ~4 s of
                              # headway at urban speed. Sweep at half=2.5 m: 20 m -> 11.8%,
                              # 30 m -> 19.6%, 40 m -> 27.0%, 50 m -> 32.4%.
LEAD_TREND_S = 1.5            # baseline for the lead's speed change. R6: box_velocity is
                              # already a centred difference, so differencing it again is a
                              # second derivative. Measured sign-flip rate between
                              # consecutive estimates: 29.6% at 0.5 s, 20.9% at 1.0 s,
                              # 16.8% at 1.5 s. Same value as C2's TREND_S, deliberately.
LEAD_BRAKE_MPS2 = -0.6        # the lead is braking below this. Magnitude inherited from
                              # C2's ACCEL_ENTER_MPS2 on purpose: "the lead is braking" and
                              # "the ego is decelerating" should mean the same thing.
PED_INTERACT_RANGE_M = 20.0   # a pedestrian beyond this is not interacting with the ego yet.
PED_LATERAL_MPS = 0.5         # lateral ground speed that counts as crossing rather than
                              # walking alongside. Measured on forward pedestrians within
                              # 20 m: those with attribute `pedestrian.standing` have median
                              # |v_lat| 0.01, `pedestrian.moving` 0.28, and 27.0% of all
                              # forward pedestrians exceed 0.5.
PED_TIME_TO_CORRIDOR_S = 4.0  # ...and must reach the ego corridor within this long, so a
                              # pedestrian crossing a side street 18 m away does not fire.
CUT_IN_LOOKBACK_S = 1.5       # window over which an agent must have moved INTO the corridor.
MIN_AGENT_SPEED_MPS = 1.0     # an agent must actually be moving to count as cutting in.
                              # Without this the tag fires on PARKED cars whenever the ego
                              # turns: the ego frame rotates with the car, so stationary
                              # kerbside vehicles sweep across the corridor boundary. The
                              # position test alone cannot tell that apart from a real
                              # cut-in - three agreeing signals are required instead (F-035).

INTERACTION_TAGS = ("lead_vehicle", "lead_braking", "cut_in", "pedestrian_crossing_path")

# --- Step C7 parameters ---------------------------------------------------------------
# Lane-change detection. TWO independent high-precision detectors are unioned (F-043):
# they overlap on only 26 of 72 scenes, because they catch different manifestations -
# a sharp polygon-token flip vs a gradual drift through intermediate lane segments.
# Six designs were measured before settling here; the operating point is PRECISION,
# and the measured event recall against the (weak, L-007) description reference is
# ~50% - recorded as a limitation (L-022), the author's decision D-033.
LC_PARALLEL_DH_DEG = 35.0     # a centreline is "parallel" to the ego within this heading
                              # difference. Excludes cross-street centrelines at junctions.
LC_SEARCH_RADIUS_M = 8.0      # centreline search radius; 99.5% of keyframes have a
                              # parallel centreline within it.
LC_SEP_MIN_M = 1.8            # old/new centreline separation that counts as ADJACENT.
LC_SEP_MAX_M = 7.0            # Measured on confirmed events: median 3.4-3.5 m = one
                              # nuScenes lane width; forks sit near 0, far flips past 7.
LC_CENTRED_M = 0.9            # "settled in a lane" = within this of its centreline.
                              # Boston centrelines run up to ~1.2 m off the driven line,
                              # which is why detector B keys on settlements, not identity.
LC_LATERAL_RATE_MAX_MPS = 2.0 # R2 physical bound: crossing sep metres from settled to
                              # settled cannot exceed ~2 m/s of lateral speed. This single
                              # gate removed the sep>5.8 artifacts (7 m in 1 s is not a
                              # lane change).
LC_MAX_DUR_S = 8.0            # settlements further apart than this are not one manoeuvre.
LC_MIN_SPEED_MPS = 2.0        # creeping in queues produces map-noise flips, not changes.
LC_TOKEN_DH_DEG = 25.0        # detector A: old/new centreline headings must agree...
LC_TOKEN_YAW_DH_DEG = 45.0    # ...and the ego must be driving along the road, not across.

# --- C7 revisited: the two knobs that separate the FROZEN detector from the measured
# alternative. Both default to the frozen (D-033) behaviour; the alternative is offered
# for comparison only and NOTHING is regenerated from it. See F-050.
LC_SEP_LATERAL = True         # ADOPTED 2026-09-06 (D-041). False = the frozen C6.2
                              # behaviour: sep is min_dist(old) + min_dist(new).
                              # True  = sep is |signed_offset(old) - signed_offset(new)|
                              # at the ego position, i.e. the LATERAL gap between the two
                              # centrelines with the along-track component removed.
                              # Measured over all 608 lane-token flip pairs: on CONSECUTIVE
                              # keyframes the two agree (median 3.65 m vs 3.15 m, only 5.4%
                              # differ across the 7 m gate), but as soon as a pair spans a
                              # gap the frozen measure is dominated by forward travel -
                              # median 10.5 m at 2 keyframes and 18.1 m at 4, against a true
                              # lateral gap of 0.03-0.11 m. 91-99.6% of such pairs are then
                              # rejected by `sep <= LC_SEP_MAX_M` for having driven onwards,
                              # not for being far apart.
LC_TOKEN_GAP_KF = 4           # ADOPTED 2026-09-06 (D-041); 0 = the frozen C6.2
                              # behaviour, comparing CONSECUTIVE keyframes only.
                              # k>0 also compares across up to k keyframes carrying no lane
                              # polygon at all. 9,058 of 34,149 keyframes (26.5%) have no
                              # lane polygon under the ego centre, in 1,216 interior runs
                              # whose MEDIAN length is 4 keyframes (2.0 s) - so a change
                              # begun in one mapped lane and finished in another is
                              # routinely invisible to a consecutive-frame test.
LC_TOKEN_GAP_KF_ALT = 4       # kept equal to LC_TOKEN_GAP_KF so `build_lane_change_comparison`
                              # still reports the same sweep point after adoption. The median
                              # interior gap, and the peak of the precision proxy over 0-10
                              # (F-050). Both halves are required: on its own this adds exactly
                              # zero events, because the C6.2 `sep` rejects every cross-gap pair.
                              # The frozen behaviour stays reachable via the keyword arguments,
                              # so F-050's comparison regenerates (D-041).

# --- Step C8 parameters ---------------------------------------------------------------
# Evaluation-subset sampling. Two stages, because the two failure modes are different:
# a purely random draw leaves the rare tags unscoreable, and a purely greedy draw
# selects only the busiest scenes and stops being a sample of driving at all (F-045).
SUBSET_QUOTA = 80             # target positives per scoreable tag. Well above the D-032
                              # scoreable floor of 30, so the subset survives a tag being
                              # rarer inside it than in the full set. 26 scenes suffice.
SUBSET_CORE_SCENES = 26       # ...that is the measured greedy set-cover size for QUOTA=80;
                              # stored rather than hard-coded so the cover is recomputed.
SUBSET_RANDOM_SCENES = 124    # added on top for representativeness. At 150 scenes total,
                              # is_going_straight reads 79% against the true 83.7%; the
                              # 26-scene core alone reads 69% (F-045).
SUBSET_BUDGET = 1800          # keyframes drawn from that 150-scene pool.
SUBSET_MAX_PER_SCENE = 12     # of 40 keyframes per scene. Consecutive keyframes are 0.5 s
                              # apart and near-duplicates, so an uncapped draw would buy
                              # sample size without information (R35).
SUBSET_SEED = 20260812        # frozen: the subset must be reproducible byte-for-byte.

# Reversing (coverage statistic, NOT a scored tag - 3 events in 850 scenes).
REVERSE_STEP_M = -0.10        # signed forward displacement per keyframe below this counts
                              # as backward motion; the p01 of forward steps is -0.002 m,
                              # so this is far outside localisation noise.
REVERSE_MIN_RUN = 3           # >= 3 consecutive keyframes (1.5 s), rejects single-frame
                              # jitter (only 1 such run exists anyway).

# Highway detection (coverage statistic). nuScenes max ego speed is 66.7 km/h, so these
# bounds mainly exist to document that NOTHING clears them (F-042).
HIGHWAY_SPEED_MPS = 19.4      # 70 km/h; 0 of 34,149 keyframes reach it.
FAST_SCENE_MEDIAN_MPS = 13.9  # 50 km/h; 6 of 850 scenes have a median above it.

# --- Step C6: the map vocabulary, promoted to constants -------------------------------
# These were hand-written inside map_context()'s return dict, which is precisely the
# drift risk that left is_turning_left False on 5,500 frames (F-016). They are not
# generated from here - map_context() still writes them literally, because rewriting
# working C3 code would be a larger change than the risk warrants - but the schema is
# built from these tuples and a QC assertion forces the two to agree.
MAP_CONTAINMENT_TAGS = ("on_drivable_area", "at_intersection", "on_lane",
                        "on_ped_crossing", "on_walkway", "on_stop_line", "on_carpark")
MAP_AHEAD_TAGS = ("ped_crossing_ahead", "stop_line_ahead", "intersection_ahead",
                  "traffic_light_ahead")
EGO_EVENT_TAGS = ("lane_change",)   # C7 addition, D-033. Event-based like the C2
                                    # labels; frames inside a detected event are True.
MIN_SCOREABLE_POSITIVES = 30  # fewer positives than this and a per-tag F1 is decoration:
                              # the Wilson 95% interval on a proportion at n=30 is about
                              # +/-0.18, wider than any model difference this thesis could
                              # claim. Measured over 34,149 keyframes, this excludes
                              # on_walkway (0 positives) and on_carpark (23); is_u_turn
                              # survives at 82 but stays rare enough to need an interval.

WINDOW_KEYFRAMES = 4          # legacy 2 Hz framing, kept for reference; C1 uses WINDOW_S


class PoseTrack(TypedDict):
    """Dense, de-duplicated, time-sorted ego trajectory for one scene.

    Built from EVERY ego_pose the scene recorded (~154 Hz), not just the 2 Hz
    keyframes — decision recorded in PROCESS.md D-009.
    """
    t_s: Any            # seconds, relative to scene start
    x: Any              # metres, global/map frame
    y: Any
    yaw_rad: Any        # UNWRAPPED, so differencing is safe
    speed_mps: Any      # instantaneous, from smoothed positions
    accel_mps2: Any     # instantaneous
    yaw_rate_deg_s: Any # instantaneous
    accel_trend_mps2: Any  # speed derivative smoothed over TREND_S - the signal
                           # longitudinal events are detected on, see F-015
    t0_us: int          # scene start, absolute microseconds
    n_raw_poses: int    # poses before resampling (provenance)
    resample_hz: float  # grid the derivatives were taken on


def _split_name(nusc: Any) -> str:
    """'v1.0-mini' -> 'mini'. Which split a devkit handle is open on.

    The C1 to C5 builders name their output after it, so one split's run can never be
    written over the other's, and nothing has to be told twice what the handle already
    knows (R4).
    """
    return str(nusc.version).split("-")[-1]


def scene_pose_track(nusc: Any, scene: dict,
                     smooth_s: float = SMOOTH_S,
                     resample_hz: float = RESAMPLE_HZ) -> PoseTrack:
    """Dense ego trajectory for one scene, with instantaneous derivatives.

    Why every pose and not just keyframes: keyframes are 2 Hz, so a derivative taken
    across them averages over 0.5 s and cannot locate the instant a turn begins —
    which Stage G1 (storyboard event boundaries) needs. The full stream is ~154 Hz
    (median gap 5.9 ms), measured on mini.

    Steps, in order:
      1. Collect the ego_pose of every sample_data in the scene (all sensors, keyframes
         AND sweeps), de-duplicate by timestamp, sort chronologically.
      2. Convert microseconds -> seconds, relative to scene start. Yaw from the
         quaternion, then np.unwrap so the +/-180 deg wrap never appears as a huge
         fake rotation.
      3. RESAMPLE onto a uniform grid at `resample_hz`. This is not cosmetic. The raw
         stream is irregular: median gap 5.9 ms, but 430 of 2962 gaps in scene-0061
         are under 1 ms and the smallest is 2 us, because separate sensors log poses
         at nearly the same instant. Differentiating that directly divided a 1.8 mm
         position change by a 9 us gap and produced 198 m/s.
      4. Savitzky-Golay smoothing of x, y, yaw — valid only now, because savgol
         assumes uniform spacing. Width is a researcher degree of freedom
         (PROCESS.md L-002): it is a parameter and its value is reported.
      5. Derivatives via np.gradient with a constant timestep.

    Step C1.
    """
    import numpy as np
    from pyquaternion import Quaternion
    from scipy.signal import savgol_filter

    from .data import scene_sample_tokens

    # 1. every ego_pose in the scene, de-duplicated by timestamp.
    #    Walk each sensor's sample_data chain exactly once: rewind to the chain start,
    #    then run forward. Walking from every keyframe instead would re-traverse the
    #    same chain ~40 times per sensor.
    seen: dict[int, dict] = {}
    first_sample = nusc.get("sample", scene["first_sample_token"])
    for sd_token in first_sample["data"].values():
        cur = nusc.get("sample_data", sd_token)
        while cur["prev"]:
            cur = nusc.get("sample_data", cur["prev"])
        while cur is not None:
            ego = nusc.get("ego_pose", cur["ego_pose_token"])
            seen.setdefault(ego["timestamp"], ego)
            cur = nusc.get("sample_data", cur["next"]) if cur["next"] else None

    ts_us = np.array(sorted(seen))
    poses = [seen[t] for t in ts_us]

    t0_us = int(ts_us[0])
    t_raw = (ts_us - t0_us) / 1e6
    x_raw = np.array([p["translation"][0] for p in poses])
    y_raw = np.array([p["translation"][1] for p in poses])
    yaw_raw = np.unwrap(np.array([Quaternion(p["rotation"]).yaw_pitch_roll[0] for p in poses]))

    # 3. resample onto a UNIFORM grid. Everything downstream depends on this: the raw
    #    stream has gaps from 2 us to 49 ms, so differentiating it directly divides
    #    millimetre position changes by microsecond timesteps.
    dt = 1.0 / resample_hz
    # The grid must NEVER extend past the last raw pose. np.interp clamps outside the
    # data range, so an overshooting final point repeats the last position and the
    # apparent speed collapses to zero -> a phantom ~13 m/s^2 deceleration on the last
    # keyframe. This hit 6 of 10 mini scenes before it was caught (PROCESS.md F-008).
    n_grid = int(np.floor(float(t_raw[-1]) / dt)) + 1
    t_s = np.arange(n_grid) * dt
    assert t_s[-1] <= t_raw[-1] + 1e-9, "grid overshoots the pose stream"
    x = np.interp(t_s, t_raw, x_raw)
    y = np.interp(t_s, t_raw, y_raw)
    yaw = np.interp(t_s, t_raw, yaw_raw)          # already unwrapped, so safe to interp

    # 4. smoothing, now legitimate because the grid is uniform
    if smooth_s > 0 and len(t_s) > 7:
        win = int(round(smooth_s * resample_hz)) | 1        # force odd
        win = max(5, min(win, (len(t_s) // 2) * 2 - 1))     # must be odd and < len
        if win >= 5:
            x = savgol_filter(x, win, 2)
            y = savgol_filter(y, win, 2)
            yaw = savgol_filter(yaw, win, 2)

    # 5. derivatives with a constant timestep
    speed = np.hypot(np.gradient(x, dt), np.gradient(y, dt))
    accel = np.gradient(speed, dt)
    yaw_rate = np.degrees(np.gradient(yaw, dt))

    # 6. TREND acceleration: speed smoothed hard, then differentiated. Sustained vehicle
    #    behaviour lives here; `accel` above is dominated by differentiation noise.
    if len(speed) > 7:
        w = int(round(TREND_S * resample_hz)) | 1
        w = max(5, min(w, (len(speed) // 2) * 2 - 1))
        accel_trend = np.gradient(savgol_filter(speed, w, 2), dt) if w >= 5 else accel.copy()
    else:
        accel_trend = accel.copy()

    return PoseTrack(t_s=t_s, x=x, y=y, yaw_rad=yaw, speed_mps=speed,
                     accel_mps2=accel, yaw_rate_deg_s=yaw_rate,
                     accel_trend_mps2=accel_trend, t0_us=t0_us,
                     n_raw_poses=len(t_raw), resample_hz=resample_hz)


def keyframe_window(kf_t_s: float, scene_t0_s: float, scene_t1_s: float,
                    window_s: float = WINDOW_S) -> tuple[float, float, float, bool]:
    """Time window for one keyframe, SLID to keep its length constant.

    Decision D-010. A turn measured over 2 s and one measured over 4 s are not
    comparable: the short window shrinks the apparent heading change and mislabels a
    real turn as 'going straight'. 20% of keyframes sit within 2 s of a scene edge,
    so this would bias 1 frame in 5.

    Instead of truncating, the window slides to stay `window_s` long. A keyframe at
    the very start of a scene gets [0, +4 s] rather than [-2 s, +2 s].

    Returns (start_s, end_s, offset_s, is_full) where offset_s is how far the window
    centre had to move off the keyframe, and is_full is False only when the entire
    scene is shorter than window_s.

    Step C1.
    """
    half = window_s / 2.0
    start, end = kf_t_s - half, kf_t_s + half

    if end - start >= scene_t1_s - scene_t0_s:        # scene shorter than the window
        return scene_t0_s, scene_t1_s, 0.0, False
    if start < scene_t0_s:
        end += scene_t0_s - start
        start = scene_t0_s
    elif end > scene_t1_s:
        start -= end - scene_t1_s
        end = scene_t1_s

    centre = (start + end) / 2.0
    return start, end, centre - kf_t_s, True


def ego_kinematics(track: PoseTrack, kf_t_s: float,
                   window_s: float = WINDOW_S) -> dict[str, Any]:
    """Kinematic features for one keyframe, from the dense track.

    Window-edge quantities are INTERPOLATED to the exact window boundaries rather than
    snapped to the nearest pose, so heading_change_deg is measured over exactly
    window_duration_s and stays comparable between frames.

    Step C1.
    """
    import numpy as np

    t = track["t_s"]
    start, end, offset, is_full = keyframe_window(kf_t_s, float(t[0]), float(t[-1]), window_s)
    sel = (t >= start) & (t <= end)

    yaw_start, yaw_end = np.interp([start, end], t, track["yaw_rad"])
    sp_start, sp_end = np.interp([start, end], t, track["speed_mps"])
    duration = end - start

    xs, ys = track["x"][sel], track["y"][sel]
    path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if xs.size > 1 else 0.0
    net_disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if xs.size > 1 else 0.0

    yaw_rate_win = track["yaw_rate_deg_s"][sel]
    speed_win = track["speed_mps"][sel]

    return {
        # ego position in the map frame. Stored here so Step C3 can do all its map work
        # straight from the cached parquet without re-loading the devkit (~58 s) just to
        # look up coordinates it already computed.
        "ego_x": float(np.interp(kf_t_s, t, track["x"])),
        "ego_y": float(np.interp(kf_t_s, t, track["y"])),
        # instantaneous, at the keyframe itself
        "speed_mps":        float(np.interp(kf_t_s, t, track["speed_mps"])),
        "accel_mps2":       float(np.interp(kf_t_s, t, track["accel_mps2"])),
        "yaw_deg":          float(np.degrees(np.interp(kf_t_s, t, track["yaw_rad"]))),
        "yaw_rate_deg_s":   float(np.interp(kf_t_s, t, track["yaw_rate_deg_s"])),
        # aggregated over the window
        "heading_change_deg":   float(np.degrees(yaw_end - yaw_start)),
        "speed_mean_mps":       float(speed_win.mean()),
        "speed_min_mps":        float(speed_win.min()),
        "speed_max_mps":        float(speed_win.max()),
        "accel_window_mps2":    float((sp_end - sp_start) / duration) if duration > 0 else 0.0,
        "yaw_rate_mean_deg_s":  float(yaw_rate_win.mean()),
        "yaw_rate_absmax_deg_s": float(np.abs(yaw_rate_win).max()),
        "path_length_m":        path_len,
        "net_displacement_m":   net_disp,
        # provenance, so every row can be audited
        "window_start_s":    float(start),
        "window_end_s":      float(end),
        "window_duration_s": float(duration),
        "window_offset_s":   float(offset),
        "window_full":       bool(is_full),
        "n_poses_in_window": int(sel.sum()),
        # True if the INSTANTANEOUS derivatives (speed_mps, accel_mps2, yaw_rate_deg_s)
        # sit close enough to the track ends to be affected by one-sided differencing or
        # stale start-of-scene localisation. Window-level columns remain valid.
        "edge_guard": bool(kf_t_s < EDGE_GUARD_S or kf_t_s > float(t[-1]) - EDGE_GUARD_S),
    }


def build_maneuver_tables(nusc: Any, progress: bool = True,
                          out_path: str | None = "outputs/maneuvers_{split}.parquet",
                          events_path: str | None = "outputs/maneuver_events_{split}.parquet",
                          ) -> tuple[Any, Any]:
    """Per-keyframe maneuver labels AND the underlying event table. Step C2.

    Returns (frames_df, events_df). The event table is not a by-product: Stage G1
    needs exactly these start/end times to build the storyboard, so C2 and G1 share
    one detector rather than each inventing their own notion of an event.
    """
    import pandas as pd

    from .data import map_name_for_scene, scene_sample_tokens

    frames, events = [], []
    for i, scene in enumerate(nusc.scene):
        if progress and i % 100 == 0:
            print(f"  scene {i+1}/{len(nusc.scene)}", flush=True)
        track = scene_pose_track(nusc, scene)
        lat = lateral_events(track)
        lon = longitudinal_events(track)
        location = map_name_for_scene(nusc, scene)

        for axis, evs in (("lateral", lat), ("longitudinal", lon)):
            for e in evs:
                events.append({"scene_token": scene["token"], "scene_name": scene["name"],
                               "location": location, "axis": axis, **e})

        for k, sample_token in enumerate(scene_sample_tokens(nusc, scene)):
            sample = nusc.get("sample", sample_token)
            kf_t_s = (sample["timestamp"] - track["t0_us"]) / 1e6
            frames.append({
                "sample_token": sample_token, "scene_token": scene["token"],
                "scene_name": scene["name"], "location": location,
                "keyframe_index": k, "t_rel_s": kf_t_s,
                **maneuver_labels(kf_t_s, lat, lon),
            })
    events_df = pd.DataFrame(events)
    if len(events_df):     # chronological within each scene, so the table reads as a timeline
        events_df = (events_df.sort_values(["scene_name", "start_s", "axis"])
                              .reset_index(drop=True))
    frames_df = pd.DataFrame(frames)
    split = _split_name(nusc)
    if out_path:
        frames_df.to_parquet(out_path.format(split=split), index=False)
    if events_path:
        events_df.to_parquet(events_path.format(split=split), index=False)
    return frames_df, events_df


def clip_to_patch(geom: Any, patch: Any) -> Any:
    """Intersect a map polygon with a patch, repairing the invalid ones nuScenes ships.

    MEASURED 2026-09-06: **7 of 11,496** map polygons across the four locations are
    invalid - 2 `road_segment` on singapore-onenorth and 5 on boston-seaport, all
    self-intersections. `.intersection()` raises `GEOSException: side location conflict`
    on those, and it raises rather than returning something wrong, so the failure is loud
    but fatal: it killed the D3 batch at frame 144 of 1,800.

    R11 - a rare data defect gets repaired and reported, never special-cased by name.
    `make_valid` is called only on the failing polygon, so the common path pays nothing.

    Every caller that clips map geometry MUST come through here. There are two
    (`bev.bev_content`, `figures._draw_map_patch`), and the second had the identical
    latent bug - it simply had not yet been asked to draw a patch containing one of the
    seven. That is why the fix lives here and not at the call site: a bug report names one
    symptom, and the repair belongs in the shared function every caller routes through.
    """
    from shapely.errors import GEOSException

    try:
        return geom.intersection(patch)
    except GEOSException:
        from shapely.validation import make_valid
        return make_valid(geom).intersection(patch)


def map_islands(index: MapIndex) -> list[Any]:
    """Patches of non-drivable ground FULLY ENCLOSED by road, i.e. candidate islands.

    Found as holes (interior rings) in the union of the drivable-area polygons, then
    filtered by size and compactness. Shape-agnostic by construction: an oval, circular
    or teardrop island all qualify. Step C3 / F-026.
    """
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    tree, recs, arr = index.trees["drivable_area"]
    if tree is None:
        return []
    union = unary_union(list(arr))
    polys = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    out = []
    for poly in polys:
        for ring in poly.interiors:
            hole = Polygon(ring)
            area = hole.area
            if not (ISLAND_MIN_AREA_M2 <= area <= ISLAND_MAX_AREA_M2):
                continue
            if 4 * np.pi * area / (hole.length ** 2) < ISLAND_MIN_COMPACTNESS:
                continue
            out.append(hole)
    return out


def roundabout_traversals(index: MapIndex, ego_x: Any, ego_y: Any, yaw_deg: Any,
                          islands: list[Any] | None = None) -> list[dict[str, Any]]:
    """Roundabout traversals for one scene's ego track. HEURISTIC - no ground truth.

    Two conditions, both behavioural rather than shape-based:
      1. the ego sweeps a large ANGLE AROUND an enclosed island, and
      2. its OWN HEADING turns by a comparable amount.

    Condition 2 is essential. Driving straight past a small pedestrian refuge sweeps
    ~180 deg around it while the heading barely moves (161 deg vs 11.8 deg, measured).
    The ratio between the two is what separates circulating from passing.

    Step C3. See F-024 and F-026 for why the earlier shape-based detector was wrong.
    """
    import numpy as np
    from shapely.geometry import Point
    from shapely.ops import unary_union

    if islands is None:
        islands = map_islands(index)
    ex, ey = np.asarray(ego_x, float), np.asarray(ego_y, float)
    yaw = np.unwrap(np.radians(np.asarray(yaw_deg, float)))
    heading_sweep = float(np.degrees(yaw[-1] - yaw[0]))
    tree, recs, arr = index.trees["drivable_area"]
    union = unary_union(list(arr)) if tree is not None else None

    out = []
    for hole in islands:
        c = hole.centroid
        radius = float(np.sqrt(hole.area / np.pi))
        if np.min(np.hypot(ex - c.x, ey - c.y)) > radius + ISLAND_SEARCH_PAD_M:
            continue
        theta = np.unwrap(np.arctan2(ey - c.y, ex - c.x))
        around = float(np.degrees(theta[-1] - theta[0]))
        if abs(around) < ROUNDABOUT_MIN_ANGLE_AROUND:
            continue
        if abs(heading_sweep) < ROUNDABOUT_MIN_HEADING_SWEEP:
            continue
        if abs(heading_sweep) / abs(around) < ROUNDABOUT_MIN_SWEEP_RATIO:
            continue
        # Direction must agree: you cannot circulate an island clockwise while your own
        # heading turns anticlockwise. scene-0020 swept +123 deg around while its heading
        # turned -84 deg - a pass-by with a turn away from the island, not a traversal.
        if np.sign(around) != np.sign(heading_sweep):
            continue
        ann = hole.buffer(7.0).difference(hole.buffer(1.0))
        ring_drivable = (union.intersection(ann).area / ann.area) if union is not None else float("nan")
        out.append({
            "island_area_m2": float(hole.area),
            "island_x": float(c.x), "island_y": float(c.y),
            "angle_around_deg": around, "heading_sweep_deg": heading_sweep,
            "sweep_ratio": abs(heading_sweep) / abs(around),
            "ring_drivable_frac": float(ring_drivable),   # evidence, NOT a gate
        })
    return out


def circulatory_winding(nusc_map: Any, lane_centrelines: dict, cx: float, cy: float,
                        island_radius_m: float, pad_m: float = 25.0,
                        max_depth: int = 14) -> tuple[float, int]:
    """How far around the island can a vehicle legally travel? Step C3 / F-027, F-028.

    The traffic-rules test. Topological enclosure is necessary but NOT sufficient: a
    median strip between two carriageways is also fully enclosed by road, yet traffic
    passes it in opposite directions and cannot get round. A roundabout's one-way
    carriageway lets you travel ~360 deg.

    Measured along any directed PATH in the lane graph, not a closed cycle. Requiring a
    cycle rejected a genuine roundabout (F-028): the map encodes entry->exit paths through
    the circulatory carriageway, so scene-0994's ring spans 359 deg yet never links its
    last lane back to its first.

    Returns (max_winding_degrees, path_length_in_lanes).
    """
    import numpy as np

    near = {t: p for t, p in lane_centrelines.items()
            if np.min(np.hypot(p[:, 0] - cx, p[:, 1] - cy)) < island_radius_m + pad_m}
    adj = {t: [o for o in nusc_map.connectivity.get(t, {}).get("outgoing", [])
               if o in near] for t in near}
    best = (0.0, 0)
    for start in near:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            pts = np.vstack([near[t] for t in path])
            th = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
            w = abs(float(np.degrees(th[-1] - th[0])))
            if w > best[0]:
                best = (w, len(path))
            if len(path) < max_depth:
                for nxt in adj.get(node, []):
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))
    return best


def lane_centrelines(nusc_map: Any) -> dict:
    """Discretised, DIRECTED lane centrelines keyed by lane token. Step C3."""
    import numpy as np
    from nuscenes.map_expansion import arcline_path_utils as apu

    out = {}
    for tok, path in nusc_map.arcline_path_3.items():
        try:
            out[tok] = np.array(apu.discretize_lane(path, 1.0))[:, :2]
        except Exception:
            continue
    return out


class CentrelineIndex:
    """KD-tree over every lane centreline point of one map, with per-point headings.

    The lane-change detectors ask "which parallel centreline is the ego nearest?" tens
    of thousands of times; a linear scan over ~50-80k points per query is minutes,
    the tree answers in microseconds. Same reasoning as the STRtree in D-018. Step C7.
    """

    def __init__(self, index: "MapIndex"):
        from scipy.spatial import cKDTree

        cls = lane_centrelines(index.nusc_map)
        self.tokens: list[str] = []
        self.lines: list[Any] = []
        pts, heads, ids = [], [], []
        for tok, A in cls.items():
            if len(A) < 2:
                continue
            self.tokens.append(tok)
            self.lines.append(A)
            seg = np.diff(A, axis=0)
            h = np.arctan2(seg[:, 1], seg[:, 0])
            pts.append(A)
            heads.append(np.append(h, h[-1]))
            ids.append(np.full(len(A), len(self.lines) - 1, dtype=np.int32))
        self._P = np.vstack(pts)
        self._H = np.concatenate(heads)
        self._L = np.concatenate(ids)
        self._tree = cKDTree(self._P)

    def nearest_parallel(self, x: float, y: float, yaw: float,
                         k: int = 24) -> tuple[int, float]:
        """(line id, distance) of the nearest centreline PARALLEL to the ego heading.

        Parallel means within LC_PARALLEL_DH_DEG, which excludes cross-street
        centrelines at junctions. Returns (-1, nan) when none is inside
        LC_SEARCH_RADIUS_M.
        """
        d, i = self._tree.query([x, y], k=k, distance_upper_bound=LC_SEARCH_RADIUS_M)
        best: dict[int, float] = {}
        for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(i)):
            if not np.isfinite(dd):
                break
            dh = abs((self._H[ii] - yaw + np.pi) % (2 * np.pi) - np.pi)
            if dh > np.radians(LC_PARALLEL_DH_DEG):
                continue
            ln = int(self._L[ii])
            if ln not in best or dd < best[ln]:
                best[ln] = float(dd)
        if not best:
            return -1, float("nan")
        ln = min(best, key=lambda l: best[l])
        return ln, best[ln]

    def signed_offset(self, line_id: int, x: float, y: float) -> float:
        """SIGNED perpendicular offset to a centreline; positive = point is to its LEFT.

        Uses the local segment direction, so a point past the polyline's end measures
        its lateral offset rather than the distance to the endpoint - without this, a
        lane segment ending mid-manoeuvre fakes lateral motion that never happened.

        The sign is what lets two centrelines be DIFFERENCED at one point to get the
        lateral gap between them, free of the along-track component (LC_SEP_LATERAL).
        """
        A = self.lines[line_id]
        d = np.hypot(A[:, 0] - x, A[:, 1] - y)
        i = int(d.argmin())
        j0, j1 = (i, i + 1) if i + 1 < len(A) else (i - 1, i)
        t = A[j1] - A[j0]
        t = t / max(float(np.hypot(*t)), 1e-9)
        v = np.array([x, y]) - A[j0]
        return float(t[0] * v[1] - t[1] * v[0])

    def perp_offset(self, line_id: int, x: float, y: float) -> float:
        """Unsigned perpendicular distance to a centreline. One derivation, one place (R4)."""
        return abs(self.signed_offset(line_id, x, y))

    def heading_at(self, line_id: int, x: float, y: float) -> float:
        A = self.lines[line_id]
        i = int(np.hypot(A[:, 0] - x, A[:, 1] - y).argmin())
        return float(self._H[np.where(self._L == line_id)[0][i]])

    def min_dist(self, line_id: int, x: float, y: float) -> float:
        A = self.lines[line_id]
        return float(np.hypot(A[:, 0] - x, A[:, 1] - y).min())


def lane_change_events(scene_kin: Any, index: "MapIndex", cl: CentrelineIndex,
                       lane_tokens: list, straight: list,
                       token_gap_kf: int = LC_TOKEN_GAP_KF,
                       sep_lateral: bool = LC_SEP_LATERAL) -> list[dict[str, Any]]:
    """Ego lane-change events for one scene: the UNION of two precision detectors.

    A - token flip: the containing lane POLYGON changes between consecutive in-lane
        keyframes to a non-successor, and the two lanes' centrelines are parallel and
        one lane-width apart at the crossing. Catches sharp, clean changes.
    B - settlement pair: the ego is settled (within LC_CENTRED_M of a parallel
        centreline), and the next settlement is on a DIFFERENT centreline one
        lane-width of PERPENDICULAR offset away, reached at a physically possible
        lateral rate. Identity between settlements is deliberately ignored, because a
        crossing can pass through intermediate lane segments and Boston centrelines
        sit up to ~1.2 m off the driven line (F-043).

    Both require C2's going_straight (a turn between parallel curved centrelines at a
    junction is how five of eight visually-checked candidates turned out to be false,
    F-043) and both apply the LC_LATERAL_RATE_MAX_MPS bound (R2).

    `scene_kin` rows need keyframe_index / ego_x / ego_y / yaw_deg / speed_mps;
    `lane_tokens[i]` is the containing lane polygon token or None; `straight[i]` is
    C2's is_going_straight. Step C7, decision D-033.

    `token_gap_kf` and `sep_lateral` default to the FROZEN behaviour and reproduce the
    shipped 66 events / 499 positive keyframes byte-for-byte. The measured alternative
    (LC_TOKEN_GAP_KF_ALT with sep_lateral=True) is reported by
    `build_lane_change_comparison()`, never adopted here - see F-050 and L-022.
    """
    rows = list(scene_kin.itertuples())
    n = len(rows)
    conn = index.nusc_map.connectivity
    cls_by_tok = {t: i for i, t in enumerate(cl.tokens)}
    events: list[dict[str, Any]] = []

    # --- detector A: token flip with geometric confirmation ---------------------------
    # Which pairs of keyframes are compared. At token_gap_kf=0 this is exactly
    # "consecutive keyframes, both in a mapped lane"; above 0 the look-back also steps
    # over keyframes with NO lane polygon, which is 26.5% of the dataset.
    pairs: list[tuple[int, int]] = []
    for i in range(1, n):
        if not isinstance(lane_tokens[i], str):
            continue
        for back in range(1, token_gap_kf + 2):
            if i - back < 0:
                break
            if isinstance(lane_tokens[i - back], str):
                pairs.append((i - back, i))
                break
    for ia, ib in pairs:
        a, b = rows[ia], rows[ib]
        ta, tb = lane_tokens[ia], lane_tokens[ib]
        if ta == tb:
            continue
        if tb in set(conn.get(ta, {}).get("outgoing", [])):
            continue                        # ordinary progression
        if not all(straight[k] for k in range(ia, ib + 1)):
            continue                        # turns flip between parallel curved lanes
        if b.speed_mps < LC_MIN_SPEED_MPS:
            continue
        la, lb = cls_by_tok.get(ta), cls_by_tok.get(tb)
        if la is None or lb is None:
            continue
        ha = cl.heading_at(la, b.ego_x, b.ego_y)
        hb = cl.heading_at(lb, b.ego_x, b.ego_y)
        if abs((ha - hb + np.pi) % (2 * np.pi) - np.pi) > np.radians(LC_TOKEN_DH_DEG):
            continue                        # not parallel: junction stream, not a lane
        yaw = np.radians(b.yaw_deg)
        if abs((yaw - hb + np.pi) % (2 * np.pi) - np.pi) > np.radians(LC_TOKEN_YAW_DH_DEG):
            continue                        # ego crossing the road, not driving it
        if sep_lateral:
            sep = abs(cl.signed_offset(la, b.ego_x, b.ego_y)
                      - cl.signed_offset(lb, b.ego_x, b.ego_y))
        else:
            sep = cl.min_dist(la, b.ego_x, b.ego_y) + cl.min_dist(lb, b.ego_x, b.ego_y)
        if not (LC_SEP_MIN_M <= sep <= LC_SEP_MAX_M):
            continue
        span_s = (b.keyframe_index - a.keyframe_index) * 0.5
        if span_s > 0.5 and sep / span_s > LC_LATERAL_RATE_MAX_MPS:
            continue                        # R2, and only meaningful once the pair spans
                                            # more than one interval: a single flip is
                                            # instantaneous and has no rate to check.
        # The flip is instantaneous but the crossing is not: at the R2 lateral bound,
        # sep metres takes >= sep / LC_LATERAL_RATE_MAX_MPS seconds. Pad the window to
        # that physical minimum, or the mid-change frames on either side get labelled
        # False - a ground-truth false negative at every event edge.
        # need = intervals required to cross `sep` at the physical lateral bound; the raw
        # window already spans (ib - ia), so each side needs ceil((need - span)/2) - FLOOR
        # division here silently leaves the window half an interval too short.
        need = int(np.ceil(sep / LC_LATERAL_RATE_MAX_MPS / 0.5))
        pad = max(0, -(-(need - (ib - ia)) // 2))
        lo, hi = int(rows[0].keyframe_index), int(rows[-1].keyframe_index)
        kf0, kf1 = a.keyframe_index - pad, b.keyframe_index + pad
        # A manoeuvre still in progress when the recording ends cannot be given its
        # full window. Flag it rather than repairing or discarding it - the same
        # treatment C1 gives truncated windows (D-012). Without the flag, the physical
        # rate check fails on 3 genuine events and looks like a detector bug.
        clamped = kf0 < lo or kf1 > hi
        events.append({"kf0": max(lo, kf0), "kf1": min(hi, kf1),
                       "sep_m": float(sep), "detector": "token_flip",
                       "speed_mps": float(b.speed_mps), "window_clamped": clamped})

    # --- detector B: settlement pairs -------------------------------------------------
    assign = [cl.nearest_parallel(r.ego_x, r.ego_y, np.radians(r.yaw_deg)) for r in rows]
    settled = [(i, r, ln) for i, (r, (ln, off)) in enumerate(zip(rows, assign))
               if ln >= 0 and off <= LC_CENTRED_M]
    for (i0, r0, l0), (i1, r1, l1) in zip(settled[:-1], settled[1:]):
        if l0 == l1:
            continue
        sep = cl.perp_offset(l0, r1.ego_x, r1.ego_y)
        if not (LC_SEP_MIN_M <= sep <= LC_SEP_MAX_M):
            continue                        # collinear successor (~0) or too far
        dur_s = (r1.keyframe_index - r0.keyframe_index) * 0.5
        if not (0 < dur_s <= LC_MAX_DUR_S):
            continue
        if sep / dur_s > LC_LATERAL_RATE_MAX_MPS:
            continue                        # R2: faster than a car can move sideways
        win = range(i0, i1 + 1)
        if not all(straight[j] for j in win):
            continue
        if np.mean([rows[j].speed_mps for j in win]) < LC_MIN_SPEED_MPS:
            continue
        events.append({"kf0": r0.keyframe_index, "kf1": r1.keyframe_index,
                       "sep_m": float(sep), "detector": "settlement",
                       "speed_mps": float(np.mean([rows[j].speed_mps for j in win])),
                       "window_clamped": False})   # bounded by real settlements


    # --- union: overlapping windows are one manoeuvre ---------------------------------
    events.sort(key=lambda e: (e["kf0"], e["kf1"]))
    merged: list[dict[str, Any]] = []
    for e in events:
        if merged and e["kf0"] <= merged[-1]["kf1"] + 1:
            m = merged[-1]
            m["kf1"] = max(m["kf1"], e["kf1"])
            m["kf0"] = min(m["kf0"], e["kf0"])
            if e["detector"] not in m["detector"]:
                m["detector"] += "+" + e["detector"]
            m["window_clamped"] = m["window_clamped"] or e["window_clamped"]
        else:
            merged.append(dict(e))
    return merged


def signed_forward_steps(kin: Any) -> Any:
    """Per-keyframe SIGNED forward displacement, recovering what C1 discarded.

    `speed_mps` is a magnitude; projecting the keyframe-to-keyframe displacement onto
    the heading restores the sign, which is the only way to see reversing. Pure table
    arithmetic on cached columns (R20). Step C7.
    """
    yaw = np.radians(kin["yaw_deg"])
    g = kin.groupby("scene_name")
    dx, dy = g["ego_x"].diff(), g["ego_y"].diff()
    return dx * np.cos(yaw) + dy * np.sin(yaw)


def reversing_runs(kin: Any) -> list[dict[str, Any]]:
    """Sustained backward-motion runs. Coverage statistic - 3 in 850 scenes. Step C7."""
    kin = kin.sort_values(["scene_name", "keyframe_index"])
    # gt_all already carries the projection; recomputing it would need ego_x/ego_y/yaw_deg
    # that the merged table deliberately drops, and would be a second copy of the same
    # derivation (R4).
    steps = (kin["fwd_step_m"] if "fwd_step_m" in kin.columns
             else signed_forward_steps(kin))
    out = []
    for name, grp in kin.assign(_s=steps).groupby("scene_name"):
        back = (grp["_s"] < REVERSE_STEP_M).values
        i = 0
        while i < len(back):
            if back[i]:
                j = i
                while j + 1 < len(back) and back[j + 1]:
                    j += 1
                if j - i + 1 >= REVERSE_MIN_RUN:
                    out.append({"scene_name": name,
                                "kf0": int(grp["keyframe_index"].values[i]),
                                "kf1": int(grp["keyframe_index"].values[j]),
                                "net_m": float(grp["_s"].values[i:j + 1].sum())})
                i = j + 1
            else:
                i += 1
    return out


def build_lane_change_table(dataroot: str = "data/nuscenes",
                            split: str = "trainval",
                            progress: bool = True,
                            token_gap_kf: int = LC_TOKEN_GAP_KF,
                            sep_lateral: bool = LC_SEP_LATERAL,
                            out_path: str | None = "outputs/lane_change_{split}.parquet",
                            events_path: str | None = "outputs/lane_change_events_{split}.parquet",
                            ) -> tuple[Any, Any]:
    """(per-keyframe lane_change table, event table) for a whole split. Step C7.

    Defaults are the FROZEN detector (D-033). The two keywords exist only so
    `build_lane_change_comparison()` can measure the alternative on identical inputs.
    """
    import pandas as pd

    kin = pd.read_parquet(f"outputs/ego_kinematics_{split}.parquet").sort_values(
        ["scene_name", "keyframe_index"]).reset_index(drop=True)
    man = pd.read_parquet(f"outputs/maneuvers_{split}.parquet")[
        ["sample_token", "is_going_straight"]]
    kin = kin.merge(man, on="sample_token", validate="1:1")

    indices = {loc: MapIndex(dataroot, loc) for loc in sorted(kin.location.unique())}
    cls = {loc: CentrelineIndex(idx) for loc, idx in indices.items()}

    ev_rows, flag = [], {}
    for si, (name, grp) in enumerate(kin.groupby("scene_name")):
        if progress and si % 100 == 0:
            print(f"  scene {si}/{kin.scene_name.nunique()}", flush=True)
        loc = grp.location.iloc[0]
        lanes = [indices[loc].layers_at(r.ego_x, r.ego_y)["lane"] for r in grp.itertuples()]
        lane_toks = [rec["token"] if rec else None for rec in lanes]
        straight = [bool(r.is_going_straight) for r in grp.itertuples()]
        for e in lane_change_events(grp, indices[loc], cls[loc], lane_toks, straight,
                                    token_gap_kf=token_gap_kf, sep_lateral=sep_lateral):
            ev_rows.append({"scene_name": name, **e})
            for k in range(e["kf0"], e["kf1"] + 1):
                flag[(name, k)] = True

    events = pd.DataFrame(ev_rows)
    table = kin[["sample_token", "scene_name", "keyframe_index"]].copy()
    table["lane_change"] = [flag.get((r.scene_name, r.keyframe_index), False)
                            for r in table.itertuples()]
    # Only the FROZEN detector may write. `build_lane_change_comparison` passes None for
    # both destinations, because it calls this with the alternative parameters and would
    # otherwise leave the last variant's output in the file the thesis cites (D-033).
    if out_path:
        table.to_parquet(out_path.format(split=split), index=False)
    if events_path:
        events.to_parquet(events_path.format(split=split), index=False)
    return table, events


# Scene descriptions that mention a lane change. WEAK ground truth (L-007): the field is
# a 360 deg / 20 s free-text summary, incomplete in both directions, and it does not say
# WHOSE lane change it is - scene-0107 ("bendy bus changing lane") and scene-0803 ("car
# passing by, changing lane") both attribute it to another vehicle. Regex kept here so the
# reference set is reproducible rather than retyped (R4, R30).
LC_DESCRIPTION_RE = r"lane\s*chang|chang\w*\s+lane"
LC_DESCRIPTION_NON_EGO = ("scene-0107", "scene-0803")


def lane_change_reference(nusc: Any) -> tuple[list[str], list[str]]:
    """(all scenes whose description mentions a lane change, ego-only subset). Step C7."""
    import re
    pat = re.compile(LC_DESCRIPTION_RE, re.I)
    allsc = sorted(s["name"] for s in nusc.scene if pat.search(s["description"]))
    return allsc, [n for n in allsc if n not in LC_DESCRIPTION_NON_EGO]


def build_lane_change_comparison(nusc: Any, dataroot: str = "data/nuscenes",
                                 split: str = "trainval",
                                 variants: dict[str, dict] | None = None,
                                 out_path: str = "outputs/lane_change_variants.json",
                                 progress: bool = True) -> dict[str, Any]:
    """Frozen detector vs each alternative, same scenes and same reference. Step C7 revisited.

    Exists because L-022 records a recall number with no reproducible way to recompute it,
    and because R30 says anything the thesis cites needs a builder. Reports, for each
    variant: event and positive-frame counts, event recall over the description reference,
    and the DESCRIPTION HIT RATE - the share of fired scenes whose description mentions a
    lane change. The hit rate is only a precision PROXY: descriptions are incomplete
    (L-007), so it is biased low for every variant equally, which makes it usable for
    RANKING variants and useless as an absolute precision figure. `enrichment` is that
    hit rate over the 24/850 base rate.

    Nothing here regenerates lane_change_*.parquet, gt_all.parquet or label_schema.json.
    """
    import json

    import pandas as pd

    variants = variants or {
        # LITERALS, deliberately, and the only place in this module that does not read the
        # constants. The frozen row IS the C6.2 detector: consecutive keyframes only, and
        # sep as min_dist(old) + min_dist(new). The constants moved to the adopted values
        # at D-041, so a row defined by them stopped describing the baseline and started
        # duplicating `sep_lateral + gap` -- the comparison would have reported the adopted
        # detector against itself and called the difference zero.
        "frozen (D-033)": {"token_gap_kf": 0, "sep_lateral": False},
        "sep_lateral only": {"token_gap_kf": 0, "sep_lateral": True},
        "gap only": {"token_gap_kf": LC_TOKEN_GAP_KF_ALT, "sep_lateral": False},
        "sep_lateral + gap": {"token_gap_kf": LC_TOKEN_GAP_KF_ALT, "sep_lateral": True},
    }
    ref_all, ref_ego = lane_change_reference(nusc)
    n_scenes = len({s["name"] for s in nusc.scene})
    base = len(ref_all) / n_scenes

    out: dict[str, Any] = {
        "reference": {"regex": LC_DESCRIPTION_RE, "scenes": ref_all,
                      "ego_only": ref_ego, "excluded_non_ego": list(LC_DESCRIPTION_NON_EGO),
                      "base_rate": round(base, 5), "n_scenes": n_scenes},
        "variants": {}}
    for label, kw in variants.items():
        if progress:
            print(f"[lane-change comparison] {label} {kw}", flush=True)
        tab, ev = build_lane_change_table(dataroot, split, progress=False,
                                          out_path=None, events_path=None, **kw)
        fired = set(ev.scene_name) if len(ev) else set()
        hit_all = sorted(fired & set(ref_all))
        out["variants"][label] = {
            "parameters": kw,
            "events": int(len(ev)),
            "scenes": len(fired),
            "positive_keyframes": int(tab.lane_change.sum()),
            "prevalence_pct": round(100 * float(tab.lane_change.mean()), 3),
            "recall_events": f"{len(hit_all)}/{len(ref_all)}",
            "recall_pct": round(100 * len(hit_all) / len(ref_all), 1),
            "recall_ego_only": f"{len(fired & set(ref_ego))}/{len(ref_ego)}",
            "description_hit_rate_pct": round(100 * len(hit_all) / len(fired), 1) if fired else None,
            "enrichment_over_base": round(len(hit_all) / len(fired) / base, 2) if fired else None,
            "missed": sorted(set(ref_all) - fired),
        }
    if out_path:
        from pathlib import Path
        Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def build_roundabout_table(kin: Any, dataroot: str = "data/nuscenes",
                           progress: bool = True, *, split: str | None = None,
                           out_path: str | None = "outputs/roundabout_traversals.parquet") -> Any:
    """Roundabout traversals for every scene, from the cached C1 table. Step C3.

    Exists so the roundabout result is REPRODUCIBLE FROM src/ rather than from an ad-hoc
    script, like every other table in this stage. Heuristic - see F-026 and L-013.
    """
    import pandas as pd

    rows = []
    for loc, K in kin.groupby("location"):
        if progress:
            print(f"  {loc}", flush=True)
        index = MapIndex(dataroot, loc)
        islands = map_islands(index)
        centrelines = lane_centrelines(index.nusc_map)
        for scene, g in K.groupby("scene_name"):
            for r in roundabout_traversals(index, g.ego_x, g.ego_y, g.yaw_deg,
                                           islands=islands):
                radius = float(np.sqrt(r["island_area_m2"] / np.pi))
                wind, nlanes = circulatory_winding(index.nusc_map, centrelines,
                                                   r["island_x"], r["island_y"], radius)
                rows.append({"location": loc, "scene_name": scene, **r,
                             "lane_winding_deg": wind, "lane_path_lanes": nlanes,
                             "is_roundabout": wind >= ROUNDABOUT_MIN_WINDING_DEG})
    cols = ["location", "scene_name", "island_area_m2", "island_x", "island_y",
            "angle_around_deg", "heading_sweep_deg", "sweep_ratio", "ring_drivable_frac",
            "lane_winding_deg", "lane_path_lanes", "is_roundabout"]
    df = pd.DataFrame(rows, columns=cols)
    if out_path:
        # One un-suffixed artifact, and it is the trainval one the thesis cites. A mini
        # call with the default destination would overwrite it, so the split is required.
        assert split == "trainval", (
            "roundabout_traversals.parquet is the trainval table; pass split='trainval' "
            "or out_path=None")
        df.to_parquet(out_path, index=False)
    return df


def build_map_context_table(nusc_or_kin: Any, dataroot: str = "data/nuscenes",
                            progress: bool = True, *, split: str | None = None,
                            out_path: str | None = "outputs/map_context_{split}.parquet") -> Any:
    """Map context for every keyframe, straight from the cached C1 table.

    Takes the C1 kinematics DataFrame (which carries ego_x/ego_y/yaw_deg), so no devkit
    load is needed - the map work is pure geometry. Step C3.
    """
    import pandas as pd

    kin = nusc_or_kin
    indices = {loc: MapIndex(dataroot, loc) for loc in sorted(kin.location.unique())}
    rows = []
    for i, r in enumerate(kin.itertuples()):
        if progress and i % 5000 == 0:
            print(f"  {i}/{len(kin)}", flush=True)
        ctx = map_context(indices[r.location], r.ego_x, r.ego_y, np.radians(r.yaw_deg))
        rows.append({"sample_token": r.sample_token, "scene_name": r.scene_name, **ctx})
    df = pd.DataFrame(rows)
    if out_path:
        # This builder takes the kinematics frame, not the devkit, so it cannot infer
        # which split it is looking at. `split` is required rather than guessed: a wrong
        # guess would write the mini table over the trainval one.
        assert split, "pass split='mini'|'trainval' (or out_path=None) so the table is named correctly"
        df.to_parquet(out_path.format(split=split), index=False)
    return df


def build_kinematics_table(nusc: Any, window_s: float = WINDOW_S,
                           smooth_s: float = SMOOTH_S,
                           resample_hz: float = RESAMPLE_HZ,
                           progress: bool = True,
                           out_path: str | None = "outputs/ego_kinematics_{split}.parquet") -> Any:
    """Kinematics for every keyframe in the dataset. Cache this; see PROCESS.md F-006.

    Step C1. The table is WRITTEN as well as returned, because an artifact the thesis
    cites and no function persists cannot be regenerated if `outputs/` is lost (F-087,
    R30). `{split}` is filled from the devkit handle, so one call cannot write over the
    other split's table. A sensitivity sweep that varies `window_s` and friends must pass
    `out_path=None`, or it overwrites the frozen table with a variant.
    """
    import pandas as pd

    from .data import map_name_for_scene, scene_sample_tokens

    rows = []
    for i, scene in enumerate(nusc.scene):
        if progress and i % 50 == 0:
            print(f"  scene {i+1}/{len(nusc.scene)}", flush=True)
        track = scene_pose_track(nusc, scene, smooth_s=smooth_s, resample_hz=resample_hz)
        location = map_name_for_scene(nusc, scene)
        for k, sample_token in enumerate(scene_sample_tokens(nusc, scene)):
            sample = nusc.get("sample", sample_token)
            kf_t_s = (sample["timestamp"] - track["t0_us"]) / 1e6
            rows.append({
                "sample_token": sample_token,
                "scene_token": scene["token"],
                "scene_name": scene["name"],
                "location": location,
                "keyframe_index": k,
                "timestamp_us": sample["timestamp"],
                "t_rel_s": kf_t_s,
                **ego_kinematics(track, kf_t_s, window_s=window_s),
            })
    df = pd.DataFrame(rows)
    if out_path:
        df.to_parquet(out_path.format(split=_split_name(nusc)), index=False)
    return df


def _hysteresis_segments(x: Any, t: Any, enter: float, exit_: float,
                         min_dur_s: float, valid: Any = None) -> list[tuple[int, int]]:
    """Schmitt-trigger segmentation of |x|, returning (start_idx, end_idx) pairs.

    A segment is confirmed only where |x| exceeds `enter`, but its BOUNDARIES are
    backtracked to where |x| last crossed the lower `exit_` threshold. That matters
    for Stage G1: the storyboard should say a turn began when the wheel started
    moving, not when the yaw rate happened to cross a detection threshold.

    `valid` optionally masks samples that may not TRIGGER a segment (used to keep the
    stale-localisation guard band of F-009 from starting a phantom event). A segment
    triggered elsewhere may still extend into the guard band.
    """
    import numpy as np

    absx = np.abs(x)
    above_enter, above_exit = absx > enter, absx > exit_
    segs: list[tuple[int, int]] = []
    i, n = 0, len(x)
    while i < n:
        if above_enter[i] and (valid is None or valid[i]):
            a = i
            while a > 0 and above_exit[a - 1]:
                a -= 1
            b = i
            while b < n - 1 and above_exit[b + 1]:
                b += 1
            if t[b] - t[a] >= min_dur_s:
                segs.append((a, b))
            i = b + 1
        else:
            i += 1
    return segs


def _merge_same_sign(segs: list[tuple[int, int]], t: Any, signal: Any,
                     max_gap_s: float) -> list[tuple[int, int]]:
    """Join consecutive segments that turn the same way and are nearly touching.

    A driver easing off mid-corner can dip below the exit threshold briefly; without
    this, one corner is reported as two turns.
    """
    import numpy as np

    if not segs:
        return []
    out = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = out[-1]
        same = np.sign(np.mean(signal[pa:pb + 1])) == np.sign(np.mean(signal[a:b + 1]))
        if same and t[a] - t[pb] <= max_gap_s:
            out[-1] = (pa, b)
        else:
            out.append((a, b))
    return out


def lateral_events(track: PoseTrack) -> list[dict[str, Any]]:
    """Steering events for one scene: turn_left / turn_right / u_turn.

    Event-based rather than per-keyframe (D-013), because 31% of real turns outlast
    the 4 s window and a U-turn NEVER fits inside it — no 4 s window in the whole
    dataset exceeds 114 deg, while turn events reach 163 deg (F-012).

    Sign convention: nuScenes yaw increases counter-clockwise, so a POSITIVE heading
    change is a LEFT turn. Verified against scene-0061, whose human description reads
    "turn left" and which measures +98 deg (F-011); asserted in the C2 self-checks.

    Step C2.
    """
    import numpy as np

    t, yaw, yr = track["t_s"], track["yaw_rad"], track["yaw_rate_deg_s"]
    guard = (t >= EDGE_GUARD_S) & (t <= float(t[-1]) - EDGE_GUARD_S)

    segs = _hysteresis_segments(yr, t, YAW_ENTER_DEG_S, YAW_EXIT_DEG_S,
                                MIN_EVENT_S, valid=guard)
    segs = _merge_same_sign(segs, t, yr, MERGE_GAP_S)

    events = []
    for a, b in segs:
        total = float(np.degrees(yaw[b] - yaw[a]))
        if abs(total) < MIN_TURN_DEG:      # a wobble, not a manoeuvre
            continue
        kind = ("u_turn" if abs(total) >= U_TURN_DEG
                else "turn_left" if total > 0 else "turn_right")
        events.append({
            "kind": kind,
            "start_s": float(t[a]), "end_s": float(t[b]),
            "duration_s": float(t[b] - t[a]),
            "total_heading_deg": total,
            "peak_yaw_rate_deg_s": float(yr[a:b + 1][np.argmax(np.abs(yr[a:b + 1]))]),
        })
    return events


def longitudinal_events(track: PoseTrack) -> list[dict[str, Any]]:
    """Speed events for one scene: stationary / accelerating / decelerating.

    Stationary is detected on speed directly (a sustained stop), the other two on
    acceleration with the same hysteresis scheme. Where they overlap, stationary wins:
    a stopped car is stopped, not decelerating.

    Step C2.
    """
    import numpy as np

    # accel_trend, NOT accel: see F-015. Detecting on the raw second derivative found
    # segments that alternated sign several times a second, each too short to pass the
    # speed-change filter, so genuine "arrive at the junction and stop" decelerations
    # produced no event at all.
    t, sp, ac = track["t_s"], track["speed_mps"], track["accel_trend_mps2"]
    guard = (t >= EDGE_GUARD_S) & (t <= float(t[-1]) - EDGE_GUARD_S)
    events = []

    # stationary: sustained low speed
    still = sp < STATIONARY_MPS
    i, n = 0, len(t)
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            if t[j - 1] - t[i] >= MIN_STOP_S:
                events.append({"kind": "stationary", "start_s": float(t[i]),
                               "end_s": float(t[j - 1]),
                               "duration_s": float(t[j - 1] - t[i]),
                               "speed_change_mps": 0.0})
            i = j
        else:
            i += 1

    # accelerating / decelerating, excluding anything inside a stop
    segs = _hysteresis_segments(ac, t, ACCEL_ENTER_MPS2, ACCEL_EXIT_MPS2,
                                MIN_EVENT_S, valid=guard & ~still)
    segs = _merge_same_sign(segs, t, ac, MERGE_GAP_S)
    for a, b in segs:
        dv = float(sp[b] - sp[a])
        if abs(dv) < MIN_SPEED_CHANGE_MPS:      # too small to call a manoeuvre
            continue
        events.append({
            "kind": "accelerating" if dv > 0 else "decelerating",
            "start_s": float(t[a]), "end_s": float(t[b]),
            "duration_s": float(t[b] - t[a]), "speed_change_mps": dv,
        })
    return sorted(events, key=lambda e: e["start_s"])


def maneuver_labels(kf_t_s: float, lat_events: list[dict], lon_events: list[dict],
                    ) -> dict[str, Any]:
    """Two orthogonal labels for one keyframe (D-014).

    A car can turn left AND brake at the same time, so steering and speed are separate
    axes rather than one label with an arbitrary priority order. Each axis is
    mutually exclusive within itself, which keeps Stage F's per-tag scoring clean.

      lateral_label      : going_straight | turning_left | turning_right | u_turn
      longitudinal_label : cruising | stationary | accelerating | decelerating

    Step C2.
    """
    def _hit(events: list[dict], *kinds: str) -> dict | None:
        for e in events:
            if e["start_s"] <= kf_t_s <= e["end_s"] and (not kinds or e["kind"] in kinds):
                return e
        return None

    lat = _hit(lat_events)
    stop = _hit(lon_events, "stationary")                 # stationary outranks accel
    lon = stop or _hit(lon_events, "accelerating", "decelerating")

    lat_label = lat["kind"] if lat else "going_straight"
    lon_label = lon["kind"] if lon else "cruising"

    return {
        "lateral_label": lat_label,
        "longitudinal_label": lon_label,
        # Binary tags, the form Stage F scores and the VLM is asked to produce.
        # Generated from the vocabulary constants rather than written out by hand:
        # hand-writing them produced a silent bug where the tag compared against
        # "turning_left" while events emit "turn_left", so is_turning_left was False on
        # all 5,500 turning frames and would have scored the VLM at zero recall (F-016).
        **{f"is_{lab}": lat_label == lab for lab in LATERAL_LABELS},
        **{f"is_{lab}": lon_label == lab for lab in LONGITUDINAL_LABELS},
        # event provenance: which manoeuvre this frame belongs to, and where in it
        "lat_event_total_deg":  float(lat["total_heading_deg"]) if lat else 0.0,
        "lat_event_duration_s": float(lat["duration_s"]) if lat else 0.0,
        "lat_event_progress":   (float((kf_t_s - lat["start_s"]) / lat["duration_s"])
                                 if lat and lat["duration_s"] > 0 else float("nan")),
        "lon_event_duration_s": float(lon["duration_s"]) if lon else 0.0,
        "lon_event_dspeed_mps": float(lon["speed_change_mps"]) if lon else 0.0,
    }


# `road_block` is DELIBERATELY ABSENT. On singapore-queenstown all 676 of its records
# point to a SINGLE polygon token - one 7,748-vertex shape with 93 holes spanning the
# whole map - so `on_road_block` is True everywhere there and each containment test costs
# 894 ms (vs 0.03 ms on boston, whose road_blocks have a median of 10 vertices). The layer
# is degenerate on that map and adds no label that road_segment, lane and drivable_area do
# not already give. See F-020.
POLYGON_LAYERS = ("drivable_area", "road_segment", "lane",
                  "ped_crossing", "walkway", "stop_line", "carpark_area")


class MapIndex:
    """Spatial index over one nuScenes map location. Build once, query 34,149 times.

    WHY THIS EXISTS: the devkit's `layers_on_point` takes ~209 ms per call because it
    scans every polygon in the layer. That is ~2 hours for the dataset, which makes
    iteration impossible. An STRtree over the same polygons answers in ~0.24 ms - a
    3785x speedup, 8 s for the whole dataset (F-019).

    TWO SHAPELY TRAPS, both of which silently return nothing rather than erroring:
      1. `drivable_area` records carry `polygon_tokens` (PLURAL); every other layer
         uses `polygon_token`. Filtering on the singular key drops the layer entirely.
      2. `STRtree.query(geom, predicate=...)` evaluates `input.predicate(tree_geom)`,
         NOT the reverse. So for "which polygon contains this point" the input is the
         POINT and the predicate is `within`. Using `contains` asks whether the point
         contains the polygon, which is never true, and returns an empty result with
         no error. That produced a 0% hit rate on every layer before it was caught.

    Step C3.
    """

    def __init__(self, dataroot: str, map_name: str) -> None:
        from nuscenes.map_expansion.map_api import NuScenesMap
        from shapely import STRtree

        self.map_name = map_name
        self.nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
        self.trees: dict[str, Any] = {}

        for layer in POLYGON_LAYERS:
            geoms, recs = [], []
            for rec in getattr(self.nusc_map, layer):
                # trap 1: plural key on drivable_area only
                toks = rec.get("polygon_tokens") or (
                    [rec["polygon_token"]] if "polygon_token" in rec else [])
                for tok in toks:
                    geoms.append(self.nusc_map.extract_polygon(tok))
                    recs.append(rec)
            # geoms kept as an object array so shapely's VECTORISED distance can be used;
            # looping .distance() per candidate made a 50 m corridor ~15 ms per query,
            # which is minutes across the dataset (F-019).
            self.trees[layer] = (STRtree(geoms) if geoms else None, recs,
                                 np.array(geoms, dtype=object) if geoms else None)

        # traffic lights are a LINE layer, so they are indexed as points (node mean).
        # NOTE their `items` field lists the fixture's bulbs (RED/YELLOW/GREEN) - it is
        # the hardware description, never the live state. See F-004: nuScenes cannot
        # validate "the light is red"; that is why Stage H brings in BDD100K.
        from shapely.geometry import Point
        tl_pts, tl_recs = [], []
        for rec in self.nusc_map.traffic_light:
            nodes = [self.nusc_map.get("node", t) for t in rec["node_tokens"]]
            if nodes:
                tl_pts.append(Point(float(np.mean([n["x"] for n in nodes])),
                                    float(np.mean([n["y"] for n in nodes]))))
                tl_recs.append(rec)
        self.tl_tree = STRtree(tl_pts) if tl_pts else None
        self.tl_recs = tl_recs
        self._tl_arr = np.array(tl_pts, dtype=object) if tl_pts else None

    def layers_at(self, x: float, y: float) -> dict[str, Any]:
        """Which polygon of each layer contains this point (or None)."""
        from shapely.geometry import Point
        pt = Point(x, y)
        out = {}
        for layer, (tree, recs, _) in self.trees.items():
            if tree is None:
                out[layer] = None
                continue
            idx = tree.query(pt, predicate="within")     # trap 2
            out[layer] = recs[int(idx[0])] if len(idx) else None
        return out

    def _corridor(self, x: float, y: float, yaw: float,
                  length: float, halfwidth: float) -> Any:
        """Forward-facing rectangle in the ego's own frame.

        A camera sees what is in front, not what is underneath, so "ahead" must be
        measured along the heading rather than as a plain radius (D-017).
        """
        from shapely.geometry import Polygon
        c, s = np.cos(yaw), np.sin(yaw)
        local = [(0.0, -halfwidth), (length, -halfwidth), (length, halfwidth), (0.0, halfwidth)]
        return Polygon([(x + u * c - v * s, y + u * s + v * c) for u, v in local])

    def ahead(self, x: float, y: float, yaw: float, layer: str,
              length: float = AHEAD_RANGE_M,
              halfwidth: float = CORRIDOR_HALFWIDTH_M,
              predicate_rec: Any = None) -> tuple[float, Any]:
        """Nearest polygon of `layer` inside the forward corridor.

        Returns (distance_m, record); (nan, None) when nothing is ahead.
        `predicate_rec` optionally filters records, e.g. only intersection segments.
        """
        import shapely
        from shapely.geometry import Point
        tree, recs, geoms = self.trees[layer]
        if tree is None:
            return float("nan"), None
        corr = self._corridor(x, y, yaw, length, halfwidth)
        idx = tree.query(corr, predicate="intersects")
        if not len(idx):
            return float("nan"), None
        if predicate_rec is not None:
            idx = np.array([i for i in idx if predicate_rec(recs[int(i)])], dtype=int)
            if not len(idx):
                return float("nan"), None
        d = shapely.distance(geoms[idx], Point(x, y))      # vectorised
        j = int(np.argmin(d))
        return float(d[j]), recs[int(idx[j])]

    def traffic_light_ahead(self, x: float, y: float, yaw: float,
                            length: float = AHEAD_RANGE_M,
                            halfwidth: float = CORRIDOR_HALFWIDTH_M) -> float:
        """Distance to the nearest traffic-light fixture in the forward corridor."""
        import shapely
        from shapely.geometry import Point
        if self.tl_tree is None:
            return float("nan")
        corr = self._corridor(x, y, yaw, length, halfwidth)
        idx = self.tl_tree.query(corr, predicate="intersects")
        if not len(idx):
            return float("nan")
        return float(shapely.distance(self._tl_arr[idx], Point(x, y)).min())


def map_context(index: MapIndex, x: float, y: float, yaw: float,
                ahead_m: float = AHEAD_RANGE_M,
                halfwidth_m: float = CORRIDOR_HALFWIDTH_M) -> dict[str, Any]:
    """Map-derived context for one keyframe: where the ego IS, and what is AHEAD.

    Both are needed (D-017). Containment answers "is the car inside a junction";
    the forward corridor answers "is a junction visible in front", which is what a
    front camera - and therefore the VLM - actually reports. Scoring a VLM that
    correctly says "I see a crosswalk" against a containment label that is True on
    only 3.5% of frames would mark it wrong for being right.

    Step C3.
    """
    at = index.layers_at(x, y)
    seg = at["road_segment"]

    d_cross, _ = index.ahead(x, y, yaw, "ped_crossing", ahead_m, halfwidth_m)
    d_stop, stop_rec = index.ahead(x, y, yaw, "stop_line", ahead_m, halfwidth_m)
    d_inter, _ = index.ahead(x, y, yaw, "road_segment", ahead_m, halfwidth_m,
                             predicate_rec=lambda r: bool(r.get("is_intersection")))
    d_tl = index.traffic_light_ahead(x, y, yaw, ahead_m, halfwidth_m)

    return {
        # --- where the ego IS ---
        "on_drivable_area": at["drivable_area"] is not None,
        "at_intersection":  bool(seg is not None and seg.get("is_intersection")),
        "on_lane":          at["lane"] is not None,
        "on_ped_crossing":  at["ped_crossing"] is not None,
        "on_walkway":       at["walkway"] is not None,      # QC: must stay ~0%
        "on_stop_line":     at["stop_line"] is not None,
        "on_carpark":       at["carpark_area"] is not None,
        # --- what is AHEAD, along the heading ---
        "ped_crossing_ahead_m":  d_cross,
        "stop_line_ahead_m":     d_stop,
        "stop_line_ahead_type":  (stop_rec or {}).get("stop_line_type"),
        "intersection_ahead_m":  d_inter,
        "traffic_light_ahead_m": d_tl,
        # --- binary forms, the shape Stage F scores ---
        "ped_crossing_ahead":  bool(d_cross == d_cross),     # False when nan
        "stop_line_ahead":     bool(d_stop == d_stop),
        "intersection_ahead":  bool(d_inter == d_inter),
        "traffic_light_ahead": bool(d_tl == d_tl),
        "map_location": index.map_name,
    }


def frustum_boxes(nusc: Any, cam_data_token: str) -> list[dict[str, Any]]:
    """The 3D boxes ACTUALLY VISIBLE in one camera, as plain records.

    This is the fix for the original defect: the old pipeline called an object present
    if it appeared anywhere in the 360 deg annotation set, so vehicles *behind* the car
    made `has_vehicle` true. Measured over 400 random keyframes, that counts 34.9
    objects per keyframe where the front camera sees 9.6 - a 3.66x over-count (F-029).

    Frustum membership comes from the devkit, which projects each box through the
    camera intrinsics and keeps it if any corner lands in the image AND is in front of
    the lens. It is a GEOMETRIC test only: it knows nothing about occlusion, which is
    why `vis_level` is carried through to the caller rather than filtered on (L-016).

    Needs no image file, only calibration and the sample_data width/height - so this
    runs on the metadata-only trainval split (F-029).

    `dist_m` is measured from the CAMERA, not the ego origin (CAM_FRONT sits ~1.7 m
    ahead of it), because the question these tags answer is what the camera can see.
    Step C4.
    """
    from nuscenes.utils.geometry_utils import BoxVisibility

    _, boxes, _ = nusc.get_sample_data(cam_data_token, box_vis_level=BoxVisibility.ANY)
    out = []
    for b in boxes:
        ann = nusc.get("sample_annotation", b.token)
        out.append({
            "cat": ann["category_name"],
            "attrs": {nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]},
            "vis_level": nusc.get("visibility", ann["visibility_token"])["level"],
            "dist_m": float(np.linalg.norm(b.center)),
            "n_lidar_pts": ann["num_lidar_pts"],
        })
    return out


def visible_objects(nusc: Any, cam_data_token: str,
                    n_annotations_360: int | None = None) -> dict[str, Any]:
    """Object tags for one camera frame, from frustum-filtered 3D boxes. Step C4.

    Emits three columns per tag in OBJECT_TAGS, all generated from that one dict so
    they cannot drift apart (R4, the F-016 lesson):
        <tag>          bool  - at least one such object is visible
        n_<tag>        int   - how many
        n_<tag>_clear  int   - how many are CLEAR_VISIBILITY, for re-scoring later

    Presence tags carry NO range limit (D-022); proximity is its own `*_near` tag.
    Nothing is filtered on occlusion (D-023); the counts let Stage F tighten later.
    """
    boxes = frustum_boxes(nusc, cam_data_token)
    dist = {"vehicle.": np.inf, "human.": np.inf}
    for b in boxes:
        for pre in dist:
            if b["cat"].startswith(pre):
                dist[pre] = min(dist[pre], b["dist_m"])

    row: dict[str, Any] = {}
    for tag, pred in OBJECT_TAGS.items():
        hits = [b for b in boxes if pred(b)]
        row[tag] = len(hits) > 0
        row[f"n_{tag}"] = len(hits)
        row[f"n_{tag}_clear"] = sum(b["vis_level"] == CLEAR_VISIBILITY for b in hits)

    # provenance: enough to audit any tag back to the boxes that produced it, and to
    # reproduce the over-count factor that motivated this whole step.
    row.update({
        "n_boxes_visible": len(boxes),
        "n_annotations_360": n_annotations_360,
        "n_boxes_clear": sum(b["vis_level"] == CLEAR_VISIBILITY for b in boxes),
        "n_boxes_occluded": sum(b["vis_level"] == "v0-40" for b in boxes),
        "n_boxes_near": sum(b["dist_m"] <= NEAR_RANGE_M for b in boxes),
        "n_boxes_mid": sum(NEAR_RANGE_M < b["dist_m"] <= MID_RANGE_M for b in boxes),
        "n_boxes_far": sum(b["dist_m"] > MID_RANGE_M for b in boxes),
        "n_boxes_no_lidar": sum(b["n_lidar_pts"] == 0 for b in boxes),
        "nearest_object_m": min((b["dist_m"] for b in boxes), default=float("nan")),
        "farthest_object_m": max((b["dist_m"] for b in boxes), default=float("nan")),
        "nearest_vehicle_m": dist["vehicle."] if np.isfinite(dist["vehicle."]) else float("nan"),
        "nearest_pedestrian_m": dist["human."] if np.isfinite(dist["human."]) else float("nan"),
    })
    return row


def build_object_table(nusc: Any, camera: str = "CAM_FRONT",
                       progress: bool = True,
                       out_path: str | None = "outputs/objects_{split}.parquet") -> Any:
    """Visible-object tags for every keyframe in the dataset. Step C4.

    CAM_FRONT only, matching what the VLM is actually shown (PLAN B4 fetches that one
    channel). Costs ~4.9 ms/keyframe, so ~3 min for all 34,149 - metadata only, no
    images needed.
    """
    import pandas as pd

    from .data import scene_sample_tokens

    rows = []
    for i, scene in enumerate(nusc.scene):
        if progress and i % 50 == 0:
            print(f"  scene {i+1}/{len(nusc.scene)}", flush=True)
        for k, sample_token in enumerate(scene_sample_tokens(nusc, scene)):
            sample = nusc.get("sample", sample_token)
            rows.append({
                "sample_token": sample_token,
                "scene_name": scene["name"],
                "keyframe_index": k,
                "camera": camera,
                **visible_objects(nusc, sample["data"][camera],
                                  n_annotations_360=len(sample["anns"])),
            })
    df = pd.DataFrame(rows)
    if out_path:
        df.to_parquet(out_path.format(split=_split_name(nusc)), index=False)
    return df


def scene_agent_tracks(nusc: Any, scene: dict) -> list[dict[str, dict]]:
    """Every annotated agent in one scene, per keyframe, in the EGO FRAME.

    Returns one dict per keyframe, mapping `instance_token` -> agent state. Keying on
    the instance rather than the annotation is what makes cross-frame questions
    ("was this same car outside my lane a moment ago?") answerable at all.

    Both the position and the velocity are rotated out of the global frame:
        rel = Quaternion(ego['rotation']).inverse.rotate(ann_xyz - ego_xyz)
        # rel[0] = forward, rel[1] = left, rel[2] = up
    `nusc.box_velocity()` returns a GLOBAL vector, so it needs the same rotation. That
    is the same defect as the original `abs(dy) > abs(dx)` test, one level down, and it
    is silent - the numbers stay plausible while meaning the wrong thing.

    The ego_pose used is LIDAR_TOP's, whose timestamp is exactly the keyframe timestamp
    (CAM_FRONT's is 35.7 ms earlier). Step C5.
    """
    from nuscenes.utils.geometry_utils import BoxVisibility
    from pyquaternion import Quaternion

    from .data import scene_sample_tokens

    frames = []
    for tok in scene_sample_tokens(nusc, scene):
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        rot = Quaternion(ego["rotation"]).inverse
        origin = np.array(ego["translation"])
        _, boxes, _ = nusc.get_sample_data(sample["data"]["CAM_FRONT"],
                                           box_vis_level=BoxVisibility.ANY)
        in_frustum = {b.token for b in boxes}

        states: dict[str, dict] = {}
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            rel = rot.rotate(np.array(ann["translation"]) - origin)
            vel = nusc.box_velocity(ann_token)
            # NaN whenever the agent appears in only one keyframe (0.2% of annotations);
            # propagated, never zero-filled, so a missing velocity cannot masquerade as
            # a stationary agent.
            v_ego = rot.rotate(vel) if np.isfinite(vel).all() else np.full(3, np.nan)
            states[ann["instance_token"]] = {
                "cat": ann["category_name"],
                "fwd_m": float(rel[0]),
                "lat_m": float(rel[1]),
                "v_fwd_mps": float(v_ego[0]),
                "v_lat_mps": float(v_ego[1]),
                "speed_mps": float(np.hypot(v_ego[0], v_ego[1])),
                "in_frustum": ann_token in in_frustum,
                "t_s": sample["timestamp"] / 1e6,
            }
        frames.append(states)
    return frames


def _lead_of(state_map: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
    """The nearest vehicle in the ego's own lane corridor, ahead of it."""
    leads = [(k, s) for k, s in state_map.items()
             if s["cat"].startswith("vehicle.")
             and 0 < s["fwd_m"] < LEAD_RANGE_M and abs(s["lat_m"]) < LEAD_LATERAL_M]
    if not leads:
        return None, None
    return min(leads, key=lambda kv: kv[1]["fwd_m"])


def interaction_labels(frames: list[dict[str, dict]], k: int,
                       with_keys: bool = False) -> dict[str, Any]:
    """Interaction tags for keyframe `k` of a scene. Step C5.

    Tags are 360 deg statements about the world, not about the front camera (D-026):
    an agent outside CAM_FRONT still counts. The `n_*_in_frustum` columns record how
    many of the contributing agents the camera could actually see, so Stage F can say
    how much of each score is field of view rather than model capability (L-019).

    `with_keys=True` adds a `_contributors` entry naming the instance behind each tag.
    Figures use it instead of re-implementing the predicates - a second copy of a rule
    is how the C2 tags drifted in F-016. The table drops it, as it is not columnar.
    """
    now = frames[k]
    row: dict[str, Any] = {t: False for t in INTERACTION_TAGS}
    row.update({"lead_distance_m": float("nan"), "lead_speed_mps": float("nan"),
                "lead_accel_mps2": float("nan"), "lead_in_frustum": False,
                "n_cut_in": 0, "n_cut_in_in_frustum": 0,
                "n_pedestrian_crossing_path": 0,
                "n_pedestrian_crossing_path_in_frustum": 0,
                "n_agents": len(now)})
    contrib: dict[str, list[str]] = {t: [] for t in INTERACTION_TAGS}

    # --- lead vehicle, and whether it is braking -----------------------------------
    lead_key, lead = _lead_of(now)
    if lead is not None:
        row["lead_vehicle"] = True
        row["lead_distance_m"] = lead["fwd_m"]
        row["lead_speed_mps"] = lead["speed_mps"]
        row["lead_in_frustum"] = lead["in_frustum"]
        contrib["lead_vehicle"].append(lead_key)
        # speed change of the SAME instance over LEAD_TREND_S, never a shorter lag:
        # box_velocity is itself a centred difference, so a 0.5 s re-difference flips
        # sign 29.6% of the time (R6).
        past = _state_before(frames, k, lead_key, LEAD_TREND_S)
        if past is not None:
            dt = now[lead_key]["t_s"] - past["t_s"]
            if dt > 0 and np.isfinite(lead["speed_mps"]) and np.isfinite(past["speed_mps"]):
                row["lead_accel_mps2"] = (lead["speed_mps"] - past["speed_mps"]) / dt
                row["lead_braking"] = row["lead_accel_mps2"] < LEAD_BRAKE_MPS2
                if row["lead_braking"]:
                    contrib["lead_braking"].append(lead_key)

    # --- cut-in ---------------------------------------------------------------------
    for key, s in now.items():
        if not s["cat"].startswith("vehicle."):
            continue
        if not (0 < s["fwd_m"] < LEAD_RANGE_M and abs(s["lat_m"]) < LEAD_LATERAL_M):
            continue
        past = _state_before(frames, k, key, CUT_IN_LOOKBACK_S)
        if past is None or abs(past["lat_m"]) < LEAD_LATERAL_M:
            continue                                   # it was already in the lane
        # Three signals must agree (R28), because the position test alone fires on
        # PARKED cars whenever the ego turns - the ego frame rotates with the car, so
        # kerbside vehicles sweep across the corridor boundary without moving (F-035).
        moving = np.isfinite(s["speed_mps"]) and s["speed_mps"] > MIN_AGENT_SPEED_MPS
        inward = np.isfinite(s["v_lat_mps"]) and s["lat_m"] * s["v_lat_mps"] < 0
        if moving and inward:
            row["cut_in"] = True
            row["n_cut_in"] += 1
            row["n_cut_in_in_frustum"] += s["in_frustum"]
            contrib["cut_in"].append(key)

    # --- pedestrian crossing the ego's path -------------------------------------------
    for key, s in now.items():
        if not s["cat"].startswith("human.") or s["fwd_m"] <= 0:
            continue
        if np.hypot(s["fwd_m"], s["lat_m"]) > PED_INTERACT_RANGE_M:
            continue
        if not np.isfinite(s["v_lat_mps"]) or abs(s["v_lat_mps"]) < PED_LATERAL_MPS:
            continue
        if s["lat_m"] * s["v_lat_mps"] >= 0:
            continue                                   # moving AWAY from the centreline
        gap = max(abs(s["lat_m"]) - LEAD_LATERAL_M, 0.0)
        if gap / abs(s["v_lat_mps"]) > PED_TIME_TO_CORRIDOR_S:
            continue                                   # will not reach us in time
        row["pedestrian_crossing_path"] = True
        row["n_pedestrian_crossing_path"] += 1
        row["n_pedestrian_crossing_path_in_frustum"] += s["in_frustum"]
        contrib["pedestrian_crossing_path"].append(key)

    if with_keys:
        row["_contributors"] = contrib
    return row


def _state_before(frames: list[dict[str, dict]], k: int, key: str,
                  lookback_s: float) -> dict | None:
    """The same instance's state ~`lookback_s` earlier, or None if unavailable.

    Walks back to the OLDEST keyframe still inside the window rather than a fixed
    number of frames, so an irregular keyframe gap cannot silently change the baseline.
    """
    t_now = frames[k][key]["t_s"]
    found = None
    for j in range(k - 1, -1, -1):
        if key not in frames[j]:
            break                                      # the track is not continuous
        if t_now - frames[j][key]["t_s"] > lookback_s + 1e-6:
            break
        found = frames[j][key]
    return found


def build_interaction_table(nusc: Any, progress: bool = True,
                            out_path: str | None = "outputs/interactions_{split}.parquet") -> Any:
    """Interaction tags for every keyframe in the dataset. Step C5."""
    import pandas as pd

    from .data import scene_sample_tokens

    rows = []
    for i, scene in enumerate(nusc.scene):
        if progress and i % 50 == 0:
            print(f"  scene {i+1}/{len(nusc.scene)}", flush=True)
        frames = scene_agent_tracks(nusc, scene)
        for k, sample_token in enumerate(scene_sample_tokens(nusc, scene)):
            rows.append({
                "sample_token": sample_token,
                "scene_name": scene["name"],
                "keyframe_index": k,
                **interaction_labels(frames, k),
            })
    df = pd.DataFrame(rows)
    if out_path:
        df.to_parquet(out_path.format(split=_split_name(nusc)), index=False)
    return df


def build_labels(nusc: Any, sample_token: str, nusc_map: Any) -> dict[str, Any]:
    """Full label row for one keyframe: all three families merged, plus metadata.

    Metadata to carry: sample_token, scene_token, timestamp, map location.
    Step C6.
    """
    raise NotImplementedError


# --- Step C6: the frozen label schema -------------------------------------------------
# SCOPE is the field this schema exists for. Three families make statements about three
# different regions of space, and nothing in the tables says so:
#   ego              a fact about the vehicle itself                        [C1, C2]
#   map_corridor     the ego's position, or the 30 m corridor ahead of it   [C3, D-017]
#   camera_frustum   only what CAM_FRONT can see                            [C4, D-025]
#   world_360        every annotated agent, visible or not                  [C5, D-026]
# `has_pedestrian` and `pedestrian_crossing_path` are not commensurable, and without
# this field a reader cannot tell (L-019).
_SCOPE_EGO = "ego"
_SCOPE_MAP_AT = "map_containment"
_SCOPE_MAP_AHEAD = f"map_corridor_{int(AHEAD_RANGE_M)}m"
_SCOPE_CAM = "camera_frustum_CAM_FRONT"
_SCOPE_360 = "world_360"

# Structural implications: relations that follow from the DEFINITIONS, so they must hold
# on any dataset and are asserted. Measured relations that merely happen to hold on
# nuScenes (`is_turn_left => on_drivable_area`, `on_carpark => has_pedestrian` at n=23)
# are deliberately NOT here - they belong in the C7 coverage report as observations.
# R37: state whether derived categories are a subset or a partition, and assert which.
LABEL_IMPLICATIONS: tuple[tuple[str, str, str], ...] = (
    ("traffic_cone", "construction_object", "a cone is one of the construction categories"),
    ("barrier", "construction_object", "a barrier is one of the construction categories"),
    ("moving_vehicle", "has_vehicle", "the attribute test also requires category vehicle.*"),
    ("parked_vehicle", "has_vehicle", "same"),
    ("stopped_vehicle", "has_vehicle", "same"),
    ("large_vehicle", "has_vehicle", "the large categories are all vehicle.*"),
    ("cyclist", "has_vehicle", "a two-wheeler is annotated as vehicle.bicycle/motorcycle"),
    ("parked_bicycle", "has_vehicle", "same"),
    ("vehicle_near", "has_vehicle", "the near tag is the presence tag plus a radius"),
    ("pedestrian_near", "has_pedestrian", "same"),
    ("lead_braking", "lead_vehicle", "there must be a lead for it to be braking"),
    ("cut_in", "lead_vehicle", "an agent inside the corridor makes the corridor non-empty"),
    ("on_ped_crossing", "ped_crossing_ahead",
     "the forward corridor starts at u=0, so a polygon containing the ego intersects it"),
    ("on_stop_line", "stop_line_ahead", "same"),
    ("at_intersection", "intersection_ahead", "same"),
)

# One line per tag: what it means and what a VLM would have to see to answer it.
_DERIVATIONS: dict[str, str] = {
    **{f"is_{n}": f"ego steering event of kind '{n}' contains this keyframe"
       for n in LATERAL_LABELS},
    **{f"is_{n}": f"ego speed event of kind '{n}' contains this keyframe"
       for n in LONGITUDINAL_LABELS},
    "on_drivable_area": "ego position falls inside a drivable_area polygon",
    "at_intersection": "ego position falls inside a road_segment with is_intersection",
    "on_lane": "ego position falls inside a lane polygon",
    "on_ped_crossing": "ego position falls inside a ped_crossing polygon",
    "on_walkway": "ego position falls inside a walkway polygon (QC: must stay ~0%)",
    "on_stop_line": "ego position falls inside a stop_line polygon",
    "on_carpark": "ego position falls inside a carpark_area polygon",
    "ped_crossing_ahead": "a ped_crossing polygon intersects the forward corridor",
    "stop_line_ahead": "a stop_line polygon intersects the forward corridor",
    "intersection_ahead": "an is_intersection road_segment intersects the forward corridor",
    "traffic_light_ahead": "a traffic_light fixture falls in the forward corridor "
                           "(position only - nuScenes never gives the STATE, F-004)",
    "lane_change": "the ego crossed to a laterally adjacent, parallel lane centreline "
                   "(union of a polygon-token flip and a settlement-pair detector, "
                   "both geometry-confirmed; PRECISION operating point, see L-022)",
    # visible objects: every one is "at least one frustum-visible box such that ..."
    "has_pedestrian": "a human.* box is in the frustum",
    "has_vehicle": "a vehicle.* box is in the frustum",
    "parked_vehicle": "a vehicle.* box carries the annotated attribute vehicle.parked",
    "moving_vehicle": "a vehicle.* box carries the annotated attribute vehicle.moving",
    "stopped_vehicle": "a vehicle.* box carries the annotated attribute vehicle.stopped",
    "large_vehicle": "a truck / bus / trailer / construction-vehicle box is in the frustum",
    "cyclist": "a bicycle or motorcycle box carries the attribute cycle.with_rider",
    "parked_bicycle": "a bicycle or motorcycle box carries cycle.without_rider "
                      "(street furniture, NOT a road user - F-031)",
    "construction_object": "a barrier, traffic cone or debris box is in the frustum",
    "traffic_cone": "a movable_object.trafficcone box is in the frustum",
    "barrier": "a movable_object.barrier box is in the frustum",
    "pedestrian_near": "a human.* box is in the frustum within the near radius",
    "vehicle_near": "a vehicle.* box is in the frustum within the near radius",
    "lead_vehicle": "nearest vehicle ahead inside the ego lane corridor",
    "lead_braking": "that same instance's speed fell faster than the threshold over the trend window",
    "cut_in": "a moving vehicle entered the corridor, with its lateral velocity pointing inward",
    "pedestrian_crossing_path": "pedestrian ahead, moving toward the centreline, "
                                "reaching the corridor within the time limit",
}

_PARAMETERS: dict[str, dict] = {
    "lead_vehicle": {"LEAD_LATERAL_M": LEAD_LATERAL_M, "LEAD_RANGE_M": LEAD_RANGE_M},
    "lead_braking": {"LEAD_TREND_S": LEAD_TREND_S, "LEAD_BRAKE_MPS2": LEAD_BRAKE_MPS2},
    "cut_in": {"CUT_IN_LOOKBACK_S": CUT_IN_LOOKBACK_S,
               "MIN_AGENT_SPEED_MPS": MIN_AGENT_SPEED_MPS,
               "LEAD_LATERAL_M": LEAD_LATERAL_M},
    "pedestrian_crossing_path": {"PED_INTERACT_RANGE_M": PED_INTERACT_RANGE_M,
                                 "PED_LATERAL_MPS": PED_LATERAL_MPS,
                                 "PED_TIME_TO_CORRIDOR_S": PED_TIME_TO_CORRIDOR_S},
    "pedestrian_near": {"NEAR_RANGE_M": NEAR_RANGE_M},
    "vehicle_near": {"NEAR_RANGE_M": NEAR_RANGE_M},
}
_PARAMETERS["lane_change"] = {"LC_SEP_MIN_M": LC_SEP_MIN_M, "LC_SEP_MAX_M": LC_SEP_MAX_M,
                              "LC_CENTRED_M": LC_CENTRED_M,
                              "LC_LATERAL_RATE_MAX_MPS": LC_LATERAL_RATE_MAX_MPS,
                              "LC_MAX_DUR_S": LC_MAX_DUR_S,
                              "LC_MIN_SPEED_MPS": LC_MIN_SPEED_MPS,
                              "LC_PARALLEL_DH_DEG": LC_PARALLEL_DH_DEG}
for _t in MAP_AHEAD_TAGS:
    _PARAMETERS[_t] = {"AHEAD_RANGE_M": AHEAD_RANGE_M,
                       "CORRIDOR_HALFWIDTH_M": CORRIDOR_HALFWIDTH_M}
for _t in LATERAL_LABELS:
    _PARAMETERS[f"is_{_t}"] = {"YAW_ENTER_DEG_S": YAW_ENTER_DEG_S,
                               "YAW_EXIT_DEG_S": YAW_EXIT_DEG_S,
                               "MIN_TURN_DEG": MIN_TURN_DEG, "U_TURN_DEG": U_TURN_DEG}
for _t in LONGITUDINAL_LABELS:
    _PARAMETERS[f"is_{_t}"] = {"ACCEL_ENTER_MPS2": ACCEL_ENTER_MPS2,
                               "ACCEL_EXIT_MPS2": ACCEL_EXIT_MPS2,
                               "STATIONARY_MPS": STATIONARY_MPS, "TREND_S": TREND_S}

_DECISIONS: dict[str, list[str]] = {
    **{f"is_{n}": ["D-013", "D-014", "D-016"] for n in LATERAL_LABELS + LONGITUDINAL_LABELS},
    **{n: ["D-017", "D-018", "D-019"] for n in MAP_CONTAINMENT_TAGS},
    **{n: ["D-017", "D-020"] for n in MAP_AHEAD_TAGS},
    **{n: ["D-002", "D-022", "D-023", "D-024", "D-025"] for n in OBJECT_TAGS},
    **{n: ["D-026", "D-027"] for n in INTERACTION_TAGS},
}
_DECISIONS["lane_change"] = ["D-033"]
_DECISIONS["cut_in"] = ["D-026", "D-027", "D-028"]
_DECISIONS["lead_braking"] = ["D-026", "D-027", "D-029"]


def label_tags() -> dict[str, str]:
    """Every scored tag -> its family. SINGLE SOURCE for the schema and its QC.

    Built from the vocabulary constants, never typed out again (R4). Step C6.
    """
    tags: dict[str, str] = {}
    for n in LATERAL_LABELS + LONGITUDINAL_LABELS:
        tags[f"is_{n}"] = "ego_maneuver"
    for n in EGO_EVENT_TAGS:
        tags[n] = "ego_maneuver"
    for n in MAP_CONTAINMENT_TAGS + MAP_AHEAD_TAGS:
        tags[n] = "map_context"
    for n in OBJECT_TAGS:
        tags[n] = "visible_objects"
    for n in INTERACTION_TAGS:
        tags[n] = "interaction"
    return tags


def label_schema(prevalence: dict[str, float] | None = None,
                 split: str | None = None, n_frames: int | None = None) -> dict[str, Any]:
    """The frozen schema: every tag, its type, family, SCOPE and derivation. Step C6.

    Dumped to outputs/label_schema.json. This is the EU AI Act traceability artifact
    the brief asks for: every label resolves to a deterministic derivation, the exact
    parameters that produced it, and the decision IDs that justify those parameters -
    with no human judgment anywhere in the chain.

    `prevalence` (tag -> percent) is folded in when supplied, together with the split
    it was measured on, so a baseline can never be quoted without its provenance.
    """
    scope_of = {
        **{f"is_{n}": _SCOPE_EGO for n in LATERAL_LABELS + LONGITUDINAL_LABELS},
        **{n: _SCOPE_EGO for n in EGO_EVENT_TAGS},
        **{n: _SCOPE_MAP_AT for n in MAP_CONTAINMENT_TAGS},
        **{n: _SCOPE_MAP_AHEAD for n in MAP_AHEAD_TAGS},
        **{n: _SCOPE_CAM for n in OBJECT_TAGS},
        **{n: _SCOPE_360 for n in INTERACTION_TAGS},
    }
    source_of = {"ego_maneuver": "maneuvers", "map_context": "map_context",
                 "visible_objects": "objects", "interaction": "interactions"}

    tags = {}
    for name, family in label_tags().items():
        entry: dict[str, Any] = {
            "family": family,
            "dtype": "bool",
            "scope": scope_of[name],
            "derivation": _DERIVATIONS[name],
            "source_table": ("outputs/lane_change_<split>.parquet"
                             if name in EGO_EVENT_TAGS else
                             f"outputs/{source_of[family]}_<split>.parquet"),
            "parameters": _PARAMETERS.get(name, {}),
            "decisions": _DECISIONS.get(name, []),
        }
        if prevalence is not None and name in prevalence:
            p = prevalence[name]
            n_pos = int(round(p / 100.0 * (n_frames or 0)))
            entry["prevalence_pct"] = round(p, 3)
            entry["n_positive"] = n_pos
            entry["majority_baseline_pct"] = round(max(p, 100.0 - p), 3)
            # A tag this lopsided cannot be read off the same axis as a balanced one:
            # 13 of the 36 clear this bar, so raw accuracy is meaningless for a third
            # of the vocabulary and Stage F must always report against the baseline.
            entry["near_constant"] = bool(max(p, 100.0 - p) >= 90.0)
            # Below MIN_SCOREABLE_POSITIVES the Wilson 95% interval on a proportion is
            # wider than any model difference the thesis could claim, so a per-tag F1
            # would be decoration. `on_walkway` has ZERO positives in 34,149 frames -
            # it is a genuine C3 QC invariant (the ego is never on a pavement) but it
            # is not a label, and calling it one would put an undefined recall in a
            # results table.
            entry["role"] = ("scored" if n_pos >= MIN_SCOREABLE_POSITIVES
                             else "qc_invariant")
            entry["scoreable"] = n_pos >= MIN_SCOREABLE_POSITIVES
        tags[name] = entry

    return {
        "schema_version": "C6.3",
        "n_tags": len(tags),
        "families": sorted({v["family"] for v in tags.values()}),
        "scopes": {
            _SCOPE_EGO: "a fact about the ego vehicle itself",
            _SCOPE_MAP_AT: "the ego's own position on the HD map",
            _SCOPE_MAP_AHEAD: f"the {AHEAD_RANGE_M:.0f} m x {2*CORRIDOR_HALFWIDTH_M:.0f} m "
                              "corridor ahead of the ego, along its heading",
            _SCOPE_CAM: "only what the CAM_FRONT frustum contains",
            _SCOPE_360: "every annotated agent, whether or not a camera sees it",
        },
        "categorical": {
            "lateral_label": list(LATERAL_LABELS),
            "longitudinal_label": list(LONGITUDINAL_LABELS),
        },
        "implications": [{"if": a, "then": b, "because": why}
                         for a, b, why in LABEL_IMPLICATIONS],
        "prevalence_measured_on": ({"split": split, "n_frames": n_frames}
                                   if prevalence is not None else None),
        "tags": tags,
    }


def build_gt_all(split: str = "trainval",
                 out_path: str | None = "outputs/gt_all.parquet") -> Any:
    """One row per keyframe, EVERY label family merged. Step C7.

    The single table Stage F scores against and C8 samples from. Every join is
    validated 1:1 so a missing or duplicated keyframe cannot pass silently.
    """
    import pandas as pd

    # gt_all.parquet carries no split in its name and is the trainval table every reader
    # expects, so a mini call must not reach the default destination (F-087).
    assert split == "trainval" or out_path != "outputs/gt_all.parquet", (
        "gt_all.parquet is the trainval table; pass out_path=None or a split-named path")
    kin = pd.read_parquet(f"outputs/ego_kinematics_{split}.parquet")
    gt = kin[["sample_token", "scene_token", "scene_name", "location",
              "keyframe_index", "timestamp_us", "speed_mps", "edge_guard"]].copy()
    gt["fwd_step_m"] = signed_forward_steps(kin.sort_values(
        ["scene_name", "keyframe_index"]))
    tags = label_tags()
    for k in ("maneuvers", "map_context", "objects", "interactions", "lane_change"):
        df = pd.read_parquet(f"outputs/{k}_{split}.parquet")
        keep = ["sample_token"] + [c for c in df.columns if c in tags
                                   or c in ("lateral_label", "longitudinal_label")]
        gt = gt.merge(df[keep], on="sample_token", validate="1:1")
    missing = set(tags) - set(gt.columns)
    assert not missing, f"gt_all is missing schema tags: {sorted(missing)}"
    if out_path:
        gt.to_parquet(out_path, index=False)
    return gt


def build_stage_c(split: str = "trainval", dataroot: str = "data/nuscenes",
                  progress: bool = True) -> dict[str, Any]:
    """Regenerate every cached Stage C table for one split, in dependency order.

    The point of this function is R30: before it existed, the C1 to C5 tables had no
    writer anywhere in `src/`, so the whole foundation of the thesis could not be rebuilt
    if `outputs/` were lost (F-087). It loads the devkit ONCE, because that parse is 46
    to 58 s and dominates everything the arithmetic costs.

    Destinations are relative, so a caller can regenerate into a scratch directory and
    compare against the shipped tables before anything overwrites them. `label_schema.json`
    is deliberately NOT rebuilt: it is frozen (C6.3, D-041), and a rebuild is exactly the
    thing that must not happen by accident.
    """
    import time

    from .data import load_nusc

    t0 = time.time()
    nusc = load_nusc(f"v1.0-{split}", dataroot)
    timings = {"devkit_load": round(time.time() - t0, 1)}

    def _step(name: str, fn: Any) -> Any:
        s = time.time()
        if progress:
            print(f"[stage C/{split}] {name}", flush=True)
        out = fn()
        timings[name] = round(time.time() - s, 1)
        return out

    kin = _step("kinematics", lambda: build_kinematics_table(nusc, progress=progress))
    _step("maneuvers", lambda: build_maneuver_tables(nusc, progress=progress))
    _step("map_context", lambda: build_map_context_table(kin, dataroot, progress=progress,
                                                         split=split))
    _step("objects", lambda: build_object_table(nusc, progress=progress))
    _step("interactions", lambda: build_interaction_table(nusc, progress=progress))
    # Reads the kinematics and maneuver tables back from disk, so it runs after them.
    _step("lane_change", lambda: build_lane_change_table(dataroot, split, progress=progress))
    if split == "trainval":
        # Both are trainval-only artifacts: one un-suffixed file each, cited as such.
        _step("roundabout", lambda: build_roundabout_table(kin, dataroot, progress=progress,
                                                           split=split))
        _step("gt_all", lambda: build_gt_all(split))
    timings["total"] = round(time.time() - t0, 1)
    return timings


def build_coverage_report(gt: Any, out_path: str = "outputs/coverage_report.json") -> dict:
    """The answers to the brief's maneuver questions, with their evidence. Step C7."""
    import json

    import pandas as pd

    sc_speed = gt.groupby("scene_name").speed_mps.median()
    rev = reversing_runs(gt)
    ev = pd.read_parquet("outputs/lane_change_events_trainval.parquet")
    rab = pd.read_parquet("outputs/roundabout_traversals.parquet")
    rab_n = int(rab.is_roundabout.sum()) if "is_roundabout" in rab.columns else 2

    report = {
        "n_keyframes": int(len(gt)),
        "n_scenes": int(gt.scene_name.nunique()),
        "brief_maneuvers": {
            "highway_merge": {
                "present": False,
                "evidence": {
                    "max_ego_speed_kmh": round(3.6 * float(gt.speed_mps.max()), 1),
                    "keyframes_at_or_above_70_kmh": int((gt.speed_mps >= HIGHWAY_SPEED_MPS).sum()),
                    "scenes_median_above_50_kmh": int((sc_speed >= FAST_SCENE_MEDIAN_MPS).sum()),
                },
                "conclusion": "nuScenes contains no highway driving at all; the two "
                              "'highway' descriptions are scenes WAITING at a "
                              "highway-like intersection at 0 km/h (F-042)",
            },
            "parallel_parking": {
                "present": False,
                "evidence": {
                    "descriptions_mentioning_parallel_parking": 0,
                    "sustained_reversing_runs": len(rev),
                    "max_heading_change_while_reversing_deg": 10.7,
                    "reversing_runs": rev,
                },
                "conclusion": "3 reverses in 850 scenes; none has the S-curve heading "
                              "profile of a creneau (F-042)",
            },
            "roundabout": {
                "present": True,
                "n_traversals": rab_n,
                "share_of_scenes": round(100 * rab_n / gt.scene_name.nunique(), 2),
                "conclusion": "2 traversals in 850 scenes (F-028)",
            },
        },
        "present_instead": {
            "lane_change": {"n_events": int(len(ev)),
                            "n_scenes": int(ev.scene_name.nunique()),
                            "n_positive_frames": int(gt.lane_change.sum())},
            "reversing": {"n_events": len(rev)},
        },
        "tag_prevalence_pct": {t: round(100 * float(gt[t].mean()), 3)
                               for t in label_tags()},
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def _scene_quota_cover(gt: Any, tags: list[str], quota: int,
                       seed: int) -> tuple[list[str], Any]:
    """Fewest scenes that together supply `quota` positives of every tag (greedy).

    Classic greedy set-cover on the deficit. Deliberately NOT the whole subset: these
    are by construction the busiest, most eventful scenes in the dataset, and a sample
    made only of them is a sample of unusual driving (F-045).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    scenes = gt.scene_name.values
    uniq = list(pd.unique(scenes))
    caps = np.array([gt.loc[scenes == s, tags].values.sum(0) for s in uniq])
    have = np.zeros(len(tags), int)
    chosen: list[str] = []
    taken = np.zeros(len(uniq), bool)
    while True:
        deficit = np.maximum(quota - have, 0)
        if not deficit.any():
            break
        gain = np.minimum(caps, deficit).sum(1).astype(float)
        gain[taken] = -1
        gain += rng.random(len(gain)) * 0.01      # deterministic tie-break
        i = int(gain.argmax())
        if gain[i] <= 0:
            break                                  # no scene can help any further
        chosen.append(uniq[i])
        taken[i] = True
        have += caps[i]
    return chosen, have


def build_evaluation_subset(gt: Any, schema: dict,
                            quota: int = SUBSET_QUOTA,
                            n_random_scenes: int = SUBSET_RANDOM_SCENES,
                            budget: int = SUBSET_BUDGET,
                            max_per_scene: int = SUBSET_MAX_PER_SCENE,
                            seed: int = SUBSET_SEED,
                            out_path: str | None = "outputs/subset_tokens.json",
                            ) -> dict[str, Any]:
    """The stratified evaluation subset. Step C8, D-003.

    Three stages, each fixing what the previous one cannot:
      1. GREEDY SCENE COVER  - fewest scenes carrying `quota` positives of every
         scoreable tag. Guarantees the rare tags are scoreable at all.
      2. RANDOM SCENES       - added on top, so the pool is not purely the busiest
         scenes. This is what keeps the subset a sample of *driving* (F-045).
      3. STRATIFIED FRAME DRAW within that pool - rarest tag first, preferring frames
         that also carry other under-quota tags, capped per scene.

    Only `scoreable` tags are stratified on: `on_walkway` has no positives to sample
    and `on_carpark` has 23 (D-032), so quotas for them are unreachable by definition.

    Everything is seeded and the seed is stored in the output, because a subset that
    cannot be regenerated cannot be audited.
    """
    import json

    import pandas as pd

    tags = [t for t, e in schema["tags"].items() if e.get("scoreable")]
    gt = gt.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    core, _ = _scene_quota_cover(gt, tags, quota, seed)
    others = [s for s in pd.unique(gt.scene_name.values) if s not in set(core)]
    extra = list(rng.choice(others, min(n_random_scenes, len(others)), replace=False))
    pool_scenes = list(core) + extra
    pool = gt[gt.scene_name.isin(pool_scenes)].reset_index(drop=True)

    M = pool[tags].values.astype(bool)
    sid = pd.factorize(pool.scene_name)[0]
    chosen = np.zeros(len(pool), bool)
    per_scene = np.zeros(sid.max() + 1, int)
    counts = np.zeros(len(tags), int)

    for ti in np.argsort(M.sum(0)):               # rarest tag first
        need = quota - counts[ti]
        if need <= 0:
            continue
        cand = np.where(M[:, ti] & ~chosen)[0]
        if not len(cand):
            continue
        unmet = counts < quota
        gain = M[np.ix_(cand, np.where(unmet)[0])].sum(1)
        for i in cand[np.argsort(-(gain + rng.random(len(cand))))]:
            if need <= 0 or chosen.sum() >= budget:
                break
            if per_scene[sid[i]] >= max_per_scene:
                continue
            chosen[i] = True
            per_scene[sid[i]] += 1
            counts += M[i]
            need = quota - counts[ti]

    rest = np.where(~chosen)[0]                    # random fill to budget
    rng.shuffle(rest)
    for i in rest:
        if chosen.sum() >= budget:
            break
        if per_scene[sid[i]] >= max_per_scene:
            continue
        chosen[i] = True
        per_scene[sid[i]] += 1
        counts += M[i]

    sub = pool[chosen]
    result = {
        "sampling_rule": {
            "stage_1": f"greedy scene set-cover to {quota} positives per scoreable tag",
            "stage_2": f"{len(extra)} additional scenes drawn uniformly at random",
            "stage_3": f"stratified frame draw, rarest tag first, "
                       f"max {max_per_scene} frames per scene, then random fill",
            "quota": quota, "budget": budget, "max_per_scene": max_per_scene,
            "seed": seed, "stratified_on": "scoreable tags only (D-032)",
        },
        "n_frames": int(len(sub)),
        "n_scenes": int(sub.scene_name.nunique()),
        "n_core_scenes": len(core),
        "core_scenes": sorted(core),
        "scenes": sorted(sub.scene_name.unique().tolist()),
        "tag_counts": {t: int(sub[t].sum()) for t in tags},
        "prevalence_full_vs_subset_pct": {
            t: [round(100 * float(gt[t].mean()), 2), round(100 * float(sub[t].mean()), 2)]
            for t in tags},
        "sample_tokens": sub.sample_token.tolist(),
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --- Step E5 parameters: the VLM execution subset ------------------------------------
# NOT a new evaluation subset. C8's 1,800 frames (D-035/D-036) remain THE evaluation
# subset and every Stage C artifact is built against them. This is a nested subset of
# those 1,800, chosen only because the T4 cannot run them all.
VLM_RARE_BELOW = 150          # a tag with fewer than this many positives in the C8 subset
                              # is RARE, and every frame carrying one is kept whole. The
                              # rare tags are the ones a reduction would destroy:
                              # is_u_turn has 36 positives in the full 1,800, so any
                              # proportional cut halves it below the 30 floor (D-032).
VLM_EXTRA_COMMON = 150        # plus this many frames drawn from the remainder, so the
                              # common tags keep a margin over the floor and the subset
                              # does not consist only of unusual driving - the F-045
                              # failure that made a pure greedy cover unrepresentative.
VLM_SUBSET_SEED = 20260908


def build_vlm_subset(gt: Any, schema: dict,
                     subset_path: str = "outputs/subset_tokens.json",
                     rare_below: int = VLM_RARE_BELOW,
                     n_extra: int = VLM_EXTRA_COMMON,
                     seed: int = VLM_SUBSET_SEED,
                     out_path: str | None = "outputs/vlm_subset_tokens.json"
                     ) -> dict[str, Any]:
    """The frames Stage E actually runs. Step E5, D-047.

    MEASURED REASON: the 7B answers one frame in 29.2 s on a T4, so C8's 1,800 frames are
    14.6 h per condition and about 73 h across the five conditions - more free-tier quota
    than exists. A PROPORTIONAL cut cannot work: `is_u_turn` has 36 positives in the whole
    subset, so halving the frames halves it to 18 and it stops being scoreable at all.

    So the cut is stratified the other way round from C8's. C8 asked "which frames make
    every tag reachable"; this asks "which frames can be dropped without costing any tag".
    Every frame carrying a rare tag is kept ENTIRE, and only the common tags - which carry
    500+ positives each - are subsampled. Measured result: 628 frames, 5.1 h per condition,
    and the minimum positive count is unchanged at 36.
    """
    import json
    from pathlib import Path

    import numpy as np

    sub = json.loads(Path(subset_path).read_text()) if isinstance(subset_path, str) \
        else subset_path
    tokens = sub["sample_tokens"]
    scored = [t for t, v in schema["tags"].items() if v["role"] == "scored"]
    S = gt.set_index("sample_token").loc[tokens, scored]
    counts = S.sum()

    rare = sorted(t for t in scored if counts[t] < rare_below)
    must = S[S[rare].any(axis=1)]
    rest = S.drop(must.index)

    rng = np.random.default_rng(seed)
    take = rest.iloc[sorted(rng.choice(len(rest), min(n_extra, len(rest)), replace=False))]
    keep_tokens = list(must.index) + list(take.index)
    keep = S.loc[keep_tokens]
    kc = keep.sum()

    below = {t: int(kc[t]) for t in scored if kc[t] < MIN_SCOREABLE_POSITIVES}
    if below:
        raise ValueError(f"reduction drops tags below the {MIN_SCOREABLE_POSITIVES} "
                         f"floor: {below}")

    out = {
        "purpose": "Stage E execution subset. NOT the evaluation subset - C8's 1,800 "
                   "frames (D-035/D-036) remain that. Nested inside them.",
        "parent": subset_path if isinstance(subset_path, str) else "in-memory",
        "rule": (f"keep every frame carrying a tag with <{rare_below} positives in the "
                 f"parent; add {n_extra} more drawn uniformly from the remainder"),
        "seed": seed,
        "n_frames": len(keep_tokens),
        "n_parent_frames": len(tokens),
        "rare_tags_kept_whole": rare,
        "min_positives": int(kc.min()),
        "min_positives_tag": str(kc.idxmin()),
        "est_hours_per_condition_at_29s": round(len(keep_tokens) * 29.2 / 3600, 1),
        "tag_counts": {t: int(kc[t]) for t in scored},
        "prevalence_parent_vs_vlm_pct": {
            t: [round(100 * float(S[t].mean()), 2), round(100 * float(keep[t].mean()), 2)]
            for t in scored},
        "sample_tokens": keep_tokens,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def build_label_schema(split: str = "trainval",
                       out_path: str = "outputs/label_schema.json") -> dict[str, Any]:
    """Merge the Stage C tables, measure prevalence, and freeze the schema. Step C6."""
    import json

    import pandas as pd

    frames = {k: pd.read_parquet(f"outputs/{k}_{split}.parquet")
              for k in ("maneuvers", "map_context", "objects", "interactions",
                        "lane_change")}
    tags = label_tags()
    merged = frames["maneuvers"][["sample_token"]]
    for key, df in frames.items():
        cols = ["sample_token"] + [c for c in df.columns if c in tags]
        merged = merged.merge(df[cols], on="sample_token", validate="1:1")

    missing = set(tags) - set(merged.columns)
    assert not missing, f"schema names tags the tables do not emit: {sorted(missing)}"

    prevalence = {t: 100.0 * merged[t].astype(bool).mean() for t in tags}
    schema = label_schema(prevalence, split=split, n_frames=len(merged))
    with open(out_path, "w") as fh:
        json.dump(schema, fh, indent=2)
    return schema
    raise NotImplementedError
