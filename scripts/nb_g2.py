"""Source for notebooks/stage_g2_kaggle.ipynb — Step G2b, the VLM segmentation arm.

Build:  ./venv/bin/python scripts/make_notebook.py scripts/nb_g2.py
"""

OUT = "notebooks/stage_g2_kaggle.ipynb"

CELLS = [
("md", r"""
# Step G2b — can a VLM find the manoeuvre boundaries?

Answers the second half of **RQ5**. PLAN G2's criterion is "VLM segmentation vs
fixed-interval baseline"; the baselines, the tolerance sweep and the chance floor are
already built and published (`outputs/g2_segmentation.csv`, F-084). This session supplies
the missing column.

**951 calls over 25 scenes, ~3.2 h at 12 s each.** Two images per call at full resolution —
the `v4_both` configuration, already verified over 628 frames — and a one-boolean answer, so
`max_new_tokens` is 32 instead of 700 and most of the per-call cost disappears with it.

**Setup, once:**
1. Settings → Accelerator → **GPU T4 x2** (Turing 7.5; P100 is Pascal 6.0 and 4-bit will
   not run).
2. Settings → Internet → **On**.
3. Add Data → Upload `outputs/g2_bundle.tar.gz` (146 MB) as a private Dataset named
   `thesis-g2`. Build it with
   `./venv/bin/python -c "import src.storyboard as S; S.vlm_pair_items(); S.build_g2_bundle()"`.

**Why the question is pairwise rather than "segment this scene".** The obvious design —
show the model K frames of a 20 s scene and ask which start a new manoeuvre — is not
answerable at the resolution the metric scores. E9 measured the ceiling at **five images**
(at 1280×720 the 4-bit path emits degenerate all-`!` output from four up, F-069), and five
frames spread over 40 keyframes places a boundary no better than ±4. Scored at ±1 that fails
for reasons that have nothing to do with the model. So the unit of work is a **keyframe**:
show the frame one keyframe before and one after, and ask whether the car changed what it
was doing. A `true` marks a boundary at that keyframe, in the same coordinate as the
reference — no snapping and no off-by-one to argue about afterwards.

**The null is already fixed, and it is large.** Boundaries are 0.098 per keyframe in this
draw, so random placement with the same boundary count scores **0.294 at ±1 keyframe** and
**0.490 at ±2**. Those numbers were computed before this run, deliberately (F-084/R45). An
always-yes answer scores 0.187 and the matched null scores 0.187 too — enthusiasm buys
exactly nothing here.
"""),

("code", r"""
import os, sys, json, glob, pathlib, shutil, tarfile

_marker = glob.glob('/kaggle/input/**/outputs/g2_vlm_items.json', recursive=True)
if _marker:
    WORK = pathlib.Path(_marker[0]).parent.parent
    print('bundle found at:', WORK)
else:
    _tars = glob.glob('/kaggle/input/**/*.tar.gz', recursive=True)
    if not _tars:
        print('NOTHING MATCHED. Actual layout under /kaggle/input:')
        for p in sorted(glob.glob('/kaggle/input/*/*'))[:40]:
            print('   ', p)
        raise SystemExit('upload outputs/g2_bundle.tar.gz as a Kaggle Dataset first')
    WORK = pathlib.Path('/kaggle/working/bundle'); WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_tars[0]) as t:
        t.extractall(WORK)
    print('extracted', _tars[0], '->', WORK)

OUT = pathlib.Path('/kaggle/working/results'); OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(WORK/'results/*.jsonl')):
    shutil.copy(f, OUT)
print('images in bundle:', len(glob.glob(str(WORK/'images/*.jpg'))))
"""),

("code", r"""
!pip -q install -U "transformers>=4.49" accelerate bitsandbytes
import torch

_cap = torch.cuda.get_device_capability(0)
_name = torch.cuda.get_device_name(0)
assert _cap >= (7, 5), (
    f'{_name} has compute capability {_cap[0]}.{_cap[1]}; bitsandbytes 4-bit needs 7.5+. '
    'Switch Settings -> Accelerator to GPU T4 x2 (Turing, 7.5). P100 is Pascal 6.0.')
print(f'{_name}, capability {_cap[0]}.{_cap[1]} - ok')
"""),

("code", r"""
# Same model as every other arm. G2b is compared against the fixed-interval and random
# baselines, not against another model, so the checkpoint is held constant for the same
# reason E10 varied it deliberately and alone (D-048).
MODEL = None          # None = the verified Qwen2.5-VL-7B
LIMIT = None          # e.g. 20 for a smoke run
"""),

("code", r"""
sys.path.insert(0, str(WORK))
import src.storyboard as S, src.vlm as V

spec = json.loads((WORK/'outputs/g2_vlm_items.json').read_text())
items = {it['item_id']: it for it in spec['items']}
tokens = list(items)
print(f"{spec['n_items']} items over {spec['n_scenes']} scenes, "
      f"{spec['n_gold_boundaries']} gold boundaries")
print(f"estimate: {spec['est_hours_at_12s']} h at 12 s/call, "
      f"{spec['est_hours_at_20s']} h at 20 s/call")
print('CHANCE FLOOR, fixed before the run:', spec['chance_f1_closed_form'])
print(f"{spec['n_scenes_with_no_boundary']} of {spec['n_scenes']} scenes contain NO "
      "boundary - those can only produce false positives, and that is the point")

PROMPT = S.VLM_PAIR_PROMPT
_asks = 'think step by step' in PROMPT or 'reasoning first' in PROMPT
assert not (_asks and 'nothing else' in PROMPT), 'stale src/storyboard.py in the bundle'
assert f'"{S.G2_VLM_TAG}"' in PROMPT, 'the prompt does not ask for the tag the scorer reads'
print(f'prompt ok ({len(PROMPT)} chars), tag = {S.G2_VLM_TAG!r}')

# HARD FAIL on a missing frame. A hole here does not shorten the run visibly - it removes
# one keyframe from the predicted boundary set, which lowers recall for a reason that is
# ours (F-048).
_missing = [t for it in spec['items'] for t in (it['before_token'], it['after_token'])
            if not (WORK/f'images/{t}.jpg').exists()]
assert not _missing, f'{len(_missing)} frames absent from the bundle, e.g. {_missing[:3]}'

def images_for(item_id):
    it = items[item_id]
    # ORDER IS THE EXPERIMENT. The prompt says the first image is the earlier moment; a
    # swapped pair asks a different question and nothing downstream could detect it.
    return [WORK/f"images/{it['before_token']}.jpg", WORK/f"images/{it['after_token']}.jpg"]
if LIMIT: tokens = tokens[:LIMIT]
"""),

("code", r"""
# Two images at FULL resolution, not E9's 640x360 cap. The cap was measured at four images
# and up (F-069); v4_both sends two at full size over 628 frames with no degeneracy, and
# applying a cap that was never needed would hand the arm a resolution confound (L-024) it
# does not have to carry.
be = V.TransformersBackend(**({'model': MODEL} if MODEL else {}))
print('loaded:', be.model_name)

_smoke = V.run_batch(be, tokens[:2], 'g2_vlm_pairs', OUT/'_smoke.jsonl',
                     images_for=images_for, prompt_text=PROMPT,
                     tags=[S.G2_VLM_TAG], max_new_tokens=32, progress=False)
_rows = [json.loads(l) for l in (OUT/'_smoke.jsonl').open()]
assert all(r['parse_reason'].startswith('ok') for r in _rows), \
    f"smoke failed: {[(r['parse_reason'], r['raw'][:80]) for r in _rows]}"
assert all(r['n_answered'] == 1 for r in _rows), \
    f"the model answered something other than the one tag: {[r['values'] for r in _rows]}"
print('smoke ok:', [(r['parse_reason'], r['values'], round(r['latency_s'], 1)) for r in _rows])
print(f"projected: {len(tokens) * _rows[-1]['latency_s'] / 3600:.1f} h at this latency")
"""),

("code", r"""
RESULT = OUT/V.result_filename('g2_vlm_pairs', MODEL)
stats = V.run_batch(be, tokens, 'g2_vlm_pairs', RESULT,
                    images_for=images_for, prompt_text=PROMPT,
                    tags=[S.G2_VLM_TAG], max_new_tokens=32, progress=True)
print(stats)
"""),

("md", r"""
## Score it here

The reference travels in the bundle, so the number exists before the file is downloaded.

**Read the gap to the floor, never the absolute F1.** `random_matched_gold` is random
placement with the reference's boundary count; `random_matched_vlm` is random placement
with *the model's own* count, which is the one that matters if the model answers `true`
generously — a segmenter that calls everything a boundary must be compared with a random
segmenter that does the same, or its recall is credit for enthusiasm rather than timing.
"""),

("code", r"""
df = S.score_vlm_boundaries(results_stem=str(RESULT),
                            items_path=str(WORK/'outputs/g2_vlm_items.json'),
                            out_csv='/kaggle/working/g2_vlm_scores.csv')
import pandas as pd
pd.set_option('display.width', 200)
print(df.to_string(index=False))

vlm = df[(df.method == 'vlm_pairwise')].set_index('tolerance_kf')
rnd = df[(df.method == 'random_matched_vlm')].set_index('tolerance_kf')
# AGAINST THE NULL'S 97.5th PERCENTILE, not its mean. The model gets ONE draw, and a
# single random draw has a standard deviation of ~0.03 here - a fake model answering at
# random cleared the null's MEAN by +0.051 in rehearsal and was 1.1 sd from it.
print('\nAGAINST THE MATCHED NULL (this is the result):')
for tol in sorted(vlm.index):
    print(f"  tol {tol}: VLM {vlm.f1[tol]:.3f}   null {rnd.f1[tol]:.3f} "
          f"[{rnd.f1_lo[tol]:.3f}, {rnd.f1_hi[tol]:.3f}]   "
          f"{'ABOVE CHANCE' if vlm.f1[tol] > rnd.f1_hi[tol] else 'inside the null band'}")
print(f"\nanswer coverage {df.answer_coverage.iloc[0]:.3f}, "
      f"parse failures {df.parse_failure_rate.iloc[0]:.4f}")
print('\nTake home: results/ into outputs/results/, and g2_vlm_scores.csv beside it.')
"""),
]
