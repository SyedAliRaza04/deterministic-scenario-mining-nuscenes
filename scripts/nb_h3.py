"""Source for notebooks/stage_h3_kaggle.ipynb — Step H3, BDD100K signage.

Build:  ./venv/bin/python scripts/make_notebook.py scripts/nb_h3.py
"""

OUT = "notebooks/stage_h3_kaggle.ipynb"

CELLS = [
("md", r"""
# Step H3 — signage on BDD100K

Answers **RQ4**. nuScenes has no traffic-sign or traffic-light annotation and its map's
`items[].color` is the fixture's lamp inventory rather than what is lit — 626 of 634 records
carry RED and YELLOW and GREEN at once (F-004/F-076). H2 established that **presence** is
verifiable there and **state** is not. BDD100K has `trafficLightColor`, so state is the one
signage question this run exists to measure.

**586 images, ~1.6 h. Much smaller than a Stage E condition** — five tags instead of 35, one
image per call, and a 32 MB bundle instead of 542 MB.

**Setup, once:**
1. Settings → Accelerator → **GPU T4 x2** (Turing 7.5; P100 is Pascal 6.0 and `bitsandbytes`
   4-bit will not run on it).
2. Settings → Internet → **On** (the model downloads from HuggingFace).
3. Add Data → Upload `outputs/h3_bundle.tar.gz` as a private Dataset named `thesis-h3`.
   Build it first with `./venv/bin/python -c "import src.bdd as B; B.build_bundle()"`.

**Three things that are deliberate and will look wrong if you skip this paragraph.**
State is asked as **three booleans**, never as one colour: 34.0% of images with a lit light
show more than one *distinct* lit colour and BDD100K has no lane association, so
"what colour is the light?" has no well-posed answer and the best proxy (largest box) agrees
with the image's majority colour only 87.55% of the time (R22). The reference itself is
uncertain on **29.94%** of images — a light whose colour is `NA` — so state metrics are
reported twice, over all images and over the certain ones only. And the scoring runs **in
this session**: the ground truth travels inside the bundle, so the numbers do not wait on a
download (F-062 became a "first look" that had to be re-scored later exactly because they did).
"""),

("code", r"""
import os, sys, json, glob, pathlib, shutil, tarfile

# Kaggle AUTO-EXTRACTS uploaded archives and the folder it extracts into is not
# predictable. Find the bundle ROOT by its marker file rather than by name - a hardcoded
# path has broken twice already on the Stage E notebook.
_marker = glob.glob('/kaggle/input/**/outputs/h3_bdd_subset.json', recursive=True)
if _marker:
    WORK = pathlib.Path(_marker[0]).parent.parent
    print('bundle found at:', WORK)
else:
    _tars = glob.glob('/kaggle/input/**/*.tar.gz', recursive=True)
    if not _tars:
        print('NOTHING MATCHED. Actual layout under /kaggle/input:')
        for p in sorted(glob.glob('/kaggle/input/*/*'))[:40]:
            print('   ', p)
        raise SystemExit('upload outputs/h3_bundle.tar.gz as a Kaggle Dataset first')
    WORK = pathlib.Path('/kaggle/working/bundle')
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_tars[0]) as t:
        t.extractall(WORK)
    print('extracted', _tars[0], '->', WORK)

# /kaggle/input is READ-ONLY. Results and any resumed rows live in /kaggle/working.
OUT = pathlib.Path('/kaggle/working/results'); OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(WORK/'results/*.jsonl')):
    shutil.copy(f, OUT)
print('images in bundle:', len(glob.glob(str(WORK/'images/*.jpg'))))
"""),

("code", r"""
!pip -q install -U "transformers>=4.49" accelerate bitsandbytes
import torch

# Same hard stop as Stage E: bitsandbytes 4-bit NF4 needs compute capability >= 7.5.
# Checked here rather than 20 minutes into a model download.
_cap = torch.cuda.get_device_capability(0)
_name = torch.cuda.get_device_name(0)
assert _cap >= (7, 5), (
    f'{_name} has compute capability {_cap[0]}.{_cap[1]}; bitsandbytes 4-bit needs 7.5+. '
    'Switch Settings -> Accelerator to GPU T4 x2 (Turing, 7.5). P100 is Pascal 6.0.')
print(f'{_name}, capability {_cap[0]}.{_cap[1]} - ok')
"""),

("code", r"""
# The same model as every nuScenes arm. H4's whole point is a CROSS-DATASET comparison, so
# the model is the thing that must NOT vary: a different checkpoint here would confound the
# dataset difference with a model difference and there would be nothing to compare.
MODEL = None          # None = the verified Qwen2.5-VL-7B, as used for E5-E9
LIMIT = None          # e.g. 12 for a smoke run
"""),

("code", r"""
sys.path.insert(0, str(WORK))
import src.bdd as B, src.vlm as V

spec = json.loads((WORK/'outputs/h3_bdd_subset.json').read_text())
records = {B.image_id(r['filepath']): r for r in spec['records']}
tokens = list(records)
print(f"{len(tokens)} images | {spec['rule']}")
print('prevalence in the FULL 10k val split (%):', spec['prevalence_full_split_pct'])
print('positives in this subset            :', spec['positives_all'])
print(f"ground-truth state uncertain on {spec['state_gt_uncertain_pct']}% of images")

# F-065 guard, on the new prompt. preflight checks it on the laptop; it cannot see what
# Kaggle actually loaded, and a stale bundle would reinstate a prompt that asks for
# reasoning and forbids it in the same breath.
PROMPT = B.build_prompt()
_asks = 'think step by step' in PROMPT or 'reasoning first' in PROMPT
assert not (_asks and 'nothing else' in PROMPT), 'stale src/bdd.py in the bundle'
for t in B.BDD_TAGS:
    assert f'"{t}"' in PROMPT, f'{t} is never asked for'
print(f'prompt ok ({len(PROMPT)} chars, {len(B.BDD_TAGS)} tags)')

# HARD FAIL on a missing image, never a skip. A hole in the image set becomes a shorter
# results file that nothing flags (F-048), and here it would silently drop a whole tag's
# positives - the top-up images are ALL yellow positives, so losing them costs the only
# tag that needed protecting.
missing = [t for t in tokens if not (WORK/f'images/{t}.jpg').exists()]
assert not missing, f'{len(missing)} images absent from the bundle, e.g. {missing[:3]}'

def images_for(tok):
    return [WORK/f'images/{tok}.jpg']
if LIMIT: tokens = tokens[:LIMIT]
"""),

("code", r"""
be = V.TransformersBackend(**({'model': MODEL} if MODEL else {}))
print('loaded:', be.model_name)

# SMOKE FIRST. Two images, both must parse. On a 1.6 h job this costs a minute and it is
# the only thing that catches a degenerate configuration before the session is committed -
# E9's all-'!' failure was invisible in the progress line and at a normal latency (F-069).
_smoke = V.run_batch(be, tokens[:2], 'h3_bdd_signage', OUT/'_smoke.jsonl',
                     images_for=images_for, prompt_text=PROMPT,
                     tags=list(B.BDD_TAGS), max_new_tokens=200, progress=False)
_rows = [json.loads(l) for l in (OUT/'_smoke.jsonl').open()]
assert all(r['parse_reason'].startswith('ok') for r in _rows), \
    f"smoke failed: {[(r['parse_reason'], r['raw'][:80]) for r in _rows]}"
print('smoke ok:', [(r['parse_reason'], r['n_answered'], r['values']) for r in _rows])
"""),

("code", r"""
RESULT = OUT/V.result_filename('h3_bdd_signage', MODEL)

# max_new_tokens=200, not 700. Five booleans is ~90 characters of JSON; the 700 the tag
# arms use exists for chain-of-thought's ~2,000-character replies and would only buy
# decode time here. The one parse failure in the whole project was CoT truncation against
# 700 (F-066e), so the number is not arbitrary in either direction.
stats = V.run_batch(be, tokens, 'h3_bdd_signage', RESULT,
                    images_for=images_for, prompt_text=PROMPT,
                    tags=list(B.BDD_TAGS), max_new_tokens=200, progress=True)
print(stats)
"""),

("md", r"""
## Score it here, not at home

The reference travels inside the bundle, so H3's numbers exist before the file is
downloaded. Three views, and they are not interchangeable:

* **`uniform_view`** — the 450 drawn at random. Prevalences are the real ones, so these
  baselines are the population baselines. **Read presence results here.**
* **`all`** — those 450 plus the yellow top-up. The only view where
  `yellow_light_visible` clears the 30-positive floor, and the only one where its F1 means
  anything.
* **`state_gt_certain_only`** — images whose reference state is not flagged uncertain.
  29.94% of the subset carries a light whose colour is `NA`, and a single number over that
  would average a known contamination into the result.
"""),

("code", r"""
import pprint
res = B.score(results_stem=str(RESULT))
print(f"images scored      : {res['n_images']}")
print(f"parse failure rate : {res['parse_failure_rate']:.4f}   <- RQ2b on a second dataset")
print(f"state-certain only : {res['n_state_gt_certain']} images\n")
for view in ('uniform_view', 'all', 'state_gt_certain_only'):
    print(f'--- {view} ---')
    for tag, m in res[view].items():
        beat = 'BEATS' if m['f1'] > m['baseline_f1'] else '     '
        print(f"  {tag:24s} P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}  base {m['baseline_f1']:.3f}  {beat}  n={m['support']}")
    print()
json.dump(res, open('/kaggle/working/h3_scores.json','w'), indent=1)
print('results  :', RESULT)
print('scores   : /kaggle/working/h3_scores.json')
print('\nTake BOTH home: results/ into outputs/results/, scores beside it. H4 reads them.')
"""),
]
