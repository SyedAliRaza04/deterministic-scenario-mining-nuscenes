# Notebooks

Four Kaggle notebooks. They are deliberately thin: each one extracts a bundle, imports `src/`,
and runs. All of the logic they use lives in the repository, so nothing here can drift from
what the tests cover.

| Notebook | What it runs | Bundle | Time |
|---|---|---|---|
| `stage_e_kaggle.ipynb` | every inference condition, the repaired bird's eye view arms, and the NuScenes-QA validation | `colab_bundle.tar.gz`, 686 MB | ~5 h per condition |
| `stage_g2_kaggle.ipynb` | storyboard segmentation, 951 keyframe pairs over 25 scenes | `g2_bundle.tar.gz`, 146 MB | ~1.8 h |
| `stage_h3_kaggle.ipynb` | BDD100K signage, 586 images, 5 tags | `h3_bundle.tar.gz`, 32 MB | ~1.6 h |
| `qwen3_bev_kaggle.ipynb` | the Qwen3-VL-8B bird's eye view slice, 150 frames | same bundle as Stage E | ~1.5 h |

The signage and segmentation runs have **their own bundles** rather than flags on the main
one. That bundle is 542 MB of nuScenes, and re-uploading it to run a 32 MB job would spend an
hour of bandwidth for nothing. Both carry their own reference labels, so **each notebook
scores its own run inside the session**: a first look that waits on a download has to be
re-scored later, and was.

## Before you start any of them

```bash
./venv/bin/python scripts/preflight.py h3        # or  g2  |  f5  |  <a condition name>
```

It must print **ALL CHECKS PASSED**. It checks the artifact, never the reference to it: that
the bundle really carries every image, that its `src/` hashes match the working tree, that the
prompt does not ask for reasoning and forbid it in the same breath, and that no completed
results file is sitting in the bundle waiting to turn the whole session into a no-op. It has
caught a stale bundle.

**Rehearse from the extracted bundle, never from the repository.** A bundle is a subset of the
repository and a lazy import fails only when its function runs. A rehearsal with the
repository on `sys.path` proved nothing about a bundle that shipped a module without the one
it imports inside a scoring function: 951 GPU calls completed and the last cell died on an
import error. Extract the bundle to a temporary directory and run the notebook's calls in a
fresh interpreter that cannot see the repository. A smoke test must also make the **exact**
call the run will make; one that built its own call passed while the real one raised.

## Building the bundles

```bash
./venv/bin/python -c "import src.bdd as B; B.build_bundle()"                                  # signage
./venv/bin/python -c "import src.storyboard as S; S.vlm_pair_items(); S.build_g2_bundle()"    # segmentation
./venv/bin/python -c "import src.data as D; D.build_colab_bundle(temporal=True)"              # inference
```

## Editing a notebook

Do not edit the `.ipynb`. Each is generated:

```bash
./venv/bin/python scripts/make_notebook.py scripts/nb_h3.py     # or scripts/nb_g2.py
```

A notebook is JSON with its source split into escaped one line strings, and a mistake there is
invisible until the session opens the file. The `.py` specification is the source of truth.

## Host behaviour, each item learned from a failed run

| Host | Behaviour |
|---|---|
| Kaggle | Uploaded archives are **auto extracted**, so the dataset may contain a directory rather than the `.tar.gz`. The data cell finds the bundle root by its marker file and handles both. |
| Kaggle | `/kaggle/input` is **read only**. Completed rows are copied into `/kaggle/working/results/` before the runner appends. |
| Kaggle | The writable path is `/kaggle/working`. `/kaggle/work` does not exist. |
| Kaggle | GPU and internet are both gated behind phone verification. Without it the accelerator will not turn on. |
| Kaggle | Choose **T4 x2**, never P100. 4-bit NF4 needs compute capability 7.5 or above; T4 is Turing at 7.5, P100 is Pascal at 6.0. Without 4-bit the 7B needs about 15 GB in fp16 against the P100's 16 GB, leaving nothing for the cache. The notebooks assert the capability before downloading the model. |
| Colab | Overwriting a `.py` does **not** reload an already imported module. Re-extract the bundle, then restart the session. The import cell asserts this rather than failing later with a traceback whose line numbers come from the new file while the code that runs is the old one. |
| Colab | Free GPU quota is opaque and can run out mid condition. Progress is written per row, so this costs only the in-flight frame. |
| Both | The execution subset is **asserted**, never defaulted. A warn-and-continue fallback once ran the full 1,800 frames for ten hours longer than intended. |
| Both | The temporal arm only: the bundle must be built with `temporal=True`. It then carries the sequence index and the 1,008 extra neighbour frames, 542 MB and 2,808 images. Without it the prompt promises five frames and the model receives one. The notebook asserts the file is present rather than falling back. |
| Both | The temporal arm only: images are capped at 640x360. At the default 1280x720 the 4-bit path emits degenerate output from four images up, silently and at normal latency, so the whole run would produce nothing. The cap is applied **by condition**, never by image count, because the two-image arm sends full size images and is already complete. |
| Both | A two frame smoke test runs before the real batch and asserts both replies parse. The runner also aborts after five consecutive **parse** failures. Backend errors are excluded deliberately: a dead backend raises instantly, and a restart needs those rows to exist so it knows what to retry. |

## Reading the segmentation result

**Read it against the null, never as an absolute F1.** Boundaries occur 0.098 times per
keyframe, so random placement with the same boundary count scores 0.294 at a one keyframe
tolerance and 0.490 at two. Those numbers were computed **before** the run, deliberately. The
notebook compares against the null's **97.5th percentile**, not its mean: the model gets one
draw, and a single random draw has a standard deviation of about 0.03. In rehearsal a fake
backend answering `true` at random cleared the null's mean by 0.051 and would have read as a
real effect.

## Between sessions

Download `results/<condition>.jsonl`, put it in `outputs/results/`, then rebuild the bundle
with `src.data.build_colab_bundle()`. Completed rows travel inside the bundle, so the next run
on any host resumes instead of repeating work.
