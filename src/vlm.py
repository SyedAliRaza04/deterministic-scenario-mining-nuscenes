"""VLM backends and output parsing — Stage E.

Two backends, because we have two machines and no paid GPU:
    MLXBackend           Mac M4, 16 GB unified. Use the 3B model ONLY.
                         7B-4bit + a 1600x900 image + KV cache + macOS will swap.
                         For prompt iteration, not for the real runs.
    TransformersBackend  Colab free T4, 16 GB VRAM. Qwen2.5-VL-7B in 4-bit.
                         This produces the numbers that go in the thesis.

Stage A3 skeleton — signatures only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

MLX_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"     # Mac, iteration only
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"                     # Colab T4, 4-bit

# nuScenes images are 1600x900. Downscale before inference on both backends.
MAX_IMAGE_PX = (1280, 720)

# Step E9. At MAX_IMAGE_PX the 4-bit path emits degenerate all-'!' output from FOUR images
# up, silently and at a normal latency; 640x360 restores clean JSON at five. Measured on
# the 3B/MLX path, 2026-09-15:
#     images   1      2      3      4        5
#     1280x720 JSON   JSON   JSON   '!'x40   '!'x40
#     640x360  -      -      JSON   JSON     JSON  (parse ok, 34/35 answered, 31 s)
# Not memory: full-resolution 5-image peak was 4.25 GB of 16 GB. Not the chat template
# either -- it emits exactly N image placeholders for every N from 1 to 6. The binding
# quantity is the vision-token count, which crosses a threshold between ~880 and ~1,180.
#
# Applied ONLY to the temporal condition, never by image count: `v4_both` sends two images
# at MAX_IMAGE_PX and has already completed 628/628 (F-064). Downscaling on a count rule
# would silently change a finished condition and break its comparability.
#
# The cost is a genuine confound RQ3b must state: the temporal arm sees frames at a
# quarter of the pixels the single-frame arms saw, so a temporal gain is measured against
# a resolution handicap and a temporal loss could be caused by one. This is L-005's
# territory -- a property of what an accessible open-weight model can be fed, not of
# whether sequence helps.
MULTI_IMAGE_MAX_PX = (640, 360)


def result_filename(condition: str, model: str | None = None) -> str:
    """The results file a run writes. ONE definition, because two would drift (R4).

    E10 runs an ALTERNATIVE model on the same condition as the primary arm, so its output
    must not overwrite `v1_structured.jsonl` -- hence the model suffix. The notebook and
    `scripts/preflight.py` both need this name, and when preflight computed it separately
    it checked the wrong file and reported E10 as blocked by rows that belong to a
    different model. A check that cries wolf is how F-060's real warning came to be
    ignored.
    """
    tag = "" if not model else "__" + model.split("/")[-1]
    return f"{condition}{tag}.jsonl"


class VLMBackend(Protocol):
    """Any backend takes N images + a prompt and returns raw text."""

    def generate(self, images: list[Path], prompt: str, max_new_tokens: int = 512) -> str:
        ...


class MLXBackend:
    """Apple Silicon backend via mlx-vlm. Step E1.

    LIVES IN ITS OWN VIRTUALENV (`./venv-mlx`), not the project one. mlx-vlm requires
    numpy>=2 and the nuscenes-devkit requires numpy<2, so installing it alongside the
    pipeline would break every Stage C module. `src/vlm.py` and `src/prompts.py` import
    no numpy, so both environments can use them (D-046).

    3B, not 7B: on 16 GB unified memory a 7B-4bit plus a 1600x900 image plus KV cache plus
    macOS swaps (L-005). This backend is for PROMPT ITERATION, so that prompt bugs are
    found here rather than discovered on the T4 where quota is the scarce resource. The
    numbers in the thesis come from the 7B on Colab.
    """

    def __init__(self, model: str = MLX_MODEL, max_image_px: tuple[int, int] = MAX_IMAGE_PX):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_name = model
        self.max_image_px = max_image_px
        self.model, self.processor = load(model)
        self.config = load_config(model)

    def _prepared(self, images: list[Path]) -> list[str]:
        """Downscale before inference. nuScenes frames are 1600x900 and the BEV is 768;
        full resolution buys nothing here and costs memory that 16 GB does not have."""
        import tempfile

        from PIL import Image

        out = []
        for p in images:
            im = Image.open(p)
            if im.width > self.max_image_px[0] or im.height > self.max_image_px[1]:
                im.thumbnail(self.max_image_px)
                tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
                im.convert("RGB").save(tmp, quality=92)
                out.append(str(tmp))
            else:
                out.append(str(p))
        return out

    def generate(self, images: list[Path], prompt: str, max_new_tokens: int = 700,
                 text: str | None = None) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        paths = self._prepared(images)
        # `text` carries the symbolic-BEV description (E6b), which is an INPUT, not part
        # of the instruction - kept separate so the shared answer rules stay byte-identical
        # across conditions and RQ3a compares inputs rather than wording.
        full = prompt if text is None else f"{prompt}\n\nTHE SCENE:\n{text}"
        formatted = apply_chat_template(self.processor, self.config, full,
                                        num_images=len(paths))
        res = generate(self.model, self.processor, formatted, image=paths or None,
                       max_tokens=max_new_tokens, verbose=False)
        return res.text if hasattr(res, "text") else str(res)


class TransformersBackend:
    """Colab T4 backend, Qwen2.5-VL-7B in 4-bit via bitsandbytes. Step E4.

    UNTESTED ON THE DEVELOPMENT MACHINE - the M4 has no CUDA, so this path has only ever
    been read, not run. Verify it on a 12-frame slice before committing quota to 1,800.
    The MLX backend is the tested one; this is its Colab twin and the two must agree on
    the contract `run_batch` depends on: `generate(images, prompt, max_new_tokens, text)
    -> str`, and a `model_name` attribute for the provenance column.

    4-bit because a 7B in fp16 is ~15 GB and the T4 has 16 GB with the KV cache still to
    come. Quantisation is not free (REFERENCES.md R-09) and L-005 already states the
    conclusions are about ACCESSIBLE open-weight models, not frontier capability.
    """

    def __init__(self, model: str = HF_MODEL, load_in_4bit: bool = True,
                 max_image_px: tuple[int, int] = MAX_IMAGE_PX):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        # E10 needs a SECOND model (Qwen3-VL-8B) on the same code path, and this class
        # previously hard-coded `Qwen2_5_VLForConditionalGeneration` -- a different
        # architecture simply will not load with it, so E10 would have died on the first
        # cell of a metered session. The Auto class dispatches on the checkpoint's own
        # config instead. `AutoModelForImageTextToText` is the current name; older
        # transformers call it `AutoModelForVision2Seq`, and Kaggle installs whatever is
        # latest, so both are accepted.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:                       # transformers < 4.46
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self.model_name = model
        self.max_image_px = max_image_px
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        # float16, never bfloat16: the T4 is Turing (cc 7.5) and has no bf16 at all.
        self.model = _AutoVLM.from_pretrained(
            model, quantization_config=quant, device_map="auto",
            torch_dtype=torch.float16)

        # The five completed arms were produced by Qwen2_5_VLForConditionalGeneration. If
        # the Auto dispatch ever resolves the DEFAULT model to something else, those
        # numbers stop being reproducible and the change must be deliberate, not silent.
        if model == HF_MODEL:
            got = type(self.model).__name__
            assert got == "Qwen2_5_VLForConditionalGeneration", (
                f"the default model now loads as {got}; the five published arms were run "
                "under Qwen2_5_VLForConditionalGeneration and would no longer be comparable")

        # Cap the vision tokens. Qwen-VL uses dynamic resolution, so an uncapped 1600x900
        # frame becomes a very large number of patches, and the KV cache is what actually
        # exhausts a T4 - not the weights. A processor that does not accept the pixel
        # bounds (a different family) must still load rather than abort the session.
        try:
            self.processor = AutoProcessor.from_pretrained(
                model, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        except (TypeError, ValueError):
            self.processor = AutoProcessor.from_pretrained(model)

    def generate(self, images: list[Path], prompt: str, max_new_tokens: int = 700,
                 text: str | None = None) -> str:
        import torch
        from PIL import Image

        pil = []
        for p in images:
            im = Image.open(p).convert("RGB")
            if im.width > self.max_image_px[0] or im.height > self.max_image_px[1]:
                im.thumbnail(self.max_image_px)
            pil.append(im)

        full = prompt if text is None else f"{prompt}\n\nTHE SCENE:\n{text}"
        content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": full}]
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=pil or None,
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        # Strip the prompt tokens, or every row carries the 1,200-token instruction back.
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def extract_json(raw: str) -> tuple[dict | None, str]:
    """Brace-balanced JSON extraction. Returns (parsed_or_None, failure_reason). Step E2.

    Replaces `re.search(r"\\{[\\s\\S]+?\\}", raw)` from docs/nuscens.py:1029. That `+?` is
    NON-GREEDY: it stops at the FIRST `}`, so any nested or multi-line JSON yields a
    truncated fragment, `json.loads` throws, the bare `except` swallows it, and every tag
    silently defaults to False. A model that answered perfectly scored as if it had said
    "no" to everything.

    Scanning depth rather than regex also handles the three things models actually emit:
    a ```json fence, prose before the object, and a trailing comma.

    The caller MUST record `reason` - "model X emitted invalid JSON on N% of frames" is
    RQ2b, a reportable finding, not something to hide in an except block.
    """
    import json
    import re

    if not raw or not raw.strip():
        return None, "empty_output"

    # Depth scan, ignoring braces inside strings. A regex cannot do this correctly.
    start = raw.find("{")
    if start < 0:
        return None, "no_json_object"
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None, "unbalanced_braces"

    blob = raw[start:end]
    try:
        return json.loads(blob), "ok"
    except json.JSONDecodeError:
        pass
    # One repair pass for the single most common malformation, then give up honestly.
    repaired = re.sub(r",(\s*[}\]])", r"\1", blob)          # trailing comma
    try:
        return json.loads(repaired), "ok_after_trailing_comma_fix"
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e.msg}"


def coerce_tags(parsed: dict, tags: list[str]) -> tuple[dict[str, bool], list[str]]:
    """Map a parsed object onto the frozen tag vocabulary. Returns (values, problems).

    A MISSING tag is not False. That conflation is precisely what made the original
    pipeline unscoreable, and it is why `problems` is returned rather than swallowed:
    Stage F must be able to separate "the model said no" from "the model did not answer".

    Accepts the spellings models actually produce - true/false, "yes"/"no", 1/0 - because
    scoring a correct answer wrong for its formatting measures the parser, not the model.
    """
    truthy = {"true": True, "yes": True, "y": True, "1": True,
              "false": False, "no": False, "n": False, "0": False}
    flat: dict[str, Any] = {}

    def walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)                  # prompts group tags by scope; flatten the groups
            else:
                flat[str(k).strip().lower()] = v

    walk(parsed)

    values, problems = {}, []
    for t in tags:
        if t.lower() not in flat:
            problems.append(f"missing:{t}")
            continue
        v = flat[t.lower()]
        if isinstance(v, bool):
            values[t] = v
        elif isinstance(v, (int, float)) and v in (0, 1):
            values[t] = bool(v)
        elif isinstance(v, str) and v.strip().lower() in truthy:
            values[t] = truthy[v.strip().lower()]
        else:
            problems.append(f"unparseable:{t}={v!r}")
    extra = [k for k in flat if k not in {t.lower() for t in tags}]
    problems += [f"extra:{k}" for k in extra]
    return values, problems


def completed_tokens(out_jsonl: Path | str) -> set[str]:
    """Tokens already written, for resume. Tolerates a truncated final line.

    A Colab session killed mid-write leaves a partial JSON line. Treating that as
    corruption and refusing to start would make the runner useless exactly when it is
    needed; treating it as complete would lose a frame silently. It is dropped and redone.

    Rows whose BACKEND failed are also not "done". Found 2026-09-06 by the resume test:
    when a session dies the runner records an error row per remaining frame, and counting
    those as complete meant a restart skipped every frame the crash had eaten - quota
    spent, frames never produced, and nothing to show it. A row where the model REPLIED is
    done even if the reply was unparseable, because that is a real observation and RQ2b's
    parse-failure rate depends on keeping it.
    """
    import json

    p = Path(out_jsonl)
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("backend_error"):
                continue      # the model never answered: retry it
            done.add(row["sample_token"])
        except (json.JSONDecodeError, KeyError):
            continue          # torn final line from a killed session
    return done


def run_batch(backend: Any, tokens: list[str], prompt_version: str,
              out_jsonl: Path | str, images_for: Any,
              prompt_text: str | None = None,
              tags: list[str] | None = None,
              max_new_tokens: int = 700,
              text_for: Any = None,
              prompt_for: Any = None,
              max_consecutive_failures: int = 5,
              progress: bool = True) -> dict[str, Any]:
    """Checkpointed batch inference. NON-NEGOTIABLE on free Colab, sessions die. Step E4.

      - append-only JSONL, one line per frame, flushed and fsynced immediately, never a
        list in RAM
      - on start, read out_jsonl and SKIP tokens already present
      - prompt_version, model, latency and the parse-failure REASON on every row

    `images_for(token) -> list[Path]` and the optional `text_for(token) -> str` are what
    make one runner serve every condition: camera, BEV raster, both, symbolic, temporal.
    The prompt is passed in rather than rebuilt here, so the exact string that produced a
    row is the one recorded.

    `prompt_for(token) -> str` overrides `prompt_text` PER ITEM, which Step F5 needs: an
    external VQA benchmark asks a different question of the same image, so the unit of work
    is a QUESTION, not a frame. It is a separate hook rather than a reuse of `text_for`
    because the backends splice text as `f"{prompt}\n\nTHE SCENE:\n{text}"` — a question
    smuggled through there would be labelled "THE SCENE", and since a row records neither
    the prompt nor the spliced text, the provenance would be unrecoverable.

    Nothing is scored here. A row carries the raw output and the parse reason, so Stage F
    can re-score without re-running the GPU - and so "the model emitted invalid JSON on
    N% of frames" stays a measurable finding (RQ2b) instead of an exception nobody sees.
    """
    import json
    import os
    import time

    from . import prompts as P

    if prompt_text is None and prompt_for is None:
        # Only fall back to the registry when there is no per-item prompt. F5's version is
        # not a registered TAG prompt, so building one would raise here -- after the model
        # had loaded, on metered time, which is F-065's shape.
        prompt_text = P.build_prompt(prompt_version)
    if tags is None:
        tags = P.scoreable_tags(P.load_schema())

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = completed_tokens(out_jsonl)
    todo = [t for t in tokens if t not in done]
    stats = {"n_requested": len(tokens), "n_skipped": len(done & set(tokens)),
             "n_run": 0, "n_parse_fail": 0, "n_incomplete": 0, "n_backend_error": 0}
    if progress:
        print(f"{len(todo)} to run, {stats['n_skipped']} already done", flush=True)

    consecutive = 0
    with open(out_jsonl, "a") as f:
        for i, tok in enumerate(todo):
            # Wall clock AND process CPU time. Wall alone is a trap: it counts time the
            # machine spent asleep. The E1 pilot recorded a median 3,657 s per frame -
            # about 61 minutes - which read as a catastrophic performance problem. It was
            # macOS 'Maintenance Sleep' on a 60-minute cycle; the real figure is 27-35 s
            # (F-056). A large wall/cpu ratio is the signature, so both are recorded and
            # Stage F can spot it instead of publishing it.
            t0, c0 = time.time(), time.process_time()
            _prompt = prompt_for(tok) if prompt_for is not None else prompt_text
            try:
                raw = backend.generate(images_for(tok), _prompt,
                                       max_new_tokens=max_new_tokens,
                                       text=text_for(tok) if text_for else None)
                err = None
            except Exception as e:                      # noqa: BLE001
                # A single frame must never end the run. Colab OOMs on one long output
                # and the remaining 1,799 are still worth having.
                raw, err = "", f"{type(e).__name__}: {e}"
            parsed, reason = extract_json(raw)
            values, problems = coerce_tags(parsed, tags) if parsed else ({}, [])
            if err is not None:
                stats["n_backend_error"] += 1
            elif parsed is None:
                stats["n_parse_fail"] += 1
            elif len(values) < len(tags):
                stats["n_incomplete"] += 1
            stats["n_run"] += 1

            f.write(json.dumps({
                "sample_token": tok,
                "backend_error": err is not None,
                "prompt_version": prompt_version,
                "model": getattr(backend, "model_name", type(backend).__name__),
                "raw": raw,
                "parse_reason": reason if err is None else err,
                "values": values,
                "problems": problems,
                "n_answered": len(values),
                "latency_s": round(time.time() - t0, 2),
                "cpu_s": round(time.process_time() - c0, 2),
            }) + "\n")
            f.flush()
            os.fsync(f.fileno())        # a Colab kill is not a graceful close

            # FAIL FAST on a PARSE-failure streak. The row above is written and fsynced
            # first: an unparseable reply is a real observation and RQ2b's rate is built
            # from it (D-045).
            #
            # Measured 2026-09-15: Qwen2.5-VL emits degenerate all-'!' output once the
            # vision-token count runs too high, silently and at a normal latency, so at
            # 5 full-resolution images that is EVERY frame -- hours of quota producing
            # nothing. The measured parse-failure rate is 0.03% over 3,516 calls (F-066),
            # so this many in a row is a broken configuration, not bad luck.
            #
            # BACKEND ERRORS ARE DELIBERATELY EXCLUDED. A dead backend raises immediately,
            # so a full run of error rows costs seconds rather than quota, and F-055/D-045
            # require those rows to exist so the restart knows which frames to retry.
            # Aborting there would break the resume contract to prevent waste that does
            # not happen.
            if err is not None:
                pass                      # infrastructure, not a model reply
            elif parsed is None:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= max_consecutive_failures:
                raise RuntimeError(
                    f"{consecutive} consecutive frames failed to parse -- aborting after "
                    f"{stats['n_run']} of {len(todo)} rather than spending the rest of the "
                    f"run on it. Last reason: {reason!r}. Rows written so far are kept and "
                    f"the run resumes from them. If the raw text is all '!', the vision "
                    f"token count is too high: lower the per-image cap (MAX_IMAGE_PX) or "
                    f"send fewer images.")

            if progress and (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}  parse_fail={stats['n_parse_fail']}",
                      flush=True)
    return stats
