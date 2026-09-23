#!/usr/bin/env bash
# Step B4 — fetch CAM_FRONT keyframes for the evaluation subset.
#
# Source is the PUBLIC nuScenes S3 mirror, so no signed URL and no browser trip:
#   https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/
#
# Per part: download ~4 GiB of *_keyframes.tgz, extract ONLY samples/CAM_FRONT/,
# delete the tarball. Peak extra disk is one tarball (~4 GiB); the kept images are
# ~4.4 GB for all 34,149 CAM_FRONT keyframes, of which 6,025 are the subset's.
#
# Resumable at BYTE level (curl -C -) and at PART level (.done markers), because a
# 40 GiB transfer on a laptop will be interrupted. Re-run this script as often as
# needed; finished parts are skipped.
#
# Usage:  ./scripts/fetch_b4_images.sh          # all 10 parts
#         ./scripts/fetch_b4_images.sh 03 07    # only these parts
set -uo pipefail

BASE="https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/nuscenes"
WORK="$ROOT/data/_b4_tmp"
mkdir -p "$DEST" "$WORK"

PARTS=("$@"); [ ${#PARTS[@]} -eq 0 ] && PARTS=(01 02 03 04 05 06 07 08 09 10)

for n in "${PARTS[@]}"; do
  f="v1.0-trainval${n}_keyframes.tgz"
  if [ -f "$WORK/$f.done" ]; then
    echo "[$n/10] $f already extracted — skipping"; continue
  fi

  echo "[$n/10] downloading $f (~4 GiB, resumable)…"
  # Retry in the SHELL, never with curl --retry. curl's internal retry restarts the
  # transfer from byte 0 and TRUNCATES the -o file: `-C -` is evaluated once per
  # invocation, not per internal retry. Measured 2026-09-04: a "Recv failure:
  # Connection reset by peer" at 3.32 GiB of part 05 discarded all 3.32 GiB and
  # silently restarted, which read as a collapse from 4.2 to 1.2 MiB/s.
  # A fresh curl per attempt re-reads the on-disk size, so -C - genuinely resumes.
  # --speed-limit/--speed-time kill a connection that is open but stalled, so a dead
  # socket cannot hang the run indefinitely.
  ok=0
  for attempt in $(seq 1 40); do
    if curl -fL -C - --speed-limit 10240 --speed-time 60 \
            -o "$WORK/$f" "$BASE/$f"; then ok=1; break; fi
    got=$(stat -f%z "$WORK/$f" 2>/dev/null || echo 0)
    # Back off after the first few: attempts 1-10 are for a transient reset, but a
    # real network outage needs minutes, not 5 s. Measured 2026-09-05: an overnight
    # wifi drop produced 30 consecutive "Could not resolve host" and burned the whole
    # 40-attempt budget in 200 s, abandoning part 08 at 1742 MiB. The partial file
    # survived intact - that part of the fix worked - but the budget did not.
    [ "$attempt" -lt 10 ] && naptime=5 || naptime=30
    echo "  … attempt $attempt interrupted at $((got/1048576)) MiB; resuming in ${naptime}s" >&2
    sleep "$naptime"
  done
  if [ "$ok" -ne 1 ]; then
    echo "  ✗ download failed for $f — re-run the script to resume" >&2
    continue
  fi

  echo "[$n/10] extracting samples/CAM_FRONT/ …"
  if tar -xzf "$WORK/$f" -C "$DEST" --include='samples/CAM_FRONT/*'; then
    touch "$WORK/$f.done"
    rm -f "$WORK/$f"
    echo "  ✓ part $n done; tarball deleted"
  else
    echo "  ✗ extraction failed for $f — tarball kept at $WORK/$f" >&2
  fi
done

echo
echo "CAM_FRONT keyframes on disk: $(find "$DEST/samples/CAM_FRONT" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
echo "Now verify:  ./venv/bin/python -c \"import src.data,json;print(json.dumps(src.data.verify_subset_images(),indent=1))\""
