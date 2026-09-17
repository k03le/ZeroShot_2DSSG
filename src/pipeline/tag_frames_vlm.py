#!/usr/bin/env python3
"""
tag_frames_vlm.py — Stage 1 of the VLM-first detection front-end.

PURPOSE
    The VLM generates the object tags FIRST; the
    tags are then fed to GroundingDINO as grounding phrases to produce boxes.
    This replaces RAM++ (v1) and YOLOE (Goal-2, superseded before
    execution) as the tag source. This script is that first stage: one whole
    RGB frame in, one open-vocabulary tag list out.

    Deliberately path-agnostic: frames are given as explicit image paths or as
    (--frames-dir + --frames) pairs, so the SAME script runs on ARKitScenes
    lowres_wide frames and on 3DSSG sequence frames with no adapter.

INPUT
    --frames-dir + --frames   e.g. --frames-dir /data/.../40753679/lowres_wide
                                   --frames 40753679_6813.238 ...   (.png appended)
    or --images               explicit image paths, any extension

OUTPUT
    --out-json                one JSON file: { "<frame_key>": {"tags": [...],
                              "t_vlm_s": float, "model": str, "raw": str}, ... }
                              frame_key = filename stem. Atomic write; --resume
                              merges with an existing file, --overwrite replaces
                              it, neither -> refuse if it exists.
    --dump-prompt             also write the exact prompt to <out-json>.prompt.txt
                              (audit artifact for the report; the prompt is a
                              research object in this project).

USAGE
    conda activate ram_pipeline    # needs only requests + PIL + stdlib

    python tag_frames_vlm.py \
        --frames-dir /data/mmc-user/arkitscenes/raw/raw/Training/40753679/lowres_wide \
        --frames 40753679_6813.238 40753679_6841.941 40753679_6832.979 \
                 40753679_6844.890 40753679_6811.522 40753679_6804.958 \
        --out-json /data/mmc-user/outputs/vlmfirst_tags/40753679.json \
        --model qwen3-vl:8b-instruct-bf16 \
        --tag-style instances \
        --dump-prompt

DESIGN NOTES
    --tag-style categories : plain category nouns ("chair", "desk"). What RAM++
                             produced; GroundingDINO returns N boxes per phrase,
                             so duplicate instances share a tag (the duplicate-
                             instance-collapse problem persists).
    --tag-style instances  : one phrase PER VISIBLE INSTANCE, with a short
                             disambiguating attribute when two instances share a
                             category ("black office chair", "grey office chair").
                             The attempt to address duplicate-instance collapse
                             at the tag level.
                             GroundingDINO accepts attribute phrases.
    Both are legitimate prompt-engineering variants (the project's core
    contribution); A/B them qualitatively, per scope (no formal protocol).

    Determinism: temperature 0, seed 42, matching extract_relations_vlm.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    print("ERROR: `requests` not importable — wrong env? Run:\n"
          "  conda activate ram_pipeline && python -c 'import requests'",
          file=sys.stderr)
    raise

LOG = logging.getLogger("tag_frames_vlm")
_INTERRUPTED = False

PROMPT_CATEGORIES = """You are given a photograph of an indoor scene.
List every distinct physical OBJECT visible in the image.

Rules:
- Output ONLY a JSON array of lowercase noun phrases, nothing else.
- One entry per object CATEGORY (e.g. "chair" once even if several chairs).
- Objects only: no room types, no scene labels, no activities, no materials
  on their own, no parts of objects listed separately from the object.
- If the image is too blurry or empty to identify anything, output [].

Example output: ["office chair", "desk", "monitor", "trash can"]"""

PROMPT_INSTANCES = """You are given a photograph of an indoor scene.
List every distinct physical object INSTANCE visible in the image.

Rules:
- Output ONLY a JSON array of lowercase noun phrases, nothing else.
- One entry PER INSTANCE: if two chairs are visible, output two entries.
- When several instances share a category, add ONE short visual attribute that
  tells them apart (color, material, or size — e.g. "black office chair",
  "grey office chair"). Do NOT use positional words (left, right, near, front,
  behind) as the attribute.
- Objects only: no room types, no scene labels, no activities, no parts of
  objects listed separately from the whole object.
- If the image is too blurry or empty to identify anything, output [].

Example output: ["black office chair", "grey office chair", "desk", "monitor"]"""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        LOG.warning("caught signal %s — finishing current frame, then exiting clean",
                    signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_tags(raw: str) -> list[str] | None:
    """Extract a JSON array of strings from the model output.

    Tolerates markdown fences and stray prose around the array; returns None if
    no parseable array is found (caller logs the raw text — never silently
    coerce -- a past bug came from doing exactly that)."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE).strip()
    candidates = [txt]
    m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and all(isinstance(t, str) for t in obj):
            # normalise: lowercase, strip, drop empties and exact duplicates
            seen, out = set(), []
            for t in (s.strip().lower() for s in obj):
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
    return None


def call_ollama(host: str, model: str, prompt: str, image_path: Path,
                timeout: float) -> tuple[str, float]:
    with image_path.open("rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("ascii")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{host.rstrip('/')}/api/generate", json=payload,
                         timeout=timeout)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json().get("response", ""), dt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="VLM-first Stage 1: whole-frame open-vocabulary object tagging.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = p.add_argument_group("frame sources (use one style)")
    src.add_argument("--frames-dir", default=None,
                     help="directory containing the frame images")
    src.add_argument("--frames", nargs="*", default=None,
                     help="frame stems inside --frames-dir (extension appended)")
    src.add_argument("--ext", default=".png",
                     help="extension appended to --frames stems (default .png; "
                          "3DSSG color frames are .jpg)")
    src.add_argument("--images", nargs="*", default=None,
                     help="explicit image paths (alternative to --frames-dir/--frames)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--host", default="http://10.121.198.34:11434",
                   help="Ollama base URL")
    p.add_argument("--model", default="qwen3-vl:8b-instruct-bf16")
    p.add_argument("--tag-style", choices=["categories", "instances"],
                   default="instances")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="per-call HTTP timeout in seconds (first call may pay a "
                        "cold model load)")
    p.add_argument("--dump-prompt", action="store_true")
    mx = p.add_mutually_exclusive_group()
    mx.add_argument("--resume", action="store_true",
                    help="keep existing frames in --out-json, tag only missing ones")
    mx.add_argument("--overwrite", action="store_true",
                    help="replace --out-json entirely")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    _install_signal_handlers()

    # ---- resolve frame list ------------------------------------------------
    images: list[Path] = []
    if args.images:
        images = [Path(x) for x in args.images]
    elif args.frames_dir and args.frames:
        d = Path(args.frames_dir)
        if not d.is_dir():
            LOG.error("--frames-dir does not exist: %s", d)
            LOG.error("check with:  ls %s", d.parent)
            return 2
        images = [d / f"{stem}{args.ext}" for stem in args.frames]
    else:
        LOG.error("give either --images, or --frames-dir together with --frames")
        return 2

    missing = [str(x) for x in images if not x.exists()]
    if missing:
        LOG.error("missing image file(s):\n  %s", "\n  ".join(missing))
        LOG.error("nothing was tagged — fix the paths first (do not guess them; "
                  "ls the directory)")
        return 2

    out_path = Path(args.out_json)
    existing: dict = {}
    if out_path.exists():
        if args.resume:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                LOG.info("--resume: %d frame(s) already present in %s",
                         len(existing), out_path.name)
            except (OSError, json.JSONDecodeError) as exc:
                LOG.error("could not read existing %s for --resume: %s", out_path, exc)
                return 2
        elif not args.overwrite:
            LOG.error("%s exists. Pass --resume to extend it or --overwrite to "
                      "replace it. Refusing to overwrite silently.", out_path)
            return 2

    prompt = PROMPT_INSTANCES if args.tag_style == "instances" else PROMPT_CATEGORIES

    if args.dump_prompt:
        pp = out_path.parent / (out_path.name + ".prompt.txt")
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(f"# model={args.model} tag_style={args.tag_style} "
                      f"temperature=0 seed=42\n\n{prompt}\n", encoding="utf-8")
        LOG.info("prompt dumped → %s", pp)

    # ---- main loop ---------------------------------------------------------
    results = dict(existing)
    n_new = 0
    latencies = []
    for img in images:
        key = img.stem  # ARKitScenes: 40753679_6813.238 ; 3DSSG: frame-000089.color
        # 3DSSG stems carry the .color suffix inside the stem; strip it:
        if key.endswith(".color"):
            key = key[: -len(".color")]
        if _INTERRUPTED:
            LOG.warning("interrupted — writing partial results")
            break
        if key in results and args.resume:
            LOG.info("skip (resume): %s", key)
            continue
        try:
            raw, dt = call_ollama(args.host, args.model, prompt, img, args.timeout)
        except requests.ConnectionError:
            LOG.error("cannot reach Ollama at %s — is it up and bound to "
                      "0.0.0.0:11434 on wasti-old?\n"
                      "  curl -s %s/api/tags", args.host, args.host)
            break
        except requests.Timeout:
            LOG.error("timeout after %.0fs on %s — cold model load? bump "
                      "--timeout, or warm the model first", args.timeout, key)
            break
        except requests.HTTPError as exc:
            LOG.error("HTTP error on %s: %s — response body: %s",
                      key, exc, getattr(exc.response, "text", "")[:400])
            break

        tags = _parse_tags(raw)
        if tags is None:
            LOG.warning("UNPARSEABLE model output on %s — stored raw, tags=[]. "
                        "Raw (first 300 chars): %r", key, raw[:300])
            tags = []
        results[key] = {"tags": tags, "t_vlm_s": round(dt, 3),
                        "model": args.model, "tag_style": args.tag_style,
                        "raw": raw}
        latencies.append(dt)
        n_new += 1
        LOG.info("%s  →  %d tag(s) in %.2fs: %s", key, len(tags), dt, tags)

    if not results:
        LOG.error("nothing to write")
        return 1

    _atomic_write_json(results, out_path)
    LOG.info("wrote %d frame(s) (%d new) → %s", len(results), n_new, out_path)
    if latencies:
        lat_sorted = sorted(latencies)
        med = lat_sorted[len(lat_sorted) // 2]
        LOG.info("tagging latency (this stage only): n=%d  median=%.2fs  "
                 "range=[%.2f, %.2f]s  — first call may include a cold load; "
                 "report warm calls only", len(latencies), med,
                 lat_sorted[0], lat_sorted[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())