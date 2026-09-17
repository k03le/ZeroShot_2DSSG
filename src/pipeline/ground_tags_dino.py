#!/usr/bin/env python3
"""
ground_tags_dino.py — Stage 2 of the VLM-first front-end.

PURPOSE
    Consume the VLM-generated tag lists from tag_frames_vlm.py (Stage 1) and
    ground each tag with GroundingDINO into 2D boxes. Output schema matches
    what extract_relations_vlm.py reads: per-detection `tag`, `bbox_2d`
    ([x1,y1,x2,y2] pixels), `score_2d` — the score-key naming mismatch that
    broke downstream consumers previously cannot recur because we emit the
    canonical key here.

INPUT
    --tags-json   vlmfirst_tags/{scene}.json  (from tag_frames_vlm.py)
    --frames-dir  directory with the frame images
    --ext         appended to each frame key to form the filename
                  (ARKitScenes: ".png"; 3DSSG color: ".color.jpg" — the tagger
                  strips ".color" from stems, this re-appends it)
    --config / --checkpoint   GroundingDINO config .py and weights .pth.
                  REQUIRED — do not guess; locate with:
                    find /data /home/mmc-user -name "GroundingDINO_Swin*.py" 2>/dev/null
                    find /data /home/mmc-user -name "*.pth" 2>/dev/null | grep -i ground

OUTPUT
    --out-json    e.g. /data/mmc-user/outputs/vlmfirst_boxes/{scene}.json
                  { "<frame_key>": [ {tag, bbox_2d, score_2d,
                                      grounded_phrase}, ... ], ... }
                  + viz/<frame_key>_viz.jpg (boxes + tags, upscaled) unless
                  --no-viz. Atomic write; --resume / --overwrite, mutually
                  exclusive, refuse silent overwrite.

USAGE
    conda activate ram_pipeline          # GroundingDINO lives here (v1 pipeline)
    # MANDATORY before any GPU work on wasti-new (Pascal / torch pin):
    python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
    #   expect 2.5.1+cu121 with sm_60 present

    python ground_tags_dino.py \
        --tags-json /data/mmc-user/outputs/vlmfirst_tags/40753679.json \
        --frames-dir /data/mmc-user/arkitscenes/raw/raw/Training/40753679/lowres_wide \
        --ext .png \
        --out-json /data/mmc-user/outputs/vlmfirst_boxes/40753679.json \
        --config <path/to/GroundingDINO_SwinT_OGC.py> \
        --checkpoint <path/to/groundingdino_swint_ogc.pth> \
        --box-threshold 0.35 --text-threshold 0.25

    NOTE: 0.35/0.25 are the repo defaults. If the v1 RAM++->GroundingDINO stage
    used different thresholds, USE THOSE — the grounder configuration is part
    of the frozen comparison surface (front-end tag source must be the only
    variable). Check the old stage script before the real run.

DESIGN NOTES
    - Caption per frame: tags joined as "tag1 . tag2 . tag3 ." (repo format).
    - GroundingDINO may return phrases that are token-subsets of the input
      tags ("office chair" for "black office chair"). Each returned phrase is
      mapped BACK to the closest input tag so the VLM's open-vocab label —
      not the tokenizer's fragment — flows to the relation stage. Mapping
      order: exact -> substring -> token overlap -> keep raw phrase (warned:
      these are the grounding-hallucination candidates to eyeball).
    - One phrase can ground MULTIPLE boxes (two chairs from one "black office
      chair" tag). All are kept; per-tag greedy IoU dedupe (--dedupe-iou)
      removes near-identical repeats only. Whether duplicate instances
      survive as same-tag boxes is exactly the open question here — check the viz.
    - --max-area-frac ports the whole-frame filter: a tag grounded to
      ~the entire frame is the grounding-hallucination analogue of RAM++'s
      scene-label boxes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

LOG = logging.getLogger("ground_tags_dino")
_INTERRUPTED = False

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
           "#F8C471", "#82E0AA"]


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        LOG.warning("caught signal %s — finishing current frame, flushing, exiting",
                    signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _atomic_write_json(obj, path: Path) -> None:
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


def _map_phrase_to_tag(phrase: str, tags: list[str]) -> tuple[str, bool]:
    """Map a GroundingDINO output phrase back to the originating VLM tag.

    Returns (tag, mapped_ok). Falls back to the raw phrase with mapped_ok=False
    — those are the candidates for grounding hallucination / tokenizer mangling
    and are logged."""
    ph = phrase.strip().lower()
    if not ph:
        return phrase, False
    for t in tags:                       # exact
        if ph == t:
            return t, True
    for t in tags:                       # substring either way
        if ph in t or t in ph:
            return t, True
    ph_tok = set(ph.split())             # best token overlap
    best, best_n = None, 0
    for t in tags:
        n = len(ph_tok & set(t.split()))
        if n > best_n:
            best, best_n = t, n
    if best is not None and best_n > 0:
        return best, True
    return phrase, False


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe_per_tag(dets: list[dict], iou_thr: float) -> list[dict]:
    """Greedy per-tag NMS: within each tag, drop boxes overlapping a
    higher-scoring kept box above iou_thr. Cross-tag boxes are never merged
    (two different tags on one region is information, not duplication)."""
    out = []
    by_tag: dict[str, list[dict]] = {}
    for d in dets:
        by_tag.setdefault(d["tag"], []).append(d)
    for tag, group in by_tag.items():
        group = sorted(group, key=lambda d: -d["score_2d"])
        kept: list[dict] = []
        for d in group:
            if all(_iou(d["bbox_2d"], k["bbox_2d"]) < iou_thr for k in kept):
                kept.append(d)
        dropped = len(group) - len(kept)
        if dropped:
            LOG.debug("dedupe: dropped %d near-duplicate box(es) for tag %r",
                      dropped, tag)
        out.extend(kept)
    return out


def _draw_viz(img, dets, upscale):
    from PIL import Image, ImageDraw, ImageFont
    try:
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        fnt = ImageFont.load_default()
    im = img.resize((img.width * upscale, img.height * upscale),
                    Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, det in enumerate(dets):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = [c * upscale for c in det["bbox_2d"]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        badge = f"{det['tag']} {det['score_2d']:.2f}"
        tb = d.textbbox((0, 0), badge, font=fnt)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        by = max(0, y1)
        d.rectangle([x1, by, x1 + tw + 6, by + th + 6], fill=color)
        d.text((x1 + 3, by + 2), badge, fill="black", font=fnt)
    return im


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: ground VLM tags with GroundingDINO.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--tags-json", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--ext", default=".png",
                   help='".png" for ARKitScenes, ".color.jpg" for 3DSSG')
    p.add_argument("--out-json", required=True)
    p.add_argument("--config", required=True,
                   help="GroundingDINO model config .py — locate with: "
                        "find /data /home/mmc-user -name 'GroundingDINO_Swin*.py'")
    p.add_argument("--checkpoint", required=True,
                   help="GroundingDINO weights .pth — locate with: "
                        "find /data /home/mmc-user -name '*.pth' | grep -i ground")
    p.add_argument("--box-threshold", type=float, default=0.35,
                   help="repo default 0.35 — MATCH THE v1 STAGE VALUE if different")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="repo default 0.25 — MATCH THE v1 STAGE VALUE if different")
    p.add_argument("--max-area-frac", type=float, default=0.85,
                   help="drop boxes covering more than this frame fraction "
                        "(whole-frame filter, now catching grounding "
                        "hallucinations)")
    p.add_argument("--dedupe-iou", type=float, default=0.9,
                   help="per-tag greedy NMS threshold; 1.0 disables")
    p.add_argument("--frames", nargs="*", default=None,
                   help="restrict to these frame keys (default: all in tags JSON)")
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--device", default="cuda")
    mx = p.add_mutually_exclusive_group()
    mx.add_argument("--resume", action="store_true")
    mx.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    _install_signal_handlers()

    # ---- env / import guards ----------------------------------------------
    try:
        import torch
    except ImportError:
        LOG.error("torch not importable — wrong env? conda activate ram_pipeline")
        return 2
    archs = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    LOG.info("torch %s | arch_list %s", torch.__version__, archs)
    if not torch.__version__.startswith("2.5.1"):
        LOG.warning("torch is NOT the pinned 2.5.1 build — on wasti-new this "
                    "risks cudaErrorNoKernelImageForDevice (Pascal). "
                    "Stop and check the env unless you know why.")
    try:
        from groundingdino.util.inference import load_model, load_image, predict
    except ImportError as exc:
        LOG.error("groundingdino not importable in this env: %s", exc)
        LOG.error("It was importable in the v1 pipeline env — confirm: "
                  "conda activate ram_pipeline && python -c "
                  "'import groundingdino; print(groundingdino.__file__)'")
        return 2
    from PIL import Image

    for path, what in ((args.config, "config"), (args.checkpoint, "checkpoint")):
        if not Path(path).exists():
            LOG.error("%s not found: %s — do not guess this path; use the find "
                      "commands in --help", what, path)
            return 2

    # ---- load inputs -------------------------------------------------------
    tags_path = Path(args.tags_json)
    try:
        tags_data = json.loads(tags_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("cannot read %s: %s", tags_path, exc)
        return 2

    frame_keys = args.frames or sorted(tags_data.keys())
    missing = [k for k in frame_keys if k not in tags_data]
    if missing:
        LOG.error("frame key(s) not in tags JSON: %s", missing)
        return 2

    frames_dir = Path(args.frames_dir)
    img_paths = {}
    for k in frame_keys:
        ip = frames_dir / f"{k}{args.ext}"
        if not ip.exists():
            LOG.error("image not found: %s — check --ext and --frames-dir "
                      "(ls %s | head)", ip, frames_dir)
            return 2
        img_paths[k] = ip

    out_path = Path(args.out_json)
    results: dict = {}
    if out_path.exists():
        if args.resume:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            LOG.info("--resume: %d frame(s) already in %s", len(results), out_path.name)
        elif not args.overwrite:
            LOG.error("%s exists — pass --resume or --overwrite. No silent "
                      "overwrite.", out_path)
            return 2

    # ---- model -------------------------------------------------------------
    LOG.info("loading GroundingDINO (%s)...", Path(args.checkpoint).name)
    t0 = time.perf_counter()
    model = load_model(args.config, args.checkpoint, device=args.device)
    LOG.info("model loaded in %.1fs", time.perf_counter() - t0)

    latencies = []
    n_done = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(frame_keys, desc="grounding")
    except ImportError:
        iterator = frame_keys

    for key in iterator:
        if _INTERRUPTED:
            break
        if key in results and args.resume:
            continue
        tags = tags_data[key].get("tags", [])
        if not tags:
            LOG.info("%s: 0 tags from Stage 1 — recording empty detection list "
                     "(genuine abstention propagates)", key)
            results[key] = []
            continue

        caption = " . ".join(tags) + " ."
        image_source, image = load_image(str(img_paths[key]))
        h, w = image_source.shape[:2]

        t0 = time.perf_counter()
        boxes, logits, phrases = predict(
            model=model, image=image, caption=caption,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device)
        dt = time.perf_counter() - t0
        latencies.append(dt)

        dets, dropped_area, unmapped = [], [], []
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype)
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = (box * scale).tolist()
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            af = (x2 - x1) * (y2 - y1) / float(w * h)
            tag, ok = _map_phrase_to_tag(phrase, tags)
            if not ok:
                unmapped.append(phrase)
            if af > args.max_area_frac:
                dropped_area.append(f"{tag}({af:.0%})")
                continue
            dets.append({
                "tag": tag,
                "bbox_2d": [round(x1, 1), round(y1, 1),
                            round(x2, 1), round(y2, 1)],
                "score_2d": round(float(logit), 4),
                "grounded_phrase": phrase,
            })

        if args.dedupe_iou < 1.0:
            dets = _dedupe_per_tag(dets, args.dedupe_iou)

        ungrounded = [t for t in tags
                      if not any(d["tag"] == t for d in dets)]
        if dropped_area:
            LOG.info("%s: dropped whole-frame box(es): %s", key,
                     ", ".join(dropped_area))
        if unmapped:
            LOG.warning("%s: phrase(s) not mappable to any input tag "
                        "(hallucination candidates): %s", key, unmapped)
        if ungrounded:
            LOG.warning("%s: tag(s) the VLM produced but GroundingDINO did NOT "
                        "ground: %s — VLM-tag-hallucination candidates, or "
                        "threshold casualties", key, ungrounded)

        results[key] = dets
        n_done += 1
        LOG.info("%s: %d tag(s) -> %d box(es) in %.2fs: %s", key, len(tags),
                 len(dets), dt,
                 [f"{d['tag']}@{d['score_2d']:.2f}" for d in dets])

        if not args.no_viz and dets:
            viz_dir = out_path.parent / "viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(img_paths[key]).convert("RGB")
            _draw_viz(img, dets, args.upscale).save(
                viz_dir / f"{key}_viz.jpg", quality=92)

    if not results:
        LOG.error("nothing to write")
        return 1
    _atomic_write_json(results, out_path)
    LOG.info("wrote %d frame(s) (%d new) -> %s", len(results), n_done, out_path)
    if latencies:
        ls = sorted(latencies)
        LOG.info("grounding latency (stage 2): n=%d  median=%.2fs  "
                 "range=[%.2f, %.2f]s", len(ls), ls[len(ls) // 2], ls[0], ls[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())