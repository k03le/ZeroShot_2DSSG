"""
Script:  viz_gt_graph.py
Stage:   GT inspection / report figures (no VLM, no GPU).
Purpose: Draw the 3DSSG ground-truth scene graph for each frame ON TOP of the
         upright colour frame:每 visible instance gets a translucent mask tint and
         a label at its centroid, and each GT edge is drawn as an arrow from
         subject to object with the predicate written at the midpoint.

         Two uses. (1) Sanity: confirm the GT we are about to score against is
         what we think it is -- right objects, right relations, right frame.
         (2) Diagnosis: rendering the DIRECTIONAL stratum makes the camera-relative
         reference-frame mismatch visible -- stored 'left' arrows will not agree
         with what the camera sees. That disagreement is a report figure.

KEY FACTS RELIED ON (verified 2026-08-08):
    - rendered.instances.png is NATIVELY UPRIGHT (540x960): NO rotation applied.
    - mask pixel values are objectIds in the objects.json / relationships.json
      integer space.

INPUT   --census      gt_census.json (frame list: scan + frame_no)
        --frames-root the upright colour tree (<scan>/lowres_wide/<key>.png)
        --dssg-root   relationships.json, objects.json, 3dssg_data/<scan>/sequence/
OUTPUT  --out-dir     one <frame_key>__gt_<strata>.png per frame (atomic writes)
USAGE
    python viz_gt_graph.py \
        --census /data/mmc-user/outputs/3dssg_upright_v2/gt_census.json \
        --out-dir /data/mmc-user/outputs/3dssg_gt_viz \
        --strata support
    # then, to demonstrate the camera-relative reference-frame mismatch:
    #   --strata directional --out-dir /data/mmc-user/outputs/3dssg_gt_viz_dir
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOG = logging.getLogger("gtviz")

DIRECTIONAL = {"left", "right", "front", "behind", "higher than", "lower than",
               "bigger than", "smaller than"}
COMPARATIVE = {"same object type", "same color", "same shape", "same material",
               "same texture", "same state", "same as", "brighter than",
               "darker than", "cleaner than", "messier than"}
SYMMETRIC = {"close by", "same object type", "same color", "same shape",
             "same material", "same texture", "same state", "same as"}

PALETTE = [(255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 165, 0),
           (186, 85, 211), (0, 206, 209), (255, 20, 147), (154, 205, 50),
           (255, 215, 0), (100, 149, 237), (219, 112, 147), (46, 139, 87)]


def strata_of(pred: str) -> str:
    p = pred.strip().lower()
    if p == "close by":
        return "close"
    if p in DIRECTIONAL:
        return "directional"
    if p in COMPARATIVE:
        return "comparative"
    return "support"


def arrow(dr, p0, p1, colour, width=3, head=13):
    dr.line([p0, p1], fill=colour, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (+1, -1):
        a = ang + s * 2.6
        dr.line([p1, (p1[0] + head * math.cos(a), p1[1] + head * math.sin(a))],
                fill=colour, width=width)


def label(dr, xy, text, colour, font):
    x, y = xy
    try:
        l, t, r, b = dr.textbbox((x, y), text, font=font)
    except AttributeError:                       # very old PIL
        w, h = dr.textsize(text, font=font); l, t, r, b = x, y, x + w, y + h
    dr.rectangle([l - 3, t - 2, r + 3, b + 2], fill=(0, 0, 0))
    dr.text((x, y), text, fill=colour, font=font)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render 3DSSG GT scene graphs per frame.")
    ap.add_argument("--census", required=True)
    ap.add_argument("--frames-root",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strata", nargs="*", default=["support"],
                    choices=["support", "close", "directional", "comparative", "all"])
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--upscale", type=int, default=2, help="for text legibility")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    want = None if "all" in args.strata else set(args.strata)

    root = Path(args.dssg_root)
    def indexed(fn, key):
        raw = json.loads((root / fn).read_text())
        return {e["scan"]: e[key] for e in raw["scans"]}
    rels = indexed("relationships.json", "relationships")
    objs = indexed("objects.json", "objects")

    cen = json.loads(Path(args.census).read_text())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        LOG.warning("DejaVuSans not found; labels will be small")

    tag = "-".join(sorted(want)) if want else "all"
    n_done = 0
    for fr in cen["frames"]:
        key = fr["frame_key"]
        scan, no = key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1])
        dst = out / f"{key}__gt_{tag}.png"
        if dst.exists() and not args.overwrite:
            LOG.info("%s exists — skipped (use --overwrite)", dst.name); continue

        col_p = Path(args.frames_root) / scan / "lowres_wide" / f"{key}.png"
        msk_p = root / args.data_subdir / scan / "sequence" / f"frame-{no:06d}.rendered.instances.png"
        try:
            col = Image.open(col_p).convert("RGB")
            mask = np.asarray(Image.open(msk_p))          # native upright, no rotation
        except OSError as e:
            LOG.error("%s: %s — skipped", key, e); continue
        if (mask.shape[1], mask.shape[0]) != col.size:
            LOG.error("%s: mask %s vs colour %s mismatch — skipped",
                      key, mask.shape[1::-1], col.size); continue

        ids, counts = np.unique(mask, return_counts=True)
        vis = {int(i): int(c) for i, c in zip(ids, counts) if i != 0 and c >= args.min_px}
        if not vis:
            LOG.warning("%s: no visible objects — skipped", key); continue
        labels = {int(o["id"]): o.get("label", "?") for o in objs.get(scan, [])}

        U = args.upscale
        im = col.resize((col.width * U, col.height * U), Image.LANCZOS)
        tint = Image.new("RGBA", im.size, (0, 0, 0, 0))
        td = ImageDraw.Draw(tint)
        cent, colour_of = {}, {}
        for n, oid in enumerate(sorted(vis)):
            c = PALETTE[n % len(PALETTE)]
            colour_of[oid] = c
            ys, xs = np.nonzero(mask == oid)
            cent[oid] = (float(xs.mean()) * U, float(ys.mean()) * U)
            td.rectangle([xs.min() * U, ys.min() * U, xs.max() * U, ys.max() * U],
                         outline=c + (200,), width=2)
        im = Image.alpha_composite(im.convert("RGBA"), tint).convert("RGB")
        dr = ImageDraw.Draw(im)

        drawn, n_edges = set(), 0
        for e in rels.get(scan, []):
            s_id, o_id, pname = int(e[0]), int(e[1]), str(e[3])
            if s_id not in vis or o_id not in vis:
                continue
            st = strata_of(pname)
            if want and st not in want:
                continue
            if pname.strip().lower() in SYMMETRIC:
                sig = (min(s_id, o_id), max(s_id, o_id), pname)
                if sig in drawn:
                    continue
                drawn.add(sig)
            p0, p1 = cent[s_id], cent[o_id]
            arrow(dr, p0, p1, colour_of[s_id])
            label(dr, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 - 8), pname,
                  (255, 255, 255), font)
            n_edges += 1

        for oid in sorted(vis):
            label(dr, cent[oid], f"{oid}:{labels.get(oid, '?')}", colour_of[oid], font)
        label(dr, (8, 8), f"{key}  |  {len(vis)} objects, {n_edges} GT edges [{tag}]",
              (255, 255, 255), font)

        tmp = dst.with_name(f".{dst.stem}.tmp{os.getpid()}{dst.suffix}")
        im.save(tmp, "PNG"); os.replace(tmp, dst)
        n_done += 1
        LOG.info("%-46s %2d objects, %2d edges → %s", key, len(vis), n_edges, dst.name)

    print(f"\nwrote {n_done} figure(s) to {out}")
    print("browse over SSH:  python contact_sheet.py --help   (or open individually in VS Code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())