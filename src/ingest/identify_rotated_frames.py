"""
Script:  identify_rotated_frames.py
Stage:   3DSSG ingest -- pre-flight for the 15 hand-rotated colour frames.
Purpose: The hand-rotated frames sit flat as `frame-NNNNNN.color.jpg` with no scan
         in the path, and a given frame number exists in several scans. Recover,
         per file and from pixel content alone:
             (1) which scan it came from,
             (2) which np.rot90 k the hand rotation corresponds to,
             (3) whether the rotation is real pixels or only an EXIF flag.
         Matching is by mean-absolute-difference on a small grayscale thumbnail,
         which is robust to Preview's JPEG re-encode but not to a different scene.

         READ-ONLY. Writes nothing except the optional --json-out report.

Input:   --flat-dir    directory holding frame-NNNNNN.color.jpg (any names ending
                       in a 6-digit frame number before ".color")
         --dssg-root/--data-subdir  -> <root>/<subdir>/<scan>/sequence/frame-%06d.color.jpg
Output:  console table (scan, k, MAE, margin to runner-up, EXIF verdict)
         --json-out    machine-readable report for the next step (atomic write)
Usage:
    conda activate ram_pipeline
    python identify_rotated_frames.py \
        --flat-dir /data/mmc-user/outputs/3dssg_upright_v2 \
        --json-out /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json

    Exit 0 = every file identified confidently; 2 = at least one ambiguous.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

LOG = logging.getLogger("identify")

THUMB = (96, 96)          # aspect-destroying resize: same content -> same thumbnail
K_CANDIDATES = (-1, 1, 2, 0)   # cw, ccw, 180, none


def thumb(arr: np.ndarray) -> np.ndarray:
    """Grayscale, fixed-size, float thumbnail — comparison currency."""
    im = Image.fromarray(arr).convert("L").resize(THUMB, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recover (scan, rotation k) for hand-rotated 3DSSG colour frames.")
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--max-mae", type=float, default=12.0,
                    help="best match must be below this to count as identified")
    ap.add_argument("--min-margin", type=float, default=3.0,
                    help="runner-up must be at least this much worse (MAE units)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.json_out and Path(args.json_out).exists() and not args.overwrite:
        raise SystemExit(f"[refusing] {args.json_out} exists; pass --overwrite")

    data_root = Path(args.dssg_root) / args.data_subdir
    if not data_root.is_dir():
        raise SystemExit(f"[input error] {data_root} is not a directory")
    scans = sorted(p.name for p in data_root.iterdir() if (p / "sequence").is_dir())
    LOG.info("scans with a sequence/ dir: %d  %s", len(scans), scans)

    flat = sorted(p for p in Path(args.flat_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not flat:
        raise SystemExit(f"[input error] no image files directly in {args.flat_dir}")
    LOG.info("candidate rotated frames: %d", len(flat))

    results, ambiguous = [], 0
    for f in flat:
        m = re.search(r"(\d{6})", f.name)
        if not m:
            LOG.error("%s: no 6-digit frame number in name -- skipped", f.name)
            ambiguous += 1
            continue
        fno = int(m.group(1))

        im = Image.open(f)
        raw_size = im.size
        exif_ori = im.getexif().get(274, None)
        # compare using the pixels PIL hands to the pipeline (EXIF NOT applied),
        # which is exactly what tag_frames_vlm.py / ground_tags_dino.py will see
        hand = np.asarray(im.convert("RGB"))
        hand_t = thumb(hand)
        portrait = raw_size[1] > raw_size[0]
        if portrait and exif_ori in (None, 1):
            exif_verdict = "real pixel rotation"
        elif not portrait and exif_ori in (6, 8):
            exif_verdict = "**EXIF-ONLY -- pipeline sees sideways pixels**"
        elif not portrait:
            exif_verdict = "**NOT ROTATED**"
        else:
            exif_verdict = f"portrait but exif_ori={exif_ori} -- inspect"

        scored = []
        for scan in scans:
            src = data_root / scan / "sequence" / f"frame-{fno:06d}.color.jpg"
            if not src.exists():
                continue
            base = np.asarray(Image.open(src).convert("RGB"))
            for k in K_CANDIDATES:
                cand = np.rot90(base, k)
                if cand.shape[:2] != hand.shape[:2]:
                    continue          # wrong orientation for this k -- cannot be it
                scored.append((float(np.abs(thumb(cand) - hand_t).mean()), scan, k))
        scored.sort()

        if not scored:
            LOG.error("%s: no source frame-%06d.color.jpg of matching dimensions in any scan",
                      f.name, fno)
            results.append(dict(file=f.name, frame_no=fno, size=list(raw_size),
                                exif_orientation=exif_ori, exif_verdict=exif_verdict,
                                scan=None, k=None, mae=None, margin=None, ok=False))
            ambiguous += 1
            continue

        mae, scan, k = scored[0]
        margin = (scored[1][0] - mae) if len(scored) > 1 else float("inf")
        ok = (mae <= args.max_mae and margin >= args.min_margin
              and exif_verdict == "real pixel rotation")
        ambiguous += (not ok)
        results.append(dict(file=f.name, frame_no=fno, size=list(raw_size),
                            exif_orientation=exif_ori, exif_verdict=exif_verdict,
                            scan=scan, k=k, mae=round(mae, 2),
                            margin=(None if margin == float("inf") else round(margin, 2)),
                            runner_up=(None if len(scored) < 2
                                       else dict(scan=scored[1][1], k=scored[1][2],
                                                 mae=round(scored[1][0], 2))),
                            ok=bool(ok)))

    print(f"\n{'file':28s} {'size':12s} {'scan':40s} {'k':>3s} {'MAE':>7s} {'margin':>7s}  verdict")
    for r in results:
        print(f"{r['file']:28s} {str(r['size']):12s} {str(r['scan']):40s} "
              f"{str(r['k']):>3s} {str(r['mae']):>7s} {str(r['margin']):>7s}  "
              f"{'OK' if r['ok'] else '** CHECK: ' + r['exif_verdict'] + ' **'}")

    ks = {r["k"] for r in results if r["ok"]}
    scan_counts = {}
    for r in results:
        if r["ok"]:
            scan_counts[r["scan"]] = scan_counts.get(r["scan"], 0) + 1
    print(f"\nidentified: {sum(r['ok'] for r in results)}/{len(results)}")
    print(f"rotation k values seen: {sorted(ks)}"
          f"{'   <-- MIXED, must be handled per frame' if len(ks) > 1 else ''}")
    print(f"frames per scan: {scan_counts}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(dict(flat_dir=args.flat_dir, n=len(results),
                           n_ok=sum(r["ok"] for r in results),
                           k_values=sorted(ks), frames_per_scan=scan_counts,
                           frames=results), fh, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("report written: %s", args.json_out)

    return 0 if ambiguous == 0 else 2


if __name__ == "__main__":
    sys.exit(main())