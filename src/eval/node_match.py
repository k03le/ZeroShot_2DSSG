"""
Script:  node_match.py
Stage:   Recall@K evaluation, step 2 — Hungarian node matching + detection ceiling.
Purpose: For each of the 17 frozen frames, derive GT boxes from the instance
         mask (box derivation: bounding box of each objectId's mask pixels)
         and Hungarian-match them against VLM-first predicted boxes at
         IoU >= 0.5. Emits the detection ceiling (share of GT nodes
         matched) as a byproduct of the same node-matching pass.

         CONTROL: must reproduce the known matching result — 58 of the 133
         visible ground-truth objects match a predicted box at IoU >= 0.5
         (43.6%). A different number means something upstream changed. If it
         doesn't, STOP — do not proceed to object_recall_at_k.py until understood.
         Per the project's standing rule, the control is not retuned to match
         output.

READ-ONLY except --json-out.

KEY FACTS RELIED ON (verified this session, do not re-assume):
    - rendered.instances.png is natively upright (540x960), id 0 = background.
    - Predicted bbox_2d and GT-derived boxes share the same [x1,y1,x2,y2]
      convention on the same frame orientation — VERIFIED 2026-08-17
      (pred x-range 0.8-538.1, y-range 69.1-957.4; mask shape (960,540)).

INPUT
    --frames-json   ingest_identify.json
    --dssg-root     holds objects.json, 3dssg_data/<scan>/sequence/
    --rel-dir       relations_3dssg_vlmfirst/json/ (predicted boxes + relations)
    --min-px        GT visibility threshold (default 200 -- matches the value used
                    for the reproducibility control above)
    --iou-thresh    Hungarian match rejection threshold (default 0.5)
OUTPUT
    console: per-frame matched/visible table + totals + control-check verdict
    --json-out: machine-readable match records (atomic write)
USAGE
    conda activate ram_pipeline
    python node_match.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --json-out /data/mmc-user/outputs/node_match_v1.json
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

LOG = logging.getLogger("node_match")


def derive_gt_boxes(mask_path, min_px):
    """Box derivation: for each objectId, the bounding box of its mask pixels.
    id 0 (background) and anything under min_px are dropped."""
    mask = np.asarray(Image.open(mask_path))
    ids, counts = np.unique(mask, return_counts=True)
    boxes = {}
    for oid, cnt in zip(ids, counts):
        if oid == 0 or cnt < min_px:
            continue
        ys, xs = np.where(mask == oid)
        boxes[int(oid)] = (float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max()), int(cnt))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_relation_file(rel_dir, scan, frame_no):
    """Locate the predicted-objects file for one frame. Warns on ambiguity
    rather than silently picking one — multiple backends may have been run."""
    pattern = os.path.join(rel_dir, "{0}_{1:06d}__*.json".format(scan, frame_no))
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        LOG.warning(
            "%d files match %s_%06d — using first, VERIFY this is the intended "
            "backend/run: %s", len(matches), scan, frame_no, matches)
    return matches[0]


def main():
    ap = argparse.ArgumentParser(
        description="Hungarian node matching + detection ceiling (recall@K evaluation, step 2).")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.json_out and Path(args.json_out).exists() and not args.overwrite:
        raise SystemExit("[refusing] {0} exists; pass --overwrite".format(args.json_out))

    try:
        rep = json.loads(Path(args.frames_json).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit("[input error] {0}: {1}".format(args.frames_json, e))
    frames = [f for f in rep["frames"] if f.get("ok")]
    LOG.info("frames to match: %d", len(frames))

    rows = []
    total_visible = 0
    total_matched = 0
    skipped_no_pred = []

    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)

        mpath = Path(args.dssg_root) / args.data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        if not mpath.exists():
            raise SystemExit("[input error] mask not found: {0}".format(mpath))
        gt_boxes = derive_gt_boxes(mpath, args.min_px)

        relf = find_relation_file(args.rel_dir, scan, no)
        if relf is None:
            LOG.warning("%s: no predicted-relations file found — 0 matches, "
                       "all GT nodes miss this frame", key)
            skipped_no_pred.append(key)
            pred_boxes = []
        else:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]

        gt_ids = list(gt_boxes.keys())
        n_gt = len(gt_ids)
        n_pred = len(pred_boxes)
        matched = 0

        if n_gt and n_pred:
            cost = np.ones((n_pred, n_gt))
            iou_mat = np.zeros((n_pred, n_gt))
            for pi, pb in enumerate(pred_boxes):
                for gi, gid in enumerate(gt_ids):
                    v = iou(pb, gt_boxes[gid])
                    iou_mat[pi, gi] = v
                    cost[pi, gi] = 1.0 - v
            row_ind, col_ind = linear_sum_assignment(cost)
            for pi, gi in zip(row_ind, col_ind):
                if iou_mat[pi, gi] >= args.iou_thresh:
                    matched += 1

        total_visible += n_gt
        total_matched += matched
        rows.append(dict(frame_key=key, scan=scan, frame_no=no,
                         n_gt_visible=n_gt, n_pred=n_pred, n_matched=matched,
                         relation_file=relf))

    print("\n{0:46s} {1:>7s} {2:>7s} {3:>8s}".format(
        "frame_key", "gt_vis", "n_pred", "matched"))
    for r in rows:
        print("{0:46s} {1:>7d} {2:>7d} {3:>8d}".format(
            r["frame_key"], r["n_gt_visible"], r["n_pred"], r["n_matched"]))

    ceiling = 100.0 * total_matched / total_visible if total_visible else 0.0
    print("\nTOTAL: {0}/{1} matched  =  {2:.1f}% detection ceiling".format(
        total_matched, total_visible, ceiling))
    control_pass = (total_matched, total_visible) == (58, 133)
    if args.iou_thresh != 0.5:
        print("CONTROL CHECK — not applicable at --iou-thresh {0}: the control is "
              "defined at 0.5 only.".format(args.iou_thresh))
        print("  (0.3 is the documented sensitivity variant and yields 68/133; "
              "control_pass=false there is expected, not a failure.)")
    else:
        print("CONTROL CHECK — expected 58/133 (43.6%): {0}".format(
            "PASS" if control_pass else
            "MISMATCH — DO NOT PROCEED, investigate before building further"))
    if skipped_no_pred:
        print("\n{0} frame(s) had no predicted-relations file: {1}".format(
            len(skipped_no_pred), skipped_no_pred))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out + ".tmp.{0}".format(os.getpid())
        with open(tmp, "w") as fh:
            json.dump(dict(min_px=args.min_px, iou_thresh=args.iou_thresh,
                           total_matched=total_matched, total_visible=total_visible,
                           detection_ceiling_pct=ceiling, control_pass=control_pass,
                           frames=rows), fh, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()