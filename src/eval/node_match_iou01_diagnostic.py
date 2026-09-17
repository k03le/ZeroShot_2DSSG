"""
Script:  node_match_iou01_diagnostic.py
Stage:   Recall@K evaluation, step 2 diagnostic -- per-object breakdown of
         why GT objects go unmatched at IoU >= 0.1.
Purpose: node_match.py builds a full predicted-box x GT-box IoU matrix per
         frame and runs Hungarian one-to-one matching, but only ever
         persists frame-level aggregate counts (n_gt_visible/n_pred/
         n_matched) to its --json-out -- the per-object outcome is computed
         and then discarded. This script mirrors node_match.py's
         box-derivation (derive_gt_boxes) and Hungarian cost-matrix logic
         EXACTLY, fixed at --iou-thresh 0.1, and persists the per-object
         outcome node_match.py throws away: for every GT object NOT matched
         at IoU >= 0.1, classify why not --
           zero_overlap        max IoU across every predicted box in the
                                frame is exactly 0.
           below_thresh_nonzero 0 < max IoU < 0.1 -- some overlap exists but
                                would not clear the threshold even with no
                                competing GT object.
           lost_to_competition max IoU >= 0.1 against some predicted box,
                                but the Hungarian assignment gave that box
                                (or this GT's only viable slot) to a
                                competing GT object instead.

         Written to source the 75.9% (pigeonhole cap, 113 boxes / 133
         objects) vs 60.9% (IoU>=0.1 ceiling) gap discussion in the report:
         of the 52 GT objects unmatched at IoU>=0.1, 34 lost a real (>=0.1)
         candidate to a competing GT object, 15 never had a candidate
         precise enough to clear 0.1 even absent competition, and 3 had
         zero overlap with anything in the frame.

         CONTROL: must reproduce node_match.py's IoU>=0.1 totals as a
         cross-check before the per-object breakdown is trusted -- 81 of
         133 visible GT objects matched (60.9%), of which 3 unmatched
         objects have zero overlap with every predicted box in their frame.
         A different number means either the underlying predicted-box/mask
         inputs changed since this was written, or the two scripts have
         drifted out of sync -- STOP and investigate; per the project's
         standing rule, this is not retuned to match output.

READ-ONLY except --json-out.

KEY FACTS RELIED ON:
    - Box derivation, IoU, and the predicted-relation-file lookup are
      copied verbatim from node_match.py (same derive_gt_boxes, same iou,
      same find_relation_file) so the two scripts cannot silently diverge
      on how a box is built or matched -- only on which outcome fields get
      kept.

INPUT
    --frames-json   ingest_identify.json (default: the 3dssg_upright_v2 copy
                    node_match.py's own USAGE block uses)
    --dssg-root     holds objects.json, 3dssg_data/<scan>/sequence/
    --rel-dir       relations_3dssg_vlmfirst/json/ (predicted boxes + relations)
    --min-px        GT visibility threshold (default 200 -- matches
                    node_match.py's default / the control gate)
    --iou-thresh    fixed conceptually at 0.1 for this diagnostic; exposed
                    as a flag only so the control check can be re-run at
                    other thresholds without editing the file, default 0.1
OUTPUT
    console: totals, bucket counts, control-check verdict
    --json-out: machine-readable per-object breakdown (atomic write)
USAGE
    /data/mmc-user/envs/miniconda3/envs/ram_pipeline/bin/python \
        node_match_iou01_diagnostic.py \
        --json-out /data/mmc-user/outputs/node_match_v1_iou01_diagnostic.json
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

LOG = logging.getLogger("node_match_iou01_diagnostic")


def derive_gt_boxes(mask_path, min_px):
    """Copied verbatim from node_match.py: bounding box of each objectId's
    mask pixels. id 0 (background) and anything under min_px are dropped."""
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
    """Copied verbatim from node_match.py."""
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
    """Copied verbatim from node_match.py."""
    pattern = os.path.join(rel_dir, "{0}_{1:06d}__*.json".format(scan, frame_no))
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        LOG.warning(
            "%d files match %s_%06d -- using first, VERIFY this is the intended "
            "backend/run: %s", len(matches), scan, frame_no, matches)
    return matches[0]


def main():
    ap = argparse.ArgumentParser(
        description="Per-object breakdown of why GT objects go unmatched at "
                     "IoU >= 0.1 (diagnostic sibling of node_match.py).")
    ap.add_argument("--frames-json",
                     default="/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir",
                     default="/data/mmc-user/outputs/relations_3dssg_vlmfirst/json")
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--iou-thresh", type=float, default=0.1)
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

    total_visible = 0
    total_matched = 0
    zero_overlap = []
    below_thresh_nonzero = []
    lost_to_competition = []
    matched_list = []

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
            LOG.warning("%s: no predicted-relations file found", key)
            pred_boxes = []
        else:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]

        gt_ids = list(gt_boxes.keys())
        n_gt, n_pred = len(gt_ids), len(pred_boxes)
        total_visible += n_gt

        if n_gt == 0:
            continue

        if n_pred == 0:
            for gid in gt_ids:
                zero_overlap.append(dict(frame_key=key, gt_id=gid))
            continue

        iou_mat = np.zeros((n_pred, n_gt))
        for pi, pb in enumerate(pred_boxes):
            for gi, gid in enumerate(gt_ids):
                iou_mat[pi, gi] = iou(pb, gt_boxes[gid])

        cost = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_gi_to_pi = {gi: pi for pi, gi in zip(row_ind, col_ind)}

        for gi, gid in enumerate(gt_ids):
            col = iou_mat[:, gi]
            max_iou = float(col.max())
            assigned_pi = assigned_gi_to_pi.get(gi)
            assigned_iou = float(iou_mat[assigned_pi, gi]) if assigned_pi is not None else 0.0

            if assigned_iou >= args.iou_thresh:
                total_matched += 1
                matched_list.append(dict(frame_key=key, gt_id=gid, assigned_iou=assigned_iou))
                continue

            if max_iou == 0.0:
                zero_overlap.append(dict(frame_key=key, gt_id=gid))
            elif max_iou < args.iou_thresh:
                below_thresh_nonzero.append(dict(frame_key=key, gt_id=gid, max_iou=max_iou))
            else:
                best_pi = int(np.argmax(col))
                lost_to_competition.append(dict(
                    frame_key=key, gt_id=gid, max_iou=max_iou,
                    competing_pred_idx=best_pi))

    n_unmatched = total_visible - total_matched
    ceiling = 100.0 * total_matched / total_visible if total_visible else 0.0

    print("total_visible:", total_visible)
    print("total_matched (IoU>={0}): {1}".format(args.iou_thresh, total_matched))
    print("ceiling_pct: {0:.1f}%".format(ceiling))
    print("n_unmatched:", n_unmatched)
    print()
    print("zero_overlap:", len(zero_overlap))
    print("below_thresh_nonzero:", len(below_thresh_nonzero))
    print("lost_to_competition:", len(lost_to_competition))
    bucket_sum = len(zero_overlap) + len(below_thresh_nonzero) + len(lost_to_competition)
    print("check sum == n_unmatched:", bucket_sum == n_unmatched)

    control_pass = (total_matched, total_visible, len(zero_overlap)) == (81, 133, 3)
    if abs(args.iou_thresh - 0.1) < 1e-9:
        print("\nCONTROL CHECK -- expected 81/133 matched (60.9%), 3 zero-overlap: {0}".format(
            "PASS" if control_pass else
            "MISMATCH -- DO NOT TRUST THE BREAKDOWN, investigate before citing"))
    else:
        print("\nCONTROL CHECK -- not applicable at --iou-thresh {0}: the control is "
              "defined at 0.1 only.".format(args.iou_thresh))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out + ".tmp.{0}".format(os.getpid())
        with open(tmp, "w") as fh:
            json.dump(dict(
                min_px=args.min_px, iou_thresh=args.iou_thresh,
                total_matched=total_matched, total_visible=total_visible,
                detection_ceiling_pct=ceiling, control_pass=control_pass,
                n_zero_overlap=len(zero_overlap),
                n_below_thresh_nonzero=len(below_thresh_nonzero),
                n_lost_to_competition=len(lost_to_competition),
                zero_overlap=zero_overlap,
                below_thresh_nonzero=below_thresh_nonzero,
                lost_to_competition=lost_to_competition,
                matched=matched_list,
            ), fh, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()
