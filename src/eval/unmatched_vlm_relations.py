"""
Script:  unmatched_vlm_relations.py
Stage:   Supporting evidence for the recall@K evaluation — qualitative precision evidence (GT-incompleteness).
Purpose: List every VLM relation whose BOTH endpoints matched a GT node
         (Hungarian, IoU >= 0.5) but for which GT records NO edge on that pair.

         These are the candidates for the §4 note on GT incompleteness: the
         VLM asserted something about two correctly-localised objects and the
         annotation simply has nothing to say. They are NOT scoreable as
         precision — absence from GT is not evidence of falsehood (this is the
         standard justification in the SGG literature for using recall rather
         than mAP). The output is evidence for a QUALITATIVE paragraph, and
         must not be reported as a precision figure.

         Reports two tiers:
           tier A — no GT edge on the pair AT ALL (any predicate, incl. the
                    attribute/comparative families excluded from the scored vocabulary)
           tier B — no GT edge on the pair within the 20 SCORED predicates
                    (tier A is a subset of tier B)

READ-ONLY except --json-out.

USAGE
    conda activate ram_pipeline
    python unmatched_vlm_relations.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --json-out /data/mmc-user/outputs/unmatched_vlm_relations.json
"""

import argparse
import glob
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

LOG = logging.getLogger("unmatched_vlm")

PREDICATE_VOCAB = {
    "left", "right", "close by", "attached to", "standing on",
    "higher than", "lower than", "same as", "lying on", "hanging on",
    "supported by", "standing in", "build in", "leaning against",
    "connected to", "belonging to", "lying in", "part of", "cover",
    "hanging in",
}


def derive_gt_boxes(mask_path, min_px):
    mask = np.asarray(Image.open(mask_path))
    ids, counts = np.unique(mask, return_counts=True)
    out = {}
    for oid, cnt in zip(ids, counts):
        if oid == 0 or cnt < min_px:
            continue
        ys, xs = np.where(mask == oid)
        out[int(oid)] = (float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max()))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def find_relation_file(rel_dir, scan, frame_no):
    pat = os.path.join(rel_dir, "{0}_{1:06d}__*.json".format(scan, frame_no))
    m = sorted(glob.glob(pat))
    if not m:
        return None
    if len(m) > 1:
        LOG.warning("%d files match %s_%06d — using first", len(m), scan, frame_no)
    return m[0]


def atomic_write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(
        description="VLM relations on matched pairs with no GT edge.")
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

    if args.json_out and os.path.exists(args.json_out) and not args.overwrite:
        raise SystemExit("[refusing] {0} exists; pass --overwrite".format(args.json_out))

    root = Path(args.dssg_root)
    rels_raw = json.loads((root / "relationships.json").read_text())
    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}
    objs_raw = json.loads((root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}

    rep = json.loads(Path(args.frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    tier_a, tier_b = [], []
    n_vlm_total = 0
    n_both_matched = 0

    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)

        mpath = root / args.data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt_boxes = derive_gt_boxes(mpath, args.min_px)
        visible = set(gt_boxes)

        relf = find_relation_file(args.rel_dir, scan, no)
        if relf is None:
            continue
        rd = json.loads(Path(relf).read_text())
        pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]
        vlm_rels = rd.get("relations_readable", [])
        n_vlm_total += len(vlm_rels)

        box_to_gt = {}
        gt_ids = list(gt_boxes.keys())
        if gt_ids and pred_boxes:
            cost = np.ones((len(pred_boxes), len(gt_ids)))
            im = np.zeros_like(cost)
            for pi, pb in enumerate(pred_boxes):
                for gi, gid in enumerate(gt_ids):
                    v = iou(pb, gt_boxes[gid])
                    im[pi, gi] = v
                    cost[pi, gi] = 1.0 - v
            ri, ci = linear_sum_assignment(cost)
            for pi, gi in zip(ri, ci):
                if im[pi, gi] >= args.iou_thresh:
                    box_to_gt[pi] = gt_ids[gi]

        # GT edges on this frame, indexed by unordered visible pair
        gt_any = defaultdict(list)
        gt_scored = defaultdict(list)
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in visible or o_id not in visible:
                continue
            pair = frozenset((s_id, o_id))
            gt_any[pair].append(pname)
            if pname in PREDICATE_VOCAB:
                gt_scored[pair].append(pname)

        lab = labels_by_scan.get(scan, {})
        for r in vlm_rels:
            try:
                si = int(str(r["subject"]).split(":", 1)[0])
                oi = int(str(r["object"]).split(":", 1)[0])
            except (ValueError, KeyError):
                continue
            if si not in box_to_gt or oi not in box_to_gt:
                continue
            sg, og = box_to_gt[si], box_to_gt[oi]
            if sg == og:
                continue
            n_both_matched += 1
            pair = frozenset((sg, og))
            rec = dict(frame_key=key,
                       vlm_subject=r["subject"], vlm_relation=r["relation"],
                       vlm_object=r["object"],
                       gt_subject_id=sg, gt_subject_label=lab.get(sg, "?"),
                       gt_object_id=og, gt_object_label=lab.get(og, "?"),
                       gt_predicates_any=gt_any.get(pair, []),
                       gt_predicates_scored=gt_scored.get(pair, []))
            if not gt_any.get(pair):
                tier_a.append(rec)
            if not gt_scored.get(pair):
                tier_b.append(rec)

    print("\n=== VLM relations on matched pairs with NO GT edge ===")
    print("  total VLM relations emitted:            {0}".format(n_vlm_total))
    print("  with BOTH endpoints matched to GT:      {0}".format(n_both_matched))
    print("  tier A - no GT edge at all:             {0}".format(len(tier_a)))
    print("  tier B - no GT edge in scored 20-vocab: {0}".format(len(tier_b)))

    print("\n--- TIER A (no GT edge of any kind on this pair) ---")
    for r in tier_a:
        print("  [{0}]".format(r["frame_key"]))
        print("      VLM: {0}  --{1}-->  {2}".format(
            r["vlm_subject"], r["vlm_relation"], r["vlm_object"]))
        print("      GT nodes: {0}(id {1})  /  {2}(id {3})   GT edges: NONE".format(
            r["gt_subject_label"], r["gt_subject_id"],
            r["gt_object_label"], r["gt_object_id"]))

    only_b = [r for r in tier_b if r["gt_predicates_any"]]
    print("\n--- TIER B only (GT has edges, but none in the scored vocab) ---")
    for r in only_b:
        print("  [{0}]".format(r["frame_key"]))
        print("      VLM: {0}  --{1}-->  {2}".format(
            r["vlm_subject"], r["vlm_relation"], r["vlm_object"]))
        print("      GT has (unscored): {0}".format(", ".join(r["gt_predicates_any"])))

    print("\nNOTE: absence from GT is NOT evidence of falsehood. These are")
    print("      qualitative candidates only — never report as precision.")

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            min_px=args.min_px, iou_thresh=args.iou_thresh,
            n_vlm_total=n_vlm_total, n_both_matched=n_both_matched,
            tier_a=tier_a, tier_b=tier_b))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()