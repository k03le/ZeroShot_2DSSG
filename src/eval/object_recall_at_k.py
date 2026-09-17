"""
Script:  object_recall_at_k.py
Stage:   Recall@K evaluation, step 3 — Object R@1/3/5.
Purpose: For each of the 133 visible GT nodes (17 frames), determine hit@K
         against the 160-class vocabulary, for all three candidate encoders.
         Denominator = ALL 133 GT nodes (locked decision), micro-aggregated
         across frames (locked decision).

         A GT node scores an automatic MISS (contributes to denominator, not
         numerator) if EITHER:
           (a) it was not matched to any predicted box (Hungarian, IoU>=0.5), or
           (b) its raw label is outside the 160-class vocabulary (4/133 cases:
               poster x2, office chair, stereo equipment — DECLARED CHOICE,
               not silently dropped; see console summary).

         READ-ONLY except --json-out.

INPUT
    --frames-json   ingest_identify.json
    --dssg-root     holds objects.json, 3dssg_data/<scan>/sequence/
    --rel-dir       relations_3dssg_vlmfirst/json/
    --classes       official classes.txt (160 lines, gate-enforced)
    --minilm/--mpnet/--clip   encoder paths or cached names
    --min-px        GT visibility threshold (default 200)
    --iou-thresh    Hungarian match threshold (default 0.5)
OUTPUT
    console: per-encoder R@1/3/5 + stray-label summary
    --json-out: machine-readable results (atomic write)
USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python object_recall_at_k.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --json-out ~/object_rk_results.json
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

LOG = logging.getLogger("object_recall")
TOP_KS = [1, 3, 5]


def load_classes(path):
    with open(path, "r", encoding="utf-8") as fh:
        classes = [ln.strip() for ln in fh if ln.strip()]
    if len(classes) != 160:
        LOG.error("classes.txt has %d entries, expected 160 (flag 16). Refusing.",
                  len(classes))
        raise SystemExit(2)
    return classes


def derive_gt(mask_path, min_px, labels_for_scan):
    mask = np.asarray(Image.open(mask_path))
    ids, counts = np.unique(mask, return_counts=True)
    out = {}
    for oid, cnt in zip(ids, counts):
        if oid == 0 or cnt < min_px:
            continue
        ys, xs = np.where(mask == oid)
        out[int(oid)] = {
            "box": (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            "label": labels_for_scan.get(int(oid), "MISSING"),
        }
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_relation_file(rel_dir, scan, frame_no):
    pattern = os.path.join(rel_dir, "{0}_{1:06d}__*.json".format(scan, frame_no))
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        LOG.warning("%d files match %s_%06d — using first: %s",
                    len(matches), scan, frame_no, matches)
    return matches[0]


def collect_matched_pairs(frames_json, dssg_root, data_subdir, rel_dir, min_px, iou_thresh):
    """Returns: list of dicts, one per GT node, with match info.
    This is EVERY visible GT node (133), matched or not, in-vocab or not —
    the full population the R@K denominator is drawn from."""
    dssg_root = Path(dssg_root)
    objs_raw = json.loads((dssg_root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}

    rep = json.loads(Path(frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    all_gt = []
    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)

        mpath = dssg_root / data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt = derive_gt(mpath, min_px, labels_by_scan.get(scan, {}))

        relf = find_relation_file(rel_dir, scan, no)
        if relf is None:
            pred_boxes, pred_tags = [], []
        else:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]
            pred_tags = [o["tag"] for o in rd["objects"]]

        gt_ids = list(gt.keys())
        matched_tag = {gid: None for gid in gt_ids}

        if gt_ids and pred_boxes:
            n_gt, n_pred = len(gt_ids), len(pred_boxes)
            cost = np.ones((n_pred, n_gt))
            iou_mat = np.zeros((n_pred, n_gt))
            for pi, pb in enumerate(pred_boxes):
                for gi, gid in enumerate(gt_ids):
                    v = iou(pb, gt[gid]["box"])
                    iou_mat[pi, gi] = v
                    cost[pi, gi] = 1.0 - v
            row_ind, col_ind = linear_sum_assignment(cost)
            for pi, gi in zip(row_ind, col_ind):
                if iou_mat[pi, gi] >= iou_thresh:
                    matched_tag[gt_ids[gi]] = pred_tags[pi]

        for gid in gt_ids:
            all_gt.append(dict(frame_key=key, scan=scan, frame_no=no,
                               gt_id=gid, gt_label=gt[gid]["label"],
                               matched_tag=matched_tag[gid]))
    return all_gt


def score_encoder(model, all_gt, classes160, classes_set):
    from sentence_transformers import util

    vocab_emb = model.encode(classes160, convert_to_tensor=True)
    hits = {k: 0 for k in TOP_KS}
    n_total = len(all_gt)
    n_scoreable = 0
    stray = []

    for item in all_gt:
        if item["gt_label"] not in classes_set:
            stray.append(item)
            continue  # automatic miss, denominator already includes it via n_total
        if item["matched_tag"] is None:
            continue  # unmatched GT node, automatic miss

        n_scoreable += 1
        q_emb = model.encode(item["matched_tag"], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, vocab_emb)[0]
        order = sims.argsort(descending=True).tolist()
        rank = order.index(classes160.index(item["gt_label"])) + 1
        for k in TOP_KS:
            if rank <= k:
                hits[k] += 1

    return {
        "n_total": n_total,
        "n_scoreable_matched_invocab": n_scoreable,
        "n_stray_out_of_vocab": len(stray),
        "n_unmatched_or_stray_miss": n_total - n_scoreable,
        "hits": hits,
        "recall_at_k": {k: hits[k] / n_total for k in TOP_KS},
    }


def atomic_write_json(path, payload):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = None, None
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description="Object R@1/3/5, three encoders (recall@K evaluation, step 3).")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--minilm", default="all-MiniLM-L6-v2")
    ap.add_argument("--mpnet", required=True)
    ap.add_argument("--clip", required=True)
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

    classes160 = load_classes(args.classes)
    classes_set = set(classes160)

    all_gt = collect_matched_pairs(args.frames_json, args.dssg_root, args.data_subdir,
                                   args.rel_dir, args.min_px, args.iou_thresh)
    LOG.info("total GT node population: %d (control: expect 133)", len(all_gt))
    if len(all_gt) != 133:
        LOG.warning("Population is %d, not 133 — cross-check against node_match.py "
                   "before trusting anything downstream.", len(all_gt))

    stray_items = [g for g in all_gt if g["gt_label"] not in classes_set]
    print("\nStray labels outside 160-vocab ({0} instances) — DECLARED as automatic miss:"
          .format(len(stray_items)))
    for s in stray_items:
        print("  {0:30s} frame={1}".format(s["gt_label"], s["frame_key"]))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers not importable: {0}".format(exc))

    encoders = [("all-MiniLM-L6-v2", args.minilm),
               ("all-mpnet-base-v2", args.mpnet),
               ("clip-ViT-L-14", args.clip)]

    results = {}
    for name, path in encoders:
        LOG.info("Loading %s from %s", name, path)
        try:
            model = SentenceTransformer(path)
        except Exception as exc:
            LOG.error("FAILED to load %s: %s", name, exc)
            results[name] = {"load_error": str(exc)}
            continue
        res = score_encoder(model, all_gt, classes160, classes_set)
        results[name] = res
        print("\n--- {0} ---".format(name))
        print("  population (denominator): {0}".format(res["n_total"]))
        print("  scoreable (matched + in-vocab): {0}".format(res["n_scoreable_matched_invocab"]))
        print("  automatic misses (unmatched or out-of-vocab): {0}".format(
            res["n_unmatched_or_stray_miss"]))
        for k in TOP_KS:
            print("  R@{0}: {1}/{2} = {3:.1%}".format(
                k, res["hits"][k], res["n_total"], res["recall_at_k"][k]))

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            n_total=len(all_gt), min_px=args.min_px, iou_thresh=args.iou_thresh,
            stray_labels=[s["gt_label"] for s in stray_items],
            encoders=results))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()