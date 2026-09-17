"""
Script:  per_frame_summary.py
Stage:   Supporting evidence for the recall@K evaluation — per-frame view of the same pipeline triplet_recall_at_k_baseline.py pools.
Purpose: For each of the 17 frames: detection rate, canonical GT edge count,
         how many reached scoring, and triplet hits@5/10/20 (one encoder,
         product score fn, same first-emitted/no-oracle rule as triplet_recall_at_k_baseline.py).

         CROSS-CHECK: summed per-frame hits/population MUST equal the pooled
         numbers already in triplet_rk_product.json. If they don't match,
         this script or triplet_recall_at_k_baseline.py has a bug — stop and compare before
         trusting either.

READ-ONLY except --json-out.

USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python per_frame_summary.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --encoder ~/hf_stage/all-mpnet-base-v2 \
        --encoder-name all-mpnet-base-v2 \
        --json-out /data/mmc-user/outputs/per_frame_summary.json
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

LOG = logging.getLogger("per_frame")
TOP_KS = [5, 10, 20]

PREDICATE_VOCAB = [
    "left", "right", "close by", "attached to", "standing on",
    "higher than", "lower than", "same as", "lying on", "hanging on",
    "supported by", "standing in", "build in", "leaning against",
    "connected to", "belonging to", "lying in", "part of", "cover",
    "hanging in",
]
VOCAB_SET = set(PREDICATE_VOCAB)
SYMMETRIC = {"close by", "same as"}
INVERSE = {"left": "right", "right": "left",
           "higher than": "lower than", "lower than": "higher than"}


def canonical_gt(s_id, o_id, pred):
    if pred not in VOCAB_SET:
        return None
    if pred in SYMMETRIC:
        return (min(s_id, o_id), max(s_id, o_id), pred)
    if pred in INVERSE:
        if s_id <= o_id:
            return (s_id, o_id, pred)
        return (o_id, s_id, INVERSE[pred])
    return (s_id, o_id, pred)


def load_classes(path):
    with open(path, "r", encoding="utf-8") as fh:
        classes = [ln.strip() for ln in fh if ln.strip()]
    if len(classes) != 160:
        raise SystemExit("[gate] classes.txt has {0} entries, expected 160".format(len(classes)))
    return classes


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


def strip_index(s):
    return str(s).split(":", 1)[-1].strip()


def triplet_rank_product(cos_s, cos_p, cos_o, i_s, i_p, i_o):
    """Same rule as triplet_recall_at_k_baseline.py: product with the negative-cosine sign correction,
    pessimistic ties. Kept in sync deliberately — if that script's rule
    changes, this one must change too."""
    raw = cos_s[:, None, None] * cos_p[None, :, None] * cos_o[None, None, :]
    neg = ((cos_s < 0).astype(np.int8)[:, None, None]
           + (cos_p < 0).astype(np.int8)[None, :, None]
           + (cos_o < 0).astype(np.int8)[None, None, :])
    score = np.where(neg == 2, -raw, raw)
    gt_score = score[i_s, i_p, i_o]
    return int((score > gt_score).sum()) + int((score == gt_score).sum())


def main():
    ap = argparse.ArgumentParser(description="Per-frame summary of the recall@K pipeline.")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--encoder", required=True, help="path to the encoder to use")
    ap.add_argument("--encoder-name", default="encoder")
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
    class_index = {c: i for i, c in enumerate(classes160)}
    pred_index = {p: i for i, p in enumerate(PREDICATE_VOCAB)}

    root = Path(args.dssg_root)
    rels_raw = json.loads((root / "relationships.json").read_text())
    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}
    objs_raw = json.loads((root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}
    rep = json.loads(Path(args.frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    from sentence_transformers import SentenceTransformer, util
    LOG.info("Loading %s", args.encoder_name)
    model = SentenceTransformer(args.encoder)
    node_emb = model.encode(classes160, convert_to_tensor=True)
    pred_emb = model.encode(PREDICATE_VOCAB, convert_to_tensor=True)
    cache = {}

    def cos_vec(phrase, emb, tag):
        ck = tag + "::" + phrase
        if ck in cache:
            return cache[ck]
        q = model.encode(phrase, convert_to_tensor=True)
        v = util.cos_sim(q, emb)[0].cpu().numpy().astype(np.float64)
        cache[ck] = v
        return v

    rows = []
    totals = dict(n_gt_visible=0, n_matched=0, n_canonical_edges=0,
                 n_scoreable=0, hits={k: 0 for k in TOP_KS})

    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)
        lab = labels_by_scan.get(scan, {})

        mpath = root / args.data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt_boxes = derive_gt_boxes(mpath, args.min_px)
        visible = set(gt_boxes)

        relf = find_relation_file(args.rel_dir, scan, no)
        pred_boxes, vlm_rels = [], []
        has_file = relf is not None
        if has_file:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]
            vlm_rels = rd.get("relations_readable", [])
        abstained = has_file and not vlm_rels

        box_to_gt = {}
        gt_ids = list(gt_boxes.keys())
        n_matched_frame = 0
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
                    n_matched_frame += 1

        vlm_by_pair = defaultdict(list)
        for emit_i, r in enumerate(vlm_rels):
            try:
                si = int(str(r["subject"]).split(":", 1)[0])
                oi = int(str(r["object"]).split(":", 1)[0])
            except (ValueError, KeyError):
                continue
            if si in box_to_gt and oi in box_to_gt:
                sg, og = box_to_gt[si], box_to_gt[oi]
                if sg != og:
                    vlm_by_pair[frozenset((sg, og))].append(
                        (emit_i, sg, og, strip_index(r["subject"]),
                         r["relation"], strip_index(r["object"])))

        n_canonical = 0
        n_scoreable = 0
        hits = {k: 0 for k in TOP_KS}
        seen = set()
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in visible or o_id not in visible:
                continue
            can = canonical_gt(s_id, o_id, pname)
            if can is None or can in seen:
                continue
            seen.add(can)
            n_canonical += 1
            cs, co, cp = can

            pair_cands = sorted(vlm_by_pair.get(frozenset((cs, co)), []),
                                key=lambda t: t[0])
            if not pair_cands:
                continue
            _e, sg, og, vsub, vrel, vobj = pair_cands[0]

            if (sg, og) == (cs, co):
                ts, tp, to = lab.get(cs, "?"), cp, lab.get(co, "?")
            elif cp in SYMMETRIC:
                ts, tp, to = lab.get(og, "?"), cp, lab.get(sg, "?")
            elif cp in INVERSE:
                ts, tp, to = lab.get(co, "?"), INVERSE[cp], lab.get(cs, "?")
            else:
                continue  # direction error, miss

            if ts not in class_index or to not in class_index:
                continue  # out of vocab, miss

            cs_v = cos_vec(vsub, node_emb, "N")
            cp_v = cos_vec(vrel, pred_emb, "P")
            co_v = cos_vec(vobj, node_emb, "N")
            rank = triplet_rank_product(cs_v, cp_v, co_v,
                                        class_index[ts], pred_index[tp],
                                        class_index[to])
            n_scoreable += 1
            for k in TOP_KS:
                if rank <= k:
                    hits[k] += 1

        row = dict(frame_key=key, scan=scan, frame_no=no,
                   n_gt_visible=len(gt_ids), n_matched=n_matched_frame,
                   detection_rate=n_matched_frame / len(gt_ids) if gt_ids else 0.0,
                   abstained=abstained,
                   n_canonical_edges=n_canonical, n_scoreable=n_scoreable,
                   hits=hits)
        rows.append(row)

        totals["n_gt_visible"] += row["n_gt_visible"]
        totals["n_matched"] += row["n_matched"]
        totals["n_canonical_edges"] += row["n_canonical_edges"]
        totals["n_scoreable"] += row["n_scoreable"]
        for k in TOP_KS:
            totals["hits"][k] += hits[k]

    print("\n=== PER-FRAME SUMMARY — encoder: {0} ===\n".format(args.encoder_name))
    print("{0:<46} {1:>5} {2:>5} {3:>6}  {4:>4} {5:>5}   {6}".format(
        "frame_key", "gt", "det", "rate%", "edg", "scor", "hits@5/10/20"))
    for r in rows:
        flag = " (abstained)" if r["abstained"] else ""
        print("{0:<46} {1:>5} {2:>5} {3:>6.1f}  {4:>4} {5:>5}   {6}/{7}/{8}{9}".format(
            r["frame_key"], r["n_gt_visible"], r["n_matched"],
            100 * r["detection_rate"], r["n_canonical_edges"], r["n_scoreable"],
            r["hits"][5], r["hits"][10], r["hits"][20], flag))

    print("\nTOTALS: gt={0} matched={1} ({2:.1f}%)  canonical_edges={3}  "
          "scoreable={4}  hits@5/10/20={5}/{6}/{7}".format(
              totals["n_gt_visible"], totals["n_matched"],
              100 * totals["n_matched"] / totals["n_gt_visible"],
              totals["n_canonical_edges"], totals["n_scoreable"],
              totals["hits"][5], totals["hits"][10], totals["hits"][20]))
    print("\nCROSS-CHECK against triplet_rk_product.json: canonical_edges should")
    print("  be 216, and hits@K should match that file's {0} entry exactly.".format(
        args.encoder_name))
    if totals["n_canonical_edges"] != 216:
        LOG.warning("canonical_edges = %d, not 216 — investigate before trusting this",
                   totals["n_canonical_edges"])
    if totals["n_matched"] != 58:
        LOG.warning("total matched = %d, not the 58 control — investigate",
                   totals["n_matched"])

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        d = os.path.dirname(os.path.abspath(args.json_out)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(encoder=args.encoder_name, rows=rows, totals=totals),
                      fh, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()