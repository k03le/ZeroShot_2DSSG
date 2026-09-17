"""
Script:  pull_inverse_evidence.py
Stage:   Pre-registration evidence for the orientation-handling fix.
Purpose: Answer three questions FROM DATA, before any map is hand-written:

  A. For each GT predicate p, when (s,o,p) exists, what predicates appear on
     the REVERSED pair (o,s)? This derives SYMMETRIC and INVERSE empirically
     instead of asserting them. Fixes the defect that our current maps were
     never verified (the assumed count of inverse-paired predicates was 5;
     our map has only 2 pairs).

  B. Exactly which distinct free-text predicate phrases does the VLM emit,
     with counts? This is the domain the new natural-language inverse map
     must cover exhaustively.

  C. How many of the 216 GT items currently DIE as direction errors -- i.e.
     a VLM relation exists on the correct grounded pair, but in reversed
     orientation with a GT predicate that has no inverse? This quantifies the
     expected size of the fix BEFORE running it, so the effect can be
     pre-registered rather than discovered.

READ-ONLY. Writes nothing.

USAGE
    conda activate ram_pipeline
    python pull_inverse_evidence.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.dssg_root)
    rels_raw = json.loads((root / "relationships.json").read_text())

    # ---------------- A: empirical reverse-pair structure ----------------
    print("=" * 78)
    print("A. EMPIRICAL REVERSE-PAIR STRUCTURE (all 1335 scans, full corpus)")
    print("   For each predicate p on (s,o): what appears on (o,s)?")
    print("=" * 78)

    reverse_of = defaultdict(Counter)
    total_of = Counter()
    for scan in rels_raw["scans"]:
        pairs = defaultdict(set)
        for e in scan["relationships"]:
            pairs[(int(e[0]), int(e[1]))].add(str(e[3]))
        for (s, o), preds in pairs.items():
            back = pairs.get((o, s), set())
            for p in preds:
                total_of[p] += 1
                if not back:
                    reverse_of[p]["<NO REVERSE EDGE>"] += 1
                for q in back:
                    reverse_of[p][q] += 1

    print("\n{0:<18} {1:>8}   {2}".format("predicate", "total", "what appears on the reversed pair (top 4)"))
    for p in PREDICATE_VOCAB:
        if total_of[p] == 0:
            print("{0:<18} {1:>8}   (absent from corpus)".format(p, 0))
            continue
        top = reverse_of[p].most_common(4)
        desc = ", ".join("{0}={1}".format(q, n) for q, n in top)
        print("{0:<18} {1:>8}   {2}".format(p, total_of[p], desc))

    print("\n  READ THIS AS:")
    print("   p reversed -> p          => SYMMETRIC (should be in SYMMETRIC set)")
    print("   p reversed -> q          => INVERSE PAIR (should be in INVERSE map)")
    print("   p reversed -> <NO REVERSE EDGE> dominant => genuinely DIRECTED")

    # ---------------- B: VLM phrase inventory ----------------
    print("\n" + "=" * 78)
    print("B. DISTINCT VLM PREDICATE PHRASES (the domain the NL inverse map must cover)")
    print("=" * 78)
    phrases = Counter()
    for fp in sorted(glob.glob(os.path.join(args.rel_dir, "*.json"))):
        d = json.loads(Path(fp).read_text())
        for r in d.get("relations_readable", []):
            phrases[r["relation"]] += 1
    print("\n{0:>6}  {1}".format("count", "phrase"))
    for ph, n in phrases.most_common():
        print("{0:>6}  {1!r}".format(n, ph))
    print("\n  distinct phrases: {0}   total mentions: {1}".format(
        len(phrases), sum(phrases.values())))

    # ---------------- C: how many items would the fix touch? ----------------
    print("\n" + "=" * 78)
    print("C. ITEMS CURRENTLY LOST TO DIRECTION ERRORS (the fix's target population)")
    print("=" * 78)

    objs_raw = json.loads((root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}
    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}
    rep = json.loads(Path(args.frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    lost = []
    n_pop = 0
    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)
        lab = labels_by_scan.get(scan, {})

        mpath = root / args.data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt_boxes = derive_gt_boxes(mpath, args.min_px)
        visible = set(gt_boxes)

        pat = os.path.join(args.rel_dir, "{0}_{1:06d}__*.json".format(scan, no))
        m = sorted(glob.glob(pat))
        vlm_rels, pred_boxes = [], []
        if m:
            rd = json.loads(Path(m[0]).read_text())
            vlm_rels = rd.get("relations_readable", [])
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]

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

        by_pair = defaultdict(list)
        for r in vlm_rels:
            try:
                si = int(str(r["subject"]).split(":", 1)[0])
                oi = int(str(r["object"]).split(":", 1)[0])
            except (ValueError, KeyError):
                continue
            if si in box_to_gt and oi in box_to_gt:
                sg, og = box_to_gt[si], box_to_gt[oi]
                if sg != og:
                    by_pair[frozenset((sg, og))].append((sg, og, r["relation"]))

        seen = set()
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in visible or o_id not in visible:
                continue
            can = canonical_gt(s_id, o_id, pname)
            if can is None or can in seen:
                continue
            seen.add(can)
            n_pop += 1
            cs, co, cp = can
            cands = by_pair.get(frozenset((cs, co)), [])
            if not cands:
                continue
            sg, og, phrase = cands[0]
            if (sg, og) != (cs, co) and cp not in SYMMETRIC and cp not in INVERSE:
                lost.append(dict(frame=key, gt="{0} {1} {2}".format(
                    lab.get(cs, "?"), cp, lab.get(co, "?")),
                    vlm_phrase=phrase,
                    vlm_order="{0}->{1}".format(lab.get(sg, "?"), lab.get(og, "?"))))

    print("\n  population checked: {0} (expect 216)".format(n_pop))
    print("  items lost to direction errors under the FIRST-EMITTED rule: {0}".format(len(lost)))
    for x in lost:
        print("    [{0}]".format(x["frame"]))
        print("       GT : {0}".format(x["gt"]))
        print("       VLM: {0!r} on {1}  (reversed, no inverse defined)".format(
            x["vlm_phrase"], x["vlm_order"]))
    print("\n  NOTE: this counts only the FIRST-EMITTED candidate, matching")
    print("  triplet_recall_at_k_baseline.py's no-oracle rule. The ceiling's loss is larger because")
    print("  it silently skips the reversed test for ALL directed predicates.")


if __name__ == "__main__":
    main()