"""
Stage: Recall@K evaluation, step 4 — Predicate R@K + mR@K + B4 vocabulary
ceiling, with a per-stratum breakdown (strata from strata.stratum_of: S1-S5
+ SAME_AS, see strata.py). Per-stratum mR@K is deliberately omitted. Also
adds item-weighted (mention-weighted) B4 ceiling reach, pooled and per
stratum, distinct from the class-weighted ceiling.

Purpose: for each GT predicate class, can ANY emitted VLM phrase rank this
         exact class <= K (not best-of-acceptable-set).
Denominator: canonical visible GT edge-predicate items (216), micro,
         abstention included.
READ-ONLY except --json-out.

USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python predicate_recall_at_k_per_stratum.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --json-out /data/mmc-user/outputs/predicate_rk_v3_stratum.json
"""

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gt"))
from strata import stratum_of, STRATA

LOG = logging.getLogger("predicate_recall_per_stratum")
TOP_KS = [1, 3, 5]

# The 22-predicate scored set minus front/behind = 20 scored predicates.
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
    """Return the canonical (subject, object, predicate) for one GT record,
    per the SYMMETRIC/INVERSE/directed canonicalisation regimes defined above.
    Returns None for predicates outside the scored set."""
    if pred not in VOCAB_SET:
        return None
    if pred in SYMMETRIC:
        return (min(s_id, o_id), max(s_id, o_id), pred)
    if pred in INVERSE:
        if s_id <= o_id:
            return (s_id, o_id, pred)
        return (o_id, s_id, INVERSE[pred])
    return (s_id, o_id, pred)


def load_classes_check(path):
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
    import glob
    pat = os.path.join(rel_dir, "{0}_{1:06d}__*.json".format(scan, frame_no))
    m = sorted(glob.glob(pat))
    if not m:
        return None
    if len(m) > 1:
        LOG.warning("%d files match %s_%06d — using first: %s", len(m), scan, frame_no, m)
    return m[0]


def build_population(frames_json, dssg_root, data_subdir, rel_dir, min_px, iou_thresh):
    """One record per canonical, visible GT edge-predicate item, annotated with
    whether its endpoints matched, what the VLM said about that pair, and its
    stratum."""
    dssg_root = Path(dssg_root)
    rels_raw = json.loads((dssg_root / "relationships.json").read_text())
    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}

    # labels, needed to derive each item's stratum
    objs_raw = json.loads((dssg_root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}

    rep = json.loads(Path(frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    population = []
    n_abstain = 0

    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)
        lab = labels_by_scan.get(scan, {})

        mpath = dssg_root / data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt_boxes = derive_gt_boxes(mpath, min_px)
        visible = set(gt_boxes)

        relf = find_relation_file(rel_dir, scan, no)
        pred_boxes, pred_tags, vlm_rels = [], [], []
        has_relation_file = relf is not None
        if has_relation_file:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]
            pred_tags = [o["tag"] for o in rd["objects"]]
            vlm_rels = rd.get("relations_readable", [])
        frame_abstained = has_relation_file and not vlm_rels
        if frame_abstained:
            n_abstain += 1

        # ---- Hungarian: predicted box index -> GT objectId ----
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
                if im[pi, gi] >= iou_thresh:
                    box_to_gt[pi] = gt_ids[gi]

        # ---- VLM relations, expressed over GT ids ----
        vlm_by_pair = defaultdict(list)
        for r in vlm_rels:
            try:
                si = int(str(r["subject"]).split(":", 1)[0])
                oi = int(str(r["object"]).split(":", 1)[0])
            except (ValueError, KeyError):
                LOG.warning("%s: unparseable relation %r", key, r)
                continue
            if si in box_to_gt and oi in box_to_gt:
                sg, og = box_to_gt[si], box_to_gt[oi]
                if sg != og:  # two boxes collapsing to one id -> self-loop, drop
                    vlm_by_pair[frozenset((sg, og))].append((sg, og, r["relation"]))

        # ---- canonical GT items for this frame ----
        seen = set()
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in visible or o_id not in visible:
                continue
            can = canonical_gt(s_id, o_id, pname)
            if can is None:
                continue
            if can in seen:
                continue
            seen.add(can)
            cs, co, cp = can

            both_matched = cs in box_to_gt.values() and co in box_to_gt.values()
            cands = vlm_by_pair.get(frozenset((cs, co)), [])

            resolved = []
            for (sg, og, phrase) in cands:
                if (sg, og) == (cs, co):
                    resolved.append({"phrase": phrase, "target": cp, "reversed": False})
                elif cp in SYMMETRIC:
                    resolved.append({"phrase": phrase, "target": cp, "reversed": False})
                elif cp in INVERSE:
                    resolved.append({"phrase": phrase, "target": INVERSE[cp],
                                     "reversed": True})
                else:
                    resolved.append({"phrase": phrase, "target": None,
                                     "reversed": True})

            has_scoreable = any(c["target"] is not None for c in resolved)

            if has_scoreable:
                miss_reason = "scored"
            elif not has_relation_file:
                miss_reason = "no_relation_file"
            elif frame_abstained:
                miss_reason = "abstained"
            elif not both_matched:
                miss_reason = "endpoints_not_matched"
            elif not resolved:
                miss_reason = "no_relation_for_pair"
            else:
                miss_reason = "direction_error"

            sl, ol = lab.get(cs, "?"), lab.get(co, "?")
            stratum = stratum_of(cp, sl, ol)

            population.append(dict(
                frame_key=key, subject=cs, object=co, gt_predicate=cp,
                subject_label=sl, object_label=ol, stratum=stratum,
                both_matched=both_matched,
                frame_abstained=frame_abstained,
                candidates=resolved,
                has_scoreable_candidate=has_scoreable,
                miss_reason=miss_reason,
            ))

    return population, n_abstain


def rank_of(model, util, phrase, target, vocab_emb):
    q = model.encode(phrase, convert_to_tensor=True)
    sims = util.cos_sim(q, vocab_emb)[0]
    order = sims.argsort(descending=True).tolist()
    return order.index(PREDICATE_VOCAB.index(target)) + 1


def score_encoder(model, population, distinct_phrases):
    from sentence_transformers import util
    vocab_emb = model.encode(PREDICATE_VOCAB, convert_to_tensor=True)

    n_total = len(population)
    hits = {k: 0 for k in TOP_KS}
    cond_hits = {k: 0 for k in TOP_KS}
    n_cond = 0
    per_class_tot = defaultdict(int)
    per_class_hit = {k: defaultdict(int) for k in TOP_KS}
    n_direction_errors = 0

    # per-stratum accumulators
    strat_n_total = {st: 0 for st in STRATA}
    strat_hits = {st: {k: 0 for k in TOP_KS} for st in STRATA}

    for item in population:
        gtp = item["gt_predicate"]
        st = item["stratum"]
        per_class_tot[gtp] += 1
        strat_n_total[st] += 1

        scoreable = [c for c in item["candidates"] if c["target"] is not None]
        if any(c["target"] is None for c in item["candidates"]):
            n_direction_errors += 1

        if not scoreable:
            continue

        n_cond += 1
        best = None
        for c in scoreable:
            r = rank_of(model, util, c["phrase"], c["target"], vocab_emb)
            if best is None or r < best:
                best = r

        for k in TOP_KS:
            if best <= k:
                hits[k] += 1
                cond_hits[k] += 1
                per_class_hit[k][gtp] += 1
                strat_hits[st][k] += 1

    classes_present = sorted(per_class_tot)
    mrk = {}
    for k in TOP_KS:
        per = [per_class_hit[k][c] / per_class_tot[c] for c in classes_present]
        mrk[k] = float(np.mean(per)) if per else 0.0

    # ---- B4: vocabulary ceiling, per EXACT predicate (UNCHANGED from v2) ----
    ceiling = {k: {} for k in TOP_KS}
    for cls in classes_present:
        best = None
        for ph in distinct_phrases:
            r = rank_of(model, util, ph, cls, vocab_emb)
            if best is None or r < best:
                best = r
        for k in TOP_KS:
            ceiling[k][cls] = best <= k

    # item-weighted (mention-weighted) ceiling, pooled and per stratum --
    # a different aggregation of the same per-class bools as `ceiling` above.
    item_ceiling_hits = {k: sum(1 for it in population
                                if ceiling[k].get(it["gt_predicate"], False))
                         for k in TOP_KS}
    item_weighted_reach = {k: (item_ceiling_hits[k] / n_total if n_total else 0.0)
                           for k in TOP_KS}

    strat_ceiling_hits = {st: {k: 0 for k in TOP_KS} for st in STRATA}
    for it in population:
        st = it["stratum"]
        for k in TOP_KS:
            if ceiling[k].get(it["gt_predicate"], False):
                strat_ceiling_hits[st][k] += 1

    # consistency gates: stratum sums must equal pooled totals
    s_total = sum(strat_n_total.values())
    if s_total != n_total:
        raise SystemExit("[consistency] stratum n_total sum {0} != pooled "
                         "n_total {1}".format(s_total, n_total))
    for k in TOP_KS:
        s_hits = sum(strat_hits[st][k] for st in STRATA)
        if s_hits != hits[k]:
            raise SystemExit("[consistency] stratum hits sum {0} != pooled "
                             "hits {1} at K={2}".format(s_hits, hits[k], k))
        s_ceil = sum(strat_ceiling_hits[st][k] for st in STRATA)
        if s_ceil != item_ceiling_hits[k]:
            raise SystemExit("[consistency] stratum item-ceiling sum {0} != "
                             "pooled item-ceiling {1} at K={2}".format(
                                 s_ceil, item_ceiling_hits[k], k))

    per_stratum = {}
    for st in STRATA:
        nt = strat_n_total[st]
        per_stratum[st] = dict(
            n_total=nt,
            hits=strat_hits[st],
            R_at_K={k: (strat_hits[st][k] / nt if nt else 0.0) for k in TOP_KS},
            item_weighted_ceiling_hits=strat_ceiling_hits[st],
            item_weighted_ceiling_reach={
                k: (strat_ceiling_hits[st][k] / nt if nt else 0.0) for k in TOP_KS})

    return {
        "n_total": n_total,
        "n_conditional": n_cond,
        "n_direction_errors": n_direction_errors,
        "R_at_K": {k: hits[k] / n_total if n_total else 0.0 for k in TOP_KS},
        "hits": hits,
        "conditional_R_at_K": {k: cond_hits[k] / n_cond if n_cond else 0.0 for k in TOP_KS},
        "mR_at_K": mrk,
        "classes_present": classes_present,
        "per_class_totals": dict(per_class_tot),
        "vocab_ceiling": {k: {c: bool(v) for c, v in ceiling[k].items()} for k in TOP_KS},
        "item_weighted_ceiling_reach_v3": item_weighted_reach,
        "per_stratum": per_stratum,
    }


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
        description="Predicate R@K + mR@K + per-stratum breakdown (v3), recall@K evaluation step 4.")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
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

    population, n_abstain = build_population(
        args.frames_json, args.dssg_root, args.data_subdir,
        args.rel_dir, args.min_px, args.iou_thresh)

    if len(population) != 216:
        LOG.warning("population %d != 216 — cross-check triplet_recall_at_k_baseline.py / "
                    "stratify_population.py", len(population))

    strat_pop_counts = defaultdict(int)
    for p in population:
        strat_pop_counts[p["stratum"]] += 1

    distinct_phrases = sorted({c["phrase"] for it in population for c in it["candidates"]})

    print("\n=== POPULATION ===")
    print("  canonical visible GT edge-predicate items (PRIMARY denominator): {0}".format(
        len(population)))
    print("  items with >=1 scoreable VLM candidate: {0}".format(
        sum(1 for i in population if i["has_scoreable_candidate"])))
    print("  frames where VLM abstained: {0}".format(n_abstain))
    print("  distinct VLM phrases reaching a matched pair: {0}".format(len(distinct_phrases)))
    print("\n=== STRATUM COUNTS (4th independent derivation) ===")
    for st in STRATA:
        print("  {0:<8} {1:>4}".format(st, strat_pop_counts.get(st, 0)))
    cls_tot = defaultdict(int)
    for i in population:
        cls_tot[i["gt_predicate"]] += 1
    print("\n  GT predicate class distribution:")
    for c, n in sorted(cls_tot.items(), key=lambda kv: -kv[1]):
        print("    {0:5d}  {1}".format(n, c))

    # ---- miss-reason breakdown (encoder-independent — computed once) ----
    reason_order = ["scored", "abstained", "endpoints_not_matched",
                    "no_relation_for_pair", "direction_error", "no_relation_file"]
    reason_counts = defaultdict(int)
    for it in population:
        reason_counts[it["miss_reason"]] += 1
    n_total_pop = len(population)

    print("\n  --- miss-reason breakdown (why each of the {0} items did or "
          "didn't get a shot) ---".format(n_total_pop))
    reason_label = {
        "scored": "has >=1 scoreable VLM candidate (goes to R@K)",
        "abstained": "frame's VLM output was empty ([])",
        "endpoints_not_matched": "one or both endpoints failed Hungarian match",
        "no_relation_for_pair": "endpoints matched, VLM proposed nothing for this pair",
        "direction_error": "VLM proposed a relation, but backwards (directed predicate)",
        "no_relation_file": "no predicted-relations file found for this frame",
    }
    for r in reason_order:
        n = reason_counts.get(r, 0)
        pct = 100.0 * n / n_total_pop if n_total_pop else 0.0
        print("    {0:5d}  ({1:5.1f}%)  {2:22s}  {3}".format(
            n, pct, r, reason_label[r]))
    n_abstain_items = reason_counts.get("abstained", 0)
    print("  --> abstention alone accounts for {0}/{1} items ({2:.1f}% of the "
          "primary denominator)".format(
              n_abstain_items, n_total_pop,
              100.0 * n_abstain_items / n_total_pop if n_total_pop else 0.0))

    # same miss-reason breakdown, per stratum
    reason_by_stratum = {st: defaultdict(int) for st in STRATA}
    for it in population:
        reason_by_stratum[it["stratum"]][it["miss_reason"]] += 1
    s_check = sum(sum(d.values()) for d in reason_by_stratum.values())
    if s_check != n_total_pop:
        raise SystemExit("[consistency] miss-reason-by-stratum sum {0} != "
                         "population {1}".format(s_check, n_total_pop))
    print("\n  --- miss-reason breakdown, PER STRATUM ---")
    for st in STRATA:
        row = reason_by_stratum[st]
        n_st = strat_pop_counts.get(st, 0)
        print("    {0:<8} n={1:<3} ".format(st, n_st) +
              "  ".join("{0}={1}".format(r, row.get(r, 0)) for r in reason_order
                       if row.get(r, 0) > 0))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers not importable: {0}".format(exc))

    results = {}
    for name, path in [("all-MiniLM-L6-v2", args.minilm),
                       ("all-mpnet-base-v2", args.mpnet),
                       ("clip-ViT-L-14", args.clip)]:
        LOG.info("Loading %s", name)
        try:
            model = SentenceTransformer(path)
        except Exception as exc:
            LOG.error("FAILED to load %s: %s", name, exc)
            results[name] = {"load_error": str(exc)}
            continue
        res = score_encoder(model, population, distinct_phrases)
        results[name] = res

        print("\n--- {0} ---".format(name))
        print("  PRIMARY (all {0} GT items):".format(res["n_total"]))
        for k in TOP_KS:
            print("    R@{0}:  {1:>3}/{2} = {3:6.1%}     mR@{0}: {4:6.1%}".format(
                k, res["hits"][k], res["n_total"], res["R_at_K"][k], res["mR_at_K"][k]))
        print("  PER STRATUM (mR@K omitted, 2026-08-19):")
        for st in STRATA:
            b = res["per_stratum"][st]
            row = "    {0:<8} n={1:<3}".format(st, b["n_total"])
            for k in TOP_KS:
                row += "  R@{0}={1:>3}/{2:<3}={3:6.1%}".format(
                    k, b["hits"][k], b["n_total"], b["R_at_K"][k])
            print(row)
        print("  CONDITIONAL (n={0}, both endpoints matched + VLM relation present):".format(
            res["n_conditional"]))
        for k in TOP_KS:
            print("    R@{0}:  {1:6.1%}".format(k, res["conditional_R_at_K"][k]))
        print("  direction errors (support relation emitted backwards): {0}".format(
            res["n_direction_errors"]))
        print("  B4 vocabulary ceiling — GT classes reachable by ANY emitted phrase "
              "(CLASS-weighted, unchanged from v2):")
        for k in TOP_KS:
            reach = sum(1 for v in res["vocab_ceiling"][k].values() if v)
            print("    @{0}: {1}/{2} classes".format(k, reach, len(res["classes_present"])))
            unreachable = [c for c, v in res["vocab_ceiling"][k].items() if not v]
            if k == 5 and unreachable:
                print("       unreachable even @5: {0}".format(unreachable))
        print("  B4 ceiling, ITEM-weighted (v3 NEW — NOT the same number as above):")
        for k in TOP_KS:
            print("    @{0}: {1:6.1%}  ({2}/{3} items)".format(
                k, res["item_weighted_ceiling_reach_v3"][k],
                round(res["item_weighted_ceiling_reach_v3"][k] * res["n_total"]),
                res["n_total"]))
        print("  B4 ceiling, ITEM-weighted, PER STRATUM:")
        for st in STRATA:
            b = res["per_stratum"][st]
            row = "    {0:<8} n={1:<3}".format(st, b["n_total"])
            for k in TOP_KS:
                row += "  @{0}={1:>3}/{2:<3}={3:6.1%}".format(
                    k, b["item_weighted_ceiling_hits"][k], b["n_total"],
                    b["item_weighted_ceiling_reach"][k])
            print(row)

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            v3_note=("Adds per_stratum (R@K, mR@K OMITTED) to each encoder's "
                     "block. Adds item_weighted_ceiling_reach_v3 (pooled) and "
                     "per_stratum.item_weighted_ceiling_reach -- a NEW figure, "
                     "distinct from the pre-existing CLASS-weighted "
                     "vocab_ceiling, which is UNCHANGED. Consistency-gated: "
                     "stratum sums == pooled, for hits AND item-ceiling-hits, "
                     "asserted at runtime. All pre-existing pooled fields MUST "
                     "match predicate_rk_results.json bit-for-bit."),
            min_px=args.min_px, iou_thresh=args.iou_thresh,
            predicate_vocab=PREDICATE_VOCAB,
            n_population=len(population), n_abstain_frames=n_abstain,
            stratum_population_counts=dict(strat_pop_counts),
            miss_reason_breakdown_by_stratum={
                st: dict(reason_by_stratum[st]) for st in STRATA},
            distinct_phrases=distinct_phrases,
            gt_class_distribution=dict(cls_tot),
            miss_reason_breakdown=dict(reason_counts),
            encoders=results))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()
