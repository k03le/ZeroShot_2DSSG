"""
Stage:   Recall@K evaluation, step 5 — Triplet R@5/10/20 + diagnostics, with
         a per-stratum breakdown (strata from strata.stratum_of, see
         strata.py: S1-S5 + SAME_AS). Pooled numbers are unchanged; per-
         stratum mR@K is omitted since most strata carry only 1-2 predicate
         classes, where it would mostly restate R@K. A consistency gate
         asserts stratum sums reproduce the pooled numbers exactly.
Scope:   *** 3DSSG SUBSET (4 scans / 17 frames / 216 GT items) ***
         NEVER report these numbers as "3DSSG". The subset label is printed in
         every header and stored in the JSON.

WHAT THIS MEASURES:
  PRIMARY — GT-CONDITIONED SEMANTIC RETRIEVAL (headline). Given the VLM's
  free-text triple, the full 160x20x160 = 512,000 ontology triplets are scored
  and the rank of the GT triplet in that single composite list gives hit@K.
  This is NOT "where does the VLM's prediction rank among the VLM's own
  predictions" — that is the STRICT metric below, reported alongside as a
  diagnostic.

  DIAGNOSTIC A — INSTANCE PAIR RECALL. Of GT edges, how many had BOTH endpoints
  correctly grounded (Hungarian, IoU>=0.5)? Separates grounding failure from
  relation failure. Encoder-independent.

  DIAGNOSTIC B — STRICT INSTANCE-LEVEL TRIPLET RECALL. Each VLM relation is
  mapped to its nearest ontology predicate over MATCHED GT instance ids, giving
  (gt_id_s, predicate, gt_id_o). A GT triplet is retrieved if an exact match
  appears in the frame's top-K predictions. Closer to the published 3DSSG
  convention. Ranking key is VLM EMISSION ORDER (the VLM emits no confidence
  scores). CAVEAT: the VLM emits ~4-5 relations/frame, so any K >= 20 contains
  every prediction and strict R@20 degenerates to plain unranked set recall;
  false positives are NOT penalised at large K.

CANDIDATE SELECTION: for each GT pair, only the FIRST-EMITTED VLM candidate is
scored — no selection among multiple candidates. If the first candidate is a
direction error, the item MISSES. Items whose pair had >1 candidate are
counted and reported.

ORIENTATION:
  orientation_equivalent — VLM named the endpoints the other way round for a
                           SYMMETRIC or INVERSE-PAIRED predicate. NOT an error;
                           targets are realigned to the VLM's ordering.
  direction_error        — VLM reversed a DIRECTED support/attachment relation,
                           for which no inverse exists. A genuine error; the
                           item misses and is counted separately.

SCORE FUNCTIONS (shifted_product is the primary one used throughout):
  product          raw product of the three cosines, with a sign correction
                   for the n==2-negatives case (the only case with a lying sign).
  shifted_product  (cos + SHIFT) per component, then product. No sign
                   discontinuity. NOT rank-preserving w.r.t. `product` — SHIFT
                   acts as a continuous knob from product-like (SHIFT=0) to
                   sum-like (large SHIFT), since log(x+c) ~ log c + x/c.
  sum              cos_s + cos_p + cos_o.
  mean             sum / 3 — identical ranking to `sum` (monotonic transform).
  TIES: pessimistic — the GT triplet takes the WORST rank in its tie block.

DENOMINATORS
  PRIMARY     = ALL canonical visible GT edge-predicate items (216), micro,
                abstained frames included.
  CONDITIONAL = items with a usable first-emitted candidate (secondary row).
  GT labels outside the 160-class vocabulary are automatic misses retained in
  the denominator.

READ-ONLY except --json-out.

USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python triplet_recall_at_k_per_stratum.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --score-fn shifted_product --shift 1.0 \
        --json-out /data/mmc-user/outputs/triplet_rk_v3_shifted_product.json
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gt"))
from strata import stratum_of, STRATA

LOG = logging.getLogger("triplet_recall_per_stratum")
TOP_KS = [5, 10, 20]
SUBSET_LABEL = "3DSSG subset (4 scans / 17 frames)"

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


def build_population(frames_json, dssg_root, data_subdir, rel_dir, min_px, iou_thresh):
    root = Path(dssg_root)
    rels_raw = json.loads((root / "relationships.json").read_text())
    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}
    objs_raw = json.loads((root / "objects.json").read_text())
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?") for o in s["objects"]}
                      for s in objs_raw["scans"]}

    rep = json.loads(Path(frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]

    population = []
    frame_predictions = {}
    n_abstain_frames = 0
    n_items_multi_candidate = 0

    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)
        lab = labels_by_scan.get(scan, {})

        mpath = root / data_subdir / scan / "sequence" / \
            "frame-{0:06d}.rendered.instances.png".format(no)
        gt_boxes = derive_gt_boxes(mpath, min_px)
        visible = set(gt_boxes)

        relf = find_relation_file(rel_dir, scan, no)
        pred_boxes, vlm_rels = [], []
        has_file = relf is not None
        if has_file:
            rd = json.loads(Path(relf).read_text())
            pred_boxes = [tuple(o["bbox_2d"]) for o in rd["objects"]]
            vlm_rels = rd.get("relations_readable", [])
        abstained = has_file and not vlm_rels
        if abstained:
            n_abstain_frames += 1

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

        # ---- VLM relations over GT ids, PRESERVING EMISSION ORDER ----
        ordered_preds = []
        vlm_by_pair = defaultdict(list)
        for emit_i, r in enumerate(vlm_rels):
            try:
                si = int(str(r["subject"]).split(":", 1)[0])
                oi = int(str(r["object"]).split(":", 1)[0])
            except (ValueError, KeyError):
                LOG.warning("%s: unparseable relation %r", key, r)
                continue
            if si not in box_to_gt or oi not in box_to_gt:
                continue
            sg, og = box_to_gt[si], box_to_gt[oi]
            if sg == og:
                continue
            rec = (emit_i, sg, og, strip_index(r["subject"]),
                   r["relation"], strip_index(r["object"]))
            ordered_preds.append(rec)
            vlm_by_pair[frozenset((sg, og))].append(rec)
        frame_predictions[key] = ordered_preds

        # ---- canonical GT items ----
        seen = set()
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in visible or o_id not in visible:
                continue
            can = canonical_gt(s_id, o_id, pname)
            if can is None or can in seen:
                continue
            seen.add(can)
            cs, co, cp = can

            matched_ids = set(box_to_gt.values())
            both_matched = (cs in matched_ids) and (co in matched_ids)

            pair_cands = sorted(vlm_by_pair.get(frozenset((cs, co)), []),
                                key=lambda t: t[0])
            if len(pair_cands) > 1:
                n_items_multi_candidate += 1

            chosen = None
            if pair_cands:
                _emit_i, sg, og, vsub, vrel, vobj = pair_cands[0]
                if (sg, og) == (cs, co):
                    chosen = dict(vlm_subject=vsub, vlm_relation=vrel, vlm_object=vobj,
                                  target_subject=lab.get(cs, "?"),
                                  target_predicate=cp,
                                  target_object=lab.get(co, "?"),
                                  orientation_equivalent=False, direction_error=False)
                elif cp in SYMMETRIC:
                    chosen = dict(vlm_subject=vsub, vlm_relation=vrel, vlm_object=vobj,
                                  target_subject=lab.get(og, "?"),
                                  target_predicate=cp,
                                  target_object=lab.get(sg, "?"),
                                  orientation_equivalent=True, direction_error=False)
                elif cp in INVERSE:
                    chosen = dict(vlm_subject=vsub, vlm_relation=vrel, vlm_object=vobj,
                                  target_subject=lab.get(co, "?"),
                                  target_predicate=INVERSE[cp],
                                  target_object=lab.get(cs, "?"),
                                  orientation_equivalent=True, direction_error=False)
                else:
                    chosen = dict(vlm_subject=vsub, vlm_relation=vrel, vlm_object=vobj,
                                  target_subject=None, target_predicate=None,
                                  target_object=None,
                                  orientation_equivalent=False, direction_error=True)

            stratum = stratum_of(cp, lab.get(cs, "?"), lab.get(co, "?"))

            population.append(dict(
                frame_key=key, subject_id=cs, object_id=co, gt_predicate=cp,
                gt_subject_label=lab.get(cs, "?"), gt_object_label=lab.get(co, "?"),
                stratum=stratum,
                both_endpoints_matched=both_matched,
                n_candidates_for_pair=len(pair_cands),
                chosen=chosen,
                frame_abstained=abstained))

    return population, frame_predictions, n_abstain_frames, n_items_multi_candidate


def composite_scores(cos_s, cos_p, cos_o, score_fn, shift):
    """Full 512k enumeration."""
    if score_fn == "product":
        raw = cos_s[:, None, None] * cos_p[None, :, None] * cos_o[None, None, :]
        neg = ((cos_s < 0).astype(np.int8)[:, None, None]
               + (cos_p < 0).astype(np.int8)[None, :, None]
               + (cos_o < 0).astype(np.int8)[None, None, :])
        return np.where(neg == 2, -raw, raw)
    if score_fn == "shifted_product":
        s, p, o = cos_s + shift, cos_p + shift, cos_o + shift
        return s[:, None, None] * p[None, :, None] * o[None, None, :]
    if score_fn == "sum":
        return cos_s[:, None, None] + cos_p[None, :, None] + cos_o[None, None, :]
    if score_fn == "mean":
        return (cos_s[:, None, None] + cos_p[None, :, None]
                + cos_o[None, None, :]) / 3.0
    raise ValueError("unknown score_fn {0!r}".format(score_fn))


def triplet_rank(cos_s, cos_p, cos_o, i_s, i_p, i_o, score_fn, shift):
    score = composite_scores(cos_s, cos_p, cos_o, score_fn, shift)
    gt_score = score[i_s, i_p, i_o]
    n_greater = int((score > gt_score).sum())
    n_equal = int((score == gt_score).sum())
    return n_greater + n_equal, float(gt_score), int(score.size)


def instance_pair_recall(population):
    """DIAGNOSTIC A — encoder-independent. Also computes the same recall
    per stratum."""
    def _one(pop_subset):
        n_items = len(pop_subset)
        n_grounded = sum(1 for p in pop_subset if p["both_endpoints_matched"])
        pairs_all, pairs_grounded = set(), set()
        for p in pop_subset:
            k = (p["frame_key"], p["subject_id"], p["object_id"])
            pairs_all.add(k)
            if p["both_endpoints_matched"]:
                pairs_grounded.add(k)
        return dict(
            n_gt_items=n_items,
            n_gt_items_both_endpoints_grounded=n_grounded,
            item_level_pair_recall=n_grounded / n_items if n_items else 0.0,
            n_distinct_gt_pairs=len(pairs_all),
            n_distinct_gt_pairs_grounded=len(pairs_grounded),
            pair_level_recall=len(pairs_grounded) / len(pairs_all) if pairs_all else 0.0)

    pooled = _one(population)

    strat_groups = defaultdict(list)
    for p in population:
        strat_groups[p["stratum"]].append(p)
    by_stratum = {st: _one(strat_groups.get(st, [])) for st in STRATA}

    # consistency gate: stratum sums must equal pooled totals
    s_items = sum(v["n_gt_items"] for v in by_stratum.values())
    s_grounded = sum(v["n_gt_items_both_endpoints_grounded"] for v in by_stratum.values())
    if s_items != pooled["n_gt_items"]:
        raise SystemExit(
            "[consistency] instance_pair_recall: stratum n_gt_items sum {0} "
            "!= pooled {1}".format(s_items, pooled["n_gt_items"]))
    if s_grounded != pooled["n_gt_items_both_endpoints_grounded"]:
        raise SystemExit(
            "[consistency] instance_pair_recall: stratum grounded sum {0} "
            "!= pooled {1}".format(s_grounded, pooled["n_gt_items_both_endpoints_grounded"]))

    pooled["by_stratum"] = by_stratum
    return pooled


def strict_triplet_recall(model, util, population, frame_predictions, pred_emb):
    """DIAGNOSTIC B — instance-level; VLM emission order is the ranking key.
    Not stratified."""
    cache = {}

    def nearest_pred(phrase):
        if phrase in cache:
            return cache[phrase]
        q = model.encode(phrase, convert_to_tensor=True)
        sims = util.cos_sim(q, pred_emb)[0].cpu().numpy()
        out = PREDICATE_VOCAB[int(sims.argmax())]
        cache[phrase] = out
        return out

    mapped = {}
    for fk, preds in frame_predictions.items():
        mapped[fk] = [(sg, nearest_pred(vrel), og)
                      for (_e, sg, og, _vs, vrel, _vo) in preds]

    gt_by_frame = defaultdict(list)
    for p in population:
        gt_by_frame[p["frame_key"]].append(
            (p["subject_id"], p["gt_predicate"], p["object_id"]))

    out = {}
    n_pred_total = sum(len(v) for v in mapped.values())
    for k in TOP_KS:
        hit, tot = 0, 0
        for fk, gts in gt_by_frame.items():
            topk = set(mapped.get(fk, [])[:k])
            for g in gts:
                tot += 1
                if g in topk:
                    hit += 1
        out[str(k)] = dict(hits=hit, total=tot, recall=hit / tot if tot else 0.0)
    out["n_mapped_predictions"] = n_pred_total
    out["mean_predictions_per_frame"] = (n_pred_total / len(mapped)) if mapped else 0.0
    return out


def score_encoder(model, population, classes160, class_index, pred_index,
                  score_fn, shift):
    from sentence_transformers import util
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

    n_total = len(population)
    hits = {k: 0 for k in TOP_KS}
    cond_hits = {k: 0 for k in TOP_KS}
    n_cond = 0
    per_class_tot = defaultdict(int)
    per_class_hit = {k: defaultdict(int) for k in TOP_KS}
    n_oov, n_dir_err, n_orient = 0, 0, 0
    detail = []

    # per-stratum totals/hits, same denominator convention as pooled
    strat_n_total = {st: 0 for st in STRATA}
    strat_hits = {st: {k: 0 for k in TOP_KS} for st in STRATA}

    for item in population:
        gtp = item["gt_predicate"]
        st = item["stratum"]
        per_class_tot[gtp] += 1
        strat_n_total[st] += 1
        c = item["chosen"]
        if c is None:
            continue
        if c["direction_error"]:
            n_dir_err += 1
            continue
        if c["orientation_equivalent"]:
            n_orient += 1

        ts, tp, to = c["target_subject"], c["target_predicate"], c["target_object"]
        if ts not in class_index or to not in class_index:
            n_oov += 1
            continue

        cs_v = cos_vec(c["vlm_subject"], node_emb, "N")
        cp_v = cos_vec(c["vlm_relation"], pred_emb, "P")
        co_v = cos_vec(c["vlm_object"], node_emb, "N")

        rank, gt_score, n_cand = triplet_rank(
            cs_v, cp_v, co_v, class_index[ts], pred_index[tp], class_index[to],
            score_fn, shift)

        r_s = int((cs_v > cs_v[class_index[ts]]).sum()) + 1
        r_p = int((cp_v > cp_v[pred_index[tp]]).sum()) + 1
        r_o = int((co_v > co_v[class_index[to]]).sum()) + 1

        n_cond += 1
        detail.append(dict(
            frame_key=item["frame_key"], gt_predicate=gtp, stratum=st,
            subject_id=item["subject_id"], object_id=item["object_id"],
            gt_triplet=[ts, tp, to],
            vlm_triple=[c["vlm_subject"], c["vlm_relation"], c["vlm_object"]],
            orientation_equivalent=c["orientation_equivalent"],
            triplet_rank=rank, gt_score=gt_score,
            component_ranks=dict(subject=r_s, predicate=r_p, object=r_o),
            product_of_component_ranks=r_s * r_p * r_o,
            n_candidates_for_pair=item["n_candidates_for_pair"],
            n_candidates_enumerated=n_cand))

        for k in TOP_KS:
            if rank <= k:
                hits[k] += 1
                cond_hits[k] += 1
                per_class_hit[k][gtp] += 1
                strat_hits[st][k] += 1

    classes_present = sorted(per_class_tot)
    mrk = {}
    for k in TOP_KS:
        per = [per_class_hit[k][cc] / per_class_tot[cc] for cc in classes_present]
        mrk[k] = float(np.mean(per)) if per else 0.0

    strict = strict_triplet_recall(model, util, population,
                                   score_encoder.frame_predictions, pred_emb)

    # per-stratum block; mR@K omitted (see module docstring)
    per_stratum = {}
    for st in STRATA:
        nt = strat_n_total[st]
        per_stratum[st] = dict(
            n_total=nt,
            hits=strat_hits[st],
            R_at_K={k: (strat_hits[st][k] / nt if nt else 0.0) for k in TOP_KS})

    # consistency gate: stratum sums must equal pooled totals
    s_total = sum(v["n_total"] for v in per_stratum.values())
    if s_total != n_total:
        raise SystemExit(
            "[consistency] score_encoder: stratum n_total sum {0} != pooled "
            "n_total {1}".format(s_total, n_total))
    for k in TOP_KS:
        s_hits = sum(v["hits"][k] for v in per_stratum.values())
        if s_hits != hits[k]:
            raise SystemExit(
                "[consistency] score_encoder: stratum hits sum {0} != pooled "
                "hits {1} at K={2}".format(s_hits, hits[k], k))

    return {
        "n_total": n_total, "n_conditional": n_cond,
        "n_out_of_vocab_misses": n_oov,
        "n_direction_errors": n_dir_err,
        "n_orientation_equivalent": n_orient,
        "hits": hits,
        "R_at_K": {k: hits[k] / n_total if n_total else 0.0 for k in TOP_KS},
        "conditional_R_at_K": {k: cond_hits[k] / n_cond if n_cond else 0.0
                               for k in TOP_KS},
        "mR_at_K": mrk,
        "classes_present": classes_present,
        "strict_instance_level": strict,
        "detail": detail,
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
        description="Triplet R@5/10/20 + per-stratum breakdown (v3) — {0}".format(SUBSET_LABEL))
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rel-dir", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--minilm", default="all-MiniLM-L6-v2")
    ap.add_argument("--mpnet", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--score-fn", default="product",
                    choices=["product", "shifted_product", "sum", "mean"])
    ap.add_argument("--shift", type=float, default=1.0,
                    help="vertical shift for shifted_product (DECLARED parameter)")
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

    if args.score_fn == "mean":
        LOG.warning("score-fn 'mean' produces IDENTICAL rankings to 'sum' "
                    "(monotonic transform). Do not report both as distinct results.")

    classes160 = load_classes(args.classes)
    class_index = {c: i for i, c in enumerate(classes160)}
    pred_index = {p: i for i, p in enumerate(PREDICATE_VOCAB)}

    population, frame_preds, n_abstain, n_multi = build_population(
        args.frames_json, args.dssg_root, args.data_subdir,
        args.rel_dir, args.min_px, args.iou_thresh)
    score_encoder.frame_predictions = frame_preds

    print("\n" + "=" * 74)
    print("SCOPE: {0}  [per-stratum breakdown]".format(SUBSET_LABEL))
    print("       Report as SUBSET results. Never as '3DSSG'.")
    print("=" * 74)
    print("\n=== POPULATION ===")
    print("  canonical visible GT items (PRIMARY denominator): {0}".format(len(population)))
    print("  candidate space: {0} x {1} x {0} = {2:,} triplets (FULL, no truncation)".format(
        len(classes160), len(PREDICATE_VOCAB),
        len(classes160) ** 2 * len(PREDICATE_VOCAB)))
    print("  score function: {0}{1}".format(
        args.score_fn,
        "  (shift={0})".format(args.shift) if args.score_fn == "shifted_product" else ""))
    print("  ties: pessimistic (GT takes worst rank in its tie block)")
    print("  candidate selection: FIRST-EMITTED (oracle removed)")
    print("  GT items whose pair had >1 VLM candidate: {0}".format(n_multi))
    print("  frames abstained: {0}".format(n_abstain))
    if len(population) != 216:
        LOG.warning("population %d != 216 — cross-check predicate_recall_at_k_baseline.py", len(population))

    strat_pop_counts = defaultdict(int)
    for p in population:
        strat_pop_counts[p["stratum"]] += 1
    print("\n=== STRATUM COUNTS (this run's population) ===")
    for st in STRATA:
        print("  {0:<8} {1:>4}".format(st, strat_pop_counts.get(st, 0)))

    ipr = instance_pair_recall(population)
    print("\n=== DIAGNOSTIC A — INSTANCE PAIR RECALL (encoder-independent) ===")
    print("  POOLED: {0}/{1} = {2:.1%}  (distinct pairs: {3}/{4} = {5:.1%})".format(
        ipr["n_gt_items_both_endpoints_grounded"], ipr["n_gt_items"],
        ipr["item_level_pair_recall"], ipr["n_distinct_gt_pairs_grounded"],
        ipr["n_distinct_gt_pairs"], ipr["pair_level_recall"]))
    print("  PER STRATUM:")
    for st in STRATA:
        b = ipr["by_stratum"][st]
        print("    {0:<8} {1:>3}/{2:<3} = {3:6.1%}   pairs {4:>3}/{5:<3} = {6:6.1%}".format(
            st, b["n_gt_items_both_endpoints_grounded"], b["n_gt_items"],
            b["item_level_pair_recall"], b["n_distinct_gt_pairs_grounded"],
            b["n_distinct_gt_pairs"], b["pair_level_recall"]))

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

        res = score_encoder(model, population, classes160, class_index,
                            pred_index, args.score_fn, args.shift)
        results[name] = res
        strict = res["strict_instance_level"]

        print("\n--- {0} ---".format(name))
        print("  PRIMARY — GT-conditioned semantic retrieval (all {0} GT items):".format(
            res["n_total"]))
        for k in TOP_KS:
            star = "   <== HEADLINE" if k == 10 else ""
            print("    R@{0:<3} {1:>3}/{2} = {3:6.2%}    mR@{0:<3} {4:6.2%}{5}".format(
                k, res["hits"][k], res["n_total"], res["R_at_K"][k],
                res["mR_at_K"][k], star))
        print("  PER STRATUM (mR@K omitted):")
        for st in STRATA:
            b = res["per_stratum"][st]
            row = "    {0:<8} n={1:<3}".format(st, b["n_total"])
            for k in TOP_KS:
                row += "  R@{0}={1:>3}/{2:<3}={3:6.2%}".format(
                    k, b["hits"][k], b["n_total"], b["R_at_K"][k])
            print(row)
        print("  CONDITIONAL (n={0}, POOLED ONLY):".format(res["n_conditional"]))
        for k in TOP_KS:
            print("    R@{0:<3} {1:6.2%}".format(k, res["conditional_R_at_K"][k]))
        print("  orientation-equivalent (NOT errors): {0}   direction errors: {1}"
              "   OOV misses: {2}".format(
                  res["n_orientation_equivalent"], res["n_direction_errors"],
                  res["n_out_of_vocab_misses"]))

        print("  DIAGNOSTIC B — strict instance-level triplet recall (POOLED ONLY)")
        print("    (ranking key: VLM emission order; {0:.1f} predictions/frame — "
              "K>=20 is set recall)".format(strict["mean_predictions_per_frame"]))
        for k in TOP_KS:
            s = strict[str(k)]
            print("    strict R@{0:<3} {1}/{2} = {3:6.2%}".format(
                k, s["hits"], s["total"], s["recall"]))

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            scope=SUBSET_LABEL,
            v3_note=("Adds per_stratum (R@K only, mR@K omitted) to "
                     "instance_pair_recall and to each encoder's block. "
                     "Consistency-gated: stratum sums == pooled."),
            min_px=args.min_px, iou_thresh=args.iou_thresh,
            score_fn=args.score_fn, shift=args.shift,
            tie_policy="pessimistic",
            candidate_selection="first-emitted (no oracle)",
            strict_ranking_key="VLM emission order",
            n_candidates=len(classes160) ** 2 * len(PREDICATE_VOCAB),
            n_population=len(population), n_abstain_frames=n_abstain,
            n_items_multi_candidate=n_multi,
            stratum_population_counts=dict(strat_pop_counts),
            instance_pair_recall=ipr,
            predicate_vocab=PREDICATE_VOCAB,
            encoders=results))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()
