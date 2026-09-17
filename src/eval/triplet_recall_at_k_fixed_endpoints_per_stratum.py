"""
Stage:   E2 -- fixed-object triplet ranking. Reuses the population machinery
         (build_population, instance_pair_recall, strict_triplet_recall,
         cos_vec/predicate-rank formula) from triplet_recall_at_k_per_stratum.py.
Scope:   *** 3DSSG SUBSET (4 scans / 17 frames / 216 GT items) ***

WHAT THIS MEASURES: subject/object identity for each GT pair is already known
from Hungarian box matching (IoU >= 0.5) upstream, so re-ranking it against
the 160-class vocabulary via a text encoder would just re-litigate a decision
already made geometrically. This script instead hardcodes
component_ranks.subject/object to 1 for every row (no embedding call touches
subject/object text at all) and scores only the predicate: `triplet_rank`
here equals the predicate's marginal rank (cosine similarity vs.
PREDICATE_VOCAB, optimistic ties). n_candidates_enumerated is still reported
as 160x20x160 = 512,000 to document the nominal candidate-space size, though
the rank itself is computed only over the 20-entry predicate vocabulary.

CONSEQUENCE: predicate rank is always in [1, 20], so R@20 == R@100 == R@200
== R@500 == R@1000 exactly for every encoder — an exact plateau, not a bug.

USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python triplet_recall_at_k_fixed_endpoints_per_stratum.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --json-out /data/mmc-user/outputs/triplet_rk_v4_fixedobj.json

READ-ONLY except --json-out.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gt"))
import triplet_recall_at_k_per_stratum as _v3mod
from strata import STRATA
from triplet_recall_at_k_per_stratum import (
    build_population,
    load_classes,
    instance_pair_recall,
    strict_triplet_recall,
    atomic_write_json,
    PREDICATE_VOCAB,
    SUBSET_LABEL,
)

LOG = logging.getLogger("triplet_recall_fixed_endpoints")

TOP_KS = [5, 10, 20, 100, 200, 500, 1000]
# strict_triplet_recall() reads its own module's global TOP_KS; override it
# (without editing that module on disk) so Diagnostic B uses this script's
# K-schedule.
_v3mod.TOP_KS = TOP_KS

DEFAULT_V3_BASELINE = "/data/mmc-user/outputs/triplet_rk_v3_shifted_product.json"


def score_encoder_fixedobj(model, population, class_index, pred_index, n_candidates):
    """E2 scoring. Subject/object component ranks are hardcoded to 1 — no
    model.encode() call below ever touches subject or object text."""
    from sentence_transformers import util

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
        # same OOV/abstention gate as v3, kept so n_cond is comparable
        if ts not in class_index or to not in class_index:
            n_oov += 1
            continue

        # ---- predicate rank: byte-identical mechanism to v3 ----
        cp_v = cos_vec(c["vlm_relation"], pred_emb, "P")
        r_p = int((cp_v > cp_v[pred_index[tp]]).sum()) + 1

        # subject/object rank: fixed by construction, no embedding call
        r_s = 1
        r_o = 1

        product_of_component_ranks = r_s * r_p * r_o
        if product_of_component_ranks != r_p:
            raise SystemExit(
                "[consistency] product_of_component_ranks {0} != "
                "component_ranks['predicate'] {1} -- E2 invariant "
                "violated (subject={2}, object={3})".format(
                    product_of_component_ranks, r_p, r_s, r_o))

        rank = product_of_component_ranks

        n_cond += 1
        detail.append(dict(
            frame_key=item["frame_key"], gt_predicate=gtp, stratum=st,
            subject_id=item["subject_id"], object_id=item["object_id"],
            gt_triplet=[ts, tp, to],
            vlm_triple=[c["vlm_subject"], c["vlm_relation"], c["vlm_object"]],
            orientation_equivalent=c["orientation_equivalent"],
            triplet_rank=rank,
            component_ranks=dict(subject=r_s, predicate=r_p, object=r_o),
            product_of_component_ranks=product_of_component_ranks,
            n_candidates_for_pair=item["n_candidates_for_pair"],
            n_candidates_enumerated=n_candidates))

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

    strict = strict_triplet_recall(
        model, util, population, score_encoder_fixedobj.frame_predictions, pred_emb)

    per_stratum = {}
    for st in STRATA:
        nt = strat_n_total[st]
        per_stratum[st] = dict(
            n_total=nt,
            hits=strat_hits[st],
            R_at_K={k: (strat_hits[st][k] / nt if nt else 0.0) for k in TOP_KS})

    # Consistency gate: stratum sums MUST reproduce pooled exactly (same
    # pattern as v3).
    s_total = sum(v["n_total"] for v in per_stratum.values())
    if s_total != n_total:
        raise SystemExit(
            "[consistency] score_encoder_fixedobj: stratum n_total sum {0} "
            "!= pooled n_total {1}".format(s_total, n_total))
    for k in TOP_KS:
        s_hits = sum(v["hits"][k] for v in per_stratum.values())
        if s_hits != hits[k]:
            raise SystemExit(
                "[consistency] score_encoder_fixedobj: stratum hits sum {0} "
                "!= pooled hits {1} at K={2}".format(s_hits, hits[k], k))

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


def main():
    ap = argparse.ArgumentParser(
        description="E2 -- fixed-object triplet ranking: subject/object "
                     "component ranks hardcoded to 1 via node-match identity; "
                     "predicate rank is the only real signal. {0}".format(
                         SUBSET_LABEL))
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
    ap.add_argument("--v3-baseline-json", default=DEFAULT_V3_BASELINE,
                    help="v3 output to diff n_candidates_enumerated against "
                         "(DoD requirement); pass '' to skip the check")
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
    n_candidates = len(classes160) ** 2 * len(PREDICATE_VOCAB)

    if args.v3_baseline_json:
        if os.path.exists(args.v3_baseline_json):
            v3_payload = json.loads(Path(args.v3_baseline_json).read_text())
            v3_n_candidates = v3_payload.get("n_candidates")
            if v3_n_candidates != n_candidates:
                raise SystemExit(
                    "[consistency] n_candidates_enumerated {0} != v3 baseline "
                    "{1} value {2} ({3}) -- E2 must not silently change the "
                    "candidate-space size".format(
                        n_candidates, args.v3_baseline_json, v3_n_candidates,
                        args.v3_baseline_json))
            LOG.info("n_candidates_enumerated %s confirmed == v3 baseline (%s)",
                     n_candidates, args.v3_baseline_json)
        else:
            LOG.warning("v3 baseline JSON not found at %s -- skipping "
                        "n_candidates_enumerated diff check", args.v3_baseline_json)

    population, frame_preds, n_abstain, n_multi = build_population(
        args.frames_json, args.dssg_root, args.data_subdir,
        args.rel_dir, args.min_px, args.iou_thresh)
    score_encoder_fixedobj.frame_predictions = frame_preds

    print("\n" + "=" * 74)
    print("SCOPE: {0}  [v4 -- E2 fixed-object triplet ranking]".format(SUBSET_LABEL))
    print("       Report as SUBSET results. Never as '3DSSG'.")
    print("=" * 74)
    print("\n=== POPULATION ===")
    print("  canonical visible GT items (PRIMARY denominator): {0}".format(len(population)))
    print("  candidate space: {0} x {1} x {0} = {2:,} triplets (SIZE unchanged; "
          "subject/object no longer scored within it)".format(
              len(classes160), len(PREDICATE_VOCAB), n_candidates))
    print("  object_scoring_rule: fixed_rank_1_via_node_match")
    print("  score_fn: predicate_rank_only (component_ranks.subject/object "
          "hardcoded to 1 -- no embedding call for either)")
    print("  ties (predicate rank): optimistic (best rank in tie block, "
          "strict-greater-count + 1)")
    print("  candidate selection: FIRST-EMITTED (oracle removed) -- unchanged from v3")
    print("  GT items whose pair had >1 VLM candidate: {0}".format(n_multi))
    print("  frames abstained: {0}".format(n_abstain))
    if len(population) != 216:
        LOG.warning("population %d != 216 -- cross-check predicate_recall_at_k_baseline.py", len(population))

    strat_pop_counts = defaultdict(int)
    for p in population:
        strat_pop_counts[p["stratum"]] += 1
    print("\n=== STRATUM COUNTS (this run's population) ===")
    for st in STRATA:
        print("  {0:<8} {1:>4}".format(st, strat_pop_counts.get(st, 0)))

    ipr = instance_pair_recall(population)
    print("\n=== DIAGNOSTIC A -- INSTANCE PAIR RECALL (encoder-independent, "
          "unchanged mechanism, reused verbatim) ===")
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

        res = score_encoder_fixedobj(model, population, class_index, pred_index,
                                     n_candidates)
        results[name] = res
        strict = res["strict_instance_level"]

        print("\n--- {0} ---".format(name))
        print("  PRIMARY -- predicate-only ranking (all {0} GT items):".format(
            res["n_total"]))
        for k in TOP_KS:
            print("    R@{0:<4} {1:>3}/{2} = {3:6.2%}    mR@{0:<4} {4:6.2%}".format(
                k, res["hits"][k], res["n_total"], res["R_at_K"][k],
                res["mR_at_K"][k]))
        print("  PER STRATUM:")
        for st in STRATA:
            b = res["per_stratum"][st]
            row = "    {0:<8} n={1:<3}".format(st, b["n_total"])
            for k in TOP_KS:
                row += "  R@{0}={1:>3}/{2:<3}={3:6.2%}".format(
                    k, b["hits"][k], b["n_total"], b["R_at_K"][k])
            print(row)
        print("  CONDITIONAL (n={0}, POOLED ONLY):".format(res["n_conditional"]))
        for k in TOP_KS:
            print("    R@{0:<4} {1:6.2%}".format(k, res["conditional_R_at_K"][k]))
        print("  orientation-equivalent (NOT errors): {0}   direction errors: {1}"
              "   OOV misses: {2}".format(
                  res["n_orientation_equivalent"], res["n_direction_errors"],
                  res["n_out_of_vocab_misses"]))

        print("  DIAGNOSTIC B -- strict instance-level triplet recall (POOLED ONLY, "
              "unchanged mechanism, reused verbatim)")
        print("    (ranking key: VLM emission order; {0:.1f} predictions/frame)".format(
            strict["mean_predictions_per_frame"]))
        for k in TOP_KS:
            s = strict[str(k)]
            print("    strict R@{0:<4} {1}/{2} = {3:6.2%}".format(
                k, s["hits"], s["total"], s["recall"]))

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            scope=SUBSET_LABEL,
            v4_note=(
                "E2 -- fixed-object triplet ranking. component_ranks.subject "
                "and component_ranks.object are hardcoded to 1 for every row "
                "(identity already fixed by Hungarian node-matching at "
                "IoU>=0.5; no embedding call touches subject/object text). "
                "triplet_rank == product_of_component_ranks == the "
                "predicate's marginal rank (asserted at runtime). Predicate "
                "rank is always in [1, 20], so R@20 == R@100 == R@200 == "
                "R@500 == R@1000 exactly for every encoder -- an exact "
                "plateau, not a bug. n_candidates_enumerated is still "
                "160x20x160 = 512,000, documenting the nominal candidate-"
                "space size, not the space the rank is computed within."),
            min_px=args.min_px, iou_thresh=args.iou_thresh,
            score_fn="predicate_rank_only",
            shift=None,
            tie_policy="optimistic (best rank in tie block: strict-greater-count + 1)",
            candidate_selection="first-emitted (no oracle)",
            strict_ranking_key="VLM emission order",
            object_scoring_rule="fixed_rank_1_via_node_match",
            n_candidates=n_candidates,
            n_population=len(population), n_abstain_frames=n_abstain,
            n_items_multi_candidate=n_multi,
            stratum_population_counts=dict(strat_pop_counts),
            instance_pair_recall=ipr,
            predicate_vocab=PREDICATE_VOCAB,
            encoders=results))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()
