"""
Script:  predicate_recall_14frame_conditional.py
Stage:   Recall@K evaluation, step 4 diagnostic — the 14-responding-frame
         conditional predicate recall.
Purpose: The evaluation protocol (D21(6)) locks a secondary reporting row
         restricted to the 14 of 17 frames where the VLM did not abstain,
         as a check on whether the headline 216-item denominator is being
         inflated by frames that produced no answer at all. That row was
         locked as protocol but never actually computed or saved anywhere
         in this repository -- this script fills that gap.

         Reuses build_population(), rank_of() and PREDICATE_VOCAB verbatim
         from predicate_recall_at_k_baseline.py (imported, not reimplemented)
         so the match/candidate/miss-reason logic cannot silently diverge
         from the primary 216-item figure already reported in the chapter.
         The only new step is filtering the population to items whose
         frame_key belongs to a non-abstained frame before scoring, and
         reporting against the resulting 162-item denominator instead of
         216. MiniLM only, matching this chapter's primary-encoder
         reporting convention; mpnet/CLIP are not computed here.

         CONTROL (checked at runtime, in order, before anything is
         trusted or written):
           1. Exactly 3 of 17 frames abstained (triples_indexed empty),
              14 did not -- checked two independent ways: once directly
              from the relation files' triples_indexed field, and once
              via build_population()'s own frame_abstained flag. Both
              must agree and both must give (3, 14).
           2. Filtering the 216-item population to the 14 non-abstained
              frames must yield exactly 162 items (216 - 54, where 54 is
              the already-reported abstained-item count). A different
              count means the frame_key join is silently dropping or
              duplicating items -- STOP.
           3. All 19 items already scored under the primary 216-item
              MiniLM run must survive the filter unchanged (an abstained
              frame contributes zero scored items by construction, so
              restricting to responding frames can only ever remove
              already-missed items, never a scored one). Hit counts at
              K=1/3/5 over the filtered population are therefore expected
              to equal the primary run's hit counts (5/8/9) exactly --
              only the denominator changes, 216 to 162. This script
              asserts that equality before writing output; a mismatch
              means the reuse of build_population/rank_of has diverged
              from the primary script, and it refuses to write.

READ-ONLY except --json-out.

INPUT
    --frames-json / --dssg-root / --rel-dir / --classes as in
    predicate_recall_at_k_baseline.py (same defaults)
OUTPUT
    console: control-check results, then the 162-denominator R@1/3/5 vs.
             the primary 216-denominator figure, side by side
    --json-out: machine-readable result (atomic write)
USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python predicate_recall_14frame_conditional.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --rel-dir /data/mmc-user/outputs/relations_3dssg_vlmfirst/json \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --json-out /data/mmc-user/outputs/predicate_rk_v3_14frame_conditional.json
"""

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger("predicate_14frame_conditional")
TOP_KS = [1, 3, 5]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predicate_recall_at_k_baseline import (  # noqa: E402
    build_population, rank_of, PREDICATE_VOCAB,
)


def find_abstained_frames_from_relfiles(rel_dir):
    """Independent check #1: derive abstained/answered frames directly from
    the relation files' triples_indexed field, not via build_population."""
    abstained, answered = [], []
    for f in sorted(glob.glob(os.path.join(rel_dir, "*.json"))):
        d = json.loads(Path(f).read_text())
        (abstained if not d.get("triples_indexed") else answered).append(d["frame_id"])
    return sorted(abstained), sorted(answered)


def atomic_write_json(path, payload):
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
        description="14-responding-frame conditional predicate R@K (MiniLM), "
                     "a locked-but-never-run secondary row (D21(6)).")
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

    # ---- control check 1: abstained/answered frame counts, two ways ----
    abstained_a, answered_a = find_abstained_frames_from_relfiles(args.rel_dir)
    print("[check 1a] from relation files: {0} abstained, {1} answered".format(
        len(abstained_a), len(answered_a)))
    if len(abstained_a) != 3 or len(answered_a) != 14:
        raise SystemExit(
            "[STOP] expected 3 abstained / 14 answered from relation files, "
            "got {0}/{1}".format(len(abstained_a), len(answered_a)))

    population, n_abstain = build_population(
        args.frames_json, args.dssg_root, args.data_subdir, args.rel_dir,
        args.min_px, args.iou_thresh)
    abstained_b = sorted({it["frame_key"] for it in population if it["frame_abstained"]})
    all_frames_b = sorted({it["frame_key"] for it in population})
    answered_b = sorted(set(all_frames_b) - set(abstained_b))
    print("[check 1b] from build_population(): {0} abstained, {1} answered".format(
        len(abstained_b), len(answered_b)))
    if len(abstained_b) != 3 or len(answered_b) != 14:
        raise SystemExit(
            "[STOP] expected 3 abstained / 14 answered from build_population(), "
            "got {0}/{1}".format(len(abstained_b), len(answered_b)))
    if abstained_a != abstained_b:
        raise SystemExit(
            "[STOP] the two independent abstained-frame lists disagree:\n"
            "  from relation files:   {0}\n"
            "  from build_population: {1}".format(abstained_a, abstained_b))
    print("[check 1] PASS -- both methods agree: {0}".format(abstained_a))

    # ---- control check 2: filtered population size ----
    n_total = len(population)
    filtered = [it for it in population if not it["frame_abstained"]]
    print("[check 2] population {0} -> filtered {1} (expect 162)".format(
        n_total, len(filtered)))
    if len(filtered) != 162:
        raise SystemExit(
            "[STOP] filtered population is {0}, not 162 -- frame_key join is "
            "dropping or duplicating items, investigate before scoring "
            "anything.".format(len(filtered)))
    print("[check 2] PASS")

    # ---- score MiniLM: primary (216, unfiltered) and 14-frame (162) ----
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise SystemExit("sentence-transformers not importable: {0}".format(exc))

    LOG.info("loading all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vocab_emb = model.encode(PREDICATE_VOCAB, convert_to_tensor=True)

    def score(pop):
        hits = {k: 0 for k in TOP_KS}
        n_cond = 0
        for item in pop:
            scoreable = [c for c in item["candidates"] if c["target"] is not None]
            if not scoreable:
                continue
            n_cond += 1
            best = min(rank_of(model, util, c["phrase"], c["target"], vocab_emb)
                       for c in scoreable)
            for k in TOP_KS:
                if best <= k:
                    hits[k] += 1
        return hits, n_cond

    hits_216, n_cond_216 = score(population)
    hits_162, n_cond_162 = score(filtered)

    print("\n[primary 216-item run via this script] hits={0} n_conditional={1}".format(
        hits_216, n_cond_216))
    print("(cross-check against predicate_rk_results.json: expect hits {{1:5,3:8,5:9}})")

    # ---- control check 3: scored items survive the filter unchanged ----
    print("\n[check 3] hits over filtered 162-item population: {0}".format(hits_162))
    if hits_162 != hits_216:
        raise SystemExit(
            "[STOP] hit counts differ between the 216-item and 162-item "
            "populations ({0} vs {1}) -- an abstained frame should "
            "contribute zero scored items, so this should be impossible. "
            "Refusing to write output; investigate the frame_key "
            "join.".format(hits_216, hits_162))
    if n_cond_162 != n_cond_216:
        raise SystemExit(
            "[STOP] n_conditional differs between populations ({0} vs {1}); "
            "expected both to equal 19.".format(n_cond_216, n_cond_162))
    print("[check 3] PASS -- hit counts and n_conditional unchanged by the "
          "filter, as required; only the denominator moves (216 -> 162).")

    r_216 = {k: hits_216[k] / 216 for k in TOP_KS}
    r_162 = {k: hits_162[k] / 162 for k in TOP_KS}

    print("\n=== RESULT (all-MiniLM-L6-v2) ===")
    print("  216-item denominator (primary, headline): R@1={0:.1%} R@3={1:.1%} R@5={2:.1%}".format(
        r_216[1], r_216[3], r_216[5]))
    print("  162-item denominator (14-frame conditional): R@1={0:.1%} R@3={1:.1%} R@5={2:.1%}".format(
        r_162[1], r_162[3], r_162[5]))
    print("  delta (162 minus 216), percentage points: R@1={0:+.2f} R@3={1:+.2f} R@5={2:+.2f}".format(
        100 * (r_162[1] - r_216[1]), 100 * (r_162[3] - r_216[3]), 100 * (r_162[5] - r_216[5])))

    if args.json_out:
        atomic_write_json(args.json_out, dict(
            min_px=args.min_px, iou_thresh=args.iou_thresh,
            n_abstain_frames=3, n_responding_frames=14,
            abstained_frame_keys=abstained_a,
            n_population_primary=216, n_population_14frame=162,
            checks=dict(
                check1_agree=True, check2_count_162=True,
                check3_hits_unchanged=True,
            ),
            encoder="all-MiniLM-L6-v2",
            hits=hits_216, n_conditional=n_cond_216,
            primary_216=dict(denominator=216, hits=hits_216, R_at_K=r_216),
            conditional_14frame_162=dict(denominator=162, hits=hits_162, R_at_K=r_162),
            delta_pp_162_minus_216={k: 100 * (r_162[k] - r_216[k]) for k in TOP_KS},
        ))
        LOG.info("written: %s", args.json_out)


if __name__ == "__main__":
    main()
