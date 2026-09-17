"""
Script:  encoder_probe.py
Stage:   Recall@K evaluation, step 1 — headline-encoder designation (flag 17).
Purpose: Compare all-MiniLM-L6-v2 / all-mpnet-base-v2 / clip-ViT-L-14 on a
         PRE-REGISTERED semantic probe, so the primary encoder is selected on
         evidence INDEPENDENT of the headline metric (triplet R@10).

         This follows the same precedent as an earlier probe exactly: hand-built
         cases, pass bar fixed BEFORE running, no retuning if an encoder fails
         by one.

         READ-ONLY except for --json-out.

PRE-REGISTRATION (fixed 2026-08-17, before any encoder was run):
  * Cases          = every distinct SCOREABLE predicate phrase the VLM actually
                     emitted (12 of 14; front/behind excluded), and
                     12 real node labels drawn from the same corpus.
  * Scoring        = best-of-acceptable-set, top-3. Declared as the GENEROUS
                     reading ('on' has no single correct GT target).
  * Pass bar       = >= 10/12 in top-3, per sub-probe. Carried over from an
                     earlier probe's 9/11 (81.8% -> 83.3%). NOT retuned on failure.
  * Selection rule = (1) predicate top-3 accuracy; (2) fewer antonym inversions
                     on the 3 directional cases; (3) node top-3 accuracy;
                     (4) tie -> MiniLM for predicates, CLIP for nodes.
  * Excluded nodes = black rolling cart / white dresser / vertical framed
                     artwork / white drawer unit. No defensible 160-class
                     target. Reported as vocabulary-ceiling instances.

VOCABULARIES
  Predicates: the scored vocabulary's 22 MINUS front/behind = 20.
  Nodes:      official 3DSSG classes.txt = 160 (flag 16 closed 2026-08-16).

INPUT
    --minilm / --mpnet / --clip   model paths or cached names
    --classes                     path to official classes.txt (160 lines)
OUTPUT
    console: per-encoder per-case table + summary + selection verdict
    --json-out: machine-readable result (atomic write)
USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python encoder_probe.py \
        --classes ~/3dssg_subset_official/3DSSG_subset/classes.txt \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --json-out ~/encoder_probe_results.json
"""

import argparse
import json
import logging
import os
import sys
import tempfile

logger = logging.getLogger("encoder_probe")

# --------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS — do not edit after first run
# --------------------------------------------------------------------------

# The scored vocabulary's 22 minus front/behind = 20 scored predicates.
PREDICATE_VOCAB = [
    "left", "right", "close by", "attached to", "standing on",
    "higher than", "lower than", "same as", "lying on", "hanging on",
    "supported by", "standing in", "build in", "leaning against",
    "connected to", "belonging to", "lying in", "part of", "cover",
    "hanging in",
]

ANTONYMS = {
    "left": "right",
    "right": "left",
    "higher than": "lower than",
    "lower than": "higher than",
}

# Every distinct SCOREABLE phrase from the real 73-relation VLM-predicted corpus,
# with its emitted frequency. Front/behind phrases excluded.
PREDICATE_CASES = [
    {"query": "on",                    "n": 20, "accept": ["standing on", "lying on", "supported by"]},
    {"query": "next to",               "n": 12, "accept": ["close by"]},
    {"query": "above",                 "n": 7,  "accept": ["higher than"]},
    {"query": "under",                 "n": 6,  "accept": ["lower than"]},
    {"query": "on top of",             "n": 5,  "accept": ["standing on", "lying on", "supported by"]},
    {"query": "left of",               "n": 3,  "accept": ["left"], "directional": True},
    {"query": "below",                 "n": 3,  "accept": ["lower than"]},
    {"query": "on the floor next to",  "n": 2,  "accept": ["close by"]},
    {"query": "beside",                "n": 1,  "accept": ["close by"]},
    {"query": "to the right of",       "n": 1,  "accept": ["right"], "directional": True},
    {"query": "to the left of",        "n": 1,  "accept": ["left"],  "directional": True},
    {"query": "near",                  "n": 1,  "accept": ["close by"]},
]

# Real node labels from the same corpus, mapped to 160-class targets.
NODE_CASES = [
    {"query": "white desk",                        "n": 13, "accept": ["desk", "computer desk"]},
    {"query": "white pillow",                      "n": 6,  "accept": ["pillow", "cushion"]},
    {"query": "grey armchair with yellow pillow",  "n": 6,  "accept": ["armchair"]},
    {"query": "white shelf",                       "n": 6,  "accept": ["shelf", "bookshelf", "rack"]},
    {"query": "grey ottoman",                      "n": 4,  "accept": ["ottoman", "footstool", "stool"]},
    {"query": "children's book",                   "n": 4,  "accept": ["book", "books"]},
    {"query": "green chair",                       "n": 4,  "accept": ["chair", "dining chair", "armchair", "sofa chair", "rocking chair"]},
    {"query": "glass vase with white flowers",     "n": 4,  "accept": ["vase"]},
    {"query": "white lamp",                        "n": 3,  "accept": ["lamp", "table lamp", "light"]},
    {"query": "floor tile",                        "n": 3,  "accept": ["floor"]},
    {"query": "small potted plant",                "n": 3,  "accept": ["plant", "flower", "flowers"]},
    {"query": "colorful patterned rug",            "n": 2,  "accept": ["carpet"]},
]

EXCLUDED_NODES = [
    {"query": "black rolling cart",      "n": 8, "reason": "no cart/trolley in the 160-class vocabulary"},
    {"query": "white dresser",           "n": 3, "reason": "dresser absent; commode vs wardrobe both arguable"},
    {"query": "vertical framed artwork", "n": 4, "reason": "picture vs frame genuinely ambiguous"},
    {"query": "white drawer unit",       "n": 4, "reason": "drawer vs commode vs cabinet ambiguous"},
]

TOP_K = 3
PASS_BAR = 10  # out of 12, per sub-probe


# --------------------------------------------------------------------------


def load_classes(path):
    """Read the official classes.txt. Fails loud on the wrong line count."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            classes = [ln.strip() for ln in fh if ln.strip()]
    except OSError as exc:
        logger.error("Cannot read classes file %s: %s", path, exc)
        raise
    if len(classes) != 160:
        logger.error(
            "classes.txt has %d entries, expected 160 (flag 16). Refusing to "
            "proceed on the wrong vocabulary.", len(classes)
        )
        raise SystemExit(2)
    logger.info("Loaded %d node classes from %s", len(classes), path)
    return classes


def rank_vocab(model, query, vocab):
    """Return vocab entries ordered by descending cosine against query."""
    from sentence_transformers import util

    q_emb = model.encode(query, convert_to_tensor=True)
    v_emb = model.encode(vocab, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, v_emb)[0]
    order = sims.argsort(descending=True).tolist()
    return [(vocab[i], float(sims[i])) for i in order]


def run_subprobe(model, cases, vocab, label):
    """Score one encoder on one sub-probe. Returns (results, hits, inversions)."""
    from tqdm import tqdm

    results = []
    hits = 0
    inversions = 0

    for case in tqdm(cases, desc=label, leave=False):
        ranked = rank_vocab(model, case["query"], vocab)
        ranked_names = [name for name, _ in ranked]

        # rank of the best acceptable target (1-indexed)
        best_rank = None
        best_target = None
        for target in case["accept"]:
            if target not in ranked_names:
                logger.warning("Target %r not in vocabulary — skipping", target)
                continue
            r = ranked_names.index(target) + 1
            if best_rank is None or r < best_rank:
                best_rank = r
                best_target = target

        hit = best_rank is not None and best_rank <= TOP_K
        if hit:
            hits += 1

        # antonym guard: does the antonym outrank the correct answer?
        inverted = False
        if case.get("directional") and best_target is not None:
            anto = ANTONYMS.get(best_target)
            if anto is not None and anto in ranked_names:
                anto_rank = ranked_names.index(anto) + 1
                if anto_rank < best_rank:
                    inverted = True
                    inversions += 1

        results.append({
            "query": case["query"],
            "mentions": case["n"],
            "accept": case["accept"],
            "best_target": best_target,
            "best_rank": best_rank,
            "top3": ranked_names[:3],
            "hit": hit,
            "antonym_inverted": inverted,
        })

    return results, hits, inversions


def print_table(results, encoder, label, hits, inversions):
    print("\n--- {0} | {1} ---".format(encoder, label))
    print("{0:<30} {1:>4} {2:>6}  {3:<28} {4}".format(
        "query", "rank", "hit", "best target", "top-3"))
    for r in results:
        mark = "OK " if r["hit"] else "MISS"
        if r["antonym_inverted"]:
            mark = "INV"
        rank_s = str(r["best_rank"]) if r["best_rank"] is not None else "-"
        print("{0:<30} {1:>4} {2:>6}  {3:<28} {4}".format(
            r["query"][:30], rank_s, mark,
            str(r["best_target"])[:28], ", ".join(r["top3"])))
    verdict = "PASS" if hits >= PASS_BAR else "FAIL"
    print("  => {0}/{1} in top-{2}  [{3}]  antonym inversions: {4}".format(
        hits, len(results), TOP_K, verdict, inversions))


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
        description="Pre-registered 3-encoder semantic probe (flag 17).")
    ap.add_argument("--classes", required=True,
                    help="official 3DSSG classes.txt (must be 160 lines)")
    ap.add_argument("--minilm", default="all-MiniLM-L6-v2",
                    help="path or cached name for all-MiniLM-L6-v2")
    ap.add_argument("--mpnet", required=True, help="path to all-mpnet-base-v2")
    ap.add_argument("--clip", required=True, help="path to clip-ViT-L-14")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s")

    if args.json_out and os.path.exists(args.json_out) and not args.overwrite:
        logger.error("Output %s exists. Pass --overwrite to replace it.",
                     args.json_out)
        raise SystemExit(1)

    node_vocab = load_classes(args.classes)

    print("=" * 78)
    print("PRE-REGISTERED PROBE  |  pass bar {0}/12 in top-{1}  |  NOT retuned".format(
        PASS_BAR, TOP_K))
    print("predicate vocab: {0}  |  node vocab: {1}".format(
        len(PREDICATE_VOCAB), len(node_vocab)))
    print("excluded node labels ({0}), vocabulary-ceiling instances:".format(
        len(EXCLUDED_NODES)))
    for e in EXCLUDED_NODES:
        print("   {0:<26} n={1:<3} {2}".format(e["query"], e["n"], e["reason"]))
    print("=" * 78)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        logger.error("sentence-transformers not importable: %s", exc)
        raise SystemExit(3)

    encoders = [
        ("all-MiniLM-L6-v2", args.minilm),
        ("all-mpnet-base-v2", args.mpnet),
        ("clip-ViT-L-14", args.clip),
    ]

    payload = {
        "pass_bar": PASS_BAR,
        "top_k": TOP_K,
        "n_cases": len(PREDICATE_CASES),
        "predicate_vocab": PREDICATE_VOCAB,
        "n_node_vocab": len(node_vocab),
        "excluded_nodes": EXCLUDED_NODES,
        "encoders": {},
    }

    for name, path in encoders:
        logger.info("Loading %s from %s", name, path)
        try:
            model = SentenceTransformer(path)
        except Exception as exc:
            logger.error("FAILED to load %s: %s: %s", name, type(exc).__name__, exc)
            payload["encoders"][name] = {"load_error": str(exc)}
            continue

        p_res, p_hits, p_inv = run_subprobe(
            model, PREDICATE_CASES, PREDICATE_VOCAB, "predicates")
        n_res, n_hits, n_inv = run_subprobe(
            model, NODE_CASES, node_vocab, "nodes")

        print_table(p_res, name, "PREDICATES", p_hits, p_inv)
        print_table(n_res, name, "NODES", n_hits, n_inv)

        payload["encoders"][name] = {
            "path": path,
            "predicate": {"results": p_res, "hits": p_hits,
                          "inversions": p_inv,
                          "pass": p_hits >= PASS_BAR},
            "node": {"results": n_res, "hits": n_hits,
                     "pass": n_hits >= PASS_BAR},
        }

    # ---- selection rule, applied exactly as pre-registered ----
    print("\n" + "=" * 78)
    print("SELECTION (rule fixed before running)")
    print("=" * 78)
    scored = [(k, v) for k, v in payload["encoders"].items()
              if "load_error" not in v]
    if not scored:
        print("No encoder loaded. Nothing to select.")
    else:
        ranked = sorted(
            scored,
            key=lambda kv: (-kv[1]["predicate"]["hits"],
                            kv[1]["predicate"]["inversions"],
                            -kv[1]["node"]["hits"]))
        for k, v in ranked:
            print("  {0:<20} pred {1}/12 (inv {2})  node {3}/12".format(
                k, v["predicate"]["hits"], v["predicate"]["inversions"],
                v["node"]["hits"]))
        winner = ranked[0]
        print("\n  -> primary encoder by pre-registered rule: {0}".format(winner[0]))
        if not winner[1]["predicate"]["pass"]:
            print("  !! WINNER DID NOT MEET THE PASS BAR. Bar NOT retuned.")
            print("     This is a logged failure.")
        payload["selected"] = winner[0]
        payload["selected_met_bar"] = winner[1]["predicate"]["pass"]

    if args.json_out:
        atomic_write_json(args.json_out, payload)
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()