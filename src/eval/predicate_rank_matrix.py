"""
Script:  predicate_rank_matrix.py
Purpose: For every distinct VLM phrase (14) x every scored GT predicate (20),
         report the RANK the GT predicate receives under that phrase, per
         encoder. Answers directly: does 'next to' find left/right? does
         'on top of' find lying on / supported by / standing in? does 'above'
         find close by? -- and where each encoder fails.

         These are NOT inverse-pair questions (Section A proved the inverse
         structure is exactly left<->right and higher<->lower). They are
         semantic-distance questions, which is what actually decides whether a
         correct-in-spirit VLM phrase ranks its GT predicate high or buries it.

READ-ONLY.
USAGE
    conda activate ram_pipeline
    export HF_HUB_OFFLINE=1
    python predicate_rank_matrix.py \
        --mpnet ~/hf_stage/all-mpnet-base-v2 \
        --clip  ~/hf_stage/clip-ViT-L-14 \
        --json-out /data/mmc-user/outputs/predicate_rank_matrix.json
"""
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np

PREDICATE_VOCAB = [
    "left", "right", "close by", "attached to", "standing on",
    "higher than", "lower than", "same as", "lying on", "hanging on",
    "supported by", "standing in", "build in", "leaning against",
    "connected to", "belonging to", "lying in", "part of", "cover",
    "hanging in",
]
# the 14 phrases the VLM actually emitted, with corpus counts (Section B)
VLM_PHRASES = [
    ("on", 20), ("next to", 12), ("above", 7), ("in front of", 6),
    ("under", 6), ("behind", 5), ("on top of", 5), ("left of", 3),
    ("below", 3), ("on the floor next to", 2), ("beside", 1),
    ("to the right of", 1), ("to the left of", 1), ("near", 1),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minilm", default="all-MiniLM-L6-v2")
    ap.add_argument("--mpnet", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.json_out and os.path.exists(args.json_out) and not args.overwrite:
        raise SystemExit("[refusing] exists; pass --overwrite")

    from sentence_transformers import SentenceTransformer, util
    out = {}
    for name, path in [("all-MiniLM-L6-v2", args.minilm),
                       ("all-mpnet-base-v2", args.mpnet),
                       ("clip-ViT-L-14", args.clip)]:
        try:
            model = SentenceTransformer(path)
        except Exception as exc:
            print("FAILED to load {0}: {1}".format(name, exc))
            continue
        pv = model.encode(PREDICATE_VOCAB, convert_to_tensor=True)
        ranks = {}
        print("\n" + "=" * 100)
        print("{0}   — rank of each GT predicate (column) under each VLM phrase (row)".format(name))
        print("=" * 100)
        hdr = "{0:<22}".format("VLM phrase")
        for p in PREDICATE_VOCAB:
            hdr += "{0:>5}".format(p[:4])
        print(hdr)
        for ph, n in VLM_PHRASES:
            q = model.encode(ph, convert_to_tensor=True)
            sims = util.cos_sim(q, pv)[0].cpu().numpy()
            order = np.argsort(-sims)
            r = {PREDICATE_VOCAB[i]: int(np.where(order == i)[0][0]) + 1
                 for i in range(len(PREDICATE_VOCAB))}
            ranks[ph] = r
            row = "{0:<22}".format(ph[:22])
            for p in PREDICATE_VOCAB:
                row += "{0:>5}".format(r[p])
            print(row)
        out[name] = ranks

        # the specific pairings the user asked about
        print("\n  -- the pairings in question (rank of GT predicate under that phrase) --")
        QUERIES = [
            ("next to", ["left", "right", "close by"]),
            ("on top of", ["lying on", "supported by", "standing in", "standing on"]),
            ("to the right of", ["close by", "right"]),
            ("to the left of", ["close by", "left"]),
            ("above", ["close by", "higher than"]),
            ("on", ["standing on", "lying on", "supported by"]),
            ("under", ["lower than", "standing on"]),
            ("on the floor next to", ["close by", "standing on"]),
        ]
        for ph, targets in QUERIES:
            parts = ", ".join("{0}={1}".format(t, ranks[ph][t]) for t in targets)
            print("    {0:<24} {1}".format(repr(ph), parts))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        d = os.path.dirname(os.path.abspath(args.json_out))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(dict(predicate_vocab=PREDICATE_VOCAB,
                           vlm_phrases=[p for p, _ in VLM_PHRASES],
                           ranks=out), fh, indent=2)
        os.replace(tmp, args.json_out)
        print("\nwritten: {0}".format(args.json_out))

if __name__ == "__main__":
    main()