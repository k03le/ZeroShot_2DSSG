#!/usr/bin/env python3
"""
count_predicate_diversity.py — settle the open-vocabulary-relations claim with a count.

PURPOSE
    The claim under test is that the VLM produces open-vocabulary relations geometry
    cannot. It is currently ⚠ WEAKENED on the strength of "1 of 19 relations was a
    rich/semantic type" — computed on the qwen2.5vl corpus only, by an unrecorded
    method. qwen3-vl@26 is now the canonical corpus and NOBODY HAS COUNTED IT.

    Three modes, deliberately separated so the lexicon is fixed before the count is seen:

      probe      Dump the JSON structure of one relation file. Assumes nothing.
      inventory  Every distinct predicate string + frequency -> CSV with a blank `bucket`
                 column, plus a codebook. You fill `bucket` BEFORE seeing totals.
      classify   Read the filled CSV back; report bucket totals, per-frame breakdown, and
                 a like-for-like comparison against a second corpus.

INPUT
    --rel-dir /data/mmc-user/outputs/vlm_relations
    --suffix  __ii_qwen3vl.json   (or __ii_ollama.json)

USAGE
    conda activate ram_pipeline

    python count_predicate_diversity.py --mode probe \
        --rel-dir /data/mmc-user/outputs/vlm_relations --suffix __ii_qwen3vl.json

    python count_predicate_diversity.py --mode inventory \
        --rel-dir /data/mmc-user/outputs/vlm_relations --suffix __ii_qwen3vl.json \
        --out ./predicates_qwen3vl.csv

    #  ... fill the `bucket` column ...

    python count_predicate_diversity.py --mode classify \
        --csv ./predicates_qwen3vl.csv --json-out ./predicate_diversity_qwen3vl.json

    # then the identical treatment on the other corpus, and compare:
    python count_predicate_diversity.py --mode classify \
        --csv ./predicates_qwen3vl.csv --compare-csv ./predicates_ollama.csv

NOTES
    * Run the SAME lexicon over both corpora. The "1 of 19" figure was produced by an
      unrecorded method; re-deriving it under this codebook is what makes the two numbers
      comparable. Do not cite the old figure alongside a new one.
    * The honest outcome may be that the open-vocabulary-relations claim stays weakened.
      If qwen3-vl's 26 relations contain
      one or two rich-semantic types, the core-contribution claim becomes a limitation.
      Better to learn that now than in §4.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

LOG = logging.getLogger("predicate_diversity")

BUCKETS = ("camera_relative", "topological_support", "rich_semantic", "junk")

# Token references stripped for publication (2026-09). Bucket definitions unchanged.
CODEBOOK = """\
# predicate bucket codebook (fix these BEFORE looking at totals)

One row per distinct predicate string across the corpus. Assign exactly one bucket.

| bucket | definition | examples |
|---|---|---|
| `camera_relative` | True only from a viewpoint. Swapping the camera changes or destroys the relation — the project's most-confirmed finding, and not evidence for the open-vocabulary-relations claim. | left of, right of, in front of, behind, above, below, next to (when viewpoint-dependent) |
| `topological_support` | Viewpoint-invariant, but derivable in principle from geometry alone — two boxes plus gravity. These do NOT support the open-vocabulary-relations claim, because that claim is specifically about relations geometry *cannot* produce. | on, on top of, under, in, inside, supports, supported by, attached to, touching |
| `rich_semantic` | Requires knowledge about what the objects ARE and how they are used — not recoverable from geometry at any precision. **This bucket is the entire content of the open-vocabulary-relations claim.** | leads into, connected with, has part, part of, plugged into, belongs to, used with, faces |
| `junk` | Parser artifacts, empty strings, malformed output, relations involving a hallucinated object. Excluded from the denominator; reported separately. | "", "relation", "n/a" |

## Rules
1. Fill every row before running `--mode classify`. Unfilled rows are excluded and flagged.
2. Judge the predicate STRING, not the specific instance. If a string is genuinely
   ambiguous across instances (e.g. "next to" used both ways), assign the bucket that
   fits the majority and note it — do not split one string across buckets.
3. `attached to` is `topological_support`, not `rich_semantic`. Contact is geometric.
   This matters: "attached to" was the single relation carrying the open-vocabulary-relations
   claim under qwen2.5vl. If it moves to topological_support, that claim's old evidence was
   already thinner than stated.
4. Apply this identical codebook to BOTH corpora before comparing them.

## What the result means
- rich_semantic count is materially > 1 across 26 relations -> the open-vocabulary-relations
  claim survives, restated with the real number and a per-model caveat.
- rich_semantic count is still ~1 -> the open-vocabulary-relations claim becomes a Limitation:
  "under an Ollama-served 7-8B open-weight VLM, open-vocabulary output collapses toward the
  camera-relative and topological families." That is a real, publishable negative finding —
  it is the empirical evidence, in your own data, for whether VLMs genuinely understand
  relations rather than merely detecting geometry.
"""


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _relation_list(blob):
    """Find the relation container without assuming the key."""
    if isinstance(blob, list):
        return blob, "<root list>"
    if not isinstance(blob, dict):
        return [], None
    for key in ("relations", "triples", "relations_readable", "scene_graph", "edges", "output"):
        v = blob.get(key)
        if isinstance(v, list) and v:
            return v, key
    for key, v in blob.items():
        if isinstance(v, list) and v and isinstance(v[0], (dict, list, str)):
            return v, key
    return [], None


def _predicate(item):
    """Pull the predicate out of a relation item of unknown shape."""
    if isinstance(item, dict):
        for k in ("relation", "predicate", "rel", "pred", "type"):
            if item.get(k):
                return str(item[k]).strip().lower()
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return str(item[1]).strip().lower()
    if isinstance(item, str):
        # "desk supports monitor" — cannot be split reliably; flag for manual handling
        return None
    return None


def cmd_probe(args) -> int:
    files = sorted(Path(args.rel_dir).glob(f"*{args.suffix}"))
    if not files:
        LOG.error("no files matching *%s in %s", args.suffix, args.rel_dir)
        return 1
    f = files[0]
    blob = json.load(f.open(encoding="utf-8"))
    print(f"\n--- {f.name} ---")
    print(f"  top-level type : {type(blob).__name__}")
    if isinstance(blob, dict):
        print(f"  top-level keys : {list(blob)}")
        for k, v in blob.items():
            kind = type(v).__name__
            extra = f" (len {len(v)})" if isinstance(v, (list, dict, str)) else ""
            print(f"     {k:<24} {kind}{extra}")
    rels, key = _relation_list(blob)
    print(f"\n  relation container key : {key}")
    print(f"  n relations            : {len(rels)}")
    for i, r in enumerate(rels[:4]):
        print(f"    [{i}] {type(r).__name__}: {r}")
    if rels:
        print(f"\n  predicate extracted from [0]: {_predicate(rels[0])!r}")
        if _predicate(rels[0]) is None:
            print("  ** relations look like free-form strings — inventory mode cannot")
            print("     split them automatically. Tell me the shape and I'll adapt the parser. **")
    print(f"\n  files in corpus: {len(files)}")
    print()
    return 0


def cmd_inventory(args) -> int:
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        LOG.error("%s exists — --overwrite discards any buckets already assigned", out)
        return 1

    files = sorted(Path(args.rel_dir).glob(f"*{args.suffix}"))
    if not files:
        LOG.error("no files matching *%s in %s", args.suffix, args.rel_dir)
        return 1

    counts, per_frame, unparsed = Counter(), defaultdict(Counter), 0
    total = 0
    for f in files:
        try:
            blob = json.load(f.open(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOG.error("malformed JSON %s: %s", f.name, exc)
            return 1
        rels, _ = _relation_list(blob)
        frame = f.name.replace(args.suffix, "")
        for r in rels:
            total += 1
            pred = _predicate(r)
            if pred is None:
                unparsed += 1
                continue
            counts[pred] += 1
            per_frame[frame][pred] += 1

    if unparsed:
        LOG.warning("%d/%d relation items had no extractable predicate — run --mode probe "
                    "and report the shape rather than guessing", unparsed, total)
    if not counts:
        LOG.error("no predicates extracted; run --mode probe")
        return 1

    LOG.info("%d files, %d relation items, %d distinct predicates",
             len(files), total, len(counts))
    LOG.info("corpus total (the denominator the open-vocabulary-relations claim is stated over): %d", total)

    rows = [{"predicate": p, "count": n,
             "n_frames": sum(1 for fr in per_frame.values() if fr.get(p)),
             "bucket": "", "note": ""}
            for p, n in counts.most_common()]
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    _atomic_write(out, sio.getvalue())
    _atomic_write(out.with_suffix(".codebook.md"), CODEBOOK)

    LOG.info("inventory -> %s  (%d rows)", out, len(rows))
    LOG.info("codebook  -> %s  (read BEFORE filling `bucket`)", out.with_suffix(".codebook.md"))
    LOG.info("allowed buckets: %s", " | ".join(BUCKETS))
    return 0


def _load_buckets(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    bad = sorted({r["bucket"].strip() for r in rows
                  if r["bucket"].strip() and r["bucket"].strip() not in BUCKETS})
    return rows, bad


def cmd_classify(args) -> int:
    rows, bad = _load_buckets(Path(args.csv))
    if bad:
        LOG.error("unrecognized buckets: %s (allowed: %s)", bad, " | ".join(BUCKETS))
        return 1
    pending = [r for r in rows if not r["bucket"].strip()]
    if pending:
        LOG.warning("%d/%d predicate rows unbucketed — excluded from all totals",
                    len(pending), len(rows))
        for r in pending[:10]:
            LOG.warning("   pending: %r (count %s)", r["predicate"], r["count"])

    tally = Counter()
    for r in rows:
        b = r["bucket"].strip()
        if b in BUCKETS:
            tally[b] += int(r["count"])
    denom = sum(tally[b] for b in ("camera_relative", "topological_support", "rich_semantic"))

    print()
    print(f"  corpus : {Path(args.csv).name}")
    print(f"  distinct predicates : {len(rows)}   bucketed: {len(rows) - len(pending)}")
    print(f"  relation instances (denominator, junk excluded): {denom}")
    print(f"  junk excluded : {tally.get('junk', 0)}")
    print()
    for b in ("camera_relative", "topological_support", "rich_semantic"):
        k = tally.get(b, 0)
        frac = k / denom if denom else 0.0
        print(f"    {b:<22} {k:>4}   {frac:>6.1%}")
    print()
    rich = tally.get("rich_semantic", 0)
    print(f"  >>> open-vocabulary-relations claim verdict input: {rich} rich-semantic relation instances "
          f"of {denom} ({rich/denom:.1%} of the corpus)" if denom else "  >>> empty corpus")
    strings = [r["predicate"] for r in rows if r["bucket"].strip() == "rich_semantic"]
    print(f"      distinct rich-semantic predicate strings: {len(strings)}")
    if strings:
        print(f"      {strings}")
    print()

    if args.compare_csv:
        other_rows, other_bad = _load_buckets(Path(args.compare_csv))
        if other_bad:
            LOG.error("comparison CSV has unrecognized buckets: %s", other_bad)
            return 1
        other = Counter()
        for r in other_rows:
            b = r["bucket"].strip()
            if b in BUCKETS:
                other[b] += int(r["count"])
        odenom = sum(other[b] for b in ("camera_relative", "topological_support", "rich_semantic"))
        print(f"  like-for-like vs {Path(args.compare_csv).name}")
        print(f"    {'bucket':<22}{'this':>12}{'other':>12}")
        for b in ("camera_relative", "topological_support", "rich_semantic"):
            a = f"{tally.get(b,0)} ({tally.get(b,0)/denom:.0%})" if denom else "-"
            o = f"{other.get(b,0)} ({other.get(b,0)/odenom:.0%})" if odenom else "-"
            print(f"    {b:<22}{a:>12}{o:>12}")
        print(f"    {'denominator':<22}{denom:>12}{odenom:>12}")
        print()

    if args.json_out:
        payload = {
            "csv": str(args.csv),
            "n_distinct_predicates": len(rows),
            "n_pending": len(pending),
            "denominator": denom,
            "junk": tally.get("junk", 0),
            "buckets": {b: tally.get(b, 0) for b in BUCKETS},
            "rich_semantic_strings": strings,
        }
        _atomic_write(Path(args.json_out), json.dumps(payload, indent=2))
        LOG.info("summary -> %s", args.json_out)

    return 2 if pending else 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Count how many distinct relation types the VLM produces, and how "
                     "many require semantic knowledge that geometry alone cannot supply.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="buckets: " + " | ".join(BUCKETS),
    )
    p.add_argument("--mode", choices=("probe", "inventory", "classify"), required=True)
    p.add_argument("--rel-dir", default="/data/mmc-user/outputs/vlm_relations")
    p.add_argument("--suffix", default="__ii_qwen3vl.json")
    p.add_argument("--out", default=None, help="[inventory] CSV output path")
    p.add_argument("--csv", default=None, help="[classify] filled inventory CSV")
    p.add_argument("--compare-csv", default=None, help="[classify] second corpus, same codebook")
    p.add_argument("--json-out", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.mode == "probe":
        return cmd_probe(args)
    if args.mode == "inventory":
        if not args.out:
            LOG.error("--mode inventory requires --out")
            return 1
        return cmd_inventory(args)
    if not args.csv:
        LOG.error("--mode classify requires --csv")
        return 1
    return cmd_classify(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.warning("interrupted")
        sys.exit(130)