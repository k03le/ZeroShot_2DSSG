"""
Script:  relation_pair_ordering_check.py
Purpose: Verify, per (s,o)<->(o,s) pair, 3DSSG's directional inverse-pair storage
         regime — and, as a by-product, decode the ENDPOINT ORDERING convention
         of relationships.json records.

         An earlier count inferred "distinct facts = records / 2" for directional
         predicates from AGGREGATE COUNT TIES (left 122 = right 122, front 59 = behind 59).
         A tie is necessary but NOT sufficient: it is satisfiable without any
         individual record having a partner. This script checks partnership
         PER PAIR.

         The inverse map is DISCOVERED from the data, not assumed: for every
         record [a,b,P], we tally which predicates Q appear in records [b,a,Q].
         Q == P  -> symmetric regime;  Q == some other predicate at high rate
         -> inverse-paired regime;  no reverse record -> directed regime.

         ORDERING DECODE: inverse pairing is symmetric under both readings of
         [a,b,P] ("a is P of b" vs "b is P of a"), so it CANNOT settle ordering.
         Support predicates can: a `floor` is stood ON, never standing. We tally
         which POSITION architectural supports (floor/ground/wall) occupy in
         standing on / lying on / supported by / attached to records.

Input:   --relationships /data/mmc-user/3dssg/relationships.json
         --dssg-root     /data/mmc-user/3dssg/3dssg_data  (semseg.v2.json labels)
Output:  console tables; optional --json-out (atomic write).
Usage:   python relation_pair_ordering_check.py --json-out /data/mmc-user/outputs/relation_pair_ordering.json
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

LOG = logging.getLogger("pair_order_check")

SCANS = ["0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca",
         "5341b7a5-8a66-2cdd-8751-70b98263cb8d",
         "5341b7bf-8a66-2cdd-8794-026113b7c312",
         "5630cfd8-12bf-2860-8773-e3dde9da2aff"]
SUPPORT = ("standing on", "lying on", "supported by", "attached to",
           "hanging on", "standing in", "lying in", "leaning against")
ARCH = ("floor", "ground", "wall", "ceiling")


def main():
    ap = argparse.ArgumentParser(description="Per-pair reverse-record verification + endpoint-ordering decode")
    ap.add_argument("--relationships", default="/data/mmc-user/3dssg/relationships.json")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg/3dssg_data")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(levelname)s %(name)s: %(message)s")
    if a.json_out and Path(a.json_out).exists() and not a.overwrite:
        raise SystemExit("[refusing] %s exists; pass --overwrite" % a.json_out)

    try:
        rel = json.loads(Path(a.relationships).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit("[input error] %s: %s" % (a.relationships, e))
    by_scan = {r["scan"]: r["relationships"] for r in rel["scans"] if r["scan"] in SCANS}
    missing = [s for s in SCANS if s not in by_scan]
    if missing:
        raise SystemExit("[input error] scans absent from relationships.json: %s" % missing)

    labels = {}
    for s in SCANS:
        p = Path(a.dssg_root) / s / "semseg.v2.json"
        try:
            labels[s] = {int(g["objectId"]): g["label"]
                         for g in json.loads(p.read_text())["segGroups"]}
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise SystemExit("[input error] %s: %s" % (p, e))

    # ---------------- reverse-predicate discovery + per-pair partnership -------
    rev_tally = defaultdict(Counter)     # P -> Counter of Q found at (b,a)
    n_rec = Counter()
    orphans = defaultdict(list)          # P -> [(scan8, a, b), ...]
    selfloops = []
    dupes = Counter()

    for scan, recs in by_scan.items():
        s8 = scan[:8]
        index = defaultdict(set)         # (a,b) -> {predicates}
        for a_, b_, _pid, p in recs:
            if a_ == b_:
                selfloops.append((s8, a_, p))
            index[(a_, b_)].add(p)
            dupes[(s8, a_, b_, p)] += 1
        for (a_, b_), preds in index.items():
            back = index.get((b_, a_), set())
            for p in preds:
                n_rec[p] += 1
                if not back:
                    orphans[p].append((s8, a_, b_))
                for q in back:
                    rev_tally[p][q] += 1

    print("\n== Per-pair reverse-record analysis (4 scans, whole-scan) ==")
    print("%-22s %6s %8s %8s   %s" % ("predicate", "n", "hasRev%", "orphan%",
                                      "dominant reverse predicate(s)"))
    regimes = {}
    for p in sorted(n_rec, key=lambda x: -n_rec[x]):
        n = n_rec[p]
        orp = len(orphans[p])
        has = 100.0 * (n - orp) / n
        top = rev_tally[p].most_common(3)
        dom_q, dom_n = (top[0] if top else ("-", 0))
        if dom_q == p and dom_n >= 0.95 * n:
            reg = "SYMMETRIC"
        elif dom_q != p and dom_n >= 0.95 * n:
            reg = "INVERSE-PAIRED"
        elif orp >= 0.95 * n:
            reg = "DIRECTED"
        else:
            reg = "MIXED**"
        regimes[p] = dict(n=n, orphans=orp, has_rev_pct=round(has, 1),
                          top_reverse=top, regime=reg)
        desc = ", ".join("%s=%d" % (q, c) for q, c in top) or "(none)"
        print("%-22s %6d %7.1f%% %7.1f%%   [%s] %s" % (p, n, has, 100.0*orp/n, reg, desc))

    print("\n**MIXED means the /2 canonicalisation is NOT uniformly justified for that"
          "\n  predicate — the aggregate count tie would have hidden it.")
    if selfloops:
        print("\nSELF-LOOPS (a==b): %d  e.g. %s" % (len(selfloops), selfloops[:5]))
    dup = {k: v for k, v in dupes.items() if v > 1}
    if dup:
        print("EXACT DUPLICATE RECORDS: %d  e.g. %s" % (len(dup), list(dup.items())[:5]))

    # ---------------- ordering decode via architectural supports --------------
    print("\n== endpoint ordering decode (support predicates, architectural endpoint) ==")
    print("If [a,b,P] means 'a is P of b' (SUBJECT-FIRST), a floor/wall should sit at"
          "\nposition b (index 1) for `standing on` / `attached to`.")
    pos = defaultdict(lambda: [0, 0])    # predicate -> [arch at index0, arch at index1]
    examples = defaultdict(list)
    for scan, recs in by_scan.items():
        lab = labels[scan]
        for a_, b_, _pid, p in recs:
            if p not in SUPPORT:
                continue
            la = (lab.get(a_, "?") or "").lower()
            lb = (lab.get(b_, "?") or "").lower()
            a_arch = any(k in la for k in ARCH)
            b_arch = any(k in lb for k in ARCH)
            if a_arch != b_arch:                       # exactly one is architectural
                idx = 0 if a_arch else 1
                pos[p][idx] += 1
                if len(examples[p]) < 3:
                    examples[p].append("[%d:%s, %d:%s, '%s']" % (a_, la, b_, lb, p))
    print("%-18s %10s %10s   %s" % ("predicate", "arch@idx0", "arch@idx1", "example"))
    tot = [0, 0]
    for p in sorted(pos, key=lambda x: -sum(pos[x])):
        c0, c1 = pos[p]
        tot[0] += c0
        tot[1] += c1
        print("%-18s %10d %10d   %s" % (p, c0, c1, examples[p][0] if examples[p] else ""))
    if sum(tot):
        verdict = ("SUBJECT-FIRST  [subject, object, pid, predicate]" if tot[1] > tot[0]
                   else "OBJECT-FIRST  [object, subject, pid, predicate]")
        print("\nTOTAL arch@idx0=%d  arch@idx1=%d  ->  %s  (%.1f%% consistent)"
              % (tot[0], tot[1], verdict, 100.0*max(tot)/sum(tot)))
    else:
        print("\n[inconclusive] no support record had exactly one architectural endpoint")

    if a.json_out:
        payload = dict(scans=SCANS, regimes=regimes,
                       ordering=dict(arch_idx0=tot[0], arch_idx1=tot[1],
                                     per_predicate={k: v for k, v in pos.items()}),
                       self_loops=len(selfloops), duplicate_records=len(dup))
        jp = Path(a.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        tmp = jp.with_name(".%s.tmp%d%s" % (jp.stem, os.getpid(), jp.suffix))
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, jp)
        LOG.info("json written: %s", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
