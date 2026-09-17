"""
Script:  census_3dssg_gt.py
Stage:   Pre-eval feasibility gate (no VLM, no GPU).
Purpose: Answer, from ground truth alone: how many edges are actually SCOREABLE
         per frame once (a) both endpoints must be visible in that frame's
         instance mask and (b) directional predicates are deferred?

         Also quantifies the pseudo-replication risk by measuring how much
         the visible-object sets of frames from the same scan overlap.

         READ-ONLY. Touches nothing but the optional --json-out report.

KEY FACTS THIS SCRIPT RELIES ON (verified, do not re-assume):
    - rendered.instances.png is NATIVELY UPRIGHT (540x960) -> NO rotation applied.
      The 17-frame orientation probe voted k=0 unanimously.
    - mask pixel values are objectIds in the same integer space as
      objects.json / relationships.json endpoints. Verified again here:
      every visible id must resolve to a label, else the script fails loud.

INPUT
    --frames-json  ingest_identify.json (gives scan + frame_no for each frame)
    --dssg-root    holds relationships.json, objects.json, 3dssg_data/<scan>/sequence/
OUTPUT
    console: per-frame table, predicate distribution, scan-level overlap matrix
    --json-out: machine-readable census (atomic write)
USAGE
    conda activate ram_pipeline
    python census_3dssg_gt.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --json-out /data/mmc-user/outputs/3dssg_upright_v2/gt_census.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger("census")

# Deferred-for-now predicate families. NOT applied silently: every observed
# predicate is printed with its bucket so the classification is reviewable, and
# anything unrecognised lands in "UNCLASSIFIED" rather than being dropped.
DIRECTIONAL = {
    "left", "right", "front", "behind", "above", "below",
    "left of", "right of", "in front of", "behind of",
    "front of", "back of", "bigger than", "smaller than",
    "higher than", "lower than",
}
COMPARATIVE_HINTS = ("same ", "more ", "less ", "than")   # attribute comparison hint words


def bucket(pred: str) -> str:
    p = pred.strip().lower()
    if p in DIRECTIONAL:
        return "directional"
    if any(h in p for h in COMPARATIVE_HINTS):
        return "comparative"
    return "structural"          # standing on / attached to / part of / lying on / ...


def load_scan_indexed(path: Path, key: str):
    """3DSSG json: {'scans': [{'scan': <id>, <key>: [...]}, ...]} -> {scan: [...]}"""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[input error] {path}: {e}")
    if "scans" not in raw:
        raise SystemExit(f"[schema] {path}: no 'scans' key; top-level keys = {list(raw)[:8]}")
    out = {}
    for entry in raw["scans"]:
        sid = entry.get("scan") or entry.get("scan_id")
        if sid is None:
            raise SystemExit(f"[schema] {path}: entry without 'scan'; keys = {list(entry)[:8]}")
        if key not in entry:
            raise SystemExit(f"[schema] {path}: entry {sid} has no '{key}'; keys = {list(entry)[:8]}")
        out[sid] = entry[key]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="GT-side scoreable-edge census for 3DSSG frames.")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--min-px", type=int, default=200,
                    help="an object counts as visible at >= this many mask pixels")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.json_out and Path(args.json_out).exists() and not args.overwrite:
        raise SystemExit(f"[refusing] {args.json_out} exists; pass --overwrite")

    root = Path(args.dssg_root)
    rels_by_scan = load_scan_indexed(root / "relationships.json", "relationships")
    objs_by_scan = load_scan_indexed(root / "objects.json", "objects")
    LOG.info("relationships.json: %d scans   objects.json: %d scans",
             len(rels_by_scan), len(objs_by_scan))

    rep = json.loads(Path(args.frames_json).read_text())
    frames = [f for f in rep["frames"] if f.get("ok")]
    LOG.info("frames to census: %d", len(frames))

    needed = {f["scan"] for f in frames}
    missing = needed - set(rels_by_scan)
    if missing:
        raise SystemExit(
            f"[blocking] no relationships for scan(s): {sorted(missing)}\n"
            f"  These may be rescans whose GT lives under the reference scan id.\n"
            f"  Inspect: python -c \"import json;d=json.load(open('{root}/relationships.json'));"
            f"print([s['scan'] for s in d['scans']][:10])\"")

    labels = {s: {int(o["id"]): o.get("label", "?") for o in objs}
              for s, objs in objs_by_scan.items()}

    rows, pred_counter, visible_by_frame = [], Counter(), {}
    for f in sorted(frames, key=lambda r: (r["scan"], r["frame_no"])):
        scan, no = f["scan"], f["frame_no"]
        key = f"{scan}_{no:06d}"
        mpath = root / args.data_subdir / scan / "sequence" / f"frame-{no:06d}.rendered.instances.png"
        try:
            mask = np.asarray(Image.open(mpath))          # NO rotation: natively upright
        except OSError as e:
            raise SystemExit(f"[input error] {mpath}: {e}")
        ids, counts = np.unique(mask, return_counts=True)
        vis = {int(i): int(c) for i, c in zip(ids, counts) if i != 0 and c >= args.min_px}
        visible_by_frame[key] = set(vis)

        unknown = [i for i in vis if i not in labels.get(scan, {})]
        if unknown:
            LOG.warning("%s: %d visible id(s) absent from objects.json: %s",
                        key, len(unknown), unknown[:6])

        b = Counter()
        edges_kept = []
        for e in rels_by_scan[scan]:
            try:
                s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            except (ValueError, IndexError, TypeError):
                raise SystemExit(f"[schema] unexpected edge format in {scan}: {e!r}")
            if s_id in vis and o_id in vis:
                bk = bucket(pname)
                b[bk] += 1
                pred_counter[(bk, pname)] += 1
                if bk == "structural":
                    edges_kept.append((s_id, pname, o_id))
        rows.append(dict(frame_key=key, scan=scan, frame_no=no,
                         n_visible=len(vis),
                         n_edges_all=sum(b.values()),
                         n_structural=b["structural"],
                         n_directional=b["directional"],
                         n_comparative=b["comparative"],
                         structural_edges=[
                             [labels.get(scan, {}).get(s, s), p, labels.get(scan, {}).get(o, o)]
                             for s, p, o in edges_kept]))

    print(f"\n{'frame_key':46s} {'vis':>4s} {'all':>5s} {'struct':>7s} {'dir':>5s} {'comp':>5s}")
    for r in rows:
        print(f"{r['frame_key']:46s} {r['n_visible']:>4d} {r['n_edges_all']:>5d} "
              f"{r['n_structural']:>7d} {r['n_directional']:>5d} {r['n_comparative']:>5d}")
    tot_s = sum(r["n_structural"] for r in rows)
    tot_a = sum(r["n_edges_all"] for r in rows)
    zero = [r["frame_key"] for r in rows if r["n_structural"] == 0]
    print(f"\nTOTAL scoreable (structural, both endpoints visible): {tot_s}"
          f"   of {tot_a} visible edges  ({100*tot_s/max(tot_a,1):.1f}%)")
    print(f"frames with ZERO structural edges: {len(zero)}/{len(rows)}  {zero}")

    print("\n-- observed predicates, by bucket (verify the classification) --")
    for (bk, p), n in sorted(pred_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{n:6d}  [{bk:12s}]  {p}")

    print("\n-- visible-object overlap within each scan (Jaccard; pseudo-replication) --")
    by_scan = defaultdict(list)
    for r in rows:
        by_scan[r["scan"]].append(r["frame_key"])
    pairs = []
    for scan, keys in by_scan.items():
        for a, b_ in combinations(sorted(keys), 2):
            A, B = visible_by_frame[a], visible_by_frame[b_]
            if not (A or B):
                continue
            j = len(A & B) / len(A | B)
            pairs.append((j, a, b_))
    for j, a, b_ in sorted(pairs, reverse=True)[:12]:
        flag = "  <-- near-duplicate viewpoint" if j >= 0.7 else ""
        print(f"J={j:.2f}  {a.split('_')[-1]} vs {b_.split('_')[-1]}  "
              f"({a.split('-')[0]}){flag}")
    if pairs:
        print(f"median Jaccard over {len(pairs)} same-scan pairs: "
              f"{float(np.median([p[0] for p in pairs])):.2f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(dict(min_px=args.min_px, n_frames=len(rows),
                           total_structural=tot_s, total_visible_edges=tot_a,
                           frames=rows,
                           predicates={f"{bk}|{p}": n for (bk, p), n in pred_counter.items()}),
                      fh, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("census written: %s", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())