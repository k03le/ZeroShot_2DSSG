#!/usr/bin/env python3
"""
count_support_gt.py — GT starvation check for the handpicked 3DSSG frames.

PURPOSE
    Before committing to the support-only fallback (or to the directional-predicate
    recomputation approach), count how many GT edges each handpicked frame can actually
    score: an edge is scoreable in a frame iff BOTH endpoints are visible in
    that frame's instance mask AND its predicate is inside the evaluated
    subset. If the support-subset count across all frames is tiny (~<15), a
    support-only evaluation is too starved to report and directional predicates
    stop being optional — run this BEFORE the time-boxed V3 attempt, because
    its result changes what V3's failure would cost.

INPUT
    --dssg-root   /data/mmc-user/3dssg
    --pairs       scan_id:frame_no pairs, e.g.
                  5341b7a5-8a66-2cdd-8751-70b98263cb8d:115
                  (frame_no int, zero-padded internally to frame-XXXXXX)
    --min-px      minimum instance-mask pixels for an object to count as
                  visible (default 200; lower it for small objects)

OUTPUT
    Per-frame and total counts of scoreable GT edges, split into
    support/structural+proximity vs directional; per-predicate breakdown.
    Read-only; writes nothing.

USAGE
    conda activate ram_pipeline
    python count_support_gt.py --dssg-root /data/mmc-user/3dssg \
        --pairs <scan>:6 <scan>:30 <scan>:41 <scan>:75 <scan>:110 <scan>:115
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger("count_support_gt")

# Scored spatial-predicate subset, split by whether pose recomputation is required.
SUPPORT_PREDICATES = {
    "standing on", "attached to", "lying on", "hanging on", "supported by",
    "standing in", "lying in", "leaning against", "connected to", "part of",
    "build in", "cover", "hanging in", "belonging to",
    "close by",  # proximity — viewpoint-invariant, needs no pose
}
DIRECTIONAL_PREDICATES = {
    "left", "right", "front", "behind", "higher than", "lower than",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Count scoreable GT edges (support vs directional) for "
                    "handpicked 3DSSG frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    p.add_argument("--data-subdir", default="3dssg_data")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="scan_id:frame_no, one per handpicked frame")
    p.add_argument("--min-px", type=int, default=200)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-7s %(message)s")

    root = Path(args.dssg_root)
    rel_path = root / "relationships.json"
    try:
        rel_data = json.loads(rel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("cannot read %s: %s", rel_path, exc)
        return 2
    rel_by_scan = {e.get("scan"): e.get("relationships", [])
                   for e in rel_data.get("scans", [])}

    pairs = []
    for spec in args.pairs:
        try:
            scan, fno = spec.rsplit(":", 1)
            pairs.append((scan, int(fno)))
        except ValueError:
            LOG.error("bad --pairs entry %r — expected scan_id:frame_no", spec)
            return 2

    grand_support = Counter()
    grand_dir = Counter()
    grand_excluded = Counter()
    total_support = total_dir = 0

    print()
    print(f"{'scan[:8]':<10s} {'frame':>6s} {'vis_obj':>7s} "
          f"{'support':>8s} {'direct.':>8s} {'excluded':>9s}")
    print("-" * 55)
    for scan, fno in pairs:
        seq = root / args.data_subdir / scan / "sequence"
        inst_path = seq / f"frame-{fno:06d}.rendered.instances.png"
        if not inst_path.exists():
            LOG.error("missing %s — check scan id / frame no (ls %s | head)",
                      inst_path.name, seq)
            return 2
        if scan not in rel_by_scan:
            LOG.error("scan %s not in relationships.json", scan)
            return 2

        inst = np.array(Image.open(inst_path))
        ids, counts = np.unique(inst, return_counts=True)
        visible = {int(i) for i, c in zip(ids, counts)
                   if int(i) != 0 and c >= args.min_px}

        n_sup = n_dir = n_exc = 0
        for row in rel_by_scan[scan]:
            s, o, _pid, pred = row[0], row[1], row[2], str(row[3]).lower()
            if s not in visible or o not in visible:
                continue
            if pred in SUPPORT_PREDICATES:
                n_sup += 1
                grand_support[pred] += 1
            elif pred in DIRECTIONAL_PREDICATES:
                n_dir += 1
                grand_dir[pred] += 1
            else:
                n_exc += 1
                grand_excluded[pred] += 1
        total_support += n_sup
        total_dir += n_dir
        print(f"{scan[:8]:<10s} {fno:>6d} {len(visible):>7d} "
              f"{n_sup:>8d} {n_dir:>8d} {n_exc:>9d}")

    print("-" * 55)
    print(f"{'TOTAL':<25s} {total_support:>8d} {total_dir:>8d}")
    print()
    print("support/proximity breakdown:",
          dict(grand_support.most_common()) or "(none)")
    print("directional breakdown:      ",
          dict(grand_dir.most_common()) or "(none)")
    print("top excluded (outside the scored subset):  ",
          dict(grand_excluded.most_common(6)) or "(none)")
    print()
    if total_support < 15:
        print(f"⚠ STARVED: only {total_support} support-subset edge(s) across all "
              f"frames. A support-only evaluation cannot produce a reportable "
              f"number at this n — directional predicates are no longer "
              f"optional, OR pick different/more frames.")
    else:
        print(f"Support-only is viable: n = {total_support} edges. The "
              f"time-box decision rule stands — attempt V3 once; on ambiguity, "
              f"fall back to support-only with this n.")
    # NOTE: counts are per-frame occurrences; the same GT edge visible in two
    # frames counts twice. Report as "edge-frame instances", or dedupe when
    # consumed in eval/ — state which convention is used.
    return 0


if __name__ == "__main__":
    sys.exit(main())