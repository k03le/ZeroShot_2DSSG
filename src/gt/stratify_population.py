"""
Script:  stratify_population.py
Stage:   Independent re-derivation of the 216-item canonical GT population, tagged with its stratum, gated on a
         PRE-REGISTERED fingerprint.

Purpose: (1) Prove the strata.py rule reproduces 35/62/45/68/5/1 = 216.
         (2) Emit the JOIN SPINE for every later per-stratum table: one row
             per item carrying GT IDs, which the existing triplet_recall_at_k_baseline.py
             detail records do NOT carry (their labels-only
             key collides -- e.g. six pillows in 000238).
         This script scores nothing and loads no encoder.

Input:   --frames-json ingest_identify.json (frames with ok=true)
         --dssg-root   dir holding relationships.json + objects.json
         --data-subdir subdir holding <scan>/sequence/*.rendered.instances.png
Output:  --json-out  absolute path; population table + control block.
         READ-ONLY except --json-out.

Usage:
    conda activate ram_pipeline
    python stratify_population.py \
        --frames-json /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json \
        --dssg-root /data/mmc-user/3dssg \
        --json-out /data/mmc-user/outputs/population_table.json
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from strata import (STRATA, STRATUM_MEANING, ARCH_KEYS, PREDICATE_VOCAB,
                    EXPECTED_STRATA, EXPECTED_PREDICATE_HISTOGRAM,
                    stratum_of, arch_trigger)

LOG = logging.getLogger("stratify_population")

VOCAB_SET = set(PREDICATE_VOCAB)
SYMMETRIC = {"close by", "same as"}
INVERSE = {"left": "right", "right": "left",
           "higher than": "lower than", "lower than": "higher than"}


def canonical_gt(s_id, o_id, pred):
    """VERBATIM copy of predicate_recall_at_k_baseline.py / triplet_recall_at_k_baseline.py canonical_gt
    (its three canonicalisation regimes). Copied, not imported, so this stays
    an INDEPENDENT path."""
    if pred not in VOCAB_SET:
        return None
    if pred in SYMMETRIC:
        return (min(s_id, o_id), max(s_id, o_id), pred)
    if pred in INVERSE:
        if s_id <= o_id:
            return (s_id, o_id, pred)
        return (o_id, s_id, INVERSE[pred])
    return (s_id, o_id, pred)


def visible_ids(mask_path, min_px):
    """Same visibility rule as derive_gt_boxes(): id != 0 and area >= min_px.
    Boxes are not needed here, only membership."""
    mask = np.asarray(Image.open(mask_path))
    ids, counts = np.unique(mask, return_counts=True)
    return {int(i) for i, c in zip(ids, counts) if i != 0 and c >= min_px}


def atomic_write_json(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)   # G-row: object_recall_at_k.py crashed for want of this
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    os.replace(tmp, str(p))


def main():
    ap = argparse.ArgumentParser(
        description="Stratum-tagged canonical GT population + control gate.")
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--min-px", type=int, default=200,
                    help="EVALUATION visibility threshold. NOT the 400 used "
                         "internally by upright_projection_direction_check.py "
                         "-- never merge the two.")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-control-check", action="store_true",
                    help="write output even if the pre-registered fingerprint "
                         "fails. Records control.bypassed=true. Use only if "
                         "the RECORDED counts are what is wrong.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = ap.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(levelname)s %(message)s")

    if a.min_px != 200:
        LOG.warning("min_px=%d: the evaluation population is defined at 200.",
                    a.min_px)

    out = Path(a.json_out)
    if out.exists() and not a.overwrite:
        raise SystemExit("[exists] {0} -- pass --overwrite to replace".format(out))

    root = Path(a.dssg_root)
    try:
        rels_raw = json.loads((root / "relationships.json").read_text())
        objs_raw = json.loads((root / "objects.json").read_text())
        rep = json.loads(Path(a.frames_json).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit("[io] cannot read inputs: {0}".format(exc))

    rels_by_scan = {s["scan"]: s["relationships"] for s in rels_raw["scans"]}
    labels_by_scan = {s["scan"]: {int(o["id"]): o.get("label", "?")
                                  for o in s["objects"]}
                      for s in objs_raw["scans"]}
    frames = [f for f in rep["frames"] if f.get("ok")]
    LOG.info("frames ok=%d", len(frames))

    items = []
    arch_hits = Counter()
    for f in tqdm(sorted(frames, key=lambda r: (r["scan"], r["frame_no"])),
                  desc="frames"):
        scan, no = f["scan"], f["frame_no"]
        key = "{0}_{1:06d}".format(scan, no)
        lab = labels_by_scan.get(scan, {})
        mpath = (root / a.data_subdir / scan / "sequence" /
                 "frame-{0:06d}.rendered.instances.png".format(no))
        try:
            vis = visible_ids(mpath, a.min_px)
        except OSError as exc:
            raise SystemExit(
                "[io] mask unreadable: {0}\n"
                "      check --data-subdir (given {1!r}); error: {2}"
                .format(mpath, a.data_subdir, exc))

        seen = set()
        for e in rels_by_scan.get(scan, []):
            s_id, o_id, _pid, pname = int(e[0]), int(e[1]), e[2], str(e[3])
            if s_id not in vis or o_id not in vis:
                continue
            can = canonical_gt(s_id, o_id, pname)
            if can is None or can in seen:
                continue
            seen.add(can)
            cs, co, cp = can
            sl, ol = lab.get(cs, "?"), lab.get(co, "?")
            st = stratum_of(cp, sl, ol)
            # Strata: S1 = support between two ordinary objects, S2 = support
            # where one endpoint is architecture (floor/wall/ceiling),
            # S3 = proximity ("close by"), S4 = lateral (left / right),
            # S5 = vertical (higher than / lower than), SAME_AS = same-instance
            # identity.
            if st == "S2":
                for side, lb in (("subject", sl), ("object", ol)):
                    k = arch_trigger(lb)
                    if k:
                        arch_hits["{0}|{1}|{2}".format(k, side, lb)] += 1
            items.append(dict(frame_key=key, scan=scan, frame_no=no,
                              subject_id=cs, object_id=co, gt_predicate=cp,
                              subject_label=sl, object_label=ol, stratum=st))

    # ---- control block: pre-registered, evaluated AFTER the build ----
    strat_counts = Counter(i["stratum"] for i in items)
    pred_hist = Counter(i["gt_predicate"] for i in items)

    id_key = Counter((i["frame_key"], i["subject_id"], i["object_id"],
                      i["gt_predicate"]) for i in items)
    label_key = Counter((i["frame_key"], i["subject_label"],
                         i["gt_predicate"], i["object_label"]) for i in items)
    n_id_collisions = sum(v - 1 for v in id_key.values() if v > 1)
    n_label_collisions = sum(v - 1 for v in label_key.values() if v > 1)

    checks = {
        "n_population_is_216": len(items) == 216,
        "strata_match_expected": dict(strat_counts) == EXPECTED_STRATA,
        "predicate_histogram_matches_preregistered":
            dict(pred_hist) == EXPECTED_PREDICATE_HISTOGRAM,
        "id_key_unique": n_id_collisions == 0,
    }
    passed = all(checks.values())

    print("\n=== POPULATION ===")
    print("  items: {0}".format(len(items)))
    print("\n=== STRATA (got vs pre-registered) ===")
    for st in STRATA:
        print("  {0:<8} {1:>4}  expected {2:>4}   {3}".format(
            st, strat_counts.get(st, 0), EXPECTED_STRATA[st],
            STRATUM_MEANING[st]))
    print("\n=== PREDICATE HISTOGRAM (got vs pre-registered) ===")
    for p in sorted(set(pred_hist) | set(EXPECTED_PREDICATE_HISTOGRAM)):
        g, e = pred_hist.get(p, 0), EXPECTED_PREDICATE_HISTOGRAM.get(p, 0)
        print("  {0:<14} {1:>4}  expected {2:>4}{3}".format(
            p, g, e, "" if g == e else "   <-- MISMATCH"))
    print("\n=== ARCH_KEYS triggers on S2 (audit the substring rule) ===")
    for k, v in sorted(arch_hits.items(), key=lambda kv: -kv[1]):
        print("  {0:>4}  {1}".format(v, k))
    print("\n=== JOIN-KEY UNIQUENESS ===")
    print("  (frame,subj_id,obj_id,pred) collisions : {0}".format(n_id_collisions))
    print("  (frame,subj_lbl,pred,obj_lbl) collisions: {0}"
          "   <-- items unreachable via existing detail JSONs"
          .format(n_label_collisions))
    print("\n=== CONTROL ===")
    for k, v in checks.items():
        print("  {0:<34} {1}".format(k, "PASS" if v else "FAIL"))

    if not passed and not a.no_control_check:
        print("\n[abort] pre-registered fingerprint FAILED. Nothing written.")
        print("        Do not relax the criterion -- diagnose the rule.")
        return 2

    atomic_write_json(out, dict(
        scope="3DSSG subset (4 scans / 17 frames)",
        min_px=a.min_px,
        rule=dict(arch_keys=list(ARCH_KEYS),
                  stratum_meaning=STRATUM_MEANING,
                  partition="exhaustive over the 20 scored predicates"),
        n_population=len(items),
        stratum_counts=dict(strat_counts),
        predicate_histogram=dict(pred_hist),
        arch_key_triggers=dict(arch_hits),
        join_key=dict(fields=["frame_key", "subject_id", "object_id",
                              "gt_predicate"],
                      id_key_collisions=n_id_collisions,
                      label_key_collisions=n_label_collisions),
        control=dict(expected_strata=EXPECTED_STRATA,
                     expected_predicate_histogram=EXPECTED_PREDICATE_HISTOGRAM,
                     checks=checks, passed=passed,
                     bypassed=bool(a.no_control_check and not passed)),
        items=items))
    print("\nwrote {0}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
