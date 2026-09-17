"""
Script:  depth_axis_check.py
Purpose: Decide whether stored 3DSSG front/behind annotations are camera-frame
         (in which case per-pose recomputation, already used for the left/right
         axis, extends validly to this depth axis too) or a global/room
         convention (recomputation is the WRONG operation for this sub-axis).
         For every stored front/behind fact whose BOTH endpoints are visible in
         a frame, compare the stored direction against the R2 recomputation
         "A front B <=> z_cam(A) < z_cam(B)". Also records |dz| per pair to
         test the small-depth-separation hypothesis from 000115.
         Pre-registered decision rule: flip>=80% -> consistent non-camera
         convention, EXCLUDE front/behind from --directional (finding);
         flip<=20% -> 000115 was an outlier, SCORE front/behind;
         20-80% -> exclude on unreliability, report the |dz| split.

Input:   relationships.json (subject-first, verified), semseg.v2.json,
         per-frame pose.txt + _info.txt, ingest_identify.json (k),
         instance masks (visibility only, --min-px 200).
         
Output:  console table; --json-out (atomic).
Usage:   python depth_axis_check.py --json-out .../depth_axis.json
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger("depthaxis")


def main():
    ap = argparse.ArgumentParser(description="Stored vs z_cam front/behind, 17 frames")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg/3dssg_data")
    ap.add_argument("--relationships", default="/data/mmc-user/3dssg/relationships.json")
    ap.add_argument("--rotation-json",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json")
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(levelname)s %(name)s: %(message)s")
    if a.json_out and Path(a.json_out).exists() and not a.overwrite:
        raise SystemExit("[refusing] %s exists; pass --overwrite" % a.json_out)

    ing = json.loads(Path(a.rotation_json).read_text())["frames"]
    rel = {r["scan"]: r["relationships"]
           for r in json.loads(Path(a.relationships).read_text())["scans"]}
    groups = {}
    rows = []

    for fr in sorted(ing, key=lambda f: (f["scan"], int(f["frame_no"]))):
        scan, fno = fr["scan"], int(fr["frame_no"])
        sdir = Path(a.dssg_root) / scan
        seq = sdir / "sequence"
        if scan not in groups:
            ss = json.loads((sdir / "semseg.v2.json").read_text())
            groups[scan] = {int(g["objectId"]): g for g in ss["segGroups"]}
        grp = groups[scan]
        try:
            mask = np.array(Image.open(seq / ("frame-%06d.rendered.instances.png" % fno)))
            pose = np.array([float(x) for x in
                             (seq / ("frame-%06d.pose.txt" % fno)).read_text().split()])
        except OSError as e:
            LOG.error("%s/%06d: %s -- skipped", scan[:8], fno, e)
            continue
        if pose.size != 16:
            LOG.error("%s/%06d: pose != 16 floats -- skipped", scan[:8], fno)
            continue
        Ti = np.linalg.inv(pose.reshape(4, 4))
        ids, cts = np.unique(mask, return_counts=True)
        vis = {int(i) for i, c in zip(ids, cts) if i != 0 and c >= a.min_px}

        z = {}
        for oid in vis:
            if oid not in grp:
                continue
            p = Ti @ np.append(np.asarray(grp[oid]["obb"]["centroid"], float), 1.0)
            if p[2] > 0.05:
                z[oid] = float(p[2])

        # stored front/behind facts, canonicalised to ONE record per unordered
        # pair via the front member (every front record has a matching behind record)
        for s, o, _pid, p in rel.get(scan, []):
            if p != "front" or s == o or s not in z or o not in z:
                continue
            stored_front = s                       # subject-first: s is in front of o
            recomp_front = s if z[s] < z[o] else o
            rows.append(dict(scan=scan[:8], frame=fno, a=s, b=o,
                             a_label=grp[s]["label"], b_label=grp[o]["label"],
                             z_a=round(z[s], 3), z_b=round(z[o], 3),
                             dz=round(abs(z[s] - z[o]), 3),
                             agree=bool(stored_front == recomp_front)))

    if not rows:
        raise SystemExit("[result] zero visible front/behind facts -- nothing to decide")

    n = len(rows)
    flips = [r for r in rows if not r["agree"]]
    print("\n== stored front/behind vs z_cam recomputation, 17 frames ==")
    print("%-8s %6s %-16s %-16s %7s %7s %7s %6s" %
          ("scan", "frame", "front(stored)", "other", "z_front", "z_other", "|dz|", "flip"))
    for r in rows:
        print("%-8s %6d %-16s %-16s %7.2f %7.2f %7.2f %6s" %
              (r["scan"], r["frame"], ("%d:%s" % (r["a"], r["a_label"]))[:16],
               ("%d:%s" % (r["b"], r["b_label"]))[:16],
               r["z_a"], r["z_b"], r["dz"], "" if r["agree"] else "FLIP"))
    fr_ = len(flips) / n
    print("\nfacts compared: %d   flips: %d  = %.1f%%" % (n, len(flips), 100 * fr_))
    med = sorted(r["dz"] for r in rows)[n // 2]
    lo = [r for r in rows if r["dz"] <= med]
    hi = [r for r in rows if r["dz"] > med]
    print("|dz| split at median %.2f m:  small-dz flips %d/%d (%.0f%%)   "
          "large-dz flips %d/%d (%.0f%%)"
          % (med,
             sum(not r["agree"] for r in lo), len(lo),
             100 * sum(not r["agree"] for r in lo) / max(1, len(lo)),
             sum(not r["agree"] for r in hi), len(hi),
             100 * sum(not r["agree"] for r in hi) / max(1, len(hi))))
    verdict = ("CONSISTENT NON-CAMERA CONVENTION -> exclude front/behind, cite rate"
               if fr_ >= 0.8 else
               "CAMERA-FRAME HOLDS -> front/behind scoreable via z_cam"
               if fr_ <= 0.2 else
               "UNRELIABLE (mixed) -> exclude front/behind, report |dz| split")
    print("pre-registered verdict band: %s" % verdict)

    if a.json_out:
        jp = Path(a.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(n=n, flips=len(flips), flip_rate=round(fr_, 4),
                       median_dz=med, verdict=verdict, rows=rows)
        tmp = jp.with_name(".%s.tmp%d%s" % (jp.stem, os.getpid(), jp.suffix))
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, jp)
        LOG.info("json written: %s", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
