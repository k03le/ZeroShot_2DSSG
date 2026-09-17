"""
Script:  dir_rule_divergence.py
Purpose: Quantify + visualise the disagreement between two candidate rules for
         per-frame directional GT (part of the directional-predicate evaluation
         work, flagged as an open question):
           R2: object position = 3D OBB centroid projected through the composition
               verified by upright_projection_direction_check.py
               (world -> inv(pose) -> intrinsics -> np_rot90_k) into
               upright pixels. May legitimately fall outside the frame.
           R3: object position = visible-extent centroid of the instance mask
               (read in place, natively upright, using the corrected rotation
               handling). No pose/intrinsics.
         For every visible object pair, compare the left/right (u) and
         higher/lower (v) orderings under R2 vs R3. front/behind is R2-only
         (a 2D mask centroid has no depth) and is reported, not compared.
Input:   3dssg_data/<scan>/{semseg.v2.json, sequence/*}, ingest_identify.json,
         upright colour from 3dssg_upright_v2 (NEVER the retracted tree).
Output:  console table; --json-out stats (atomic); --viz-out divergence portrait
         (yellow dot = R3 mask centroid, red dot = R2 projection, grey tether).
Usage:   python dir_rule_divergence.py --scan <uuid> --frame N \
             --json-out .../f.json --viz-out .../f.jpg [--min-px 200]
"""
import argparse, json, logging, os, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger("dirrule")
ROT = {-1: lambda u, v, W, H: (H - 1 - v, u),
        1: lambda u, v, W, H: (v, W - 1 - u)}

# A centroid summarises an object's apparent position only if that object HAS a
# single apparent position. Containment is NOT the test (a concave but connected
# region legitimately excludes its own mean); DOMINANCE is: what fraction of the
# visible pixels lie in the largest connected component.
FRAG_THRESHOLD = 0.70
try:
    from scipy import ndimage as _ndi
except ImportError:
    _ndi = None


def mask_components(m_bbox):
    """(n_components, largest_fraction) over a bbox-cropped boolean array."""
    if _ndi is None:
        raise SystemExit("[input error] scipy required for component analysis; "
                         "check: python -c 'import scipy; print(scipy.__version__)'")
    lab, n = _ndi.label(m_bbox)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())

def main():
    ap = argparse.ArgumentParser(description="R2 vs R3 directional-GT rule divergence, one frame.")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg/3dssg_data")
    ap.add_argument("--upright-root",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training")
    ap.add_argument("--rotation-json",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--min-px", type=int, default=200,
                    help="visibility threshold -- matches the value used across the "
                         "evaluation scripts (200), NOT the 400 used internally by "
                         "upright_projection_direction_check.py")
    ap.add_argument("--json-out"); ap.add_argument("--viz-out")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(levelname)s %(name)s: %(message)s")
    for o in (a.json_out, a.viz_out):
        if o and Path(o).exists() and not a.overwrite:
            raise SystemExit(f"[refusing] {o} exists; pass --overwrite")

    scan_dir = Path(a.dssg_root) / a.scan
    seq = scan_dir / "sequence"
    # rotation k (ingest_identify schema only -- the legacy tree is retracted)
    ing = json.loads(Path(a.rotation_json).read_text())
    ent = next((f for f in ing["frames"]
                if f["scan"] == a.scan and int(f["frame_no"]) == a.frame), None)
    if ent is None:
        raise SystemExit(f"[input error] {a.scan}/{a.frame} not in {a.rotation_json}")
    k = int(ent["k"])
    if k not in ROT:
        raise SystemExit(f"[input error] unsupported k={k}")

    semseg = json.loads((scan_dir / "semseg.v2.json").read_text())
    groups = {int(g["objectId"]): g for g in semseg["segGroups"]}
    mask = np.array(Image.open(seq / f"frame-{a.frame:06d}.rendered.instances.png"))
    if mask.ndim != 2:
        raise SystemExit("[input error] mask must be single-channel (one object id per pixel)")
    Hu, Wu = mask.shape                                   # natively upright, corrected rotation handling

    # intrinsics + stored colour geometry, per scan
    kv = {}
    for line in (seq / "_info.txt").read_text().splitlines():
        if "=" in line:
            key, _, val = line.partition("="); kv[key.strip()] = val.strip()
    m = [float(x) for x in kv["m_calibrationColorIntrinsic"].split()]
    fx, fy, cx, cy = m[0], m[5], m[2], m[6]
    W, H = int(float(kv["m_colorWidth"])), int(float(kv["m_colorHeight"]))
    LOG.info("scan intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f  stored %dx%d  k=%d",
             fx, fy, cx, cy, W, H, k)
    pose = [float(x) for x in (seq / f"frame-{a.frame:06d}.pose.txt").read_text().split()]
    if len(pose) != 16:
        raise SystemExit("[input error] pose: expected 16 floats")
    Tinv = np.linalg.inv(np.array(pose).reshape(4, 4))

    rows = []
    for oid in sorted(int(x) for x in np.unique(mask) if x != 0):
        ys, xs = np.nonzero(mask == oid)
        if xs.size < a.min_px or oid not in groups:
            continue
        g = groups[oid]
        um, vm = float(xs.mean()), float(ys.mean())
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        ncomp, lrg = mask_components((mask == oid)[y0:y1 + 1, x0:x1 + 1])
        ui, vi = int(round(um)), int(round(vm))
        inside = bool(mask[min(vi, Hu - 1), min(ui, Wu - 1)] == oid)
        r = dict(objectId=oid, label=g["label"], area_px=int(xs.size),
                 u_mask=um, v_mask=vm, u_proj=None, v_proj=None, z_cam=None,
                 ncomp=ncomp, lrg_frac=round(lrg, 3), contained=inside,
                 fragmented=bool(lrg < FRAG_THRESHOLD),
                 u_p05=float(np.percentile(xs, 5)), u_p95=float(np.percentile(xs, 95)))
        pc = Tinv @ np.append(np.asarray(g["obb"]["centroid"], float), 1.0)
        x, y, z = pc[:3]
        if z > 0.05:
            uu, vv = ROT[k](fx * x / z + cx, fy * y / z + cy, W, H)
            r.update(u_proj=float(uu), v_proj=float(vv), z_cam=float(z))
        else:
            LOG.warning("id %d (%s): z_cam=%.3f <= 0 -- R2 undefined here", oid, g["label"], z)
        rows.append(r)
    if len(rows) < 2:
        raise SystemExit("[input error] <2 visible objects at --min-px; nothing to compare")

    print(f"\n== dir-rule divergence  {a.scan[:8]}_{a.frame:06d}  "
          f"(n={len(rows)} objects, min_px={a.min_px}) ==")
    print(f"{'id':>4} {'label':16} {'area':>7} {'u_mask':>7} {'u_proj':>7} "
          f"{'|du|':>6} {'v_mask':>7} {'v_proj':>7} {'z_cam':>6} "
          f"{'ncomp':>5} {'lrg%':>5} {'in?':>4} {'FRAG':>5}")
    for r in sorted(rows, key=lambda r: r["u_mask"]):
        up = f"{r['u_proj']:7.1f}" if r["u_proj"] is not None else "   n/a "
        vp = f"{r['v_proj']:7.1f}" if r["v_proj"] is not None else "   n/a "
        du = (f"{abs(r['u_proj']-r['u_mask']):6.1f}" if r["u_proj"] is not None else "   n/a")
        zc = f"{r['z_cam']:6.2f}" if r["z_cam"] is not None else "  n/a "
        print(f"{r['objectId']:>4} {r['label'][:16]:16} {r['area_px']:>7} "
              f"{r['u_mask']:>7.1f} {up} {du} {r['v_mask']:>7.1f} {vp} {zc} "
              f"{r['ncomp']:>5} {r['lrg_frac']*100:>5.1f} "
              f"{'yes' if r['contained'] else 'NO':>4} "
              f"{'FRAG' if r['fragmented'] else '':>5}")
    nfrag = sum(r["fragmented"] for r in rows)
    frag_names = [str(r["objectId"]) + ":" + r["label"]
                  for r in rows if r["fragmented"]]
    tail = ("  -> " + ", ".join(frag_names)) if nfrag else ""
    print("\nfragmented (largest component < %d%% of visible px): %d/%d%s"
          % (int(FRAG_THRESHOLD * 100), nfrag, len(rows), tail))

    pairs, dis_u, dis_v = [], [], []
    for A, B in combinations([r for r in rows if r["u_proj"] is not None], 2):
        su_m = np.sign(A["u_mask"] - B["u_mask"]); su_p = np.sign(A["u_proj"] - B["u_proj"])
        sv_m = np.sign(A["v_mask"] - B["v_mask"]); sv_p = np.sign(A["v_proj"] - B["v_proj"])
        p = dict(a=A["objectId"], b=B["objectId"],
                 sep_u_mask=round(abs(A["u_mask"] - B["u_mask"]), 1),
                 sep_v_mask=round(abs(A["v_mask"] - B["v_mask"]), 1),
                 agree_u=bool(su_m == su_p), agree_v=bool(sv_m == sv_p),
                 frag=bool(A["fragmented"] or B["fragmented"]))
        pairs.append(p)
        if not p["agree_u"]: dis_u.append(p)
        if not p["agree_v"]: dis_v.append(p)
    lab = {r["objectId"]: r["label"] for r in rows}
    clean = [p for p in pairs if not p["frag"]]
    cu = [p for p in dis_u if not p["frag"]]
    cv = [p for p in dis_v if not p["frag"]]
    print(f"\npairs compared: {len(pairs)}   "
          f"LEFT/RIGHT flips: {len(dis_u)}   HIGHER/LOWER flips: {len(dis_v)}")
    print(f"  excluding pairs touching a fragmented object: {len(clean)} pairs, "
          f"{len(cu)} u-flips, {len(cv)} v-flips")
    for tag, lst in (("u (left/right)", dis_u), ("v (higher/lower)", dis_v)):
        for p in lst:
            print(f"  FLIP {tag}: {p['a']}:{lab[p['a']]} vs {p['b']}:{lab[p['b']]}  "
                  f"mask-sep u={p['sep_u_mask']} v={p['sep_v_mask']} px")

    if a.viz_out:
        img_p = (Path(a.upright_root) / a.scan / "lowres_wide" / f"{a.scan}_{a.frame:06d}.png")
        if not img_p.exists():
            LOG.error("upright colour missing: %s (stats unaffected)", img_p)
        else:
            im = Image.open(img_p).convert("RGB"); dr = ImageDraw.Draw(im)
            for r in rows:
                xm, ym = r["u_mask"], r["v_mask"]
                if r["u_proj"] is not None:
                    xp = min(max(r["u_proj"], -40), Wu + 40)   # clamp for drawability
                    yp = min(max(r["v_proj"], -40), Hu + 40)
                    dr.line([xm, ym, xp, yp], fill=(160, 160, 160), width=2)
                    dr.ellipse([xp - 6, yp - 6, xp + 6, yp + 6], outline=(255, 40, 40), width=3)
                dr.ellipse([xm - 6, ym - 6, xm + 6, ym + 6], outline=(255, 220, 0), width=3)
                tx = min(xm + 8, Wu - 90)                      # right-edge overflow lesson
                dr.text((tx, ym - 8), f"{r['objectId']}:{r['label']}", fill=(255, 220, 0))
            vp = Path(a.viz_out); vp.parent.mkdir(parents=True, exist_ok=True)
            tmp = vp.with_name(f".{vp.stem}.tmp{os.getpid()}{vp.suffix}")
            im.save(tmp, "JPEG", quality=92); os.replace(tmp, vp)
            LOG.info("viz written: %s", a.viz_out)

    if a.json_out:
        payload = dict(scan=a.scan, frame=a.frame, min_px=a.min_px, k=k,
                       intrinsics=dict(fx=fx, fy=fy, cx=cx, cy=cy),
                       objects=[{k2: (round(v2, 3) if isinstance(v2, float) else v2)
                                 for k2, v2 in r.items()} for r in rows],
                       pairs=pairs,
                       flips=dict(u=len(dis_u), v=len(dis_v), n_pairs=len(pairs)))
        jp = Path(a.json_out); jp.parent.mkdir(parents=True, exist_ok=True)
        tmp = jp.with_name(f".{jp.stem}.tmp{os.getpid()}{jp.suffix}")
        tmp.write_text(json.dumps(payload, indent=2)); os.replace(tmp, jp)
        LOG.info("json written: %s", a.json_out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
