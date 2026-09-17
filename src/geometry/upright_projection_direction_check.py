"""
Script:  upright_projection_direction_check.py
Gate:    Data-touching verification of the pose-direction x rotation-handling
         directional composition.
Purpose: On ONE upright 3DSSG frame, verify that the composed mapping

           world --(inv(pose), assuming pose=Twc, camera-to-world)--> camera
                 --(OpenCV intrinsics)--> stored (u,v)
                 --(rotation.json np_rot90_k)--> upright (u',v')

         (CHECK 1) lands each object's OBB centroid inside that object's own
                   instance-mask region in the UPRIGHT frame, and
         (CHECK 2) satisfies the directional formula the eval/ pipeline implements:
                   u_upright ~ -y_cam   =>   "A right of B" <=> y_cam(A) < y_cam(B),
                   anchored on the MASK-derived left/right ordering (independent of
                   the projection under test -- the mask is rotated mechanically via
                   np.rot90, never via the formula).

         Reads ONLY primary data (3dssg_data/<scan>/{semseg.v2.json, sequence/}) plus
         rotation.json. Deliberately does NOT trust outputs/3dssg_upright/instances/
         (that tree is the retracted 3dssg_upright tree, not 3dssg_upright_v2, and its
         rotation correctness was never independently verified).

Input:   --dssg-root/--data-subdir  -> <root>/<subdir>/<scan>/semseg.v2.json
                                       <root>/<subdir>/<scan>/sequence/frame-%06d.pose.txt
                                       <root>/<subdir>/<scan>/sequence/frame-%06d.rendered.instances.png
                                       <root>/<subdir>/<scan>/sequence/_info.txt   (intrinsics)
         --rotation-json            -> per-frame np_rot90_k + stored/upright sizes
Output:  console PASS/FAIL verdict (exit 0 pass, 2 fail, 3 input error);
         optional --json-out (atomic write) with this script's own results;
         optional --viz-out upright JPEG with projected centroids drawn, for the
         mandated eyeball confirmation ("what the image plainly shows").
Usage:
    conda activate ram_pipeline   # needs numpy + PIL only
    python upright_projection_direction_check.py \
        --scan 5341b7bf-8a66-2cdd-8794-026113b7c312 --frame 6 \
        --json-out /data/mmc-user/outputs/g10/g10_5341b7bf_000006.json \
        --viz-out  /data/mmc-user/outputs/g10/g10_5341b7bf_000006.jpg
    # then open the .jpg and confirm the highlighted pair matches the printed verdict.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger("upright_proj_check")

# ---------------------------------------------------------------- rotation ---
# stored pixel (u, v) -> upright pixel (u', v'), for a stored image of size (W, H).
# k is the np.rot90 k that make_upright applied to the ARRAY (rotation.json: np_rot90_k).
ROT_FORMULAS = {
    -1: lambda u, v, W, H: (H - 1 - v, u),        # cw  (u' = H-1-v, v' = u)
    1:  lambda u, v, W, H: (v, W - 1 - u),        # ccw
}


def _selftest_rot(k: int) -> None:
    """Assert the closed-form pixel map agrees with np.rot90 on a marked array."""
    H, W = 3, 5
    a = np.zeros((H, W), dtype=int)
    u, v = 4, 1                       # col=4, row=1
    a[v, u] = 7
    b = np.rot90(a, k)
    up, vp = ROT_FORMULAS[k](u, v, W, H)
    if b[vp, up] != 7:
        raise AssertionError(
            f"rotation self-test FAILED for k={k}: formula ({up},{vp}) does not "
            f"match np.rot90 -- do not trust any output of this run")
    LOG.debug("rotation self-test passed for k=%d", k)


# ------------------------------------------------------------------ parsing ---
def load_pose_twc(path: Path) -> np.ndarray:
    """frame-%06d.pose.txt: 4x4 row-major camera-to-world (Twc)."""
    try:
        vals = [float(x) for x in path.read_text().split()]
    except OSError as e:
        raise SystemExit(f"[input error] cannot read pose {path}: {e}")
    if len(vals) != 16:
        raise SystemExit(f"[input error] pose {path}: expected 16 floats, got {len(vals)}")
    T = np.array(vals, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 0.01:
        LOG.warning("pose rotation det=%.4f (expected ~1) -- pose file suspicious", np.linalg.det(R))
    return T


def load_intrinsics(info_path: Path, args) -> tuple:
    """Parse _info.txt -> (fx, fy, cx, cy, W_color, H_color). CLI overrides win."""
    if all(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        LOG.info("intrinsics: using CLI overrides fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                 args.fx, args.fy, args.cx, args.cy)
        return args.fx, args.fy, args.cx, args.cy, None, None
    if not info_path.exists():
        raise SystemExit(
            f"[input error] {info_path} not found and no --fx/--fy/--cx/--cy given.\n"
            f"  Inspect the sequence dir:  ls {info_path.parent} | head")
    kv = {}
    for line in info_path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    key = "m_calibrationColorIntrinsic"
    if key not in kv:
        raise SystemExit(f"[input error] {key} missing in {info_path}; keys: {sorted(kv)}")
    m = [float(x) for x in kv[key].split()]
    if len(m) != 16:
        raise SystemExit(f"[input error] {key}: expected 16 values, got {len(m)}")
    fx, fy, cx, cy = m[0], m[5], m[2], m[6]
    Wc = int(float(kv.get("m_colorWidth", 0))) or None
    Hc = int(float(kv.get("m_colorHeight", 0))) or None
    LOG.info("intrinsics(_info.txt): fx=%.2f fy=%.2f cx=%.2f cy=%.2f color=%sx%s",
             fx, fy, cx, cy, Wc, Hc)
    return fx, fy, cx, cy, Wc, Hc


# --------------------------------------------------------------------- core ---
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify the pose-direction x rotation-handling directional composition on one upright frame.")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg")
    ap.add_argument("--data-subdir", default="3dssg_data")
    ap.add_argument("--rotation-json",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json",
                    help="per-frame rotation source: the 3dssg_upright_v2 ingest_identify.json "
                         "(default: /data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json); "
                         "a legacy rotation.json {<scan>_<NNNNNN>: {np_rot90_k, upright_size_wh}} "
                         "is also accepted")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--objects", type=int, nargs=2, default=None,
                    help="two objectIds for CHECK 2; default: auto-pick the widest-separated pair")
    ap.add_argument("--min-px", type=int, default=400,
                    help="min visible mask pixels for an object to be considered")
    ap.add_argument("--win", type=int, default=9,
                    help="odd window side for the mask-majority test around the projected point")
    ap.add_argument("--fx", type=float); ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float); ap.add_argument("--cy", type=float)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--viz-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")

    for out in (args.json_out, args.viz_out):
        if out and Path(out).exists() and not args.overwrite:
            raise SystemExit(f"[refusing] {out} exists; pass --overwrite to replace it")

    scan_dir = Path(args.dssg_root) / args.data_subdir / args.scan
    seq = scan_dir / "sequence"
    key = f"{args.scan}_{args.frame:06d}"

    # ---- rotation entry
    try:
        rot_raw = json.loads(Path(args.rotation_json).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[input error] cannot read {args.rotation_json}: {e}")
    # Two schemas accepted:
    #   (1) legacy rotation.json : {"<scan>_<NNNNNN>": {"np_rot90_k":k, "upright_size_wh":[W,H]}}
    #   (2) ingest_identify.json : {"frames":[{"scan":..,"frame_no":..,"k":..,"size":[W,H]}]}
    # (2) is the ONLY rotation source for 3dssg_upright_v2; (1)'s tree is RETRACTED.
    if isinstance(rot_raw, dict) and isinstance(rot_raw.get("frames"), list):
        rot_all = {f"{fr['scan']}_{int(fr['frame_no']):06d}":
                   {"np_rot90_k": int(fr["k"]), "upright_size_wh": list(fr["size"])}
                   for fr in rot_raw["frames"] if fr.get("ok", True)}
        LOG.info("rotation source: ingest_identify schema, %d frames", len(rot_all))
    else:
        rot_all = rot_raw
        LOG.warning("rotation source: LEGACY rotation.json schema -- confirm this is NOT "
                    "the retracted 3dssg_upright tree")
    if key not in rot_all:
        raise SystemExit(f"[input error] {key} not in {args.rotation_json} "
                         f"({len(rot_all)} keys, e.g. {list(rot_all)[:2]})")
    ent = rot_all[key]
    k = int(ent["np_rot90_k"])
    if k not in ROT_FORMULAS:
        raise SystemExit(f"[input error] unsupported np_rot90_k={k}")
    _selftest_rot(k)

    # ---- primary data
    try:
        semseg = json.loads((scan_dir / "semseg.v2.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[input error] semseg.v2.json: {e}")
    groups = {int(g["objectId"]): g for g in semseg["segGroups"]}

    mask_path = seq / f"frame-{args.frame:06d}.rendered.instances.png"
    color_path = seq / f"frame-{args.frame:06d}.color.jpg"
    try:
        mask_native = np.array(Image.open(mask_path))
    except OSError as e:
        raise SystemExit(f"[input error] instance mask {mask_path}: {e}")
    if mask_native.ndim != 2:
        raise SystemExit(f"[input error] mask is not single-channel (shape {mask_native.shape}), "
                         f"stop")
    # CORRECTED ROTATION HANDLING (2026-08-08): the rendered instance mask is NATIVELY UPRIGHT
    # (540x960) and takes k=0 -- read in place, no rotation. ONLY the photographic
    # colour frame (960x540) is rotated. The stored frame the intrinsics live in is
    # therefore the COLOUR frame, not the mask.
    try:
        color_stored = np.array(Image.open(color_path).convert("RGB"))
    except OSError as e:
        raise SystemExit(f"[input error] colour frame {color_path}: {e}")
    H, W = color_stored.shape[:2]          # stored (landscape) frame of the intrinsics
    mask_up = mask_native                  # k=0, per the corrected rotation handling above
    Hu, Wu = mask_up.shape
    rot_shape = np.rot90(color_stored, k).shape[:2]
    if (Hu, Wu) != rot_shape:
        raise SystemExit(
            f"[fail] upright-rotation invariant violated: rot90(colour, k={k}) -> {rot_shape} "
            f"but the instance mask is {(Hu, Wu)}. Colour and mask are not in "
            f"complementary orientations; stop before trusting any verdict.")
    LOG.info("stored colour %dx%d -> upright %dx%d ; mask native upright %dx%d "
             "dtype=%s n_ids=%d", W, H, Wu, Hu, Wu, Hu, mask_native.dtype,
             len(np.unique(mask_native)))

    T = load_pose_twc(seq / f"frame-{args.frame:06d}.pose.txt")
    fx, fy, cx, cy, Wc, Hc = load_intrinsics(seq / "_info.txt", args)
    if Wc and (Wc, Hc) != (W, H):                       # _info.txt colour res != on-disk colour res
        sx, sy = W / Wc, H / Hc
        LOG.warning("mask %dx%d != color %dx%d -> scaling intrinsics by (%.3f, %.3f)",
                    W, H, Wc, Hc, sx, sy)
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy

    Tinv = np.linalg.inv(T)
    # mask_up was bound above (k=0, per the corrected rotation handling) -- deliberately NOT rotated here.
    if [Wu, Hu] != list(ent["upright_size_wh"]):
        LOG.warning("upright mask %dx%d != rotation-json upright_size_wh %s",
                    Wu, Hu, ent["upright_size_wh"])

    # ---- per-object projection through the composition under test
    rows, half = [], args.win // 2
    for oid, g in sorted(groups.items()):
        if oid == 0:
            continue
        vis = mask_up == oid
        area = int(vis.sum())
        if area < args.min_px:
            continue
        c = np.asarray(g["obb"]["centroid"], dtype=np.float64)
        pc = Tinv @ np.append(c, 1.0)
        x, y, z = pc[:3]
        if z <= 0.05:
            LOG.debug("id %d (%s): behind camera (z_cam=%.3f) -- skipped", oid, g["label"], z)
            continue
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (0 <= u < W and 0 <= v < H):
            LOG.info("id %d (%-14s): projects outside stored frame (u=%.0f v=%.0f) -- "
                     "skipped (large-object centroid)",
                     oid, g["label"], u, v)
            continue
        uu, vu = ROT_FORMULAS[k](u, v, W, H)            # the rotation half of the composition
        ui, vi = int(round(uu)), int(round(vu))
        patch = mask_up[max(0, vi - half):vi + half + 1, max(0, ui - half):ui + half + 1]
        winfrac = float((patch == oid).mean()) if patch.size else 0.0
        ys, xs = np.nonzero(vis)
        dist = float(np.min(np.hypot(xs - uu, ys - vu)))
        rows.append(dict(objectId=oid, label=g["label"], area_px=area,
                         y_cam=round(float(y), 4), z_cam=round(float(z), 3),
                         u_stored=round(float(u), 1), v_stored=round(float(v), 1),
                         u_upright=round(float(uu), 1), v_upright=round(float(vu), 1),
                         u_mask=round(float(xs.mean()), 1), v_mask=round(float(ys.mean()), 1),
                         window_frac=round(winfrac, 3), dist_px=round(dist, 1),
                         check1=bool(winfrac >= 0.5 or dist <= 8.0)))

    if not rows:
        raise SystemExit("[fail] no visible objects passed the filters -- try another frame")

    print(f"\n== {key}  (np_rot90_k={k}, stored {W}x{H} -> upright {Wu}x{Hu}) ==")
    print(f"{'id':>4} {'label':16} {'area':>7} {'y_cam':>7} {'u_up(proj)':>10} "
          f"{'u_up(mask)':>10} {'win%':>6} {'dist':>6}  Check-1")
    for r in sorted(rows, key=lambda r: r["u_mask"]):
        print(f"{r['objectId']:>4} {r['label'][:16]:16} {r['area_px']:>7} {r['y_cam']:>7.3f} "
              f"{r['u_upright']:>10.1f} {r['u_mask']:>10.1f} {r['window_frac']:>6.2f} "
              f"{r['dist_px']:>6.1f}  {'PASS' if r['check1'] else 'FAIL'}")

    n_pass = sum(r["check1"] for r in rows)
    check1_ok = n_pass >= 2 and n_pass >= 0.6 * len(rows)

    # ---- CHECK 2: sign formula vs MASK-anchored left/right
    pool = [r for r in rows if r["check1"]] or rows
    if args.objects:
        sel = {r["objectId"]: r for r in rows}
        missing = [o for o in args.objects if o not in sel]
        if missing:
            raise SystemExit(f"[input error] --objects {missing} not among visible/scored ids")
        A, B = sel[args.objects[0]], sel[args.objects[1]]
    else:
        pool = sorted(pool, key=lambda r: r["u_mask"])
        A, B = pool[0], pool[-1]                        # extreme left / extreme right by MASK
    left, right = (A, B) if A["u_mask"] < B["u_mask"] else (B, A)
    sep = abs(right["u_mask"] - left["u_mask"])
    if sep < 0.15 * Wu:
        LOG.warning("pair separation %.0f px < 15%% of width -- not 'unmistakable'; "
                    "re-run with --objects on a clearer pair", sep)
    check2_ok = right["y_cam"] < left["y_cam"]
    print(f"\nCHECK1 (centroid lands in own mask): {n_pass}/{len(rows)} objects -> "
          f"{'PASS' if check1_ok else 'FAIL'}")
    print(f"CHECK2 pair: LEFT  = id {left['objectId']} '{left['label']}'  "
          f"(u_mask={left['u_mask']:.0f}, y_cam={left['y_cam']:+.3f})")
    print(f"             RIGHT = id {right['objectId']} '{right['label']}' "
          f"(u_mask={right['u_mask']:.0f}, y_cam={right['y_cam']:+.3f})")
    print(f"CHECK2 (right-of <=> smaller y_cam):  {'PASS' if check2_ok else 'FAIL'}")
    verdict = "PASS" if (check1_ok and check2_ok) else "FAIL"
    print(f"\nUPRIGHT PROJECTION / DIRECTION CHECK VERDICT: {verdict}   (eyeball the --viz-out image before logging)\n")

    # ---- viz
    if args.viz_out:
        try:
            col = np.array(Image.open(color_path).convert("RGB"))
            if col.shape[:2] != (H, W):
                col = np.array(Image.fromarray(col).resize((W, H), Image.NEAREST))
            im = Image.fromarray(np.rot90(col, k).copy())
            dr = ImageDraw.Draw(im)
            for r in rows:
                hot = r["objectId"] in (left["objectId"], right["objectId"])
                fill = (255, 40, 40) if hot else (255, 220, 0)
                x0, y0 = r["u_upright"], r["v_upright"]
                dr.ellipse([x0 - 7, y0 - 7, x0 + 7, y0 + 7], outline=fill, width=3)
                dr.text((x0 + 9, y0 - 7), f"{r['objectId']}:{r['label']}", fill=fill)
            Path(args.viz_out).parent.mkdir(parents=True, exist_ok=True)
            vp = Path(args.viz_out)
            tmp = vp.with_name(f".{vp.stem}.tmp{os.getpid()}{vp.suffix}")
            im.save(tmp, "JPEG", quality=92); os.replace(tmp, vp)
            LOG.info("viz written: %s", args.viz_out)
        except (OSError, ValueError) as e:
            LOG.error("viz failed (verdict unaffected): %s", e)

    # ---- json (atomic)
    if args.json_out:
        # NOTE: the default output directory for this script is outputs/g10/
        # (see the module docstring's Usage example) -- that directory name
        # predates the "gate" field value below and is not renamed here.
        payload = dict(gate="upright_projection_direction", key=key, np_rot90_k=k,
                       stored_wh=[W, H], upright_wh=[Wu, Hu],
                       intrinsics=dict(fx=fx, fy=fy, cx=cx, cy=cy),
                       objects=rows,
                       pair=dict(left=left["objectId"], right=right["objectId"],
                                 sep_px=round(sep, 1)),
                       check1=dict(n_pass=n_pass, n=len(rows), ok=check1_ok),
                       check2=dict(ok=check2_ok), verdict=verdict)
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, args.json_out)
        LOG.info("json written: %s", args.json_out)

    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())