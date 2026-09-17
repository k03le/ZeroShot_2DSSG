#!/usr/bin/env python3
"""
verify_3dssg.py — dataset-integrity and camera-convention checks for 3DSSG.
Read-only. Pass --skip v3 to skip the slow one.

CHECKS
    V1  rendered.instances.png is still 16-bit. Transfer/unzip can silently
        recompress 16-bit PNGs to 8-bit, corrupting any objectId above 255.
        Passes if any pixel value exceeds 255 (proof 16 bits survived).

    V2  objects.json + relationships.json exist, parse, and cover all
        requested scans.

    V3  Camera pose convention, measured rather than assumed. pose.txt is
        assumed camera-to-world (Twc) from its homogeneous form, but this
        tests both hypotheses:

            H_twc : pose is Twc (camera->world)  =>  p_cam = pose^-1 . p_world
            H_tcw : pose is Tcw (world->camera)  =>  p_cam = pose   . p_world

        and, for each, finds which camera axis is horizontal / vertical /
        forward and its sign — no intrinsics needed:
          * forward: must match the instance mask's depth in absolute metres.
          * horizontal/vertical: must correlate with the mask's pixel
            centroid (column / row).
        A silent sign flip would invert every directional-predicate score,
        so this is measured rather than assumed.

INPUT
    --dssg-root  /data/mmc-user/3dssg  (3dssg_data/, objects.json,
                                        relationships.json)
    --scan / --frame  which frame V1 and V3 run on

OUTPUT
    stdout report; --json-out for a machine-readable summary. Writes nothing
    else.

USAGE
    python verify_3dssg.py --dssg-root /data/mmc-user/3dssg

    # drop floor/wall if they pollute the V3 axis fit (huge OBBs, centroid
    # far from the visible surface):
    python verify_3dssg.py --dssg-root /data/mmc-user/3dssg --exclude-ids 1 7

CAVEAT
    V3's forward check compares OBB centroid depth to visible-surface depth,
    so expect ~0.1-0.4 m offset (worst for walls/floors/ceilings, flagged
    automatically) — the axis is picked by correlation + order-of-magnitude
    agreement, not exact equality.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
from pathlib import Path

LOG = logging.getLogger("verify_3dssg")

DEFAULT_SCANS = [
    "0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca",
    "5341b7a5-8a66-2cdd-8751-70b98263cb8d",
    "5341b7bf-8a66-2cdd-8794-026113b7c312",
    "5630cfd8-12bf-2860-8773-e3dde9da2aff",
]
AXIS_NAMES = ("x", "y", "z")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("ERROR: numpy not importable. Confirm the env:\n"
          "  conda activate ram_pipeline && python -c 'import numpy; print(numpy.__version__)'",
          file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("ERROR: PIL not importable. Confirm the env:\n"
          "  conda activate ram_pipeline && python -c 'import PIL; print(PIL.__version__)'",
          file=sys.stderr)
    raise


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        LOG.warning("caught signal %s — exiting", signum)
        sys.exit(130)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _frame_stem(scan_dir: Path, frame_no: int) -> Path:
    return scan_dir / "sequence" / f"frame-{frame_no:06d}"


# ---------------------------------------------------------------- V1
def verify_instances_encoding(scan_dir: Path, frame_no: int) -> dict:
    _hr("V1 — instance-mask encoding on THIS machine (re-verification)")
    path = _frame_stem(scan_dir, frame_no).with_suffix(".rendered.instances.png")
    # .with_suffix on a stem containing dots is fragile; build explicitly:
    path = _frame_stem(scan_dir, frame_no).parent / (
        _frame_stem(scan_dir, frame_no).name + ".rendered.instances.png")

    res = {"check": "instances_uint16", "path": str(path), "passed": False}
    if not path.exists():
        LOG.error("not found: %s", path)
        LOG.error("list what is actually there:  ls %s | head -20", path.parent)
        res["error"] = "missing"
        return res

    try:
        img = Image.open(path)
        arr = np.array(img)
    except OSError as exc:
        LOG.error("could not open %s: %s", path, exc)
        res["error"] = str(exc)
        return res

    ids, counts = np.unique(arr, return_counts=True)
    over_255 = bool(ids.max() > 255)
    single_channel = arr.ndim == 2

    print(f"  path         : {path.name}")
    print(f"  PIL mode     : {img.mode}")
    print(f"  numpy dtype  : {arr.dtype}")
    print(f"  shape        : {arr.shape}")
    print(f"  unique ids   : {ids.tolist()[:24]}{' ...' if len(ids) > 24 else ''}")
    print(f"  max id       : {int(ids.max())}")
    print(f"  pixel share  : " + ", ".join(
        f"{int(i)}:{100.0 * c / arr.size:.2f}%" for i, c in list(zip(ids, counts))[:12]))
    print()

    if not single_channel:
        print("  FAIL — image is not single-channel. Expected a 2-D uint16 array;")
        print("         an RGB read means the PNG was recompressed or a palette applied.")
        res["error"] = "not_single_channel"
        return res

    if over_255:
        print("  PASS — pixel values above 255 are present, so 16-bit survived the")
        print("         transfer. Pixel value == objectId, no decode needed.")
        res["passed"] = True
    elif arr.dtype == np.uint16 or img.mode in ("I;16", "I"):
        print("  PASS — read back as 16-bit (dtype/mode). Every id in this frame is")
        print("         <= 255, so the stronger 'a pixel exceeds 255' proof is not")
        print("         available — and it is not available anywhere in this 4-scan")
        print("         subset, whose highest objectId is 251. Do not go looking for a")
        print("         frame that supplies it. An 8-bit recompression would still be")
        print("         caught here, as a uint8 read fails this check outright.")
        res["passed"] = True
        res["dtype_only"] = True
    else:
        print("  FAIL — 8-bit read. The mask no longer round-trips as written; any")
        print("         objectId above 255 would be mis-assigned. Re-transfer without")
        print("         recompression.")
        res["error"] = f"dtype={arr.dtype} mode={img.mode}"
    res["max_id"] = int(ids.max())
    res["dtype"] = str(arr.dtype)
    res["mode"] = img.mode
    return res


# ---------------------------------------------------------------- V2
def verify_split_gt(dssg_root: Path, scans: list[str]) -> dict:
    _hr("V2 — split-level GT files present, parseable, and covering the scans")
    res = {"check": "split_gt", "passed": False}

    out = {}
    for name in ("objects.json", "relationships.json"):
        p = dssg_root / name
        if not p.exists():
            LOG.error("missing: %s", p)
            LOG.error("check with:  ls -la %s", dssg_root)
            res["error"] = f"missing {name}"
            return res
        try:
            with p.open(encoding="utf-8") as fh:
                out[name] = json.load(fh)
        except json.JSONDecodeError as exc:
            LOG.error("%s does not parse as JSON: %s", p, exc)
            res["error"] = f"unparseable {name}"
            return res
        print(f"  {name:<20s} {p.stat().st_size / 1e6:8.1f} MB   top-level keys: "
              f"{list(out[name].keys())[:5]}")

    obj_scans = {e.get("scan"): e for e in out["objects.json"].get("scans", [])}
    rel_scans = {e.get("scan"): e for e in out["relationships.json"].get("scans", [])}
    print(f"\n  objects.json       covers {len(obj_scans)} scans")
    print(f"  relationships.json covers {len(rel_scans)} scans\n")

    print(f"  {'scan':<40s} {'objects':>8s} {'relations':>10s}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 10}")
    all_present = True
    coverage = {}
    for s in scans:
        in_o, in_r = s in obj_scans, s in rel_scans
        n_o = len(obj_scans[s].get("objects", [])) if in_o else 0
        n_r = len(rel_scans[s].get("relationships", [])) if in_r else 0
        flag = "" if (in_o and in_r) else "   <-- MISSING"
        print(f"  {s:<40s} {n_o:>8d} {n_r:>10d}{flag}")
        coverage[s] = {"objects": n_o, "relationships": n_r,
                       "in_objects": in_o, "in_relationships": in_r}
        all_present &= (in_o and in_r)

    print()
    if all_present:
        print("  PASS — all requested scans present in both split files.")
        print("         Cross-check these counts against the expected per-scan totals:")
        print("         0a4b8ef6 225/30, 5341b7a5 52/11, 5341b7bf 540/40, 5630cfd8 254/40")
        print("         (relationships/objects). A mismatch means a different split file.")
        res["passed"] = True
    else:
        print("  FAIL — at least one scan is absent. Do not proceed with the eval/ pipeline.")
    res["coverage"] = coverage
    return res


# ---------------------------------------------------------------- V3
def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def verify_camera_convention(
    scan_dir: Path, frame_no: int, exclude_ids: set[int], min_px: int,
) -> dict:
    _hr("V3 — pose direction and camera axis sign convention  (DIAGNOSTIC)")
    res = {"check": "camera_convention", "passed": False,
           "diagnostic": True, "verdict": "inconclusive"}

    stem = _frame_stem(scan_dir, frame_no)
    p_inst = stem.parent / (stem.name + ".rendered.instances.png")
    p_depth = stem.parent / (stem.name + ".rendered.depth.png")
    p_pose = stem.parent / (stem.name + ".pose.txt")
    p_semseg = scan_dir / "semseg.v2.json"

    for p in (p_inst, p_depth, p_pose, p_semseg):
        if not p.exists():
            LOG.error("missing: %s", p)
            LOG.error("list the sequence dir:  ls %s | head -20", stem.parent)
            res["error"] = f"missing {p.name}"
            return res

    pose = np.loadtxt(p_pose)
    if pose.shape != (4, 4):
        LOG.error("pose.txt is %s, expected (4,4)", pose.shape)
        res["error"] = "bad_pose_shape"
        return res

    R = pose[:3, :3]
    orth_err = float(np.abs(R @ R.T - np.eye(3)).max())
    det = float(np.linalg.det(R))
    last_row_ok = bool(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6))
    print(f"  pose.txt sanity:  det(R) = {det:+.6f}   |RR^T - I|_max = {orth_err:.2e}"
          f"   last row [0,0,0,1] = {last_row_ok}")
    if abs(det - 1.0) > 1e-3:
        print("    WARNING — det(R) is not +1. A determinant of -1 means a reflection")
        print("    (a left/right-handed frame swap), which would itself flip every")
        print("    directional prediction. Resolve this before anything else.")
    print()

    inst = np.array(Image.open(p_inst))
    depth_mm = np.array(Image.open(p_depth)).astype(np.float64)
    if depth_mm.shape != inst.shape:
        print(f"  WARNING — depth {depth_mm.shape} and instances {inst.shape} differ in")
        print("  resolution. Skipping the metric forward-axis check; only the image-space")
        print("  correlations below are usable.")
        depth_mm = None

    with p_semseg.open(encoding="utf-8") as fh:
        semseg = json.load(fh)
    centroids, labels = {}, {}
    for g in semseg.get("segGroups", []):
        oid = g.get("objectId")
        obb = g.get("obb") or {}
        c = obb.get("centroid")
        if oid is None or not c:
            continue
        centroids[int(oid)] = np.array([float(c[0]), float(c[1]), float(c[2]), 1.0])
        labels[int(oid)] = g.get("label", "?")

    H, W = inst.shape
    ids, counts = np.unique(inst, return_counts=True)

    obs = []
    for oid, cnt in zip(ids.tolist(), counts.tolist()):
        oid = int(oid)
        if oid == 0 or cnt < min_px or oid in exclude_ids:
            continue
        if oid not in centroids:
            LOG.warning("objectId %d in the mask has no semseg entry — skipped", oid)
            continue
        mask = inst == oid
        rows, cols = np.nonzero(mask)
        u_frac = float(cols.mean()) / W          # 0 = left edge, 1 = right edge
        v_frac = float(rows.mean()) / H          # 0 = top edge,  1 = bottom edge
        if depth_mm is not None:
            d = depth_mm[mask]
            d = d[d > 0]
            depth_m = float(np.median(d)) / 1000.0 if d.size else float("nan")
        else:
            depth_m = float("nan")
        obs.append({
            "id": oid, "label": labels.get(oid, "?"),
            "px_share": cnt / inst.size,
            "u_frac": u_frac, "v_frac": v_frac, "depth_m": depth_m,
            "world": centroids[oid],
        })

    if len(obs) < 3:
        print(f"  Only {len(obs)} usable object(s) — need at least 3 for a correlation.")
        print("  Pick a busier frame, or lower --min-px.")
        res["error"] = "too_few_objects"
        return res

    hypotheses = {
        "H_twc (pose is camera->world; p_cam = pose^-1 . p_world)": np.linalg.inv(pose),
        "H_tcw (pose is world->camera; p_cam = pose . p_world)": pose,
    }

    res["hypotheses"] = {}
    for hname, M in hypotheses.items():
        print("-" * 78)
        print(f"  {hname}")
        print("-" * 78)
        for o in obs:
            o["cam"] = (M @ o["world"])[:3]

        print(f"  {'id':>4s} {'label':<16s} {'px%':>6s} {'u':>6s} {'v':>6s} "
              f"{'depth_m':>8s} {'cam_x':>8s} {'cam_y':>8s} {'cam_z':>8s}")
        for o in sorted(obs, key=lambda d: d["u_frac"]):
            big = "  <-- large extent, centroid unreliable" if o["px_share"] > 0.25 else ""
            print(f"  {o['id']:>4d} {o['label'][:16]:<16s} {100 * o['px_share']:>5.1f}% "
                  f"{o['u_frac']:>6.3f} {o['v_frac']:>6.3f} {o['depth_m']:>8.3f} "
                  f"{o['cam'][0]:>8.3f} {o['cam'][1]:>8.3f} {o['cam'][2]:>8.3f}{big}")

        cues = {
            "u_frac (image left->right)": [o["u_frac"] for o in obs],
            "v_frac (image top->bottom)": [o["v_frac"] for o in obs],
            "depth_m (surface distance)": [o["depth_m"] for o in obs],
        }
        print(f"\n  Pearson r, camera axis vs. cue   (n={len(obs)})")
        print(f"  {'cue':<28s} {'r(cam_x)':>10s} {'r(cam_y)':>10s} {'r(cam_z)':>10s}   best")
        hyp_rec = {"axes": {}}
        for cue_name, cue_vals in cues.items():
            if any(math.isnan(v) for v in cue_vals):
                print(f"  {cue_name:<28s} {'--':>10s} {'--':>10s} {'--':>10s}   (unavailable)")
                continue
            rs = [_pearson([float(o["cam"][k]) for o in obs], cue_vals) for k in range(3)]
            cells = "".join(f"{('%+.3f' % r) if r is not None else '  n/a':>11s}" for r in rs)
            valid = [(abs(r), k) for k, r in enumerate(rs) if r is not None]
            if valid:
                _, best_k = max(valid)
                sign = "+" if rs[best_k] > 0 else "-"
                print(f"  {cue_name:<28s}{cells}   {sign}{AXIS_NAMES[best_k]}")
                hyp_rec["axes"][cue_name.split()[0]] = {
                    "axis": AXIS_NAMES[best_k], "sign": sign, "r": rs[best_k]}
            else:
                print(f"  {cue_name:<28s}{cells}   (degenerate)")

        # metric discriminator: does the chosen forward axis match depth in metres?
        fwd = hyp_rec["axes"].get("depth_m")
        if fwd and depth_mm is not None:
            k = AXIS_NAMES.index(fwd["axis"])
            resid = [abs(abs(float(o["cam"][k])) - o["depth_m"]) for o in obs
                     if not math.isnan(o["depth_m"])]
            mad = sum(resid) / len(resid) if resid else float("nan")
            hyp_rec["forward_metric_mae_m"] = mad
            print(f"\n  metric check: mean |{fwd['sign']}{fwd['axis']}| vs. observed depth "
                  f"= {mad:.3f} m mean abs difference")
            print("    Expect a few tens of centimetres (OBB centroid sits behind the")
            print("    visible surface). Metres of disagreement => wrong hypothesis.")
        print()
        res["hypotheses"][hname] = hyp_rec

    scored = {h: v.get("forward_metric_mae_m") for h, v in res["hypotheses"].items()}
    scored = {h: v for h, v in scored.items() if v is not None and not math.isnan(v)}
    print("=" * 78)
    if not scored:
        print("  VERDICT: inconclusive — no usable depth. Fix the depth read, or")
        print("  fall back to the image-space correlations plus a manual RGB check.")
        res["verdict"] = "inconclusive (no usable depth)"
        print("=" * 78)
        return res

    best = min(scored, key=scored.get)
    axes = res["hypotheses"][best]["axes"]
    res["winner"] = best
    roles = (("horizontal (left->right)", "u_frac"),
             ("vertical   (top->bottom)", "v_frac"),
             ("forward    (into scene) ", "depth_m"))
    assigned = [axes[k]["axis"] for _, k in roles if k in axes]

    print(f"  best metric fit: {best.split(' ')[0]}, mean depth error {scored[best]:.3f} m")
    for h in scored:
        if h != best:
            print(f"                   ({h.split(' ')[0]} gives {scored[h]:.3f} m)")
    print("\n  Axis assignment under that hypothesis:")
    for cue, key in roles:
        a = axes.get(key)
        print(f"    {cue}:  {a['sign']}{a['axis']}   (r = {a['r']:+.3f})" if a
              else f"    {cue}:  (unavailable)")

    # The per-cue argmax runs independently per role, so nothing stops one axis
    # winning two of them. When it does, the fit carries no information at all.
    reasons = []
    if len(assigned) != len(set(assigned)):
        reasons.append(f"one axis was assigned to more than one role ({', '.join(assigned)})")
    if len(obs) < 5:
        reasons.append(f"n={len(obs)} objects is too few to separate three axes")
    if any(o["px_share"] > 0.25 for o in obs):
        reasons.append("an object fills >25% of the frame, so its OBB centroid does not "
                       "stand for its visible position")

    print()
    if reasons:
        res["verdict"] = "INCONCLUSIVE"
        print("  VERDICT: INCONCLUSIVE — do not cite this as the camera convention.")
        for why in reasons:
            print(f"    * {why}")
    else:
        res["verdict"] = f"measured {best.split(' ')[0]}"
        print(f"  VERDICT: {best.split(' ')[0]} — roles are distinct and n is adequate.")
        h = axes.get("u_frac")
        if h:
            print(f"    'to the right of' = a LARGER value of {h['sign']}{h['axis']} in camera coords.")

    print()
    print("  WHAT THE PIPELINE ASSUMES, INDEPENDENT OF THIS CHECK:")
    print("    geometry/upright_projection_direction_check.py, gt/dir_rule_divergence.py,")
    print("    gt/depth_axis_check.py and viz/viz_gt_vlm_panels.py all apply inv(pose) —")
    print("    i.e. H_twc. An H_tcw result here is NOT grounds to change them; on a frame")
    print("    this small the fit cannot separate the two hypotheses. Confirm against the")
    print("    RGB before touching anything:")
    print(f"      open {stem.parent / (stem.name + '.color.jpg')}")
    print("    The object with the largest u_frac above must be visibly the rightmost.")
    print("=" * 78)
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Three pre-code verifications for the 3DSSG evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dssg-root", default="/data/mmc-user/3dssg",
                   help="root containing 3dssg_data/, objects.json, relationships.json")
    p.add_argument("--data-subdir", default="3dssg_data",
                   help="nested level holding the per-scan folders")
    p.add_argument("--scan", default="5341b7a5-8a66-2cdd-8751-70b98263cb8d",
                   help="scan id used for V1 and V3")
    p.add_argument("--frame", type=int, default=89, help="frame number for V1/V3")
    p.add_argument("--scans", nargs="*", default=DEFAULT_SCANS,
                   help="scan ids to check coverage for in V2")
    p.add_argument("--exclude-ids", nargs="*", type=int, default=[],
                   help="objectIds to drop from the V3 fit (e.g. 1 7 for floor/wall)")
    p.add_argument("--min-px", type=int, default=200,
                   help="minimum mask pixels for an object to enter the V3 fit")
    p.add_argument("--skip", nargs="*", default=[], choices=["v1", "v2", "v3"],
                   help="skip individual checks")
    p.add_argument("--json-out", default=None, help="write a machine-readable summary here")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-7s %(message)s")
    _install_signal_handlers()

    root = Path(args.dssg_root)
    scan_dir = root / args.data_subdir / args.scan
    if not scan_dir.exists():
        LOG.error("scan dir not found: %s", scan_dir)
        LOG.error("check the nesting level explicitly -- a past run failed on this:")
        LOG.error("  find %s -maxdepth 2 -type d | head -20", root)
        return 2

    results = []
    if "v1" not in args.skip:
        results.append(verify_instances_encoding(scan_dir, args.frame))
    if "v2" not in args.skip:
        results.append(verify_split_gt(root, args.scans))
    if "v3" not in args.skip:
        results.append(verify_camera_convention(
            scan_dir, args.frame, set(args.exclude_ids), args.min_px))

    _hr("SUMMARY")
    gate_ok = True
    for r in results:
        if r.get("diagnostic"):
            print(f"  DIAG  {r['check']}  -> {r.get('verdict', 'inconclusive')}")
            continue
        mark = "PASS" if r.get("passed") else "FAIL"
        extra = "  (dtype/mode only — see above)" if r.get("dtype_only") else ""
        err = f"   [{r['error']}]" if r.get("error") else ""
        print(f"  {mark:<5s} {r['check']}{extra}{err}")
        gate_ok &= bool(r.get("passed"))
    print()
    print("  V1 and V2 gate; both must pass. V3 is a DIAGNOSTIC and gates nothing:")
    print("  the scored eval path (node_match, *_recall_at_k) never loads pose.")
    print("  Gates cleared." if gate_ok
          else "  Gate failure above — fix it before trusting anything downstream.")
    print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        serialisable = json.loads(json.dumps(results, default=lambda o: str(o)))
        out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        LOG.info("summary → %s", out)

    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())