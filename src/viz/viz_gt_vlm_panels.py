"""
Script:  viz_gt_vlm_panels.py  (v4)
Purpose: Per-frame GT scene graphs (one panel per predicate category) beside the
         VLM-predicted graph (one un-bucketed panel). Human-eye aid; no scoring.
v4:      (1) directional/vertical GT canonicalised via the INVERSE MAP so each
             unordered pair is ONE arrow carrying all its facts -- v3 merged only
             per ORDERED pair, so [a,b,left] and [b,a,front] still overprinted.
         (2) dir_r2 recomputes BOTH axes: lateral from projected u, depth from
             z_cam ("A front B" <=> z(A) < z(B)). No more "(u-rule n/a)".
         (3) display names (left -> "left of", front -> "in front of"); the raw
             predicate is untouched everywhere in scoring.
         (4) VLM panel draws BOXES + centre crosshair, and flags part-pair
             candidates (overlap coefficient inter/min(areas) >= --part-oc).
Input:   3dssg_data/<scan>/{semseg.v2.json, sequence/*}, relationships.json,
         relations_3dssg_vlmfirst/json/*.json, upright colour from
         3dssg_upright_v2 (NEVER the retracted 3dssg_upright tree).
Output:  <out-dir>/<scan8>_<frame>.png (atomic write)
Usage:   python viz_gt_vlm_panels.py --out-dir .../gt_vlm_panels [--frame-id ...]
         [--scale 2] [--cols 3] [--part-oc 0.8]
"""
import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

LOG = logging.getLogger("panels")
ROT = {-1: lambda u, v, W, H: (H - 1 - v, u), 1: lambda u, v, W, H: (v, W - 1 - u)}

SUPPORT = ("standing on", "attached to", "lying on", "hanging on", "supported by",
           "standing in", "lying in", "connected to", "leaning against", "part of",
           "build in", "cover", "hanging in", "belonging to")
PROX = ("close by",)
DIRECTIONAL = ("left", "right", "front", "behind")
VERTICAL = ("higher than", "lower than")
INV = {"left": "right", "right": "left", "front": "behind", "behind": "front",
       "higher than": "lower than", "lower than": "higher than"}
DISPLAY = {"left": "left of", "right": "right of", "front": "in front of",
           "behind": "behind", "higher than": "higher than",
           "lower than": "lower than", "close by": "close by"}

CAT_COLOR = dict(support=(80, 210, 130), proximity=(110, 180, 255),
                 dir_gt=(255, 195, 60), dir_r2=(255, 130, 60),
                 vertical=(205, 140, 255), vlm=(255, 90, 90))
CAT_TITLE = dict(support="GT: support / attachment",
                 proximity="GT: proximity (close by)",
                 dir_gt="GT: directional (AS STORED)",
                 dir_r2="GT: directional (recomputed from stored 3D geometry, not the original stored label)",
                 vertical="GT: vertical (higher/lower)",
                 vlm="VLM PREDICTED (open-vocabulary, not mapped to a fixed predicate bucket)")
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def load_font(px):
    for p in FONTS:
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            continue
    LOG.warning("no TrueType font; labels will use the tiny default bitmap font")
    return ImageFont.load_default()


def boxed_text(dr, x, y, text, font, col, W, H, anchor_left=True, alpha_bg=210):
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    w, h = r - l, b - t
    tx = min(max(x if anchor_left else x - w, 2), W - w - 4)
    ty = min(max(y, 2), H - h - 4)
    dr.rectangle([tx - 3, ty - 2, tx + w + 3, ty + h + 3], fill=(16, 16, 20, alpha_bg))
    dr.text((tx, ty), text, font=font, fill=col)


def arrow(dr, p0, p1, col, width, head):
    dr.line([p0, p1], fill=col, width=width)
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    n = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    ux, uy = vx / n, vy / n
    dr.polygon([(p1[0] - ux * head - uy * head * 0.45,
                 p1[1] - uy * head + ux * head * 0.45),
                (p1[0] - ux * head + uy * head * 0.45,
                 p1[1] - uy * head - ux * head * 0.45),
                (p1[0] - ux * 3, p1[1] - uy * 3)], fill=col)


def panel(base, nodes, node_labels, edges, title, col, S, fonts, boxes=None,
          extra=""):
    f_node, f_edge, f_title = fonts
    im = base.convert("RGBA")
    dr = ImageDraw.Draw(im)
    W, H = im.size
    lw, head = max(2, int(1.6 * S)), int(9 * S)
    if boxes:
        for k, (x1, y1, x2, y2) in boxes.items():
            dr.rectangle([x1, y1, x2, y2], outline=col + (140,), width=max(2, int(S)))
    for a, b, _lb in edges:
        if a in nodes and b in nodes:
            arrow(dr, nodes[a], nodes[b], col, lw, head)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    for i, (a, b, lb) in enumerate(edges):
        if a in nodes and b in nodes:
            mx = (nodes[a][0] + nodes[b][0]) / 2
            my = (nodes[a][1] + nodes[b][1]) / 2 + (i % 5 - 2) * int(11 * S)
            boxed_text(do, mx + 4, my, lb, f_edge, col + (128,), W, H, alpha_bg=110)
    r = int(4.5 * S)
    for k, (x, y) in nodes.items():
        dr.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2],
                   outline=(255, 255, 255), width=max(2, int(1.3 * S)))
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    for k, (x, y) in nodes.items():
        left_ok = x < W * 0.72
        boxed_text(do, x + r + 5 if left_ok else x - r - 5, y - int(7 * S),
                   node_labels.get(k, str(k)), f_node, (240, 240, 240, 128), W, H,
                   anchor_left=left_ok, alpha_bg=110)
    im = Image.alpha_composite(im, ov).convert("RGB")
    hh = int(17 * S)
    strip = Image.new("RGB", (W, hh), (24, 24, 28))
    ImageDraw.Draw(strip).text(
        (int(4 * S), int(3 * S)),
        "%s   [%d edges, %d nodes]%s" % (title, len(edges), len(nodes), extra),
        font=f_title, fill=col)
    out = Image.new("RGB", (W, H + hh), (24, 24, 28))
    out.paste(strip, (0, 0))
    out.paste(im, (0, hh))
    return out


def main():
    ap = argparse.ArgumentParser(description="GT-vs-Arm-B per-category panels, v4")
    ap.add_argument("--dssg-root", default="/data/mmc-user/3dssg/3dssg_data")
    ap.add_argument("--relationships", default="/data/mmc-user/3dssg/relationships.json")
    ap.add_argument("--rel-glob",
                    default="/data/mmc-user/outputs/relations_3dssg_vlmfirst/json/*.json")
    ap.add_argument("--upright-root",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training")
    ap.add_argument("--rotation-json",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json")
    ap.add_argument("--out-dir", default="/data/mmc-user/outputs/gt_vlm_panels")
    ap.add_argument("--categories",
                    default="support,proximity,dir_gt,dir_r2,vertical,vlm")
    ap.add_argument("--frame-id", default=None)
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--dim", type=float, default=0.55)
    ap.add_argument("--part-oc", type=float, default=0.8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(levelname)s %(name)s: %(message)s")
    cats = [c.strip() for c in a.categories.split(",") if c.strip()]
    bad = [c for c in cats if c not in CAT_TITLE]
    if bad:
        raise SystemExit("[input error] unknown categories %s; valid: %s"
                         % (bad, sorted(CAT_TITLE)))
    S = a.scale
    fonts = (load_font(int(11 * S)), load_font(int(10 * S)), load_font(int(12 * S)))

    files = sorted(glob.glob(a.rel_glob))
    if not files:
        raise SystemExit("[input error] no relation files at %s" % a.rel_glob)
    scan_rel = {r["scan"]: r["relationships"]
                for r in json.loads(Path(a.relationships).read_text())["scans"]}
    ing = json.loads(Path(a.rotation_json).read_text())
    kmap = {"%s_%06d" % (f["scan"], int(f["frame_no"])): int(f["k"]) for f in ing["frames"]}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    labels, wrote = {}, []

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    for fp in tqdm(files, desc="frames"):
        d = json.loads(Path(fp).read_text())
        fid = d["frame_id"]
        if a.frame_id and fid != a.frame_id:
            continue
        scan, fno = fid[:-7], int(fid[-6:])
        outp = Path(a.out_dir) / ("%s_%06d.png" % (scan[:8], fno))
        if outp.exists() and not a.overwrite:
            LOG.warning("skip existing %s (use --overwrite)", outp.name)
            continue
        sdir, seq = Path(a.dssg_root) / scan, Path(a.dssg_root) / scan / "sequence"
        if scan not in labels:
            ss = json.loads((sdir / "semseg.v2.json").read_text())
            labels[scan] = {int(g["objectId"]): g for g in ss["segGroups"]}
        grp = labels[scan]
        mask = np.array(Image.open(seq / ("frame-%06d.rendered.instances.png" % fno)))
        Hu, Wu = mask.shape
        img_p = Path(a.upright_root) / scan / "lowres_wide" / ("%s_%06d.png" % (scan, fno))
        if not img_p.exists():
            LOG.error("missing upright colour %s -- skipping", img_p)
            continue
        Wp, Hp = int(Wu * S), int(Hu * S)
        base = ImageEnhance.Brightness(
            Image.open(img_p).convert("RGB").resize((Wp, Hp), Image.LANCZOS)).enhance(a.dim)

        gnodes, glab, vis = {}, {}, {}
        for oid in np.unique(mask):
            oid = int(oid)
            if oid == 0 or oid not in grp:
                continue
            ys, xs = np.nonzero(mask == oid)
            if xs.size < a.min_px:
                continue
            gnodes[oid] = (float(xs.mean()) * S, float(ys.mean()) * S)
            glab[oid] = "%d:%s" % (oid, grp[oid]["label"])
            vis[oid] = int(xs.size)

        # R2: projected u (lateral) AND z_cam (depth)
        u_r2, z_r2 = {}, {}
        if "dir_r2" in cats:
            try:
                kv = {}
                for ln in (seq / "_info.txt").read_text().splitlines():
                    if "=" in ln:
                        kk, _, vv = ln.partition("=")
                        kv[kk.strip()] = vv.strip()
                m = [float(x) for x in kv["m_calibrationColorIntrinsic"].split()]
                fx, fy, cx, cy = m[0], m[5], m[2], m[6]
                W0, H0 = int(float(kv["m_colorWidth"])), int(float(kv["m_colorHeight"]))
                Ti = np.linalg.inv(np.array(
                    [float(x) for x in
                     (seq / ("frame-%06d.pose.txt" % fno)).read_text().split()]).reshape(4, 4))
                k = kmap["%s_%06d" % (scan, fno)]
                for oid in vis:
                    p = Ti @ np.append(np.asarray(grp[oid]["obb"]["centroid"], float), 1.0)
                    if p[2] > 0.05:
                        uu, _ = ROT[k](fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy, W0, H0)
                        u_r2[oid] = float(uu)
                        z_r2[oid] = float(p[2])
            except (OSError, KeyError, ValueError) as e:
                LOG.warning("R2 unavailable for %s: %s", fid, e)

        # ---- canonicalise: one entry per UNORDERED pair, inverse map applied
        E = defaultdict(list)
        dirf, vertf, prox = defaultdict(set), defaultdict(set), set()
        for s, o, _pid, p in scan_rel.get(scan, []):
            if s == o or s not in gnodes or o not in gnodes:
                continue
            if p in SUPPORT:
                E["support"].append((s, o, p))
            elif p in PROX:
                prox.add(frozenset((s, o)))
            elif p in DIRECTIONAL:
                dirf[(s, o)].add(p) if s < o else dirf[(o, s)].add(INV[p])
            elif p in VERTICAL:
                vertf[(s, o)].add(p) if s < o else vertf[(o, s)].add(INV[p])
        for pair in sorted(prox):
            s, o = sorted(pair)
            E["proximity"].append((s, o, "close by"))
        for (s, o), ps in sorted(dirf.items()):
            E["dir_gt"].append((s, o, " + ".join(DISPLAY[p] for p in sorted(ps))))
        for (s, o), ps in sorted(vertf.items()):
            E["vertical"].append((s, o, " + ".join(DISPLAY[p] for p in sorted(ps))))
        n_flip = 0
        for (s, o), ps in sorted(dirf.items()):
            parts = []
            for pos, neg, src in (("left", "right", u_r2), ("front", "behind", z_r2)):
                if s not in src or o not in src:
                    continue
                rc = pos if src[s] < src[o] else neg
                st = pos if pos in ps else (neg if neg in ps else None)
                flip = st is not None and st != rc
                n_flip += int(flip)
                parts.append(DISPLAY[rc] + (" [FLIPPED]" if flip else ""))
            if parts:
                E["dir_r2"].append((s, o, " + ".join(parts)))

        # ---- VLM: boxes + centres + part-pair candidates
        vnodes, vlab, vbox, vedges = {}, {}, {}, []
        for ob in d["objects"]:
            x1, y1, x2, y2 = ob["bbox_2d"]
            vnodes[ob["index"]] = ((x1 + x2) / 2 * S, (y1 + y2) / 2 * S)
            vbox[ob["index"]] = (x1 * S, y1 * S, x2 * S, y2 * S)
            vlab[ob["index"]] = "%d:%s" % (ob["index"], ob["tag"])
        for t in d.get("triples_indexed") or []:
            vedges.append((t[0], t[2], str(t[1])))
        npart = 0
        for i, j in combinations(sorted(vbox), 2):
            A, B = vbox[i], vbox[j]
            iw = max(0, min(A[2], B[2]) - max(A[0], B[0]))
            ih = max(0, min(A[3], B[3]) - max(A[1], B[1]))
            aA = max(1.0, (A[2] - A[0]) * (A[3] - A[1]))
            aB = max(1.0, (B[2] - B[0]) * (B[3] - B[1]))
            if iw * ih / min(aA, aB) >= a.part_oc:
                npart += 1

        panels = []
        for c in cats:
            if c == "vlm":
                ex = ("   ** ABSTAINED **" if not vedges else "")
                ex += ("   %d part-pair cand." % npart) if npart else ""
                panels.append(panel(base, vnodes, vlab, vedges, CAT_TITLE[c],
                                    CAT_COLOR[c], S, fonts, boxes=vbox, extra=ex))
            else:
                ex = ("   %d flipped" % n_flip) if (c == "dir_r2" and n_flip) else ""
                panels.append(panel(base, gnodes, glab, E[c], CAT_TITLE[c],
                                    CAT_COLOR[c], S, fonts, extra=ex))

        cols = max(1, a.cols)
        rows = (len(panels) + cols - 1) // cols
        pw, ph = panels[0].size
        gap, top = int(4 * S), int(15 * S)
        sheet = Image.new("RGB", (cols * pw + gap * (cols - 1),
                                  top + rows * ph + gap * (rows - 1)), (24, 24, 28))
        ImageDraw.Draw(sheet).text(
            (int(4 * S), int(2 * S)),
            "%s   (mask nodes >= %d px, scale x%g; GT arrows canonicalised per "
            "unordered pair via the inverse map)" % (fid, a.min_px, S),
            font=fonts[2], fill=(235, 235, 235))
        for i, p in enumerate(panels):
            sheet.paste(p, ((i % cols) * (pw + gap), top + (i // cols) * (ph + gap)))
        tmp = outp.with_name(".%s.tmp%d%s" % (outp.stem, os.getpid(), outp.suffix))
        sheet.save(tmp, "PNG")
        os.replace(tmp, outp)
        wrote.append(outp.name)

    print("\nwrote %d panel sheets to %s" % (len(wrote), a.out_dir))
    for w in wrote:
        print("  " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
