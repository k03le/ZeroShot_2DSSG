"""
Script:  overlap_clause_ablation_viz.py
Stage:   Diagnostic ablation visualisation (companion to overlap_clause_ablation_probe.py).
Purpose: For each probed frame, render ONE image with four panels side by side —
         V0 (frozen prompt) | V1 (overlap-caution clause removed) | V2 (overlap-caution
         clause replaced by inert line) | V3 (overlap-caution clause kept + inert line
         inserted before it) — so the abstention flip is visible rather than inferred
         from counts.

         Uses the CANONICAL draw_numbered() and viz_relations() from
         extract_relations_vlm.py: the panels are drawn exactly as the pipeline
         draws them, so nothing here can diverge from the real output.

         READ-ONLY w.r.t. every arm artifact. Writes only into --out-dir.

KNOWN LIMITATION (inherited, do not mistake for a bug): viz_relations draws each
         edge between object CENTROIDS, so reciprocal pairs (i rel j AND j rel' i)
         land on the same line and their labels overpaint each other. The console
         listing printed by this script is authoritative for what was produced;
         the image is for structure, not for counting.

INPUT   --probe-dir  probe_c7/  ({frame}__V{0,1,2,3}.json from overlap_clause_ablation_probe.py)
        --json-dir   relations_3dssg_vlmfirst/json  (for the `objects` list)
        --raw-root   the upright colour tree
OUTPUT  --out-dir    {frame}__overlap_clause_probe_V0-V3.png, one per frame (atomic writes)
        console      per frame x variant, the readable relation list
USAGE
    conda activate ram_pipeline
    cd /data/mmc-user/projects/src
    python viz/overlap_clause_ablation_viz.py --out-dir /data/mmc-user/outputs/probe_c7/viz
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# extract_relations_vlm.py lives in ../pipeline/ (this script lives in viz/);
# make it importable at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from PIL import Image, ImageDraw

LOG = logging.getLogger("overlap_viz")

VARIANT_NOTE = {
    "V0": "frozen prompt",
    "V1": "overlap-caution clause removed",
    "V2": "overlap-caution clause replaced with an inert placebo sentence",
    "V3": "overlap-caution clause kept, plus an inert control line",
}
HEADER_H = 34


def main() -> int:
    ap = argparse.ArgumentParser(description="Render V0-V3 probe panels per frame.")
    # NOTE: directory name "probe_c7" predates this rename and no longer matches
    # either script's filename; left as-is because real output files already
    # live there under this path.
    ap.add_argument("--probe-dir", default="/data/mmc-user/outputs/probe_c7")
    ap.add_argument("--json-dir",
                    default="/data/mmc-user/outputs/relations_3dssg_vlmfirst/json")
    ap.add_argument("--suffix", default="__ii_vlmfirst_qwen3vl_3dssg.json")
    ap.add_argument("--raw-root",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variants", nargs="*", default=["V0", "V1", "V2", "V3"])
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    try:
        import extract_relations_vlm as erv
    except SystemExit:
        raise SystemExit("[abort] importing extract_relations_vlm ran its main(); "
                         "needs an `if __name__ == '__main__':` guard")
    except ImportError as e:
        raise SystemExit(f"[abort] {e}; run from /data/mmc-user/projects/src")

    probe = Path(args.probe_dir)
    frames = sorted({p.name.rsplit("__V", 1)[0] for p in probe.glob("*__V?.json")})
    if not frames:
        raise SystemExit(f"[abort] no *__V?.json in {probe}")
    LOG.info("frames found: %d", len(frames))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        fnt = erv.font(20)
    except Exception:
        fnt = None

    for fk in frames:
        rp = Path(args.json_dir) / f"{fk}{args.suffix}"
        try:
            rec = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            LOG.error("%s: %s — skipped", fk, e); continue
        objects = rec["objects"]

        scan = fk.rsplit("_", 1)[0]
        ip = Path(args.raw_root) / scan / "lowres_wide" / f"{fk}.png"
        try:
            img = Image.open(ip)
        except OSError as e:
            LOG.error("%s: %s — skipped", fk, e); continue
        annotated = erv.draw_numbered(img, objects, args.upscale, "pass")
        W, H = annotated.size

        print(f"\n=== {fk}  ({len(objects)} objects) ===")
        panels = []
        for v in args.variants:
            pp = probe / f"{fk}__{v}.json"
            if not pp.exists():
                LOG.warning("%s %s: missing — placeholder panel", fk, v)
                panels.append((v, annotated.copy(), None)); continue
            d = json.loads(pp.read_text())
            triples = d.get("triples", [])
            panels.append((v, erv.viz_relations(annotated, objects, triples,
                                                args.upscale), len(triples)))
            print(f"  {v:3s} [{VARIANT_NOTE.get(v, ''):22s}] {len(triples)} relation(s)")
            tag = {o["index"]: o["tag"] for o in objects}
            for t in triples:
                try:
                    s, r, o = t[0], t[1], t[2]
                    print(f"        {s}:{tag.get(s, '?'):<28s} --{r}--> {o}:{tag.get(o, '?')}")
                except (IndexError, TypeError):
                    print(f"        (unparseable triple: {t!r})")

        sheet = Image.new("RGB", (W * len(panels), H + HEADER_H), (18, 18, 18))
        dr = ImageDraw.Draw(sheet)
        for i, (v, im, n) in enumerate(panels):
            sheet.paste(im, (i * W, HEADER_H))
            label = (f"{v}  {VARIANT_NOTE.get(v, '')}  —  "
                     f"{'no output file' if n is None else f'{n} relation(s)'}")
            colour = (255, 210, 60) if n else (150, 150, 150)
            dr.text((i * W + 8, 8), label, fill=colour, font=fnt)
            if i:
                dr.line([(i * W, 0), (i * W, H + HEADER_H)], fill=(90, 90, 90), width=2)

        dst = out / f"{fk}__overlap_clause_probe_{'-'.join(args.variants)}.png"
        if dst.exists() and not args.overwrite:
            LOG.info("%s exists — skipped (use --overwrite)", dst.name); continue
        tmp = dst.with_name(f".{dst.stem}.tmp{os.getpid()}{dst.suffix}")
        sheet.save(tmp, "PNG"); os.replace(tmp, dst)
        LOG.info("wrote %s  (%dx%d)", dst.name, *sheet.size)

    print(f"\nsheets in {out}")
    print("NOTE: coincident/reciprocal edges overpaint each other in the panels — "
          "the console listing above is authoritative for counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())