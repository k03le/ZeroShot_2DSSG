"""
Script:  contact_sheet.py
Purpose: Render a labelled grid of candidate frames so you can pick frames
         by eye over SSH. Filters frames by detected-object count, samples a
         spread across the scene, draws detection boxes, labels each cell with
         frame_id + object count. Output is ONE jpg to scp down and browse.
Input:   /data/mmc-user/outputs/localization_3d/{scene}.json   (per-frame dets)
         /data/mmc-user/arkitscenes/raw/raw/Training/{scene}/lowres_wide/{frame_id}.png
Output:  /data/mmc-user/outputs/contact_sheets/{scene}_obj{min}-{max}.jpg
Usage:   python contact_sheet.py --scene 40753679 --min-obj 3 --max-obj 5
         python contact_sheet.py --scene 40753679 --min-obj 8 --max-obj 11 --n 30
Run on:  wasti-new | conda activate ram_pipeline
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAW_ROOT = Path("/data/mmc-user/arkitscenes/raw/raw/Training")
LOC_DIR  = Path("/data/mmc-user/outputs/localization_3d")
OUT_DIR  = Path("/data/mmc-user/outputs/contact_sheets")

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
           "#F8C471", "#82E0AA"]


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def render_cell(scene, frame_id, dets, cell_w, upscale, draw_boxes):
    """Render one frame with optional detection boxes + a header label."""
    img_path = RAW_ROOT / scene / "lowres_wide" / f"{frame_id}.png"
    if not img_path.exists():
        # placeholder cell
        ph = Image.new("RGB", (cell_w, cell_w), (40, 20, 20))
        d = ImageDraw.Draw(ph)
        d.text((6, 6), f"MISSING\n{frame_id}", fill="#ff8888", font=font(11))
        return ph

    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    W, H = img.size
    d = ImageDraw.Draw(img)

    if draw_boxes:
        for i, det in enumerate(dets):
            color = PALETTE[i % len(PALETTE)]
            x1, y1, x2, y2 = [c * upscale for c in det["bbox_2d"]]
            d.rectangle([x1, y1, x2, y2], outline=color, width=2)
            d.text((x1 + 2, y1 + 1), str(i), fill=color, font=font(13))

    # header bar with frame_id + object count.
    # Sized relative to the upscaled image (not the final shrunk cell), and
    # rendered at a fixed minimum font size so it survives the later resize.
    bar_h = 44
    out = Image.new("RGB", (W, H + bar_h), (20, 20, 24))
    out.paste(img, (0, bar_h))
    d2 = ImageDraw.Draw(out)
    d2.text((6, 4), frame_id, fill="white", font=font(26))
    d2.text((6, 4 + 24), f"{len(dets)} obj", fill="#9aa0aa", font=font(18))

    # Scale to requested cell width, keep aspect. If the cell width forces
    # heavy downscaling, the label text shrinks too -- warn rather than
    # silently produce illegible labels.
    if out.width != cell_w:
        ratio = cell_w / out.width
        if ratio < 0.5:
            print(f"  [warn] {frame_id}: downscaling by {ratio:.2f}x -- labels may "
                 f"be small. Increase --cell-w or --upscale, or reduce --cols.")
        out = out.resize((cell_w, int(out.height * ratio)), Image.LANCZOS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--min-obj", type=int, default=3)
    ap.add_argument("--max-obj", type=int, default=6)
    ap.add_argument("--n", type=int, default=30, help="frames to show in the grid")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-w", type=int, default=420, help="px width per cell")
    ap.add_argument("--upscale", type=int, default=5, help="frames are tiny (256x192); upscale")
    ap.add_argument("--no-boxes", action="store_true", help="don't draw detection boxes")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="ignore detections below this 2D score when counting/drawing")
    args = ap.parse_args()

    loc_path = LOC_DIR / f"{args.scene}.json"
    if not loc_path.exists():
        raise SystemExit(f"No localization_3d for scene {args.scene}")
    loc = json.load(open(loc_path))

    # filter detections by score, then frames by object count
    candidates = []
    for fid, dets in loc.items():
        if not isinstance(dets, list):
            continue
        kept = [d for d in dets if d.get("score_2d", 1.0) >= args.min_score]
        if args.min_obj <= len(kept) <= args.max_obj:
            candidates.append((fid, kept))

    if not candidates:
        raise SystemExit(f"No frames with {args.min_obj}-{args.max_obj} objects "
                         f"(score>={args.min_score}).")

    # sort by frame_id (chronological-ish) and sample an even spread
    candidates.sort(key=lambda x: x[0])
    step = max(1, len(candidates) // args.n)
    picked = candidates[::step][:args.n]
    print(f"{len(candidates)} frames match; showing {len(picked)} spread across the scene")

    cells = [render_cell(args.scene, fid, dets, args.cell_w, args.upscale,
                         not args.no_boxes) for fid, dets in picked]

    cw = args.cell_w
    ch = max(c.height for c in cells)
    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * rows + 40), (15, 15, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"Scene {args.scene}  |  {args.min_obj}-{args.max_obj} objects/frame  "
                   f"|  {len(picked)} of {len(candidates)} matching frames",
           fill="white", font=font(20))
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        sheet.paste(cell, (c * cw, r * ch + 40))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.scene}_obj{args.min_obj}-{args.max_obj}.jpg"
    sheet.save(out_path, quality=90)
    print(f"saved -> {out_path}")
    print("\nframe_ids shown (for picking):")
    for fid, dets in picked:
        print(f"  {fid}   ({len(dets)} obj)")


if __name__ == "__main__":
    main()