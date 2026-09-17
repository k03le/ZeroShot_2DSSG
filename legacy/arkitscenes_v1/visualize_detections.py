"""
Script:  visualize_detections.py
Purpose: Visualise 2D grounding boxes and back-projected 3D bbox centroids
         on RGB frames for qualitative inspection.
Usage:
    python visualize_detections.py --scene_id 47331541 --n_frames 6
    python visualize_detections.py --scene_id 40777060 --n_frames 6
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/data/mmc-user/arkitscenes/raw/raw/Training")
BOXES_DIR  = Path("/data/mmc-user/outputs/grounding_boxes")
LOC3D_DIR  = Path("/data/mmc-user/outputs/localization_3d")
OUT_DIR    = Path("/data/mmc-user/outputs/visualizations")

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#73C6B6", "#AED6F1",
]


def load_font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def draw_box(draw, bbox, color, label, font):
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    tb = draw.textbbox((0, 0), f" {label} ", font=font)
    tw = int(tb[2]) - int(tb[0])
    th = int(tb[3]) - int(tb[1])
    draw.rectangle([x1, y1 - th - 4, x1 + tw, y1], fill=color)
    draw.text((x1 + 2, y1 - th - 2), f" {label} ", fill="black", font=font)


def visualize_frame(
    rgb_path: Path,
    detections_2d: list,
    detections_3d: list,
    frame_id: str,
) -> Image.Image:
    """Draw 2D grounding boxes on the RGB frame."""
    img = Image.open(rgb_path).convert("RGB")
    # Upscale for visibility (256×192 is very small)
    img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
    W, H = img.size
    scale = 4

    draw = ImageDraw.Draw(img)
    font = load_font(14)
    font_small = load_font(10)

    for i, det in enumerate(detections_2d):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = det["bbox_2d"]
        # Scale up coordinates to match upscaled image
        bbox_scaled = [x1*scale, y1*scale, x2*scale, y2*scale]
        draw_box(draw, bbox_scaled, color, det["tag"][:20], font)

        # Show confidence score
        draw.text(
            (x1*scale + 2, y2*scale - 14),
            f"{det['score']:.2f}",
            fill=color,
            font=font_small,
        )

    # Add frame ID header
    draw.rectangle([0, 0, W, 20], fill="black")
    draw.text((4, 2), f"Frame: {frame_id} | {len(detections_2d)} detections",
              fill="white", font=font_small)

    return img


def make_grid(images: list, cols: int = 3) -> Image.Image:
    """Arrange images in a grid."""
    rows = (len(images) + cols - 1) // cols
    W, H = images[0].size
    grid = Image.new("RGB", (W * cols, H * rows + 40), color=(30, 30, 30))

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        grid.paste(img, (col * W, row * H + 40))

    return grid


def add_title(grid: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, grid.width, 38], fill=(20, 20, 20))
    draw.text((10, 8), title, fill="white", font=font)
    return grid


def visualize_scene(scene_id: str, n_frames: int, seed: int = 42):
    print(f"\n{'='*60}")
    print(f"Visualising scene: {scene_id}")
    print(f"{'='*60}")

    rgb_dir   = DATA_ROOT / scene_id / "lowres_wide"
    boxes_path = BOXES_DIR / f"{scene_id}.json"
    loc_path   = LOC3D_DIR / f"{scene_id}.json"

    if not boxes_path.exists():
        print(f"ERROR: No grounding boxes for {scene_id}")
        return

    # Load grounding boxes
    with open(boxes_path) as f:
        all_boxes: dict = json.load(f)

    # Load 3D localisation
    all_3d = {}
    if loc_path.exists():
        with open(loc_path) as f:
            all_3d = json.load(f)

    # Pick frames that have detections
    frames_with_dets = [
        fid for fid, dets in all_boxes.items() if len(dets) >= 2
    ]

    if not frames_with_dets:
        print(f"No frames with detections for {scene_id}")
        return

    # Sample evenly across the scene (not just the first N frames)
    random.seed(seed)
    step = max(1, len(frames_with_dets) // n_frames)
    selected = [frames_with_dets[i * step] for i in range(n_frames)]
    selected = selected[:n_frames]

    print(f"Selected {len(selected)} frames from {len(frames_with_dets)} with detections")

    # Render each frame
    images = []
    for frame_id in selected:
        rgb_path = rgb_dir / f"{frame_id}.png"
        if not rgb_path.exists():
            continue

        dets_2d = all_boxes.get(frame_id, [])
        dets_3d = all_3d.get(frame_id, [])

        img = visualize_frame(rgb_path, dets_2d, dets_3d, frame_id)
        images.append(img)
        print(f"  Rendered {frame_id}: {len(dets_2d)} boxes")

    if not images:
        print("No images rendered.")
        return

    # Build grid
    cols = 3
    grid = make_grid(images, cols=cols)

    # Add title with scene stats
    n_total = len([f for f, d in all_boxes.items() if d])
    unique_tags = set(d["tag"] for dets in all_boxes.values() for d in dets)
    title = (
        f"Scene {scene_id} | "
        f"{n_total} frames with detections | "
        f"{len(unique_tags)} unique tags"
    )
    grid = add_title(grid, title)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scene_id}_detections.jpg"
    grid.save(out_path, quality=92)
    print(f"\nSaved → {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_ids", nargs="+",
        default=["47331541", "40777060"],
        help="Scene IDs to visualise"
    )
    parser.add_argument(
        "--n_frames", type=int, default=6,
        help="Number of frames to show per scene"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scene_id in args.scene_ids:
        visualize_scene(scene_id, args.n_frames)

    print(f"\nAll visualisations saved to {OUT_DIR}")