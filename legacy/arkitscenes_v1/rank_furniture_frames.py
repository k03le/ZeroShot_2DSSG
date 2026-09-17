"""
Script:  rank_furniture_frames.py
Purpose: Rank frames by how furniture-rich and clean their strict-17
         detections are, to help pick frames for the relation study after
         dropping RAM++ (Stage 5 now prompts Grounding DINO with the fixed
         17 ARKitScenes categories).
Input:   /data/mmc-user/outputs/grounding_boxes/{scene}.json  (strict-17 dets)
Output:  prints a ranked list of frames (frame_id, n_furniture, mean_score,
         category breakdown) so you can pick furniture-dominated frames.
Usage:   python rank_furniture_frames.py --scene 40753679 --min-objects 3 --top 30
Run on:  wasti-new | conda activate ram_pipeline
"""

import argparse
import json
from collections import Counter
from pathlib import Path

BOXES_DIR = Path("/data/mmc-user/outputs/grounding_boxes")

# The 17 ARKitScenes detection categories (must match Stage 5 prompt).
ARKIT_CATEGORIES = {
    "cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer",
    "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool", "chair",
    "table", "tv monitor", "sofa",
}


def score_frame(dets, min_score):
    """Return (n_furniture, mean_score, category_counter) for confident
    in-vocabulary detections in a frame."""
    kept = [d for d in dets
            if d.get("score", 0) >= min_score and d.get("tag") in ARKIT_CATEGORIES]
    if not kept:
        return 0, 0.0, Counter()
    mean_score = sum(d["score"] for d in kept) / len(kept)
    cats = Counter(d["tag"] for d in kept)
    return len(kept), mean_score, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--min-objects", type=int, default=3,
                    help="minimum confident furniture objects to include a frame")
    ap.add_argument("--min-score", type=float, default=0.35,
                    help="per-detection score floor")
    ap.add_argument("--top", type=int, default=30,
                    help="how many top-ranked frames to print")
    ap.add_argument("--max-objects", type=int, default=99,
                    help="cap (to find 'simple' frames, set e.g. 5)")
    args = ap.parse_args()

    path = BOXES_DIR / f"{args.scene}.json"
    if not path.exists():
        raise SystemExit(f"No grounding_boxes for scene {args.scene}")
    boxes = json.load(open(path))

    scored = []
    for fid, dets in boxes.items():
        if not isinstance(dets, list):
            continue
        n, mean_s, cats = score_frame(dets, args.min_score)
        if args.min_objects <= n <= args.max_objects:
            # rank key: prefer more furniture AND higher mean confidence
            rank = n + mean_s  # simple combined score
            scored.append((rank, fid, n, mean_s, cats))

    scored.sort(reverse=True)
    print(f"Scene {args.scene}: {len(scored)} frames with "
          f"{args.min_objects}-{args.max_objects} confident furniture objects "
          f"(score>={args.min_score})\n")
    print(f"{'frame_id':<24} {'n':>3} {'mean':>5}  categories")
    print("-" * 70)
    for rank, fid, n, mean_s, cats in scored[:args.top]:
        cat_str = ", ".join(f"{c}x{k}" if c > 1 else k for k, c in cats.most_common())
        print(f"{fid:<24} {n:>3} {mean_s:>5.2f}  {cat_str}")

    # category coverage summary across the top frames
    print("\nCategory frequency across all matching frames:")
    total = Counter()
    for _, _, _, _, cats in scored:
        total.update(cats)
    for cat, cnt in total.most_common():
        print(f"  {cat:<14} {cnt}")


if __name__ == "__main__":
    main()