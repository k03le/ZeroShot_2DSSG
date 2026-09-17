"""
Script: extract_crops.py
Stage:  8A-prep — Object Crop Extraction & Label Normalization
Purpose: Group per-frame detections from localization_3d by tag, select the
         best detection per unique canonical label (CLIP-clustered), crop the
         RGB frame, and write crops + metadata for describe_objects.py.
Input:   /data/mmc-user/outputs/localization_3d/{scene_id}.json
         /data/mmc-user/arkitscenes/raw/raw/Training/{scene_id}/lowres_wide/{frame_id}.png
Output:  /data/mmc-user/outputs/object_crops/{scene_id}/metadata.json
         /data/mmc-user/outputs/object_crops/{scene_id}/{obj_id}.jpg
Usage:   python extract_crops.py [--scene_ids 47331541 47333462 ...] [--resume]
         python extract_crops.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path("/data/mmc-user")
LOC3D_DIR   = BASE / "outputs" / "localization_3d"
CROPS_DIR   = BASE / "outputs" / "object_crops"
ARKITSCENES = BASE / "arkitscenes" / "raw" / "raw" / "Training"
TMP_DIR     = BASE / "tmp"
LOGS_DIR    = BASE / "logs" / "ram_pipeline" / "stage_8a_prep"

ALL_SCENE_IDS = [
    "47331541", "47333462", "40753679", "40776203", "40777060", "40809740",
    "40958733", "40966448", "41007548", "41007581", "41007602", "41007608",
]

# ─── Globals for signal handling ──────────────────────────────────────────────
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    logging.warning("Interrupt received — finishing current scene then exiting.")
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─── I/O helpers ──────────────────────────────────────────────────────────────
def atomic_write_json(data: dict, final_path: Path) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / (final_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.rename(tmp, final_path)


def rgb_frame_path(scene_id: str, frame_id: str) -> Path:
    return ARKITSCENES / scene_id / "lowres_wide" / f"{frame_id}.png"


# ─── Label normalization (CLIP clustering, no spaCy) ──────────────────────────
def normalize_labels(
    tags: list[str],
    distance_threshold: float,
    embedder,
) -> dict[str, str]:
    """
    Maps each raw tag to a canonical label via agglomerative CLIP clustering.
    Falls back to identity mapping for single tags or on any error.
    """
    if len(tags) <= 1:
        return {t: t for t in tags}

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    cleaned = [t.lower().strip() for t in tags]
    try:
        embeddings = embedder.encode(cleaned, normalize_embeddings=True,
                                     show_progress_bar=False)
        dists = pdist(embeddings, metric="cosine")
        Z = linkage(dists, method="average")
        cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")
    except Exception as e:
        logging.warning(f"Label normalization failed ({e}); using identity mapping.")
        return {t: t for t in tags}

    mapping: dict[str, str] = {}
    for cid in set(cluster_ids):
        idxs    = [i for i, c in enumerate(cluster_ids) if c == cid]
        members = [tags[i] for i in idxs]
        embs    = embeddings[idxs]
        centroid = embs.mean(axis=0)
        closest  = members[int(np.argmin(np.linalg.norm(embs - centroid, axis=1)))]
        for m in members:
            mapping[m] = closest
    return mapping


# ─── Per-scene processing ─────────────────────────────────────────────────────
def process_scene(
    scene_id: str,
    args: argparse.Namespace,
    embedder,
    writer,
) -> dict | None:
    """
    Returns the metadata dict on success, None on skip or error.
    """
    out_dir = CROPS_DIR / scene_id
    meta_path = out_dir / "metadata.json"

    # Resume / overwrite check
    if meta_path.exists():
        if args.resume:
            logging.info(f"[{scene_id}] Skipping — already done (--resume).")
            return None
        if not args.overwrite:
            logging.error(
                f"[{scene_id}] Output exists. Pass --resume to skip or "
                f"--overwrite to replace."
            )
            sys.exit(1)

    # Load localization_3d
    loc_path = LOC3D_DIR / f"{scene_id}.json"
    if not loc_path.exists():
        logging.error(f"[{scene_id}] localization_3d not found: {loc_path}")
        return None

    loc_data: dict[str, list[dict]] = json.loads(loc_path.read_text())

    # ── Group detections by tag across ALL frames ──────────────────────────
    # tag → list of candidate dicts
    tag_candidates: dict[str, list[dict]] = defaultdict(list)
    for frame_id, detections in loc_data.items():
        for det in detections:
            if det["score_2d"] < args.min_score:
                continue
            if det["n_points"] < args.min_points:
                continue
            tag_candidates[det["tag"]].append({
                "frame_id":   frame_id,
                "score_2d":   det["score_2d"],
                "n_points":   det["n_points"],
                "bbox_2d":    det["bbox_2d"],
                "bbox_3d":    det["bbox_3d"],
            })

    unique_tags = sorted(tag_candidates.keys())
    logging.info(f"[{scene_id}] {len(loc_data)} frames, "
                 f"{len(unique_tags)} unique tags after filtering.")

    if not unique_tags:
        logging.warning(f"[{scene_id}] No tags passed filters — skipping.")
        return None

    # ── Select best detection per raw tag (max n_points) ──────────────────
    best_per_tag: dict[str, dict] = {
        tag: max(cands, key=lambda d: d["n_points"])
        for tag, cands in tag_candidates.items()
    }

    # ── Label normalization ────────────────────────────────────────────────
    tag_to_canonical = normalize_labels(unique_tags, args.label_dist_threshold,
                                        embedder)
    n_canonical = len(set(tag_to_canonical.values()))
    logging.info(f"[{scene_id}] Normalized {len(unique_tags)} tags → "
                 f"{n_canonical} canonical labels.")

    # ── Select best detection per canonical label ──────────────────────────
    canonical_best: dict[str, tuple[str, dict]] = {}  # canonical → (raw_tag, det)
    for tag, det in best_per_tag.items():
        canonical = tag_to_canonical[tag]
        if canonical not in canonical_best:
            canonical_best[canonical] = (tag, det)
        else:
            _, existing = canonical_best[canonical]
            if det["n_points"] > existing["n_points"]:
                canonical_best[canonical] = (tag, det)

    # Apply --max-objects cap (sort by n_points descending)
    ranked = sorted(
        canonical_best.items(),
        key=lambda kv: kv[1][1]["n_points"],
        reverse=True,
    )[: args.max_objects]

    # ── Crop and save ─────────────────────────────────────────────────────
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    objects = []
    pad = args.crop_padding

    for i, (canonical, (raw_tag, det)) in enumerate(
        tqdm(ranked, desc=f"  {scene_id} crops", leave=False)
    ):
        obj_id     = f"obj_{i:04d}"
        crop_path  = out_dir / f"{obj_id}.jpg"
        frame_path = rgb_frame_path(scene_id, det["frame_id"])

        if not frame_path.exists():
            logging.warning(f"  Frame not found: {frame_path} — skipping {raw_tag}.")
            continue

        if not args.dry_run:
            try:
                img = Image.open(frame_path).convert("RGB")
                w, h = img.size
                x1, y1, x2, y2 = det["bbox_2d"]
                crop = img.crop((
                    max(0,  x1 - pad),
                    max(0,  y1 - pad),
                    min(w,  x2 + pad),
                    min(h,  y2 + pad),
                ))
                crop.save(str(crop_path), quality=90)
            except Exception as e:
                logging.warning(f"  Crop failed for {raw_tag}: {e} — skipping.")
                continue

        # Derive centroid + volume from bbox_3d
        b = det["bbox_3d"]
        centroid = [
            round((b["x_min"] + b["x_max"]) / 2, 4),
            round((b["y_min"] + b["y_max"]) / 2, 4),
            round((b["z_min"] + b["z_max"]) / 2, 4),
        ]
        volume = (
            (b["x_max"] - b["x_min"]) *
            (b["y_max"] - b["y_min"]) *
            (b["z_max"] - b["z_min"])
        )

        objects.append({
            "object_id":       obj_id,
            "label":           raw_tag,
            "label_normalized": canonical,
            "bbox_3d":         b,
            "centroid":        centroid,
            "volume_m3":       round(volume, 4),
            "n_points":        det["n_points"],
            "grounding_score": det["score_2d"],
            "best_frame_id":   det["frame_id"],
            "bbox_2d":         det["bbox_2d"],
            "crop_path":       str(crop_path),
        })

    metadata = {
        "scene_id":            scene_id,
        "n_objects_raw":       len(unique_tags),
        "n_objects_normalized": n_canonical,
        "n_objects_saved":     len(objects),
        "objects":             objects,
    }

    if not args.dry_run:
        atomic_write_json(metadata, meta_path)

    # ── TensorBoard ───────────────────────────────────────────────────────
    if writer is not None:
        step = ALL_SCENE_IDS.index(scene_id) if scene_id in ALL_SCENE_IDS else 0
        writer.add_scalar("tags/n_raw",        len(unique_tags), step)
        writer.add_scalar("tags/n_canonical",   n_canonical,      step)
        writer.add_scalar("tags/n_saved",       len(objects),     step)
        writer.add_scalar("tags/reduction_ratio",
                          1 - n_canonical / max(len(unique_tags), 1), step)
        if objects:
            scores   = [o["grounding_score"] for o in objects]
            n_points = [o["n_points"]        for o in objects]
            writer.add_histogram("detections/grounding_score", np.array(scores),   step)
            writer.add_histogram("detections/n_points",        np.array(n_points), step)

    logging.info(
        f"[{scene_id}] Done: {len(unique_tags)} raw tags → "
        f"{n_canonical} canonical → {len(objects)} crops saved."
    )
    return metadata


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 8A-prep: extract object crops from localization_3d outputs."
    )
    parser.add_argument(
        "--scene_ids", nargs="+", default=None,
        help="Scene IDs to process. Default: all scenes in localization_3d/.",
    )
    parser.add_argument("--min-score",  type=float, default=0.3,
                        help="Min grounding score_2d to include (default: 0.3).")
    parser.add_argument("--min-points", type=int,   default=100,
                        help="Min n_points to include (default: 100).")
    parser.add_argument("--max-objects", type=int,  default=300,
                        help="Max canonical objects per scene (default: 300).")
    parser.add_argument("--crop-padding", type=int, default=20,
                        help="Pixel padding around each bbox_2d (default: 20).")
    parser.add_argument("--label-dist-threshold", type=float, default=0.05,
                        help="Cosine distance threshold for CLIP clustering (default: 0.05).")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip scenes whose metadata.json already exists.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process scenes even if output already exists.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate inputs and print counts without writing output.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.resume and args.overwrite:
        logging.error("--resume and --overwrite are mutually exclusive.")
        sys.exit(1)

    # Discover scene IDs
    scene_ids = args.scene_ids or [
        p.stem for p in sorted(LOC3D_DIR.glob("*.json"))
    ]
    if not scene_ids:
        logging.error(f"No scenes found in {LOC3D_DIR}")
        sys.exit(1)
    logging.info(f"Processing {len(scene_ids)} scenes: {scene_ids}")

    # Load embedding model once
    logging.info("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Embedder ready.")

    # TensorBoard
    writer = None
    if not args.dry_run:
        try:
            from torch.utils.tensorboard import SummaryWriter
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(LOGS_DIR))
        except Exception as e:
            logging.warning(f"TensorBoard init failed ({e}); continuing without it.")

    # Main loop
    t0 = time.time()
    for scene_id in tqdm(scene_ids, desc="Scenes"):
        if _shutdown:
            logging.warning("Shutdown requested — stopping after current scene.")
            break
        try:
            process_scene(scene_id, args, embedder, writer)
        except Exception as e:
            logging.error(f"[{scene_id}] Unhandled error: {e}", exc_info=True)

    elapsed = time.time() - t0
    logging.info(f"Finished in {elapsed:.1f}s.")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()