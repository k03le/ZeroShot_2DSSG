"""
Script:  evaluation.py
Stage:   7 — Evaluation & Analysis
Purpose: Compare pipeline 3D detections against ARKitScenes ground truth
         3D annotations. Uses embedding-based label matching (cosine similarity)
         to handle vocabulary differences between RAM++ tags and annotation labels.
Input:   /data/mmc-user/outputs/localization_3d/{scene_id}.json
         /data/mmc-user/arkitscenes/raw/raw/Training/{scene_id}/{scene_id}_3dod_annotation.json
Output:  /data/mmc-user/outputs/evaluation/{scene_id}.json
Usage:
    python evaluation.py
    python evaluation.py --scene_ids 47331541
    python evaluation.py --sim_threshold 0.5
    python evaluation.py --dry_run
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/data/mmc-user/arkitscenes/raw/raw/Training")
LOC3D_DIR    = Path("/data/mmc-user/outputs/localization_3d")
OUTPUT_DIR   = Path("/data/mmc-user/outputs/evaluation")
LOG_DIR      = Path("/data/mmc-user/logs/ram_pipeline/evaluation")
TMP_DIR      = Path("/data/tmp")


# ─── CLI Arguments ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 7: Evaluate 3D detections against ARKitScenes GT."
    )
    parser.add_argument("--scene_ids", nargs="+", default=None)
    parser.add_argument(
        "--sim_threshold", type=float, default=0.7,
        help="Cosine similarity threshold for label matching (default: 0.7)."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(level: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "evaluation.log"),
        ],
    )
    return logging.getLogger("eval")


# ─── Atomic Write ─────────────────────────────────────────────────────────────
def atomic_write_json(data: dict, final_path: Path) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / (final_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp_path, final_path)


# ─── Load Ground Truth Annotations ───────────────────────────────────────────
def load_gt_annotations(scene_id: str) -> list[dict]:
    """
    Load ARKitScenes 3D annotation JSON.

    Each annotation has:
        label:    category name (e.g. "cabinet", "bed")
        obb:      oriented bounding box with centroid, axesLengths, normalizedAxes
    """
    anno_path = DATA_ROOT / scene_id / f"{scene_id}_3dod_annotation.json"
    if not anno_path.exists():
        return []

    with open(anno_path) as f:
        raw = json.load(f)

    annotations = []
    for item in raw.get("data", []):
        label = item.get("label", "unknown")
        obb = item.get("segments", {}).get("obb", {})

        if not obb or "centroid" not in obb:
            continue

        annotations.append({
            "label": label,
            "centroid": obb["centroid"],
            "axesLengths": obb.get("axesLengths", [0, 0, 0]),
            "normalizedAxes": obb.get("normalizedAxes", [1,0,0, 0,1,0, 0,0,1]),
        })

    return annotations


# ─── Load Pipeline Detections ─────────────────────────────────────────────────
def load_detections(scene_id: str) -> list[dict]:
    """
    Aggregate all unique detected tags with their 3D bounding boxes
    across all frames for a scene.

    Since Stage 6 is per-frame only (no cross-frame merging), the same
    physical object appears in many frames. We collect all unique tag names
    and their average 3D position for matching purposes.
    """
    loc_path = LOC3D_DIR / f"{scene_id}.json"
    with open(loc_path) as f:
        loc_data: dict[str, list[dict]] = json.load(f)

    # Aggregate by tag: collect all centroids per tag
    tag_centroids: dict[str, list[list[float]]] = defaultdict(list)
    tag_counts: dict[str, int] = defaultdict(int)

    for frame_id, dets in loc_data.items():
        for det in dets:
            tag = det["tag"]
            bbox = det["bbox_3d"]
            cx = (bbox["x_min"] + bbox["x_max"]) / 2
            cy = (bbox["y_min"] + bbox["y_max"]) / 2
            cz = (bbox["z_min"] + bbox["z_max"]) / 2
            tag_centroids[tag].append([cx, cy, cz])
            tag_counts[tag] += 1

    # Average centroid per tag
    detections = []
    for tag, centroids in tag_centroids.items():
        avg = np.mean(centroids, axis=0).tolist()
        detections.append({
            "tag": tag,
            "avg_centroid": avg,
            "n_frames": tag_counts[tag],
        })

    return detections


# ─── Embedding-Based Label Matching ───────────────────────────────────────────
def load_embedding_model(logger: logging.Logger):
    """
    Load a sentence-transformer model for computing label similarity.
    Uses all-MiniLM-L6-v2: lightweight, fast, 384-dim embeddings.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model loaded.")
    return model


def compute_similarity_matrix(
    detected_tags: list[str],
    gt_labels: list[str],
    embed_model,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between detected tags and GT labels.

    Returns: (n_detected, n_gt) similarity matrix in range [-1, 1].
    """
    all_texts = detected_tags + gt_labels
    embeddings = embed_model.encode(all_texts, normalize_embeddings=True)

    n_det = len(detected_tags)
    det_emb = embeddings[:n_det]     # (n_det, 384)
    gt_emb  = embeddings[n_det:]     # (n_gt, 384)

    # Cosine similarity = dot product of normalised vectors
    sim_matrix = det_emb @ gt_emb.T  # (n_det, n_gt)
    return sim_matrix


def match_labels(
    detected_tags: list[str],
    gt_labels: list[str],
    sim_matrix: np.ndarray,
    threshold: float,
) -> dict:
    """
    Match each detected tag to its best GT label (if above threshold).

    Returns dict with:
        matches:      list of {detected, gt_label, similarity}
        unmatched_det: detected tags with no GT match
        unmatched_gt:  GT labels with no detection match
    """
    n_det, n_gt = sim_matrix.shape

    # For each detection, find the best matching GT label
    matches = []
    matched_gt = set()
    unmatched_det = []

    for i in range(n_det):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])

        if best_sim >= threshold:
            matches.append({
                "detected": detected_tags[i],
                "gt_label": gt_labels[best_j],
                "similarity": round(best_sim, 4),
            })
            matched_gt.add(best_j)
        else:
            unmatched_det.append({
                "detected": detected_tags[i],
                "best_gt": gt_labels[best_j],
                "best_similarity": round(best_sim, 4),
            })

    # GT labels that were never matched
    unmatched_gt = [
        gt_labels[j] for j in range(n_gt) if j not in matched_gt
    ]

    return {
        "matches": matches,
        "unmatched_detections": unmatched_det,
        "unmatched_gt_labels": unmatched_gt,
    }


# ─── Metrics Computation ─────────────────────────────────────────────────────
def compute_metrics(match_result: dict, n_det: int, n_gt: int) -> dict:
    """
    Compute evaluation metrics:
        coverage:         % of GT categories that have at least one matched detection
        false_positive:   % of detected tags that match no GT category
        precision:        matched / total detected
        recall:           matched GT / total GT
    """
    n_matches    = len(match_result["matches"])
    n_unmatched  = len(match_result["unmatched_detections"])
    n_gt_missed  = len(match_result["unmatched_gt_labels"])

    # Unique matched GT labels
    matched_gt_labels = set(m["gt_label"] for m in match_result["matches"])

    coverage = len(matched_gt_labels) / n_gt * 100 if n_gt > 0 else 0
    false_positive_rate = n_unmatched / n_det * 100 if n_det > 0 else 0
    precision = n_matches / n_det * 100 if n_det > 0 else 0
    recall = len(matched_gt_labels) / n_gt * 100 if n_gt > 0 else 0

    return {
        "n_detected": n_det,
        "n_gt_categories": n_gt,
        "n_matched": n_matches,
        "n_false_positives": n_unmatched,
        "n_gt_missed": n_gt_missed,
        "coverage_pct": round(coverage, 1),
        "false_positive_rate_pct": round(false_positive_rate, 1),
        "precision_pct": round(precision, 1),
        "recall_pct": round(recall, 1),
    }


# ─── Scene Evaluation ─────────────────────────────────────────────────────────
def evaluate_scene(
    scene_id: str,
    embed_model,
    threshold: float,
    logger: logging.Logger,
    writer,
    scene_idx: int,
) -> dict:
    """Full evaluation for one scene."""

    # Load ground truth
    gt_annotations = load_gt_annotations(scene_id)
    gt_labels = sorted(set(a["label"] for a in gt_annotations))
    logger.info(f"GT annotations: {len(gt_annotations)} objects, {len(gt_labels)} unique categories")

    # Load pipeline detections
    detections = load_detections(scene_id)
    detected_tags = [d["tag"] for d in detections]
    logger.info(f"Pipeline detections: {len(detected_tags)} unique tags")

    if not gt_labels or not detected_tags:
        logger.warning(f"Scene {scene_id}: empty GT or detections — skipping.")
        return {}

    # Compute similarity matrix
    logger.info(f"Computing {len(detected_tags)} × {len(gt_labels)} similarity matrix ...")
    sim_matrix = compute_similarity_matrix(detected_tags, gt_labels, embed_model)

    # Match labels
    match_result = match_labels(detected_tags, gt_labels, sim_matrix, threshold)

    # Compute metrics
    metrics = compute_metrics(match_result, len(detected_tags), len(gt_labels))

    # Log results
    logger.info(f"Coverage:           {metrics['coverage_pct']}%")
    logger.info(f"Precision:          {metrics['precision_pct']}%")
    logger.info(f"Recall:             {metrics['recall_pct']}%")
    logger.info(f"False positive rate: {metrics['false_positive_rate_pct']}%")

    logger.info(f"Matched ({len(match_result['matches'])}):")
    for m in match_result["matches"]:
        logger.info(f"  {m['detected']:30s} ↔ {m['gt_label']:20s}  sim={m['similarity']:.3f}")

    if match_result["unmatched_detections"]:
        logger.info(f"Unmatched detections ({len(match_result['unmatched_detections'])}):")
        for u in match_result["unmatched_detections"]:
            logger.info(f"  {u['detected']:30s}  best: {u['best_gt']:20s}  sim={u['best_similarity']:.3f}")

    if match_result["unmatched_gt_labels"]:
        logger.info(f"Missed GT categories ({len(match_result['unmatched_gt_labels'])}):")
        for label in match_result["unmatched_gt_labels"]:
            logger.info(f"  {label}")

    # TensorBoard
    writer.add_scalar("eval/coverage_pct",          metrics["coverage_pct"],          scene_idx)
    writer.add_scalar("eval/precision_pct",          metrics["precision_pct"],          scene_idx)
    writer.add_scalar("eval/recall_pct",             metrics["recall_pct"],             scene_idx)
    writer.add_scalar("eval/false_positive_rate_pct", metrics["false_positive_rate_pct"], scene_idx)

    # Similarity score distribution
    for i, tag in enumerate(detected_tags):
        best_sim = float(sim_matrix[i].max())
        writer.add_scalar("eval/best_sim_per_tag", best_sim, scene_idx * 100 + i)

    return {
        "scene_id": scene_id,
        "sim_threshold": threshold,
        "metrics": metrics,
        "matches": match_result["matches"],
        "unmatched_detections": match_result["unmatched_detections"],
        "unmatched_gt_labels": match_result["unmatched_gt_labels"],
        "gt_categories": gt_labels,
        "detected_tags": detected_tags,
        "detections_detail": [
            {**d, "avg_centroid": [round(c, 3) for c in d["avg_centroid"]]}
            for d in detections
        ],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    logger.info(f"TensorBoard logs → {LOG_DIR}")
    logger.info(f"Similarity threshold: {args.sim_threshold}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    if args.scene_ids:
        scene_ids = args.scene_ids
    else:
        scene_ids = sorted(p.stem for p in LOC3D_DIR.glob("*.json"))

    if not scene_ids:
        logger.error("No localization files found — run Stage 6 first.")
        sys.exit(1)

    logger.info(f"Found {len(scene_ids)} scene(s): {scene_ids}")

    # Check existing outputs
    to_process = []
    for sid in scene_ids:
        out_path = OUTPUT_DIR / f"{sid}.json"
        if out_path.exists():
            if args.resume:
                logger.info(f"Scene {sid}: output exists — skipping (--resume).")
                continue
            elif args.overwrite:
                to_process.append(sid)
            else:
                logger.error(
                    f"Scene {sid}: output exists. "
                    "Pass --resume to skip or --overwrite to replace."
                )
                sys.exit(1)
        else:
            to_process.append(sid)

    if not to_process:
        logger.info("Nothing to process.")
        writer.close()
        return

    if args.dry_run:
        logger.info(f"[DRY RUN] Would evaluate: {to_process}")
        writer.close()
        return

    # Load embedding model once
    embed_model = load_embedding_model(logger)

    # Evaluate each scene
    all_results = []
    for i, scene_id in enumerate(to_process):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating scene {i+1}/{len(to_process)}: {scene_id}")
        logger.info(f"{'='*60}")

        result = evaluate_scene(
            scene_id, embed_model, args.sim_threshold, logger, writer, i
        )

        if not result:
            continue

        out_path = OUTPUT_DIR / f"{scene_id}.json"
        atomic_write_json(result, out_path)
        logger.info(f"Scene {scene_id}: saved → {out_path}")
        all_results.append(result)

    # Aggregate summary across all scenes
    if all_results:
        avg_coverage  = np.mean([r["metrics"]["coverage_pct"] for r in all_results])
        avg_precision = np.mean([r["metrics"]["precision_pct"] for r in all_results])
        avg_recall    = np.mean([r["metrics"]["recall_pct"] for r in all_results])
        avg_fp_rate   = np.mean([r["metrics"]["false_positive_rate_pct"] for r in all_results])

        logger.info(f"\n{'='*60}")
        logger.info(f"AGGREGATE RESULTS ({len(all_results)} scenes)")
        logger.info(f"{'='*60}")
        logger.info(f"  Avg coverage:            {avg_coverage:.1f}%")
        logger.info(f"  Avg precision:           {avg_precision:.1f}%")
        logger.info(f"  Avg recall:              {avg_recall:.1f}%")
        logger.info(f"  Avg false positive rate:  {avg_fp_rate:.1f}%")

        # TensorBoard aggregate
        writer.add_scalar("eval/avg_coverage",  avg_coverage,  0)
        writer.add_scalar("eval/avg_precision",  avg_precision,  0)
        writer.add_scalar("eval/avg_recall",     avg_recall,     0)
        writer.add_scalar("eval/avg_fp_rate",    avg_fp_rate,    0)

    writer.close()
    logger.info("Stage 7 complete.")


if __name__ == "__main__":
    main()