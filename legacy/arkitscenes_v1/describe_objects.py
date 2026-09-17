"""
Script: describe_objects.py
Stage:  8A-infer — Object Description Generation
Purpose: Reads object crops locally, sends each to the VLM HTTP server on
         wasti-old (vlm_server.py), and writes objects_with_descriptions/.
Input:   /data/mmc-user/outputs/object_crops/{scene_id}/metadata.json
         /data/mmc-user/outputs/object_crops/{scene_id}/{obj_id}.jpg
Output:  /data/mmc-user/outputs/objects_with_descriptions/{scene_id}.json
Usage:   python describe_objects.py [--scene_ids 47331541 ...] [--resume]
         python describe_objects.py --dry-run

Run on:  wasti-new (Port 6190) | conda activate ram_pipeline
Requires: vlm_server.py running on wasti-old at VLM_SERVER_URL below.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ─── Server config ────────────────────────────────────────────────────────────
VLM_SERVER_URL = "http://10.121.198.34:8765/describe"
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3

# ─── Paths (local to wasti-new) ───────────────────────────────────────────────
BASE      = Path("/data/mmc-user")
CROPS_DIR = BASE / "outputs" / "object_crops"
DESCS_DIR = BASE / "outputs" / "objects_with_descriptions"
TMP_DIR   = BASE / "tmp"
LOGS_DIR  = BASE / "logs" / "ram_pipeline" / "stage_8a_infer"

ALL_SCENE_IDS = [
    "47331541", "47333462", "40753679", "40776203", "40777060", "40809740",
    "40958733", "40966448", "41007548", "41007581", "41007602", "41007608",
]

# ─── Signal handling ──────────────────────────────────────────────────────────
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


def encode_image(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ─── HTTP client ──────────────────────────────────────────────────────────────
def describe_via_server(image_path: Path, label: str) -> str | None:
    try:
        image_b64 = encode_image(image_path)
    except Exception as e:
        logging.warning(f"  Failed to encode {image_path.name}: {e}")
        return None

    payload = json.dumps({"image_b64": image_b64, "label": label}).encode()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                VLM_SERVER_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                return json.loads(resp.read()).get("description")
        except urllib.error.URLError as e:
            logging.warning(f"  HTTP attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1.0)
    return None


def check_server() -> bool:
    health_url = VLM_SERVER_URL.replace("/describe", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception as e:
        logging.error(
            f"VLM server not reachable at {VLM_SERVER_URL}\n"
            f"  Start on wasti-old: conda activate mmc && "
            f"python /data/mmc-user/projects/src/vlm_server.py\n"
            f"  Error: {e}"
        )
        return False


# ─── Per-scene processing ─────────────────────────────────────────────────────
def process_scene(scene_id: str, args: argparse.Namespace, writer) -> dict | None:
    out_path  = DESCS_DIR / f"{scene_id}.json"
    meta_path = CROPS_DIR / scene_id / "metadata.json"

    if out_path.exists():
        if args.resume:
            logging.info(f"[{scene_id}] Skipping — already done (--resume).")
            return None
        if not args.overwrite:
            logging.error(f"[{scene_id}] Output exists. Pass --resume or --overwrite.")
            sys.exit(1)

    if not meta_path.exists():
        logging.error(f"[{scene_id}] metadata.json not found — run extract_crops.py first.")
        return None

    metadata   = json.loads(meta_path.read_text())
    objects_in = metadata.get("objects", [])
    if not objects_in:
        logging.warning(f"[{scene_id}] No objects — skipping.")
        return None

    if args.dry_run:
        objects_in = objects_in[:1]
        logging.info(f"[{scene_id}] Dry-run: 1 / {len(metadata['objects'])} objects.")

    logging.info(f"[{scene_id}] Describing {len(objects_in)} objects ...")
    enriched, times = [], []

    for obj in tqdm(objects_in, desc=f"  {scene_id}", leave=False):
        if _shutdown:
            break

        crop_path = Path(obj["crop_path"])
        if not crop_path.exists():
            logging.warning(f"  Crop not found: {crop_path.name}")
            enriched.append({**obj, "description": None,
                             "description_model": None, "description_frame": None})
            continue

        t0          = time.perf_counter()
        description = describe_via_server(crop_path, obj["label_normalized"])
        elapsed     = time.perf_counter() - t0

        if description is not None:
            times.append(elapsed)
            logging.debug(f"  {obj['object_id']}: {elapsed:.1f}s → {description[:80]}")

        enriched.append({
            **obj,
            "description":       description,
            "description_model": "Qwen2.5-VL-7B-Instruct",
            "description_frame": obj["best_frame_id"],
        })

    output = {"scene_id": scene_id, "n_objects": len(enriched), "objects": enriched}

    if not args.dry_run:
        DESCS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, out_path)

    if writer is not None and times:
        import numpy as np
        step = ALL_SCENE_IDS.index(scene_id) if scene_id in ALL_SCENE_IDS else 0
        writer.add_scalar("vlm/mean_latency_s", float(np.mean(times)), step)
        writer.add_scalar("vlm/total_time_s",   float(np.sum(times)),  step)
        writer.add_scalar("vlm/n_descriptions", len(times),             step)
        writer.add_scalar("vlm/n_failed",
                          sum(1 for o in enriched if o["description"] is None), step)

    n_ok = sum(1 for o in enriched if o["description"] is not None)
    mean_t = sum(times) / len(times) if times else 0.0
    logging.info(f"[{scene_id}] Done: {n_ok} OK, {len(enriched)-n_ok} failed, "
                 f"mean {mean_t:.2f}s/call.")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    global VLM_SERVER_URL

    parser = argparse.ArgumentParser(
        description="Stage 8A-infer: describe objects via VLM server on wasti-old."
    )
    parser.add_argument("--scene_ids",  nargs="+", default=None)
    parser.add_argument("--server-url", default=VLM_SERVER_URL)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    VLM_SERVER_URL = args.server_url

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.resume and args.overwrite:
        logging.error("--resume and --overwrite are mutually exclusive.")
        sys.exit(1)

    logging.info(f"Checking VLM server at {VLM_SERVER_URL} ...")
    if not check_server():
        sys.exit(1)
    logging.info("VLM server reachable.")

    scene_ids = args.scene_ids or (
        [p.name for p in sorted(CROPS_DIR.iterdir()) if p.is_dir()]
        if CROPS_DIR.exists() else []
    )
    if not scene_ids:
        logging.error(f"No scenes in {CROPS_DIR}. Run extract_crops.py first.")
        sys.exit(1)

    logging.info(f"Processing {len(scene_ids)} scenes: {scene_ids}")

    writer = None
    if not args.dry_run:
        try:
            from torch.utils.tensorboard import SummaryWriter
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(LOGS_DIR))
        except Exception as e:
            logging.warning(f"TensorBoard unavailable ({e}); skipping.")

    t0 = time.time()
    for scene_id in tqdm(scene_ids, desc="Scenes"):
        if _shutdown:
            break
        try:
            process_scene(scene_id, args, writer)
        except Exception as e:
            logging.error(f"[{scene_id}] Unhandled error: {e}", exc_info=True)

    logging.info(f"All done in {time.time()-t0:.1f}s.")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()