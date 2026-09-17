"""
Grounding pipeline -- orchestrator.

Single responsibility: coordinate the three layers (VLM, parser, renderer)
in the correct order, persist results, and report statistics.

This file contains NO domain logic of its own -- only coordination.
All heavy lifting is delegated to:
  vlm.py           -> model inference
  parsing.py       -> response interpretation
  visualization.py -> image annotation
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vlm import VLM
from parsing import Detection, parse_detections
from visualization import render_grounding


# ─── Prompt ───────────────────────────────────────────────────────
GROUNDING_PROMPT = """Detect all objects in this indoor scene image.

For EVERY object, return its name and bounding box.

Output format — return ONLY this JSON, nothing else:
[
  {"label": "object_name", "bbox": [x1, y1, x2, y2]},
  {"label": "another_object", "bbox": [x1, y1, x2, y2]}
]

CRITICAL RULES:
- Every object MUST have a "bbox" field with 4 integer pixel coordinates
- Objects without bbox will be discarded
- bbox = [left, top, right, bottom] in pixels
- Include furniture, decor, electronics, appliances
- Skip floor, wall, ceiling
- Use specific names like "glass_table" not just "table"

Example for a 1000x800 image:
[
  {"label": "wooden_chair", "bbox": [100, 300, 350, 700]},
  {"label": "desk_lamp", "bbox": [400, 100, 500, 350]}
]"""


# ─── Result container ─────────────────────────────────────────────
@dataclass
class GroundingResult:
    """All outputs produced for a single image."""
    name: str
    detections: list[Detection]
    inference_seconds: float
    image_size: tuple[int, int]
    raw_response: str


# ─── Pipeline steps ───────────────────────────────────────────────
def ground_image(image_path: Path) -> GroundingResult:
    """
    Runs VLM inference and parsing for a single image.
    This is a pure function with no file output or visualization.
    """
    image = Image.open(image_path).convert("RGB")

    start = time.perf_counter()
    response = VLM.ask(image, GROUNDING_PROMPT)
    elapsed = time.perf_counter() - start

    detections = parse_detections(response)

    return GroundingResult(
        name=image_path.stem,
        detections=detections,
        inference_seconds=round(elapsed, 1),
        image_size=image.size,
        raw_response=response,
    )


def save_result(
    result: GroundingResult, image_path: Path, output_dir: Path
) -> None:
    """
    Writes two artifacts to disk:
      1. Annotated image  (<name>_grounded.jpg)
      2. JSON report      (<name>_objects.json), fully reproducible
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotated image
    render_grounding(
        image_path=image_path,
        detections=result.detections,
        output_path=output_dir / f"{result.name}_grounded.jpg",
    )

    # 2. JSON report
    report = {
        "image": str(image_path),
        "size": list(result.image_size),
        "inference_time": result.inference_seconds,
        "detections": [d.to_dict() for d in result.detections],
        "raw_response": result.raw_response,
    }
    report_path = output_dir / f"{result.name}_objects.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )


def process_image(image_path: Path, output_dir: Path) -> GroundingResult:
    """Full pipeline for a single image: inference, parse, save, and log."""
    print(f"\n{'='*50}\nProcessing: {image_path.stem}\n{'='*50}")

    result = ground_image(image_path)

    print(f"  Size:       {result.image_size}")
    print(f"  Inference:  {result.inference_seconds}s")
    print(f"  Detections: {len(result.detections)}")
    for i, det in enumerate(result.detections, 1):
        print(f"    {i:2}. {det.label}: {list(det.bbox)}")

    save_result(result, image_path, output_dir)
    print(f"  Saved to {output_dir}")

    return result


def process_directory(
    image_dir: Path, output_dir: Path
) -> list[GroundingResult]:
    """
    Runs the full pipeline on every image in a directory.
    Supported extensions: .jpg, .jpeg, .png
    """
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in extensions
    )
    print(f"Found {len(images)} images in {image_dir}")

    results = [process_image(img, output_dir) for img in images]

    # Summary table
    total_objects = sum(len(r.detections) for r in results)
    total_time    = sum(r.inference_seconds for r in results)
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for r in results:
        print(
            f"  {r.name:20} {len(r.detections):3} detections"
            f"   {r.inference_seconds:5.1f}s"
        )
    print(
        f"  {'TOTAL':20} {total_objects:3} detections"
        f"   {total_time:5.1f}s"
    )

    return results


# ─── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path("/data/mmc-user/projects")
    process_directory(
        image_dir=PROJECT_ROOT / "test_images",
        output_dir=PROJECT_ROOT / "output",
    )