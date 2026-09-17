"""
Parser for VLM responses containing bounding-box detections.

Single responsibility: raw model output -> structured Detection list.

Why a separate module? Because VLMs are unreliable JSON producers.
Observed failure modes in Qwen2.5-VL output:
  - Markdown code fences:   triple-backtick JSON ... triple-backtick
  - Swapped brackets:       "}" becomes ">"
  - Trailing commas:        [...,]
  - Wrong key names:        "bbox_2d" instead of "bbox"
  - Repetition loops:       same object generated 54 times

Strategy: three stages of increasing aggressiveness.
  Stage 1: Standard json.loads after minimal cleaning
  Stage 2: Patch common errors, retry parsing
  Stage 3: Regex fallback -- extract detections directly from raw text
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Type alias for a 4-element bounding box
BBox = tuple[int, int, int, int]


# ─── Data container ───────────────────────────────────────────────
@dataclass
class Detection:
    """A single object detection: label + bounding box."""
    label: str
    bbox: BBox  # (x1, y1, x2, y2) in pixels

    def to_dict(self) -> dict:
        return {"label": self.label, "bbox": list(self.bbox)}


# ─── Helper functions ─────────────────────────────────────────────
def _clean_response(response: str) -> str:
    """Strips markdown fences and repairs the most common JSON errors."""
    text = re.sub(r"```json\s*", "", response)
    text = re.sub(r"```\s*", "", text)
    text = text.replace("}>", "}")                  # ">" instead of "}"
    text = re.sub(r",(\s*[])}])", r"\1", text)      # trailing commas
    return text


def _extract_array(text: str) -> Optional[str]:
    """Returns the first JSON array found in the text, or None."""
    match = re.search(r"\[[\s\S]*]", text)
    return match.group() if match else None


def _coerce_detection(item: dict) -> Optional[Detection]:
    """
    Converts a raw dict to a Detection, accepting multiple key names
    for the bounding box (bbox, bbox_2d, bounding_box).
    """
    if "label" not in item:
        return None

    bbox_raw = item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box")
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        return None

    try:
        x1, y1, x2, y2 = int(float(bbox_raw[0])), int(float(bbox_raw[1])), \
                          int(float(bbox_raw[2])), int(float(bbox_raw[3]))
    except (TypeError, ValueError):
        return None

    return Detection(label=str(item["label"]), bbox=(x1, y1, x2, y2))


def _regex_fallback(text: str) -> list[Detection]:
    """
    Last resort: extract detections via regex directly from raw text.
    Used when the JSON is too broken to repair structurally.
    """
    pattern = (
        r'"label"\s*:\s*"([^"]+)"'
        r'.*?'
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*]"
    )
    results: list[Detection] = []
    for m in re.finditer(pattern, text, flags=re.DOTALL):
        bbox: BBox = (
            int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
        results.append(Detection(label=m.group(1), bbox=bbox))
    return results


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    """
    Removes identical (label, bbox) pairs.
    Critical defense against repetition loops in autoregressive generation.
    """
    seen: set[tuple[str, BBox]] = set()
    unique: list[Detection] = []
    for d in detections:
        key = (d.label, d.bbox)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


# ─── Public API ───────────────────────────────────────────────────
def parse_detections(response: str) -> list[Detection]:
    """
    Robust three-stage parsing strategy:
      1. Clean response and attempt standard JSON parsing
      2. On failure: fall back to regex extraction
      3. Deduplicate to remove repetition-loop artifacts

    Never raises -- returns an empty list in the worst case.
    """
    cleaned = _clean_response(response)
    array_text = _extract_array(cleaned)

    if array_text is None:
        print("Warning: no JSON array found in response.")
        return []

    # Stage 1 + 2: structured JSON parsing
    try:
        raw_items = json.loads(array_text)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parse failed ({e}); falling back to regex.")
        return _deduplicate(_regex_fallback(array_text))

    # Convert raw dicts to typed Detection objects
    detections: list[Detection] = [
        d for d in (_coerce_detection(item) for item in raw_items)
        if d is not None
    ]

    # Stage 3: deduplicate
    deduped = _deduplicate(detections)
    if len(deduped) < len(detections):
        print(f"  Deduplicated: {len(detections)} -> {len(deduped)} detections")

    return deduped