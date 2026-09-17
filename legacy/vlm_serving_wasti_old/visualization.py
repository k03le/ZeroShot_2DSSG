"""
Renderer: annotates images with bounding boxes and a legend.

Single responsibility: detections + image path -> annotated JPG.
This module has no knowledge of the VLM and reads no raw responses --
it receives everything it needs as typed parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

from parsing import Detection, BBox

# Font type that covers both truetype and default fallback
FontType = Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]


# ─── Configuration ────────────────────────────────────────────────
PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#73C6B6", "#AED6F1",
]

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ─── Helper functions ─────────────────────────────────────────────
def _load_font(path: str, size: int) -> FontType:
    """Loads a TrueType font; falls back to the PIL default if unavailable."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _assign_colors(detections: list[Detection]) -> dict[str, str]:
    """
    Maps each unique label to a color from the palette.
    Same label always gets the same color -- stable across the full list.
    """
    colors: dict[str, str] = {}
    for det in detections:
        if det.label not in colors:
            colors[det.label] = PALETTE[len(colors) % len(PALETTE)]
    return colors


def _clip(bbox: BBox, w: int, h: int) -> BBox:
    """Clamps a bounding box to the image boundaries."""
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    color: str,
    font: FontType,
) -> None:
    """Draws a colored badge with black shadow at the given position."""
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = int(tb[2]) - int(tb[0]), int(tb[3]) - int(tb[1])

    # Black shadow for contrast
    draw.rectangle([x - 2, y - 2, x + tw + 4, y + th + 6], fill="black")
    # Colored background
    draw.rectangle([x, y, x + tw + 2, y + th + 4], fill=color)
    # Text
    draw.text((x + 2, y + 2), text, fill="black", font=font)


# ─── Drawing routines ─────────────────────────────────────────────
def _draw_box_with_id(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    box_id: int,
    color: str,
    font: FontType,
) -> None:
    """Draws the rectangle outline and a numbered badge in the top-left corner."""
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    badge_text = f" {box_id} "
    tb = draw.textbbox((0, 0), badge_text, font=font)
    th = int(tb[3]) - int(tb[1])
    badge_y = max(0, y1 - th - 6)

    _draw_badge(draw, x1, badge_y, badge_text, color, font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    detections: list[Detection],
    colors: dict[str, str],
    image_size: tuple[int, int],
    font: FontType,
) -> None:
    """
    Draws a legend in the bottom-left corner:
    colored square + '1 = chair', '2 = table', ...
    """
    w, h = image_size
    line_height = 20
    y = h - 15 - len(detections) * line_height

    longest = max(
        (f"{i + 1} = {d.label}" for i, d in enumerate(detections)),
        key=len,
        default="",
    )
    panel_w = len(longest) * 9 + 40

    # Dark background panel
    draw.rectangle([0, y - 10, panel_w, h], fill=(0, 0, 0))

    for i, det in enumerate(detections):
        draw.rectangle([10, y + 1, 24, y + 15], fill=colors[det.label])
        draw.text((30, y), f"{i + 1} = {det.label}", fill="white", font=font)
        y += line_height


# ─── Public API ───────────────────────────────────────────────────
def render_grounding(
    image_path: Path | str,
    detections: list[Detection],
    output_path: Path | str,
) -> Path:
    """
    Draws bounding boxes and a legend onto a copy of the image.

    Draw order is intentional:
      1. Boxes first (behind everything)
      2. Legend last (always on top)
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = Image.open(image_path).convert("RGB")
    canvas = ImageDraw.Draw(image)
    font_id    = _load_font(FONT_BOLD, 24)
    font_small = _load_font(FONT_REG, 14)

    colors = _assign_colors(detections)

    # Draw boxes (layer 1)
    for i, det in enumerate(detections):
        clipped = _clip(det.bbox, *image.size)
        _draw_box_with_id(canvas, clipped, i + 1, colors[det.label], font_id)

    # Draw legend (layer 2 -- always on top)
    if detections:
        _draw_legend(canvas, detections, colors, image.size, font_small)

    image.save(output_path, quality=95)
    return output_path