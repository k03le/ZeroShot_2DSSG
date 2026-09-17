"""
Script:  extract_relations_vlm.py
Purpose: Per-frame VLM relation extraction (VLM-primary, NO geometry injected).
         Draws numbered boxes on the frame for visual grounding, sends the
         annotated frame + numbered object list to the VLM, asks for open-
         vocabulary relation triples [subj_idx, relation, obj_idx], parses
         them, and renders the frame with relation arrows. Dumps raw VLM
         output for inspection.

Input:   localization_3d/{scene}.json  (per-frame 2D detections)
         raw/raw/Training/{scene}/lowres_wide/{frame_id}.png

VLM:     http://10.121.198.34:8765/ask  (wasti-old)

Output:  /data/mmc-user/outputs/vlm_relations/{frame_id}.json   (triples + raw)
         /data/mmc-user/outputs/vlm_relations/{frame_id}_viz.jpg (arrows)
         
Usage:   python extract_relations_vlm.py --frames 40753679_6813.238 40753679_6841.941
         python extract_relations_vlm.py --frames <id> --min-score 0.3
Run on:  wasti-new | conda activate ram_pipeline
"""

import argparse
import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

RAW_ROOT = Path("/data/mmc-user/arkitscenes/raw/raw/Training")
LOC_DIR  = Path("/data/mmc-user/outputs/localization_3d")
OUT_DIR  = Path("/data/mmc-user/outputs/vlm_relations")
VLM_SERVER_URL = "http://10.121.198.34:8765/ask"          # vlm_server.py (bf16)
OLLAMA_URL     = "http://10.121.198.34:11434/api/generate" # Ollama (GGUF)

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
           "#F8C471", "#82E0AA"]


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Prompt — NO geometry injected. Open vocabulary. Indices for grounding.
# ---------------------------------------------------------------------------
def build_prompt(objects, detector_label_mode, inject_geometry=False):
    if detector_label_mode == "suppress":
        lines = "\n".join(f"  ({i})" for i, o in enumerate(objects))
    else:
        lines = "\n".join(f"  ({i}) {o['tag']}" for i, o in enumerate(objects))

    geom_block = ""
    if inject_geometry:
        geom_block = geometry_hint(objects)

    return (
        "You are given an indoor image. Numbered boxes mark detected objects:\n"
        f"{lines}\n"
        f"{geom_block}\n"
        "List the spatial relationships you can see BETWEEN these numbered objects. "
        "Use the numbers to refer to objects. Describe each relationship in your own "
        "words (open vocabulary — do not restrict to a fixed list).\n\n"
        "IMPORTANT: The boxes are 2D detections and may OVERLAP on the image even when "
        "the objects do not touch and are not parts of each other. Do NOT infer "
        "'part of', 'has part', or 'contains' merely because one box overlaps or sits "
        "inside another box. Judge the actual physical relationship from what you see "
        "in the image" + (" (using the 3D context only to focus attention)" if inject_geometry else "") +
        ", not from box overlap.\n\n"
        "Return ONLY a JSON array of triples, each as "
        "[subject_number, \"relation\", object_number]. Example:\n"
        "[[0, \"on top of\", 1], [2, \"next to\", 0]]\n"
        "Only include relationships you are confident are visible. "
        "Return [] if you see none."
    )


# ---------------------------------------------------------------------------
# Geometry features (method iii). ARKit world is Y-up: vertical axis = index 1.
# Computed per object PAIR from per-frame bbox_3d (already outlier-filtered in
# Stage 6). Used as PROMPT CONTEXT only — not as a hard filter/validator.
# ---------------------------------------------------------------------------
VERTICAL = "y"  # ARKit Y-up


def _centroid(b):
    return ((b["x_min"] + b["x_max"]) / 2,
            (b["y_min"] + b["y_max"]) / 2,
            (b["z_min"] + b["z_max"]) / 2)


def _interval_gap(a_min, a_max, b_min, b_max):
    """Gap between two 1D intervals; 0 if they overlap."""
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def pair_geometry(bi, bj):
    """Compute (d_surf, delta_v, omega) between two 3D boxes.

    delta_v : centroid_j.y - centroid_i.y. Positive => j is HIGHER than i
              (gravity-up, since ARKit world is Y-up).
    d_surf  : 3D gap between the nearest faces of the two boxes (0 if they
              overlap/touch). Euclidean over the per-axis interval gaps.
    omega   : footprint overlap on the X-Z (floor) plane, as a fraction of
              the smaller footprint. 1.0 => one sits fully over the other.
    """
    ci, cj = _centroid(bi), _centroid(bj)
    delta_v = cj[1] - ci[1]

    gx = _interval_gap(bi["x_min"], bi["x_max"], bj["x_min"], bj["x_max"])
    gy = _interval_gap(bi["y_min"], bi["y_max"], bj["y_min"], bj["y_max"])
    gz = _interval_gap(bi["z_min"], bi["z_max"], bj["z_min"], bj["z_max"])
    d_surf = (gx**2 + gy**2 + gz**2) ** 0.5

    # X-Z footprint overlap (floor plane, since Y is up)
    ox = max(0.0, min(bi["x_max"], bj["x_max"]) - max(bi["x_min"], bj["x_min"]))
    oz = max(0.0, min(bi["z_max"], bj["z_max"]) - max(bi["z_min"], bj["z_min"]))
    inter = ox * oz
    area_i = (bi["x_max"] - bi["x_min"]) * (bi["z_max"] - bi["z_min"])
    area_j = (bj["x_max"] - bj["x_min"]) * (bj["z_max"] - bj["z_min"])
    denom = min(area_i, area_j)
    omega = inter / denom if denom > 1e-9 else 0.0

    return d_surf, delta_v, omega


def geometry_hint(objects, near_dsurf=0.15, near_omega=0.20):
    """Option (b): attention-steering hint, NOT per-pair relation labels.

    Method (iii) v1 failed by pre-phrasing every pair's vertical offset
    ("object 1 is higher than object 0"), which the VLM transcribed verbatim
    — flooding output with monotonous 'above/below' relations and inventing
    spurious long-range ones. This version instead just tells the VLM WHICH
    pairs are physically close (so it focuses there) and that the rest are
    far apart (so it doesn't relate them), WITHOUT naming any relation. The
    VLM's own visual reasoning decides what the relation is.

    A pair is 'close' if surfaces nearly touch (d_surf < near_dsurf) OR
    footprints meaningfully overlap (omega > near_omega).

    Returns a hint string, or "" if no bbox_3d available.
    """
    n = len(objects)
    close, far = [], []
    any_geom = False
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = objects[i].get("bbox_3d"), objects[j].get("bbox_3d")
            if not bi or not bj:
                continue
            any_geom = True
            d_surf, delta_v, omega = pair_geometry(bi, bj)
            if d_surf < near_dsurf or omega > near_omega:
                close.append((i, j))
            else:
                far.append((i, j))

    if not any_geom:
        return ""

    close_str = (", ".join(f"[{i}&{j}]" for i, j in close)
                 if close else "(none)")
    parts = [
        "\n3D depth-sensor context (to ground your reasoning in real space, "
        "not just 2D appearance):",
        f"- Physically close pairs (near-touching or overlapping in 3D): {close_str}. "
        "These are the pairs most likely to have a direct physical relationship.",
    ]
    if far:
        parts.append(
            "- All other pairs are far apart in 3D and are usually NOT directly "
            "related, even if their 2D boxes overlap in the image."
        )
    parts.append(
        "Use this only to focus attention and avoid relating distant objects. "
        "Decide the actual relationship from the image itself.\n"
    )
    return "\n".join(parts)


def _short_tag(tag, n=18):
    """Trim long multi-word tags for the badge (full tag stays in the prompt)."""
    t = tag.strip()
    return t if len(t) <= n else t[:n - 1] + "\u2026"


def draw_numbered(img, objects, upscale, detector_label_mode):
    """Draw boxes with 'index: tag' badges on an upscaled copy."""
    im = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    fnt = font(15)
    for i, o in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = [c * upscale for c in o["bbox_2d"]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 'index: tag' badge, sized to the text
        badge = f"{i}" if detector_label_mode == "suppress" else f"{i}: {_short_tag(o['tag'])}"
        tb = d.textbbox((0, 0), badge, font=fnt)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        by = max(0, y1)
        d.rectangle([x1, by, x1 + tw + 6, by + th + 6], fill=color)
        d.text((x1 + 3, by + 2), badge, fill="black", font=fnt)
    return im


def call_vlm(annotated_img, prompt, backend="server", model="qwen2.5vl:7b-fp16"):
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()

    if backend == "ollama":
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0, "top_p": 1.0, "seed": 42},
        }, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")

    # backend == "server" (unchanged contract)
    resp = requests.post(VLM_SERVER_URL, json={"image_b64": b64, "prompt": prompt}, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _as_index(v):
    """Coerce a triple endpoint to int. VLM may return 1 or "1"; accept both.
    Returns None if not a usable index."""
    if isinstance(v, bool):          # bool is a subclass of int — reject
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        # handle "obj 1", "(1)", "#1" etc.
        m = re.search(r"\d+", s)
        if m:
            return int(m.group(0))
    return None


def parse_triples(raw, n_objects):
    """Extract [subj, rel, obj] triples robustly from VLM text.
    Tolerant of string indices ("1") and int indices (1)."""
    # strip code fences
    txt = re.sub(r"```(?:json)?", "", raw).strip()
    triples = []
    # try whole-string JSON first, then the first bracketed span
    span = re.search(r"\[.*\]", txt, re.DOTALL)
    for candidate in (txt, span.group(0) if span else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, list):
            continue
        for t in data:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            s = _as_index(t[0])
            o = _as_index(t[2])
            rel = str(t[1]).strip()
            if (s is not None and o is not None and rel
                    and 0 <= s < n_objects and 0 <= o < n_objects and s != o):
                triples.append([s, rel, o])
        if triples:
            return triples
    # regex fallback: [ 1 , "text" , 0 ] with optional quotes around indices
    for m in re.finditer(r"\[\s*\"?(\d+)\"?\s*,\s*\"([^\"]+)\"\s*,\s*\"?(\d+)\"?\s*\]", txt):
        s, rel, o = int(m.group(1)), m.group(2), int(m.group(3))
        if 0 <= s < n_objects and 0 <= o < n_objects and s != o:
            triples.append([s, rel, o])
    return triples


def viz_relations(annotated_img, objects, triples, upscale):
    """Draw arrows + relation labels between box centers."""
    im = annotated_img.copy()
    d = ImageDraw.Draw(im)

    def center(o):
        x1, y1, x2, y2 = [c * upscale for c in o["bbox_2d"]]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    for k, (s, rel, o) in enumerate(triples):
        cs, co = center(objects[s]), center(objects[o])
        color = PALETTE[k % len(PALETTE)]
        d.line([cs, co], fill=color, width=2)
        # arrowhead at object end
        d.ellipse([co[0]-4, co[1]-4, co[0]+4, co[1]+4], fill=color)
        # relation label at midpoint
        mx, my = (cs[0]+co[0])/2, (cs[1]+co[1])/2
        label = f"{s} {rel} {o}"
        tb = d.textbbox((0, 0), label, font=font(13))
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        d.rectangle([mx-2, my-2, mx+tw+2, my+th+2], fill=(0, 0, 0))
        d.text((mx, my), label, fill=color, font=font(13))
    return im


def _area_frac(bbox, img_w, img_h):
    """Fraction of the frame covered by a 2D box [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(img_w * img_h)


def process_frame(frame_id, loc, args):
    scene = frame_id.split("_")[0]
    if frame_id not in loc:
        print(f"  [skip] {frame_id} not in localization_3d")
        return

    img_path = RAW_ROOT / scene / "lowres_wide" / f"{frame_id}.png"
    if not img_path.exists():
        print(f"  [skip] {frame_id}: image not found")
        return
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size

    # filter: score floor AND area ceiling (drop whole-frame scene labels)
    dets = []
    dropped_area, dropped_score = [], []
    for d in loc[frame_id]:
        score_val = d.get("score_2d", d.get("score", 1.0))
        if score_val < args.min_score:
            dropped_score.append(d["tag"])
            continue
        af = _area_frac(d["bbox_2d"], iw, ih)
        if af > args.max_area_frac:
            dropped_area.append(f"{d['tag']}({af:.0%})")
            continue
        dets.append(d)

    if dropped_area:
        print(f"    dropped {len(dropped_area)} whole-frame box(es): {', '.join(dropped_area)}")
    if dropped_score:
        print(f"    dropped {len(dropped_score)} low-score box(es)")

    if len(dets) < 2:
        print(f"  [skip] {frame_id}: <2 objects after filtering")
        return

    annotated = draw_numbered(img, dets, args.upscale, args.detector_label_mode)
    if args.dump_input:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        in_path = OUT_DIR / f"{frame_id}__{args.method}_input.jpg"
        annotated.save(in_path, quality=92)
        print(f"    saved input overlay -> {in_path}")
        if args.dump_only:
            return
    inject = "iii" in args.method
    prompt = build_prompt(dets, args.detector_label_mode, inject_geometry=inject)

    print(f"  {frame_id}: {len(dets)} objects -> calling VLM"
          f"{' (geometry injected)' if inject else ''}...")
    raw = call_vlm(annotated, prompt, backend=args.backend, model=args.model)    
    triples = parse_triples(raw, len(dets))
    print(f"    parsed {len(triples)} relations from VLM")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # readable triples with tags
    readable = [{"subject": f"{s}:{dets[s]['tag']}", "relation": rel,
                 "object": f"{o}:{dets[o]['tag']}"} for s, rel, o in triples]
    out = {
        "frame_id": frame_id,
        "n_objects": len(dets),
        "objects": [{"index": i, "tag": d["tag"], "bbox_2d": d["bbox_2d"],
                     "score_2d": d.get("score_2d", d.get("score", 1.0)), "bbox_3d": d.get("bbox_3d")}
                    for i, d in enumerate(dets)],
        "raw_vlm_response": raw,
        "triples_indexed": triples,
        "relations_readable": readable,
        "prompt_used": prompt,
        "geometry_injected": inject,
        "method": args.method,
    }
    json.dump(out, open(OUT_DIR / f"{frame_id}__{args.method}.json", "w"), indent=2)

    viz = viz_relations(annotated, dets, triples, args.upscale)
    viz.save(OUT_DIR / f"{frame_id}__{args.method}_viz.jpg", quality=92)

    print(f"    saved -> {OUT_DIR / (frame_id + '__' + args.method + '.json')}")
    print(f"    saved -> {OUT_DIR / (frame_id + '__' + args.method + '_viz.jpg')}")
    # print relations to terminal for quick read
    for r in readable:
        print(f"      {r['subject']}  --{r['relation']}-->  {r['object']}")


def main():
    global RAW_ROOT, LOC_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True, help="frame_ids to process")
    ap.add_argument("--method", default="ii",
                    help="comparison method tag for output filenames: "
                         "'i' (VLM does everything), 'ii' (VLM relations on pipeline "
                         "objects, current), 'iii' (geometry injected). Free-form; "
                         "also append model name e.g. 'ii_qwen', 'ii_llava'.")
    ap.add_argument("--backend", choices=["server", "ollama"], default="server",
                    help="VLM backend: 'server' = vlm_server.py bf16, 'ollama' = GGUF via Ollama")
    ap.add_argument("--model", default="qwen2.5vl:7b-fp16",
                    help="Ollama model tag (only used when --backend ollama)")
    ap.add_argument("--dump-input", action="store_true",
                    help="save the numbered input overlay (exactly what the VLM sees)")
    ap.add_argument("--detector-label-mode", choices=["pass", "suppress"], default="pass",
                    help="pass: detector tag visible in prompt text AND box badge (current behavior). "
                    "suppress: VLM sees only index numbers — no tag text, no tag badge.")
    ap.add_argument("--dump-only", action="store_true",
                    help="with --dump-input: save the overlay and skip the VLM call")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="drop detections below this 2D score (keep low: real "
                         "objects in blurry frames score low; area filter handles junk)")
    ap.add_argument("--max-area-frac", type=float, default=0.85,
                    help="drop boxes covering more than this fraction of the frame "
                         "(removes whole-frame scene-label detections like "
                         "'office building'/'meeting room')")
    ap.add_argument("--upscale", type=int, default=5)
    ap.add_argument("--loc-dir", default=str(LOC_DIR),
                    help="directory of per-scene detection JSONs "
                         "(default: /data/mmc-user/outputs/localization_3d). "
                         "NOTE: localization_3d and vlmfirst_boxes are two different "
                         "detection front ends and must not be mixed.")
    ap.add_argument("--raw-root", default=str(RAW_ROOT),
                    help="root holding <scene>/lowres_wide/<frame_id>.png "
                         "(3DSSG: the upright mirror tree)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="where per-frame relation JSON/viz are written")
    args = ap.parse_args()

    RAW_ROOT = Path(args.raw_root)
    LOC_DIR  = Path(args.loc_dir)
    OUT_DIR  = Path(args.out_dir)

    # group frames by scene to load each localization file once
    by_scene = {}
    for fid in args.frames:
        by_scene.setdefault(fid.split("_")[0], []).append(fid)

    for scene, frames in by_scene.items():
        loc_path = LOC_DIR / f"{scene}.json"
        if not loc_path.exists():
            print(f"[error] no localization_3d for scene {scene}")
            continue
        loc = json.load(open(loc_path))
        print(f"Scene {scene}: {len(frames)} frame(s)")
        for fid in frames:
            process_frame(fid, loc, args)


if __name__ == "__main__":
    main()