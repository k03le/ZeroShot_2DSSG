#!/usr/bin/env python3
"""
detection_label_quality_and_confidence_auc.py — turn the "~1/3 correct / ~1/3
plausible / ~1/3 confidently wrong" eyeball estimate into a counted result.

PURPOSE
    "~1/3 correct / ~1/3 plausible / ~1/3 confidently wrong" for prompt-free
    YOLOE is currently the most quotable Goal-2 result available, and it is an
    eyeball estimate with no n. This script makes it a hard number, and as a
    byproduct produces the quantitative backing for the claim that detection
    confidence does not separate correct from incorrect labels, at zero extra
    rating effort.

    Two modes:
      emit  — read the YOLOE detections JSON, write a CSV with one row per box and
              an empty `verdict` column. You fill `verdict` while looking at the
              corresponding _viz.jpg.
      tally — read the filled CSV, print counts with explicit n, per-verdict
              confidence statistics, the AUC of confidence-as-correctness-classifier,
              and a paste-ready LaTeX table.

RUBRIC (fix before scoring; record verbatim in the report's Method section)
    correct   the label denotes the object actually in the box, at any reasonable
              granularity. Synonyms and hypernyms count ("office chair" / "swivel
              chair" / "chair" for a swivel chair are all correct).
    plausible wrong, but a human could see the confusion, or the box spans a
              compound object and the label picks one constituent
              (e.g. "bureau" for a desk, "monitor" for a CRT tower+screen).
    wrong     a different category entirely, OR a scene/room label rather than an
              object ("computer room", "hospital room", "kitchen floor"), OR the
              box contains nothing identifiable.
    skip      excluded from the tally (record why in `note`).

INPUT
    emit  : --boxes  /data/mmc-user/outputs/yoloe_boxes/40753679.json
    tally : --csv    ./detection_label_quality_tally.csv   (the emitted file, with verdict filled in)

OUTPUT
    emit  : CSV at --csv (atomic write; refuses to clobber without --overwrite)
    tally : report to stdout; optional --latex-out for the table fragment

USAGE
    conda activate ram_pipeline

    python detection_label_quality_and_confidence_auc.py emit \
        --boxes /data/mmc-user/outputs/yoloe_boxes/40753679.json \
        --csv   ./detection_label_quality_tally.csv

    # ... open each viz/<frame>_viz.jpg, fill the verdict column ...

    python detection_label_quality_and_confidence_auc.py tally --csv ./detection_label_quality_tally.csv --latex-out ./detection_label_quality_table.tex

NOTES
    - Nothing is written to /data/. Both modes are read-only with respect to the
      pipeline outputs.
    - --img-w/--img-h default to ARKitScenes lowres_wide (256x192) and are used
      ONLY for the informational area_frac column. Confirm with:
        python -c "from PIL import Image; \
                   print(Image.open('/data/mmc-user/arkitscenes/raw/raw/Training/40753679/lowres_wide/40753679_6813.238.png').size)"
    - The first raw detection dict is logged at INFO in emit mode so you can
      confirm the bbox_2d ordering before trusting area_frac.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import statistics
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger("label_quality_tally")

VERDICTS = ("correct", "plausible", "wrong", "skip")
FIELDS = [
    "frame", "box_idx", "tag", "score",
    "x1", "y1", "x2", "y2", "area_frac",
    "verdict", "note",
]

_INTERRUPTED = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        LOG.warning("caught signal %s — finishing current item, then exiting clean", signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _atomic_write_csv(rows: list[dict], path: Path) -> None:
    """Write to a tmp file in the same directory, then os.replace(). No partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _normalize_score(det: dict) -> float | None:
    """YOLOE writes `score`; the RAM++/GroundingDINO era wrote `score_2d`.

    This is the same class of score-key naming mismatch seen elsewhere, in
    miniature. Return None rather than a default so a missing
    score is visible in the CSV instead of silently becoming 1.0.
    """
    for key in ("score", "score_2d", "confidence", "conf"):
        if key in det and det[key] is not None:
            try:
                return float(det[key])
            except (TypeError, ValueError):
                LOG.warning("un-floatable %s=%r", key, det[key])
                return None
    return None


def cmd_emit(args: argparse.Namespace) -> int:
    boxes_path = Path(args.boxes)
    csv_path = Path(args.csv)

    if csv_path.exists() and not args.overwrite:
        LOG.error(
            "%s already exists and --overwrite was not given. Refusing to clobber "
            "hand-entered verdicts. Pass --overwrite only if you mean it.", csv_path
        )
        return 2

    try:
        with boxes_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        LOG.error("detections file not found: %s", boxes_path)
        LOG.error("check with:  ls -la %s", boxes_path.parent)
        return 2
    except json.JSONDecodeError as exc:
        LOG.error("could not parse %s as JSON: %s", boxes_path, exc)
        return 2
    except OSError as exc:
        LOG.error("could not read %s: %s", boxes_path, exc)
        return 2

    if not isinstance(data, dict):
        LOG.error("expected a top-level dict keyed by frame id, got %s", type(data).__name__)
        return 2

    frames = sorted(data.keys())
    if args.frames:
        missing = [f for f in args.frames if f not in data]
        if missing:
            LOG.error("frames not present in %s: %s", boxes_path.name, missing)
            LOG.error("available keys: %s", frames)
            return 2
        frames = list(args.frames)

    LOG.info("%d frame(s) selected from %s", len(frames), boxes_path.name)

    peeked = False
    rows: list[dict] = []
    img_area = float(args.img_w * args.img_h)

    for frame in frames:
        if _INTERRUPTED:
            break
        dets = data[frame]
        if not isinstance(dets, list):
            LOG.warning("frame %s: expected a list of detections, got %s — skipped",
                        frame, type(dets).__name__)
            continue
        if dets and not peeked:
            LOG.info("first raw detection (confirm bbox_2d ordering before trusting "
                     "area_frac): %s", json.dumps(dets[0], indent=2)[:800])
            peeked = True

        for i, det in enumerate(dets):
            bbox = det.get("bbox_2d") or []
            if len(bbox) == 4:
                x1, y1, x2, y2 = (float(v) for v in bbox)
                area_frac = abs(x2 - x1) * abs(y2 - y1) / img_area
            else:
                x1 = y1 = x2 = y2 = ""
                area_frac = ""
                LOG.warning("frame %s box %d: bbox_2d has %d values, expected 4",
                            frame, i, len(bbox))
            score = _normalize_score(det)
            rows.append({
                "frame": frame,
                "box_idx": i,
                "tag": det.get("tag", ""),
                "score": "" if score is None else f"{score:.4f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area_frac": "" if area_frac == "" else f"{area_frac:.4f}",
                "verdict": "",
                "note": "",
            })

    if not rows:
        LOG.error("no detections found — nothing to write")
        return 2

    _atomic_write_csv(rows, csv_path)
    LOG.info("wrote %d rows across %d frames → %s", len(rows), len(frames), csv_path)
    LOG.info("Now fill the `verdict` column with one of: %s", ", ".join(VERDICTS))
    LOG.info("Viz files to rate against: <yoloe_boxes dir>/viz/<frame>_viz.jpg")
    return 0


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """AUC of a score separating `pos` from `neg`, via the rank / Mann-Whitney identity.

    Handles ties by averaging ranks (so a score with zero discriminative power
    lands at exactly 0.5 rather than being flattered by tie-breaking).
    Returns None if either class is empty.
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    combined = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, combined) if lab == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _describe(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    if len(vals) == 1:
        return f"n=1  mean={vals[0]:.3f}  sd=—      min={vals[0]:.3f}  med={vals[0]:.3f}  max={vals[0]:.3f}"
    return (f"n={len(vals):<3d} mean={statistics.mean(vals):.3f}  "
            f"sd={statistics.stdev(vals):.3f}  min={min(vals):.3f}  "
            f"med={statistics.median(vals):.3f}  max={max(vals):.3f}")


def cmd_tally(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    try:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        LOG.error("no such file: %s — run `emit` first", csv_path)
        return 2
    except OSError as exc:
        LOG.error("could not read %s: %s", csv_path, exc)
        return 2

    bad, unscored = [], 0
    clean = []
    for lineno, r in enumerate(rows, start=2):  # +2: header is line 1
        v = (r.get("verdict") or "").strip().lower()
        if not v:
            unscored += 1
            continue
        if v not in VERDICTS:
            bad.append((lineno, v))
            continue
        r["verdict"] = v
        clean.append(r)

    if bad:
        LOG.error("invalid verdict values (line: value): %s", bad)
        LOG.error("allowed: %s", ", ".join(VERDICTS))
        return 2
    if unscored:
        LOG.warning("%d row(s) still have an empty verdict — excluded from the tally. "
                    "Finish them or the reported n is wrong.", unscored)
    if not clean:
        LOG.error("no scored rows")
        return 2

    scored = [r for r in clean if r["verdict"] != "skip"]
    n = len(scored)
    n_skip = len(clean) - n
    frames = sorted({r["frame"] for r in scored})

    counts = {v: sum(1 for r in scored if r["verdict"] == v) for v in VERDICTS[:3]}

    print()
    print("=" * 74)
    print("Prompt-free YOLOE label quality, counted")
    print("=" * 74)
    print(f"source        : {csv_path}")
    print(f"frames        : {len(frames)}  ({', '.join(frames)})")
    print(f"boxes scored  : n = {n}" + (f"   (+{n_skip} skipped)" if n_skip else ""))
    print(f"boxes/frame   : {n / len(frames):.1f} mean")
    print()
    for v in VERDICTS[:3]:
        c = counts[v]
        bar = "#" * int(round(40 * c / n)) if n else ""
        print(f"  {v:<10s} {c:>3d}  {100.0 * c / n:5.1f}%  {bar}")
    print()

    # --- does confidence separate correct from not-correct? -------------
    def scores_for(pred) -> list[float]:
        out = []
        for r in scored:
            if not pred(r):
                continue
            s = (r.get("score") or "").strip()
            if s:
                try:
                    out.append(float(s))
                except ValueError:
                    pass
        return out

    print("-" * 74)
    print("Confidence vs. correctness")
    print("-" * 74)
    for v in VERDICTS[:3]:
        print(f"  {v:<10s} {_describe(scores_for(lambda r, v=v: r['verdict'] == v))}")

    pos = scores_for(lambda r: r["verdict"] == "correct")
    neg = scores_for(lambda r: r["verdict"] in ("plausible", "wrong"))
    auc = _auc(pos, neg)
    print()
    if auc is None:
        print("  AUC: not computable (one class is empty)")
    else:
        print(f"  AUC (confidence predicting `correct` vs not) = {auc:.3f}"
              f"   [n_correct={len(pos)}, n_other={len(neg)}]")
        if abs(auc - 0.5) < 0.10:
            print("  → indistinguishable from chance: confidence carries essentially no")
            print("    information about label correctness, now quantified.")
        elif auc > 0.5:
            print("  → some signal: higher confidence weakly associates with correctness.")
            print("    Must be stated as 'weak separability', not 'none'.")
        else:
            print("  → inverted: higher confidence associates with INCORRECT labels.")
            print("    Surprising — re-check the rubric application before reporting.")
        if pos and neg:
            print(f"  overlap: correct range [{min(pos):.2f}, {max(pos):.2f}] vs "
                  f"other range [{min(neg):.2f}, {max(neg):.2f}]")
    print()
    print("  Caveat for the report: n is small and boxes within a frame are not")
    print("  independent. Report the AUC with n, do not attach a p-value.")
    print()

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prompt-free YOLOE label quality on the frozen "
        f"{len(frames)}-frame set (scene 40753679), $n={n}$ detections, single rater, "
        r"rubric fixed before scoring.}",
        r"\label{tab:detection-label-quality}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Verdict & Count & Share \\",
        r"\midrule",
    ]
    for v in VERDICTS[:3]:
        latex.append(f"{v.capitalize()} & {counts[v]} & {100.0 * counts[v] / n:.1f}\\% \\\\")
    latex += [
        r"\midrule",
        f"Total & {n} & 100.0\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_str = "\n".join(latex)

    if args.latex_out:
        out = Path(args.latex_out)
        if out.exists() and not args.overwrite:
            LOG.error("%s exists; pass --overwrite to replace", out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=".tmp_", suffix=".tex")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(latex_str + "\n")
            os.replace(tmp, out)
            LOG.info("LaTeX table → %s", out)
    else:
        print("-" * 74)
        print(latex_str)
        print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tabulate YOLOE label quality and derive confidence separability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="write the scoring CSV skeleton")
    e.add_argument("--boxes", required=True,
                   help="YOLOE detections JSON, e.g. /data/mmc-user/outputs/yoloe_boxes/40753679.json")
    e.add_argument("--csv", required=True, help="output CSV path")
    e.add_argument("--frames", nargs="*", default=None,
                   help="restrict to these frame keys (default: all in the JSON)")
    e.add_argument("--img-w", type=int, default=256, help="image width for area_frac (informational)")
    e.add_argument("--img-h", type=int, default=192, help="image height for area_frac (informational)")
    e.add_argument("--overwrite", action="store_true",
                   help="replace an existing CSV (destroys hand-entered verdicts)")
    e.set_defaults(func=cmd_emit)

    t = sub.add_parser("tally", help="summarise the filled CSV")
    t.add_argument("--csv", required=True, help="the filled scoring CSV")
    t.add_argument("--latex-out", default=None, help="write the LaTeX table fragment here")
    t.add_argument("--overwrite", action="store_true", help="allow replacing --latex-out")
    t.set_defaults(func=cmd_tally)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_signal_handlers()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())