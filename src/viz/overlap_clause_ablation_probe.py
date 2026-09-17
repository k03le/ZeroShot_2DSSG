"""
Script:  overlap_clause_ablation_probe.py
Stage:   DIAGNOSTIC ABLATION. Not part of any scored pipeline; results are probe
         artifacts and must be labelled as such wherever they are cited.
Purpose: Three 3DSSG frames abstained ([] relations) in the VLM-predicted-pipeline
         Stage-3 run: 000144, 000446, 000041. Hypothesis: the overlap-caution
         paragraph in the FROZEN relation prompt over-fires on CONTAINMENT geometry
         (small box fully inside a large one — exactly the shape of every 'X on
         shelf/desk/bed' support relation), suppressing support relations; where a
         frame is mostly containment pairs, suppression is total and the '[] if you
         see none' license does the rest.

         A single run with the paragraph removed cannot distinguish that from
         generic perturbation sensitivity (the first-token bifurcation account).
         So three variants:
             V0  frozen prompt, verbatim              (reproduction control)
             V1  overlap-caution paragraph REMOVED    (the hypothesis)
             V2  overlap-caution paragraph REPLACED by a semantically inert line
                 of similar length                     (placebo — perturbation control)
         V1 fires and V2 does not  -> overlap-caution-paragraph-specific, real finding.
         V1 and V2 both fire       -> perturbation sensitivity, not the paragraph's content.
         Neither fires             -> abstention is image-driven; hypothesis dies.

         THE FROZEN PROMPT IS NOT MODIFIED. extract_relations_vlm.py is imported,
         never edited; only the prompt STRING is varied, in memory, here.

DESIGN
    The detection set is NOT re-derived (no filter divergence): the stored
    `objects` list is read from each frame's existing relation JSON — that is
    definitionally what the VLM saw. build_prompt() is then re-run over it and
    ASSERTED byte-equal to the stored `prompt_used`. Mismatch -> abort.

INPUT   --json-dir   relations_3dssg_vlmfirst/json   (Stage-3 output)
        --raw-root   the upright colour tree
OUTPUT  --out-dir    probe_c7/<frame>__<variant>.json + the prompt actually sent
        console table: relations parsed per frame x variant + containment stats
USAGE
    conda activate ram_pipeline
    cd /data/mmc-user/projects/src
    python viz/overlap_clause_ablation_probe.py --out-dir /data/mmc-user/outputs/probe_c7
"""

import argparse
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

# extract_relations_vlm.py lives in ../pipeline/ (this script lives in viz/);
# make it importable at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from PIL import Image

LOG = logging.getLogger("overlap_probe")

# Exact substring identifying the overlap-caution paragraph inside the frozen prompt.
# Matched, not reconstructed: if it is absent the probe aborts rather than guess.
OVERLAP_CLAUSE_ANCHOR = "The boxes are 2D detections and may OVERLAP"

# Placebo: no constraint on relation inference, no mention of overlap,
# containment, or physical contact. Length-matched within ~30%.
PLACEBO = ("NOTE: The numbered boxes were produced by an automatic detector and "
           "their numbering order carries no meaning; index 0 is not more important "
           "than the highest index. Treat all numbered objects as equally relevant "
           "when considering the scene.")

FRAMES_ABSTAINED = [
    "5341b7bf-8a66-2cdd-8794-026113b7c312_000144",
    "0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca_000446",
    "5630cfd8-12bf-2860-8773-e3dde9da2aff_000041",
]
FRAMES_CONTROL = [                      # produced relations under the frozen prompt
    "5341b7bf-8a66-2cdd-8794-026113b7c312_000030",   # emitted a containment-shaped
                                                     # 'book on bookshelf' — the
                                                     # counterexample to the hypothesis
    "5341b7bf-8a66-2cdd-8794-026113b7c312_000175",
]


def containment_stats(objects, thr=0.8):
    """Fraction of object pairs in a containment configuration.

    containment = intersection / area(smaller box). 1.0 => small box fully inside
    the large one. This is the geometry the overlap-caution paragraph targets; IoU would NOT
    capture it (a tiny box inside a huge one has near-zero IoU).
    """
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    n_pair = n_contained = 0
    for a, b in combinations(objects, 2):
        ba, bb = a["bbox_2d"], b["bbox_2d"]
        ix = max(0.0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
        iy = max(0.0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
        small = min(area(ba), area(bb))
        n_pair += 1
        if small > 1e-9 and (ix * iy) / small >= thr:
            n_contained += 1
    return n_contained, n_pair


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlap-caution-paragraph ablation (diagnostic).")
    ap.add_argument("--json-dir",
                    default="/data/mmc-user/outputs/relations_3dssg_vlmfirst/json")
    ap.add_argument("--suffix", default="__ii_vlmfirst_qwen3vl_3dssg.json")
    ap.add_argument("--raw-root",
                    default="/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct-bf16")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--frames", nargs="*", default=None,
                    help="override the default abstainer+control frame set")
    ap.add_argument("--variants", nargs="*", default=["V0", "V1", "V2", "V3"],
                    choices=["V0", "V1", "V2", "V3"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    # import the CANONICAL script; never reimplement overlay / call / parse
    try:
        import extract_relations_vlm as erv
    except SystemExit:
        raise SystemExit("[abort] importing extract_relations_vlm executed its main(); "
                         "it needs an `if __name__ == '__main__':` guard before this "
                         "probe can import it")
    except ImportError as e:
        raise SystemExit(f"[abort] cannot import extract_relations_vlm ({e}); "
                         f"run this from /data/mmc-user/projects/src")
    LOG.info("backend URL in use: %s", getattr(erv, "OLLAMA_URL", "<not found>"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = args.frames or (FRAMES_ABSTAINED + FRAMES_CONTROL)
    rows = []

    for fk in frames:
        jp = Path(args.json_dir) / f"{fk}{args.suffix}"
        try:
            rec = json.loads(jp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            LOG.error("%s: %s — skipped", fk, e)
            continue
        objects = rec["objects"]
        stored_prompt = rec.get("prompt_used")
        if stored_prompt is None:
            raise SystemExit(f"[abort] {jp} has no prompt_used; cannot verify reconstruction")

        # ---- self-check: our rebuild must equal what was actually sent
        p0 = erv.build_prompt(objects, "pass", inject_geometry=False)
        if p0 != stored_prompt:
            raise SystemExit(
                f"[abort] rebuilt prompt != stored prompt_used for {fk}.\n"
                f"  The probe would not be comparing against the frozen prompt.\n"
                f"  len rebuilt={len(p0)} stored={len(stored_prompt)}")
        LOG.info("%s: prompt reconstruction verified (%d objects, %d chars)",
                 fk, len(objects), len(p0))

        # ---- build the variants by exact-line surgery on the frozen prompt
        lines = p0.splitlines()
        hits = [i for i, l in enumerate(lines) if OVERLAP_CLAUSE_ANCHOR in l]
        if len(hits) != 1:
            raise SystemExit(f"[abort] overlap-caution anchor found {len(hits)}x in {fk} — "
                             f"prompt structure changed; inspect before proceeding")
        i = hits[0]
        overlap_clause_line = lines[i]
        variants = {
            "V0": p0,
            "V1": "\n".join(lines[:i] + lines[i + 1:]),
            "V2": "\n".join(lines[:i] + [PLACEBO] + lines[i + 1:]),
            # V3: overlap-caution paragraph RETAINED, inert line inserted immediately before it.
            # This is the true placebo — V2 also deleted the constraint.
            "V3": "\n".join(lines[:i] + [PLACEBO] + lines[i:]),
        }
        LOG.debug("overlap-caution line %d chars; placebo %d chars", len(overlap_clause_line), len(PLACEBO))

        # ---- image + overlay, identical to the canonical path
        scan = fk.rsplit("_", 1)[0]
        ip = Path(args.raw_root) / scan / "lowres_wide" / f"{fk}.png"
        try:
            img = Image.open(ip)
        except OSError as e:
            LOG.error("%s: %s — skipped", fk, e)
            continue
        annotated = erv.draw_numbered(img, objects, args.upscale, "pass")

        n_cont, n_pair = containment_stats(objects)
        row = dict(frame=fk, n_obj=len(objects), n_pair=n_pair, n_contained=n_cont,
                   contained_frac=round(n_cont / n_pair, 3) if n_pair else 0.0,
                   frozen_n=next((len(v) for k, v in rec.items()
                                  if k != "objects" and isinstance(v, list)), -1))

        for v in args.variants:
            dst = out / f"{fk}__{v}.json"
            if dst.exists() and not args.overwrite:
                LOG.info("%s %s exists — skipped", fk, v)
                continue
            try:
                raw = erv.call_vlm(annotated, variants[v], backend="ollama", model=args.model)
            except Exception as e:                      # network / server errors
                LOG.error("%s %s: VLM call failed: %s", fk, v, e)
                row[v] = None
                continue
            triples = erv.parse_triples(raw, len(objects))
            row[v] = len(triples)
            tmp = dst.with_name(f".{dst.stem}.tmp{os.getpid()}{dst.suffix}")
            with open(tmp, "w") as fh:
                json.dump(dict(frame_id=fk, variant=v, model=args.model,
                               n_objects=len(objects), prompt_sent=variants[v],
                               raw_response=raw, triples=triples,
                               PROBE_ARTIFACT="diagnostic ablation — NOT an arm result"),
                          fh, indent=2)
            os.replace(tmp, dst)
            LOG.info("%s %s -> %d relation(s)", fk, v, len(triples))
        rows.append(row)

    print(f"\n{'frame':46s} {'obj':>4s} {'contain%':>9s} {'frozen':>7s} "
          f"{'V0':>4s} {'V1':>4s} {'V2':>4s} {'V3':>4s}")
    for r in rows:
        print(f"{r['frame']:46s} {r['n_obj']:>4d} {r['contained_frac']*100:>8.1f}% "
              f"{r['frozen_n']:>7d} {str(r.get('V0','-')):>4s} "
              f"{str(r.get('V1','-')):>4s} {str(r.get('V2','-')):>4s} "
              f"{str(r.get('V3','-')):>4s}")
    print("\nV0 must reproduce 'frozen' (temp 0, seed 42). If it does not, the run "
          "is non-deterministic and NOTHING here is interpretable.")
    print("V1>0 and V2==0  -> overlap-caution-paragraph-specific suppression (the hypothesis).")
    print("V1>0 and V2>0   -> generic perturbation sensitivity (first-token bifurcation).")
    print("V1==0 and V2==0 -> image-driven abstention; hypothesis dies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())