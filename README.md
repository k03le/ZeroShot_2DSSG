# fm-vlm-scene-graphs

Prompt-engineered VLM pipeline for extracting open-vocabulary spatial and
semantic relations between detected objects in single RGB frames, producing
per-frame 2D scene graphs. Evaluated on a 17-frame subset of 3DSSG.

## Quick facts
- Two machines: detection/evaluation runs locally (NVIDIA TITAN X Pascal);
  VLM inference is served remotely over HTTP (Ollama). See
  [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full split.
- torch is pinned to 2.5.1+cu121 for Pascal GPU compatibility — do not
  upgrade it. See `environment.yml` and docs/REPRODUCE.md §3.
- An earlier geometry-primary detection front end exists under `legacy/`
  and is not part of the reported results; see `legacy/*/README.md`.

## Getting started
See [docs/REPRODUCE.md](docs/REPRODUCE.md) for environment setup, external
assets to download, the full run order, and the two control gates that
confirm a working setup (§8).

## Layout
    src/       code (ingest, gt, geometry, pipeline, eval, viz)
    legacy/    retired front ends and serving code, kept for provenance
    data/      the 17 upright frames used in the reported results
    docs/      REPRODUCE.md
    configs/   the frozen VLM prompt (reference copy)
