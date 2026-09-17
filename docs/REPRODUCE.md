# Reproducing the results in this repository

Everything below was checked by direct inspection on the host that produced
the reported numbers. Where something could not be verified from this machine
it is marked **(not verified)**. Absolute paths are the ones the scripts use as
defaults on that host; a fresh clone must override them (see section 7).

---

## 1. What this repo produces

The scripts build per-frame 2D scene graphs over RGB frames: a vision-language
model (VLM) proposes object tags for a frame, GroundingDINO grounds each tag to
a 2D box, and the VLM is then shown the numbered boxes and asked for relation
triples between them. The reported results are computed on a 17-frame subset
of 3DSSG (4 scans), scoring the predicted graphs against the 3DSSG ground truth
with Hungarian box matching followed by object, predicate and triplet
Recall@K, pooled and per stratum. An earlier ARKitScenes pass exists under
`legacy/` and is not part of the reported results.

Repo layout, top level:

```
.
├── .gitignore
├── data/    (the 17 shipped upright frames, section 5)
├── docs/    (this file)
├── legacy/  (superseded ARKitScenes v1 pass, not part of the reported results)
└── src/     (ingest/ gt/ geometry/ pipeline/ eval/ viz/ — everything run in section 6)
```

`legacy/`, `data/` and `docs/` are siblings of `src/`, not nested inside it.

---

## 2. Two machines

| Role | Hardware | What runs there |
|---|---|---|
| Detection / evaluation / local compute host (the machine this doc was written on) | NVIDIA TITAN X (Pascal), 12288 MiB, compute capability 6.1, driver 580.173.02 | every script in this repo: GroundingDINO grounding, all GT and eval scripts, visualisation |
| VLM-serving host, reached at `http://10.121.198.34:11434` | Ollama 0.30.10 (version read over HTTP). Reported by the project as an RTX 3090 **(GPU model not verified from here)** | Ollama only. It serves `qwen3-vl:8b-instruct-bf16`, `qwen2.5vl:7b-fp16`, `llama3.2-vision:11b`, `llava:7b-v1.6-vicuna-fp16` (listed via `/api/tags`) |

No script needs both machines at once except over HTTP: the two VLM stages
(`pipeline/tag_frames_vlm.py`, `pipeline/extract_relations_vlm.py`) and the
ablation probe encode the image into the request body, so the serving host
needs no access to the dataset.

`pipeline/extract_relations_vlm.py` also knows a second backend,
`--backend server` at `http://10.121.198.34:8765/ask` (a bf16 `vlm_server.py`,
now under `legacy/vlm_serving_wasti_old/`). That endpoint was **unreachable**
when this document was written; the reported 3DSSG relations were produced
with `--backend ollama --model qwen3-vl:8b-instruct-bf16` (the archived files
carry the method tag `ii_vlmfirst_qwen3vl_3dssg` and the tag files record
`"model": "qwen3-vl:8b-instruct-bf16"`).

---

## 3. Environment

conda env `ram_pipeline`, Python 3.10.20 (`/data/mmc-user/envs/miniconda3/envs/ram_pipeline`).

```bash
source /data/mmc-user/envs/miniconda3/etc/profile.d/conda.sh   # needed in a non-login shell
conda activate ram_pipeline
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# expected: 2.5.1+cu121 ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

A bare `conda activate ram_pipeline` fails with `CondaError: Run 'conda init'
before 'conda activate'` unless the shell has been `conda init`-ed; the `source`
line above is the portable fix. Calling the interpreter by absolute path
(`/data/mmc-user/envs/miniconda3/envs/ram_pipeline/bin/python`) works equally
well and is what the verification run used.

**torch MUST be 2.5.1+cu121.** The GPU is compute capability 6.1 (Pascal).
torch 2.6 and later drop Pascal from their build and fail at the first kernel
launch with `cudaErrorNoKernelImageForDevice`. sm_61 is not listed in the arch
list above; that is fine because sm_60 code is binary-compatible with sm_61
hardware, which is why 2.5.1+cu121 works on this card.

Do not run the scripts from the `base` env on this host: it carries Python
3.13 and torch 2.12.1+cu130 (arch list sm_75 and up), which prints a
"not compatible" warning and cannot run GroundingDINO on this GPU.

Other versions in `ram_pipeline`, for reference: sentence-transformers 2.7.0,
transformers 4.35.2, numpy 2.2.6, scipy 1.15.3, groundingdino 0.1.0 (editable
install from `/data/mmc-user/repos/GroundingDINO`, CUDA extension `_C` builds
and imports once torch is imported first).

The encoder-based eval scripts are run with `HF_HUB_OFFLINE=1`.

---

## 4. External assets NOT in this repo

All of these are gitignored or live outside the tree. On a first run they must
be downloaded or copied into place; nothing below ships with the repo except
the 17 upright frames (section 5).

| Asset | Expected path (script defaults) | How to obtain |
|---|---|---|
| 3DSSG data for the 4 scans: `3dssg_data/<scan>/{semseg.v2.json, sequence/frame-NNNNNN.{color.jpg,pose.txt,rendered.instances.png,rendered.depth.png}, sequence/_info.txt}` | `/data/mmc-user/3dssg/3dssg_data/` | 3RScan / 3DSSG download for scans `0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca`, `5341b7a5-8a66-2cdd-8751-70b98263cb8d`, `5341b7bf-8a66-2cdd-8794-026113b7c312`, `5630cfd8-12bf-2860-8773-e3dde9da2aff` (the host holds them unzipped from `3dssg_data.zip`, 531 MB) |
| 3DSSG annotations `objects.json`, `relationships.json` | `/data/mmc-user/3dssg/objects.json`, `/data/mmc-user/3dssg/relationships.json` | 3DSSG release |
| Official 160-class list `classes.txt` (160 entries) | `~/3dssg_subset_official/3DSSG_subset/classes.txt` | 3DSSG subset release; the eval scripts gate on exactly 160 entries. **The file has no trailing newline, so `wc -l` reports 159** — that is the correct file, not a truncated one. Verify with `python -c "print(len(open(p).read().splitlines()))"`, which is how the scripts count, and which gives 160 |
| Sentence encoder `all-mpnet-base-v2` | `~/hf_stage/all-mpnet-base-v2` | `sentence-transformers/all-mpnet-base-v2` from the Hugging Face Hub, saved to that directory; passed via `--mpnet` |
| Sentence encoder `clip-ViT-L-14` | `~/hf_stage/clip-ViT-L-14` | `sentence-transformers/clip-ViT-L-14`, saved to that directory; passed via `--clip` |
| Sentence encoder `all-MiniLM-L6-v2` | Hugging Face cache (`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`), resolved by name with `HF_HUB_OFFLINE=1` | `--minilm` defaults to the name `all-MiniLM-L6-v2`; download it once online, or pass a local directory |
| `bert-base-uncased` (GroundingDINO text encoder) | Hugging Face cache (`~/.cache/huggingface/hub/models--bert-base-uncased`) | pulled automatically by GroundingDINO on first use if the cache is empty and the machine is online |
| GroundingDINO code + config | `/data/mmc-user/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` | `git clone https://github.com/IDEA-Research/GroundingDINO.git` then `pip install -e .` inside `ram_pipeline` with `CUDA_HOME` set; passed via `--config` |
| GroundingDINO Swin-T weights | `~/hf_stage/groundingdino/groundingdino_swint_ogc.pth` (same staging convention as the two encoders above) | see below; passed via `--checkpoint` |

GroundingDINO weights, as downloaded for this document:

```bash
mkdir -p ~/hf_stage/groundingdino
wget -O ~/hf_stage/groundingdino/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

That URL returned HTTP 200 and a 693,997,677-byte file (a zip-format torch
checkpoint). Its checksum:

```
sha256  3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
```

This hash was **computed locally and is not verified against a published
source**: the GroundingDINO repository README links the same URL but publishes
no checksum, and none was found elsewhere in the repository.

---

## 5. The upright frame tree

3DSSG colour frames (`frame-NNNNNN.color.jpg`) are stored sideways. They must
be rotated upright before the VLM sees them, because the VLM carries a gravity
prior: on a rotated image its judgement of support relations ("standing on",
"lying on", "hanging on") degrades. Rotate by hand in any image viewer, or in
bulk with a script. The instance masks (`rendered.instances.png`) are already
upright (540x960) and are never rotated.

The 17 frames used for the reported results ship in this repo, already
upright, in the exact layout the pipeline reads. Paths below are relative to
the repo root (`$ROOT`, holding `src/`, `legacy/`, `data/`, `docs/`), not to
`src/`:

```
$ROOT/data/upright_frames_17/<scan>/lowres_wide/<scan>_NNNNNN.png   (17 files, 7.1 MB)
$ROOT/data/upright_frames_17/ingest_identify.json
```

so the rotation step can be skipped. On the original host the same tree lives
at `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training/`, which is what
the script defaults point to. The frame key everywhere downstream is
`<scan>_<NNNNNN>`.

`ingest/identify_rotated_frames.py` exists because the hand-rotated files sit
flat as `frame-NNNNNN.color.jpg` with no scan in the name, and a frame number
is not unique: frame `000078` exists in four scans. It matches each file back
to its scan and frame index by pixel content and writes `ingest_identify.json`
(`frames[*].scan`, `frames[*].frame_no`, `frames[*].k`, `frames[*].ok`), which
18 scripts in this repo read (17 consumers plus the writer). The shipped copy
identifies all 17 frames (`n_ok = 17`): 11 from `5341b7bf...`, 3 from
`5341b7a5...`, 2 from `0a4b8ef6...`, 1 from `5630cfd8...`.

The shipped tree and the original host tree at
`/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training/` were verified
byte-identical (all 17 PNGs, plus an md5-identical `ingest_identify.json`), so
substituting one for the other changes nothing.

### 5a. What a fresh clone can and cannot reproduce

The clone carries 69 tracked files: code, the 17 upright frames, and
`ingest_identify.json`. **No VLM output ships** — no tag files, no
GroundingDINO boxes, no relation JSONs. That splits the run order in two:

| Steps | Needs | Reproducible from the clone + the 3DSSG download alone? |
|---|---|---|
| 1-4, 14 | CPU only | **Yes.** Fully deterministic. Control gate 1 is re-derivable here. |
| 5, 7, 16 | the Ollama host | No. Requires `10.121.198.34:11434` serving `qwen3-vl:8b-instruct-bf16`. |
| 6 | the local CUDA GPU | No. Requires GroundingDINO weights + a Pascal-compatible torch. |
| 8-13, 15 | output of step 7 | No — not directly. They read relation JSONs that are not shipped. |

So **control gate 2 and every reported metric depend on regenerating stages
1-3 first.** Gate 2's `58/133` is a fingerprint of one specific VLM run, not a
property a reproducer can derive from what the repo contains.

---

## 6. Run order

Each step is tagged `[VLM]` (needs the Ollama host), `[GPU]` (needs the local
CUDA GPU) or `[local]` (CPU only; the encoder steps load sentence-transformers
on whatever device is available). Commands are written for the original host
paths; substitute yours. `ROOT` is the repo root (holding `src/`, `legacy/`,
`data/`, `docs/`); `SRC` is the code directory the scripts are run from.

```bash
conda activate ram_pipeline
export HF_HUB_OFFLINE=1
ROOT=/data/mmc-user/projects
SRC=$ROOT/src
cd $SRC
DSSG=/data/mmc-user/3dssg
UPRIGHT=/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training   # or $ROOT/data/upright_frames_17
FRAMES_JSON=/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json   # or $ROOT/data/upright_frames_17/ingest_identify.json
CLASSES=~/3dssg_subset_official/3DSSG_subset/classes.txt
OUT=/data/mmc-user/outputs
```

⚠ **`$OUT` is also where the archived reference artifacts live** (section 10).
On the original host the run directory and the archive were the same directory,
so following the commands below verbatim **overwrites the reference outputs in
place**, with no prompt. Two consequences worth knowing before you start:

* Point `OUT` at an empty directory if you want the archive preserved, and read
  the reference copies from `/data/mmc-user/outputs/` instead.
* Step 12's third command consumes `$OUT/triplet_rk_v3_shifted_product.json`,
  which step 12's *second* command writes. Run them in order, or
  `--v3-baseline-json` silently chains onto whatever archived file is already
  there — the flag's default points at the archive.

1. `[local]` Identify the hand-rotated frames (skip when using the shipped `ingest_identify.json`; needs the flat `frame-NNNNNN.color.jpg` files, which are not shipped):
   ```bash
   python ingest/identify_rotated_frames.py --flat-dir /data/mmc-user/outputs/3dssg_upright_v2 \
       --dssg-root $DSSG --json-out $OUT/3dssg_upright_v2/ingest_identify.json
   ```
2. `[local]` Optional pre-checks on the dataset copy. Read-only.
   ```bash
   python geometry/verify_3dssg.py --dssg-root $DSSG
   python geometry/upright_projection_direction_check.py --dssg-root $DSSG --rotation-json $FRAMES_JSON \
       --scan 5341b7bf-8a66-2cdd-8794-026113b7c312 --frame 6 --json-out $OUT/g10/g10_5341b7bf_000006.json --viz-out $OUT/g10/g10_5341b7bf_000006.jpg
   ```
   `verify_3dssg.py` runs three checks, and only the first two gate anything.
   Read its exit code as V1 ∧ V2; V3 is labelled `DIAG` and is excluded.

   * **V1, mask encoding.** Confirms the instance mask still reads back as
     16-bit. Its stronger proof — "some pixel exceeds 255" — is unreachable on
     this subset: the highest objectId across all 1,120 frames of the four
     scans is **251**, so no frame can supply it. Do not hunt for one. The
     dtype/mode test is sufficient here, because an 8-bit recompression reads
     back as `uint8` and fails the check outright.
   * **V2, annotation coverage.** Meaningful as written. Expect per-scan
     totals `0a4b8ef6 225/30`, `5341b7a5 52/11`, `5341b7bf 540/40`,
     `5630cfd8 254/40` (relationships/objects).
   * **V3, camera convention — diagnostic only, and expected to print
     `INCONCLUSIVE` on the default frame.** It fits each of the three camera
     axis roles by independent per-cue argmax, so one axis can win two roles;
     on the default scan/frame it does exactly that (n=3 usable objects, two of
     them filling >25% of the frame). It also reports `H_tcw` as the better
     metric fit, which is the *opposite* of the `inv(pose)` (`H_twc`)
     convention hardcoded in the four scripts that actually consume pose —
     `geometry/upright_projection_direction_check.py`, `gt/dir_rule_divergence.py`,
     `gt/depth_axis_check.py`, `viz/viz_gt_vlm_panels.py`. **That disagreement
     is not a reason to change those scripts**: at this n the fit cannot
     separate the hypotheses. Nothing in the scored path is affected either
     way — `eval/node_match.py` and every `*_recall_at_k` script derive GT
     boxes from the instance mask and never load a pose file at all.
3. `[local]` GT census of scoreable edges per frame:
   ```bash
   python gt/census_3dssg_gt.py --frames-json $FRAMES_JSON --dssg-root $DSSG --json-out $OUT/3dssg_upright_v2/gt_census.json
   ```
4. `[local]` **Control gate 1**: the 216-item stratified population (section 8):
   ```bash
   python gt/stratify_population.py --frames-json $FRAMES_JSON --dssg-root $DSSG --json-out $OUT/population_table.json
   ```
5. `[VLM]` Stage 1, tags. One call per scan; `--frames` are the frame keys of that scan (from `ingest_identify.json`), `.png` is appended:
   ```bash
   python pipeline/tag_frames_vlm.py \
       --frames-dir $UPRIGHT/5341b7bf-8a66-2cdd-8794-026113b7c312/lowres_wide \
       --frames 5341b7bf-8a66-2cdd-8794-026113b7c312_000006 5341b7bf-8a66-2cdd-8794-026113b7c312_000030 ... \
       --out-json $OUT/3dssg_vlmfirst_tags/5341b7bf-8a66-2cdd-8794-026113b7c312.json \
       --host http://10.121.198.34:11434 --model qwen3-vl:8b-instruct-bf16 --tag-style instances --dump-prompt
   ```
6. `[GPU]` Stage 2, boxes. One call per scan:
   ```bash
   python pipeline/ground_tags_dino.py \
       --tags-json $OUT/3dssg_vlmfirst_tags/5341b7bf-8a66-2cdd-8794-026113b7c312.json \
       --frames-dir $UPRIGHT/5341b7bf-8a66-2cdd-8794-026113b7c312/lowres_wide --ext .png \
       --out-json $OUT/3dssg_vlmfirst_boxes/5341b7bf-8a66-2cdd-8794-026113b7c312.json \
       --config /data/mmc-user/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
       --checkpoint ~/hf_stage/groundingdino/groundingdino_swint_ogc.pth \
       --box-threshold 0.35 --text-threshold 0.25
   ```
7. `[VLM]` Stage 3, relations. Frames from several scans may be passed at once; the script groups them by scan and reads `<loc-dir>/<scan>.json`. It writes `<frame>__<method>.json` and `<frame>__<method>_viz.jpg` side by side into `--out-dir`. (The archived run keeps its files under `relations_3dssg_vlmfirst/json/`, `viz/` and `input/`; the eval scripts only need the directory holding the JSON files.)
   ```bash
   python pipeline/extract_relations_vlm.py \
       --frames 5341b7bf-8a66-2cdd-8794-026113b7c312_000006 5341b7bf-8a66-2cdd-8794-026113b7c312_000030 ... \
       --backend ollama --model qwen3-vl:8b-instruct-bf16 --method ii_vlmfirst_qwen3vl_3dssg \
       --loc-dir $OUT/3dssg_vlmfirst_boxes --raw-root $UPRIGHT --out-dir $OUT/relations_3dssg_vlmfirst/json
   ```
8. `[local]` **Control gate 2**: Hungarian node matching and the detection ceiling (section 8):
   ```bash
   python eval/node_match.py --frames-json $FRAMES_JSON --dssg-root $DSSG \
       --rel-dir $OUT/relations_3dssg_vlmfirst/json --iou-thresh 0.5 --json-out $OUT/node_match_v1.json
   # sensitivity variant (section 8); expect 68/133 and control_pass=false, both by design
   python eval/node_match.py --frames-json $FRAMES_JSON --dssg-root $DSSG \
       --rel-dir $OUT/relations_3dssg_vlmfirst/json --iou-thresh 0.3 --json-out $OUT/node_match_v1_iou03.json
   ```
   `--iou-thresh 0.5` is the default and is written out here only because
   section 8 names it as part of the gate. Every `_iou03.json` artifact in
   section 10 is produced the same way: rerun the same command with
   `--iou-thresh 0.3` and an `_iou03` output path.
9. `[local]` Encoder designation probe (independent of the headline metric):
   ```bash
   python eval/encoder_probe.py --classes $CLASSES --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --json-out ~/encoder_probe_results.json
   ```
10. `[local]` Object R@K:
    ```bash
    python eval/object_recall_at_k.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --classes $CLASSES --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --json-out $OUT/object_rk_results.json
    ```
11. `[local]` Predicate R@K, pooled, then per stratum:
    ```bash
    python eval/predicate_recall_at_k_baseline.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --json-out $OUT/predicate_rk_results.json
    python eval/predicate_recall_at_k_per_stratum.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --json-out $OUT/predicate_rk_v3_stratum.json
    ```
12. `[local]` Triplet R@K, pooled (one run per score function: `product`, `shifted_product`, `sum`, `mean`), then per stratum with the `shifted_product` primary, then the fixed-endpoint variant (which reads the per-stratum JSON via `--v3-baseline-json`):
    ```bash
    # one run per score function; product shown in full, the other three differ only in these two flags
    python eval/triplet_recall_at_k_baseline.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --classes $CLASSES --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --score-fn product --json-out $OUT/triplet_rk_product.json
    #   ... --score-fn shifted_product --json-out $OUT/triplet_rk_shifted_product.json
    #   ... --score-fn sum            --json-out $OUT/triplet_rk_sum.json
    #   ... --score-fn mean           --json-out $OUT/triplet_rk_mean.json
    python eval/triplet_recall_at_k_per_stratum.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --classes $CLASSES --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --score-fn shifted_product --json-out $OUT/triplet_rk_v3_shifted_product.json
    python eval/triplet_recall_at_k_fixed_endpoints_per_stratum.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --classes $CLASSES --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 \
        --v3-baseline-json $OUT/triplet_rk_v3_shifted_product.json --json-out $OUT/triplet_rk_v4_fixedobj.json
    ```
13. `[local]` Supporting evidence (each reads the same inputs as step 8):
    ```bash
    python eval/per_frame_summary.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json \
        --classes $CLASSES --encoder ~/hf_stage/all-mpnet-base-v2 --encoder-name all-mpnet-base-v2 --json-out $OUT/per_frame_summary.json
    python eval/unmatched_vlm_relations.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json --json-out $OUT/unmatched_vlm_relations.json
    python eval/predicate_rank_matrix.py --mpnet ~/hf_stage/all-mpnet-base-v2 --clip ~/hf_stage/clip-ViT-L-14 --json-out $OUT/predicate_rank_matrix.json
    python eval/frame_forensics.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json --frames 000199,000175
    python gt/pull_inverse_evidence.py --frames-json $FRAMES_JSON --dssg-root $DSSG --rel-dir $OUT/relations_3dssg_vlmfirst/json
    ```
14. `[local]` GT-only analyses (no VLM output needed):
    ```bash
    python gt/relation_pair_ordering_check.py --relationships $DSSG/relationships.json --dssg-root $DSSG/3dssg_data --json-out $OUT/relation_pair_ordering.json
    python gt/depth_axis_check.py --dssg-root $DSSG/3dssg_data --relationships $DSSG/relationships.json --rotation-json $FRAMES_JSON --json-out $OUT/depth_axis.json
    python gt/dir_rule_divergence.py --dssg-root $DSSG/3dssg_data --upright-root $UPRIGHT --rotation-json $FRAMES_JSON \
        --scan 5341b7bf-8a66-2cdd-8794-026113b7c312 --frame 6 --json-out $OUT/dir_rule_divergence_5341b7bf_000006.json --viz-out $OUT/dir_rule_divergence_5341b7bf_000006.jpg
    python gt/count_support_gt.py --dssg-root $DSSG --pairs 5341b7bf-8a66-2cdd-8794-026113b7c312:6 5341b7bf-8a66-2cdd-8794-026113b7c312:30
    ```
15. `[local]` Figures:
    ```bash
    python viz/viz_gt_graph.py --census $OUT/3dssg_upright_v2/gt_census.json --frames-root $UPRIGHT --dssg-root $DSSG --out-dir $OUT/3dssg_gt_viz --strata support
    python viz/viz_gt_graph.py --census $OUT/3dssg_upright_v2/gt_census.json --frames-root $UPRIGHT --dssg-root $DSSG --out-dir $OUT/3dssg_gt_viz_dir --strata directional
    python viz/viz_gt_vlm_panels.py --dssg-root $DSSG/3dssg_data --relationships $DSSG/relationships.json \
        --rel-glob "$OUT/relations_3dssg_vlmfirst/json/*.json" --upright-root $UPRIGHT --rotation-json $FRAMES_JSON --out-dir $OUT/gt_vlm_panels
    ```
16. `[VLM]` then `[local]` Prompt-clause ablation (diagnostic only, not part of any reported metric):
    ```bash
    python viz/overlap_clause_ablation_probe.py --json-dir $OUT/relations_3dssg_vlmfirst/json --raw-root $UPRIGHT --out-dir $OUT/probe_c7
    python viz/overlap_clause_ablation_viz.py --probe-dir $OUT/probe_c7 --json-dir $OUT/relations_3dssg_vlmfirst/json --raw-root $UPRIGHT --out-dir $OUT/probe_c7/viz
    ```
---

## 7. Path table

Every absolute-path default in the 30 non-legacy scripts. A fresh clone must
override each one, either on the command line or by editing the default.

**`--dssg-root` defaults to `/data/mmc-user/3dssg` in 18 scripts** (the number
was counted by grep, not assumed):
`eval/frame_forensics.py`, `eval/node_match.py`, `eval/object_recall_at_k.py`,
`eval/per_frame_summary.py`, `eval/predicate_recall_at_k_baseline.py`,
`eval/predicate_recall_at_k_per_stratum.py`, `eval/triplet_recall_at_k_baseline.py`,
`eval/triplet_recall_at_k_fixed_endpoints_per_stratum.py`,
`eval/triplet_recall_at_k_per_stratum.py`, `eval/unmatched_vlm_relations.py`,
`geometry/upright_projection_direction_check.py`,
`geometry/verify_3dssg.py`, `gt/census_3dssg_gt.py`, `gt/count_support_gt.py`,
`gt/pull_inverse_evidence.py`, `gt/stratify_population.py`,
`ingest/identify_rotated_frames.py`, `viz/viz_gt_graph.py`.
Each pairs it with `--data-subdir 3dssg_data`.

Four more scripts take the subdirectory directly, `--dssg-root /data/mmc-user/3dssg/3dssg_data`, plus `--relationships /data/mmc-user/3dssg/relationships.json`:
`gt/depth_axis_check.py`, `gt/dir_rule_divergence.py`,
`gt/relation_pair_ordering_check.py`, `viz/viz_gt_vlm_panels.py`.

| Script | Flag or constant | Default |
|---|---|---|
| `geometry/upright_projection_direction_check.py` | `--rotation-json` | `/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json` |
| `gt/depth_axis_check.py` | `--rotation-json` | `/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json` |
| `gt/dir_rule_divergence.py` | `--upright-root` | `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training` |
| `gt/dir_rule_divergence.py` | `--rotation-json` | `/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json` |
| `viz/viz_gt_graph.py` | `--frames-root` | `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training` |
| `viz/viz_gt_vlm_panels.py` | `--rel-glob` | `/data/mmc-user/outputs/relations_3dssg_vlmfirst/json/*.json` |
| `viz/viz_gt_vlm_panels.py` | `--upright-root` | `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training` |
| `viz/viz_gt_vlm_panels.py` | `--rotation-json` | `/data/mmc-user/outputs/3dssg_upright_v2/ingest_identify.json` |
| `viz/viz_gt_vlm_panels.py` | `--out-dir` | `/data/mmc-user/outputs/gt_vlm_panels` |
| `viz/overlap_clause_ablation_probe.py` | `--json-dir` | `/data/mmc-user/outputs/relations_3dssg_vlmfirst/json` |
| `viz/overlap_clause_ablation_probe.py` | `--raw-root` | `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training` |
| `viz/overlap_clause_ablation_viz.py` | `--probe-dir` | `/data/mmc-user/outputs/probe_c7` |
| `viz/overlap_clause_ablation_viz.py` | `--json-dir` | `/data/mmc-user/outputs/relations_3dssg_vlmfirst/json` |
| `viz/overlap_clause_ablation_viz.py` | `--raw-root` | `/data/mmc-user/outputs/3dssg_upright_v2/raw/raw/Training` |
| `eval/triplet_recall_at_k_fixed_endpoints_per_stratum.py` | `--v3-baseline-json` | `/data/mmc-user/outputs/triplet_rk_v3_shifted_product.json` |
| `pipeline/tag_frames_vlm.py` | `--host` | `http://10.121.198.34:11434` |
| `pipeline/extract_relations_vlm.py` | `--loc-dir` (module constant `LOC_DIR`) | `/data/mmc-user/outputs/localization_3d` |
| `pipeline/extract_relations_vlm.py` | `--raw-root` (module constant `RAW_ROOT`) | `/data/mmc-user/arkitscenes/raw/raw/Training` |
| `pipeline/extract_relations_vlm.py` | `--out-dir` (module constant `OUT_DIR`) | `/data/mmc-user/outputs/vlm_relations` |
| `pipeline/extract_relations_vlm.py` | module constant `OLLAMA_URL` (no flag) | `http://10.121.198.34:11434/api/generate` |
| `pipeline/extract_relations_vlm.py` | module constant `VLM_SERVER_URL` (no flag) | `http://10.121.198.34:8765/ask` |

Paths that appear only in docstring usage examples, not as defaults:
`~/3dssg_subset_official/3DSSG_subset/classes.txt` (`--classes`, required),
`~/hf_stage/all-mpnet-base-v2` (`--mpnet`, required),
`~/hf_stage/clip-ViT-L-14` (`--clip`, required).
`pipeline/ground_tags_dino.py` has no path defaults at all: `--tags-json`,
`--frames-dir`, `--out-json`, `--config` and `--checkpoint` are all required.

---

## 8. Control gates

Two fingerprint checks. If either fails, something upstream changed; do not
proceed and do not retune the control to match.

**Gate 1: `gt/stratify_population.py`** (step 4). Must report 216 items, strata
S1=35 / S2=62 / S3=45 / S4=68 / S5=5 / SAME_AS=1, and four CONTROL lines that
all PASS (population is 216, strata match, predicate histogram matches the
pre-registered one, id key unique). The script refuses to write its JSON
otherwise unless `--no-control-check` is given.

**Gate 2: `eval/node_match.py --iou-thresh 0.5`** (step 8). Must report
58 of 133 visible ground-truth objects matched (43.6%) and `CONTROL PASS`.

0.5 is the control threshold and the threshold every reported number uses (all
11 scripts that expose `--iou-thresh` default to 0.5). `--iou-thresh 0.3` is a
separate sensitivity variant, which gives 68/133 and is stored as
`*_iou03.json` files; its `control_pass` field is `false` by design. Do not
conflate the two. At any threshold other than 0.5 the script now says the
control is *not applicable* rather than printing a mismatch warning.

**Gate 2 is not derivable from the clone.** It fingerprints one archived VLM
run, and no VLM output ships (section 5a). Reaching it means regenerating
stages 1-3 against the Ollama host first. If it fails after a legitimate
regeneration, suspect the serving stack — model build, Ollama version, sampling
defaults — before suspecting the GT or matcher code. Gate 1, by contrast, is
GT-only and fully re-derivable; it was re-run for this document and reproduced
216 items with all four CONTROL lines passing and an item-for-item identical
population table.

---

## 9. Known quirks

- **`PREDICATE_VOCAB` is duplicated.** The 20-predicate scored vocabulary is
  defined at module level in 11 files (`eval/encoder_probe.py`,
  `eval/frame_forensics.py`, `eval/per_frame_summary.py`,
  `eval/predicate_rank_matrix.py`, `eval/predicate_recall_at_k_baseline.py`,
  `eval/predicate_recall_at_k_per_stratum.py`, `eval/triplet_recall_at_k_baseline.py`,
  `eval/triplet_recall_at_k_per_stratum.py`, `eval/unmatched_vlm_relations.py`,
  `gt/pull_inverse_evidence.py`, `gt/strata.py`) and imported into a twelfth
  (`eval/triplet_recall_at_k_fixed_endpoints_per_stratum.py`, which imports it
  from `eval/triplet_recall_at_k_per_stratum.py`, not from `strata`). That
  leaves **10 copies besides `gt/strata.py`, and none is byte-identical to it**:
  `strata.py` holds a tuple, eight files hold a list, and
  `eval/frame_forensics.py` + `eval/unmatched_vlm_relations.py` hold a set. All
  10 carry the same 20 strings in the same order — verified, not assumed. The
  canonical definition is `gt/strata.py` lines 33-39. Change it there and
  propagate to all 10.
- **`pipeline/extract_relations_vlm.py --raw-root` defaults to an ARKitScenes
  path** (`/data/mmc-user/arkitscenes/raw/raw/Training`), as do `--loc-dir`
  and `--out-dir`. The reported results are 3DSSG: always pass all three.
  `--loc-dir` must point at GroundingDINO output from step 6; `localization_3d`
  and `vlmfirst_boxes` are two different detection front ends and must not be
  mixed. `--backend` defaults to `server`, whose endpoint was unreachable when
  this was written; the 3DSSG run used `--backend ollama`.
- **`eval/per_frame_summary.py`'s cross-check is manual.** Its docstring says
  the summed per-frame totals "MUST equal" the pooled numbers in
  `triplet_rk_product.json`; the code never reads that file. It prints its own
  totals and you compare them by hand.
- **`--method` is not just a label.** `pipeline/extract_relations_vlm.py:368` is
  `inject = "iii" in args.method`, so a method tag containing `iii` silently
  switches geometry injection on and changes the prompt. Renaming a run changes
  what the VLM is asked. Every `iii*` file in the archive carries
  `geometry_injected: true`; every other file carries `false`.
- **The relation JSON records no model, no backend and no timestamp** — only the
  free-text `method` string. Provenance for `$OUT/vlm_relations/` rests entirely
  on that tag plus file mtimes. The one exception is the VLM-first lineage,
  whose model is recorded upstream in `vlmfirst_tags/<scene>.json`.
- **`extract_relations_vlm.py` has no resume.** It never checks for existing
  outputs; a rerun overwrites in place. `tag_frames_vlm.py` and
  `ground_tags_dino.py` do refuse, requiring `--resume` or `--overwrite`.

---

## 10. Example outputs

Artifacts of the reported run, as found on the original host under
`/data/mmc-user/outputs/` (`$OUT`). Scripts marked *console only* write no
file unless `--json-out` is given, and no such file was archived.

| Script | Example artifact |
|---|---|
| `ingest/identify_rotated_frames.py` | `$OUT/3dssg_upright_v2/ingest_identify.json` (also shipped as `$ROOT/data/upright_frames_17/ingest_identify.json`) |
| `geometry/verify_3dssg.py` | console only |
| `geometry/upright_projection_direction_check.py` | `$OUT/g10/` (per-frame JSON + JPEG, 12 files) |
| `gt/census_3dssg_gt.py` | `$OUT/3dssg_upright_v2/gt_census.json` |
| `gt/stratify_population.py` | `$OUT/population_table.json` |
| `gt/count_support_gt.py` | console only |
| `gt/depth_axis_check.py` | `$OUT/depth_axis.json` (console only without `--json-out`) |
| `gt/dir_rule_divergence.py` | `$OUT/dir_rule_divergence_<scan>_<frame>.{json,jpg}` (console only without `--json-out`/`--viz-out`) |
| `gt/pull_inverse_evidence.py` | console only (writes nothing by design) |
| `gt/relation_pair_ordering_check.py` | `$OUT/relation_pair_ordering.json` |
| `gt/strata.py` | none (library module) |
| `pipeline/tag_frames_vlm.py` | `$OUT/3dssg_vlmfirst_tags/<scan>.json` (4 scans) + `<scan>.json.prompt.txt`; ARKitScenes: `$OUT/vlmfirst_tags/40753679.json` (Appendix A) |
| `pipeline/ground_tags_dino.py` | `$OUT/3dssg_vlmfirst_boxes/<scan>.json` (4 scans) + `viz/<frame>_viz.jpg`; ARKitScenes: `$OUT/vlmfirst_boxes/40753679.json` (Appendix A) |
| `pipeline/extract_relations_vlm.py` | `$OUT/relations_3dssg_vlmfirst/json/<frame>__ii_vlmfirst_qwen3vl_3dssg.json` (17 files) + `viz/`, `input/`; ARKitScenes: `$OUT/vlm_relations/<frame>__<method>.{json,_viz.jpg}` (43 JSON + 56 JPG, 1 scene, 6 frames — Appendix A) |
| `eval/node_match.py` | `$OUT/node_match_v1.json` (and `node_match_v1_iou03.json`) |
| `eval/encoder_probe.py` | `~/encoder_probe_results.json` |
| `eval/object_recall_at_k.py` | `$OUT/object_rk_results.json` (and `_iou03`) |
| `eval/predicate_recall_at_k_baseline.py` | `$OUT/predicate_rk_results.json` |
| `eval/predicate_recall_at_k_per_stratum.py` | `$OUT/predicate_rk_v3_stratum.json` (and `_iou03`) |
| `eval/triplet_recall_at_k_baseline.py` | `$OUT/triplet_rk_product.json`, `triplet_rk_shifted_product.json`, `triplet_rk_sum.json`, `triplet_rk_mean.json` |
| `eval/triplet_recall_at_k_per_stratum.py` | `$OUT/triplet_rk_v3_shifted_product.json` (and `_iou03`, `_iou03_K50_100_500`) |
| `eval/triplet_recall_at_k_fixed_endpoints_per_stratum.py` | `$OUT/triplet_rk_v4_fixedobj.json` (and `_iou03`) |
| `eval/per_frame_summary.py` | `$OUT/per_frame_summary.json` |
| `eval/unmatched_vlm_relations.py` | `$OUT/unmatched_vlm_relations.json` |
| `eval/predicate_rank_matrix.py` | `$OUT/predicate_rank_matrix.json` |
| `eval/frame_forensics.py` | console only |
| `viz/viz_gt_graph.py` | `$OUT/3dssg_gt_viz/` (17 PNGs) and `$OUT/3dssg_gt_viz_dir/` (17 PNGs, `--strata directional`) |
| `viz/viz_gt_vlm_panels.py` | `$OUT/gt_vlm_panels/` (18 files) |
| `viz/overlap_clause_ablation_probe.py` | `$OUT/probe_c7/<frame>__V{0,1,2,3}.json` |
| `viz/overlap_clause_ablation_viz.py` | `$OUT/probe_c7/viz/` (5 PNGs) |

---

## Appendix A. The ARKitScenes v1 corpus (archive, not a reported result)

`$OUT/vlm_relations/` holds the earlier ARKitScenes pass. **No number in the
report comes from it** — section 1 already places it out of scope, and it feeds
no control gate. It is documented here because the archive is on disk, readers
find it, and one of the two lineages inside it is still live code. Reproducing
it reproduces an *archive*, not a result.

The directory contains 43 JSON + 56 JPG covering **one scene (`40753679`) and
six frames**. Those files came from two different front ends:

| Lineage | Method tags | Detections from | Reproducible? |
|---|---|---|---|
| VLM-first | `ii_vlmfirst_qwen3vl` | `$OUT/vlmfirst_boxes/` | **Yes** — A.1 below |
| Geometry-primary | bare, `ii`, `ii_server_check`, `ii_ollama`, `ii_qwen3vl`, `iii_qwen`, `iii_v2_qwen` | `$OUT/localization_3d/` | **No** — see A.3 |

Tell them apart by `objects[*].bbox_3d`: `null` in the VLM-first lineage,
populated in the geometry-primary one.

### A.1 Reproducing the VLM-first variant

Same three stages as section 6 steps 5-7, pointed at ARKitScenes. Only the
stage-2 thresholds differ from the 3DSSG run — see the warning after step A3.

```bash
source /data/mmc-user/envs/miniconda3/etc/profile.d/conda.sh
conda activate ram_pipeline
ROOT=/data/mmc-user/projects; SRC=$ROOT/src; cd $SRC
ARK=/data/mmc-user/arkitscenes/raw/raw/Training
SCENE=40753679
AOUT=<a fresh directory>     # NOT /data/mmc-user/outputs — that IS the archive
FR="40753679_6804.958 40753679_6811.522 40753679_6813.238 \
    40753679_6832.979 40753679_6841.941 40753679_6844.890"
DINO_CFG=/data/mmc-user/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
DINO_W=~/hf_stage/groundingdino/groundingdino_swint_ogc.pth
```

⚠ All three `extract_relations_vlm.py` path defaults point *into* the archive,
and this is the only run in this document where they are the correct values.
Override them anyway, or a rerun overwrites the archive in place — that script
has no resume and does not ask.

**A1** `[VLM]` Stage 1, tags:
```bash
python pipeline/tag_frames_vlm.py --frames-dir $ARK/$SCENE/lowres_wide --frames $FR --ext .png \
    --out-json $AOUT/vlmfirst_tags/$SCENE.json \
    --host http://10.121.198.34:11434 --model qwen3-vl:8b-instruct-bf16 \
    --tag-style instances --dump-prompt
```

**A2** `[GPU]` Stage 2, boxes:
```bash
python pipeline/ground_tags_dino.py --tags-json $AOUT/vlmfirst_tags/$SCENE.json \
    --frames-dir $ARK/$SCENE/lowres_wide --ext .png \
    --out-json $AOUT/vlmfirst_boxes/$SCENE.json \
    --config $DINO_CFG --checkpoint $DINO_W \
    --box-threshold 0.30 --text-threshold 0.25 --dedupe-iou 0.6
```

**A3** `[VLM]` Stage 3, relations:
```bash
python pipeline/extract_relations_vlm.py --frames $FR \
    --backend ollama --model qwen3-vl:8b-instruct-bf16 --method ii_vlmfirst_qwen3vl \
    --loc-dir $AOUT/vlmfirst_boxes --raw-root $ARK --out-dir $AOUT/vlm_relations
```

**`--box-threshold 0.30 --dedupe-iou 0.6` is not a typo.** Section 6 step 6 uses
`0.35` and the default dedupe of `0.9`; those values do **not** reproduce this
archive. The archived boxes contain a detection scoring `0.3112`, which `0.35`
cannot admit (`score_2d` is the same logit the threshold filters on,
`pipeline/ground_tags_dino.py:364`), and at `--dedupe-iou 0.9` a near-duplicate
`wooden table` box survives on frame `6804.958` that the archive does not have.
With `0.30` / `0.6` the regenerated boxes are byte-identical to the archive on
all six frames.

Three more things worth knowing before running A3:

* `--backend` defaults to `server` — the retired `:8765` endpoint. It must be
  set to `ollama` explicitly.
* `--loc-dir` takes the **boxes** directory, not `localization_3d`. The two
  front ends must not be mixed (section 9).
* ARKit `lowres_wide` frames are 256x192. The script upscales x5 NEAREST and
  draws numbered boxes on; that annotated image is what the VLM actually sees.

### A.2 Expected output

Six `<frame>__ii_vlmfirst_qwen3vl.json` plus six `_viz.jpg`, flat in
`--out-dir`. `n_objects` per frame, in the frame order above: **5, 7, 5, 4, 2,
5**. Every file must carry `geometry_injected: false`. Detections are filtered
at `--min-score 0.3` and `--max-area-frac 0.85`, and a frame left with fewer
than two detections is skipped entirely.

### A.3 What cannot be re-run, and why

Stages 4-6 of the original ARKitScenes chain — RAM++ tagging, GroundingDINO
boxes, 3D localization — **were not committed to this repo**.

| Stage | Surviving artifact | Evidence it ran |
|---|---|---|
| 4, RAM++ tags | `$OUT/ram_tags/<scene>.json` (12 scenes) | `/data/mmc-user/logs/ram_pipeline/ram_inference/`; `environment.yml` pins `ram==0.0.1`; weights at `/data/checkpoints/ram_plus_plus/ram_plus_swin_large_14m.pth`; clone at `/data/mmc-user/repos/recognize-anything` |
| 5, GroundingDINO | `$OUT/grounding_boxes/<scene>.json` (12) | `/data/mmc-user/logs/ram_pipeline/grounding_dino/` |
| 6, 3D localization | `$OUT/localization_3d/<scene>.json` (12, + `.preoutlier`) | `/data/mmc-user/logs/ram_pipeline/localization/` |

So `localization_3d/<scene>.json` is an **external input that can be copied but
not regenerated**, and every method tag in the geometry-primary lineage depends
on it. Two of those tags could not be reproduced even with it: the bare files
predate the `--method` flag and carry no `method` key, and `iii_qwen` used a
retired v1 geometry hint that the current `geometry_hint()` no longer emits.

### A.4 Getting the raw data

ARKitScenes is not in this repo. The host holds 12 usable scenes (5.3 GB) at
`$ARK/<scene>/lowres_wide/<scene>_<timestamp>.png`. Upstream downloader and
split manifest: `/data/mmc-user/repos/ARKitScenes/download_data.py` and
`raw/raw_train_val_splits.csv`. Only scene `40753679` is needed for A.1.
