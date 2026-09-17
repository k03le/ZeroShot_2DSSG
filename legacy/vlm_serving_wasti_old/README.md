Retired, not part of the reported pipeline. wasti-old cannot see the ARKitScenes
data tree and takes no part in any reproducible run — these scripts are kept
for provenance only.

vlm.py / vlm_server.py — Qwen2.5-VL bf16-direct serving path (superseded by
the Ollama stack). Retained because it is the bf16 arm of a reported
serving-stack comparison: bf16-direct and Ollama-served Qwen2.5-VL produced
different relation outputs on the same frame, which is cited in the report.
Do not run as part of reproduction; kept so the comparison is checkable.

grounding.py / parsing.py / visualization.py — standalone open-vocabulary
detection demo built on vlm.py. No reported number traces to this code.
Retained for completeness only; safe to ignore.
