# configs/

- `prompt_relations_frozen.txt` — reference copy of the frozen VLM relation
  prompt. The actual prompt used at runtime is inlined in
  `src/pipeline/extract_relations_vlm.py`'s `build_prompt()` function
  (verified during this repo's prep: the returned text matches this file
  as of the last check). If they ever diverge, the script is the source
  of truth — this file exists for readability, not execution.

- 160-class vocabulary (`classes.txt`) and the 20-predicate scored
  vocabulary are NOT duplicated here. The former is an external asset
  from the 3DSSG subset release (see docs/REPRODUCE.md §4); the latter
  is defined once, canonically, in `src/gt/strata.py` (PREDICATE_VOCAB,
  lines 33-39) and duplicated internally across 11 evaluation scripts —
  see docs/REPRODUCE.md §9, Known quirks.