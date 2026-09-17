Retired, not part of the reported pipeline. This is the earlier
geometry-primary (RAM++ / GroundingDINO) front end and its supporting
diagnostics, superseded by the VLM-primary front end that produced the
reported results: a VLM tags the whole frame in open vocabulary first
(tag_frames_vlm.py), each tag is grounded into a 2D box
(ground_tags_dino.py), then the numbered-box frame is sent back to the VLM
to extract open-vocabulary [subject, relation, object] triples
(extract_relations_vlm.py). No 3D geometry is given to the VLM at any stage.

Retained because this code produced a negative result argued in the report:
coverage and label-quality figures from this front end motivate the switch
to the VLM-primary method. Deleting the code that produced those figures
would leave that argument unverifiable. Not part of any reproduction run.

`count_predicate_diversity.py` is retired for the same reason: it exists to
test whether the VLM's relation vocabulary is genuinely "rich-semantic" or
merely geometry-derivable, but the project's objective was open-vocabulary
relation extraction from the outset — that question should never have been
live. It was also run against the ARKitScenes relation corpus, not 3DSSG.
