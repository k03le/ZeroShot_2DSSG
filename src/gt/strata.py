"""
Module:  strata.py
Stage:   Step 1 — SINGLE SOURCE for stratum assignment.

Purpose: Map one canonical GT item to exactly one stratum. Imported by
         stratify_population.py now, and by predicate_recall_at_k_baseline.py /
         triplet_recall_at_k_baseline.py afterwards, so the decomposition
         cannot drift between three independent implementations.

Input:   gt_predicate (one of the 20 scored predicates), subject label,
         object label (3DSSG objects.json free-text labels).
Output:  one of "S1","S2","S3","S4","S5","SAME_AS":
           S1      = support between two ordinary objects
           S2      = support where one endpoint is architecture (floor/wall/ceiling)
           S3      = proximity ("close by")
           S4      = lateral (left / right)
           S5      = vertical (higher than / lower than)
           SAME_AS = same-instance identity
Usage:   from strata import stratum_of, STRATA, ARCH_KEYS, EXPECTED_STRATA

RULE PROVENANCE
  ARCH_KEYS is copied VERBATIM from the original eval script (since deleted,
  superseded by src/eval/) — the gate that reproduced the pre-registered
  S1=35 / S2=62 counts from an independent code path.
  The predicate->family map is an EXHAUSTIVE PARTITION of the 20-predicate
  scored vocabulary, not a subset test: any predicate outside the 20
  raises rather than silently falling through to a default stratum.
"""

ARCH_KEYS = ("wall", "floor", "ceiling", "ground")

# The 20-predicate scored vocabulary: the 22-predicate scored set minus front/behind.
PREDICATE_VOCAB = (
    "left", "right", "close by", "attached to", "standing on",
    "higher than", "lower than", "same as", "lying on", "hanging on",
    "supported by", "standing in", "build in", "leaning against",
    "connected to", "belonging to", "lying in", "part of", "cover",
    "hanging in",
)

S3_PREDICATES = frozenset({"close by"})                        # proximity
S4_PREDICATES = frozenset({"left", "right"})                   # lateral
S5_PREDICATES = frozenset({"higher than", "lower than"})       # vertical
SAMEAS_PREDICATES = frozenset({"same as"})
SUPPORT_PREDICATES = (frozenset(PREDICATE_VOCAB)
                      - S3_PREDICATES - S4_PREDICATES
                      - S5_PREDICATES - SAMEAS_PREDICATES)     # 14 predicates

STRATA = ("S1", "S2", "S3", "S4", "S5", "SAME_AS")

STRATUM_MEANING = {
    "S1": "support/attachment, object-object",
    "S2": "support/attachment, >=1 architectural endpoint",
    "S3": "proximity (close by)",
    "S4": "lateral (left/right)",
    "S5": "vertical (higher/lower than)",
    "SAME_AS": "same as",
}

# PRE-REGISTERED counts for S1/S2/S3 and S4/S5/SAME_AS, each verified via an
# independent code path. Fixed before the run.
EXPECTED_STRATA = {"S1": 35, "S2": 62, "S3": 45, "S4": 68, "S5": 5, "SAME_AS": 1}

# PRE-REGISTERED canonical predicate histogram over the 216 items, verified
# via an independent code path.
EXPECTED_PREDICATE_HISTOGRAM = {
    "attached to": 24, "lying on": 21, "hanging on": 4, "standing on": 36,
    "supported by": 9, "standing in": 2, "connected to": 1,
    "close by": 45, "left": 38, "right": 30,
    "higher than": 3, "lower than": 2, "same as": 1,
}


def arch_trigger(label):
    """Return the ARCH_KEY that fired on this label, or None. Returning the
    KEY rather than a bool makes false positives (e.g. a hypothetical
    'wall lamp') visible in the audit output instead of silent."""
    low = (label or "").lower()
    for k in ARCH_KEYS:
        if k in low:
            return k
    return None


def stratum_of(gt_predicate, subject_label, object_label):
    p = str(gt_predicate).strip().lower()
    if p in S3_PREDICATES:
        return "S3"
    if p in S4_PREDICATES:
        return "S4"
    if p in S5_PREDICATES:
        return "S5"
    if p in SAMEAS_PREDICATES:
        return "SAME_AS"
    if p in SUPPORT_PREDICATES:
        return "S2" if (arch_trigger(subject_label)
                        or arch_trigger(object_label)) else "S1"
    raise ValueError(
        "predicate {0!r} is outside the 20-item scored vocabulary; "
        "the partition is exhaustive by design".format(gt_predicate))
