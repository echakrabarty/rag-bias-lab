"""Controlled pair construction for the bias audits.

Each pair has a known-correct answer and a plausible distractor, so audit
metrics have ground truth. Distractors are *other* gold answer spans from
the same paragraph: topically related, same style, but wrong for the
question at hand.

Pair dict: {"question", "correct", "distractor", "reference"}.
"""

import random

# Verbose-but-empty padding used to build the "long wrong answer" condition.
PADDING = (
    "To elaborate in comprehensive detail, considering the many different "
    "facets and various aspects of this matter from multiple perspectives, "
    "a thorough and exhaustive analysis would clearly indicate the following."
)


def build_pairs(questions, n=200, seed=0):
    rng = random.Random(seed)
    pool = [q for q in questions if q.get("sibling_answers")]
    rng.shuffle(pool)
    pairs = []
    for q in pool:
        gold = q["answers"][0]
        distractors = [a for a in q["sibling_answers"] if a.lower() != gold.lower()]
        if not distractors:
            continue
        pairs.append(
            {
                "question": q["question"],
                "correct": gold,
                "distractor": distractors[0],
                "long_distractor": f"{distractors[0]}. {PADDING}",
                "reference": gold,
            }
        )
        if len(pairs) >= n:
            break
    return pairs
