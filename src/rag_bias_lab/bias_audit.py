"""Bias audit suite for pairwise LLM judges.

Audits (each returns a dict of metrics):
  - position_bias_audit: present every pair in both orders. An unbiased judge
    makes the same call both times (consistency = 1.0) and shows no
    preference for the first-presented answer (first_position_rate = 0.5).
  - length_bias_audit: the wrong answer is padded to be much longer; order is
    randomized per pair so position cannot confound the result. An unbiased
    judge picks the long wrong answer ~never.
  - accuracy: order-randomized fraction of pairs where the judge picks the
    known-correct answer (sanity check that the judge is competent).

Thresholds for the verdicts below are documented in the report.
"""

import random


def position_bias_audit(judge, pairs):
    consistent = 0
    first_wins = 0
    total = 0
    for p in pairs:
        r1 = judge.judge(p["question"], p["correct"], p["distractor"], p["reference"])
        r2 = judge.judge(p["question"], p["distractor"], p["correct"], p["reference"])
        # Map both verdicts back to content: did it pick "correct"?
        c1 = "correct" if r1["winner"] == "A" else "distractor"
        c2 = "correct" if r2["winner"] == "B" else "distractor"
        total += 1
        if c1 == c2:
            consistent += 1
        # "first presented" won in each ordering?
        first_wins += (r1["winner"] == "A") + (r2["winner"] == "A")
    return {
        "n": total,
        "consistency": consistent / total if total else 0.0,
        "first_position_rate": first_wins / (2 * total) if total else 0.0,
    }


def length_bias_audit(judge, pairs, seed=1):
    rng = random.Random(seed)
    long_wins = 0
    for p in pairs:
        if rng.random() < 0.5:
            a, b, long_label = p["correct"], p["long_distractor"], "B"
        else:
            a, b, long_label = p["long_distractor"], p["correct"], "A"
        if judge.judge(p["question"], a, b, p["reference"])["winner"] == long_label:
            long_wins += 1
    return {
        "n": len(pairs),
        "long_wrong_win_rate": long_wins / len(pairs) if pairs else 0.0,
    }


def accuracy_audit(judge, pairs, seed=2):
    """Order-randomized accuracy on the controlled pairs (judge sanity check)."""
    rng = random.Random(seed)
    correct = 0
    for p in pairs:
        if rng.random() < 0.5:
            a, b, correct_label = p["correct"], p["distractor"], "A"
        else:
            a, b, correct_label = p["distractor"], p["correct"], "B"
        if judge.judge(p["question"], a, b, p["reference"])["winner"] == correct_label:
            correct += 1
    return {"n": len(pairs), "accuracy": correct / len(pairs) if pairs else 0.0}


def run_full_audit(judges, pairs):
    """judges: {name: judge}. Returns {name: {audit: metrics}}."""
    results = {}
    for name, judge in judges.items():
        results[name] = {
            "position": position_bias_audit(judge, pairs),
            "length": length_bias_audit(judge, pairs),
            "accuracy": accuracy_audit(judge, pairs),
        }
    return results
