"""Pairwise LLM judges and controlled bias-injection wrappers.

Judges implement:

    judge(question: str, a: str, b: str, reference: str)
        -> {"winner": "A" | "B", ...}

The audit suite validates itself with bias injection: wrap the unbiased
HeuristicJudge in a wrapper with a *known* bias strength p, then check the
audit recovers ~p. That makes the whole suite verifiable offline.
"""

import hashlib
import random
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s):
    return _TOKEN.findall(s.lower())


def token_f1(pred, ref):
    pt, rt = _tokens(pred), _tokens(ref)
    if not pt or not rt:
        return 0.0
    # multiset overlap
    pc, rc = Counter(pt), Counter(rt)
    common = sum((pc & rc).values())
    if not common:
        return 0.0
    precision = common / len(pt)
    recall = common / len(rt)
    return 2 * precision * recall / (precision + recall)


def _order_independent_tiebreak(a, b):
    # md5, not hash(): str hashing is salted per-process in CPython.
    ha = hashlib.md5(a.encode()).hexdigest()
    hb = hashlib.md5(b.encode()).hexdigest()
    return "A" if ha <= hb else "B"


class HeuristicJudge:
    """Unbiased-by-construction reference judge.

    Scores each answer by token F1 against the reference. F1 normalizes for
    length, and the tie-break is content-hash based, so the verdict cannot
    depend on presentation order or verbosity.
    """

    def judge(self, question, a, b, reference):
        sa, sb = token_f1(a, reference), token_f1(b, reference)
        if abs(sa - sb) < 1e-9:
            winner = _order_independent_tiebreak(a, b)
        else:
            winner = "A" if sa > sb else "B"
        return {"winner": winner, "score_a": sa, "score_b": sb}


class PositionBiasedJudge:
    """Wraps a base judge; with probability p picks 'A' (first) regardless."""

    def __init__(self, base, p=0.7, seed=0):
        self.base, self.p = base, p
        self.rng = random.Random(seed)

    def judge(self, question, a, b, reference):
        if self.rng.random() < self.p:
            return {"winner": "A", "score_a": 1.0, "score_b": 0.0, "injected_bias": "position"}
        return self.base.judge(question, a, b, reference)


class LengthBiasedJudge:
    """Wraps a base judge; with probability p picks the longer answer."""

    def __init__(self, base, p=0.7, seed=0):
        self.base, self.p = base, p
        self.rng = random.Random(seed)

    def judge(self, question, a, b, reference):
        if self.rng.random() < self.p:
            winner = "A" if len(a) >= len(b) else "B"
            return {"winner": winner, "score_a": 1.0, "score_b": 0.0, "injected_bias": "length"}
        return self.base.judge(question, a, b, reference)


class LLMPairwiseJudge:
    """Real LLM pairwise judge. Takes any generator with .generate(prompt, contexts).

    Example:
        judge = LLMPairwiseJudge(OpenAIChatGenerator(model="gpt-4o-mini"))
        judge = LLMPairwiseJudge(AnthropicChatGenerator(model="claude-haiku-4-5"))
    """

    SYSTEM = (
        "You compare two answers to a question against a reference answer. "
        "Reply with exactly one line: WINNER: A or WINNER: B, then a one-sentence reason."
    )

    def __init__(self, generator):
        self.generator = generator

    def judge(self, question, a, b, reference):
        prompt = (
            f"{self.SYSTEM}\n\nQuestion: {question}\nReference: {reference}\n"
            f"Answer A: {a}\nAnswer B: {b}\nVerdict:"
        )
        raw = self.generator.generate(prompt, [])
        winner = "A" if "WINNER: A" in raw.upper() else "B"
        return {"winner": winner, "raw": raw}
