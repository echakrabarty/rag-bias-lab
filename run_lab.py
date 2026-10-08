"""End-to-end lab run: retrieve -> generate -> judge -> bias-audit -> report.

Usage:
    PYTHONPATH=src python run_lab.py [--n-pairs 200] [--bias-p 0.7] [--seed 0]

    # Real LLM judge instead of the offline heuristic:
    PYTHONPATH=src python run_lab.py --judge-backend openai --n-pairs 50
    PYTHONPATH=src python run_lab.py --judge-backend anthropic --judge-model claude-haiku-4-5 --n-pairs 50
    PYTHONPATH=src python run_lab.py --judge-backend gemini --n-pairs 50

Runs fully offline with the heuristic judge by default. The bias-injection
controls (position / length, strength --bias-p) validate that the audit
suite recovers known biases.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_bias_lab.bias_audit import run_full_audit
from rag_bias_lab.corpus import load_slice
from rag_bias_lab.generation import (
    AnthropicChatGenerator,
    ExtractiveGenerator,
    OpenAIChatGenerator,
)
from rag_bias_lab.judges import (
    HeuristicJudge,
    LLMPairwiseJudge,
    LengthBiasedJudge,
    PositionBiasedJudge,
    token_f1,
)
from rag_bias_lab.pairs import build_pairs
from rag_bias_lab.report import write_report
from rag_bias_lab.retrieval import HybridRetriever

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def retrieval_eval(retriever, generator, questions, top_k=3):
    hits, f1s = 0, []
    for q in questions:
        ranked = retriever.retrieve(q["question"], top_k=top_k)
        if any(h["id"] == q["context_id"] for h in ranked):
            hits += 1
        answer = generator.generate(q["question"], ranked)
        gold = q["answers"][0]
        f1s.append(max(token_f1(answer, g) for g in q["answers"]))
    n = len(questions)
    return {"n_eval": n, "recall_at_3": hits / n if n else 0.0,
            "answer_f1": sum(f1s) / n if n else 0.0}


def build_judges(backend, model, bias_p, seed):
    """Heuristic backend runs the reference judge plus injected-bias controls.

    Real-LLM backends run just the one judge (each pair costs API calls).
    """
    base = HeuristicJudge()
    if backend == "heuristic":
        return {
            "heuristic (reference)": base,
            f"position-biased (p={bias_p})": PositionBiasedJudge(base, p=bias_p, seed=seed),
            f"length-biased (p={bias_p})": LengthBiasedJudge(base, p=bias_p, seed=seed),
        }
    if backend == "openai":
        gen = OpenAIChatGenerator(model=model or "gpt-4o-mini")
    elif backend == "anthropic":
        gen = AnthropicChatGenerator(model=model or "claude-haiku-4-5")
    elif backend == "gemini":
        gen = OpenAIChatGenerator(
            model=model or "gemini-2.5-flash", base_url=GEMINI_OPENAI_BASE
        )
    else:
        raise ValueError(f"unknown --judge-backend: {backend}")
    name = f"{backend}:{gen.model} (LLM judge)"
    return {name: LLMPairwiseJudge(gen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="data/squad_slice.json")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--bias-p", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="report.md")
    ap.add_argument(
        "--judge-backend",
        choices=["heuristic", "openai", "anthropic", "gemini"],
        default="heuristic",
        help="heuristic runs offline with injected-bias controls; the rest call a real LLM judge",
    )
    ap.add_argument("--judge-model", default=None, help="override the backend's default model")
    args = ap.parse_args()

    slice_path = Path(args.slice)
    if not slice_path.exists():
        sys.exit(f"missing {slice_path}; run: python scripts/fetch_squad.py")
    contexts, questions = load_slice(slice_path)
    print(f"loaded {len(contexts)} contexts, {len(questions)} questions")

    retriever = HybridRetriever(contexts)
    generator = ExtractiveGenerator()
    print("running retrieval eval ...")
    ret_metrics = retrieval_eval(retriever, generator, questions)
    print(f"  recall@3={ret_metrics['recall_at_3']:.3f} answer_f1={ret_metrics['answer_f1']:.3f}")

    pairs = build_pairs(questions, n=args.n_pairs, seed=args.seed)
    print(f"built {len(pairs)} controlled pairs")
    judges = build_judges(args.judge_backend, args.judge_model, args.bias_p, args.seed)
    print("running bias audits ...")
    audit_results = run_full_audit(judges, pairs)
    for name, res in audit_results.items():
        p, l, a = res["position"], res["length"], res["accuracy"]
        print(f"  [{name}] consistency={p['consistency']:.3f} "
              f"first_pos={p['first_position_rate']:.3f} "
              f"long_wrong={l['long_wrong_win_rate']:.3f} acc={a['accuracy']:.3f}")

    config = {
        "slice": str(slice_path),
        "retriever": "hybrid BM25 + TF-IDF (RRF k=60)",
        "generator": "extractive (offline)",
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model or "backend default",
        "n_pairs": len(pairs),
        "injected_bias_p": args.bias_p,
        "seed": args.seed,
    }
    write_report(args.report, config, ret_metrics, audit_results)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
