# rag-bias-lab

A small, fully-offline RAG system with a **bias audit suite for LLM judges** — the part most RAG demos skip.

The pipeline (SQuAD corpus → hybrid BM25 + TF-IDF retrieval with RRF fusion → generation) is the vehicle. The headline is the audit suite: controlled experiments that measure **position bias**, **length bias**, and judge competence in any pairwise LLM judge, validated by *bias injection* — judges with a known injected bias strength, so you can verify the audit actually recovers it.

## Results (verified run, 2026-09-30)

Retrieval + generation on a 400-paragraph / 600-question SQuAD v1.1 slice:

| metric | value |
| --- | --- |
| recall@3 (hybrid BM25 + TF-IDF, RRF) | 0.793 |
| answer token-F1 (offline extractive baseline) | 0.069 |

Bias audits on 200 controlled pairs (correct answer vs. plausible distractor drawn from the same paragraph):

| judge | position consistency | first-position rate | long-wrong win rate | accuracy |
| --- | --- | --- | --- | --- |
| heuristic (reference) | 1.000 | 0.500 | 0.000 | 1.000 |
| position-biased (p=0.7, injected control) | 0.275 | 0.863 | 0.330 | 0.600 |
| length-biased (p=0.7, injected control) | 0.720 | 0.550 | 0.700 | 0.605 |

Reading it: the unbiased judge is perfectly consistent and never fooled by verbosity. The injected controls are caught red-handed — the audit recovers the 70% position bias as an 0.863 first-position preference and the 70% length bias as an 0.700 long-wrong win rate. If the audit couldn't flag a 70% injected bias, the audit itself would be broken; that's the point of the controls.

## The bias audits

- **Position bias**: every pair is judged in both presentation orders. An unbiased judge decides identically both times (consistency = 1.0) with no preference for the first-presented answer (rate = 0.5).
- **Length bias**: the wrong answer is padded with verbose filler and the presentation order is randomized per pair, so position can't confound the result. An unbiased judge almost never picks the long wrong answer.
- **Accuracy**: order-randomized sanity check that the judge is competent enough for its biases to matter.

Pairs are built with ground truth: distractors are *other* gold answer spans from the same paragraph — topically related and stylistically matched, but wrong for the question.

## Run it

```bash
pip install -r requirements.txt        # numpy, pandas, scikit-learn only
python scripts/fetch_squad.py         # downloads SQuAD v1.1 dev, builds the slice
python run_lab.py                     # full run; writes report.md
```

Everything runs offline. The heuristic judge and extractive generator are deterministic baselines so the suite is verifiable without API keys.

## Plug in real models

```bash
export RAG_BIAS_LAB_OPENAI_KEY=...       # OpenAI (gpt-4o-mini default)
export RAG_BIAS_LAB_ANTHROPIC_KEY=...    # Anthropic (claude-haiku-4-5 default)
export RAG_BIAS_LAB_OPENAI_KEY=...       # Gemini free tier: same var, see below
```

```bash
python run_lab.py --judge-backend openai --n-pairs 50
python run_lab.py --judge-backend anthropic --n-pairs 50
python run_lab.py --judge-backend gemini --n-pairs 50        # free tier, see note
python run_lab.py --judge-backend anthropic --judge-model claude-sonnet-4-6 --n-pairs 50
```

- `--judge-backend openai` uses `OpenAIChatGenerator` (any OpenAI-compatible endpoint works via `RAG_BIAS_LAB_OPENAI_BASE`).
- `--judge-backend anthropic` uses the native `AnthropicChatGenerator` (Messages API, no extra dependency). Default model `claude-haiku-4-5` (~$1/M input) keeps a full run under a dollar.
- `--judge-backend gemini` points the OpenAI-compatible client at Google's endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`); a free AI Studio key works, so the demo costs $0.

Start with `--n-pairs 50` (200 judge calls) before scaling to 200. The interesting demo is running two backends and comparing their bias fingerprints side by side.

## Layout

```
scripts/fetch_squad.py      corpus slice builder (deterministic)
src/rag_bias_lab/
  corpus.py                 slice loading
  retrieval.py              BM25 + TF-IDF + RRF hybrid
  generation.py             extractive baseline + OpenAI / Anthropic LLM clients (urllib, no extra deps)
  judges.py                 heuristic / bias-injected / real-LLM pairwise judges
  pairs.py                  controlled pair construction
  bias_audit.py             position, length, and accuracy audits
  report.py                 markdown report writer
run_lab.py                  end-to-end entry point
```

## Why this exists

LLM-as-judge is load-bearing in modern RAG evaluation, and judges are known to favor the first-presented answer and the longer answer. Most eval harnesses *use* a judge without *auditing* it. This repo makes the audit itself the artifact — with injected-bias controls, so the audit's own validity is testable, not assumed.
