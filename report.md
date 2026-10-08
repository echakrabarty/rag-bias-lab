# rag-bias-lab run report

Date: 2026-09-30

## Config

- slice: data/squad_slice.json
- retriever: hybrid BM25 + TF-IDF (RRF k=60)
- generator: extractive (offline)
- n_pairs: 200
- injected_bias_p: 0.7
- seed: 0

## Retrieval + generation (SQuAD slice)

- recall@3: 0.793
- answer token-F1 (extractive): 0.069
- eval questions: 600

## Judge bias audits

| judge | position consistency | first-position rate | long-wrong win rate | accuracy |
| --- | --- | --- | --- | --- |
| heuristic (reference) | 1.000 (n=200) | 0.500 | 0.000 (n=200) | 1.000 (n=200) |
| position-biased (p=0.7) | 0.275 (n=200) | 0.863 | 0.330 (n=200) | 0.600 (n=200) |
| length-biased (p=0.7) | 0.720 (n=200) | 0.550 | 0.700 (n=200) | 0.605 (n=200) |

## Verdicts

### heuristic (reference)
- PASS: no position bias detected
- PASS: no length bias detected
- PASS: judge is competent
### position-biased (p=0.7)
- FLAG: position bias detected
- FLAG: length bias detected
- FLAG: judge accuracy is low
### length-biased (p=0.7)
- FLAG: position bias detected
- FLAG: length bias detected
- FLAG: judge accuracy is low

## How to read this
- *Position consistency* = fraction of pairs judged identically in both presentation orders. 1.0 means order never mattered.
- *First-position rate* = how often the first-presented answer wins. 0.5 means no positional preference.
- *Long-wrong win rate* = how often a padded, verbose *wrong* answer beats the short correct one. Near 0.0 is unbiased.
- The `position-biased (p=0.7)` and `length-biased (p=0.7)` judges are controls: they inject a *known* bias so you can verify the audit recovers it. If the audit can't flag a 70% injected bias, the audit itself is broken.
