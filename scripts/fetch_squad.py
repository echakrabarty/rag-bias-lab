"""Build a SQuAD v1.1 dev slice for rag-bias-lab.

Downloads dev-v1.1.json, then deterministically takes the first
--n-contexts paragraphs (in file order) and up to --n-eval questions,
attaching sibling answer spans per paragraph for distractor construction.

Writes data/squad_slice.json.
"""

import argparse
import json
import urllib.request
from pathlib import Path

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contexts", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--out", default="data/squad_slice.json")
    args = ap.parse_args()

    print(f"downloading {SQUAD_DEV_URL} ...")
    with urllib.request.urlopen(SQUAD_DEV_URL, timeout=120) as resp:
        squad = json.loads(resp.read())

    contexts, questions = [], []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            if len(contexts) >= args.n_contexts:
                break
            cid = f"c{len(contexts)}"
            contexts.append({"id": cid, "text": para["context"]})
            spans = [a["text"] for qa in para["qas"] for a in qa["answers"]]
            for qa in para["qas"]:
                if len(questions) >= args.n_eval:
                    break
                questions.append(
                    {
                        "id": f"q{len(questions)}",
                        "question": qa["question"],
                        "answers": [a["text"] for a in qa["answers"]],
                        "sibling_answers": spans,
                        "context_id": cid,
                    }
                )
        if len(contexts) >= args.n_contexts:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"contexts": contexts, "questions": questions}), encoding="utf-8")
    print(f"wrote {out}: {len(contexts)} contexts, {len(questions)} questions")


if __name__ == "__main__":
    main()
