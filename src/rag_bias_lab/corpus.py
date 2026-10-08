"""Corpus and eval-set loading.

The corpus is a slice of SQuAD v1.1 dev: each context paragraph is a
retrievable document, each question has gold answer spans plus sibling
answer spans (other answers from the same paragraph) used to build
plausible distractor answers for the bias audits.

Slice file format (JSON):
    {
      "contexts": [{"id": "c0", "text": "..."}],
      "questions": [{
          "id": "q0",
          "question": "...",
          "answers": ["gold span", ...],
          "sibling_answers": ["other span from same paragraph", ...],
          "context_id": "c0",
      }],
    }
"""

import json
from pathlib import Path


def load_slice(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["contexts"], obj["questions"]
