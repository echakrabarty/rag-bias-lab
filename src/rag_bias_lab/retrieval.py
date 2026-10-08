"""Retrievers: BM25, TF-IDF cosine, and a hybrid fused with RRF.

Runs offline on numpy + scikit-learn. Each retriever implements:

    retrieve(query: str, top_k: int = 3) -> list[{"id", "text", "score"}]
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 with k1=1.5, b=0.75 (standard defaults)."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in corpus]
        self.doc_len = np.array([sum(d.values()) for d in self.tf], dtype=float)
        self.avgdl = self.doc_len.mean() if len(self.doc_len) else 1.0
        df = Counter()
        for d in self.tf:
            for t in d:
                df[t] += 1
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def retrieve(self, query, top_k=3):
        scores = np.zeros(len(self.corpus))
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf_d in enumerate(self.tf):
                tf = tf_d.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self.corpus[i]["id"], "text": self.corpus[i]["text"], "score": float(scores[i])}
            for i in ranked
        ]


class TfidfRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in corpus])

    def retrieve(self, query, top_k=3):
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"id": self.corpus[i]["id"], "text": self.corpus[i]["text"], "score": float(scores[i])}
            for i in ranked
        ]


class HybridRetriever:
    """Fuses BM25 and TF-IDF rankings with reciprocal rank fusion."""

    def __init__(self, corpus, rrf_k=60):
        self.bm25 = BM25Retriever(corpus)
        self.tfidf = TfidfRetriever(corpus)
        self.rrf_k = rrf_k

    def retrieve(self, query, top_k=3):
        fused = {}
        for retriever in (self.bm25, self.tfidf):
            for rank, hit in enumerate(retriever.retrieve(query, top_k=25)):
                entry = fused.setdefault(hit["id"], {"hit": hit, "score": 0.0})
                entry["score"] += 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.values(), key=lambda e: -e["score"])[:top_k]
        return [
            {"id": e["hit"]["id"], "text": e["hit"]["text"], "score": e["score"]} for e in ranked
        ]
