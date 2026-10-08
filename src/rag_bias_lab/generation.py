"""Generators: offline extractive default + real LLM via env key.

Generators implement:

    generate(prompt: str, contexts: list[{"id", "text", "score"}]) -> str

Set RAG_BIAS_LAB_OPENAI_KEY to use OpenAIChatGenerator (any
OpenAI-compatible endpoint works via RAG_BIAS_LAB_OPENAI_BASE).
No extra dependency: the client is plain urllib.
"""

import json
import os
import re
import urllib.request


class ExtractiveGenerator:
    """Offline default: the top chunk's most question-relevant sentence, cited."""

    _SENT = re.compile(r"(?<=[.!?])\s+")
    _TOK = re.compile(r"[a-z0-9]+")

    def generate(self, prompt, contexts):
        top = contexts[0]
        q_toks = set(self._TOK.findall(prompt.lower()))
        sents = [s.strip() for s in self._SENT.split(top["text"]) if s.strip()]
        if sents:
            best = max(sents, key=lambda s: len(q_toks & set(self._TOK.findall(s.lower()))))
        else:
            best = top["text"][:600]
        return f"Based on [{top['id']}]: {best}"


class OpenAIChatGenerator:
    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None, timeout=60):
        self.model = model
        self.api_key = api_key or os.environ.get("RAG_BIAS_LAB_OPENAI_KEY")
        if not self.api_key:
            raise RuntimeError("Set RAG_BIAS_LAB_OPENAI_KEY to use OpenAIChatGenerator.")
        self.base_url = base_url or os.environ.get(
            "RAG_BIAS_LAB_OPENAI_BASE", "https://api.openai.com/v1"
        )
        self.timeout = timeout

    def generate(self, prompt, contexts):
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]


class AnthropicChatGenerator:
    """Native Anthropic Messages API client (no OpenAI-compatible endpoint exists)."""

    def __init__(self, model="claude-haiku-4-5", api_key=None, timeout=60, max_tokens=512):
        self.model = model
        self.api_key = api_key or os.environ.get("RAG_BIAS_LAB_ANTHROPIC_KEY")
        if not self.api_key:
            raise RuntimeError("Set RAG_BIAS_LAB_ANTHROPIC_KEY to use AnthropicChatGenerator.")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, prompt, contexts):
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
