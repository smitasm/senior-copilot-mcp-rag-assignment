"""Embedding providers for the RAG retriever.

Two implementations of the same tiny interface (`embed(texts) -> vectors`):

  * HashingEmbeddingProvider - deterministic, dependency-free, no network.
    Used by every test and by CI. It is NOT a semantic embedding (it only
    captures shared vocabulary via a bag-of-words hashing trick), but that is
    exactly what makes it useful here: it lets the retrieval PIPELINE
    (chunking -> BM25 -> semantic scoring -> hybrid merge -> injection guard)
    be tested deterministically with zero external dependencies.

  * OllamaEmbeddingProvider - real semantic embeddings via a running Ollama
    server (nomic-embed-text by default). Used only at runtime, wired up in
    the orchestrator (Phase 5); no test imports or instantiates this against
    a real server.

Both return L2-normalized vectors so retrieval.py's cosine similarity is a
plain dot product.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Deterministic bag-of-words vector via the hashing trick (fixed dim).
    Same text -> same vector, always; no I/O of any kind."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in tokenize(text):
            idx = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        return _l2_normalize(v)


class OllamaEmbeddingProvider:
    """Real embeddings via Ollama's /api/embed. Synchronous (a plain blocking
    httpx.Client call) to keep the EmbeddingProvider interface simple - the
    retriever itself does no I/O of its own either way."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text",
                timeout_s: float = 30.0) -> None:
        self.base_url, self.model, self.timeout_s = base_url, model, timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout_s) as client:
            resp = client.post("/api/embed", json={"model": self.model, "input": texts})
            resp.raise_for_status()
            vectors = resp.json()["embeddings"]
        return [_l2_normalize(v) for v in vectors]