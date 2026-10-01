"""Splits documents into retrievable chunks along their own "## " section
headings, so citations are meaningful ("KB-0001 - Common Root Causes") rather
than an arbitrary character window. A length-safety split further breaks up
any single section that runs long, with a one-paragraph overlap so a sentence
near the cut point still has its immediate context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.documents import Document

DEFAULT_MAX_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}#{n}"
    doc_id: str
    doc_title: str
    doc_category: str
    heading: str  # nearest "## " heading, or the doc title if the chunk precedes any heading
    text: str
    tags: tuple[str, ...]  # inherited from the parent document


_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _sections(body: str) -> list[tuple[str, str]]:
    """Split a document body into (heading, text) pairs on '## ' lines."""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())]
    out = []
    if matches[0].start() > 0:
        lead = body[: matches[0].start()].strip()
        if lead:
            out.append(("", lead))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group(1).strip(), body[m.end():end].strip()))
    return out


def _split_long(text: str, max_chars: int) -> list[str]:
    """Greedily pack blank-line-separated paragraphs into pieces <= max_chars,
    repeating the last paragraph of each piece as the first of the next one
    (simple overlap) so a reader never loses the sentence right at a cut."""
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[list[str]] = [[]]
    for p in paras:
        current = pieces[-1]
        current_len = sum(len(x) for x in current) + 2 * max(0, len(current) - 1)
        if current and current_len + 2 + len(p) > max_chars:
            pieces.append([current[-1], p])  # overlap: carry the last paragraph forward
        else:
            current.append(p)
    return ["\n\n".join(piece) for piece in pieces if piece]


def chunk_document(doc: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for heading, text in _sections(doc.body):
        if not text:
            continue
        for piece in _split_long(text, max_chars):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{n}", doc_id=doc.doc_id, doc_title=doc.title,
                doc_category=doc.category, heading=heading or doc.title, text=piece, tags=doc.tags,
            ))
            n += 1
    return chunks


def chunk_documents(docs, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, max_chars)]