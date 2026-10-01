"""Chunking tests. Uses the real corpus for integration-style checks, and
small synthetic Document fixtures for the edge cases (no headings, an
oversized section) where I need precise control over input shape."""

from rag.chunking import DEFAULT_MAX_CHARS, chunk_document, chunk_documents
from rag.documents import DOCUMENTS, Document


def test_chunk_ids_are_unique_across_the_whole_corpus():
    chunks = chunk_documents(DOCUMENTS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_every_chunk_carries_its_parent_documents_tags_and_metadata():
    doc = next(d for d in DOCUMENTS if d.doc_id == "KB-0001")
    chunks = chunk_document(doc)
    assert len(chunks) > 1  # this doc has multiple "## " sections
    for c in chunks:
        assert (c.doc_id, c.doc_title, c.doc_category, c.tags) == (doc.doc_id, doc.title, doc.category, doc.tags)


def test_chunk_headings_match_the_documents_own_section_titles():
    doc = next(d for d in DOCUMENTS if d.doc_id == "KB-0005")
    headings = {c.heading for c in chunk_document(doc)}
    assert {"Chattering Alarms", "Stale Alarms", "Recurring Alarms"} <= headings


def test_document_with_no_headings_becomes_one_chunk_titled_with_the_doc_title():
    doc = Document(doc_id="X-1", title="No Headings Here", category="reference", tags=("x",),
                   body="Just a single paragraph with no section markers at all.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1 and chunks[0].heading == doc.title


def test_leading_text_before_the_first_heading_becomes_its_own_chunk():
    doc = Document(doc_id="X-2", title="Intro Then Sections", category="reference", tags=("x",),
                   body="An introductory paragraph.\n\n## First Section\nBody text here.")
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].heading == doc.title and "introductory" in chunks[0].text
    assert chunks[1].heading == "First Section"


def test_oversized_section_is_split_with_paragraph_overlap():
    para = "Sentence about the topic. " * 10  # ~270 chars per paragraph
    body = "## Long Section\n\n" + "\n\n".join(f"{para}(paragraph {i})" for i in range(6))
    doc = Document(doc_id="X-3", title="Long Doc", category="reference", tags=("x",), body=body)
    chunks = chunk_document(doc, max_chars=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 300 for c in chunks)  # allows one overlapped paragraph over the limit
    # overlap: each piece after the first repeats the previous piece's last paragraph
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_last_para = prev.text.split("\n\n")[-1]
        assert prev_last_para in nxt.text


def test_short_section_is_not_split():
    doc = Document(doc_id="X-4", title="Short", category="reference", tags=("x",),
                   body="## Section\nA short paragraph well under the limit.")
    assert len(chunk_document(doc, max_chars=DEFAULT_MAX_CHARS)) == 1


def test_empty_section_between_two_headings_produces_no_empty_chunk():
    doc = Document(doc_id="X-5", title="Empty Middle", category="reference", tags=("x",),
                   body="## A\nSome text.\n\n## B\n\n## C\nMore text.")
    chunks = chunk_document(doc)
    assert all(c.text for c in chunks)
    assert {c.heading for c in chunks} == {"A", "C"}  # B had no body text, so it produced no chunk


def test_chunk_documents_flattens_all_documents_in_order():
    subset = [d for d in DOCUMENTS if d.doc_id in ("KB-0001", "KB-0002")]
    combined = chunk_documents(subset)
    assert [c.doc_id for c in combined if c.doc_id == "KB-0001"] == \
           [c.doc_id for c in chunk_document(subset[0])]