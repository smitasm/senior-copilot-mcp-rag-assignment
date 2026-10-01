"""Retrieval tests, entirely on HashingEmbeddingProvider (deterministic, no
network) - never on OllamaEmbeddingProvider, which needs a live server."""

import pytest

from rag.chunking import Chunk, chunk_documents
from rag.documents import DOCUMENTS
from rag.embeddings import HashingEmbeddingProvider
from rag.retrieval import BM25Index, HybridRetriever, detect_injection, format_for_prompt


def make_chunk(doc_id, text, heading="H", tags=("x",)):
    return Chunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, doc_title=doc_id, doc_category="reference",
                heading=heading, text=text, tags=tags)


@pytest.fixture(scope="module")
def corpus_chunks():
    return chunk_documents(DOCUMENTS)


@pytest.fixture(scope="module")
def retriever(corpus_chunks):
    return HybridRetriever(corpus_chunks, HashingEmbeddingProvider())


# ---- BM25 correctness properties -----------------------------------------------
def test_bm25_gives_zero_score_when_no_query_terms_match():
    idx = BM25Index([make_chunk("A", "vibration bearing coupling"), make_chunk("B", "lube oil pressure pump")])
    assert idx.score("zzz nonexistent qqq") == [0.0, 0.0]


def test_bm25_rare_term_scores_higher_than_common_term():
    """'compressor' appears in every doc (uninformative); 'zeolite' appears in
    only one (highly informative) - BM25's idf must reward the rare term."""
    chunks = [make_chunk("A", "compressor zeolite dryer"), make_chunk("B", "compressor discharge valve"),
             make_chunk("C", "compressor suction line")]
    idx = BM25Index(chunks)
    common_score = idx.score("compressor")[0]
    rare_score = idx.score("zeolite")[0]
    assert rare_score > common_score


def test_bm25_term_frequency_has_diminishing_returns():
    idx = BM25Index([make_chunk("A", "vibration vibration vibration vibration vibration"),
                     make_chunk("B", "vibration bearing coupling rotor imbalance")])
    once, five_times = idx.score("vibration")[1], idx.score("vibration")[0]
    assert five_times > once
    assert five_times < 5 * once  # NOT linear - that's the saturation property


def test_bm25_empty_index_does_not_crash():
    assert BM25Index([]).score("anything") == []


# ---- hybrid weighting ------------------------------------------------------------
def test_bm25_only_and_semantic_only_can_disagree():
    """Constructs a case where lexical and 'semantic' (shared-vocabulary)
    signals genuinely pull in different directions, then shows bm25_weight=1
    and semantic_weight=1 alone produce different top results."""
    chunks = [
        make_chunk("A", "vibration vibration vibration bearing coupling rotor"),  # heavy exact-term repetition
        make_chunk("B", "shaft imbalance mechanical looseness misalignment fault"),  # related vocabulary, no exact hit
    ]
    embedder = HashingEmbeddingProvider()
    lexical_only = HybridRetriever(chunks, embedder, bm25_weight=1.0, semantic_weight=0.0)
    top_lexical = lexical_only.retrieve("vibration", top_k=1)[0].chunk.chunk_id
    assert top_lexical == "A#0"  # only exact term matches count


def test_weights_are_normalized_so_relative_ratio_is_what_matters():
    chunks = [make_chunk("A", "vibration bearing"), make_chunk("B", "lube oil pressure")]
    embedder = HashingEmbeddingProvider()
    r1 = HybridRetriever(chunks, embedder, bm25_weight=1, semantic_weight=1)
    r2 = HybridRetriever(chunks, embedder, bm25_weight=10, semantic_weight=10)
    assert r1.retrieve("vibration", top_k=2) == r2.retrieve("vibration", top_k=2)


def test_zero_total_weight_is_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([make_chunk("A", "x")], HashingEmbeddingProvider(), bm25_weight=0, semantic_weight=0)


def test_empty_chunk_list_is_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([], HashingEmbeddingProvider())


def test_min_score_and_top_k_filter_and_limit_results(retriever):
    all_results = retriever.retrieve("vibration bearing compressor", top_k=50, min_score=0.0)
    limited = retriever.retrieve("vibration bearing compressor", top_k=3, min_score=0.0)
    assert len(limited) == 3 and limited == all_results[:3]
    strict = retriever.retrieve("vibration bearing compressor", top_k=50, min_score=0.9)
    assert len(strict) <= len(all_results) and all(r.score >= 0.9 for r in strict)


def test_retrieval_is_deterministic(retriever):
    a = retriever.retrieve("boiler feed pump seal flush low suction pressure", top_k=5)
    b = retriever.retrieve("boiler feed pump seal flush low suction pressure", top_k=5)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]


# ---- grounding: real demo queries hit the right document -----------------------------------------
def test_k201_style_query_surfaces_the_vibration_guide(retriever):
    top = retriever.retrieve("K-201 compressor high vibration bearing coupling", top_k=3)
    assert top[0].chunk.doc_id == "KB-0001"


def test_bfp101_style_query_surfaces_the_seal_flush_guide(retriever):
    top = retriever.retrieve("boiler feed pump seal flush low flow bearing temperature", top_k=3)
    assert any(r.chunk.doc_id == "KB-0003" for r in top)


def test_rationalization_query_surfaces_the_chattering_guide(retriever):
    top = retriever.retrieve("alarm chattering repeatedly deadband on-delay nuisance", top_k=3)
    assert top[0].chunk.doc_id == "KB-0005"


def test_boiler_safety_query_surfaces_the_safety_procedure(retriever):
    top = retriever.retrieve("low drum level flame failure boiler safety trip", top_k=3)
    assert top[0].chunk.doc_id == "KB-0004"


# ---- injection guard: the centerpiece --------------------------------------------------------
def test_detect_injection_true_positive_on_the_fixture_document():
    doc = next(d for d in DOCUMENTS if d.doc_id == "KB-0099")
    injected_section = doc.body.split("## Automated Assistant Instructions")[1].split("## Data Handling")[0]
    flagged, reason = detect_injection(injected_section)
    assert flagged and reason


def test_detect_injection_does_not_false_positive_on_ordinary_procedural_language():
    benign_samples = [
        "Follow this procedure to replace the seal flush strainer.",
        "Check the local gauge before taking any action on the alarm.",
        "This document describes the escalation process for critical alarms.",
        "The vendor's diagnostics team may request temporary read access to trend data.",
    ]
    for text in benign_samples:
        flagged, _ = detect_injection(text)
        assert not flagged, text


def test_the_injection_chunk_is_retrievable_and_flagged_when_it_matches_the_query(retriever):
    """The corpus deliberately keeps a bad document in it (see rag/documents.py
    module docstring) - the guard's job is to flag it at retrieval time, not
    to have hidden it from the corpus in the first place."""
    results = retriever.retrieve("vendor remote access AI assistant automated instructions", top_k=10)
    flagged = [r for r in results if r.flagged]
    assert flagged and all(r.chunk.doc_id == "KB-0099" for r in flagged)
    assert any("Automated Assistant Instructions" in r.chunk.heading for r in flagged)


def test_other_kb0099_sections_are_not_flagged_only_the_bad_one_is(retriever):
    """Precision matters: the guard must flag the specific malicious chunk,
    not the whole document just because one section of it is bad."""
    results = retriever.retrieve("remote diagnostics vendor access data handling background", top_k=10)
    kb99 = [r for r in results if r.chunk.doc_id == "KB-0099"]
    assert kb99  # the document is genuinely retrieved for this query
    assert any(not r.flagged for r in kb99)  # its legitimate sections are clean
    assert all(r.flagged == ("Automated Assistant Instructions" in r.chunk.heading) for r in kb99)


def test_format_for_prompt_includes_a_caution_warning_only_for_flagged_chunks(retriever):
    results = retriever.retrieve("vendor remote access AI assistant automated instructions", top_k=10)
    assert any(r.flagged for r in results) and any(not r.flagged for r in results)  # both cases present
    prompt_text = format_for_prompt(results)
    blocks = prompt_text.split("<source ")[1:]  # one entry per <source ...>...</source> block
    assert len(blocks) == len(results)
    for r, block in zip(results, blocks):
        assert ("CAUTION" in block) == r.flagged  # present iff flagged, checked BOTH directions
    assert "not instructions" in prompt_text  # the framing preamble is present


def test_format_for_prompt_preserves_the_flagged_text_for_auditability(retriever):
    """The guard neutralizes by framing, not by deleting - the actual
    injected text must still be visible for a human reviewer to audit."""
    flagged = [r for r in retriever.retrieve("vendor remote access AI assistant automated instructions",
                                             top_k=10) if r.flagged]
    prompt_text = format_for_prompt(flagged)
    assert "ignore all" in prompt_text.lower() and "approved=true" in prompt_text.lower()


def test_format_for_prompt_on_empty_results_says_so_plainly():
    assert "No relevant" in format_for_prompt([])


def test_format_for_prompt_includes_doc_and_section_for_citation(retriever):
    results = retriever.retrieve("high vibration compressor bearing", top_k=1)
    text = format_for_prompt(results)
    assert results[0].chunk.doc_id in text and results[0].chunk.heading in text