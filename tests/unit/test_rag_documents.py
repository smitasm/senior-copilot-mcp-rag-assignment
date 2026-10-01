"""The document corpus must be structurally sound and must cover the alarm
storylines the rest of the project already relies on - otherwise a demo
query has nothing real to retrieve."""

from rag.documents import DOCUMENTS


def test_doc_ids_are_unique():
    ids = [d.doc_id for d in DOCUMENTS]
    assert len(ids) == len(set(ids))


def test_every_document_has_required_fields():
    for d in DOCUMENTS:
        assert d.doc_id and d.title and d.category and d.tags and d.body.strip()
        assert d.category in {"troubleshooting", "safety", "procedure", "reference"}


def test_every_document_has_at_least_one_heading():
    for d in DOCUMENTS:
        assert "## " in d.body, d.doc_id


def test_corpus_covers_the_demo_storyline_alarm_names():
    all_tags = {t for d in DOCUMENTS for t in d.tags}
    required = {"High Vibration", "Low Lube Oil Pressure", "Seal Flush Low Flow", "Low Drum Level",
               "Low Discharge Flow", "High Winding Temperature", "Overload Trip"}
    assert required <= all_tags


def test_corpus_covers_priority_and_correlation_reference_material():
    all_tags = {t for d in DOCUMENTS for t in d.tags}
    assert {"priority", "correlation", "flood"} <= all_tags


def test_injection_fixture_document_is_present_and_shaped_as_expected():
    """Sanity-checks the fixture itself, not the guard: KB-0099 exists and its
    'Automated Assistant Instructions' section is the one that actually
    contains injection-style phrasing - if this ever stops being true, the
    retrieval tests that rely on it need to be revisited."""
    doc = next(d for d in DOCUMENTS if d.doc_id == "KB-0099")
    assert "Automated Assistant Instructions" in doc.body
    assert "ignore all" in doc.body.lower() and "approved=true" in doc.body.lower()
    # the OTHER sections of this document must read as ordinary, legitimate content
    background = doc.body.split("## Background")[1].split("## Automated")[0]
    assert "ignore" not in background.lower() and "approved" not in background.lower()