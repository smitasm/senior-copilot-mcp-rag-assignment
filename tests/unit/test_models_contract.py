"""Contract test: every request body in the provided Postman collections must
be accepted by our Pydantic request models. If Postman and models drift, this
fails first."""

import json
import re
from pathlib import Path

import pytest

from alarm_api import models as m

POSTMAN_DIR = Path(__file__).resolve().parents[2] / "postman"

# (method, path-regex) -> request model
BODY_MODELS = {
    "/alarms/summary": m.SummaryRequest,
    "/alarms/trends": m.TrendRequest,
    "/alarms/correlation": m.CorrelationRequest,
    "/alarms/flood-analysis": m.FloodRequest,
    "/alarms/rationalization-candidates": m.RationalizationRequest,
    "/alarms/priority-score": m.PriorityScoreRequest,
    "/recommendations/operator-actions": m.RecommendationRequest,
    "/calculation-code/generate": m.GenerateCalculationRequest,
    "/calculation-code/execute": m.ExecuteCalculationRequest,
}

VARS = {
    "asset_id": "AST-0001", "asset_id_2": "AST-0002", "asset_id_3": "AST-0003",
    "alarm_id": "ALM-0001", "calculation_id": "CALC-0001",
    "start_time": "2026-05-01T00:00:00Z", "end_time": "2026-07-01T00:00:00Z",
    "window_start": "2026-05-01T00:00:00Z", "window_end": "2026-07-01T00:00:00Z",
}


def _walk(items):
    for it in items:
        if "item" in it:
            yield from _walk(it["item"])
        else:
            yield it


def _cases():
    for f in sorted(POSTMAN_DIR.rglob("*.json")):
        coll = json.loads(f.read_text())
        for it in _walk(coll["item"]):
            req = it["request"]
            raw = (req.get("body") or {}).get("raw")
            if not raw:
                continue
            url = req["url"]["raw"] if isinstance(req["url"], dict) else req["url"]
            path = "/" + url.split("}}/", 1)[1]
            yield pytest.param(f.name, it["name"], path, raw, id=f"{f.stem}::{it['name']}")


@pytest.mark.parametrize("coll,name,path,raw", list(_cases()))
def test_postman_body_validates_against_model(coll, name, path, raw):
    body = re.sub(r"\{\{(\w+)\}\}", lambda g: VARS[g.group(1)], raw)
    model = BODY_MODELS[path]
    model.model_validate(json.loads(body))  # raises if incompatible


def test_time_range_rejects_inverted_window():
    with pytest.raises(ValueError):
        m.TimeRange(start_time="2026-07-01T00:00:00Z", end_time="2026-05-01T00:00:00Z")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError):
        m.PriorityScoreRequest.model_validate({"alarm_id": "A1", "typo_field": 1})


def test_alarm_list_query_defaults_and_bounds():
    q = m.AlarmListQuery()
    assert (q.page, q.page_size, q.sort_by, q.sort_order) == (1, 50, "start_time", "desc")
    with pytest.raises(ValueError):
        m.AlarmListQuery(page_size=500)