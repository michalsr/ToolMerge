"""Tests for the planner JSON parser."""

from toolmerge.planner import parse_planner_response


def test_parse_fenced_json():
    raw = """\
Reasoning sentence here. The frame should look like X.
```json
{"queries": [{"tool": "siglip", "query": "person crossing river", "id": "Q1"}], "combine": "Q1"}
```
"""
    queries, combine = parse_planner_response(raw)
    assert queries == [{"tool": "siglip", "query": "person crossing river", "id": "Q1"}]
    assert combine == "Q1"


def test_parse_bare_json():
    raw = '{"queries": [{"tool": "tren", "query": "red car", "id": "Q1"}], "combine": "Q1"}'
    queries, combine = parse_planner_response(raw)
    assert queries == [{"tool": "tren", "query": "red car", "id": "Q1"}]
    assert combine == "Q1"


def test_parse_multi_query_compound_combine():
    raw = """\
```json
{"queries": [
  {"tool": "tren", "query": "man in blue jacket", "id": "Q1"},
  {"tool": "siglip", "query": "person at podium", "id": "Q2"}
], "combine": "Q1 AND Q2"}
```
"""
    queries, combine = parse_planner_response(raw)
    assert len(queries) == 2
    assert combine == "Q1 AND Q2"


def test_parse_drops_invalid_query_entries():
    raw = """\
```json
{"queries": [
  {"tool": "siglip", "query": "valid", "id": "Q1"},
  {"tool": "siglip", "no_query_field": "yes", "id": "Q2"}
], "combine": "Q1"}
```
"""
    queries, combine = parse_planner_response(raw)
    assert len(queries) == 1
    assert queries[0]["id"] == "Q1"
    assert combine == "Q1"


def test_parse_empty_on_no_json():
    queries, combine = parse_planner_response("just some text, no JSON here")
    assert queries == []
    assert combine == ""


def test_parse_empty_on_malformed_json():
    queries, combine = parse_planner_response("```json\n{not valid json}\n```")
    assert queries == []
    assert combine == ""
