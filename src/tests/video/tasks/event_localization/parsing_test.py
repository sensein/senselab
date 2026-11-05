"""Prompt, schema and reply validation."""

import json

import pytest

from senselab.video.tasks.event_localization import build_prompt, parse_reply, segment_list_schema

LABELS = ["hands flapping", "jumping"]


def test_schema_has_label_enum() -> None:
    """The label is a closed enum, times are numbers, nothing else is allowed."""
    s = segment_list_schema(LABELS, extra={"child_reacts": ["yes", "no"]}, max_items=20)
    item = s["items"]
    assert s["type"] == "array" and s["maxItems"] == 20
    assert item["properties"]["label"]["enum"] == LABELS
    assert item["properties"]["start_s"] == {"type": "number"}
    assert item["properties"]["child_reacts"]["enum"] == ["yes", "no"]
    assert item["required"] == ["label", "start_s", "end_s", "child_reacts"] and item["additionalProperties"] is False
    json.dumps(s)
    with pytest.raises(ValueError):
        segment_list_schema([])


def test_prompt_lists_labels_and_window() -> None:
    """The prompt names the window length, every label with its definition, and the output format."""
    p = build_prompt({"task": "List intervals.", "labels": {"jumping": "bouncing", "rocking": "swaying"}}, 12.5)
    assert "12.5 seconds" in p and "- jumping: bouncing" in p and '"label": "jumping"' in p
    assert build_prompt({"head": "Clip of {win} s.", "task": "t", "labels": {"a": "b"}}, 3.0).startswith(
        "Clip of 3.0 s."
    )


def test_parse_valid_and_folded() -> None:
    """Exact labels pass; folded spellings count as folded and are refused when strict."""
    s, c = parse_reply('[{"label": "jumping", "start_s": 2.5, "end_s": 6}]', LABELS, 30.0, strict=True)
    assert s == [{"label": "jumping", "start_s": 2.5, "end_s": 6.0}] and c["folded"] == 0
    s, c = parse_reply('```json\n[{"label": "Hands_flapping", "start": "00:03", "end": "0:05.5"}]\n```', LABELS, 30.0)
    assert s[0]["label"] == "hands flapping" and (s[0]["start_s"], s[0]["end_s"]) == (3.0, 5.5) and c["folded"] == 1
    s, c = parse_reply('[{"label": "Hands_flapping", "start_s": 1, "end_s": 2}]', LABELS, 30.0, strict=True)
    assert s == [] and c["bad_label"] == 1


def test_parse_drops_and_counts() -> None:
    """Unknown labels, reversed or missing times, non-objects are dropped; out-of-window times are clamped."""
    reply = (
        '[{"label": "dancing", "start_s": 1, "end_s": 2}, {"label": "jumping", "start_s": 9, "end_s": 4},'
        ' {"label": "jumping", "start_s": 28, "end_s": 45}, {"label": "jumping"}, "x"]'
    )
    s, c = parse_reply(reply, LABELS, 30.0)
    assert len(s) == 1 and s[0]["end_s"] == 30.0
    assert (c["bad_label"], c["bad_order"], c["clamped"], c["bad_item"]) == (1, 1, 1, 2)


def test_parse_empty_unparsed_salvaged_point() -> None:
    """An empty list is a result, prose is unparsed, a truncated list keeps complete items, points need point_labels."""
    assert parse_reply("[]", LABELS, 30.0)[1]["empty"] == 1
    assert parse_reply("None.", LABELS, 30.0)[1]["empty"] == 1
    s, c = parse_reply("I cannot tell.", LABELS, 30.0)
    assert s == [] and c["unparsed"] == 1
    s, c = parse_reply(
        '[{"label": "jumping", "start_s": 1.0, "end_s": 2.0}, {"label": "jumping", "start_s": 5.0, "en', LABELS, 30.0
    )
    assert len(s) == 1 and c["salvaged"] == 1
    s, c = parse_reply('[{"label": "jumping", "start_s": 4.0, "end_s": 4.0}]', LABELS, 30.0)
    assert s == [] and c["bad_order"] == 1
    s, c = parse_reply(
        '<think>x</think>{"segments": [{"label": "call", "start_s": 4.0, "end_s": 4.0, "reacts": "yes"}]}',
        ["call"],
        30.0,
        point_labels=["call"],
    )
    assert s[0]["extra"] == {"reacts": "yes"}
