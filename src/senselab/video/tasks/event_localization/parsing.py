"""Prompt, JSON schema and reply validation for a model that lists labelled time segments in a window."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from senselab.video.tasks.event_localization.windows import Segment

PROMPT_HEAD = (
    "You are watching {win} seconds of a recording, with its soundtrack. "
    "Time in this clip starts at 0.0 seconds and ends at {win} seconds."
)
PROMPT_RULES = (
    "Rules:\n"
    "- Report every separate occurrence as its own item; the same label may occur several times, "
    "and different labels may occur in the same clip.\n"
    "- start_s and end_s are seconds from the start of this clip, between 0.0 and {win}, with one decimal; "
    "end_s is after start_s.\n"
    "- Use only the labels listed above, spelled exactly.\n"
    "- If nothing from the list occurs in this clip, reply with an empty list: []\n"
    "Reply with only a JSON list and no other text, for example "
    '[{{"label": "{ex}", "start_s": 2.5, "end_s": 6.0}}]'
)

START_KEYS = ("start_s", "start", "start_time", "begin", "onset", "from")
END_KEYS = ("end_s", "end", "end_time", "stop", "offset", "to")
LABEL_KEYS = ("label", "behavior", "behaviour", "class")
COUNT_KEYS = ("unparsed", "salvaged", "bad_item", "bad_label", "folded", "bad_order", "clamped", "empty")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_OBJECT = re.compile(r"\{[^{}]*\}")
_NUM = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?\s*$", re.I)
_CLOCK = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")
_NONE_REPLY = re.compile(r"(?i)\W*(none|no events?|nothing|\[\s*\])\W*")


def build_prompt(family: Mapping[str, Any], window_s: float) -> str:
    """Build the prompt for one window.

    Args:
        family: ``task`` (one sentence of what to list), ``labels`` (label to definition, in prompt order), an optional
            ``head`` (opening text with a ``{win}`` placeholder for the window length, replacing `PROMPT_HEAD`) and an
            optional ``extra`` line of rules.
        window_s: Window length in seconds.

    Returns:
        The prompt text.
    """
    win = f"{window_s:.1f}"
    labels = family["labels"]
    lines = [family.get("head", PROMPT_HEAD).format(win=win), "", family["task"], "", "Labels:"]
    lines += [f"- {k}: {v}" for k, v in labels.items()]
    if family.get("extra"):
        lines += ["", family["extra"]]
    lines += ["", PROMPT_RULES.format(win=win, ex=next(iter(labels)))]
    return "\n".join(lines)


def segment_list_schema(
    labels: Sequence[str], extra: Optional[Mapping[str, Sequence[str]]] = None, max_items: Optional[int] = None
) -> Dict[str, Any]:
    """JSON schema of a segment list whose labels come from a closed set, for constrained decoding.

    The reply is an array of objects ``{"label": <one of labels>, "start_s": <number>, "end_s": <number>}``; the array
    may be empty (nothing occurs). Range checks (times inside the window, end after start) are not expressed, because
    not every grammar backend supports them; `parse_reply` enforces them.

    Args:
        labels: The closed label set.
        extra: Further required string properties of each item, mapped to their allowed values.
        max_items: Optional cap on the number of items.

    Returns:
        A JSON schema as a dict.

    Raises:
        ValueError: If ``labels`` is empty.
    """
    if not labels:
        raise ValueError("labels must not be empty")
    props: Dict[str, Any] = {
        "label": {"type": "string", "enum": [str(x) for x in labels]},
        "start_s": {"type": "number"},
        "end_s": {"type": "number"},
    }
    for k, vals in (extra or {}).items():
        props[k] = {"type": "string", "enum": [str(v) for v in vals]}
    schema: Dict[str, Any] = {
        "type": "array",
        "items": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
    }
    if max_items is not None:
        schema["maxItems"] = int(max_items)
    return schema


def _seconds(v: object) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        m = _NUM.match(v)
        if m:
            return float(m.group(1))
        m = _CLOCK.match(v)
        if m:
            h, mi, s = m.groups()
            return int(h or 0) * 3600 + int(mi) * 60 + float(s)
    return None


def _loads(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _first_list(text: str) -> Optional[List[Any]]:
    """The first JSON list in ``text`` (also inside a dict under ``segments``/``events``/...), else ``None``."""
    for cand in [text] + _FENCE.findall(text):
        cand = cand.strip()
        val = _loads(cand)
        if val is None:
            i, j = cand.find("["), cand.rfind("]")
            if 0 <= i < j:
                val = _loads(cand[i : j + 1])
        if isinstance(val, dict):
            for k in ("segments", "events", "items", "result", "results"):
                if isinstance(val.get(k), list):
                    return val[k]
            continue
        if isinstance(val, list):
            return val
    return None


def _norm(x: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(x).strip().lower())


def parse_reply(
    text: str,
    labels: Sequence[str],
    window_s: float,
    aliases: Optional[Mapping[str, str]] = None,
    point_labels: Sequence[str] = (),
    strict: bool = False,
) -> Tuple[List[Segment], Dict[str, int]]:
    """Validate a model reply into segments relative to the window, and count what was dropped or repaired.

    A reply is read as a JSON list (optionally in a code fence, after a ``<think>`` block, or in a dict under
    ``segments``). A reply cut off in the middle of the list keeps its complete objects (``salvaged``). Items are
    dropped when they are not objects, have no usable time, name a label outside ``labels``, or end before they
    start; times outside ``[0, window_s]`` are clamped. A segment of zero length is kept only for a label in
    ``point_labels``.

    Args:
        text: The model's reply.
        labels: The closed label set.
        window_s: Window length in seconds.
        aliases: Extra spellings mapped to a label (only used when ``strict`` is false).
        point_labels: Labels that are point events (``start_s`` may equal ``end_s``).
        strict: Accept a label only when it is spelled exactly as in ``labels`` and times only as numbers.

    Returns:
        ``(segments, counts)``. Counts: ``unparsed`` (no JSON list and not an explicit "none"), ``salvaged`` (a
        truncated list was repaired), ``bad_item``, ``bad_label``, ``folded`` (label accepted after case/underscore
        folding or an alias), ``bad_order``, ``clamped``, ``empty`` (the reply is an empty list). Extra string/number
        keys of an item are carried in ``extra``.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    body = _THINK.sub("", text or "").strip()
    val = _first_list(body)
    if val is None and body.lstrip("`").lstrip().lstrip("json").lstrip().startswith("["):
        objs = [_loads(m) for m in _OBJECT.findall(body)]
        val = [o for o in objs if isinstance(o, dict)] or None
        counts["salvaged"] = 1 if val else 0
    if val is None:
        counts["empty" if _NONE_REPLY.fullmatch(body or "") else "unparsed"] = 1
        return [], counts
    if not val:
        counts["empty"] = 1
        return [], counts
    exact = {str(x): str(x) for x in labels}
    fold = {_norm(x): str(x) for x in labels}
    for a, b in (aliases or {}).items():
        fold[_norm(a)] = b
    out: List[Segment] = []
    for it in val:
        if not isinstance(it, dict):
            counts["bad_item"] += 1
            continue
        raw = next((it[k] for k in LABEL_KEYS if k in it), "")
        lab = exact.get(str(raw))
        if lab is None and not strict:
            lab = fold.get(_norm(raw))
            counts["folded"] += lab is not None
        if lab is None:
            counts["bad_label"] += 1
            continue
        keys_s = START_KEYS[:1] if strict else START_KEYS
        keys_e = END_KEYS[:1] if strict else END_KEYS
        s = next((_seconds(it[k]) for k in keys_s if k in it and _seconds(it[k]) is not None), None)
        e = next((_seconds(it[k]) for k in keys_e if k in it and _seconds(it[k]) is not None), None)
        if s is None and e is None:
            counts["bad_item"] += 1
            continue
        s = e if s is None else s
        e = s if e is None else e
        assert s is not None and e is not None
        if e < s:
            counts["bad_order"] += 1
            continue
        s2, e2 = min(max(s, 0.0), window_s), min(max(e, 0.0), window_s)
        if (s2, e2) != (s, e):
            counts["clamped"] += 1
        if e2 <= s2 and lab not in point_labels:
            counts["bad_order"] += 1
            continue
        seg: Segment = {"label": lab, "start_s": round(s2, 3), "end_s": round(e2, 3)}
        skip = set(LABEL_KEYS) | set(START_KEYS) | set(END_KEYS)
        extra = {k: str(v) for k, v in it.items() if k not in skip and isinstance(v, (str, int, float, bool))}
        if extra:
            seg["extra"] = extra
        out.append(seg)
    return out, counts
