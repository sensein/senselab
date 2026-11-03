"""The `transformers` backend of `pii_spans`: prompt from the profile slice, parsed spans, faults and the worker."""

from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE
from senselab.text.tasks.pii_spans import (
    engine,
    transformers_backend,  # noqa: F401 - declares the backend
)
from senselab.text.tasks.pii_spans import transformers as worker
from senselab.text.tasks.pii_spans.profile import PiiProfile, packaged_profile
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.kernel.results import TextSpans
from senselab_core.kernel.signals import PlainText
from senselab_core.kernel.wire import from_wire_form
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import Call, build_key
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.testing.contracts import BASE_CONTEXT, contract_cases
from senselab_core.testing.workers import RealWorkers, run_alone
from senselab_worker.faults import ContentFault, FaultClass, ResourceFault

LOCK_RECORDS = packaged_lock_records("senselab")

HOST = HardwareInfo(
    cpu_count=1,
    memory_bytes=1,
    gpus=(GpuInfo(name="H100", memory_bytes=80 * 2**30, driver=None, index=0, uuid="GPU-0", compute_capability="9.0"),),
    apple_silicon=False,
    platform="linux-x86_64",
)
PROFILE = packaged_profile().model_dump(mode="json")
PARAMS: dict[str, Any] = {"profile": PROFILE, "family": None, "backend_params": {}}
NAMES = [c["name"] for c in PROFILE["categories"]]


def _item(name: str, content: str) -> dict[str, Any]:
    return {"id": name, "inputs": {"text": PlainText(content=content).model_dump()}}


def _run(items: list[dict[str, Any]], faults: dict[str, Any] | None = None, **params: Any) -> Any:  # noqa: ANN401
    state = worker.load({"stub": True, "device": "cpu"})
    return worker.handle(state, items, {**PARAMS, **params, **({"_fault": faults} if faults else {})})


def _spans(result: dict[str, Any]) -> TextSpans:
    out = from_wire_form(result["outputs"]["spans"])
    assert isinstance(out, TextSpans)
    return out


def _answer(*spans: tuple[str, str, Any]) -> str:
    quoted = ", ".join(f'{{"quote": "{q}", "category": "{c}", "score": {s}}}' for q, c, s in spans)
    return f'Here you go:\n```json\n{{"spans": [{quoted}]}}\n```'


@pytest.mark.logic
def test_prompt_built_from_profile_slice() -> None:
    """The prompt carries the slice's instruction, every category, the matching family's guidance only, and the text."""
    profile = {
        "instruction": "MARK-THE-IDENTIFYING",
        "categories": [{"name": "ALPHA", "description": "first kind"}, {"name": "BETA", "description": "second kind"}],
        "families": [
            {"names": ["story-recall"], "guidance": "GUIDE-STORY"},
            {"names": ["open-response-questions"], "guidance": "GUIDE-OPEN"},
        ],
    }
    prompt = engine.build_prompt(profile, "Call me Sam.", "story-recall-b")
    for part in ("MARK-THE-IDENTIFYING", "- ALPHA: first kind", "- BETA: second kind", "GUIDE-STORY"):
        assert part in prompt
    assert "GUIDE-OPEN" not in prompt and engine.text_of(prompt) == "Call me Sam."
    assert "GUIDE-STORY" not in engine.build_prompt(profile, "Call me Sam.", "story")
    packaged = engine.build_prompt(PROFILE, "x", "productive-vocabulary")
    assert PROFILE["instruction"] in packaged and PROFILE["families"][0]["guidance"] in packaged


@pytest.mark.logic
def test_packaged_slice_declares_its_derivation_and_refuses_another_format() -> None:
    """The packaged file states how its values were derived, and holds only fields the profile model takes."""
    from importlib import resources

    raw = resources.files("senselab").joinpath("data", "profiles", "pii_spans.yaml").read_text("utf-8")
    assert "derivation:" in raw.split("instruction:")[0]
    assert raw.startswith("format: profile-slice@1.0") and len(packaged_profile().categories) == len(NAMES)
    with pytest.raises(ValueError, match="bogus"):
        PiiProfile.model_validate({**PROFILE, "bogus": 1})


@pytest.mark.logic
def test_findings_parsed_into_text_spans() -> None:
    """Every occurrence of a quoted span becomes a span of the quote's category and score, in code points."""
    text = "😀 Ann met Ann at 555-0100."
    completion = _answer(("Ann", "PERSON", 0.9), ("555-0100", "CONTACT", 2))
    spans = engine.parse_completion(completion, text, NAMES)
    assert spans == [(2, 5, "PERSON", 0.9), (10, 13, "PERSON", 0.9), (17, 25, "CONTACT", 1.0)]
    out = from_wire_form(engine.spans_wire_form(PlainText(content=text).model_dump(), spans))
    assert isinstance(out, TextSpans) and out.length == len(text) and out.text == PlainText(content=text).identity
    assert out.start.tolist() == [2, 10, 17] and out.end.tolist() == [5, 13, 25]
    assert (
        out.label.tolist() == ["PERSON", "PERSON", "CONTACT"]
        and out.score is not None
        and numpy.allclose(out.score, [0.9, 0.9, 1.0])
    )
    assert [text[a:b] for a, b in zip(out.start, out.end, strict=True)] == ["Ann", "Ann", "555-0100"]


@pytest.mark.logic
def test_overlapping_quotes_of_one_category_merge() -> None:
    """Overlapping spans of one category become their union at the higher score."""
    completion = _answer(("Ann Lee", "PERSON", 0.8), ("Lee Park", "PERSON", 0.9))
    assert engine.parse_completion(completion, "Ann Lee Park", NAMES) == [(0, 12, "PERSON", 0.9)]


@pytest.mark.logic
def test_overlapping_quotes_of_two_categories_are_a_content_fault() -> None:
    """Spans of different categories over the same characters are not resolved by dropping one."""
    completion = _answer(("Ann Lee", "PERSON", 0.8), ("Lee Park", "LOCATION", 0.9))
    with pytest.raises(ContentFault, match="overlap"):
        engine.parse_completion(completion, "Ann Lee Park", NAMES)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("completion", "match"),
    [
        ("no json at all", "span object"),
        ('{"spans": [{"quote": "Ann"}]}', "span object"),
        (_answer(("Ann", "COLOUR", 0.5)), "COLOUR"),
        (_answer(("Bob", "PERSON", 0.5)), "does not occur"),
        (_answer(("Ann", "PERSON", '"high"')), "not a number"),
    ],
    ids=["no-json", "missing-field", "category", "quote", "score"],
)
def test_malformed_answers_are_content_faults(completion: str, match: str) -> None:
    """An answer that is not the object asked for, or names a span it cannot locate, fails loudly."""
    with pytest.raises(ContentFault, match=match):
        engine.parse_completion(completion, "Ann", NAMES)


@pytest.mark.logic
def test_nothing_identifying_gives_no_spans() -> None:
    """An empty list is an answer: zero spans over the text, not a fault."""
    assert engine.parse_completion('{"spans": []}', "The weather is fine.", NAMES) == []
    [result] = _run([_item("a", "The weather is fine.")])
    assert len(_spans(result).start) == 0


@pytest.mark.logic
def test_stub_marks_digit_runs_and_ignores_batch_mates() -> None:
    """The stub answers each item from its own text, alone or in a batch, as typed `TextSpans`."""
    a, b = _item("a", "Dial 555 then 0100."), _item("b", "No digits here.")
    together = _run([a, b])
    assert _spans(together[0]).label.tolist() == ["ID", "ID"] and len(_spans(together[1]).start) == 0
    assert numpy.array_equal(_spans(together[0]).start, _spans(_run([a])[0]).start)


@pytest.mark.logic
def test_faults_stay_with_their_item() -> None:
    """A content fault and an out of memory leave the third item unchanged, typed, never a gap."""
    items = [_item(i, f"{i} rang 555{n}") for n, i in enumerate("abc")]
    clean = _run(items)
    faulted = _run(items, faults={"fail_items": ["a"], "oom_items": ["b"]})
    assert isinstance(faulted[0]["error"], ContentFault) and isinstance(faulted[1]["error"], ResourceFault)
    assert numpy.array_equal(_spans(faulted[2]).start, _spans(clean[2]).start)


@pytest.mark.logic
def test_unknown_backend_param_is_refused() -> None:
    """The backend takes no params; one given is an error, not ignored."""
    with pytest.raises(ValueError, match="bogus"):
        _run([_item("a", "x")], backend_params={"bogus": 1})


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": worker.MODEL_ID, "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            worker.load(init)


@pytest.mark.logic
def test_profile_edit_changes_the_key() -> None:
    """The profile is a param of the call, so a different instruction is a different logical key."""
    spec = REGISTRY.spec("pii_spans")
    backend = next(b for b in REGISTRY.backends("pii_spans") if b.name == "transformers")
    entry = CATALOGUE.entry(worker.MODEL_ID)
    placed = resolve_execution(spec, entry, ExecutionProfile.load(), HOST, REGISTRY, CATALOGUE)
    context = ExecutionContext.model_validate(BASE_CONTEXT)

    def key(profile: dict[str, Any]) -> Any:  # noqa: ANN401
        params = spec.params.model_validate({"profile": profile})
        inputs = {"text": PlainText(content="x").identity}
        return build_key(Call(task=spec, params=params, inputs=inputs), placed, context)

    assert backend.name == placed.backend.name
    assert key(PROFILE) == key(dict(PROFILE)) and key(PROFILE) != key({**PROFILE, "instruction": "Mark names."})


@pytest.mark.contract
def test_backend_serves_the_catalogue_model_and_declares_its_isolated_env() -> None:
    """One isolated backend `transformers` over the catalogue's default `pii_spans` model, pinned by commit."""
    spec = next(b for b in REGISTRY.backends("pii_spans") if b.name == "transformers")
    assert spec.name == "transformers" and spec.models == (worker.MODEL_ID,) and spec.batch_invariant
    assert spec.env.kind == "isolated" and spec.env.name == "pii-spans" and spec.worker in spec.worker_sources
    entry = CATALOGUE.entry(worker.MODEL_ID)
    assert list(entry.backends) == ["transformers", "vllm"] and spec.stack in entry.stacks
    assert list(entry.default_for) == ["pii_spans"]
    placed = resolve_execution(REGISTRY.spec("pii_spans"), entry, ExecutionProfile.load(), HOST, REGISTRY, CATALOGUE)
    assert placed.backend.name == "transformers"


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("pii_spans::", "pii_spans/transformers::", "transformers::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


@pytest.mark.env
@pytest.mark.model
@pytest.mark.gpu
def test_pii_spans_transformers_finds_the_identifying_span_with_real_weights(real_worker: RealWorkers) -> None:
    """On the pinned model a name and a number are covered by spans; a content fault leaves batch-mates unchanged."""
    text = "My name is Margaret Hollis and you can reach me on 617-555-0142."
    items = [_item("a", text), _item("b", text), _item("c", "The sky is blue today.")]
    pii = real_worker("pii_spans", "transformers")
    clean = pii.run(items, PARAMS)
    covered = {text[a:b] for a, b in zip(_spans(clean[0]).start, _spans(clean[0]).end, strict=True)}
    assert any("Hollis" in c for c in covered) and any("555-0142" in c for c in covered), covered
    faulted = pii.run(items, {**PARAMS, "_fault": {"fail_items": ["b"]}})
    assert faulted[1]["error"]["fault"] is FaultClass.CONTENT
    assert numpy.array_equal(_spans(faulted[0]).start, _spans(clean[0]).start)
    assert numpy.array_equal(_spans(faulted[2]).start, _spans(clean[2]).start)


@pytest.mark.limits
@pytest.mark.model
@pytest.mark.gpu
def test_pii_spans_transformers_edge_inputs_with_real_weights(real_worker: RealWorkers) -> None:
    """Empty, non-BMP, repeated and long texts each give spans inside the text or a typed fault."""
    client = real_worker("pii_spans", "transformers")
    cases = {
        "empty": "",
        "emoji": "😀😀 I am Dana Wu 😀",
        "repeated": "Dana Wu said Dana Wu lives here.",
        "long": "The meeting ran long. " * 400 + "Call Dana Wu on 617-555-0142.",
    }
    for name, outputs in run_alone(client, {n: _item(n, c) for n, c in cases.items()}, PARAMS).items():
        out = _spans({"outputs": outputs})
        assert out.length == len(cases[name]) and all(
            0 <= a <= b <= out.length for a, b in zip(out.start, out.end, strict=True)
        )
