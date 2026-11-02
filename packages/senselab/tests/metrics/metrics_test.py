"""The metric tasks: reference rows, each metric's definition, content faults, and the generated contract."""

from typing import Any

import numpy
import pytest
from pydantic import BaseModel
from senselab.metrics import der, eer, resample_error, span_f1, wer
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.results import Intervals, Metric, TextSpans, Transcript, TranscriptNode, Trials
from senselab_core.kernel.signals import PlainText, RegularGroup, audio_channels
from senselab_core.kernel.wire import from_wire_form
from senselab_core.testing.contracts import contract_cases
from senselab_core.testing.lock_licenses import packaged_lock_records

LOCK_RECORDS = packaged_lock_records("senselab")
CLOCK = "clock"


def _score(module: Any, inputs: dict[str, Any], params: dict[str, Any] | None = None) -> Metric:  # noqa: ANN401
    [result] = module.handle(module.load({}), [{"id": "a", "inputs": inputs}], params or {})
    assert "outputs" in result, result
    metric = from_wire_form(result["outputs"]["metric"])
    assert isinstance(metric, Metric)
    return metric


def _fault(module: Any, inputs: dict[str, Any], params: dict[str, Any] | None = None) -> Exception:  # noqa: ANN401
    [result] = module.handle(module.load({}), [{"id": "a", "inputs": inputs}], params or {})
    return result["error"]  # type: ignore[no-any-return]


def transcript(*texts: str) -> Transcript:
    """Return a transcript of one untimed segment per text."""
    return Transcript(segments=tuple(TranscriptNode(text=t) for t in texts))


def turns(*rows: tuple[str, float, float]) -> Intervals:
    """Return the speaker turns `(label, start_s, end_s)`."""
    return Intervals(
        start_ns=[round(s * 1e9) for _, s, _ in rows],
        end_ns=[round(e * 1e9) for _, _, e in rows],
        label=numpy.array([lab for lab, _, _ in rows]),
        clock_id=CLOCK,
    )


def trials(scores: list[float], same: list[bool]) -> Trials:
    """Return trials whose cosine similarities are `scores`: unit vectors at those angles' cosines."""
    first = numpy.array([[1.0, 0.0]] * len(scores))
    second = numpy.array([[s, numpy.sqrt(1.0 - s * s)] for s in scores])
    return Trials(first=first, second=second, same=numpy.array(same), space="space")


TEXT = PlainText(content="Ada Lovelace met Charles Babbage in London in 1833")


def spans(*rows: tuple[int, int, str]) -> TextSpans:
    """Return spans `(start, end, label)` over `TEXT`."""
    return TextSpans.over(
        TEXT,
        start=numpy.array([s for s, _, _ in rows], dtype=numpy.int64),
        end=numpy.array([e for _, e, _ in rows], dtype=numpy.int64),
        label=numpy.array([lab for _, _, lab in rows], dtype=str),
    )


def series(rows: list[float]) -> RegularGroup:
    """Return a one-channel series of the samples `rows`."""
    return RegularGroup(channels=audio_channels(1), sampling_rate=16000.0, data=numpy.array([rows], numpy.float32))


# pyannote.metrics' documented example (its `DiarizationErrorRate` tutorial): DER 0.516 over 31 s of reference speech.
PYANNOTE_REF = turns(("A", 0, 10), ("B", 12, 20), ("A", 24, 27), ("C", 30, 40))
PYANNOTE_HYP = turns(("a", 2, 13), ("d", 13, 14), ("b", 14, 20), ("c", 22, 38), ("d", 38, 40))

ROWS: list[Any] = [
    (wer, {"reference": transcript("hello world"), "hypothesis": transcript("hello duck")}, {}, 0.5),
    (wer, {"reference": transcript("the quick brown fox"), "hypothesis": transcript("the brown fox jumps")}, {}, 0.5),
    (
        wer,
        {
            "reference": transcript("Mr. Smith has twenty two dogs"),
            "hypothesis": transcript("mister smith has 22 dogs"),
        },
        {},
        0.0,
    ),
    (wer, {"reference": transcript("the colour red"), "hypothesis": transcript("the color red")}, {}, 0.0),
    (
        wer,
        {"reference": transcript("the colour red"), "hypothesis": transcript("the color red")},
        {"normalise": "none"},
        1 / 3,
    ),
    (
        wer,
        {"reference": transcript("a white paper"), "hypothesis": transcript("a whitepaper")},
        {"normalise": "none"},
        0.0,
    ),
    (wer, {"reference": transcript("red green", "blue black"), "hypothesis": transcript("red green blue")}, {}, 0.25),
    (der, {"reference": PYANNOTE_REF, "hypothesis": PYANNOTE_HYP}, {}, 16.0 / 31.0),
    (der, {"reference": turns(("A", 0, 10)), "hypothesis": turns(("x", 0, 10))}, {}, 0.0),
    (der, {"reference": turns(("A", 0, 10)), "hypothesis": turns(("x", 5, 15))}, {}, 1.0),
    (der, {"reference": turns(("A", 0, 10)), "hypothesis": turns(("x", 0, 5))}, {}, 0.5),
    (eer, {"trials": trials([0.9, 0.8, 0.7, 0.6, 0.4, 0.3], [True, True, False, True, False, False])}, {}, 1 / 3),
    (eer, {"trials": trials([0.9, 0.8, 0.2, 0.1], [True, True, False, False])}, {}, 0.0),
    (eer, {"trials": trials([0.1, 0.2, 0.8, 0.9], [True, True, False, False])}, {}, 1.0),
    (
        span_f1,
        {"reference": spans((0, 12, "name"), (17, 32, "name")), "hypothesis": spans((0, 12, "name"), (17, 32, "name"))},
        {},
        1.0,
    ),
    (
        span_f1,
        {
            "reference": spans((0, 12, "n"), (17, 32, "n"), (36, 42, "loc")),
            "hypothesis": spans((0, 12, "n"), (36, 42, "loc"), (43, 45, "x")),
        },
        {},
        2 / 3,
    ),
    (span_f1, {"reference": spans((0, 12, "n")), "hypothesis": spans((0, 12, "loc"))}, {}, 0.0),
    (resample_error, {"reference": series([1, 2, 3, 4]), "hypothesis": series([1, 2, 3, 5])}, {}, 1 / 30**0.5),
    (resample_error, {"reference": series([1, 2, 3, 4]), "hypothesis": series([1, 2, 3, 4])}, {}, 0.0),
]


@pytest.mark.logic
@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r[0].__name__.rsplit('.', 1)[-1]}-{ROWS.index(r)}")
def test_metric_matches_reference_rows(row: Any) -> None:  # noqa: ANN401
    """Each metric reproduces its published or hand-computed value."""
    module, inputs, params, expected = row
    assert _score(module, inputs, params).value == pytest.approx(expected, abs=1e-9)


@pytest.mark.logic
def test_metric_carries_name_unit_and_items() -> None:
    """The metric names its task, is a ratio, and counts the reference words."""
    metric = _score(wer, {"reference": transcript("red green blue black"), "hypothesis": transcript("red green blue")})
    assert (metric.name, metric.unit, metric.items) == ("wer", "ratio", 4)


@pytest.mark.logic
def test_span_f1_of_no_spans_against_no_spans_is_one_over_one_item() -> None:
    """A text with no entities that the model also leaves clean is a correct result, not a fault."""
    metric = _score(span_f1, {"reference": spans(), "hypothesis": spans()})
    assert (metric.value, metric.items) == (1.0, 1)


@pytest.mark.logic
def test_der_ignores_speaker_names_and_order() -> None:
    """Relabelling hypothesis speakers or reordering turns leaves the DER unchanged."""
    renamed = turns(("z9", 2, 13), ("q", 13, 14), ("m", 14, 20), ("k", 22, 38), ("q", 38, 40))
    shuffled = turns(("d", 38, 40), ("c", 22, 38), ("b", 14, 20), ("d", 13, 14), ("a", 2, 13))
    base = _score(der, {"reference": PYANNOTE_REF, "hypothesis": PYANNOTE_HYP}).value
    for hyp in (renamed, shuffled):
        assert _score(der, {"reference": PYANNOTE_REF, "hypothesis": hyp}).value == pytest.approx(base)


@pytest.mark.logic
def test_der_collar_forgives_boundary_error() -> None:
    """A collar of 1 s removes 0.5 s either side of each reference boundary, so a 0.4 s boundary slip scores zero."""
    inputs = {"reference": turns(("A", 0, 10)), "hypothesis": turns(("x", 0.4, 10.4))}
    assert _score(der, inputs).value == pytest.approx(0.08)
    assert _score(der, inputs, {"collar": 1.0}).value == 0.0


@pytest.mark.logic
def test_der_skip_overlap_leaves_overlapped_reference_out() -> None:
    """Time with two reference speakers is not scored when `skip_overlap` is set."""
    ref = turns(("A", 0, 10), ("B", 5, 10))
    hyp = turns(("x", 0, 10))
    assert _score(der, {"reference": ref, "hypothesis": hyp}).value == pytest.approx(5 / 15)
    assert _score(der, {"reference": ref, "hypothesis": hyp}, {"skip_overlap": True}).value == 0.0


@pytest.mark.logic
def test_der_counts_a_false_alarm_speaker() -> None:
    """Hypothesis speech where the reference is silent is a false alarm, over the reference's speech."""
    inputs = {"reference": turns(("A", 0, 10)), "hypothesis": turns(("x", 0, 10), ("y", 10, 15))}
    assert _score(der, inputs).value == pytest.approx(0.5)


@pytest.mark.logic
def test_eer_is_invariant_to_a_monotone_rescoring() -> None:
    """Scaling every embedding of a pair leaves the cosines, and so the EER, unchanged."""
    t = trials([0.9, 0.8, 0.7, 0.6, 0.4, 0.3], [True, True, False, True, False, False])
    scaled = Trials(first=t.first * 3.0, second=t.second * 0.5, same=t.same, space=t.space)
    assert _score(eer, {"trials": scaled}).value == _score(eer, {"trials": t}).value


@pytest.mark.logic
@pytest.mark.parametrize(
    ("module", "inputs", "params"),
    [
        (wer, {"reference": transcript(""), "hypothesis": transcript("a")}, {}),
        (
            der,
            {
                "reference": turns(("A", 0, 1)),
                "hypothesis": Intervals(start_ns=[0], end_ns=[1], label=numpy.array(["x"]), clock_id="other"),
            },
            {},
        ),
        (der, {"reference": turns(("A", 3, 3)), "hypothesis": turns(("x", 0, 1))}, {}),
        (eer, {"trials": trials([0.9, 0.8], [True, True])}, {}),
        (
            span_f1,
            {
                "reference": spans((0, 3, "n")),
                "hypothesis": TextSpans.over(PlainText(content="other"), start=[0], end=[1], label=numpy.array(["n"])),
            },
            {},
        ),
        (resample_error, {"reference": series([0, 0]), "hypothesis": series([1, 1])}, {}),
        (resample_error, {"reference": series([1, 2]), "hypothesis": series([1, 2, 3])}, {}),
    ],
    ids=[
        "wer-empty",
        "der-clock",
        "der-silent",
        "eer-one-kind",
        "span-f1-text",
        "rerr-silent",
        "rerr-length",
    ],
)
def test_input_with_nothing_to_score_is_a_content_fault(
    module: Any,  # noqa: ANN401
    inputs: dict[str, Any],
    params: dict[str, Any],
) -> None:
    """An input that has no defined score fails that item as a content fault."""
    from senselab_worker.faults import ContentFault  # noqa: PLC0415

    assert isinstance(_fault(module, inputs, params), ContentFault)


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("wer::", "der::", "eer::", "span_f1::", "resample_error::"))
        or name.startswith(("numpy-wer::", "numpy-der::", "numpy-eer::", "numpy-span-f1::", "numpy-resample-error::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """Each metric task and backend satisfies the generated contract."""
    case()


class _NoParams(BaseModel):
    pass


@pytest.mark.logic
def test_a_scorer_bug_is_infrastructure_with_a_gap_not_a_content_fault() -> None:
    """A bare `RuntimeError` out of a scorer is infrastructure with `gap: true`, never the item's content."""
    from senselab.metrics._worker import serve
    from senselab_worker.fake import FakeFaults
    from senselab_worker.faults import error_record

    def broken(inputs: Any, params: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("CUDA error: device-side assert triggered")

    [result] = serve(FakeFaults.from_json({}), [{"id": "a", "inputs": {}}], {}, _NoParams, broken)
    record = error_record(result["error"])
    assert (record["fault"], record["gap"]) == ("infrastructure", True)
