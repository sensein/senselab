"""The browser viewer: the page is in sync with its parts, and its JavaScript decodes the schema.

Two kinds of test. The drift guards read the JavaScript as text and assert that every byte layout
and enumeration it carries is the one ``recording_vectors.py`` declares; those run everywhere. The
behaviour of the decoders is asserted by ``node --test`` over the ``.mjs`` suites beside this file,
which run when node is on PATH and skip when it is not.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from senselab.audio.workflows.triage import recording_vectors as rv
from senselab.audio.workflows.triage import viewer

VIEWER_DIR = viewer.SOURCE_DIR
DECODE_JS = (VIEWER_DIR / "decode.js").read_text(encoding="utf-8")
AXES_JS = (VIEWER_DIR / "axes.js").read_text(encoding="utf-8")
JS_TESTS = Path(__file__).parent / "viewer"


def _js_array(source: str, name: str) -> list[str]:
    """The terms of a single-line JavaScript string array.

    Args:
        source: The JavaScript.
        name: The variable name, e.g. ``SPAN_ROWS``.

    Returns:
        The quoted terms, in order.
    """
    match = re.search(rf"var {re.escape(name)} = \[(?P<body>[^\]]*)\]", source)
    assert match is not None, f"{name} is not declared as a flat array in the JavaScript"
    return re.findall(r"'([^']*)'", match.group("body"))


def _js_number(source: str, name: str) -> float:
    """One numeric constant out of the JavaScript.

    Args:
        source: The JavaScript.
        name: The variable name.

    Returns:
        Its value.
    """
    match = re.search(rf"var {re.escape(name)} = (-?[\d.]+);", source)
    assert match is not None, f"{name} is not declared in the JavaScript"
    return float(match.group(1))


def _js_pair(source: str, name: str) -> tuple[float, float]:
    """One two-element numeric range out of the JavaScript.

    Args:
        source: The JavaScript.
        name: The variable name.

    Returns:
        The low and high bounds.
    """
    match = re.search(rf"var {re.escape(name)} = \[(-?[\d.]+), *(-?[\d.]+)\]", source)
    assert match is not None, f"{name} is not declared as a range in the JavaScript"
    return float(match.group(1)), float(match.group(2))


# ---------------------------------------------------------------- the drift guards


class TestTheJavaScriptCarriesThePythonLayout:
    """A byte enum that silently stops matching the producer is the failure these exist for."""

    def test_the_span_rows_match(self) -> None:
        """``SPAN_ROWS`` is one list, and the row byte indexes it."""
        assert _js_array(DECODE_JS, "SPAN_ROWS") == list(rv.SPAN_ROWS)

    def test_the_classifiers_match(self) -> None:
        """``CLASSIFIERS`` is one list, and the classifier byte indexes it."""
        assert _js_array(DECODE_JS, "CLASSIFIERS") == list(rv.CLASSIFIERS)

    def test_the_word_outcomes_match(self) -> None:
        """``WORD_OUTCOMES`` is one list, and the outcome byte indexes it."""
        assert _js_array(DECODE_JS, "WORD_OUTCOMES") == list(rv.WORD_OUTCOMES)

    def test_the_lanes_match(self) -> None:
        """``LANES`` is one list, and the lane byte indexes it."""
        assert _js_array(DECODE_JS, "LANES") == list(rv.LANES)

    def test_the_quantiser_scales_match(self) -> None:
        """Full scale is 65535 and a trace is 256 points, on both sides."""
        assert _js_number(DECODE_JS, "TIME_SCALE") == rv.TIME_SCALE
        assert _js_number(DECODE_JS, "TRACE_POINTS") == rv.TRACE_POINTS
        assert _js_number(DECODE_JS, "UNKNOWN_CODE") == rv.UNKNOWN_CODE
        assert _js_number(DECODE_JS, "SCHEMA_VERSION") == rv.SCHEMA_VERSION

    def test_the_value_ranges_match(self) -> None:
        """Envelope, continuity and score ranges are the producer's own."""
        assert _js_pair(DECODE_JS, "ENVELOPE_DBFS_RANGE") == rv.ENVELOPE_DBFS_RANGE
        assert _js_pair(DECODE_JS, "CONTINUITY_RANGE") == rv.CONTINUITY_RANGE
        assert _js_pair(DECODE_JS, "SCORE_RANGE") == rv.SCORE_RANGE

    def test_the_squim_ranges_match(self) -> None:
        """Each SQUIM metric keeps its own range; swapping two is the mutation this kills."""
        for name, (low, high) in rv.SQUIM_RANGES:
            match = re.search(rf"{name}: \[(-?[\d.]+), *(-?[\d.]+)\]", DECODE_JS)
            assert match is not None, f"{name} has no range in the JavaScript"
            assert (float(match.group(1)), float(match.group(2))) == (low, high)

    def test_the_record_sizes_match_the_struct_layouts(self) -> None:
        """Each block's record size is what ``struct`` packs for its layout."""
        import struct

        expected = {
            "spans": struct.calcsize("<" + rv.SPANS_LAYOUT),
            "span_labels": struct.calcsize("<" + rv.SPAN_LABELS_LAYOUT),
            "span_squim": struct.calcsize("<" + rv.SPAN_SQUIM_LAYOUT),
            "asr_words": struct.calcsize("<" + rv.ASR_WORDS_LAYOUT),
            "pii_marks": struct.calcsize("<" + rv.PII_MARKS_LAYOUT),
            "branch_lanes": struct.calcsize("<" + rv.BRANCH_LANES_LAYOUT),
        }
        block = re.search(r"var RECORD_SIZES = \{(?P<body>[^}]*)\}", DECODE_JS)
        assert block is not None
        found = {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", block.group("body"))}
        assert found == expected

    def test_the_measurement_names_match(self) -> None:
        """All twenty-nine names, in their four kinds, are the producer's own."""
        assert _js_array(AXES_JS, "SCALAR_MEASUREMENTS") == list(rv.SCALAR_MEASUREMENTS)
        assert _js_array(AXES_JS, "VECTOR_MEASUREMENTS") == list(rv.VECTOR_MEASUREMENTS)
        assert _js_array(AXES_JS, "MATRIX_MEASUREMENTS") == list(rv.MATRIX_MEASUREMENTS)
        assert _js_array(AXES_JS, "CATEGORICAL_MEASUREMENTS") == list(rv.CATEGORICAL_MEASUREMENTS)

    def test_the_three_owner_directed_axes_come_first(self) -> None:
        """Owner-directed 2026-09-22: participant, task, verdict, in that order."""
        assert _js_array(AXES_JS, "DEFAULT_AXES")[:3] == ["participant", "task", "verdict"]

    def test_there_are_at_most_ten_axes_and_ten_defaults(self) -> None:
        """Up to ten axes, as directed, and the shipped set fills them."""
        assert _js_number(AXES_JS, "MAX_AXES") == 10
        assert len(_js_array(AXES_JS, "DEFAULT_AXES")) == 10


class TestTheWorkedExampleIsAssertedOnBothSides:
    """The five bytes the schema quotes are asserted in Python and in JavaScript."""

    def test_the_javascript_suite_asserts_the_same_five_bytes(self) -> None:
        """The JS suite carries the literal bytes, not a round trip that cannot see an error."""
        text = (JS_TESTS / "decode.test.mjs").read_text(encoding="utf-8")
        assert "bytes(0x02, 0x00, 0x40, 0x00, 0x80)" in text
        assert "big-endian" in text

    def test_python_still_produces_those_five_bytes(self) -> None:
        """The producer side of the same example, so the two cannot drift apart silently."""
        blob = rv.encode_records(
            [(rv.SPAN_ROWS.index("A"), rv.quantise_time(1.0, 4.0), rv.quantise_time(2.0, 4.0))], "BHH"
        )
        assert blob == bytes([0x02, 0x00, 0x40, 0x00, 0x80])


# ---------------------------------------------------------------- the built page


class TestThePageIsWhatItsPartsSay:
    """The committed page is a build output; a stale one is a page that decodes yesterday."""

    def test_the_committed_page_matches_a_fresh_build(self) -> None:
        """Rebuild and compare bytes; the build carries no timestamp, so this is exact."""
        assert viewer.BUILT_PAGE.is_file(), "the built page is not committed"
        fresh = viewer.build()
        assert viewer.BUILT_PAGE.read_text(encoding="utf-8") == fresh, (
            "recording_vectors_viewer.html is stale; run scripts/triage_vectors_viewer.py"
        )

    def test_every_part_is_inlined_and_none_is_left_over(self) -> None:
        """The shell and PARTS agree, and the build refuses when they do not."""
        shell = (VIEWER_DIR / "shell.html").read_text(encoding="utf-8")
        assert set(viewer.INLINE_PATTERN.findall(shell)) == set(viewer.PARTS)

    def test_the_page_has_no_inline_markers_left(self) -> None:
        """A marker that survived the build would be a part silently dropped."""
        assert not viewer.INLINE_PATTERN.search(viewer.build())

    def test_the_page_fetches_nothing(self) -> None:
        """It is opened from file:// and reads a local File; no origin is contacted."""
        page = viewer.BUILT_PAGE.read_text(encoding="utf-8")
        for pattern in (r'src\s*=\s*["\']https?:', r'href\s*=\s*["\']https?:', r"@import\s"):
            assert not re.search(pattern, page), f"the page references {pattern}"
        assert "XMLHttpRequest" not in page
        assert not re.search(r"\bfetch\s*\(\s*['\"`]", page)

    def test_the_page_carries_the_handling_the_parquet_carries(self) -> None:
        """A page that renders PII says so before it is opened."""
        page = viewer.BUILT_PAGE.read_text(encoding="utf-8")
        assert "PII" in page
        assert "do not host" in page
        assert "noindex" in page

    def test_no_part_carries_data_from_a_recording(self) -> None:
        """Nothing under the repository may contain a transcript, a stem or a PII extent."""
        for path in [viewer.BUILT_PAGE, *(VIEWER_DIR / p for p in viewer.PARTS)] + sorted(JS_TESTS.glob("*.mjs")):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"sub-[0-9a-f]{8}-[0-9a-f]{4}", text), f"{path} carries a participant id"
            assert not re.search(r"ses-[0-9A-F]{8}-[0-9A-F]{4}", text), f"{path} carries a session id"


class TestTheAxisDefaultsAreTheOnesTheSpecDerives:
    """The defaults are a claim about what discriminates, and the claim is written down."""

    def test_the_defaults_are_all_assignable_columns(self) -> None:
        """A default that is not assignable would render as a silently empty axis."""
        defaults = _js_array(AXES_JS, "DEFAULT_AXES")
        for name in defaults:
            pattern = rf"name: '{re.escape(name)}'[^}}]*?assignable: false"
            assert not re.search(pattern, AXES_JS), f"{name} is a default but is not assignable"

    def test_every_default_is_a_real_parquet_column(self) -> None:
        """Each default names a column the producer's schema actually writes."""
        written = {field.name for field in rv.schema()}
        for name in _js_array(AXES_JS, "DEFAULT_AXES"):
            base = name.split(".")[0]
            assert base in written, f"{name} is not a column recording_vectors.py writes"

    def test_the_corpus_read_never_names_a_binary_column(self) -> None:
        """First paint must not pay for the blocks; the block list is the app's, not the axes'."""
        app = (VIEWER_DIR / "app.js").read_text(encoding="utf-8")
        block = re.search(r"var BLOCK_COLUMNS = \[(?P<body>.*?)\];", app, re.S)
        assert block is not None
        named = set(re.findall(r"'([^']+)'", block.group("body")))
        binary = {field.name for field in rv.schema() if str(field.type) == "binary"}
        assert binary <= named, f"the app never reads {sorted(binary - named)}"


# ---------------------------------------------------------------- the JavaScript suites


def _node() -> str | None:
    """Where node is, or None.

    Returns:
        The executable path, or None when node is not installed.
    """
    return shutil.which("node")


@pytest.mark.skipif(_node() is None, reason="node is not on PATH; the JavaScript suites cannot run")
@pytest.mark.parametrize("suite", ["decode.test.mjs", "axes.test.mjs"])
def test_the_javascript_suite_passes(suite: str) -> None:
    """Run one ``node --test`` suite and fail with its output when it does not pass.

    Args:
        suite: The file name under ``viewer/``.
    """
    node = _node()
    assert node is not None
    result = subprocess.run(
        [node, "--test", "--test-reporter=tap", str(JS_TESTS / suite)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"^# fail 0$", result.stdout, re.M), result.stdout


@pytest.mark.skipif(_node() is None, reason="node is not on PATH; the JavaScript suites cannot run")
def test_the_javascript_suites_are_not_empty() -> None:
    """A suite that silently stopped collecting would pass; assert it ran real tests."""
    node = _node()
    assert node is not None
    suites = sorted(str(p) for p in JS_TESTS.glob("*.test.mjs"))
    assert suites, "no JavaScript suites were found"
    result = subprocess.run(
        [node, "--test", "--test-reporter=tap", *suites],
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = re.search(r"^# pass (\d+)$", result.stdout, re.M)
    assert passed is not None, result.stdout
    assert int(passed.group(1)) >= 40, result.stdout


def test_the_vendored_reader_is_the_version_the_build_names() -> None:
    """The bundle's file name is its provenance; a swap without a rename would hide it."""
    vendored = [p for p in viewer.PARTS if p.startswith("vendor/")]
    assert vendored == ["vendor/hyparquet-1.31.1-fzstd-0.1.1.bundle.js"]
    bundle = (VIEWER_DIR / vendored[0]).read_text(encoding="utf-8")
    assert "Hyparquet" in bundle
    assert len(bundle) < 200_000, "the vendored reader grew unexpectedly"


def test_the_summary_json_shape_is_the_one_the_page_assumes(tmp_path: Path) -> None:
    """The merge writes per-column null counts; the page's null-not-zero prose cites them."""
    summary = {"rows": 3, "schema_version": rv.SCHEMA_VERSION, "null_counts": {"pii_findings_n": 1}}
    path = tmp_path / "recording_vectors.summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == rv.SCHEMA_VERSION
    assert loaded["null_counts"]["pii_findings_n"] == 1
