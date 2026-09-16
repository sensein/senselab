"""The QUALITY extend driver: the terminal node run over a store whose graph pass never reached it.

Nothing here is faked. QUALITY reads stored outputs only, so a finished run in miniature is enough:
a source WAV on disk, the store ADMIT wrote its ``recording`` stream into, and PREPROCESS's clip
spans with the amplitude measurement beside them — which is the shape the corpus is in once
``scripts/extend_clip_amplitudes.py`` has passed over it.

``specs/20260912-quality-clip-consistency/design.md`` is the design.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_verdict, live_entities, software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.preprocess import write_clip_spans
from senselab.audio.workflows.triage.nodes.quality import CLIP_FAMILY, CONTRADICTED_CLIP
from senselab.audio.workflows.triage.vocabulary import QUALITY, Outcome
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_quality.py"
_spec = importlib.util.spec_from_file_location("extend_quality_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_quality_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
"""The fixture's sampling rate, so a sample index and a time are one conversion apart."""

CLIP_RANGE = (8000, 8400)
"""The samples the seeded clip span covers."""


def _samples(*, clip_level: float, louder: float | None) -> np.ndarray:
    """A quiet bed with one plateau, and optionally one unclipped sample louder than it.

    Args:
        clip_level: The level the plateau holds.
        louder: An amplitude to place at sample 24000, or None to leave the bed there.

    Returns:
        The recording, mono float32.
    """
    grid = np.arange(2 * SR) / SR
    out = (0.05 * np.sin(2 * np.pi * 220.0 * grid)).astype(np.float32)
    out[CLIP_RANGE[0] : CLIP_RANGE[1]] = np.float32(clip_level)
    if louder is not None:
        out[24000] = np.float32(louder)
    return out


def _seed_run(
    root: Path,
    *,
    clip_level: float = 0.5,
    louder: float | None = 0.9,
    spans: bool = True,
    amplitudes: bool = True,
    preprocessed: bool = True,
) -> Path:
    """Write one finished run: a source WAV, and a store that stops before QUALITY.

    Args:
        root: The run root, created if absent.
        clip_level: The plateau's level.
        louder: An unclipped sample's amplitude, or None for a recording nothing contradicts.
        spans: Whether the store carries a clip span at all.
        amplitudes: Whether the clip-amplitude measurement is written beside the spans, as the
            clip extend pass leaves it. False is the corpus before that pass.
        preprocessed: Whether PREPROCESS recorded a verdict, which QUALITY reports having followed.

    Returns:
        The source WAV's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    source = root.parent / f"{root.name}.wav"
    sf.write(str(source), _samples(clip_level=clip_level, louder=louder), SR)
    sf.write(str(run_dir / "streams" / "enhanced.flac"), _samples(clip_level=clip_level, louder=None), SR)

    store = ProvStore(run_id=root.name)
    config = load_triage_config()
    admitted = admit(store, source, config, run_dir=run_dir)
    assert admitted.audio is not None
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
    store.was_associated_with(activity, agent)
    extents = [(CLIP_RANGE[0] / SR, CLIP_RANGE[1] / SR)] if spans else []
    if amplitudes:
        write_clip_spans(
            store,
            activity,
            agent,
            audio=admitted.audio,
            extents=extents,
            signal="recording",
            guard_samples=int(config.require("quality.clip_edge_guard_samples")),
        )
    else:
        for extent in extents:
            span_id = store.entity(
                prov_type="span", extent=extent, attributes={"family": CLIP_FAMILY, "signal": "recording"}
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, agent)
    if preprocessed:
        write_verdict(
            store,
            activity,
            agent,
            node="PREPROCESS",
            outcome=Outcome.PASS,
            kind=None,
            why="conditioning complete; absent derivatives are listed",
            detail={"absent": {}, "derivatives": {}},
        )
    store.write_jsonl(run_dir / "store.jsonl")
    return source


def _manifest(path: Path, roots: list[Path]) -> Path:
    """Write a manifest naming one enhanced stream per run root.

    Args:
        path: Where the JSONL goes.
        roots: The run roots, in manifest order.

    Returns:
        The manifest's path.
    """
    lines = [
        json.dumps({"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")}) for root in roots
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _store_of(root: Path) -> ProvStore:
    """Read one run's store back under its own run id.

    Args:
        root: The run root.

    Returns:
        The store.
    """
    return ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)


def _log(tmp_path: Path) -> list[dict[str, Any]]:
    """The slice log the driver wrote.

    Args:
        tmp_path: The directory the manifest sits in.

    Returns:
        One record per manifest row, in order.
    """
    path = tmp_path / "slices" / "quality-slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _contests(store: ProvStore) -> list[str]:
    """Every live assertion QUALITY wrote against a clip span.

    Args:
        store: The store.

    Returns:
        The contesting assertions' ids.
    """
    return [
        entity.id
        for entity in live_entities(store, "assertion")
        if entity.attributes.get("reason") == CONTRADICTED_CLIP
    ]


def _run(manifest: Path) -> int:
    """Drive the whole manifest as one slice.

    Args:
        manifest: The manifest JSONL.

    Returns:
        The driver's exit code.
    """
    return int(cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]))


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[..., tuple[Path, list[Path]]]:
    """A factory for a manifest over N finished runs that never reached QUALITY.

    Returns:
        A callable taking the run count and :func:`_seed_run`'s keyword arguments.
    """

    def _build(count: int, **kwargs: Any) -> tuple[Path, list[Path]]:  # noqa: ANN401 — _seed_run's own keywords
        roots = []
        for index in range(count):
            root = tmp_path / "corpus" / f"sub-{index:02d}_ses-1_20260912-000000"
            root.mkdir(parents=True, exist_ok=True)
            _seed_run(root, **kwargs)
            roots.append(root)
        return _manifest(tmp_path / "manifest.jsonl", roots), roots

    return _build


class TestTheVerdictAFinishedRunGains:
    """The node runs over the store it would have read, and its conclusion lands in that store."""

    def test_the_verdict_and_its_contest_are_written(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """A clip span below an unclipped sample is contested, and the flag reaches the store."""
        manifest, roots = corpus(2)
        assert find_verdict(_store_of(roots[0]), QUALITY) is None

        assert _run(manifest) == 0

        for root in roots:
            store = _store_of(root)
            verdict = find_verdict(store, QUALITY)
            assert verdict is not None
            assert verdict.attributes["outcome"] == Outcome.FLAG.value
            assert CONTRADICTED_CLIP in verdict.attributes["why"]
            assert verdict.attributes["contradicted_n"] == 1
            assert len(_contests(store)) == 1
        assert [record["status"] for record in _log(tmp_path)] == ["ok", "ok"]
        assert [record[QUALITY] for record in _log(tmp_path)] == ["flag", "flag"]

    def test_the_verdict_names_what_had_concluded_when_it_ran(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The branches never ran on this store, and the verdict says which nodes did."""
        manifest, roots = corpus(1)

        _run(manifest)

        verdict = find_verdict(_store_of(roots[0]), QUALITY)
        assert verdict is not None
        assert verdict.attributes["preceded_by"] == ["ADMIT", "PREPROCESS"]
        assert QUALITY not in verdict.attributes["preceded_by"]

    def test_the_clip_span_is_kept_and_the_contest_is_derived_from_it(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The span is PREPROCESS's reading; this pass answers it and never withdraws it."""
        manifest, roots = corpus(1)
        before = [entity.id for entity in live_entities(_store_of(roots[0]), "span")]

        _run(manifest)

        store = _store_of(roots[0])
        assert [entity.id for entity in live_entities(store, "span")] == before
        assert store.derived_from(_contests(store)[0]) == before

    def test_nothing_is_written_outside_the_recordings_own_run(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store gains entities and ``prov/`` is exported; no second tree appears."""
        manifest, roots = corpus(1)
        _run(manifest)
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]
        assert sorted(entry.name for entry in (roots[0] / "run").iterdir()) == ["store.jsonl", "streams"]

    def test_the_bep028_files_are_re_exported_from_the_merged_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        _run(manifest)

        io_graph = json.loads((roots[0] / "prov" / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        verdict = find_verdict(_store_of(roots[0]), QUALITY)
        assert verdict is not None
        assert any(verdict.id in identifier for identifier in entity_ids), verdict.id

    def test_the_host_environment_is_recorded(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """The pass ran somewhere, and the store says where; no venv is involved in this one."""
        manifest, roots = corpus(1)
        _run(manifest)
        assert [env.kind for env in _store_of(roots[0]).environments()] == ["host"]


class TestASecondPass:
    """A task that dies and reruns must add nothing, which is what makes the array restartable."""

    def test_a_rerun_over_the_same_corpus_changes_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The fingerprint is where the first pass left it, so the stores are byte-identical."""
        manifest, roots = corpus(2)
        _run(manifest)
        after_first = {root: (root / "run" / "store.jsonl").read_bytes() for root in roots}
        fingerprints = {root: _store_of(root).fingerprint() for root in roots}

        assert _run(manifest) == 0

        for root in roots:
            assert (root / "run" / "store.jsonl").read_bytes() == after_first[root]
            assert _store_of(root).fingerprint() == fingerprints[root]
            assert len([e for e in live_entities(_store_of(root), "verdict") if e.attributes["node"] == QUALITY]) == 1
        assert [record["status"] for record in _log(tmp_path)] == ["skipped", "skipped"]
        assert [record[QUALITY] for record in _log(tmp_path)] == ["present", "present"]


class TestARunWithNothingToContradict:
    """No clip span is not a refusal: nothing was asserted, so nothing can contradict it."""

    def test_a_run_with_no_clip_span_passes_rather_than_refusing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The verdict is the one a fresh run would have reached, and nothing is contested."""
        manifest, roots = corpus(1, spans=False)

        assert _run(manifest) == 0

        store = _store_of(roots[0])
        verdict = find_verdict(store, QUALITY)
        assert verdict is not None
        assert verdict.attributes["outcome"] == Outcome.PASS.value
        assert "no clip span" in verdict.attributes["why"]
        assert verdict.attributes["clip_spans_n"] == 0
        assert _contests(store) == []
        assert _log(tmp_path)[0]["status"] == "ok"
        assert _log(tmp_path)[0][QUALITY] == "pass"

    def test_a_clip_nothing_is_louder_than_passes(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The loudest sample is inside the span, which is what a clip means."""
        manifest, roots = corpus(1, clip_level=0.98, louder=None)

        assert _run(manifest) == 0

        store = _store_of(roots[0])
        verdict = find_verdict(store, QUALITY)
        assert verdict is not None
        assert verdict.attributes["outcome"] == Outcome.PASS.value
        assert verdict.attributes["checked_n"] == 1
        assert _contests(store) == []


class TestARunQualityCannotRead:
    """A missing input is an operational fact about the run, and never a quiet pass."""

    def test_clip_spans_with_no_amplitudes_are_refused_and_the_store_is_left_alone(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """This is the corpus before the clip pass; a pass verdict there would read as consistent."""
        manifest, roots = corpus(1, amplitudes=False)
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert _run(manifest) == 1

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert find_verdict(_store_of(roots[0]), QUALITY) is None
        record = _log(tmp_path)[0]
        assert record["status"] == "error"
        assert "clip_amplitude" in str(record[QUALITY])

    def test_a_run_with_no_store_is_recorded_and_the_others_still_reach_a_verdict(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert _run(manifest) == 1

        assert find_verdict(_store_of(roots[1]), QUALITY) is not None
        assert [record["status"] for record in _log(tmp_path)] == ["error", "ok"]

    def test_a_manifest_row_naming_no_run_root_is_refused_rather_than_guessed(self) -> None:
        """A wrong guess would write a verdict into the wrong recording's store."""
        with pytest.raises(ValueError, match="no run root"):
            cli.run_root_of(Path("/corpus/enhanced.flac"))
