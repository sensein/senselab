"""The extend-in-place driver: what it adds to a run's store, and what it leaves alone.

Nothing here calls ppgs. Its venv takes 810 s to build and its model is the whole cost of the run,
so ``extract_ppgs_from_audios`` is monkeypatched inside the driver module and the audio is
synthetic. Praat is real: it is fast, it has no venv, and what it returns over a synthetic tone is
still the dict a consumer reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pytest
import soundfile as sf
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import PHONEME_LABELS, PPGS_SAMPLE_RATE, PpgsPosteriorgramUnavailable
from senselab.audio.workflows.triage.consensus import ConsensusWord, word_attributes
from senselab.audio.workflows.triage.extend import CONSENSUS_TRANSCRIPT
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_measurements,
    lexical_words,
    path_attributes,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.preprocess import PPG_MEASUREMENT, PRAAT_MEASUREMENT
from senselab.utils import subprocess_venv
from senselab.utils.data_structures import DeviceType
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_ppg_praat.py"
_spec = importlib.util.spec_from_file_location("extend_ppg_praat_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_ppg_praat_under_test"] = cli
_spec.loader.exec_module(cli)

_FRAMES_PER_SECOND = 116
"""The posteriorgram frame rate the fake reproduces, so the shapes the tests read are the real ones."""


def _fake_ppgs(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
    """One deterministic posteriorgram per audio, in the library's ``(1, phonemes, frames)`` layout.

    Args:
        audios: The batch.
        device: Ignored.

    Returns:
        One tensor per audio.
    """
    out: List[Any] = []
    for index, audio in enumerate(audios):
        seconds = audio.waveform.shape[-1] / audio.sampling_rate
        frames = max(1, int(seconds * _FRAMES_PER_SECOND))
        generator = torch.Generator().manual_seed(index)
        out.append(torch.rand(1, len(PHONEME_LABELS), frames, generator=generator))
    return out


_HEARD = ("alpha", "beta")
"""The consensus a seeded run carries unless a test asks for another: two plain words."""


def _write_consensus(store: ProvStore, tokens: Sequence[str], *, seconds: float) -> None:
    """Add one ``consensus_transcript`` and its word stream, in the shape PREPROCESS writes them.

    A token wrapped in brackets is stored bracketed, which is how the corpus's ``[breath]``-only
    recordings read; ``tokens`` empty is the recognizers returning nothing, which is a transcript
    carrying no words rather than no transcript.

    Args:
        store: The run's store.
        tokens: The consensus surfaces, in stream order.
        seconds: The stream's duration, divided evenly among the words.
    """
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="consensus", parameters={"sources": ["a", "b"]})
    store.was_associated_with(activity, agent)
    step = seconds / max(len(tokens), 1)
    word_ids: list[str] = []
    for index, token in enumerate(tokens):
        extent = (index * step, (index + 1) * step)
        word = ConsensusWord(
            index=index,
            text=token,
            bracketed=token.startswith("[") and token.endswith("]"),
            outcome="agreement",
            sources=("a", "b"),
            readings={"a": token, "b": token},
            timings={"a": extent, "b": extent},
            extent=extent,
            onset_spread_s=0.0,
            offset_spread_s=0.0,
            temporal_uncertainty_s=0.0,
            variants=(),
            agreement=1.0,
        )
        word_id = store.entity(prov_type="word", extent=extent, attributes=word_attributes(word))
        store.was_generated_by(word_id, activity)
        store.was_attributed_to(word_id, agent)
        word_ids.append(word_id)
    write_measurement(
        store,
        activity,
        agent,
        name=CONSENSUS_TRANSCRIPT,
        signal="plain",
        attributes={
            "role": "consensus",
            "n_sources": 2,
            "n_words": len(tokens),
            "word_ids": word_ids,
            "text": " ".join(tokens),
        },
    )


def _seed_run(
    root: Path,
    *,
    seconds: float = 0.5,
    hz: float = 120.0,
    transcript: Optional[Sequence[str]] = _HEARD,
) -> Path:
    """Write one finished run: an ``enhanced`` stream on disk and a store that names it.

    Args:
        root: The run root, created if absent.
        seconds: The stream's duration.
        hz: Its fundamental, so Praat finds a pitch.
        transcript: The consensus surfaces this run heard; ``()`` for a consensus that heard
            nothing, and None for a run whose consensus block never wrote a transcript at all.

    Returns:
        The enhanced stream's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    enhanced = run_dir / "streams" / "enhanced.flac"
    t = np.arange(int(seconds * PPGS_SAMPLE_RATE)) / PPGS_SAMPLE_RATE
    wave = 0.4 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * 2 * hz * t)
    sf.write(str(enhanced), wave.astype(np.float32), PPGS_SAMPLE_RATE)

    store = ProvStore(run_id=root.name)
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="residual", parameters={})
    store.was_associated_with(activity, agent)
    entity = store.entity(
        prov_type="stream",
        extent=(0.0, seconds),
        attributes={
            "name": "enhanced",
            **path_attributes("streams/enhanced.flac", run_dir),
            "sampling_rate": PPGS_SAMPLE_RATE,
            "channels": 1,
        },
    )
    store.was_generated_by(entity, activity)
    store.was_attributed_to(entity, agent)
    if transcript is not None:
        _write_consensus(store, transcript, seconds=seconds)
    store.write_jsonl(run_dir / "store.jsonl")
    return enhanced


def _manifest(path: Path, roots: Sequence[Path]) -> Path:
    """Write a manifest naming one enhanced stream per run root.

    Args:
        path: Where the JSONL goes.
        roots: The run roots, in manifest order.

    Returns:
        The manifest's path.
    """
    lines = [
        json.dumps(
            {
                "stem": root.name,
                "enhanced": str(root / "run" / "streams" / "enhanced.flac"),
                "family": "voice",
                "duration_s": 0.5,
                "lexical": False,
            }
        )
        for root in roots
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _store_of(root: Path) -> ProvStore:
    """Read one run's store back under its own run id."""
    return ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)


def _log_rows(log_dir: Path) -> list[dict[str, Any]]:
    """The per-recording outcome records the slice wrote, in manifest order."""
    path = log_dir / "slices" / "slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _summary(log_dir: Path) -> dict[str, Any]:
    """The summary sidecar the slice wrote beside its log."""
    return json.loads((log_dir / "slices" / "slice-0-of-1.summary.json").read_text())


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[..., tuple[Path, list[Path]]]:
    """A factory for a manifest over N finished runs, each carrying the consensus it is given.

    ``transcripts`` names one consensus per run, in manifest order; without it every run carries
    :data:`_HEARD`, which is what puts it in the posteriorgram's scope.
    """

    def _build(count: int, transcripts: Optional[Sequence[Optional[Sequence[str]]]] = None) -> tuple[Path, list[Path]]:
        if transcripts is not None and len(transcripts) != count:
            raise ValueError(f"{len(transcripts)} transcripts for {count} runs")
        roots = []
        for index in range(count):
            root = tmp_path / "corpus" / f"sub-{index:02d}_ses-1_20260911-000000"
            _seed_run(
                root,
                hz=110.0 + 10.0 * index,
                transcript=_HEARD if transcripts is None else transcripts[index],
            )
            roots.append(root)
        return _manifest(tmp_path / "manifest.jsonl", roots), roots

    return _build


@pytest.fixture
def provisioned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the ppgs venv as built, so ``main`` does not refuse before measuring anything."""
    monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: True)


@pytest.fixture
def stub_ppgs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the model call in the driver module."""
    monkeypatch.setattr(cli, "extract_ppgs_from_audios", _fake_ppgs)


class TestTheRunRoot:
    """The manifest carries no run root, so the driver derives one from the enhanced path."""

    def test_the_run_root_is_the_enhanced_streams_third_parent(self) -> None:
        """``<root>/run/streams/enhanced.flac`` resolves to ``<root>``."""
        assert cli.run_root_of(Path("/corpus/sub-01_x/run/streams/enhanced.flac")) == Path("/corpus/sub-01_x")

    def test_a_path_of_another_shape_is_refused_rather_than_guessed(self) -> None:
        """A stream somewhere else names no run, and a wrong guess would extend the wrong store."""
        with pytest.raises(ValueError, match="no run root"):
            cli.run_root_of(Path("/corpus/enhanced.flac"))


class TestTheSlicing:
    """Task *i* of *n* takes a stride, so no task draws the corpus's long tail on its own."""

    def test_each_task_takes_every_nth_row(self) -> None:
        """The three shards of nine rows partition them, each taking every third."""
        rows = [{"stem": str(index)} for index in range(9)]
        shards = [cli.take_slice(rows, index, 3) for index in range(3)]
        assert [row["stem"] for row in shards[0]] == ["0", "3", "6"]
        assert [row["stem"] for row in shards[1]] == ["1", "4", "7"]
        assert sorted(row["stem"] for shard in shards for row in shard) == sorted(row["stem"] for row in rows)

    def test_an_index_outside_the_count_is_refused(self) -> None:
        """A shard nobody owns would silently drop its rows."""
        with pytest.raises(ValueError, match="slice-index"):
            cli.take_slice([{"stem": "0"}], 1, 1)

    def test_the_final_batch_is_as_short_as_it_needs_to_be(self) -> None:
        """Seven rows at batch three is 3 + 3 + 1, not 3 + 3 + 3."""
        rows = [{"stem": str(index)} for index in range(7)]
        assert [len(batch) for batch in cli.batches(rows, 3)] == [3, 3, 1]


class TestTheExtendPass:
    """What one pass adds to a finished run, and what it leaves untouched."""

    def test_both_measurements_are_merged_into_the_existing_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """The store keeps every record it had and gains the two the blocks wrote."""
        manifest, roots = corpus(2)
        before = {root: _store_of(root).fingerprint() for root in roots}

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        for root in roots:
            store = _store_of(root)
            assert store.fingerprint() != before[root]
            ppg = find_measurement(store, PPG_MEASUREMENT)
            praat = find_measurement(store, PRAAT_MEASUREMENT)
            assert ppg is not None and praat is not None
            assert (root / "run" / ppg.attributes["path"]).is_file()
            assert len(ppg.attributes["checksum_sha256"]) == 64
            assert praat.attributes["features"]

    def test_nothing_is_written_outside_the_recordings_own_run(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """The rejected shape wrote a second tree; this one only touches ``run/`` and ``prov/``."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]

    def test_the_bep028_files_are_re_exported_from_the_merged_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])

        prov = roots[0] / "prov"
        names = sorted(path.name for path in prov.iterdir())
        assert "prov-triage_io.json" in names
        assert "prov-triage_act.json" in names
        io_graph = json.loads((prov / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        store = _store_of(roots[0])
        for measurement in (find_measurement(store, PPG_MEASUREMENT), find_measurement(store, PRAAT_MEASUREMENT)):
            assert measurement is not None
            assert any(measurement.id in identifier for identifier in entity_ids), measurement.id
        assert json.loads((prov / "prov-triage_io.json").read_text())["@context"]

    def test_the_ppgs_venv_gets_its_own_environment_record(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A different interpreter is a different environment; the store names it beside the host's."""
        venv_dir = tmp_path / "venvs" / "ppgs-cpu"
        (venv_dir / "lib" / "python3.11" / "site-packages" / "torch-2.8.0.dist-info").mkdir(parents=True)
        (venv_dir / "pyvenv.cfg").write_text("version = 3.11.9\n")
        monkeypatch.setattr(subprocess_venv, "_ensure_venv_once", lambda *args, **kwargs: venv_dir)

        def _through_the_venv(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
            """Resolve the venv the way the real call does, then return the fake's tensors."""
            subprocess_venv.ensure_venv("ppgs", ["ppgs"], python_version="3.11")
            return _fake_ppgs(audios, device)

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _through_the_venv)

        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        environments = _store_of(roots[0]).environments()
        assert [env.label for env in environments if env.kind == "host"]
        venvs = [env for env in environments if env.kind == "venv"]
        assert [env.label for env in venvs] == ["ppgs-cpu"]
        assert venvs[0].python_version == "3.11.9"
        assert venvs[0].dependencies["torch"] == "2.8.0"

    def test_a_rerun_converges_on_the_same_graph(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """Merging is a set union, so a task that dies and reruns must add nothing the second time.

        The second pass skips the recording outright, which is what makes it converge: rewriting the
        npz would give it a new mtime and so a second, differently-identified entity.
        """
        manifest, roots = corpus(2)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        after_first = {root: _store_of(root).fingerprint() for root in roots}

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        for root in roots:
            store = _store_of(root)
            assert store.fingerprint() == after_first[root]
            assert len([e for e in store.entities("measurement") if e.attributes["name"] == PPG_MEASUREMENT]) == 1

    def test_a_half_extended_run_gains_only_what_it_is_missing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """Praat alone is redone when the posteriorgram already landed, and the entity is not doubled."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        store = _store_of(roots[0])
        ppg = find_measurement(store, PPG_MEASUREMENT)
        assert ppg is not None
        kept = ppg.id

        lines = [
            line
            for line in (roots[0] / "run" / "store.jsonl").read_text().splitlines()
            if json.loads(line).get("attributes", {}).get("name") != PRAAT_MEASUREMENT
        ]
        (roots[0] / "run" / "store.jsonl").write_text("\n".join(lines) + "\n")

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        store = _store_of(roots[0])
        assert find_measurement(store, PRAAT_MEASUREMENT) is not None
        assert [e.id for e in store.entities("measurement") if e.attributes["name"] == PPG_MEASUREMENT] == [kept]


class TestOneBadRecording:
    """A recording the model or the filesystem refuses is an outcome, not the end of the batch."""

    def test_a_typed_absence_leaves_the_rest_of_the_batch_intact(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """One unavailable posteriorgram costs that recording its npz, and nothing else."""

        def _one_absent(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
            out = _fake_ppgs(audios, device)
            out[0] = PpgsPosteriorgramUnavailable("ppgs produced no posteriorgram: RuntimeError: shapes")
            return out

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _one_absent)
        manifest, roots = corpus(3)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert find_measurement(_store_of(roots[0]), PPG_MEASUREMENT) is None
        assert find_measurement(_store_of(roots[0]), PRAAT_MEASUREMENT) is not None
        for root in roots[1:]:
            assert find_measurement(_store_of(root), PPG_MEASUREMENT) is not None

        log = [json.loads(line) for line in (tmp_path / "slices" / "slice-0-of-1.jsonl").read_text().splitlines()]
        assert log[0]["status"] == "absent"
        assert "PpgsPosteriorgramUnavailable" in log[0]["ppg"]
        assert [record["status"] for record in log[1:]] == ["ok", "ok"]

    def test_a_run_with_no_store_is_recorded_and_the_others_still_extend(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert find_measurement(_store_of(roots[1]), PPG_MEASUREMENT) is not None
        log = [json.loads(line) for line in (tmp_path / "slices" / "slice-0-of-1.jsonl").read_text().splitlines()]
        assert log[0]["status"] == "error"
        assert log[1]["status"] == "ok"

    def test_a_whole_batch_failure_keeps_every_recordings_praat_features(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A dead subprocess is one message on every row in the batch, not a lost slice."""

        def _die(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
            raise RuntimeError("worker timed out")

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _die)
        manifest, roots = corpus(2)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        for root in roots:
            store = _store_of(root)
            assert find_measurement(store, PPG_MEASUREMENT) is None
            assert find_measurement(store, PRAAT_MEASUREMENT) is not None
        log = [json.loads(line) for line in (tmp_path / "slices" / "slice-0-of-1.jsonl").read_text().splitlines()]
        assert all("worker timed out" in record["ppg"] for record in log)


class TestTheBatching:
    """One ppgs call per batch: the whole point of the driver's shape."""

    def test_the_batch_size_bounds_the_calls_not_the_recordings(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Five recordings at batch two is three calls of 2, 2 and 1 — not five calls of one."""
        sizes: list[int] = []

        def _counting(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
            sizes.append(len(audios))
            return _fake_ppgs(audios, device)

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _counting)
        manifest, _ = corpus(5)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--batch-size", "2"])
        assert sizes == [2, 2, 1]


class TestTheVenvGate:
    """A cold build outlasts the lock, so the driver refuses rather than starting one."""

    def test_a_missing_venv_refuses_and_names_the_pre_build(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exit 2, nothing measured, and the message names ``ensure_ppgs_venv``."""
        monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: False)
        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 2
        assert "ensure_ppgs_venv" in capsys.readouterr().err
        assert find_measurement(_store_of(roots[0]), PPG_MEASUREMENT) is None


class TestTheTranscriptScope:
    """The posteriorgram is derived where the recognizers produced a consensus, and nowhere else.

    The ``ppg_20260911`` pass applied this from an operator's manifest that cannot be rebuilt from
    the tree; it is the driver's rule now, so a manifest naming every recording yields the scope.
    """

    def test_a_consensus_that_heard_nothing_takes_praat_and_no_posteriorgram(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """A transcript carrying no word is the corpus's 2,376: Praat lands, ppgs is never asked."""
        manifest, roots = corpus(1, [()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        store = _store_of(roots[0])
        assert find_measurement(store, CONSENSUS_TRANSCRIPT) is not None
        assert find_measurement(store, PPG_MEASUREMENT) is None, "a wordless recording ran a phoneme classifier"
        assert find_measurement(store, PRAAT_MEASUREMENT) is not None, "Praat needs no transcript"
        assert _log_rows(tmp_path)[0]["status"] == "out-of-scope"

    def test_a_store_with_no_transcript_entity_is_out_of_scope_too(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """Two of the eighty sampled hold no ``consensus_transcript`` at all; absence is not consent."""
        manifest, roots = corpus(1, [None])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        store = _store_of(roots[0])
        assert find_measurement(store, CONSENSUS_TRANSCRIPT) is None
        assert find_measurement(store, PPG_MEASUREMENT) is None
        assert find_measurement(store, PRAAT_MEASUREMENT) is not None

    def test_a_transcript_of_only_bracketed_tokens_is_in_scope(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """The rule keys on the consensus stream, not on its unbracketed subset.

        14,836 of the corpus are ``[breath]``/``[cough]`` throughout, so ``lexical_words`` as the
        predicate would drop every one of them.
        """
        manifest, roots = corpus(1, [("[breath]", "[cough]")])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        store = _store_of(roots[0])
        assert lexical_words(store) == [], "this recording must carry no unbracketed word"
        assert find_measurement(store, PPG_MEASUREMENT) is not None, "the bracketed-only population is in scope"

    def test_only_the_recordings_in_scope_reach_the_model(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One ppgs call per batch, and the two out-of-scope rows are not in it."""
        sizes: list[int] = []

        def _counting(audios: List[Audio], device: Optional[DeviceType] = None) -> List[Any]:
            sizes.append(len(audios))
            return _fake_ppgs(audios, device)

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _counting)
        manifest, roots = corpus(4, [_HEARD, (), None, ("[breath]",)])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert sizes == [2], "the wordless and the transcript-less rows were handed to the model"
        held = [find_measurement(_store_of(root), PPG_MEASUREMENT) is not None for root in roots]
        assert held == [True, False, False, True]
        assert all(find_measurement(_store_of(root), PRAAT_MEASUREMENT) is not None for root in roots)

    def test_force_does_not_widen_the_scope(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """--force is one-sided: it overrides ``praat_pending`` alone, never the transcript rule."""
        manifest, roots = corpus(1, [()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 0
        row = _log_rows(tmp_path)[0]
        assert row["praat"] == "ok", "--force must still re-derive Praat on an out-of-scope recording"
        assert find_measurement(_store_of(roots[0]), PPG_MEASUREMENT) is None, "--force widened the PPG scope"

    def test_the_venv_gate_holds_even_where_nothing_is_in_scope(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The gate is a property of the host, not of the slice.

        A scope-conditional gate would let a later slice holding in-scope rows start a cold build
        under the array, which is what the unconditional one exists to prevent.
        """
        monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: False)
        manifest, roots = corpus(2, [(), None])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 2
        assert "ensure_ppgs_venv" in capsys.readouterr().err
        assert find_measurement(_store_of(roots[0]), PRAAT_MEASUREMENT) is None, "nothing may be measured"


class TestTheLogSaysWhichSkipIsWhich:
    """A posteriorgram the rule kept off the model and one the store already holds are two facts.

    Both used to read ``skipped``. The catch-up pass over the 2,376 stores that hold neither
    measurement turns on the difference: a regression that ran or failed ppgs on them would
    otherwise be invisible in the log it writes.
    """

    def test_an_out_of_scope_recording_reads_out_of_scope_and_not_skipped(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """The row the catch-up pass writes: Praat landed, and the rule is why ppgs did not run."""
        manifest, _ = corpus(1, [()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        row = _log_rows(tmp_path)[0]
        assert row["ppg"] == "out-of-scope"
        assert row["praat"] == "ok"
        assert row["status"] == "out-of-scope"

    def test_a_store_that_already_holds_its_posteriorgram_still_reads_skipped(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """An in-scope recording extended twice: the second pass has nothing to do, and says so."""
        manifest, _ = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        row = _log_rows(tmp_path)[0]
        assert row["ppg"] == "skipped"
        assert row["praat"] == "skipped"
        assert row["status"] == "skipped"

    def test_a_second_pass_over_an_out_of_scope_recording_still_reads_out_of_scope(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """The re-run takes the whole-recording skip, where the two facts are just as distinct."""
        manifest, _ = corpus(1, [()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        row = _log_rows(tmp_path)[0]
        assert row["ppg"] == "out-of-scope"
        assert row["praat"] == "skipped"
        assert row["status"] == "out-of-scope"

    def test_the_summary_counts_the_two_populations_apart(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """One in-scope row and one out of it, counted as what each is rather than both as ``ok``."""
        manifest, _ = corpus(2, [_HEARD, ()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert _summary(tmp_path)["counts"] == {"ok": 1, "out-of-scope": 1}

    def test_an_out_of_scope_recording_whose_praat_fails_is_not_out_of_scope(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The new value says the rule fired, not that the row is done: a failed block still shows."""

        def _refuse(store: ProvStore, config: object, *, run_dir: Path) -> str:
            raise RuntimeError("praat refused this recording")

        monkeypatch.setattr(cli, "praat_features", _refuse)
        manifest, _ = corpus(1, [()])
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        row = _log_rows(tmp_path)[0]
        assert row["ppg"] == "out-of-scope"
        assert "praat refused this recording" in row["praat"]
        assert row["status"] == "absent"


class TestForceReDerives:
    """Every stored Praat scalar was measured under the retired sex-typed F0 bin, so skipping looks like success."""

    def test_without_force_the_praat_block_is_skipped(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """The default is the restart-safe skip, which is what makes a corpus re-derivation a no-op."""
        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        before = _store_of(roots[0]).fingerprint()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        assert _store_of(roots[0]).fingerprint() == before
        assert _log_rows(tmp_path)[0]["praat"] == "skipped"

    def test_force_re_derives_a_store_that_already_holds_both(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """--force must run the Praat block on a store that holds it, not report a skip as success."""
        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 0
        row = _log_rows(tmp_path)[0]
        assert row["praat"] == "ok", "--force must re-derive, not skip"
        assert row["ppg"] == "skipped", "--force is Praat-only; the posteriorgram block must not run"

    def test_force_supersedes_a_reading_taken_under_a_different_range_rule(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        provisioned: None,
        stub_ppgs: None,
        tmp_path: Path,
    ) -> None:
        """The corpus was measured under coefficients the tree no longer holds; the stale reading must retire.

        The store is append-only, so without the supersession the forced pass leaves two live
        ``praat_features`` measurements with nothing saying which is current.
        """
        manifest, roots = corpus(1)
        stale = tmp_path / "stale.yaml"
        stale.write_text("praat_features:\n  pitch_ceiling_quartile_multiplier: 2.0\n", encoding="utf-8")
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--config", str(stale)]) == 0
        retired = find_measurement(_store_of(roots[0]), PRAAT_MEASUREMENT)
        assert retired is not None

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 0
        store = _store_of(roots[0])
        live = find_measurements(store, PRAAT_MEASUREMENT)
        assert [entity.id for entity in live] != [retired.id], "the stale reading is still the live one"
        assert len(live) == 1, f"{len(live)} live Praat measurements after --force"
        assert store.is_invalidated(retired.id)
        assert live[0].attributes["pitch_ceiling_quartile_multiplier"] == 2.5
        assert live[0].attributes["signal"] == "enhanced"

    def test_an_unchanged_re_derivation_retires_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """A forced pass that reproduces the stored reading has nothing to retire, and must not retire it."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"])

        store = _store_of(roots[0])
        assert len(find_measurements(store, PRAAT_MEASUREMENT)) == 1
        assert find_measurement(store, PRAAT_MEASUREMENT) is not None

    def test_force_does_not_touch_the_posteriorgram(
        self, corpus: Callable[..., tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """Audit steps 1 and 1b are both withdrawn, so a forced pass is Praat-only, on ``enhanced``."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        before = find_measurement(_store_of(roots[0]), PPG_MEASUREMENT)
        assert before is not None

        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"])
        after = find_measurement(_store_of(roots[0]), PPG_MEASUREMENT)
        assert after is not None
        assert after.id == before.id
        assert after.attributes["signal"] == "enhanced"

    def test_the_venv_gate_holds_on_a_forced_pass(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--force overrides ``praat_pending`` alone, so a store missing its posteriorgram still needs ppgs."""
        monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: False)
        manifest, _ = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 2
        assert "ensure_ppgs_venv" in capsys.readouterr().err
