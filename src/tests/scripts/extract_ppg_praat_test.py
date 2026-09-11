"""The PPG/Praat batch driver: its BIDS-shaped output, its resume, its batching and its outcomes.

Nothing here calls ppgs. Its venv takes 810 s to build and its model is the whole cost of the run,
so ``extract_ppgs_from_audios`` is monkeypatched inside the driver module and the audio is
synthetic. Praat is real: it is fast, it has no venv, and what it returns over a synthetic tone is
still a dict of the keys a consumer reads.
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
from senselab.audio.tasks.features_extraction import PHONEME_LABELS, PPGS_SAMPLE_RATE
from senselab.utils.data_structures import DeviceType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extract_ppg_praat.py"
_spec = importlib.util.spec_from_file_location("extract_ppg_praat_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extract_ppg_praat_under_test"] = cli
_spec.loader.exec_module(cli)

_FRAMES_PER_SECOND = 116
"""The posteriorgram frame rate the fake reproduces, so the shapes the tests read are the real ones."""


def _tone(path: Path, *, seconds: float = 0.5, hz: float = 120.0) -> None:
    """Write a mono 16 kHz tone Praat can find a pitch in.

    Args:
        path: Where the flac goes.
        seconds: Its duration.
        hz: Its fundamental.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * PPGS_SAMPLE_RATE)) / PPGS_SAMPLE_RATE
    wave = 0.4 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * 2 * hz * t)
    sf.write(str(path), wave.astype(np.float32), PPGS_SAMPLE_RATE)


def _fake_ppgs(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
    """One deterministic posteriorgram per audio, in the library's ``(1, phonemes, frames)`` layout.

    Args:
        audios: The batch.
        device: Ignored.

    Returns:
        One tensor per audio.
    """
    out = []
    for audio in audios:
        frames = max(1, int(audio.waveform.shape[1] / audio.sampling_rate * _FRAMES_PER_SECOND))
        out.append(torch.rand(1, len(PHONEME_LABELS), frames))
    return out


@pytest.fixture(autouse=True)
def ppgs_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model never runs here: its venv build is 810 s and its weights are the run's whole cost."""
    monkeypatch.setattr(cli, "extract_ppgs_from_audios", _fake_ppgs)
    monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: True)


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[[Sequence[str]], Path]:
    """A manifest over freshly written tones, one row per stem.

    Args:
        tmp_path: Where the audio and the manifest go.

    Returns:
        A factory taking stems and returning the manifest path.
    """

    def _build(stems: Sequence[str]) -> Path:
        manifest = tmp_path / "manifest.jsonl"
        lines = []
        for stem in stems:
            enhanced = tmp_path / "corpus" / stem / "run" / "streams" / "enhanced.flac"
            _tone(enhanced)
            lines.append(
                json.dumps(
                    {
                        "stem": stem,
                        "enhanced": str(enhanced),
                        "family": "speech",
                        "duration_s": 0.5,
                        "lexical": True,
                    }
                )
            )
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    return _build


def _run(manifest: Path, out: Path, **kwargs: object) -> dict[str, Any]:
    """Run one slice with the test defaults.

    Args:
        manifest: The manifest.
        out: The output root.
        **kwargs: Overrides for ``run_slice``'s keyword arguments.

    Returns:
        The slice summary.
    """
    options: dict[str, object] = {"slice_index": 0, "slice_count": 1, "batch_size": 500, "device": None}
    options.update(kwargs)
    return cli.run_slice(manifest, out, **options)


class TestTheOutputMirrorsTheBidsTree:
    """125,263 entries in one directory exceeded ARG_MAX; the entity path is not optional."""

    def test_a_stem_with_both_entities_lands_under_both(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """Each of sub and ses becomes a directory, in BIDS order, from the stem alone."""
        stem = "sub-01_ses-02_task-vowel"
        out = tmp_path / "out"
        _run(corpus([stem]), out)
        assert (out / "sub-01" / "ses-02" / f"{stem}_ppg.npz").is_file()
        assert (out / "sub-01" / "ses-02" / f"{stem}_features.json").is_file()

    def test_a_stem_with_neither_entity_lands_at_the_root(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """A stem carrying no entity is not invented one, and is not dropped either."""
        out = tmp_path / "out"
        _run(corpus(["recording"]), out)
        assert (out / "recording_features.json").is_file()


class TestThePhonemeOrderTravelsWithTheData:
    """A consumer must never have to guess which column is which phoneme."""

    def test_the_labels_are_stored_beside_the_posteriorgram(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """The npz carries the inventory in the axis order the array uses."""
        out = tmp_path / "out"
        _run(corpus(["sub-01_ses-01_task-a"]), out)
        npz = np.load(out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_ppg.npz")
        assert list(npz["phonemes"]) == list(PHONEME_LABELS)
        assert npz["posteriorgram"].shape[1] == len(PHONEME_LABELS)

    def test_the_posteriorgram_is_frame_major_float16(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """float16 is what makes the whole corpus 3.3 GB rather than 6.6."""
        out = tmp_path / "out"
        _run(corpus(["sub-01_ses-01_task-a"]), out)
        npz = np.load(out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_ppg.npz")
        assert npz["posteriorgram"].dtype == np.float16
        assert npz["posteriorgram"].shape[0] > npz["posteriorgram"].shape[1]
        assert float(npz["seconds_per_frame"]) == pytest.approx(1.0 / _FRAMES_PER_SECOND, rel=0.05)


class TestResume:
    """An array task that dies restarts without redoing work."""

    def test_a_completed_recording_is_skipped(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second run reaches ppgs with nothing, and the row it already wrote is untouched."""
        out = tmp_path / "out"
        manifest = corpus(["sub-01_ses-01_task-a", "sub-02_ses-01_task-a"])
        _run(manifest, out)
        row = out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json"
        stamped = row.stat().st_mtime_ns

        def _refuse(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
            raise AssertionError("ppgs was called for a recording that was already done")

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _refuse)
        summary = _run(manifest, out)
        assert summary["counts"] == {"skipped": 2}
        assert row.stat().st_mtime_ns == stamped

    def test_a_half_written_recording_is_redone(self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path) -> None:
        """The row is written last, so a posteriorgram without one is not a completed recording."""
        out = tmp_path / "out"
        manifest = corpus(["sub-01_ses-01_task-a"])
        _run(manifest, out)
        (out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").unlink()
        assert _run(manifest, out)["counts"] == {"ok": 1}


class TestBatching:
    """Batch size is the whole lever, so the ragged tail and the batch boundary both matter."""

    def test_the_final_batch_is_ragged_and_still_written(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seven recordings at a batch of three is 3 + 3 + 1, and the one is not lost."""
        sizes: list[int] = []

        def _counting(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
            sizes.append(len(audios))
            return _fake_ppgs(audios, device)

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _counting)
        out = tmp_path / "out"
        stems = [f"sub-{i:02d}_ses-01_task-a" for i in range(7)]
        summary = _run(corpus(stems), out, batch_size=3)
        assert sizes == [3, 3, 1]
        assert summary["counts"] == {"ok": 7}

    def test_the_slice_is_a_stride_of_the_manifest(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """Task i of n takes rows[i::n], so the shards partition the manifest exactly once."""
        stems = [f"sub-{i:02d}_ses-01_task-a" for i in range(5)]
        manifest = corpus(stems)
        first = _run(manifest, tmp_path / "a", slice_index=0, slice_count=2)
        second = _run(manifest, tmp_path / "b", slice_index=1, slice_count=2)
        assert (first["rows"], second["rows"]) == (3, 2)

    def test_a_slice_index_outside_the_count_is_refused(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """An off-by-one in the array bound is an error, not a silently empty shard."""
        with pytest.raises(ValueError, match="--slice-index"):
            _run(corpus(["sub-01_ses-01_task-a"]), tmp_path / "out", slice_index=2, slice_count=2)


class TestAFailureIsRecordedNotRaised:
    """One bad recording must not lose the other 499."""

    def test_a_nan_posteriorgram_is_an_outcome(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scalar NaN comes back for a recording the model raised on; that is a row, not a stop."""

        def _one_nan(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
            out = _fake_ppgs(audios, device)
            out[0] = torch.tensor(float("nan"))
            return out

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _one_nan)
        out = tmp_path / "out"
        summary = _run(corpus(["sub-01_ses-01_task-a", "sub-02_ses-01_task-a"]), out, batch_size=2)
        assert summary["counts"] == {"partial": 1, "ok": 1}
        row = json.loads((out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").read_text())
        assert row["ppg"]["status"] == "nan"
        assert row["praat"]["status"] == "ok"
        assert not (out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_ppg.npz").exists()
        assert (out / "sub-02" / "ses-01" / "sub-02_ses-01_task-a_ppg.npz").is_file()

    def test_an_unreadable_recording_does_not_lose_the_batch(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """A row pointing at a file that is not there is one error row, and the rest still land."""
        manifest = corpus(["sub-01_ses-01_task-a", "sub-02_ses-01_task-a"])
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["enhanced"] = str(tmp_path / "gone.flac")
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        out = tmp_path / "out"
        summary = _run(manifest, out, batch_size=2)
        assert summary["counts"] == {"error": 1, "ok": 1}
        row = json.loads((out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").read_text())
        assert "FileNotFoundError" in row["ppg"]["error"]

    def test_a_whole_batch_failure_is_recorded_per_recording(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A subprocess that dies costs the batch's posteriorgrams, not its Praat features."""

        def _die(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
            raise RuntimeError("ppgs worker exited 137")

        monkeypatch.setattr(cli, "extract_ppgs_from_audios", _die)
        out = tmp_path / "out"
        summary = _run(corpus(["sub-01_ses-01_task-a", "sub-02_ses-01_task-a"]), out, batch_size=2)
        assert summary["counts"] == {"partial": 2}
        row = json.loads((out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").read_text())
        assert "ppgs worker exited 137" in row["ppg"]["error"]
        assert row["praat"]["features"]


class TestThePraatFeaturesTravelInTheRow:
    """The Praat set is small enough to live in the row a consumer already reads."""

    def test_the_row_carries_the_features_and_the_manifest_fields(
        self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path
    ) -> None:
        """A reader gets the recording's identity and its measurements from one file."""
        out = tmp_path / "out"
        _run(corpus(["sub-01_ses-01_task-a"]), out)
        row = json.loads((out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").read_text())
        assert row["family"] == "speech"
        assert row["lexical"] is True
        assert row["praat"]["features"]


class TestTheSliceLog:
    """A finished task says what it did without anybody walking 60,000 directories."""

    def test_the_log_and_summary_name_every_row(self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path) -> None:
        """One JSONL row per recording the task handled, and the counts beside it."""
        out = tmp_path / "out"
        stems = [f"sub-{i:02d}_ses-01_task-a" for i in range(3)]
        summary = _run(corpus(stems), out, slice_index=0, slice_count=1)
        log = (out / "slices" / "slice-0-of-1.jsonl").read_text().splitlines()
        assert len(log) == 3
        assert json.loads((out / "slices" / "slice-0-of-1.summary.json").read_text())["counts"] == {"ok": 3}
        assert summary["phonemes"] == list(PHONEME_LABELS)


class TestTheExitCode:
    """The code reports whether every recording reached a determinate outcome, not what it found."""

    def test_a_clean_slice_exits_zero(self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path) -> None:
        """Both products landed for every recording, so nothing is owed a second look."""
        out = tmp_path / "out"
        manifest = corpus(["sub-01_ses-01_task-a"])
        argv = [str(manifest), str(out), "--slice-index", "0", "--slice-count", "1", "--device", "cpu"]
        assert cli.main(argv) == 0
        assert json.loads((out / "slices" / "slice-0-of-1.summary.json").read_text())["device"] == "cpu"

    def test_an_error_row_exits_one(self, corpus: Callable[[Sequence[str]], Path], tmp_path: Path) -> None:
        """A recording nothing could be measured on is worth a non-zero code; the rows are written anyway."""
        manifest = corpus(["sub-01_ses-01_task-a"])
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["enhanced"] = str(tmp_path / "gone.flac")
        manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        out = tmp_path / "out"
        assert cli.main([str(manifest), str(out), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert (out / "sub-01" / "ses-01" / "sub-01_ses-01_task-a_features.json").is_file()

    def test_a_missing_manifest_exits_two(self, tmp_path: Path) -> None:
        """Nothing was measured, so the code says the arguments did not resolve."""
        argv = [str(tmp_path / "none.jsonl"), str(tmp_path / "out"), "--slice-index", "0", "--slice-count", "1"]
        assert cli.main(argv) == 2


class TestTheVenvIsNotBuiltInsideAnArrayTask:
    """A cold build is 810 s and the lock's patience is 600 s; the driver names the pre-build step."""

    def test_a_missing_venv_refuses_before_anything_is_measured(
        self,
        corpus: Callable[[Sequence[str]], Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exit 2, the message names ``ensure_ppgs_venv``, and no output tree is created."""
        monkeypatch.setattr(cli, "ppgs_venv_is_provisioned", lambda: False)
        out = tmp_path / "out"
        manifest = corpus(["sub-01_ses-01_task-a"])
        code = cli.main([str(manifest), str(out), "--slice-index", "0", "--slice-count", "1"])
        assert code == 2
        assert "ensure_ppgs_venv" in capsys.readouterr().err
        assert not out.exists()
