"""SpeechScore: the metric table, the reference requirement, the pinned clone, the payload.

The scorer itself is never run: it needs a git clone of upstream, a venv with eighteen metric
dependencies, and three sets of neural weights. What is tested is every host-side decision, plus the
worker source's two non-negotiable constraints, which are the ones that would fail silently or
obscurely inside a subprocess.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import clearvoice_speechscore as ss


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Any]:
    """Stub the venv and the subprocess, returning one score per requested metric."""
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(ss, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(ss, "venv_python", lambda venv_dir: "python3")

    def fake_run(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        payload = json.loads(str(kwargs["input"]))
        captured["payload"] = payload
        captured["timeout"] = kwargs["timeout"]
        # Inspected here rather than after the call: the parent's TemporaryDirectory is gone by then.
        import soundfile

        captured["test_subtypes"] = [soundfile.info(path).subtype for path in payload["test_paths"]]
        captured["reference_present"] = [path is not None for path in payload["reference_paths"]]
        results = []
        for _ in payload["test_paths"]:
            scores: Dict[str, Any] = {}
            for name in payload["metrics"]:
                metric = ss.SPEECHSCORE_METRICS[name]
                scores[name] = {field: 3.0 for field in metric.fields} if metric.fields else 1.5
            results.append(scores)
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"results": results}), stderr="")

    monkeypatch.setattr(ss.subprocess, "run", fake_run)
    return captured


# ── The metric table ──────────────────────────────────────────────────


def test_the_table_holds_all_eighteen_metric_families() -> None:
    """SpeechScore's own factory dispatches on exactly these names."""
    assert len(ss.SPEECHSCORE_METRICS) == 18


def test_the_four_no_reference_metrics_are_the_ones_upstream_documents_as_such() -> None:
    """Upstream's ``intrusive`` attribute is unread and inverted for three of these."""
    assert set(ss.NO_REFERENCE_METRICS) == {"DNSMOS", "NISQA", "DISTILL_MOS", "SRMR"}
    assert len(ss.REFERENCE_METRICS) == 14


def test_the_multi_field_metrics_declare_their_sub_keys() -> None:
    """DNSMOS, NISQA and BSSEval return mappings; a caller needs to know which keys to expect."""
    assert ss.SPEECHSCORE_METRICS["DNSMOS"].fields == ("SIG", "BAK", "OVRL", "P808_MOS")
    assert ss.SPEECHSCORE_METRICS["BSSEval"].fields == ("ISR", "SAR", "SDR")
    assert ss.SPEECHSCORE_METRICS["PESQ"].fields == ()


# ── The reference requirement ─────────────────────────────────────────


def test_without_a_reference_only_the_no_reference_metrics_are_selected() -> None:
    """The default must be what the call can actually compute."""
    assert ss.resolve_speechscore_metrics(None, has_references=False) == list(ss.NO_REFERENCE_METRICS)


def test_with_a_reference_every_metric_is_selected() -> None:
    """With a reference, nothing has to be withheld."""
    assert len(ss.resolve_speechscore_metrics(None, has_references=True)) == 18


def test_asking_for_an_intrusive_metric_without_a_reference_is_refused() -> None:
    """Upstream scores it against a zero-padded copy of the test signal and returns a real number."""
    with pytest.raises(ValueError) as exc:
        ss.resolve_speechscore_metrics(["PESQ", "DNSMOS"], has_references=False)
    message = str(exc.value)
    assert "PESQ" in message
    assert "copy of the test signal" in message
    assert "DNSMOS" in message, "the message must name what the caller can use instead"


def test_an_unknown_metric_name_enumerates_the_real_ones() -> None:
    """A typo must name the metric the caller meant, not merely fail."""
    with pytest.raises(ValueError) as exc:
        ss.resolve_speechscore_metrics(["PSEQ"], has_references=True)
    assert "'PSEQ'" in str(exc.value) and "PESQ" in str(exc.value)


def test_metric_names_are_case_insensitive_and_returned_in_table_order() -> None:
    """A result dict's key order must not depend on the caller's argument order."""
    assert ss.resolve_speechscore_metrics(["srmr", "dnsmos"], has_references=False) == ["DNSMOS", "SRMR"]
    assert ss.resolve_speechscore_metrics(["DNSMOS", "SRMR"], has_references=False) == ["DNSMOS", "SRMR"]


# ── Scoring ───────────────────────────────────────────────────────────


def test_scoring_returns_one_dict_per_audio_keyed_by_metric(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """One row per audio, with each metric under its own name."""
    scores = ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["DNSMOS", "SRMR"])
    assert len(scores) == 1
    assert list(scores[0]) == ["DNSMOS", "SRMR"]
    assert scores[0]["DNSMOS"]["OVRL"] == 3.0
    assert scores[0]["SRMR"] == 1.5


def test_the_test_audio_is_handed_over_as_float(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """A PCM_16 hand-off would quantise the very signal whose quality is being measured."""
    ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["SRMR"])
    assert worker["test_subtypes"] == ["FLOAT"]


def test_a_reference_is_written_alongside_each_test_signal(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """The intrusive metrics need both, paired by position."""
    ss.extract_speechscore_metrics_from_audios(
        [mono_audio_sample], reference_audios=[mono_audio_sample], metrics=["PESQ"]
    )
    assert worker["reference_present"] == [True]


def test_no_reference_is_sent_as_null(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """SpeechScore's reader branches on ``reference_path is None``."""
    ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["SRMR"])
    assert worker["payload"]["reference_paths"] == [None]


def test_a_mismatched_reference_count_is_refused(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """Pairing is positional, so a length mismatch would score against the wrong file."""
    with pytest.raises(ValueError, match="one entry per audio"):
        ss.extract_speechscore_metrics_from_audios(
            [mono_audio_sample, mono_audio_sample], reference_audios=[mono_audio_sample]
        )


def test_the_pinned_commit_reaches_the_worker(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """The clone is the pin: it fixes both the metric code and the committed weights."""
    ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["SRMR"])
    assert worker["payload"]["commit"] == ss.SPEECHSCORE_COMMIT
    assert len(ss.SPEECHSCORE_COMMIT) == 40


def test_the_ceiling_scales_with_the_metric_count(worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """Eighteen metrics over one file is eighteen passes, three of them neural."""
    one = ss.default_speechscore_timeout_s(3600.0, 1)
    many = ss.default_speechscore_timeout_s(3600.0, 18)
    assert many > one > 0
    ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["SRMR"], timeout_s=11.0)
    assert worker["timeout"] == 11.0


def test_the_ceiling_has_a_floor(worker: Dict[str, Any]) -> None:
    """A short file still pays for the clone and three sets of weights."""
    assert ss.default_speechscore_timeout_s(0.5, 1) == ss._TIMEOUT_FLOOR_S


def test_a_non_positive_ceiling_is_refused(mono_audio_sample: Audio) -> None:
    """A zero ceiling would kill the worker instantly and blame a timeout."""
    with pytest.raises(ValueError, match="positive number of seconds"):
        ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["SRMR"], timeout_s=0)


def test_a_timeout_says_which_metrics_dominate(
    monkeypatch: pytest.MonkeyPatch, mono_audio_sample: Audio, tmp_path: Path
) -> None:
    """An actionable failure names the way out, and here that includes dropping the neural metrics."""
    monkeypatch.setattr(ss, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(ss, "venv_python", lambda venv_dir: "python3")

    def timing_out(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        raise ss.subprocess.TimeoutExpired(cmd, float(kwargs["timeout"]))  # type: ignore[arg-type]

    monkeypatch.setattr(ss.subprocess, "run", timing_out)
    with pytest.raises(RuntimeError) as exc:
        ss.extract_speechscore_metrics_from_audios([mono_audio_sample], metrics=["NISQA"], timeout_s=1.0)
    message = str(exc.value)
    assert "1s ceiling" in message
    assert "timeout_s" in message and "NISQA" in message


def test_no_audio_means_no_worker(worker: Dict[str, Any]) -> None:
    """An empty list must not clone upstream or build a venv."""
    assert ss.extract_speechscore_metrics_from_audios([]) == []
    assert "payload" not in worker


# ── The worker source ─────────────────────────────────────────────────


def test_the_worker_chdirs_into_the_package_directory() -> None:
    """DNSMOS, NISQA and DISTILL_MOS address their weights relative to the working directory."""
    assert "os.chdir(package_dir)" in ss._WORKER_SCRIPT
    assert 'package_dir = repo_dir / "speechscore"' in ss._WORKER_SCRIPT


def test_the_worker_puts_the_package_directory_itself_on_the_path() -> None:
    """Its parent would make ``speechscore`` a package, and its __init__ imports two absent modules."""
    assert "sys.path.insert(0, str(package_dir))" in ss._WORKER_SCRIPT


def test_the_worker_clones_sparsely_at_the_pinned_commit() -> None:
    """The rest of the studio carries checkpoints this never reads."""
    assert "sparse-checkout" in ss._WORKER_SCRIPT
    assert '"/speechscore/"' in ss._WORKER_SCRIPT
    assert 'args["commit"]' in ss._WORKER_SCRIPT


def test_the_worker_never_offers_windowing() -> None:
    """basis.py's windowed branch references an undefined name and raises NameError."""
    assert "window=None" in ss._WORKER_SCRIPT


def test_the_metric_requirements_name_what_the_scores_import() -> None:
    """SRMR needs gammatone; NISQA_lib imports pandas, matplotlib and tqdm at module scope."""
    named = {req.split("=")[0].split(">")[0].split("<")[0].strip().lower() for req in ss.SPEECHSCORE_REQUIREMENTS}
    assert {"gammatone", "museval", "onnxruntime", "xls_r_sqa", "pandas", "matplotlib", "tqdm"} <= named
