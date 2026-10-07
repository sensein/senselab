"""Audio extraction from video: the written file's subtype must match the container asked for.

The extractor writes into a ``TemporaryDirectory`` that is gone by the time it returns, so these
tests intercept ``read_files_from_disk`` -- called while the directory is still alive -- and copy the
real files out. The write under test is the production one, not a reimplementation.

Reasoning: ``specs/20260819-091500-wav-subtype-sweep/design.md``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import av
import numpy as np
import pytest
import soundfile as sf

from senselab.video.tasks import input_output as vio

_VIDEO = Path(__file__).resolve().parents[2] / "data_for_testing" / "video_48khz_stereo_16bits.mp4"


@pytest.fixture
def extracted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:  # noqa: ANN401 -- returns a closure
    """Return a callable that extracts audio and yields the written file paths."""

    def _extract(**kwargs: Any) -> List[Path]:  # noqa: ANN401
        captured: List[Path] = []

        def _copy_out(paths: List[str]) -> Dict[str, Any]:
            for i, path in enumerate(paths):
                destination = tmp_path / f"captured_{i}{Path(path).suffix}"
                shutil.copyfile(path, destination)
                captured.append(destination)
            return {}

        monkeypatch.setattr(vio, "read_files_from_disk", _copy_out)
        vio.extract_audios_from_local_videos(str(_VIDEO), **kwargs)
        return captured

    return _extract


def test_the_fixture_decodes_as_float_so_the_float_path_is_reachable() -> None:
    """The premise of the tests below: PyAV hands back ``fltp`` for this stream, not ``s16``."""
    with av.open(str(_VIDEO)) as container:
        stream = container.streams.audio[0]
        assert stream.format.name == "fltp"
        frame = next(container.decode(audio=0))
    assert np.asarray(frame.to_ndarray()).dtype == np.float32


def test_the_s16_codec_hint_still_writes_pcm_16(extracted: Any) -> None:  # noqa: ANN401
    """The default path is unchanged: an int16 array belongs in PCM_16."""
    paths = extracted()
    assert paths
    assert sf.info(str(paths[0])).subtype == "PCM_16"


def test_a_float_codec_hint_writes_float_not_pcm_16(extracted: Any) -> None:  # noqa: ANN401
    """Pre-fix this wrote PCM_16, clipping every decoded sample beyond +-1."""
    paths = extracted(acodec="pcm_f32le")
    assert paths
    info = sf.info(str(paths[0]))
    assert info.format == "WAV"
    assert info.subtype == "FLOAT"


def test_a_flac_container_gets_the_widest_integer_subtype_it_has(extracted: Any) -> None:  # noqa: ANN401
    """FLAC has no float subtype, so FLOAT would raise and PCM_16 would clip; PCM_24 is the answer."""
    paths = extracted(audio_format="flac", acodec="pcm_f32le")
    assert paths
    info = sf.info(str(paths[0]))
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"


def test_a_lossy_container_is_still_writable(extracted: Any) -> None:  # noqa: ANN401
    """The fix must not force FLOAT onto a container that rejects it."""
    paths = extracted(audio_format="ogg", acodec="pcm_f32le")
    assert paths
    assert sf.info(str(paths[0])).subtype == "VORBIS"


def test_a_video_with_no_audio_track_yields_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pre-existing behaviour, held while the write below it changed."""
    silent = tmp_path / "silent.mp4"
    with av.open(str(silent), "w") as out:
        stream = out.add_stream("mpeg4", rate=5)
        stream.width, stream.height = 16, 16
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))

    called: List[Any] = []

    def _record(paths: Any) -> Dict[str, Any]:  # noqa: ANN401
        called.append(paths)
        return {}

    monkeypatch.setattr(vio, "read_files_from_disk", _record)
    assert vio.extract_audios_from_local_videos(str(silent)) == {}
    assert not called, "a video with no audio track must not reach the reader at all"
