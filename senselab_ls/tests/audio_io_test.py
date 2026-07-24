"""Routing tests for load_audio (which fetch path a ref takes), without real files.

``Audio`` is monkeypatched to a stub that just records the local path it is handed, so these
tests assert routing only.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab_ls.common import audio_io


@pytest.fixture(autouse=True)
def _stub_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace senselab Audio with a stub returning its filepath."""
    monkeypatch.setattr(audio_io, "Audio", lambda filepath: ("AUDIO", filepath))


def test_s3_ref_uses_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    """An s3:// ref is fetched via the boto3 downloader (never the LS downloader)."""
    monkeypatch.setattr(audio_io, "_download_s3", lambda uri: "/tmp/from_s3.wav")
    got = audio_io.load_audio("s3://bucket/data/x.wav", http_downloader=lambda r: "/should/not/be/used")
    assert got == ("AUDIO", "/tmp/from_s3.wav")


def test_ls_upload_ref_uses_downloader() -> None:
    """A bare LS upload ref (no scheme, not a local file) goes through http_downloader."""
    seen: dict[str, Any] = {}

    def downloader(ref: str) -> str:
        seen["ref"] = ref
        return "/tmp/fetched.wav"

    got = audio_io.load_audio("upload/275267/f8a36cea-audio.wav", http_downloader=downloader)
    assert got == ("AUDIO", "/tmp/fetched.wav")
    assert seen["ref"] == "upload/275267/f8a36cea-audio.wav"  # passed through for LS SDK to resolve


def test_existing_local_file_is_opened_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing local path is opened directly, not sent to the downloader."""
    monkeypatch.setattr(audio_io.os.path, "isfile", lambda p: True)
    got = audio_io.load_audio("/data/clip.wav", http_downloader=lambda r: "/should/not/be/used")
    assert got == ("AUDIO", "/data/clip.wav")


def test_unresolvable_ref_without_downloader_raises() -> None:
    """A non-s3, non-local ref with no downloader is a clear error, not a bogus local open."""
    with pytest.raises(FileNotFoundError):
        audio_io.load_audio("upload/1/x.wav")
