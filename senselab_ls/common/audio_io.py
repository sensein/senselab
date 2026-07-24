"""Resolve an audio reference to a senselab ``Audio``.

Handles every reference form a Label Studio task can carry:

* ``s3://bucket/key`` -- read directly via boto3 (the EC2 instance role grants S3 read); this is
  the b2ai path, where the ref keys both the bytes and the dataset metadata.
* Label-Studio-hosted refs -- an uploaded file (``upload/<proj>/<file>`` or
  ``/data/upload/...``), local-storage (``/data/local-files?d=...``), or an ``http(s)://`` URL.
  All of these are resolved by ``http_downloader`` (i.e. ``LabelStudioMLBase.get_local_path``,
  which downloads with ``LABEL_STUDIO_URL`` + ``LABEL_STUDIO_API_KEY`` and even presigns cloud
  storage when given ``task_id``).
* an existing local filesystem path (dev / testing).

boto3 and requests are imported lazily so unit tests that use local paths need neither.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional
from urllib.parse import urlparse

from senselab.audio.data_structures import Audio


def load_audio(
    ref: str,
    *,
    http_downloader: Optional[Callable[[str], str]] = None,
) -> Audio:
    """Load ``ref`` into a senselab ``Audio``.

    Resolution order: ``s3://`` (boto3) → existing local file → Label-Studio-hosted (via
    ``http_downloader``) → bare ``http(s)://`` (direct GET).

    Args:
        ref: An ``s3://`` URI, a Label-Studio-hosted ref (``upload/...``, ``/data/...``,
            ``http(s)://``), or a local path.
        http_downloader: Callable that fetches a Label-Studio-hosted ref and returns a local
            path -- pass ``LabelStudioMLBase.get_local_path``. Required for LS-hosted refs.

    Returns:
        A senselab ``Audio`` constructed from the resolved local file.

    Raises:
        FileNotFoundError: If the ref is not ``s3://``/local and no ``http_downloader`` is given.
    """
    scheme = urlparse(ref).scheme
    if scheme == "s3":
        return Audio(filepath=_download_s3(ref))
    if scheme in ("", "file") and os.path.isfile(_strip_file_scheme(ref)):
        return Audio(filepath=_strip_file_scheme(ref))
    if http_downloader is not None:
        return Audio(filepath=http_downloader(ref))
    if scheme in ("http", "https"):
        return Audio(filepath=_download_http(ref))
    raise FileNotFoundError(
        f"Cannot resolve audio ref {ref!r}: not s3://, not an existing local file, and no Label "
        f"Studio downloader available (set LABEL_STUDIO_URL + LABEL_STUDIO_API_KEY)."
    )


def _strip_file_scheme(ref: str) -> str:
    """Return ``ref`` without a leading ``file://`` scheme."""
    return ref[len("file://") :] if ref.startswith("file://") else ref


def _download_s3(uri: str) -> str:
    """Download an ``s3://bucket/key`` object to a temp file and return its path.

    Args:
        uri: The ``s3://`` URI to fetch.

    Returns:
        The local path of the downloaded file.
    """
    import boto3  # lazy: only needed on the S3 path

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    suffix = os.path.splitext(key)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    boto3.client("s3").download_file(bucket, key, path)
    return path


def _download_http(url: str) -> str:
    """Download an HTTP(S) URL to a temp file, authenticating to Label Studio when configured.

    Args:
        url: The HTTP(S) URL to fetch.

    Returns:
        The local path of the downloaded file.
    """
    import requests  # lazy: only needed on the HTTP path

    headers = {}
    token = os.getenv("LABEL_STUDIO_API_KEY")
    if token:
        headers["Authorization"] = f"Token {token}"
    suffix = os.path.splitext(urlparse(url).path)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as handle:
        handle.write(response.content)
    return path
