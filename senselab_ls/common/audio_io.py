"""Resolve an audio reference to a senselab ``Audio``.

Supports the three reference forms a Label Studio task can carry:

* ``s3://bucket/key`` -- read directly via boto3 (the EC2 instance role grants S3 read),
* ``http(s)://...`` -- an LS-hosted upload or presigned URL,
* a local filesystem path (dev / testing).

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

    Args:
        ref: An ``s3://`` URI, an ``http(s)://`` URL, or a local path.
        http_downloader: Optional callable that downloads an HTTP(S) URL and returns a local
            path (e.g. ``LabelStudioMLBase.get_local_path``). When omitted, an authenticated
            GET using ``LABEL_STUDIO_API_KEY`` is used.

    Returns:
        A senselab ``Audio`` constructed from the resolved local file.
    """
    scheme = urlparse(ref).scheme
    if scheme == "s3":
        return Audio(filepath=_download_s3(ref))
    if scheme in ("http", "https"):
        local = http_downloader(ref) if http_downloader is not None else _download_http(ref)
        return Audio(filepath=local)
    return Audio(filepath=ref)


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
