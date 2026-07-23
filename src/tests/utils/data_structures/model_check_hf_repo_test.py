"""check_hf_repo_exists must distinguish 'genuinely absent' from 'could not verify'.

Regression: a transient 429/network error during verification was swallowed as
``return False`` → HFModel validation then reported the model as "missing". Only a
genuine not-found should be False; transient errors (and gated repos) must surface.
"""

import logging

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RevisionNotFoundError

from senselab.utils.data_structures.model import check_hf_repo_exists


def _hf_error(cls: type[HfHubHTTPError], message: str, status: int) -> HfHubHTTPError:
    """Build a real hf_hub HTTP error (they require an httpx Response with a request)."""
    request = httpx.Request("GET", "https://huggingface.co")
    return cls(message, response=httpx.Response(status, request=request))


def test_genuine_not_found_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo/revision that truly does not exist -> False (unchanged, correct)."""

    def _raise(*a: object, **k: object) -> None:
        raise _hf_error(RevisionNotFoundError, "no such revision", 404)

    monkeypatch.setattr("senselab.utils.dependencies.ensure_hf_model", _raise)
    assert check_hf_repo_exists("org/model", "bogus") is False


def test_transient_error_logged_and_reraised(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A transient/rate-limit error must NOT be masked as 'missing' — log and re-raise."""

    class _RateLimited(Exception):
        pass

    def _raise(*a: object, **k: object) -> None:
        raise _RateLimited("429 too many requests")

    monkeypatch.setattr("senselab.utils.dependencies.ensure_hf_model", _raise)
    with caplog.at_level(logging.WARNING, logger="senselab"), pytest.raises(_RateLimited):
        check_hf_repo_exists("org/model", "main")
    assert any("org/model" in rec.getMessage() for rec in caplog.records)


def test_gated_repo_reraised_not_reported_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated repo exists (needs auth) -> raise GatedRepoError, not report 'missing'."""

    def _raise(*a: object, **k: object) -> None:
        raise _hf_error(GatedRepoError, "gated", 403)

    monkeypatch.setattr("senselab.utils.dependencies.ensure_hf_model", _raise)
    with pytest.raises(GatedRepoError):
        check_hf_repo_exists("org/model", "main")
