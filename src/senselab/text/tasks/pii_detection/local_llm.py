"""The optional local-LLM PII detector: off by default, loopback-only.

The fourth engine #542 runs alongside the rule cascade, Presidio and GLiNER. It is the
only one that talks to a *service* rather than loading a model, which is what drives the
two design choices here.

**Loopback is a checked invariant, not a documented property.** Transcript text never
leaving the machine is this module's central promise, and a base URL is exactly the kind
of field a later edit widens without noticing — pointing it at a hosted endpoint would
send clinical speech to a third party in a single character change. Validating in
``__post_init__`` means that edit has to defeat an assertion rather than slip past a
docstring. ``localhost``, anything in ``127.0.0.0/8``, and ``::1`` are accepted; a
hostname that merely *resolves* to loopback is not, because what it resolves to is not
fixed at the time we check it.

**Off by default.** :func:`senselab.text.tasks.pii_detection.api.default_detectors`
omits ``"llm"`` while ``_KNOWN_DETECTORS`` includes it, so it counts in the
cross-detector agreement denominator when it runs but never turns itself on. Default-on
would make a scan depend on whether a server happened to be listening, and the same
corpus would score differently on two machines with no record of why.

**A detector that could not run never reports a clean pass.** :func:`scan_or_fail`
returns spans *and* an optional failure string; a refused connection, a timeout, and an
unparsable response all populate the failure, which ``detect_pii`` records in
``report.failures`` while leaving ``"llm"`` out of ``detectors_used``. Returning an empty
span list on a connection error would read identically to "the LLM found no PII", which
is the one wrong answer a PII check must not give.

Runs in the host process, not the detection venv: it needs nothing but ``urllib`` from
the standard library, and keeping it here makes the loopback invariant testable without
building a venv at all. The venv exists for ``presidio``/``spacy``/``gliner``, none of
which this touches.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT_S = 60.0

# Asks for a bare JSON array and nothing else. Models still wrap it in prose or a code
# fence often enough that `_parse_response` has to tolerate both rather than treat them
# as a failure -- a chatty model is not an unreachable one.
_PROMPT = (
    "You are a PII detector. Find every piece of personally identifiable information in "
    "the text below. Respond with ONLY a JSON array, no prose. Each element must be an "
    'object with keys "text" (the exact substring), "category" (one of PERSON, '
    "EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME, ID, ORGANIZATION, OTHER), and "
    '"score" (your confidence, 0 to 1). Return [] if there is none.\n\nTEXT:\n'
)


def _is_loopback(base_url: str) -> bool:
    """Return whether ``base_url``'s host is a literal loopback address.

    Args:
        base_url: The configured endpoint, e.g. ``"http://127.0.0.1:11434"``.

    Returns:
        ``True`` for ``localhost``, any ``127.0.0.0/8`` literal, and ``::1``. A hostname
        other than ``localhost`` returns ``False`` even if DNS would resolve it to a
        loopback address: resolution is not stable between the check and the request, so
        trusting it would make the guarantee depend on a resolver the caller does not
        control.
    """
    host = urlparse(base_url).hostname
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class LocalLlmConfig:
    """Where the optional local LLM lives and how long to wait for it.

    Attributes:
        base_url: Endpoint root. Must be loopback — see the module docstring.
        model: Model tag the server knows, e.g. ``"llama3.1:8b"``.
        timeout_s: Per-request ceiling. A hung server has to become a recorded failure
            rather than stalling a batch scan indefinitely.

    Raises:
        ValueError: If ``base_url`` is not loopback.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        """Reject any non-loopback endpoint at construction.

        Raises:
            ValueError: If ``base_url`` does not point at localhost / 127.0.0.0/8 / ::1.
        """
        if not _is_loopback(self.base_url):
            raise ValueError(
                f"LocalLlmConfig.base_url must be loopback (localhost, 127.0.0.1, or ::1); "
                f"got {self.base_url!r}. Transcript text must not leave the machine."
            )


@dataclass
class LlmScanResult:
    """One local-LLM scan's outcome.

    Attributes:
        spans: Raw span dicts in the same shape the venv worker's scans produce
            (``text`` / ``category`` / ``source`` / ``score``). Empty when the scan
            failed — read ``failure`` to tell that from "found nothing".
        failure: ``None`` on success; otherwise why the detector did not run.
    """

    spans: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None


def _parse_response(payload: str, model: str) -> list[dict[str, Any]]:
    """Turn the model's reply into span dicts, tolerating a fenced or prose-wrapped array.

    Args:
        payload: The model's raw completion text.
        model: Model tag, embedded in each span's ``source`` so a report can attribute a
            finding to the specific model that made it.

    Returns:
        Span dicts. Entries that are not objects, or that carry no ``text``, are dropped
        rather than raising: one malformed element in an otherwise good array is not a
        reason to discard the whole scan.

    Raises:
        ValueError: If no JSON array can be found at all.
    """
    start, end = payload.find("["), payload.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in response: {payload[:200]!r}")
    parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")

    spans: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        spans.append(
            {
                "text": text,
                "category": str(item.get("category") or "OTHER").upper(),
                "source": f"llm/{model}",
                # Clamped rather than trusted: a model asked for 0-1 will occasionally
                # answer 95, and `_compute_detection_confidence` treats the score as a
                # probability.
                "score": min(1.0, max(0.0, score)) if score is not None else None,
            }
        )
    return spans


def scan_or_fail(text: str, config: LocalLlmConfig) -> LlmScanResult:
    """Scan ``text`` with the local LLM, reporting failure rather than raising.

    Args:
        text: The transcript to scan.
        config: Endpoint, model and timeout. Already loopback-validated by construction.

    Returns:
        An :class:`LlmScanResult` whose ``failure`` is ``None`` only when the server
        answered and its reply parsed. Every error path — refused connection, timeout,
        HTTP error, unparsable body — returns empty spans *and* a populated ``failure``,
        so a caller can never mistake "did not run" for "found nothing".
    """
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/generate",
        data=json.dumps({"model": config.model, "prompt": _PROMPT + text, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:  # noqa: S310 — loopback-only, enforced in LocalLlmConfig
            body = json.loads(response.read().decode())
        return LlmScanResult(spans=_parse_response(body.get("response", ""), config.model))
    except urllib.error.URLError as exc:
        # Covers the refused-connection and timeout cases, which are the ordinary
        # outcome when nobody is running a server -- the reason this detector is opt-in.
        return LlmScanResult(failure=f"llm: could not connect to {config.base_url}: {exc.reason}")
    except Exception as exc:  # noqa: BLE001 — every failure mode has to become a report field, not a crash
        return LlmScanResult(failure=f"llm: {type(exc).__name__}: {exc}")
