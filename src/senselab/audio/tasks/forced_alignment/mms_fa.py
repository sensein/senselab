"""Word-sequence forced alignment via torchaudio's MMS_FA bundle.

Fills the previously-dead torchaudio slot of the forced-alignment task
(``constants.DEFAULT_ALIGN_MODELS_TORCH`` / the ``model_type == "torchaudio"``
branch — see architecture-review.md F2/T047): a self-contained, lazily-imported
aligner that maps an ordered word list onto a 16 kHz waveform. Used by the
adaptive workflow's U3 consensus re-alignment; import by full module path
(``senselab.audio.tasks.forced_alignment.mms_fa``) — deliberately NOT re-exported
from the package ``__init__`` so importing it never pulls the transformers-based
aligner stack.

The first call downloads the MMS_FA bundle (~1.2 GB); the SIGALRM timeout guard
keeps callers responsive on cold caches (fallback semantics are the caller's).
"""

from __future__ import annotations

import re
from typing import Any

TARGET_SR = 16000


def align_words_mms_fa(
    wav_16k: Any,  # noqa: ANN401 — 1-D float32 numpy array
    words: list[str],
    *,
    timeout_s: float = 600.0,
) -> tuple[list[dict[str, float]] | None, str | None]:
    """Align ``words`` (in order) to ``wav_16k`` → ``[{start, end}]`` matching 1:1.

    Returns ``(spans, None)`` on success or ``(None, reason)`` when torchaudio /
    the bundle is unavailable, a word cannot be romanized for the MMS_FA
    dictionary, the timeout fires (cold-cache bundle download), or the aligner
    returns a span count mismatch. Never raises for these expected conditions.
    """
    import signal

    try:
        import torch  # noqa: PLC0415
        from torchaudio.pipelines import MMS_FA as bundle  # noqa: PLC0415, N811
    except ImportError as exc:
        return None, f"aligner_backend_unavailable ({getattr(exc, 'name', exc)})"

    norm = [re.sub(r"[^a-z']", "", w.lower()) for w in words]
    if any(not t for t in norm):
        return None, "unalignable_tokens (non-romanizable words)"

    def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ANN401, ARG001
        raise TimeoutError(f"mms_fa_timeout ({timeout_s}s)")

    old_handler = None
    timer_armed = False
    try:
        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.1, timeout_s))
        timer_armed = True
    except ValueError:  # not in main thread — proceed unguarded
        old_handler = None
    try:
        model = bundle.get_model()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        with torch.no_grad():
            emission, _ = model(torch.from_numpy(wav_16k).unsqueeze(0))
            spans = aligner(emission[0], tokenizer(norm))
        ratio = len(wav_16k) / emission.shape[1]
        out = []
        for span in spans:
            start = span[0].start * ratio / TARGET_SR
            end = span[-1].end * ratio / TARGET_SR
            out.append({"start": round(float(start), 4), "end": round(float(end), 4)})
        if len(out) != len(words):
            return None, f"alignment_count_mismatch ({len(out)} != {len(words)})"
        return out, None
    except TimeoutError as exc:
        return None, f"aligner_timeout ({exc})"
    except Exception as exc:  # noqa: BLE001 — expected-failure envelope for callers
        return None, f"alignment_failed ({exc!r})"
    finally:
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
