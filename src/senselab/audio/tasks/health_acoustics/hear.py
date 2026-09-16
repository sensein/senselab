"""``google/hear`` — Google's Health Acoustic Representations, via an isolated TensorFlow venv.

HeAR (Baur et al., 2024, arXiv:2403.02522) is a ViT-L masked-autoencoder trained on 313 M
two-second audio clips mined for non-semantic human sounds (coughs, breaths, throat clears).
The repository ships **two** artifacts, and both are exposed here:

* the **encoder** — ``serving_default`` maps ``x: (None, 32000) float32`` to
  ``output_0: (None, 512)``;
* the **bundled event detector** — ``event_detector/event_detector_large`` (MobileNetV3-Large,
  ~3 M parameters) and ``event_detector/event_detector_small`` (MobileNetV3-Small, ~1 M), both
  mapping ``audio_wav: (1, 32000) float32`` to eight independent presence probabilities over
  :data:`HEAR_EVENT_LABELS`. Upstream's ``event_detector/README.md`` states the two share one
  frontend and one label set; the signatures were probed to confirm it, and they differ only in
  the output tensor's name (``mobilenetv3_large_model`` / ``mobilenetv3_small_model``), which is
  why the worker reads the sole output by position rather than by name.

Why a subprocess venv rather than ``transformers``
--------------------------------------------------
``google/hear`` ships TensorFlow SavedModels (``library_name: tf-keras``) and senselab is
torch-based, so TensorFlow runs in an isolated venv — the same pattern as
``classification/yamnet.py``, and for the same reason: TF in senselab's core dependency set is
both heavy and prone to conflicting with the torch stack.

A torch conversion **does** exist, ``google/hear-pytorch``, and PR #366 used it via
``transformers.AutoModel``. It was rejected here on three counts, checked rather than assumed:

1. It is separately gated, and this account is authorized for ``google/hear`` but **not** for
   ``google/hear-pytorch`` (the Hub answers ``GatedRepoError``: "not in the authorized list"),
   so the claim that it produces the same embeddings as the SavedModel cannot be verified here
   — and an unverifiable equivalence is not a basis for a default backend.
2. It is an ``image-feature-extraction`` ``ViTModel``: it takes a *spectrogram*, not a waveform.
   The matching frontend is not in that repo; PR #366 imported ``preprocess_audio`` from a
   local clone of ``github.com/google-health/hear`` (``hear.python.data_processing.audio_utils``),
   i.e. an un-pinned, non-installable third source. The SavedModel has the frontend fused in, so
   waveform in, embedding out, one artifact.
3. It carries no event detector. The detector exists **only** as a SavedModel, so TensorFlow is
   required for the second capability whatever the first one does; routing the encoder through
   torch would mean two frameworks, two provenances and two preprocessing paths for one model
   family.

The venv is deliberately its own (``hear``) rather than shared with ``yamnet``'s: ``ensure_venv``
keys reuse on the exact requirements list, so two backends sharing a venv name with different
lists would delete and rebuild each other's tree on alternate calls.

Measured behaviour this module is built around
----------------------------------------------
All measured on real recordings (see ``specs/20260819-hear-task/design.md`` for the tables):

* **The detector's window is hard-fixed at 32000 samples.** 0.5 / 1.0 / 1.5 / 3.0 / 4.0 s all
  raise ``InvalidArgumentError: Graph execution error``, and the batch dimension is pinned at 1.
  So the length is not a parameter here — only the hop is.
* **The encoder silently accepts other lengths.** Every length from 160 to 64000 samples returns
  a finite, plausible-looking 512-d vector, because the static shape is not enforced. A caller
  feeding a 0.3 s clip gets a confident vector that means much less than it looks like it does,
  which is why this module refuses sub-2 s input instead of forwarding it.
* **Padding destroys the representation.** Centred cosine between the same event under different
  framings runs 0.0–0.5 against a class margin of ~0.9: zero-padding a 0.3 s cough out to 2 s
  moves its embedding about as far as substituting unrelated audio. Hence
  :func:`plan_scan_windows` and :func:`plan_centred_windows` only ever return windows that lie
  wholly inside the recording, and the worker re-checks each window's length before inference —
  there is no code path here that pads.
* **Window shift is benign, amplitude is irrelevant.** ±50–200 ms of boundary error gives cosine
  0.93–0.98, and gains from ×0.1 to ×10 give 1.0000. That is what makes the tail-window and
  edge-clamp policies below cheap: a window nudged inward to stay inside the recording costs far
  less than one padded to sit where it was asked for.
* **Usable length falls off sharply below 2 s** (centred class margin +0.91 at 2 s, +0.46 at 1 s,
  +0.29 at 0.3 s) and 3 s is *worse* than 2 s, so 2 s is not merely the accepted length, it is
  the best one.

Revision pinning
----------------
:data:`HEAR_REVISION` is a 40-hex commit, never a ref (CLAUDE.md's rule). The parent stages that
commit with ``resolve_model`` and hands the worker the resulting local ``snapshots/<sha>/`` path,
so the worker's ``tf.saved_model.load`` is pinned by the path it is given and touches the Hub not
at all -- it never imports ``huggingface_hub``. That is the same shape as
``speech_to_text/crisperwhisper.py``, and why this file is listed in
``LOADER_CANNOT_PIN_SUBPROCESS_FILES`` in ``src/tests/utils/revision_pinning_guard_test.py``.

Licence and access
------------------
``google/hear`` is gated under the Health AI Developer Foundations terms
(https://developers.google.com/health-ai-developer-foundations/terms); access is granted
immediately on acknowledging them while logged in to Hugging Face. Staging goes through
``resolve_model``, which reads the token via ``get_huggingface_token()`` (``HF_TOKEN`` /
``HUGGING_FACE_HUB_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``, or a ``.env``) exactly like the other
gated models here (``pyannote/brouhaha``, DiariZen) — no HeAR-specific token mechanism, and no
interactive login: PR #366's ``notebook_login()`` prompt cannot work in a batch job.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import resolve_model
from senselab.utils.portable_audio_io import write_audio
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    venv_python,
)

HEAR_MODEL_ID = "google/hear"
HEAR_REVISION = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
"""The pinned commit, verified against the Hub — a SHA, never a ref.

CLAUDE.md's rule, and the reason for it: a ref-addressed download writes ``refs/main`` and
leaves the caller ref-addressed, so a later load through that pointer can read weights that a
different commit published while provenance still names the first one.
"""

HEAR_SAMPLING_RATE = 16000
HEAR_WINDOW_SAMPLES = 32000
"""2 s at 16 kHz. Not a parameter: the detector's graph rejects every other length, and the
encoder accepts them without meaning them (see the module docstring)."""

HEAR_WINDOW_SECONDS = HEAR_WINDOW_SAMPLES / HEAR_SAMPLING_RATE
HEAR_EMBEDDING_DIM = 512

HEAR_EVENT_LABELS: Tuple[str, ...] = (
    "Cough",
    "Snore",
    "Baby Cough",
    "Breathe",
    "Sneeze",
    "Throat Clear",
    "Laugh",
    "Speech",
)
"""The detector's eight outputs, **in graph order** — the tuple index is the output column.

Order is upstream's (``event_detector/README.md``), not alphabetical and not ours to sort: a
reordering here would silently relabel every score.
"""

ENCODER_SUBDIR = ""
"""The encoder is the repository root SavedModel, so its subdirectory is empty."""

EVENT_DETECTOR_SUBDIRS = {
    "large": "event_detector/event_detector_large",
    "small": "event_detector/event_detector_small",
}

# Public spec strings, in the style of ``"yamnet"`` / the S3PRL names: a plain string, not an
# ``HFModel``. An HFModel would validate and then mislead — nothing in transformers can load
# these SavedModels, so a spec that looks like an HF model spec would invite exactly the
# ``AutoModel.from_pretrained`` call this backend exists to replace.
HEAR_ENCODER_SPECS = frozenset({"hear", "google/hear", "hear-encoder"})
HEAR_EVENT_DETECTOR_SPECS: Dict[str, str] = {
    "hear-events": "large",
    "hear-event-detector": "large",
    "hear-event-detector-large": "large",
    "hear-event-detector-small": "small",
}

_HEAR_VENV = "hear"
# TensorFlow only. No tensorflow-hub (unlike yamnet: these weights come from the HF Hub, staged
# by the parent), no tf-keras (``tf.saved_model.load`` reads the graph directly and never asks
# Keras to rebuild the model), and no torch — which also means ``ensure_venv`` skips the CUDA
# probe and the PyTorch wheel index entirely for this venv.
#
# Verified pair on this host: tensorflow 2.21.0 on Python 3.11.15 loads both SavedModels and
# reproduces the signatures documented above. The range is left open below that because the
# SavedModel format is stable across 2.x; the ceiling is there because a 3.0 has not been tested
# against these graphs.
_HEAR_REQUIREMENTS = ["tensorflow>=2.16,<3", "numpy", "soundfile"]
_HEAR_PYTHON = "3.11"

_DEFAULT_TIMEOUT_S = 1800

_HEAR_WORKER = r"""
import json
import sys

try:
    import numpy as np
    import soundfile as sf
    import tensorflow as tf

    args = json.loads(sys.stdin.read())
    win = int(args["window_samples"])

    saved_model = tf.saved_model.load(args["saved_model_dir"])
    fn = saved_model.signatures["serving_default"]
    spec = fn.structured_input_signature[1]
    input_name = list(spec.keys())[0]

    # The detector's signature pins the batch dimension at 1 (``audio_wav: (1, 32000)``) while
    # the encoder's is free (``x: (None, 32000)``). Read it off the graph rather than trusting
    # the caller: feeding 2 windows to the detector does not degrade, it raises
    # InvalidArgumentError from deep inside the frontend's reshape.
    static_batch = spec[input_name].shape[0]
    batch = 1 if static_batch is not None else max(1, int(args.get("batch_size", 1)))

    results = []
    for job in args["jobs"]:
        x, sr = sf.read(job["wav"], dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        starts = [int(s) for s in job["starts"]]
        # Every window must already lie wholly inside the recording. The parent guarantees it;
        # this is the second check, because zero-padding a short window is the one failure that
        # returns a plausible number instead of an error (see the module docstring).
        for s in starts:
            if s < 0 or s + win > x.shape[0]:
                raise ValueError(
                    "window [%d, %d) is not inside the %d-sample recording %s; HeAR windows are "
                    "never padded" % (s, s + win, x.shape[0], job["wav"])
                )
        windows = np.stack([x[s:s + win] for s in starts]).astype("float32")

        outs = []
        for i in range(0, windows.shape[0], batch):
            block = windows[i:i + batch]
            out = fn(**{input_name: tf.constant(block, dtype=tf.float32)})
            # One output tensor, read by position: the two detectors name theirs
            # ``mobilenetv3_large_model`` / ``mobilenetv3_small_model`` and the encoder names
            # its ``output_0``.
            outs.append(np.asarray(list(out.values())[0]))
        array = np.concatenate(outs, axis=0).astype("float32")
        np.save(job["out"], array)
        results.append({"out": job["out"], "shape": [int(v) for v in array.shape]})

    print(json.dumps({"results": results, "batch": batch}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def resolve_event_detector(model: str) -> str:
    """Map a public detector spec string to its SavedModel subdirectory.

    Args:
        model: One of :data:`HEAR_EVENT_DETECTOR_SPECS`' keys.

    Returns:
        The repository-relative subdirectory of that detector.

    Raises:
        ValueError: If the spec is not a HeAR event-detector name.
    """
    try:
        size = HEAR_EVENT_DETECTOR_SPECS[model.strip().lower()]
    except KeyError:
        raise ValueError(
            f"{model!r} is not a HeAR event detector. Use one of: {sorted(HEAR_EVENT_DETECTOR_SPECS)}."
        ) from None
    return EVENT_DETECTOR_SUBDIRS[size]


def is_hear_encoder_spec(model: object) -> bool:
    """Whether ``model`` names the HeAR encoder (a plain string spec, not an ``HFModel``)."""
    return isinstance(model, str) and model.strip().lower() in HEAR_ENCODER_SPECS


def is_hear_event_detector_spec(model: object) -> bool:
    """Whether ``model`` names one of HeAR's bundled event detectors."""
    return isinstance(model, str) and model.strip().lower() in HEAR_EVENT_DETECTOR_SPECS


def seconds_to_hop_samples(hop_length: float) -> int:
    """Convert a hop in seconds to whole samples at HeAR's 16 kHz, with the two traps checked.

    Args:
        hop_length: Hop between successive 2 s windows, in seconds.

    Returns:
        The hop in samples, at least 1.

    Raises:
        ValueError: If ``hop_length`` is not positive, or rounds to zero samples.
    """
    if not hop_length > 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    hop_samples = int(round(hop_length * HEAR_SAMPLING_RATE))
    if hop_samples < 1:
        raise ValueError(
            f"hop_length={hop_length}s rounds to 0 samples at {HEAR_SAMPLING_RATE} Hz; "
            f"the smallest usable hop is {1 / HEAR_SAMPLING_RATE:.6f}s"
        )
    if hop_samples > HEAR_WINDOW_SAMPLES:
        # Not an error: a caller sampling a long recording sparsely is a legitimate use. But
        # the detector's response is a box-car of width (event + 2 s), so a hop wider than the
        # window can step over an event entirely and report nothing where there was something.
        warnings.warn(
            f"hop_length={hop_length}s exceeds HeAR's fixed {HEAR_WINDOW_SECONDS}s window, so "
            f"{(hop_samples - HEAR_WINDOW_SAMPLES) / HEAR_SAMPLING_RATE:.3f}s between windows is "
            "never seen by the model; events falling there are invisible, not absent.",
            stacklevel=2,
        )
    return hop_samples


def _require_two_seconds(n_samples: int, what: str) -> None:
    """Refuse input shorter than one window, with the measurement that makes padding wrong."""
    if n_samples < HEAR_WINDOW_SAMPLES:
        raise ValueError(
            f"{what} is {n_samples} samples ({n_samples / HEAR_SAMPLING_RATE:.3f}s at "
            f"{HEAR_SAMPLING_RATE} Hz), shorter than HeAR's fixed "
            f"{HEAR_WINDOW_SAMPLES}-sample ({HEAR_WINDOW_SECONDS}s) window. senselab will not "
            "pad it: measured on real audio, zero-padding a 0.3s event out to 2s moves its "
            "embedding as far as substituting unrelated audio (centred cosine 0.0-0.5 against a "
            "class margin of ~0.9), and the centred class margin falls from +0.91 at 2s to +0.29 "
            "at 0.3s. Pass the surrounding recording instead — use "
            "extract_hear_embeddings_at_times() to place windows on events inside it. The "
            "encoder would accept this length silently and return a confident-looking vector; "
            "that is the trap this error exists to close."
        )


def plan_scan_windows(n_samples: int, hop_samples: int) -> List[int]:
    """Plan the window start offsets for a sliding scan over ``n_samples``.

    Every returned window is exactly :data:`HEAR_WINDOW_SAMPLES` long and lies wholly inside the
    recording, so no caller of this function can end up padding. The final window is snapped so
    that it *ends* at the last sample rather than starting on the hop grid: without it, up to one
    hop of the tail is never looked at, and re-framing costs almost nothing (a 50-200 ms shift
    gives cosine 0.93-0.98) whereas skipping the tail costs everything in it.

    Args:
        n_samples: Length of the (already 16 kHz) recording.
        hop_samples: Hop between successive windows, in samples.

    Returns:
        Ascending window start offsets; at least one.

    Raises:
        ValueError: If the recording is shorter than one window, or the hop is not positive.
    """
    _require_two_seconds(n_samples, "audio")
    if hop_samples < 1:
        raise ValueError(f"hop_samples must be at least 1, got {hop_samples}")
    last_start = n_samples - HEAR_WINDOW_SAMPLES
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def plan_centred_windows(n_samples: int, centre_samples: Sequence[int]) -> List[int]:
    """Plan window starts centred on given offsets, clamped inward instead of padded.

    A window whose centre sits within 1 s of either edge cannot be centred and stay inside the
    recording. It is slid inward to the nearest legal position rather than padded, and the
    returned start says where it actually landed: the caller can see the shift, which is cheap
    (cosine 0.93-0.98 for 50-200 ms), whereas padding is not (cosine 0.0-0.5).

    Args:
        n_samples: Length of the (already 16 kHz) recording.
        centre_samples: Offsets to centre windows on. Order is preserved, duplicates kept —
            the result is positionally aligned with the request.

    Returns:
        One window start per requested centre.

    Raises:
        ValueError: If the recording is shorter than one window, or a centre is outside it.
    """
    _require_two_seconds(n_samples, "audio")
    last_start = n_samples - HEAR_WINDOW_SAMPLES
    starts = []
    for centre in centre_samples:
        centre = int(centre)
        if not 0 <= centre <= n_samples:
            raise ValueError(
                f"centre {centre} is outside the recording (0..{n_samples} samples, "
                f"{n_samples / HEAR_SAMPLING_RATE:.3f}s)"
            )
        starts.append(min(max(0, centre - HEAR_WINDOW_SAMPLES // 2), last_start))
    return starts


def span_to_hear_buffer(audio: Audio, start_s: float, end_s: float, *, placement: str = "centre") -> Audio:
    """Place one span inside a 2 s buffer containing nothing else.

    The detector's graph accepts exactly 2 s. A span shorter than that is placed in a silent buffer so the
    model sees the span and silence, never a neighbouring event. This is for the event detector only and
    must not be used to produce embeddings.

    Args:
        audio: The recording, at 16 kHz.
        start_s: Span onset.
        end_s: Span offset.
        placement: ``"centre"``, ``"start"`` or ``"end"`` — where in the buffer the span sits.

    Returns:
        Audio of exactly 2 s at the input's sampling rate.

    Raises:
        ValueError: If the span is longer than the window, or ``placement`` is not one of the three.
    """
    sr = audio.sampling_rate
    want = int(round(HEAR_WINDOW_SECONDS * sr))
    x = np.asarray(audio.waveform.detach().cpu(), dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    segment = x[int(start_s * sr) : int(end_s * sr)]
    if len(segment) > want:
        raise ValueError(
            f"span {start_s:.3f}-{end_s:.3f}s is {len(segment) / sr:.3f}s, longer than the "
            f"{HEAR_WINDOW_SECONDS:g} s the detector accepts. Split it or classify a sub-span."
        )
    offsets = {"centre": (want - len(segment)) // 2, "start": 0, "end": want - len(segment)}
    if placement not in offsets:
        raise ValueError(f"placement must be one of {sorted(offsets)}, got {placement!r}")
    buffer = np.zeros(want, dtype=np.float32)
    off = offsets[placement]
    buffer[off : off + len(segment)] = segment
    return Audio(waveform=buffer[None, :], sampling_rate=sr)


def span_hear_input(audio: Audio, extent: Tuple[float, float]) -> Audio:
    """Return an isolated candidate for the event detector, preserving a long span's own windows.

    A span shorter than :data:`HEAR_WINDOW_SECONDS` is placed in a silent 2 s buffer via
    :func:`span_to_hear_buffer`, so its only detector result describes the span itself. A longer
    span is passed through unchanged; the detector then returns one or more native windows over
    it, which :func:`hear_window_extent` places back on the source recording's own timeline.

    Args:
        audio: The recording the span was proposed over, at the detector's sampling rate.
        extent: The span's ``(start, end)`` in seconds.

    Returns:
        Audio ready to pass to the event detector.
    """
    start, end = extent
    if end - start <= HEAR_WINDOW_SECONDS:
        return span_to_hear_buffer(audio, start, end)
    start_sample = int(round(start * audio.sampling_rate))
    end_sample = int(round(end * audio.sampling_rate))
    return Audio(waveform=audio.waveform[..., start_sample:end_sample].clone(), sampling_rate=audio.sampling_rate)


def hear_window_extent(candidate_extent: Tuple[float, float], raw_window: Dict[str, Any]) -> Tuple[float, float]:
    """Place one native detector window back on the recording's own timeline.

    A short candidate is embedded in an isolated 2 s buffer (see :func:`span_hear_input`), so its
    only detector result describes the candidate itself. A long candidate is passed through
    unchanged and the detector returns one or more native windows relative to it; those windows
    must be offset to the source recording rather than all being written over the parent span.

    Args:
        candidate_extent: The span's own ``(start, end)`` in seconds.
        raw_window: One raw window the detector returned, carrying ``"start"``/``"end"`` relative
            to whatever :func:`span_hear_input` gave it.

    Returns:
        The window's ``(start, end)`` on the source recording's timeline.
    """
    start, end = candidate_extent
    if end - start <= HEAR_WINDOW_SECONDS:
        return candidate_extent
    return start + float(raw_window["start"]), start + float(raw_window["end"])


def stage_hear_snapshot() -> Tuple[str, Path]:
    """Stage the pinned ``google/hear`` commit and return ``(sha, snapshot_dir)``.

    The gated-access token comes from ``get_huggingface_token()`` — the same mechanism every
    other gated model in senselab uses. A missing or unauthorized token surfaces as
    ``huggingface_hub``'s own ``GatedRepoError``, which already names the repo and the URL to
    accept the terms at; wrapping it would only hide that.
    """
    return resolve_model(HEAR_MODEL_ID, HEAR_REVISION, token=get_huggingface_token())


def prepare_audio_for_hear(audio: Audio) -> Audio:
    """Return ``audio`` at HeAR's 16 kHz, resampling only when needed."""
    if audio.sampling_rate == HEAR_SAMPLING_RATE:
        return audio
    return resample_audios([audio], resample_rate=HEAR_SAMPLING_RATE)[0]


def write_hear_wav(path: "Path | str", audio: Audio) -> None:
    """Write ``audio`` as 32-bit-float mono WAV for the worker to read.

    ``FLOAT``, not the ``PCM_16`` ``Audio.save_to_file`` writes, for the reason measured in
    ``classification/yamnet.py``'s ``LOSSLESS_WAV_SUBTYPE``: 16-bit quantization noise is louder
    than a -100 dBFS signal and silences a -120 dBFS one outright. HeAR is amplitude-invariant
    (gains x0.1 to x10 give cosine 1.0000) but not quantization-invariant, and the sounds it is
    for — quiet breaths, throat clears — are exactly the faint end.

    Args:
        path: Destination WAV.
        audio: Audio at :data:`HEAR_SAMPLING_RATE`; a channel dimension is averaged to mono,
            since HeAR takes a single channel.
    """
    waveform = audio.waveform.detach().cpu().numpy()
    mono = waveform.mean(axis=0) if waveform.ndim > 1 else waveform
    write_audio(path, mono.astype(np.float32), audio.sampling_rate, channels_first=False)


def build_worker_payload(
    saved_model_dir: str,
    jobs: List[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    """Assemble the worker's stdin payload.

    Note what is *not* in it: no model id, no ref, no revision string. ``saved_model_dir`` is the
    staged ``snapshots/<sha>/`` path, so the commit is pinned by the path itself and the worker
    makes no Hub call at all (see the module docstring on revision pinning).

    Args:
        saved_model_dir: Directory holding ``saved_model.pb`` for this capability.
        jobs: One entry per audio: ``wav`` (input path), ``starts`` (window offsets, samples),
            ``out`` (``.npy`` destination).
        batch_size: Windows per graph call; the worker lowers it to 1 when the signature pins
            the batch dimension, as the event detectors' do.

    Returns:
        The JSON-serializable payload.
    """
    return {
        "saved_model_dir": saved_model_dir,
        "window_samples": HEAR_WINDOW_SAMPLES,
        "batch_size": batch_size,
        "jobs": jobs,
    }


def run_hear(
    audios: Sequence[Audio],
    starts_per_audio: Sequence[Sequence[int]],
    *,
    subdir: str,
    batch_size: int = 8,
    device: Optional[DeviceType] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> List[np.ndarray]:
    """Run one HeAR SavedModel over pre-planned windows, in the isolated TensorFlow venv.

    Args:
        audios: Audios **already at** :data:`HEAR_SAMPLING_RATE` (use
            :func:`prepare_audio_for_hear`).
        starts_per_audio: Window start offsets per audio, as planned by
            :func:`plan_scan_windows` or :func:`plan_centred_windows`.
        subdir: Repository-relative SavedModel directory: ``""`` for the encoder, or an entry of
            :data:`EVENT_DETECTOR_SUBDIRS`.
        batch_size: Windows per graph call. Ignored (lowered to 1) for the event detectors,
            whose signature pins the batch dimension.
        device: ``DeviceType.CPU`` hides every GPU from TensorFlow; ``DeviceType.CUDA`` lets it
            place ops itself. MPS is not a TensorFlow device — Apple GPUs need the separate
            ``tensorflow-metal`` plugin, which this venv does not install — so it is not offered.
        timeout: Seconds before the worker is killed.

    Returns:
        One array per audio: ``[n_windows, 512]`` for the encoder, ``[n_windows, 8]`` for a
        detector, in the order of ``starts_per_audio``.

    Raises:
        ValueError: If the two sequences disagree in length.
    """
    if len(audios) != len(starts_per_audio):
        raise ValueError(f"got {len(audios)} audios but {len(starts_per_audio)} window plans")
    if not audios:
        return []

    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CPU, DeviceType.CUDA]
    )
    sha, snapshot = stage_hear_snapshot()
    logger.debug("HeAR staged at %s@%s (%s)", HEAR_MODEL_ID, sha, snapshot)
    saved_model_dir = snapshot / subdir if subdir else snapshot

    venv_dir = ensure_venv(_HEAR_VENV, _HEAR_REQUIREMENTS, python_version=_HEAR_PYTHON)
    python = venv_python(venv_dir)

    with tempfile.TemporaryDirectory(prefix="senselab-hear-") as tmpdir:
        tmp = Path(tmpdir)
        jobs: List[Dict[str, Any]] = []
        for index, (audio, starts) in enumerate(zip(audios, starts_per_audio)):
            if audio.sampling_rate != HEAR_SAMPLING_RATE:
                raise ValueError(
                    f"audio {index} is at {audio.sampling_rate} Hz; HeAR needs "
                    f"{HEAR_SAMPLING_RATE} Hz (call prepare_audio_for_hear first)"
                )
            wav = tmp / f"audio_{index}.wav"
            write_hear_wav(wav, audio)
            jobs.append({"wav": str(wav), "starts": [int(s) for s in starts], "out": str(tmp / f"out_{index}.npy")})

        env = _clean_subprocess_env()
        if device_type is DeviceType.CPU:
            # TensorFlow has no per-call device argument here; hiding the GPUs is how a CPU
            # request is honoured.
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        result = subprocess.run(
            [python, "-c", _HEAR_WORKER],
            input=json.dumps(build_worker_payload(str(saved_model_dir), jobs, batch_size)),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = parse_subprocess_result(result, "HeAR")
        return [np.load(entry["out"]) for entry in output["results"]]
