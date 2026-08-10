"""Run every diarization backend over a generated corpus and record what it predicted.

Three outcomes, not two
------------------------
A backend can be *right* (predicted the true count), *wrong* (predicted a different
count), or *refused* (raised before producing a count at all). ``derive.py``'s
``exact_count_accuracy`` already treats ``None`` as "not correct" without conflating it
with a wrong number in the confusion this module builds — feeding it a fabricated ``0``
or a guessed count for a refusal would erase that distinction and make a crash look like
a bad answer. :func:`run_session` therefore records ``None`` *and* the exception's type
name on any failure, so a reader of the emitted profile can tell "child-adult refused
because the clip was under its 10 s window" from "child-adult crashed for an unrelated
reason" — both currently collapse to the same ``None`` in the accuracy curve, but only
one of them is informative about the backend's counting ability.

Role labels do not inflate the count
-------------------------------------
The USC-SAIL child-adult backend emits a literal ``"OVERLAP"`` label for frames it can't
assign to a single talker. ``capabilities.py`` already documents that this marks two
*known* talkers speaking at once, not a third one, and declares ``max_speakers=2``
accordingly (see ``child_adult.py``'s ``CAPABILITIES``). Naively counting distinct
``speaker`` values would treat ``{CHILD, ADULT, OVERLAP}`` as three speakers and
penalize this backend for its own documented labeling convention rather than for a real
miscount, so :func:`_count_speakers` drops ``"OVERLAP"`` for any backend whose
``speaker_label_kind`` is ``"role"`` before counting. This is a deliberate refinement of
the brief's literal "count distinct speaker values" — the brief did not anticipate this
label, and applying it naively would corrupt exactly the cell (child-adult, k<=2) the
probe most needs to get right.

Construction happens inside the per-session try, matching
``scripts/run_diarization_backends.py``'s ``run_one``: a gated repo or an unmet CUDA
requirement can raise at ``SenselabModel`` construction rather than inside
``diarize_audios``, and either way one backend's failure on one session must not abort
the sweep for every other backend or count.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization.api import capabilities_for
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel

# scripts/ is deliberately not an importable package (pyproject sets
# pythonpath = ["src"]), so a plain `from scripts.speaker_ceiling.derive import ...`
# would raise ModuleNotFoundError both under pytest and under `uv run python
# evaluate.py` directly. Put this file's own directory on sys.path instead --
# derive.py lives right next to it -- rather than duplicating exact_count_accuracy's
# logic here.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from derive import exact_count_accuracy  # noqa: E402


@dataclass(frozen=True)
class BackendSpec:
    """One diarization backend this probe knows how to construct and dispatch.

    ``kind`` decides which ``SenselabModel`` subclass :func:`_build_model` wraps
    ``model_id`` in -- the same distinction ``run_diarization_backends.py`` and
    ``model_for_task`` make, restated here rather than imported because neither
    exposes it as a small lookup table.
    """

    name: str
    model_id: str
    kind: str  # "pyannote" or "hf"


# All six backends `diarize_audios` dispatches to (see api.py's module docstring).
# child-adult is CUDA-only and sortformer needs the nemo-diarization subprocess venv;
# both are included here so a caller with real hardware can select them, but neither
# should be selected for the CPU-only, no-venv smoke test this module's tests exercise.
ALL_BACKENDS: Sequence[BackendSpec] = (
    BackendSpec("pyannote", "pyannote/speaker-diarization-community-1", "pyannote"),
    BackendSpec("sortformer", "nvidia/diar_sortformer_4spk-v1", "hf"),
    BackendSpec("vibevoice", "microsoft/VibeVoice-ASR-HF", "hf"),
    BackendSpec("child_adult", "AlexXu811/whisper-child-adult", "hf"),
    BackendSpec("moss", "OpenMOSS-Team/MOSS-Transcribe-Diarize", "hf"),
    BackendSpec("diarizen", "BUT-FIT/diarizen-wavlm-large-s80-md", "hf"),
)

BACKENDS_BY_NAME: Mapping[str, BackendSpec] = {b.name: b for b in ALL_BACKENDS}


@dataclass(frozen=True)
class SessionOutcome:
    """What one backend produced (or failed to produce) for one session file.

    Attributes:
        session: The session's file stem (e.g. ``"session_3"``), for error messages only.
        predicted: The distinct-speaker count the backend reported, or ``None`` if it
            raised before producing one -- a refusal, never a fabricated 0 or a guess.
        error_type: The exception's class name when ``predicted`` is ``None``; ``None``
            on success. Kept distinct from ``error_message`` so a caller can group by
            failure kind (e.g. "how many sessions failed with ValueError") without
            parsing free text.
        error_message: The exception's ``str()``, truncated, for a human reading the
            profile by hand. Not meant for programmatic matching -- use ``error_type``.
        elapsed_s: Wall-clock seconds for this one session, model construction included.
    """

    session: str
    predicted: Optional[int]
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_s: float = 0.0


def _build_model(backend: BackendSpec) -> SenselabModel:
    """Construct the ``SenselabModel`` for ``backend``.

    Left un-cached and re-called for every session on purpose, mirroring
    ``run_diarization_backends.py``'s ``run_one``: a gated repo raises here every time,
    identically, which is the correct behavior for a backend the sweep cannot reach --
    each of its sessions should record the same refusal rather than the sweep aborting
    after the first attempt.
    """
    if backend.kind == "pyannote":
        return PyannoteAudioModel(path_or_uri=backend.model_id)
    return HFModel(path_or_uri=backend.model_id)


def _count_speakers(lines: Sequence[ScriptLine], capabilities: DiarizationCapabilities) -> int:
    """Count distinguishable speakers in one session's diarization result.

    Excludes the literal ``"OVERLAP"`` label for role-kind backends -- see the module
    docstring's "Role labels do not inflate the count" section for why a naive distinct
    count would misrepresent the one backend this applies to.
    """
    labels = {line.speaker for line in lines if line.speaker}
    if capabilities.speaker_label_kind == "role":
        labels.discard("OVERLAP")
    return len(labels)


def run_session(backend: BackendSpec, wav_path: Path, device: Optional[DeviceType]) -> SessionOutcome:
    """Diarize one session file with one backend, turning any exception into a refusal.

    Never raises: a refusal (child-adult's ``ValueError`` under its 10 s window, its
    CUDA requirement, a gated repo without a token) is a result this probe wants to see,
    not a reason to abort the sweep for every other session or backend.
    """
    t0 = time.time()
    try:
        model = _build_model(backend)
        audio = Audio(filepath=str(wav_path))
        results = diarize_audios(audios=[audio], model=model, device=device)
        lines = results[0] if results else []
        predicted = _count_speakers(lines, capabilities_for(backend.model_id))
        return SessionOutcome(session=wav_path.stem, predicted=predicted, elapsed_s=time.time() - t0)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a measurement, not a crash to propagate
        return SessionOutcome(
            session=wav_path.stem,
            predicted=None,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            elapsed_s=time.time() - t0,
        )


def evaluate_backend(
    backend: BackendSpec,
    corpus_dir: Path,
    manifest: Mapping[str, object],
    counts: Sequence[int],
    device: Optional[DeviceType],
) -> Dict[int, List[SessionOutcome]]:
    """Run ``backend`` over every session recorded in ``manifest`` at each ``k`` in ``counts``.

    Reads session file paths from ``manifest["sessions"]`` (as written by
    :func:`~scripts.speaker_ceiling.generate.generate_corpus`) rather than globbing the
    corpus directory, so a session the manifest does not know about is never evaluated
    and a ``k`` with no matching sessions comes back as an empty (not missing) list --
    which is exactly the shape :func:`check_sweep_is_complete` needs to catch it.
    """
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError(f"manifest at {corpus_dir} has no 'sessions' list -- was it written by generate_corpus?")

    by_k: Dict[int, List[dict]] = defaultdict(list)
    for record in sessions:
        k = int(record["k"])  # type: ignore[call-overload]
        if k in counts:
            by_k[k].append(record)  # type: ignore[arg-type]

    outcomes: Dict[int, List[SessionOutcome]] = {}
    for k in counts:
        records = sorted(by_k.get(k, []), key=lambda r: r["session_index"])
        outcomes[k] = [run_session(backend, corpus_dir / str(record["wav"]), device) for record in records]
    return outcomes


def confusion_from_outcomes(outcomes: Sequence[SessionOutcome]) -> Dict[str, int]:
    """Tally predicted counts for one (backend, k) cell, keyed by predicted value or ``"refused"``.

    This is the full confusion the spec asks for, not just the accuracy verdict: a
    reader who disagrees with ``derive.py``'s 80% threshold can recompute a different
    verdict straight from these counts without re-running a single GPU session.
    """
    counts: "Counter[str]" = Counter()
    for outcome in outcomes:
        counts["refused" if outcome.predicted is None else str(outcome.predicted)] += 1
    return dict(counts)


def refusal_reasons_from_outcomes(outcomes: Sequence[SessionOutcome]) -> Dict[str, int]:
    """Tally exception types among the refusals in one (backend, k) cell.

    Kept separate from :func:`confusion_from_outcomes` because the spec's confusion
    schema is ``{predicted_or_"refused": n}`` -- flattening the exception type into that
    key would break the schema a reader expects, but the type is exactly what
    distinguishes "refused because the clip was too short" from "crashed for an
    unrelated reason", so it is recorded alongside instead of discarded.
    """
    return dict(Counter(o.error_type for o in outcomes if o.error_type is not None))


def curve_from_outcomes(outcomes_by_k: Mapping[int, Sequence[SessionOutcome]]) -> Dict[int, float]:
    """Reduce every (backend, k) cell's outcomes to one exact-count accuracy per ``k``."""
    return {k: exact_count_accuracy([o.predicted for o in outcomes], true_k=k) for k, outcomes in outcomes_by_k.items()}


class InsufficientMeasurementError(RuntimeError):
    """Raised when a sweep does not have enough measurement to derive a trustworthy profile."""


def check_sweep_is_complete(
    outcomes: Mapping[str, Mapping[int, Sequence[SessionOutcome]]],
    sessions_per_count: int,
) -> None:
    """Refuse if any (backend, k) cell has fewer than ``sessions_per_count`` recorded outcomes.

    A short cell most often means the corpus does not actually contain as many sessions
    as the caller asked this sweep to require (e.g. generated with ``--sessions 2`` but
    evaluated expecting ``--sessions 3``) -- a gap here is a measurement gap, and
    :func:`derive.derive_ceiling` cannot tell a gap from "the missing sessions would
    have passed". Following ``scripts/calibrate_detection_margin.py``'s posture: state
    what was insufficient and what would fix it, and hard-error rather than warn.

    Raises:
        InsufficientMeasurementError: naming every short (backend, k) cell and how many
            sessions it has versus how many were required.
    """
    short = [
        (backend_name, k, len(sessions))
        for backend_name, by_k in outcomes.items()
        for k, sessions in by_k.items()
        if len(sessions) < sessions_per_count
    ]
    if short:
        detail = "; ".join(f"{backend} at k={k}: {n}/{sessions_per_count} sessions" for backend, k, n in short)
        raise InsufficientMeasurementError(
            f"refusing to emit a profile: {len(short)} (backend, k) cell(s) have fewer than the "
            f"required {sessions_per_count} completed sessions -- {detail}. Generate more sessions "
            "for these counts with scripts/speaker_ceiling/generate.py (--counts ... --sessions "
            f"{sessions_per_count}) before re-running the sweep; a short cell would silently "
            "understate that backend's measured accuracy at that count."
        )


def check_smallest_count_has_successes(
    outcomes: Mapping[str, Mapping[int, Sequence[SessionOutcome]]],
    smallest_k: int,
) -> None:
    """Refuse if any backend produced zero successful sessions at ``smallest_k``.

    Zero successes at the smallest swept count means the probe measured the harness --
    a broken model id, a missing HF token, an unmet CUDA requirement -- not the
    backend's counting ability, and every larger ``k`` for that backend would silently
    inherit the same failure without saying so.

    Raises:
        InsufficientMeasurementError: naming every backend that failed every session at
            ``smallest_k``.
    """
    broken = [
        backend_name
        for backend_name, by_k in outcomes.items()
        if by_k.get(smallest_k) and all(o.predicted is None for o in by_k[smallest_k])
    ]
    if broken:
        names = ", ".join(sorted(broken))
        raise InsufficientMeasurementError(
            f"refusing to emit a profile: {names} produced zero successful sessions at "
            f"k={smallest_k}, the smallest count swept. That measures the harness -- a broken "
            "model id, a missing HF_TOKEN, an unmet CUDA requirement -- not the backend's "
            "counting ability. Check each failing session's recorded error_type and fix "
            "whatever raised there before trusting any larger k for this backend."
        )
