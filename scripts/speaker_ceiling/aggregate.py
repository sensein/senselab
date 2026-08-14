"""Assemble per-cell checkpoint files and per-shard corpus manifests into one profile.

A design choice, stated up front: aggregation is a **separate step**, not something each
evaluation task does on its own way out. An evaluation task only ever sees the backends and
counts it was handed -- it has no way to know whether every *other* task in the sweep also
finished, so it is the wrong place to apply either hard refusal (a short cell; zero
successes at the smallest ``k``). Those refusals need the full picture, which only exists
once every task has (or has not) reported in -- hence a dedicated pass, run once after
evaluation completes, that reads whatever is on disk and refuses to proceed around a hole
in it.

Both hard refusals below (:func:`~evaluate.check_sweep_is_complete`,
:func:`~evaluate.check_smallest_count_has_successes`) are reused unchanged from
``evaluate.py``. This module adds two more, for failure modes neither of those was written
to catch:

- **a missing shard** -- a cell checkpoint or a per-``k`` corpus manifest fragment that is
  simply absent, because the task that would have produced it was preempted and never
  restarted, never scheduled, or crashed before writing anything. A short cell (too few
  sessions) and a missing cell (no file at all) are different failures with the same wrong
  fix -- silently treating the gap as "would have passed" -- so both are refused, by the
  same standard ``scripts/calibrate_detection_margin.py`` already sets for this repo: state
  what is missing and what would fix it, and hard-error rather than emit a profile with a
  hole in it.
- **a corpus mismatch** -- a cell whose recorded corpus identity (seed, resolved TTS commit;
  see :func:`evaluate.corpus_identity_from_manifest`) disagrees with the corpus manifest
  being aggregated against. This is the check that actually enforces "every backend measured
  against byte-identical audio": without it, a difference between two backends' ceilings
  could just be a difference in what audio each one was shown, and nothing on disk would
  say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import soundfile as sf

# scripts/ is deliberately not an importable package (pyproject sets pythonpath = ["src"]).
# Put this file's own directory on sys.path instead, matching the convention every other
# module in this package already uses.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from evaluate import SessionOutcome, cell_path, corpus_identity_from_manifest, read_cell, read_cell_identity  # noqa: E402
from generate import CORPUS_SAMPLE_RATE  # noqa: E402


class MissingShardError(RuntimeError):
    """Raised when a cell checkpoint or a shard's corpus manifest fragment is absent.

    Distinct from :class:`evaluate.InsufficientMeasurementError`: that class means "some
    measurement was taken and it was not enough" (a short cell, an all-refusal backend).
    This one means "no measurement was taken at all for this piece" -- a task that never
    completed, never ran, or wrote nothing before dying. Keeping them separate means a
    caller catching one does not silently also catch the other.
    """


class CorpusValidationError(RuntimeError):
    """Raised when a corpus directory does not actually contain what its manifest claims.

    Phase 2 (evaluation) trusts an existing ``--corpus`` completely and never regenerates,
    so this is the check standing between "the manifest lists a session" and "that session's
    files are actually present, readable, and at the rate every diarizer here expects".
    """


class CorpusMismatchError(RuntimeError):
    """Raised when two cells -- or a cell and the corpus at hand -- disagree on corpus identity.

    See the module docstring's "a corpus mismatch" section: this is what actually enforces
    that every backend was scored against the same audio, rather than merely relying on the
    convention that it should have been.
    """


def load_cells(
    cells_dir: Path, backend_names: Sequence[str], counts: Sequence[int]
) -> Dict[str, Dict[int, List[SessionOutcome]]]:
    """Load every (backend, k) cell checkpoint required for a full profile.

    Reads with :func:`evaluate.read_cell`, which already degrades a missing or unparsable
    file to ``None`` -- this function is what turns that degradation into a hard refusal
    naming every absent cell at once, rather than raising on the first one found.

    Raises:
        MissingShardError: naming every (backend, k) cell whose checkpoint file is absent
            or does not parse.
    """
    missing: List[str] = []
    outcomes: Dict[str, Dict[int, List[SessionOutcome]]] = {}
    for name in backend_names:
        by_k: Dict[int, List[SessionOutcome]] = {}
        for k in counts:
            path = cell_path(cells_dir, name, k)
            cell = read_cell(path)
            if cell is None:
                missing.append(str(path))
            else:
                by_k[k] = cell
        outcomes[name] = by_k

    if missing:
        raise MissingShardError(
            f"refusing to aggregate: {len(missing)} cell checkpoint(s) are missing or unparsable -- "
            f"{', '.join(missing)}. Each is written by evaluate_backend (via "
            "probe_speaker_ceilings.py --mode evaluate) as soon as that (backend, k) cell completes; "
            "a missing one means that task never got that far -- preempted and never requeued, never "
            "scheduled, or it crashed before writing anything. Re-run evaluation for the missing "
            "backend/k combination(s) against the same --corpus before aggregating; assembling a "
            "profile around a hole would silently understate coverage instead of reporting it."
        )
    return outcomes


def check_cells_share_one_corpus(
    cells_dir: Path,
    backend_names: Sequence[str],
    counts: Sequence[int],
    manifest: Mapping[str, object],
) -> None:
    """Refuse if any cell's recorded corpus identity disagrees with ``manifest``'s.

    Called after :func:`load_cells` has already confirmed every cell exists and parses.
    A cell written without an identity at all (``{}`` -- a test, or a checkpoint predating
    this field) is not treated as evidence of a mismatch, since it was never asked to record
    one; see :func:`evaluate.write_cell`'s docstring for the same distinction.

    Raises:
        CorpusMismatchError: naming the first disagreeing cell and both identities.
    """
    expected = corpus_identity_from_manifest(manifest)
    for name in backend_names:
        for k in counts:
            path = cell_path(cells_dir, name, k)
            identity = read_cell_identity(path)
            if identity and identity != expected:
                raise CorpusMismatchError(
                    f"refusing to aggregate: cell {path} was evaluated against a different corpus "
                    f"({identity}) than the corpus manifest being aggregated ({expected}). Every cell "
                    "must be evaluated against the same --corpus, or a difference between backends' "
                    "ceilings could just be a difference in what audio each one was shown. Re-run "
                    "evaluation for the mismatched cell against the correct --corpus."
                )


def validate_corpus_for_counts(corpus_dir: Path, manifest: Mapping[str, object], counts: Sequence[int]) -> None:
    """Refuse if the corpus at ``corpus_dir`` does not actually contain what ``manifest`` claims.

    Phase 2 evaluation never regenerates, so this check is what stands between "the manifest
    lists a session" and "that session actually diarizes something real": every
    ``k=<k>/session_<i>.wav`` and its sibling ``.rttm`` (for ``k`` in ``counts``) must exist,
    the wav must open, and it must be at :data:`generate.CORPUS_SAMPLE_RATE` -- the rate
    every diarizer here expects (see ``generate.py``'s module docstring for the measured
    failure a wrong rate causes). Silently evaluating fewer or wrong-rate sessions would
    defeat :func:`evaluate.check_sweep_is_complete` downstream: from cells alone, a short
    corpus and a short measurement look identical.

    Raises:
        CorpusValidationError: naming every missing/unreadable/off-rate file, and every
            requested ``k`` with no sessions recorded for it at all.
    """
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise CorpusValidationError(f"corpus manifest under {corpus_dir} has no 'sessions' list")

    problems: List[str] = []
    seen_k: set = set()
    for record in sessions:
        k = int(record["k"])  # type: ignore[call-overload]
        if k not in counts:
            continue
        seen_k.add(k)
        wav_path = corpus_dir / str(record["wav"])  # type: ignore[index]
        rttm_path = corpus_dir / str(record["rttm"])  # type: ignore[index]
        if not wav_path.exists():
            problems.append(f"missing wav: {wav_path}")
            continue
        if not rttm_path.exists():
            problems.append(f"missing rttm: {rttm_path}")
            continue
        try:
            info = sf.info(str(wav_path))
        except Exception as exc:  # noqa: BLE001 -- any unreadable wav is a validation failure to report, not raise past
            problems.append(f"unreadable wav {wav_path}: {exc}")
            continue
        if info.samplerate != CORPUS_SAMPLE_RATE:
            problems.append(f"{wav_path} is {info.samplerate} Hz, expected {CORPUS_SAMPLE_RATE}")

    missing_k = sorted(set(counts) - seen_k)
    if missing_k:
        problems.insert(0, f"manifest has no sessions at all for k={missing_k}")

    if problems:
        raise CorpusValidationError(
            f"refusing to evaluate: corpus at {corpus_dir} is incomplete or malformed for k={list(counts)} "
            f"-- {'; '.join(problems)}. Phase 2 never regenerates -- re-run generation "
            "(scripts/probe_speaker_ceilings.py --mode generate) for the affected k before evaluating."
        )


def _validate_consistent_fragment_metadata(fragments: Mapping[int, dict]) -> None:
    """Refuse if per-shard manifest fragments disagree on anything but their own sessions.

    Every shard is generated by the same top-level ``generate_corpus`` call parameters
    (method, tts_model, session_params, seed) and differs only in which ``k`` it covers --
    so a mismatch here means someone pointed aggregation at fragments from two different
    runs (different seeds, different corpus git states), which would silently produce a
    profile whose corpus provenance describes no single actual corpus.
    """
    shared_keys = ("method", "tts_model", "session_params", "seed")
    reference_k = next(iter(fragments))
    reference = {key: fragments[reference_k].get(key) for key in shared_keys}
    for k, fragment in fragments.items():
        current = {key: fragment.get(key) for key in shared_keys}
        if current != reference:
            raise MissingShardError(
                f"refusing to aggregate: manifest fragment for k={k} disagrees with k={reference_k} on "
                f"{shared_keys} -- these fragments were not written by the same generate_corpus run "
                "(different seed, different corpus, or a stale leftover from an earlier attempt). "
                "Aggregating them would produce a profile whose corpus_manifest describes no single "
                "actual corpus."
            )


def merge_corpus_manifests(corpus_dir: Path, counts: Sequence[int]) -> dict:
    """Return one manifest describing the full corpus, from either a single file or shards.

    Tries the unsharded case first: if ``corpus_dir/manifest.json`` already covers every
    requested ``k`` (the shape a non-sharded, single-process run produces), it is returned
    as-is. Otherwise assembles from per-shard fragments named ``manifest.k<k>.json`` (the
    shape :func:`generate.generate_corpus`'s ``manifest_name`` parameter lets a sharded
    caller write without shards clobbering each other) -- refusing if any expected fragment
    is absent, by the same missing-shard standard as :func:`load_cells`.

    Raises:
        MissingShardError: naming every ``k`` whose manifest fragment is absent, or if the
            present fragments disagree on shared metadata (see
            :func:`_validate_consistent_fragment_metadata`).
    """
    full_path = corpus_dir / "manifest.json"
    if full_path.exists():
        manifest = json.loads(full_path.read_text())
        if set(manifest.get("counts", [])) >= set(counts):
            return manifest

    fragments: Dict[int, dict] = {}
    missing_k: List[int] = []
    for k in counts:
        fragment_path = corpus_dir / f"manifest.k{k}.json"
        if fragment_path.exists():
            fragments[k] = json.loads(fragment_path.read_text())
        else:
            missing_k.append(k)

    if missing_k:
        raise MissingShardError(
            f"refusing to aggregate: no corpus manifest for k={missing_k} under {corpus_dir} -- "
            "expected either a single manifest.json covering every requested k, or a "
            f"manifest.k<k>.json fragment per k (written by generate_corpus's manifest_name "
            "parameter when run per-shard). Re-run generation for the missing k(s) with the same "
            "--seed before aggregating."
        )

    _validate_consistent_fragment_metadata(fragments)

    merged = dict(fragments[counts[0]])
    merged["counts"] = sorted(fragments)
    merged["sessions"] = [record for k in sorted(fragments) for record in fragments[k]["sessions"]]
    return merged
