"""Utterance axis vote harvester — "what was said?".

**One voter, one question.** The axis is a resampling of fused word accuracy onto the shared time
grid: the recognizers' words are folded once per pass (``_consensus_word_doubt``), and each bucket
takes the coverage-weighted mean of ``1 - existence_confidence`` over the words reaching it
(``resample_word_doubt``).

Four things used to ride on every bucket beside it, and all four are gone:

- **per-bucket text** (``asr_text_in_window`` with ``fully_contained=True``), a reconstruction of
  what ``final/transcript.json`` already holds at word resolution. It is also what forced the
  1.0 s / 0.5 s grid: with a bucket narrower than a word, a fully-contained read returns nothing,
  so the grid had to be widened and overlapped until words fit. With the derivative as the voter
  that reason is gone, and the axis sits on ``axes.DEFAULT_TIME_GRID`` like the other three.
- **the pairwise phoneme distance** between recognizers, which was already recorded rather than
  scored (D-21 rule 6: its source closure is a subset of the consensus fold's, so counting both
  counts one body of evidence twice). Recorded-and-never-read is not a middle ground; the
  readable form of "which pair diverged" is the transcript's own ``alternates``.
- **``avg_logprob`` / ``token_entropy`` / ``alignment_ctc_score``**, three per-bucket reads that
  no longer reach a fold. The first two are a model's private doubt about a *transcript*, which
  the consensus fold already weighs per word; the third measures an aligner's path posterior given
  a possibly-hallucinated transcript, which was never scored for exactly that reason.

The word-level fields — ``existence_confidence``, ``onset_confidence``, ``offset_confidence`` —
are where localisation and per-edge doubt live. See :func:`resample_word_doubt` for why they are
deliberately not folded into this axis's number.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    _levenshtein,
    resolve_asr_result,
)


def aligned_columns(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[list[dict[str, Any]]] | None:
    """Group the recognizers' words by **sequence alignment**, one list per aligned position.

    Time-overlap grouping cannot represent an insertion, and the consequence is not a lost filler
    but a corrupted transcript. Measured on three recognizers reading "I uh think" where only
    CrisperWhisper emits the filler: the filler overlaps *"think"* in the other two, joins that
    group, loses the vote 2-to-1 and is dropped — and CrisperWhisper's own "think" cannot rejoin the
    group its model already occupies, so it forms a second group and "think" is emitted twice.

    Capturing disfluencies is the point, not a side effect: a filler one model heard and others
    discarded is evidence about the speaker, and the axis should record that the recognizers
    disagreed about it rather than quietly picking the majority reading of a different word.

    ``harmonize_transcripts`` already does this alignment — star-shaped against the median-length
    transcript, with insertions keyed between reference positions — and had no caller outside its
    own tests. Returns ``None`` when the alignment cannot be formed, so the caller falls back to
    time-overlap grouping rather than losing every word.
    """
    from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts

    by_model = {
        model: [(float(w["start"]), float(w["end"]), str(w["text"])) for w in words]
        for model, words in streams.items()
        if words
    }
    if len(by_model) < 2:
        return None
    try:
        harmonised = harmonize_transcripts(by_model)
    except Exception:  # noqa: BLE001 — a fold must not fail on an alignment edge case
        return None
    if not harmonised.slots:
        return None

    # Rebuild full word dicts per column by **index**, which is the word's identity. The fold needs
    # the rest of each word — confidence, corroboration, timing provenance — to weigh it and to
    # measure boundary agreement, and the lattice carries only surface form and span.
    #
    # Matching by onset instead looked equivalent and was not: forced aligners emit words sharing an
    # onset and words of zero duration (measured on the 5-speaker clip: "Josh" at [2.72, 2.72], two
    # words at 2.72), so an onset does not identify a word. That lookup put one word in two columns
    # and dropped another, turning "wanted to take" into "wanted take take" — a corruption of the
    # transcript introduced by the grouping meant to protect it.
    ordered = {model: list(words) for model, words in streams.items()}
    columns: list[list[dict[str, Any]]] = []
    for slot in harmonised.slots:
        members: list[dict[str, Any]] = []
        for model, position in (slot.indices or {}).items():
            if position is None:
                continue
            words_of_model = ordered.get(model) or []
            if not 0 <= int(position) < len(words_of_model):
                continue
            members.append({**dict(words_of_model[int(position)]), "model": model})
        if members:
            columns.append(members)
    return columns or None


def phoneme_similarity(a: str, b: str) -> float:
    """How close two words sound, in ``[0, 1]`` — 1.0 identical, 0.0 sharing no phoneme.

    Supplied to the ensemble so word accuracy grades its disagreements instead of counting exact
    matches. The task API stays stdlib-only and receives this as a callable, the same way it
    receives ``calibrator`` and ``speaker_at``: g2p is a workflow dependency and does not belong
    inside a model-independent voting routine.

    ARPAbet with stress markers stripped, so ``AH0`` and ``AH1`` are one phoneme — stress is not a
    lexical difference and counting it would penalise two recognizers that agree on the word.

    Falls back to **exact match** when g2p is unavailable or produces nothing for either side, not
    to grapheme overlap: letters are not sounds, and substituting one measure for the other would
    change the number's meaning invisibly. A homophone pair therefore scores 1.0 where g2p works
    and 0.0 where it does not, which is a real limitation and better than an unrecorded proxy.
    """
    if a == b:
        return 1.0
    try:
        from senselab.audio.workflows.audio_analysis.harvesters import g2p_phonemes, normalize_arpabet

        seq_a = [normalize_arpabet(p) for p in g2p_phonemes(a) if str(p).strip()]
        seq_b = [normalize_arpabet(p) for p in g2p_phonemes(b) if str(p).strip()]
    except Exception:  # noqa: BLE001 — a missing g2p must not fail a fold
        return 0.0
    if not seq_a or not seq_b:
        return 0.0
    denominator = max(len(seq_a), len(seq_b))
    return max(0.0, 1.0 - _levenshtein(seq_a, seq_b) / denominator)


_WORD_FIELDS = ("text", "start", "end", "score", "chunks", "timestamp_source", "timestamp_model")
"""The fields ``iter_word_leaves`` reads. Named so :func:`_as_plain` converts exactly them."""


def _as_plain(node: Any) -> Any:  # noqa: ANN401 — ScriptLine tree, its JSON form, or a duck-type
    """Normalise a transcript tree to the dict/list form the word walker understands.

    ``resolve_asr_result`` hands back ``ScriptLine`` *objects* from a live backend and dicts from
    the cache, and ``iter_word_leaves`` walks dicts only. Without this the fold silently found no
    words and the asr axis came out with zero contributing signals on a real run — the failure
    every unit test here missed, because they all construct dicts.

    The duck-typed branch is the same lesson applied one step further. ``harvesters`` is deliberately
    shape-tolerant (``seg_attr`` reads an attribute or a key), and the harvesters this fold replaced
    were too — so a shape that is neither a dict nor a Pydantic model used to work here and stopped
    working, silently and with an empty axis as the only symptom. Anything exposing ``text`` is
    converted field by field rather than rejected.
    """
    if isinstance(node, list):
        return [_as_plain(item) for item in node]
    if isinstance(node, dict):
        return {**node, "chunks": _as_plain(node["chunks"])} if isinstance(node.get("chunks"), list) else node
    dump = getattr(node, "model_dump", None)
    if callable(dump):
        return dump()
    if not hasattr(node, "text"):
        return node
    return {field: _as_plain(getattr(node, field, None)) for field in _WORD_FIELDS}


def _warn_if_grading_is_out_of_language(asr_resolved: Mapping[str, Any]) -> list[str]:
    """Warn when the transcripts are not English, because :func:`phoneme_similarity` is.

    ``g2p_en`` maps English text to ARPAbet and nothing else, so on a non-English transcript the
    word-accuracy grading degrades to an edit distance over pseudo-phonemes derived from English
    letter-to-sound rules. That is still applied symmetrically to both sides, so it is usable
    rather than meaningless — but it is not the measure the docstring promises, and a run has to
    say so. Whisper exposes ``language`` per segment when detection ran; any model's explicit
    ``language`` is honoured, and English is assumed otherwise.

    Returns:
        The detected two-letter language codes, sorted — recorded on the fold's provenance so the
        caveat travels with the number rather than living only in a stderr line.
    """
    languages: set[str] = set()
    for resolved in asr_resolved.values():
        items = resolved if isinstance(resolved, list) else [resolved]
        for line in items:
            language = getattr(line, "language", None)
            if language is None and isinstance(line, dict):
                language = line.get("language")
            if language:
                languages.add(str(language).lower()[:2])
    if languages and "en" not in languages:
        print(
            f"warn: asr word-accuracy grading is g2p_en-based and the transcript language(s) are "
            f"{sorted(languages)}; phoneme_similarity degrades to an edit distance over "
            "English-rule pseudo-phonemes (recorded as grading_languages on every asr row)",
            file=sys.stderr,
        )
    return sorted(languages)


def _consensus_word_doubt(
    asr_resolved: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
) -> tuple[dict[tuple[float, float], float | None], dict[str, Any]]:
    """Fold the recognizers' words once, then resample the result onto the grid (D-27).

    Returns ``(doubt_by_bucket, provenance)``. The provenance travels onto every row because the
    fold's parameters *are* its policy: a derivative whose choices are not in the artifact is the
    default argument this design keeps removing (D-21 rule 4).

    The slot parameters are the task API's own defaults, named here rather than inherited silently
    so the recorded value and the used value cannot drift.
    """
    from senselab.audio.tasks.speech_to_text_ensemble import fuse_word_streams, iter_word_leaves

    slot_overlap, slot_mid_tol_s = 0.3, 0.15
    streams: dict[str, list[dict[str, Any]]] = {}
    unreadable: list[str] = []
    for model_id, resolved in asr_resolved.items():
        words = iter_word_leaves(_as_plain(resolved))
        if words:
            streams[str(model_id)] = words
        else:
            unreadable.append(str(model_id))
    if unreadable:
        # Loud, because the quiet version already cost a wrong artifact: a shape the walker could not
        # read produced an asr axis with zero contributing signals over a whole recording, and every
        # unit test passed. "This model transcribed nothing" and "this model's shape was unreadable"
        # are indistinguishable downstream, so the distinction has to be made here.
        print(
            f"warn: asr fold extracted no words from {sorted(unreadable)} — either the model produced "
            "no transcript, or its result shape is one `_as_plain` could not convert (the axis then "
            "reports nothing for it, which is not the same as reporting no doubt)",
            file=sys.stderr,
        )
    if not streams:
        return ({b: None for b in buckets}, {})

    grading_languages = _warn_if_grading_is_out_of_language(asr_resolved)
    fused = fuse_word_streams(
        streams,
        slot_overlap=slot_overlap,
        slot_mid_tol_s=slot_mid_tol_s,
        text_similarity=phoneme_similarity,
        columns=aligned_columns(streams),
    )
    counts = sorted({int(w["timing_sources"]) for w in fused if w.get("timing_sources") is not None})
    provenance = {
        "operator": "consensus_words/resample",
        "sources": sorted(streams),
        "n_words": len(fused),
        "slot_overlap": slot_overlap,
        "slot_mid_tol_s": slot_mid_tol_s,
        # Which language the grading was performed *in*, not which language was spoken: it decides
        # whether ``phoneme_similarity`` measured phonemes or English-rule pseudo-phonemes.
        "grading_languages": grading_languages,
        # How many *independent* timing opinions the words had — an int when every word had the
        # same number, the sorted set otherwise. Two recognizers sharing an aligner count as one,
        # so this is routinely lower than ``len(sources)`` and the row has to say so rather than
        # leaving a reader to infer independence from the model count.
        "timing_sources": (counts[0] if len(counts) == 1 else counts) if counts else None,
    }
    return resample_word_doubt(fused, buckets), provenance


def resample_word_doubt(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Project **word-sequence accuracy** onto a time grid — the asr axis (D-27, revised).

    One question, one axis: how much do the recognizers disagree about *what words were said*. Each
    word contributes doubt ``1 - existence_confidence`` over its own span, and nothing else enters.

    **Temporal agreement is deliberately excluded** — from the mass and from the reach. It is not
    that localisation does not matter; it is that a single number cannot carry both and stay
    readable, and two attempts proved it. Bucketed pairwise WER made timing jitter *look* like
    textual disagreement, reporting an axis mean of 0.4266 on a pair of word-identical transcripts.
    Replacing that with a joint of accuracy × localisation fixed the conflation but not the
    legibility: 0.788 could mean either half, and a reader had no way to tell which. Localisation
    now lives on the word, split per edge (``onset_confidence`` / ``offset_confidence``), where the
    figure can show *which* boundary is in doubt instead of averaging it into a score.

    The cost, stated because it is real: on a run where every recognizer agrees, this axis is near
    zero — and that is the honest answer, since there is no disagreement about what was said. A
    poorly localised word no longer registers here at all, which is why the word-level fields and
    the figure's onset/offset marks are the place that question is answered.

    Args:
        words: Fused words carrying ``start``, ``end`` and ``existence_confidence``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → doubt}``, the coverage-weighted mean of the doubt reaching it. ``None`` where no
        word reaches at all: a bucket nothing was said in is unmeasured, and reporting ``0.0``
        there would assert that we are certain nothing was said — a claim of a different kind.
    """
    contributions: dict[tuple[float, float], list[tuple[float, float]]] = {b: [] for b in buckets}
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        accuracy = word.get("existence_confidence")
        doubt = 1.0 - float(accuracy) if isinstance(accuracy, (int, float)) else 1.0
        # Accuracy only, and the word's own span only. Temporal agreement is deliberately **not**
        # here — not in the mass and not in the reach — so this axis answers one question: how much
        # do the recognizers disagree about the word sequence. Mixing localisation in is what made
        # the axis unreadable twice, first as bucketed-WER disagreement that was really timing jitter
        # and then as a joint whose two halves could not be told apart in the number. The temporal
        # halves live on the word (``onset_confidence`` / ``offset_confidence``) where a reader can
        # see which edge is in doubt.
        lo, hi = start, end
        for bucket in buckets:
            overlap = min(hi, bucket[1]) - max(lo, bucket[0])
            if overlap > 0:
                contributions[bucket].append((overlap, max(0.0, min(1.0, doubt))))

    out: dict[tuple[float, float], float | None] = {}
    for bucket, reaching in contributions.items():
        if not reaching:
            out[bucket] = None
            continue
        total = sum(w for w, _ in reaching)
        out[bucket] = sum(w * d for w, d in reaching) / total if total > 0 else None
    return out


def harvest_asr_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    alignment_by_model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the asr axis.

    ``votes`` holds at most one entry, ``consensus_words``: ``{"value": doubt, **provenance}``,
    where ``doubt`` is the fused word accuracy resampled onto this bucket. A bucket no word reaches
    carries **no vote at all** rather than ``0.0`` — nothing was said there, which is not the same
    as nothing being in doubt, and zero-filling would manufacture confidence (FR-007).

    There is deliberately no per-model entry. A recognizer's own reading of a bucket is already
    inside the fold, weighted per word by how far the others corroborate it; emitting it again
    beside the fold would count one body of evidence twice (D-21 rule 6), and it was the
    fully-contained per-bucket text read that forced this axis onto a grid of its own.
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    asr_blocks = (pass_summary.get("asr") or {}).get("by_model") or {}
    asr_ok = {m: b for m, b in asr_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    asr_resolved = {m: resolve_asr_result(b, alignment_by_model.get(m)) for m, b in asr_ok.items()}

    # One fold per pass, then resampled per bucket — not recomputed per bucket, and deliberately
    # not "which words fell fully inside this bucket". Reach is the word's own span, so a word
    # straddling a grid line no longer reads as two models disagreeing.
    buckets = [(round(float(s), 6), round(float(e), 6)) for s, e, _ in grid.iter_buckets(duration_s)]
    word_doubt, word_doubt_provenance = _consensus_word_doubt(asr_resolved, buckets)

    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, dict[str, Any]] = {}
        doubt = word_doubt.get((round(float(start), 6), round(float(end), 6)))
        if doubt is not None:
            votes["consensus_words"] = {"value": doubt, **word_doubt_provenance}
        out.append({"start": start, "end": end, "votes": votes})
    return out
