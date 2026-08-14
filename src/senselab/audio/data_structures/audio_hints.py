"""Declared hints about what an ``Audio`` may contain.

A hint is an **assertion** -- by an operator, an acquisition protocol, or a corpus description --
about what a recording was *meant* to contain. It is never a measurement, and nothing in this
change consumes one: no task alters its behaviour because a hint is present. How a hint should
inform a decision is itself a decision, and it gets its own derivation when someone builds that
consumer.

This is deliberately not the same thing as dataset metadata resolved by a lookup (PR #543's
``AudioPlus``). A lookup's trust comes from the dataset; a hint's comes from whoever declared it.
Keeping them apart means hints work with no provider, no corpus, and no network.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ExpectedSpeech(BaseModel):
    """The text a speaker was asked to produce, for a read task.

    Attributes:
        text: The verbatim prompt. Present so the hint is self-contained -- a consumer can match
            a transcript against it without resolving anything.
        prompt_id: Identifier in an external reference set, e.g. ``"harvard-01"``.
        reference: Which reference set the id belongs to (name, version, or URI). Together with
            ``prompt_id`` this traces the prompt back to its corpus without vendoring that corpus
            into this repository.
    """

    text: Optional[str] = None
    prompt_id: Optional[str] = None
    reference: Optional[str] = None


class SpeakerEmbeddingProvenance(BaseModel):
    """Where a target-speaker embedding came from.

    Attributes:
        model_id: The embedding model, e.g. ``"speechbrain/spkrec-ecapa-voxceleb"``.
        model_commit_sha: The **resolved** 40-hex commit the vector was produced with, or ``None``.
            Never a ref: recording ``"main"`` here would be provenance that is confidently wrong,
            which is worse than recording none.
        unresolved_reason: Why ``model_commit_sha`` is ``None``. Required in that case, so an
            absent commit is always explained rather than merely missing.
        method: How the vector was aggregated, e.g. ``"spherical_mean"`` or
            ``"spherical_mean+dominant_cluster"`` when contamination rejection ran.
        source_files: What the estimate was computed from.
        window_s: Window length used, in seconds.
        hop_s: Hop between windows, in seconds.
        n_windows_used: Windows that contributed to the returned vector.
        n_windows_dropped: Windows excluded -- zero-norm, or removed by contamination rejection.
            Kept beside ``n_windows_used`` so a curated estimate cannot look like a clean one.
        created_at: ISO-8601 timestamp, stamped by the caller. Not defaulted to "now": a library
            that stamps wall-clock time makes its own output unreproducible.
    """

    model_id: str
    model_commit_sha: Optional[str] = None
    unresolved_reason: Optional[str] = None
    method: str = "spherical_mean"
    source_files: list[str] = Field(default_factory=list)
    window_s: Optional[float] = None
    hop_s: Optional[float] = None
    n_windows_used: int = 0
    n_windows_dropped: int = 0
    created_at: Optional[str] = None

    @field_validator("model_commit_sha")
    @classmethod
    def _must_be_a_sha(cls, v: Optional[str]) -> Optional[str]:
        """Reject anything that is not a full 40-hex commit.

        Args:
            v: The candidate value.

        Returns:
            The value unchanged when it is ``None`` or a 40-hex commit.

        Raises:
            ValueError: When the value is a ref name or a short hash. The field's whole purpose is
                to be immutable; a ref in it silently reintroduces the ambiguity it removes.
        """
        if v is None:
            return v
        if not _SHA_RE.match(v):
            raise ValueError(f"model_commit_sha must be a resolved 40-hex commit, got {v!r}")
        return v


class TargetSpeakerEmbedding(BaseModel):
    """A speaker embedding declared as the target for a recording.

    Attributes:
        vector: The embedding, unit-norm. Held inline rather than as a path to a stored artifact
            so a hint is interpretable on its own -- a reference that outlives its file is the
            dangling-pointer failure this avoids.
        provenance: Required. A vector with no provenance cannot be interpreted or reproduced.
        distribution: Optional statistics describing the set the vector was estimated from. Typed
            loosely here to keep ``data_structures`` from importing ``utils.tasks``; narrowed by
            the estimator's own signature.
    """

    vector: list[float]
    provenance: SpeakerEmbeddingProvenance
    distribution: Optional[Any] = None


class AudioHints(BaseModel):
    """What a recording was declared to contain.

    Attributes:
        may_contain: Open tags -- ``"read-speech"``, ``"cough"``, ``"music"``. Named *may* contain
            because a hint is an expectation, not an observation; nothing downstream should read
            it as ground truth. Open strings rather than an enum: a closed vocabulary here would
            be a taxonomy nobody fitted, and every corpus that did not fit it would force an edit.
            See ``speaker_embeddings/doc.md`` for a suggested, non-binding vocabulary.
        targeted_speaker_count: How many speakers the acquisition protocol aimed for -- intent,
            not a count of who is audible. A range is deliberately not modelled; it goes in
            ``metadata`` until a caller needs it, rather than shipping parallel min/max fields.
        environment: Open tag, e.g. ``"quiet-room"``, ``"clinic"``, ``"telephone"``.
        expected_speech: Ordered prompts for a read task. Ordered and separate rather than one
            concatenated string, because "which sentence was skipped" is a different question
            from "how close was the whole thing".
        target_speaker: The declared target speaker's embedding, with provenance.
        metadata: Escape hatch for corpus-specific extras that do not deserve a typed field.
    """

    may_contain: list[str] = Field(default_factory=list)
    targeted_speaker_count: Optional[int] = None
    environment: Optional[str] = None
    expected_speech: list[ExpectedSpeech] = Field(default_factory=list)
    target_speaker: Optional[TargetSpeakerEmbedding] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
