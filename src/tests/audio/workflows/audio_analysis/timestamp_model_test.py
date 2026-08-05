"""Word timings carry the identity of whatever produced them, not just its kind.

``TimestampSource`` is ``native | bundled_aligner | external_aligner`` — a *kind*, and a kind cannot
answer the question the asr axis asks. Qwen3-ASR's word times come from
``Qwen/Qwen3-ForcedAligner-0.6B`` shipped with it (``bundled_aligner``); Canary-Qwen carries no
timings, so ``stage_alignment`` aligns it with **the same model** (``external_aligner``). Two labels,
one aligner — and on ``english_conversation_higgs_audio_v2_20260805-034348`` their onsets came out
bit-identical across all 62 words while both differed from CrisperWhisper by the same 0.032 s mean.

So the grouping key has to be the producing model's id. These tests pin that the producers stamp it,
because the consumer (``_temporal_agreement``) is inert without it and inert in the silent
direction: it would report two independent timing sources and full temporal confidence.
"""

from __future__ import annotations

import ast
import pathlib

from senselab.utils.data_structures import ScriptLine

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


def test_script_line_can_carry_the_producing_model() -> None:
    """The field exists and is optional — an undeclared timing is still representable."""
    line = ScriptLine(text="hi", start=0.0, end=0.4)
    assert line.timestamp_model is None

    stamped = ScriptLine(
        text="hi",
        start=0.0,
        end=0.4,
        timestamp_source="external_aligner",
        timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    assert stamped.timestamp_model == "Qwen/Qwen3-ForcedAligner-0.6B"


def test_the_alignment_stage_stamps_the_aligner_it_used() -> None:
    """``stage_alignment`` knows ``aligner_model_id``; the words it returns must say so.

    Read from the source rather than by running an aligner: the wiring is the untested part, the
    same gap ``test_enhanced_perturbation_maps_to_the_enhanced_variant`` was written for.
    """
    src = (REPO_ROOT / "src/senselab/audio/workflows/audio_analysis/stages.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "stage_alignment")
    body = ast.get_source_segment(src, fn) or ""
    assert "_stamp_timing_provenance" in body, "stage_alignment must stamp who timed the words"
    assert "aligner_model_id" in body, "and must stamp the aligner it actually used"


def test_the_stamper_reaches_every_word_in_both_shapes() -> None:
    """Objects and dicts both have to end up carrying it.

    A backend returning ``ScriptLine`` objects and a cache returning their dict form must not
    disagree about provenance, because the fallback for an unstamped word is "its own timing
    source" — the permissive direction, which manufactures corroboration rather than losing it.
    """
    from senselab.audio.workflows.audio_analysis.stages import _stamp_timing_provenance

    obj = ScriptLine(text="a b", chunks=[ScriptLine(text="a", start=0.0, end=0.2)])
    as_dict = {"text": "a b", "chunks": [{"text": "a", "start": 0.0, "end": 0.2}]}
    for result in (obj, as_dict):
        _stamp_timing_provenance(result, source="external_aligner", model_id="facebook/mms-1b-all")

    assert obj.chunks is not None
    assert obj.chunks[0].timestamp_model == "facebook/mms-1b-all"
    assert obj.chunks[0].timestamp_source == "external_aligner"
    assert as_dict["chunks"][0]["timestamp_model"] == "facebook/mms-1b-all"  # type: ignore[index]


def test_the_stamper_does_not_overwrite_a_declared_source() -> None:
    """A backend that already said who timed it is the authority; the stage must not relabel it."""
    from senselab.audio.workflows.audio_analysis.stages import _stamp_timing_provenance

    native = ScriptLine(text="a", start=0.0, end=0.2, timestamp_source="native", timestamp_model="some/asr")
    _stamp_timing_provenance(native, source="external_aligner", model_id="facebook/mms-1b-all")
    assert native.timestamp_source == "native" and native.timestamp_model == "some/asr"


def test_the_qwen_backend_stamps_its_bundled_aligner() -> None:
    """Qwen3-ASR's timings are the companion aligner's, and the word has to admit it.

    Without this the two sides carry different kinds and no identity, so they group apart — which
    is the failure mode, silently reporting corroboration that does not exist.
    """
    src = (REPO_ROOT / "src/senselab/audio/tasks/speech_to_text/qwen.py").read_text()
    assert "timestamp_model" in src, "the qwen backend must stamp the aligner that produced its times"
