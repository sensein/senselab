"""This module implements the VibeVoice-ASR-HF diarization backend.

What the ``speaker`` field means here
-------------------------------------
This backend assigns its own per-file speaker labels (``Speaker`` tags parsed out of its
structured JSON output). They are labels, not
identities: the same tag in two different files carries no claim of being the same
person, which :func:`~senselab.audio.tasks.speaker_diarization.api.capabilities_for`
reports as ``labels_stable_across_files=False``.

That is true of diarizers generally -- ``SPEAKER_00`` is no more an identity than
``S01`` is. Reconciling labels from different backends into one namespace is a separate
concern with its own utility, and no backend module decides it. This one reports what it
produced and stops there.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import load_hf_resilient

try:
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    VIBEVOICE_AVAILABLE = True
except (ImportError, RuntimeError):
    # RuntimeError alongside ImportError: api.py imports this module unconditionally,
    # so a non-ImportError failure inside a transformers submodule (the pattern
    # audio.py/video.py/ppg.py/frame_posteriors.py already guard against) would
    # otherwise take down `import senselab.audio.tasks.speaker_diarization` for
    # Pyannote users too.
    VIBEVOICE_AVAILABLE = False

CAPABILITIES = DiarizationCapabilities(
    populates_text=True,  # joint ASR+diarization: measured 7/7 segments carried text
    speaker_label_kind="index",
    labels_stable_across_files=False,  # per-audio numbering; not measured otherwise
    # Seed-17 speaker-ceiling probe: at k=8, predicted counts ranged 6..16 (plus 5 refusals) —
    # the widest spread of any backend, so no structural ceiling was observed.
    max_speakers=None,
    max_speakers_evidence="measured: no saturation, emits up to 16 (probe seed-17)",
    honors_speaker_hints=False,  # api.py warns that num_speakers is dropped here
)


class VibeVoiceDiarization:
    """Factory for creating and caching **VibeVoice-ASR-HF** processor/model pairs.

    Pairs are cached per *(model.path_or_uri, revision, device)*, so repeated calls
    with the same configuration reuse the initialized model.

    Guidance:
        - VibeVoice-ASR-HF is a 7B-parameter unified ASR+diarization+timestamping
          model; CUDA is strongly recommended.
        - Supported devices: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _models: Dict[str, Tuple["AutoProcessor", "VibeVoiceAsrForConditionalGeneration"]] = {}

    @classmethod
    def _get_vibevoice_model(
        cls,
        model: HFModel,
        device: Optional[DeviceType],
    ) -> Tuple["AutoProcessor", "VibeVoiceAsrForConditionalGeneration"]:
        """Get or create a VibeVoice-ASR-HF processor/model pair.

        Args:
            model (HFModel): The VibeVoice-ASR-HF model.
            device (DeviceType | None): The device to run the model on.

        Returns:
            Tuple[AutoProcessor, VibeVoiceAsrForConditionalGeneration]: The processor and model.
        """
        if not VIBEVOICE_AVAILABLE:
            raise ModuleNotFoundError(
                "VibeVoice-ASR-HF requires `transformers>=5.3`. "
                "Please install/upgrade senselab audio dependencies using `pip install senselab`."
            )

        resolved_device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{resolved_device}"
        if key not in cls._models:
            # load_hf_resilient resolves (repo_id, revision) to an immutable commit
            # SHA once (download-once via the cross-process heartbeat lock) and
            # injects revision=<sha>, so a cached model makes no per-call Hub
            # version check — the 429 source under parallel batch load.
            token = get_huggingface_token()
            repo_id = str(model.path_or_uri)
            revision = model.revision or "main"
            processor = load_hf_resilient(
                AutoProcessor.from_pretrained, repo_id, repo_id=repo_id, revision=revision, token=token
            )
            # dtype="auto" (not the shared device-selector's fp16 default): this
            # checkpoint's config declares bfloat16, and every upstream example
            # loads it with dtype="auto" — forcing fp16 on a bf16-trained model
            # loses exponent range (overflow -> NaN), matching the model's own
            # trained precision instead. diarize_audios_with_vibevoice casts
            # inputs to whatever dtype actually loads, so this isn't tied to fp16.
            vv_model = load_hf_resilient(
                VibeVoiceAsrForConditionalGeneration.from_pretrained,
                repo_id,
                repo_id=repo_id,
                revision=revision,
                token=token,
                dtype="auto",
            )
            vv_model = vv_model.to(torch.device(resolved_device.value))  # type: ignore[arg-type]
            vv_model.eval()
            cls._models[key] = (processor, vv_model)
        return cls._models[key]

    @classmethod
    def release_all(cls) -> None:
        """Drop all cached processor/model pairs and free GPU memory.

        VibeVoice-ASR-HF is a 7B-parameter model held in a class-level cache with
        no natural eviction point — unlike the other three new diarization
        backends, which are subprocess-hosted and free their memory automatically
        on process exit. Call this once a caller is done running VibeVoice for a
        pass so its ~14GB+ footprint doesn't stay resident alongside another large
        in-process model for the rest of the run.
        """
        cls._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def diarize_audios_with_vibevoice(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    max_new_tokens: int = 4096,
) -> List[List[ScriptLine]]:
    """Diarize audios with **VibeVoice-ASR-HF**; returns per-speaker segments per audio.

    VibeVoice-ASR-HF is a unified ASR+diarization+timestamping foundation model
    (`microsoft/VibeVoice-ASR-HF`, natively supported by ``transformers>=5.3`` via
    ``AutoProcessor``/``VibeVoiceAsrForConditionalGeneration`` — no custom repo code
    or ``trust_remote_code`` needed). Each audio is transcribed in a single pass and
    the model's own diarization is parsed out of its structured JSON-shaped output.

    Args:
        audios (list[Audio]):
            Audio clips to diarize.
        model (HFModel | None):
            VibeVoice model. Defaults to ``HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")``.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``). CUDA is
            strongly recommended — this is a 7B-parameter model.
        max_new_tokens (int):
            Generation budget. Defaults to 4096; raise this for longer recordings.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker`, `start`, `end`, and `text`. Empty for an audio whose output didn't
        parse into structured segments (logged as a warning, not raised) — unless
        *every* audio in the batch failed to parse, which raises instead (see Raises).

    Raises:
        ModuleNotFoundError: If `transformers>=5.3` is not installed.
        RuntimeError: If every audio in the batch failed to parse into structured
            segments — treated as a broken decode contract, not silent no-speech.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> lines = diarize_audios_with_vibevoice([a1], device=DeviceType.CPU)  # doctest: +SKIP
        >>> len(lines) == 1  # doctest: +SKIP
        True
    """
    if model is None:
        model = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")

    processor, vv_model = VibeVoiceDiarization._get_vibevoice_model(model, device)

    results: List[List[ScriptLine]] = []
    decode_failures = 0
    for audio in audios:
        with tempfile.TemporaryDirectory(prefix="senselab-vibevoice-") as tmpdir:
            wav_path = str(Path(tmpdir) / "audio.wav")
            audio.save_to_file(wav_path)

            inputs = processor.apply_transcription_request(audio=wav_path)  # type: ignore[attr-defined]
            model_dtype = next(vv_model.parameters()).dtype
            inputs = {
                k: (
                    v.to(device=vv_model.device, dtype=model_dtype)
                    if v.is_floating_point()
                    else v.to(device=vv_model.device)
                )
                if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                output_ids = vv_model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[misc]

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]

            if generated_ids.shape[-1] >= max_new_tokens:
                # generate() only stops before max_new_tokens if it hit an EOS token;
                # reaching the budget means the output is very likely cut off mid-JSON,
                # which is a different failure mode than "no speech in this audio" —
                # without this, a caller can't tell the two apart from an empty result.
                logger.warning(
                    f"VibeVoice-ASR-HF hit max_new_tokens={max_new_tokens} without generating "
                    "an end token; output is likely truncated. Pass a higher max_new_tokens "
                    "for longer recordings."
                )

            try:
                segments = processor.decode(generated_ids, return_format="parsed")  # type: ignore[attr-defined]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                # This except clause is a defensive backstop, not the primary failure
                # path: processor.decode's own extract_speaker_dict() catches shape
                # mismatches (wrong/renamed keys, non-numeric Start/End, ...) itself
                # and returns the *original undecoded string* instead of raising —
                # only literally-invalid JSON (e.g. truncated mid-array) reaches
                # json.loads() and raises. The `isinstance(segments, str)` check
                # below is what actually catches the shape-mismatch case.
                logger.warning(f"VibeVoice-ASR-HF produced unparsable output: {exc}")
                segments = []
                decode_failures += 1

            if isinstance(segments, str):
                # extract_speaker_dict()'s own fallback: valid JSON, but not
                # shaped as expected (missing "Content", non-numeric Start/End,
                # a schema rename upstream, ...). This — not the except clause
                # above — is the realistic way a future transformers revision
                # would break this backend, so it counts toward decode_failures
                # too; otherwise a total schema break would never be distinguishable
                # from a legitimately quiet batch (see the all-batch check below).
                logger.warning("VibeVoice-ASR-HF output did not parse into structured segments.")
                segments = []
                decode_failures += 1
            elif not segments:
                segments = []

            script_lines = []
            for seg in segments:
                if not isinstance(seg, dict) or seg.get("Start") is None or seg.get("End") is None:
                    continue
                try:
                    start = float(seg["Start"])
                    end = float(seg["End"])
                except (TypeError, ValueError) as exc:
                    # A single malformed segment shouldn't abort the whole batch
                    # the same way a decode()-level parse failure wouldn't (see
                    # the except clause above) — skip just this segment.
                    logger.warning(f"VibeVoice-ASR-HF produced a segment with unparsable Start/End ({exc}); skipping.")
                    continue

                # `Speaker` gets no upstream guard the way `Start`/`End` do above, and
                # VibeVoice's own JSON omits it on some segments of a joint ASR+diarization
                # pass. str()-ing it unconditionally turns that absence into the literal
                # string "None", which is indistinguishable from a real speaker label to any
                # consumer grouping segments by speaker — observed on a real H100 run, where
                # an 11.26s recording produced speaker labels ['None', '0'] for a 2-segment
                # result. Keep a missing/None Speaker as speaker=None instead: a genuine
                # absence a consumer can actually detect.
                raw_speaker = seg.get("Speaker")
                speaker = str(raw_speaker) if raw_speaker is not None else None
                content = seg.get("Content")

                if speaker is None and not content:
                    # ScriptLine requires at least one of speaker/text; a segment with
                    # neither has nothing to construct a line from. Skip it like the
                    # unparsable-Start/End case above rather than aborting the batch.
                    logger.warning("VibeVoice-ASR-HF produced a segment with neither Speaker nor Content; skipping.")
                    continue

                script_lines.append(ScriptLine(speaker=speaker, start=start, end=end, text=content))
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

    if audios and decode_failures == len(audios):
        # Every audio in the batch failed to parse — far more likely a broken
        # decode()/return_format="parsed" contract (e.g. an upstream transformers
        # revision bump) than every clip happening to contain no speech. Raising
        # here keeps that distinguishable from a legitimate empty-segments result,
        # which would otherwise get recorded as a "status-ok" cached outcome and
        # silently persist across future runs.
        raise RuntimeError(
            f"VibeVoice-ASR-HF failed to parse output for all {len(audios)} audio(s) in this "
            "batch; this looks like a broken decode()/return_format='parsed' contract rather "
            "than an absence of speech. See the preceding warning(s) for the underlying errors."
        )

    return results
