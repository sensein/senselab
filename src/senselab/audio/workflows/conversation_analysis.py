"""Conversation analysis workflow for multi-speaker recordings."""

from __future__ import annotations

import os
import re
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.quality_control.metrics import (
    amplitude_headroom_metric,
    dynamic_range_metric,
    proportion_clipped_metric,
    proportion_silent_metric,
    spectral_gating_snr_metric,
)
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

BACKCHANNEL_CUES = {
    "yeah",
    "yep",
    "right",
    "okay",
    "ok",
    "uh-huh",
    "mm-hmm",
    "sure",
    "got it",
}
DISCOURSE_MARKERS = {
    "well",
    "so",
    "like",
    "actually",
    "basically",
    "anyway",
    "however",
    "therefore",
    "because",
}
POLITENESS_CUES = {"please", "thanks", "thank", "sorry", "excuse"}
POSITIVE_SENTIMENT_CUES = {"good", "great", "happy", "glad", "love", "excellent", "thanks"}
NEGATIVE_SENTIMENT_CUES = {"bad", "sad", "angry", "upset", "hate", "worry", "problem"}
QUESTION_PREFIXES = {"who", "what", "when", "where", "why", "how", "can", "could", "would", "will", "do", "did"}
REQUEST_PREFIXES = {"please", "can", "could", "would", "help", "let", "let's"}
COMMAND_PREFIXES = {"please", "do", "tell", "show", "look", "check", "start", "stop"}


def read_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for audio loading."""
    from senselab.audio.tasks.input_output.utils import read_audios as _read_audios

    return _read_audios(*args, **kwargs)


def downmix_audios_to_mono(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for mono downmixing."""
    from senselab.audio.tasks.preprocessing.preprocessing import downmix_audios_to_mono as _downmix

    return _downmix(*args, **kwargs)


def resample_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for resampling."""
    from senselab.audio.tasks.preprocessing.preprocessing import resample_audios as _resample

    return _resample(*args, **kwargs)


def extract_segments(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for segment extraction."""
    from senselab.audio.tasks.preprocessing.preprocessing import extract_segments as _extract_segments

    return _extract_segments(*args, **kwargs)


def diarize_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for speaker diarization."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios as _diarize_audios

    return _diarize_audios(*args, **kwargs)


def transcribe_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for transcription."""
    from senselab.audio.tasks.speech_to_text import transcribe_audios as _transcribe_audios

    return _transcribe_audios(*args, **kwargs)


def extract_features_from_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for multi-backend acoustic features."""
    from senselab.audio.tasks.features_extraction import extract_features_from_audios as _extract_features

    return _extract_features(*args, **kwargs)


def extract_ppgs_from_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for PPG extraction."""
    from senselab.audio.tasks.features_extraction.ppg import extract_ppgs_from_audios as _extract_ppgs

    return _extract_ppgs(*args, **kwargs)


def classify_emotions_from_speech(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for speech emotion recognition."""
    from senselab.audio.tasks.classification.speech_emotion_recognition import (
        classify_emotions_from_speech as _classify_emotions,
    )

    return _classify_emotions(*args, **kwargs)


def extract_speaker_embeddings_from_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for speaker embeddings."""
    from senselab.audio.tasks.speaker_embeddings import (
        extract_speaker_embeddings_from_audios as _extract_speaker_embeddings,
    )

    return _extract_speaker_embeddings(*args, **kwargs)


def extract_ssl_embeddings_from_audios(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Lazy wrapper for SSL embeddings."""
    from senselab.audio.tasks.ssl_embeddings import extract_ssl_embeddings_from_audios as _extract_ssl_embeddings

    return _extract_ssl_embeddings(*args, **kwargs)


def analyze_conversation_recordings(
    recording_file_paths: Sequence[str | os.PathLike],
    diarization_model: Optional[Any] = None,  # noqa: ANN401
    transcription_models: Optional[Sequence[Any]] = None,  # noqa: ANN401
    features_config: Optional[Dict[str, Any]] = None,
    speaker_embeddings_models: Optional[Sequence[Any]] = None,  # noqa: ANN401
    ssl_embeddings_models: Optional[Sequence[Any]] = None,  # noqa: ANN401
    emotion_model: Optional[Any] = None,  # noqa: ANN401
    include_ppgs: bool = False,
    device: Optional[DeviceType] = None,
) -> List[Dict[str, Any]]:
    """Analyze multi-speaker conversation recordings.

    Args:
        recording_file_paths: Audio or video recordings to analyze.
        diarization_model: Optional diarization model. Defaults to Pyannote community diarization.
        transcription_models: Optional ASR model list. Defaults to Whisper tiny.
        features_config: Optional feature backend configuration.
        speaker_embeddings_models: Optional speaker embedding models. If omitted, speaker embeddings are skipped.
        ssl_embeddings_models: Optional SSL embedding models. If omitted, SSL embeddings are skipped.
        emotion_model: Optional speech emotion recognition model.
        include_ppgs: Whether to attempt PPG extraction.
        device: Optional device for model-backed stages.

    Returns:
        One result dictionary per input recording.
    """
    if not recording_file_paths:
        return []

    resolved_source_paths = [str(Path(path).expanduser().resolve()) for path in recording_file_paths]
    prepared_audio_paths = _prepare_recording_audio_files(resolved_source_paths)

    if transcription_models is None:
        transcription_models = [HFModel(path_or_uri="openai/whisper-tiny")]
    if features_config is None:
        features_config = {"opensmile": True, "parselmouth": True, "torchaudio": False, "torchaudio_squim": False}

    recording_audios = read_audios(prepared_audio_paths)
    recording_audios = downmix_audios_to_mono(recording_audios)
    recording_audios = resample_audios(recording_audios, resample_rate=16000)

    recording_checks = [_build_recording_checks(audio) for audio in recording_audios]
    diarization_results = diarize_audios(audios=recording_audios, model=diarization_model, device=device)
    segments_info = _collect_segment_boundaries(recording_audios, diarization_results)
    segmented_audios_list = extract_segments(segments_info)
    flattened_segments = [segment for group in segmented_audios_list for segment in group]

    transcript_outputs: Dict[str, List[ScriptLine]] = {}
    for model in transcription_models:
        model_name = getattr(model, "path_or_uri", str(model))
        transcript_outputs[str(model_name)] = transcribe_audios(
            audios=flattened_segments,
            model=model,
            device=device,
        )

    feature_outputs = extract_features_from_audios(
        audios=flattened_segments,
        opensmile=features_config.get("opensmile", True),
        parselmouth=features_config.get("parselmouth", True),
        torchaudio=features_config.get("torchaudio", False),
        torchaudio_squim=bool(features_config.get("torchaudio_squim", False)),
        device=device,
        sparc=features_config.get("sparc"),
        ppgs=False,
    )

    emotion_outputs: Optional[List[AudioClassificationResult]] = None
    emotion_error: Optional[str] = None
    if emotion_model is not None and flattened_segments:
        try:
            emotion_outputs = classify_emotions_from_speech(flattened_segments, model=emotion_model, device=device)
        except Exception as exc:  # pragma: no cover - defensive orchestration
            emotion_error = str(exc)

    speaker_embedding_outputs: Dict[str, List[List[float]]] = {}
    speaker_embedding_error: Optional[str] = None
    if speaker_embeddings_models:
        try:
            for model in speaker_embeddings_models:
                model_name = str(getattr(model, "path_or_uri", model))
                embeddings = extract_speaker_embeddings_from_audios(
                    audios=flattened_segments,
                    model=model,
                    device=device,
                )
                speaker_embedding_outputs[model_name] = [_tensor_to_list(embedding) for embedding in embeddings]
        except Exception as exc:  # pragma: no cover - defensive orchestration
            speaker_embedding_error = str(exc)

    ssl_embedding_outputs: Dict[str, List[List[float]]] = {}
    ssl_embedding_error: Optional[str] = None
    if ssl_embeddings_models:
        try:
            for model in ssl_embeddings_models:
                model_name = str(getattr(model, "path_or_uri", model))
                embeddings = extract_ssl_embeddings_from_audios(audios=flattened_segments, model=model, device=device)
                ssl_embedding_outputs[model_name] = [_pool_embedding(embedding) for embedding in embeddings]
        except Exception as exc:  # pragma: no cover - defensive orchestration
            ssl_embedding_error = str(exc)

    ppg_outputs: Optional[List[Dict[str, Any]]] = None
    ppg_error: Optional[str] = None
    if include_ppgs and flattened_segments:
        try:
            ppg_tensors = extract_ppgs_from_audios(flattened_segments, device=device)
            ppg_outputs = [
                _summarize_ppg_segments(audio=audio, posteriorgram=posteriorgram)
                for audio, posteriorgram in zip(flattened_segments, ppg_tensors)
            ]
        except Exception as exc:  # pragma: no cover - defensive orchestration
            ppg_error = str(exc)

    results: List[Dict[str, Any]] = []
    offset = 0
    primary_model_name = next(iter(transcript_outputs.keys()))

    for index, (source_file, audio_file, recording_audio, diarization_lines, segment_group) in enumerate(
        zip(resolved_source_paths, prepared_audio_paths, recording_audios, diarization_results, segmented_audios_list)
    ):
        turns: List[Dict[str, Any]] = []

        for local_index, (segment, diarization_line) in enumerate(zip(segment_group, diarization_lines)):
            global_index = offset + local_index
            transcript_map = {
                model_name: (lines[global_index].text or "")
                for model_name, lines in transcript_outputs.items()
            }
            primary_script_line = transcript_outputs[primary_model_name][global_index]
            transcript_accuracy = _estimate_transcript_accuracy(transcript_map, primary_script_line)
            primary_text = transcript_map[primary_model_name]

            linguistic_features = _extract_linguistic_features(primary_text)
            dialogue_acts = _extract_dialogue_acts(primary_text)
            engagement_markers = _extract_engagement_markers(primary_text)
            acoustic_features = _extract_acoustic_features(feature_outputs[global_index])
            lexical_sentiment = _estimate_lexical_sentiment(primary_text)
            speech_emotion = (
                _format_emotion_result(emotion_outputs[global_index]) if emotion_outputs is not None else None
            )
            emotion_summary = {
                "top_label": speech_emotion["top_label"] if speech_emotion else lexical_sentiment["label"],
                "top_score": speech_emotion["top_score"] if speech_emotion else lexical_sentiment["score"],
                "lexical_sentiment": lexical_sentiment,
                "speech_emotion": speech_emotion,
            }

            turn_result = {
                "speaker_id": diarization_line.speaker,
                "start": diarization_line.start,
                "end": diarization_line.end,
                "duration_seconds": _segment_duration_seconds(diarization_line),
                "transcripts": transcript_map,
                "transcript_accuracy_estimate": transcript_accuracy["score"],
                "transcript_accuracy_estimate_method": transcript_accuracy["method"],
                "transcript_accuracy_details": transcript_accuracy,
                "acoustic_features": acoustic_features,
                "linguistic_features": linguistic_features,
                "dialogue_acts": dialogue_acts,
                "engagement_markers": engagement_markers,
                "emotion": emotion_summary,
                "speaker_embeddings": {
                    model_name: values[global_index] for model_name, values in speaker_embedding_outputs.items()
                },
                "ssl_embeddings": {
                    model_name: values[global_index] for model_name, values in ssl_embedding_outputs.items()
                },
                "ppg_summary": ppg_outputs[global_index] if ppg_outputs is not None else None,
                "checks": _build_turn_checks(
                    transcript_text=primary_text,
                    acoustic_features=acoustic_features,
                    ppg_summary=ppg_outputs[global_index] if ppg_outputs is not None else None,
                ),
            }
            turns.append(turn_result)

        offset += len(segment_group)
        checks = dict(recording_checks[index])
        checks["diarization"] = {"passed": len(diarization_lines) > 0, "turn_count": len(diarization_lines)}
        checks["transcription"] = {
            "passed": len(turns) == len(segment_group),
            "model_count": len(transcript_outputs),
        }
        checks["features"] = {"passed": len(segment_group) == len(turns)}
        checks["emotion"] = {"passed": emotion_error is None, "error": emotion_error}
        checks["speaker_embeddings"] = {"passed": speaker_embedding_error is None, "error": speaker_embedding_error}
        checks["ssl_embeddings"] = {"passed": ssl_embedding_error is None, "error": ssl_embedding_error}
        checks["ppgs"] = {"passed": ppg_error is None, "error": ppg_error}

        results.append(
            {
                "source_file": source_file,
                "derived_audio_file": None if source_file == audio_file else audio_file,
                "checks": checks,
                "environment_context": _summarize_environment_context(checks),
                "speaker_summary": _summarize_speakers(turns),
                "turn_taking": _summarize_turn_taking(turns),
                "turns": turns,
                "transcript_summary": _summarize_transcripts(turns, primary_model_name),
                "recording_duration_seconds": _audio_duration_seconds(recording_audio),
            }
        )

    return results


def _prepare_recording_audio_files(recording_file_paths: Sequence[str | os.PathLike]) -> List[str]:
    """Resolve recordings to audio files, extracting video audio when needed."""
    prepared_audio_paths: List[str] = []

    for file_path in recording_file_paths:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Recording file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            prepared_audio_paths.append(str(path))
            continue
        if suffix in VIDEO_EXTENSIONS:
            prepared_audio_paths.append(_extract_audio_from_video(path))
            continue

        raise ValueError(f"Unsupported recording format: {path.suffix}")

    return prepared_audio_paths


def _extract_audio_from_video(video_path: Path) -> str:
    """Extract a mono WAV audio track from a local video recording."""
    import ffmpeg

    output_dir = Path(tempfile.mkdtemp(prefix="senselab-conversation-audio-"))
    output_path = output_dir / f"{video_path.stem}.wav"

    audio_stream = ffmpeg.input(str(video_path)).audio.output(str(output_path), format="wav", acodec="pcm_s16le")
    ffmpeg.run(audio_stream, overwrite_output=True, quiet=True)
    return str(output_path)


def _collect_segment_boundaries(
    audios: Sequence[Audio],
    diarization_results: Sequence[Sequence[ScriptLine]],
) -> List[tuple[Audio, List[tuple[float, float]]]]:
    """Collect segment boundaries suitable for `extract_segments`."""
    segments_info: List[tuple[Audio, List[tuple[float, float]]]] = []

    for audio, script_lines in zip(audios, diarization_results):
        boundaries = [
            (line.start, line.end)
            for line in script_lines
            if line.start is not None and line.end is not None and line.end > line.start
        ]
        segments_info.append((audio, boundaries))

    return segments_info


def _estimate_transcript_accuracy(
    transcripts: Dict[str, str],
    primary_script_line: Optional[ScriptLine] = None,
) -> Dict[str, Any]:
    """Estimate transcript reliability from model agreement or timestamp coverage."""
    normalized_transcripts = {
        model_name: " ".join(_tokenize_text(text))
        for model_name, text in transcripts.items()
        if text and _tokenize_text(text)
    }

    if len(normalized_transcripts) > 1:
        model_names = list(normalized_transcripts.keys())
        similarities: List[float] = []
        for index, model_name in enumerate(model_names):
            for comparison_name in model_names[index + 1 :]:
                similarities.append(
                    SequenceMatcher(
                        None,
                        normalized_transcripts[model_name],
                        normalized_transcripts[comparison_name],
                    ).ratio()
                )
        score = float(mean(similarities)) if similarities else 0.0
        return {
            "score": round(score, 4),
            "method": "multi_model_consensus",
            "model_count": len(normalized_transcripts),
            "pairwise_similarity_mean": round(score, 4),
        }

    if primary_script_line is not None:
        tokens = _tokenize_text(primary_script_line.text or "")
        if tokens and primary_script_line.chunks:
            chunk_count = sum(1 for chunk in primary_script_line.chunks if chunk.text)
            score = min(1.0, chunk_count / max(len(tokens), 1))
            return {
                "score": round(score, 4),
                "method": "timestamp_coverage",
                "timestamped_chunk_count": chunk_count,
                "token_count": len(tokens),
            }

    text = next(iter(transcripts.values()), "")
    token_count = len(_tokenize_text(text))
    score = min(0.75, 0.35 + 0.05 * token_count) if token_count else 0.0
    return {
        "score": round(score, 4),
        "method": "single_model_heuristic",
        "token_count": token_count,
    }


def _extract_linguistic_features(text: str) -> Dict[str, Any]:
    """Extract lightweight lexical and syntax-oriented transcript features."""
    tokens = _tokenize_text(text)
    lower_tokens = [token.lower() for token in tokens]
    token_counter = Counter(lower_tokens)
    sentence_fragments = [fragment for fragment in re.split(r"[.!?]+", text) if fragment.strip()]

    discourse_markers = sorted({token for token in lower_tokens if token in DISCOURSE_MARKERS})
    politeness_cues = sorted({token for token in lower_tokens if token in POLITENESS_CUES})

    return {
        "token_count": len(lower_tokens),
        "unique_token_count": len(set(lower_tokens)),
        "type_token_ratio": round(len(set(lower_tokens)) / len(lower_tokens), 4) if lower_tokens else 0.0,
        "mean_token_length": round(mean(len(token) for token in lower_tokens), 4) if lower_tokens else 0.0,
        "sentence_count": len(sentence_fragments) or (1 if text.strip() else 0),
        "mean_sentence_length_tokens": round(
            len(lower_tokens) / max(len(sentence_fragments), 1),
            4,
        )
        if lower_tokens
        else 0.0,
        "pronoun_count": sum(token_counter[token] for token in {"i", "you", "we", "they", "he", "she"}),
        "discourse_markers": discourse_markers,
        "politeness_cues": politeness_cues,
    }


def _extract_dialogue_acts(text: str) -> List[str]:
    """Infer coarse dialogue acts from transcript text."""
    tokens = _tokenize_text(text)
    lower_tokens = [token.lower() for token in tokens]
    if not lower_tokens:
        return []

    stripped_text = text.strip().lower()
    acts: List[str] = []

    if text.strip().endswith("?") or lower_tokens[0] in QUESTION_PREFIXES:
        acts.append("question")
    if any(token in POLITENESS_CUES for token in lower_tokens) or lower_tokens[0] in REQUEST_PREFIXES:
        acts.append("request")
    if lower_tokens[0] in COMMAND_PREFIXES and "question" not in acts:
        acts.append("command")
    if stripped_text in {"hi", "hello", "hey"} or lower_tokens[0] in {"hi", "hello", "hey"}:
        acts.append("greeting")
    if any(cue in stripped_text for cue in {"bye", "goodbye", "see you"}):
        acts.append("closing")
    if any(token in BACKCHANNEL_CUES for token in lower_tokens):
        acts.append("acknowledgement")
    if not acts:
        acts.append("statement")

    return sorted(set(acts))


def _extract_engagement_markers(text: str) -> Dict[str, Any]:
    """Infer engagement-related cues from transcript text."""
    lower_tokens = [token.lower() for token in _tokenize_text(text)]
    backchannel_cues = sorted({token for token in lower_tokens if token in BACKCHANNEL_CUES})
    politeness_cues = sorted({token for token in lower_tokens if token in POLITENESS_CUES})

    is_backchannel = bool(backchannel_cues) and len(lower_tokens) <= 4
    return {
        "is_backchannel": is_backchannel,
        "backchannel_cues": backchannel_cues,
        "politeness_cues": politeness_cues,
        "engagement_score": round(min(1.0, 0.25 * len(backchannel_cues) + 0.25 * len(politeness_cues)), 4),
    }


def _estimate_lexical_sentiment(text: str) -> Dict[str, Any]:
    """Estimate lexical sentiment from simple polarity cues."""
    lower_tokens = [token.lower() for token in _tokenize_text(text)]
    positive = sum(1 for token in lower_tokens if token in POSITIVE_SENTIMENT_CUES)
    negative = sum(1 for token in lower_tokens if token in NEGATIVE_SENTIMENT_CUES)

    if positive > negative:
        label = "positive"
    elif negative > positive:
        label = "negative"
    else:
        label = "neutral"

    total = max(len(lower_tokens), 1)
    score = abs(positive - negative) / total
    return {
        "label": label,
        "score": round(score, 4),
        "positive_cue_count": positive,
        "negative_cue_count": negative,
    }


def _extract_acoustic_features(feature_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Map backend-specific features onto conversation-oriented acoustic summaries."""
    praat_features = feature_bundle.get("praat_parselmouth", {})
    opensmile_features = feature_bundle.get("opensmile", {})

    return {
        "pitch_hz": _first_present(praat_features, ["mean_f0_hertz", "pitch_floor"]),
        "energy_db": _first_present(praat_features, ["mean_intensity_db"]),
        "speech_rate": _first_present(praat_features, ["speaking_rate", "articulation_rate"]),
        "pause_duration_seconds": _first_present(praat_features, ["mean_pause_duration", "mean_pause_dur"]),
        "pause_rate": _first_present(praat_features, ["pause_rate"]),
        "loudness": _first_present(opensmile_features, ["loudness_sma3_amean"]),
        "raw_features": feature_bundle,
    }


def _summarize_turn_taking(turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize turn-taking dynamics for a recording."""
    if not turns:
        return {
            "turn_count": 0,
            "speaker_switch_count": 0,
            "interruption_count": 0,
            "overlap_count": 0,
            "mean_gap_duration_seconds": 0.0,
            "mean_turn_duration_seconds": 0.0,
        }

    sorted_turns = sorted(turns, key=lambda turn: (turn.get("start") or 0.0, turn.get("end") or 0.0))
    speaker_switch_count = 0
    interruption_count = 0
    overlap_count = 0
    gap_durations: List[float] = []

    for current_turn, next_turn in zip(sorted_turns, sorted_turns[1:]):
        if current_turn.get("speaker_id") != next_turn.get("speaker_id"):
            speaker_switch_count += 1

        current_end = current_turn.get("end") or 0.0
        next_start = next_turn.get("start") or 0.0
        gap = next_start - current_end
        gap_durations.append(max(gap, 0.0))

        if gap < 0:
            overlap_count += 1
            if current_turn.get("speaker_id") != next_turn.get("speaker_id"):
                interruption_count += 1

    return {
        "turn_count": len(sorted_turns),
        "speaker_switch_count": speaker_switch_count,
        "interruption_count": interruption_count,
        "overlap_count": overlap_count,
        "mean_gap_duration_seconds": round(mean(gap_durations), 4) if gap_durations else 0.0,
        "mean_turn_duration_seconds": round(
            mean(float(turn.get("duration_seconds") or 0.0) for turn in sorted_turns),
            4,
        ),
    }


def _summarize_speakers(turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize speaker-level participation."""
    speaker_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"turn_count": 0, "total_duration_seconds": 0.0, "word_count": 0}
    )

    for turn in turns:
        speaker_id = turn.get("speaker_id") or "unknown"
        speaker_stats[speaker_id]["turn_count"] += 1
        speaker_stats[speaker_id]["total_duration_seconds"] += float(turn.get("duration_seconds") or 0.0)
        primary_transcript = next(iter((turn.get("transcripts") or {}).values()), "")
        speaker_stats[speaker_id]["word_count"] += len(_tokenize_text(primary_transcript))

    for stats in speaker_stats.values():
        stats["mean_turn_duration_seconds"] = round(
            stats["total_duration_seconds"] / max(stats["turn_count"], 1),
            4,
        )

    return {
        "speaker_count": len(speaker_stats),
        "speakers": dict(speaker_stats),
    }


def _summarize_transcripts(turns: Sequence[Dict[str, Any]], primary_model_name: str) -> Dict[str, Any]:
    """Summarize transcript outputs across a recording."""
    primary_texts = [turn["transcripts"].get(primary_model_name, "") for turn in turns]
    accuracy_scores = [float(turn.get("transcript_accuracy_estimate") or 0.0) for turn in turns]

    return {
        "primary_model": primary_model_name,
        "full_text": " ".join(text.strip() for text in primary_texts if text.strip()),
        "turn_count": len(turns),
        "mean_accuracy_estimate": round(mean(accuracy_scores), 4) if accuracy_scores else 0.0,
    }


def _summarize_ppg_segments(audio: Audio, posteriorgram: torch.Tensor) -> Dict[str, Any]:
    """Convert framewise posteriorgrams into sparse onset/offset segments."""
    if not isinstance(posteriorgram, torch.Tensor) or posteriorgram.ndim != 2 or posteriorgram.numel() == 0:
        return {"frame_count": 0, "segment_count": 0, "segments": []}

    if torch.isnan(posteriorgram).all():
        return {"frame_count": 0, "segment_count": 0, "segments": []}

    frame_count = int(posteriorgram.shape[0])
    duration_seconds = _audio_duration_seconds(audio)
    seconds_per_frame = duration_seconds / max(frame_count, 1)

    best_labels = posteriorgram.argmax(dim=1).tolist()
    best_scores = posteriorgram.max(dim=1).values.tolist()

    segments: List[Dict[str, Any]] = []
    start_index = 0
    current_label = best_labels[0]
    current_scores = [float(best_scores[0])]

    for frame_index in range(1, frame_count):
        if best_labels[frame_index] == current_label:
            current_scores.append(float(best_scores[frame_index]))
            continue

        segments.append(
            {
                "label_index": int(current_label),
                "start_frame": start_index,
                "end_frame": frame_index - 1,
                "start_seconds": round(start_index * seconds_per_frame, 6),
                "end_seconds": round(frame_index * seconds_per_frame, 6),
                "mean_confidence": round(mean(current_scores), 6),
            }
        )
        start_index = frame_index
        current_label = best_labels[frame_index]
        current_scores = [float(best_scores[frame_index])]

    segments.append(
        {
            "label_index": int(current_label),
            "start_frame": start_index,
            "end_frame": frame_count - 1,
            "start_seconds": round(start_index * seconds_per_frame, 6),
            "end_seconds": round(duration_seconds, 6),
            "mean_confidence": round(mean(current_scores), 6),
        }
    )

    return {"frame_count": frame_count, "segment_count": len(segments), "segments": segments}


def _build_recording_checks(audio: Audio) -> Dict[str, Any]:
    """Build automated checks for an input recording."""
    duration_seconds = _audio_duration_seconds(audio)
    silence_ratio = _safe_metric(proportion_silent_metric, audio)
    clipped_ratio = _safe_metric(proportion_clipped_metric, audio)
    headroom = _safe_metric(amplitude_headroom_metric, audio)
    snr_db = _safe_metric(spectral_gating_snr_metric, audio)
    dynamic_range = _safe_metric(dynamic_range_metric, audio)

    return {
        "input": {
            "passed": duration_seconds > 0.0,
            "duration_seconds": round(duration_seconds, 4),
        },
        "preprocessing": {
            "passed": audio.waveform.shape[0] == 1 and audio.sampling_rate == 16000,
            "channel_count": int(audio.waveform.shape[0]),
            "sampling_rate": int(audio.sampling_rate),
        },
        "environment": {
            "passed": (clipped_ratio or 0.0) < 0.01 and (silence_ratio or 0.0) < 0.98,
            "silence_ratio": silence_ratio,
            "clipped_ratio": clipped_ratio,
            "headroom": headroom,
            "snr_db": snr_db,
            "dynamic_range": dynamic_range,
        },
    }


def _summarize_environment_context(checks: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize environmental recording conditions from QC checks."""
    environment = checks.get("environment", {})
    snr_db = environment.get("snr_db")

    if snr_db is None:
        noise_level = "unknown"
    elif snr_db < 10:
        noise_level = "high"
    elif snr_db < 20:
        noise_level = "moderate"
    else:
        noise_level = "low"

    clipped_ratio = environment.get("clipped_ratio")
    if clipped_ratio is None:
        clipping = "unknown"
    elif clipped_ratio > 0:
        clipping = "present"
    else:
        clipping = "not_detected"

    return {
        "noise_level": noise_level,
        "clipping": clipping,
        "silence_ratio": environment.get("silence_ratio"),
        "snr_db": snr_db,
        "dynamic_range": environment.get("dynamic_range"),
    }


def _build_turn_checks(
    transcript_text: str,
    acoustic_features: Dict[str, Any],
    ppg_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build per-turn automated checks."""
    return {
        "transcript_present": bool(transcript_text.strip()),
        "speech_rate_available": acoustic_features.get("speech_rate") is not None,
        "pitch_available": acoustic_features.get("pitch_hz") is not None,
        "ppg_available": bool(ppg_summary and ppg_summary.get("segment_count", 0) > 0),
    }


def _format_emotion_result(result: AudioClassificationResult) -> Dict[str, Any]:
    """Format an emotion classification result for workflow output."""
    return {
        "top_label": result.top_label(),
        "top_score": float(result.top_score()),
        "labels": result.get_labels(),
        "scores": [float(score) for score in result.get_scores()],
    }


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into simple word-like units."""
    return re.findall(r"[A-Za-z']+", text.lower())


def _first_present(values: Dict[str, Any], keys: Iterable[str]) -> Any:  # noqa: ANN401
    """Return the first non-null value found among candidate keys."""
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return None


def _tensor_to_list(value: Any) -> List[float]:  # noqa: ANN401
    """Convert tensor-like values to plain Python lists."""
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().flatten().tolist()]
    return [float(item) for item in value]


def _pool_embedding(value: Any) -> List[float]:  # noqa: ANN401
    """Mean-pool embedding tensors onto a single vector."""
    if isinstance(value, torch.Tensor):
        pooled = value
        if value.ndim > 1:
            pooled = value.mean(dim=0)
        return [float(item) for item in pooled.detach().cpu().flatten().tolist()]
    return [float(item) for item in value]


def _audio_duration_seconds(audio: Audio) -> float:
    """Return audio duration in seconds."""
    if audio.sampling_rate == 0:
        return 0.0
    return float(audio.waveform.shape[-1] / audio.sampling_rate)


def _segment_duration_seconds(line: ScriptLine) -> float:
    """Return ScriptLine duration."""
    if line.start is None or line.end is None:
        return 0.0
    return float(max(line.end - line.start, 0.0))


def _safe_metric(metric_fn: Any, audio: Audio) -> Optional[float]:  # noqa: ANN401
    """Compute a metric defensively."""
    try:
        value = metric_fn(audio)
    except Exception:
        return None
    return float(value) if value is not None else None
