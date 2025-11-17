"""This module provides the `explore_conversation` function for conversational data exploration on audio files."""

import os
from typing import Any, Dict, List, Optional, Type, Union

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.input_output.utils import read_audios
from senselab.audio.tasks.preprocessing.preprocessing import (
    downmix_audios_to_mono,
    extract_segments,
    resample_audios,
)
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.audio.tasks.ssl_embeddings import extract_ssl_embeddings_from_audios
from senselab.utils.data_structures.model import HFModel, PyannoteAudioModel, SenselabModel, SpeechBrainModel


def explore_conversation(
    audio_file_paths: List[str | os.PathLike],
    speaker_diarization_model: Optional[SenselabModel] = None,
    transcription_models: Optional[List[SenselabModel]] = None,
    features_config: Optional[Dict[str, Union[bool, Dict[str, Any]]]] = None,
    speaker_embeddings_models: Optional[List[SpeechBrainModel]] = None,
    ssl_embeddings_models: Optional[List[SenselabModel]] = None,
) -> List[List[Dict[str, Any]]]:
    """Perform conversational data exploration on audio files.

    This function processes audio files by performing speaker diarization, transcription,
    feature extraction, speaker embeddings, and SSL embeddings extraction. The results
    are assembled into a list of JSON-serializable dictionaries, one per speaker turn.

    Args:
        audio_file_paths (List[str]): List of paths to audio files.
        speaker_diarization_model (SenselabModel): Model for speaker diarization.
            If None, defaults to PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1",
                                         revision="main").
        transcription_models (List[SenselabModel]): List of models for transcription.
            If None, defaults to [HFModel(path_or_uri="openai/whisper-tiny")].
        features_config (Dict[str, Union[bool, Dict[str, Any]]]): Configuration for feature extraction backends.
            Keys can include 'opensmile', 'parselmouth', 'torchaudio', 'torchaudio_squim'.
            Values are either bool (enable/disable) or dict (backend-specific config).
            If None, defaults to {'opensmile': True, 'parselmouth': True, 'torchaudio': False,
            'torchaudio_squim': False}.
        speaker_embeddings_models (List[SpeechBrainModel]): List of models for speaker embeddings.
            If None, defaults to [SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")].
        ssl_embeddings_models (List[SenselabModel]): List of models for SSL embeddings.
            If None, defaults to [HFModel(path_or_uri="microsoft/wavlm-base")].

    Returns:
        List[Dict[str, Any]]: List of JSON-serializable dicts, one per speaker segment.
    """
    if speaker_diarization_model is None:
        speaker_diarization_model = PyannoteAudioModel(
            path_or_uri="pyannote/speaker-diarization-community-1", revision="main"
        )
    if transcription_models is None:
        transcription_models = [HFModel(path_or_uri="openai/whisper-tiny")]
    if speaker_embeddings_models is None:
        speaker_embeddings_models = [SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")]
    if ssl_embeddings_models is None:
        ssl_embeddings_models = [HFModel(path_or_uri="microsoft/wavlm-base")]
    if features_config is None:
        features_config = {"opensmile": True, "parselmouth": True, "torchaudio": False, "torchaudio_squim": False}

    # 1. Resolve absolute paths for all audio files
    abs_paths = [os.path.abspath(p) for p in audio_file_paths]

    # 2. Load audio files into Audio objects
    audios = read_audios(abs_paths)

    # 3. Downmix to mono and resample to 16kHz
    audios = downmix_audios_to_mono(audios)
    target_sr = 16000  # Target sampling rate for most of our supported models
    audios = resample_audios(audios, resample_rate=target_sr)

    # 4. Speaker diarization
    diarization_results = diarize_audios(
        audios=audios,
        model=speaker_diarization_model,
    )

    # 5. Build (Audio, segment_times) pairs
    segments_info = []
    for audio, script_lines in zip(audios, diarization_results):
        times = []
        for line in script_lines:
            # Only include segments where start and end are not None
            if line.start is not None and line.end is not None:
                times.append((line.start, line.end))
        segments_info.append((audio, times))

    # 6. Extract segments
    segmented_audios_list = extract_segments(segments_info)

    # 7. Populate segment metadata
    for i, (segments, lines) in enumerate(zip(segmented_audios_list, diarization_results)):
        src_path = abs_paths[i] if i < len(abs_paths) else None
        n = min(len(segments), len(lines))
        for j in range(n):
            seg = segments[j]
            line = lines[j]
            md = seg.metadata
            md["original_file"] = src_path
            md["sampling_rate"] = seg.sampling_rate
            md["speaker_id"] = getattr(line, "speaker", None)
            md["segment_start"] = getattr(line, "start", None)
            md["segment_end"] = getattr(line, "end", None)

    # 8. Flatten segments
    flattened_segments = [seg for group in segmented_audios_list for seg in group]

    # 9. Transcribe segments with all provided models
    transcripts_by_model = {}
    for model in transcription_models:
        trans_out = transcribe_audios(audios=flattened_segments, model=model)
        transcripts_by_model[model.path_or_uri] = [t.text for t in trans_out]

    for idx, seg in enumerate(flattened_segments):
        seg.metadata["transcripts"] = {
            model_name: transcripts_by_model[model_name][idx] for model_name in transcripts_by_model
        }

    # 10. Extract features
    feat_out = extract_features_from_audios(
        audios=flattened_segments,
        opensmile=features_config.get("opensmile", True),
        parselmouth=features_config.get("parselmouth", True),
        torchaudio=features_config.get("torchaudio", False),
        torchaudio_squim=bool(features_config.get("torchaudio_squim", False)),
    )
    for seg, feats in zip(flattened_segments, feat_out):
        seg.metadata["features"] = feats

    # 11. Extract speaker embeddings for all provided models
    speaker_embeddings_by_model = {}
    for model in speaker_embeddings_models:
        emb_out = extract_speaker_embeddings_from_audios(audios=flattened_segments, model=model)
        speaker_embeddings_by_model[model.path_or_uri] = [e.tolist() for e in emb_out]

    for idx, seg in enumerate(flattened_segments):
        seg.metadata["speaker_embeddings"] = {
            model_name: speaker_embeddings_by_model[model_name][idx] for model_name in speaker_embeddings_by_model
        }

    # 12. Extract SSL embeddings for all provided models
    ssl_embeddings_by_model = {}
    for model in ssl_embeddings_models:
        ssl_out = extract_ssl_embeddings_from_audios(audios=flattened_segments, model=model)
        # Mean-pool over time dimension if needed
        pooled = [emb.mean(dim=1).squeeze().tolist() if hasattr(emb, "mean") else emb for emb in ssl_out]
        ssl_embeddings_by_model[model.path_or_uri] = pooled

    for idx, seg in enumerate(flattened_segments):
        seg.metadata["ssl_embeddings"] = {
            model_name: ssl_embeddings_by_model[model_name][idx] for model_name in ssl_embeddings_by_model
        }

    # 13. Build JSON output per segment
    def build_json_from_segment(seg: Audio) -> Dict[str, Any]:
        md = getattr(seg, "metadata", {}) or {}
        return {
            "original_file": md.get("original_file"),
            "sampling_rate": md.get("sampling_rate"),
            "speaker_id": md.get("speaker_id"),
            "start": md.get("segment_start"),
            "end": md.get("segment_end"),
            "transcripts": md.get("transcripts"),
            "features": md.get("features"),
            "speaker_embeddings": md.get("speaker_embeddings"),
            "ssl_embeddings": md.get("ssl_embeddings"),
        }

    # Group segments by input audio file
    segments_by_audio: List[List[Dict[str, Any]]] = [[] for _ in abs_paths]
    for seg in flattened_segments:
        md = getattr(seg, "metadata", {}) or {}
        original_file = md.get("original_file")
        if original_file in abs_paths:
            idx = abs_paths.index(original_file)
            segments_by_audio[idx].append(build_json_from_segment(seg))
    return segments_by_audio
