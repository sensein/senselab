"""Forced aligns audio files using diarization and transcription models.

Functions:
    create_clip(audio: Audio, start: float, end: float) -> Optional[Audio]:
        Creates a clipped segment of the audio.
    clip_audios(audio: Audio, diarization: List[ScriptLine]) -> List[Audio]:
        Clips audio segments based on diarization.
    force_align(audios: List[Audio], diarization_model_path: str,
                transcription_model_path: str, device: DeviceType = DeviceType.CPU)
                -> List[ScriptLine]:
        Force aligns audio using diarization and transcription models.
"""

from typing import List, Optional

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


def create_clip(audio: Audio, start: float, end: float) -> Optional[Audio]:
    """Create a clipped segment of the audio.

    Args:
        audio (Audio): The original audio object.
        start (float): The start time of the clip in seconds.
        end (float): The end time of the clip in seconds.

    Returns:
        Optional[Audio]: The clipped audio segment, or None if the clip range is invalid.
    """
    start_sample = int(start * audio.sampling_rate)
    end_sample = int(end * audio.sampling_rate)

    # Ensure the start and end samples are within the audio waveform range
    if start_sample >= audio.waveform.size(1) or end_sample > audio.waveform.size(1):
        return None

    waveform_clip = audio.waveform[:, start_sample:end_sample]

    return Audio(
        waveform=waveform_clip,
        sampling_rate=audio.sampling_rate,
        orig_path_or_id=f"{audio.orig_path_or_id}_clip_{start_sample}_{end_sample}",
        metadata={**audio.metadata, "clip_start": start, "clip_end": end},
    )


def clip_audios(audio: Audio, diarization: List[ScriptLine]) -> List[Audio]:
    """Clip the audio segments based on the diarization.

    Args:
        audio (Audio): The original audio object.
        diarization (List[ScriptLine]): List of ScriptLine objects containing the start and end times.

    Returns:
        List[Audio]: List of clipped audio segments.
    """
    clips = []
    for segment_sl in diarization:
        if segment_sl.start is None or segment_sl.end is None:
            raise ValueError("Diarization must have both start and end times.")
        clip = create_clip(audio, segment_sl.start, segment_sl.end)
        if clip:
            clips.append(clip)
    return clips


def force_align(
    audios: List[Audio], diarization_model_path: str, transcription_model_path: str, device: DeviceType = DeviceType.CPU
) -> List[ScriptLine]:
    """Force align audio using diarization and transcription models.

    Args:
        audios (List[Audio]): List of Audio objects to be processed.
        diarization_model_path (str): Path or URI of the diarization model.
        transcription_model_path (str): Path or URI of the transcription model.
        device (DeviceType): The device to use for model inference.

    Returns:
        List[ScriptLine]: List of ScriptLine objects containing the transcriptions for the audio clips.
    """
    # Perform diarization
    diarization_model = HFModel(path_or_uri=diarization_model_path)
    diarizations = diarize_audios(audios=audios, model=diarization_model, device=device)

    # Process each audio with its corresponding diarization result
    all_clips = []
    for audio, diarization in zip(audios, diarizations):
        audio_clips = clip_audios(audio=audio, diarization=diarization)
        all_clips.extend(audio_clips)

    # Transcribe the audio clips
    transcription_model = HFModel(path_or_uri=transcription_model_path)
    return transcribe_audios(audios=all_clips, model=transcription_model, device=device)
