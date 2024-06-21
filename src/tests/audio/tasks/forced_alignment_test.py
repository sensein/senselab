"""Tests forced alignment functions."""
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.forced_alignment.forced_alignment import align_transcriptions
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.model import HFModel


def test_alignment_runs() -> None:
    """Ensures that the forced_alignment functionality runs without errors on a test audio."""
    # audios = [
    #     Audio.from_filepath(
    #         "/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"
    #     )
        
    # ]
    audios = [Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))]
    transcription_model = HFModel(path_or_uri="openai/whisper-tiny")
    transcriptions = transcribe_audios(audios, model=transcription_model)

    aligned_transcriptions = align_transcriptions(audios=audios, transcriptions=transcriptions)
    print(aligned_transcriptions)

test_alignment_runs()