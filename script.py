from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.audio.tasks.voice_activity_detection.api import detect_human_voice_activity_in_audios
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine

sample_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))




diarize = diarize_audios_with_pyannote(audios=[sample_audio], 
                                       model=HFModel(path_or_uri="pyannote/speaker-diarization-3.1", 
                                                     revision="main"),
                                        device=DeviceType.CPU, 
                                        num_speakers=4, 
                                        min_speakers=4,
                                        max_speakers=4)

print("diarize")
print(diarize)


vad = detect_human_voice_activity_in_audios(audios=[sample_audio],
                                             model=HFModel(path_or_uri="pyannote/speaker-diarization-3.1", 
                                                           revision="main"), 
                                                           device=DeviceType.CPU)
print("vad")
print(vad)


input("shsjsjs")





hf_model = HFModel(path_or_uri="openai/whisper-tiny")
transcripts = HuggingFaceASR.transcribe_audios_with_transformers(
    audios=[sample_audio],
    model=hf_model,
    language=Language(language_code="English"),
    return_timestamps="word",
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=1,
    device=DeviceType.CPU,
)
print(transcripts)

transcript = ScriptLine.from_dict({
    "text": "            None",
})

print(transcript.get_text())
print(transcript.get_speaker())
print(transcript.get_chunks())



