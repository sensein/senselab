from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.tasks.voice_cloning.api import clone_voices
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, TorchModel

mono_audio_from_file = Audio.from_filepath("../src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
resampled_audios = resample_audios([mono_audio_from_file], 16000)
resampled_audio = resampled_audios[0]

model = TorchModel(path_or_uri="bshall/knn-vc", revision="master")

cloned = clone_voices(source_audios=[resampled_audio, resampled_audio], target_audios=[resampled_audio, resampled_audio], model=model, device=DeviceType.CPU)

print(cloned)

transcripts = transcribe_audios(audios=[resampled_audio, cloned[0]], model=HFModel(path_or_uri="openai/whisper-tiny"))

print(transcripts)