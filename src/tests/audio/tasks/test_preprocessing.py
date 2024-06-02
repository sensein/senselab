from senselab.audio.tasks.preprocessing import resample_audio_dataset
from senselab.utils.data_structures.audio import Audio
import math

def test_resample_audio_dataset():
    resample_rate = 36000
    mono_audio = Audio.from_filepath(('src/tests/data_for_testing/audio_48khz_mono_16bits.wav'))
    resampled_expected_size = mono_audio.audio_data.shape[1]/48000*resample_rate

    resampled_audio = resample_audio_dataset([mono_audio], resample_rate)
    assert math.ceil(resampled_expected_size) == resampled_audio[0].audio_data.shape[1]

    stereo_audio = Audio.from_filepath(('src/tests/data_for_testing/audio_48khz_stereo_16bits.wav'))
    resampled_expected_size = stereo_audio.audio_data.shape[1]/48000*resample_rate

    resampled_audio = resample_audio_dataset([stereo_audio], resample_rate)
    assert math.ceil(resampled_expected_size) == resampled_audio[0].audio_data.shape[1]
