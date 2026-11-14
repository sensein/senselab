"""The Whisper front end shared by the Whisper backends of `speech_to_text`: the shortest input it takes."""

from senselab_worker.units import NS_PER_S

N_FFT = 400
RATE = 16_000
MIN_INPUT_NS = N_FFT * NS_PER_S // RATE
"""The shortest audio, in ns, whose STFT has one `N_FFT` window at `RATE`."""
