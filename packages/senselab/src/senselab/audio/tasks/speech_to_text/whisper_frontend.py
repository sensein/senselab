"""The Whisper front end shared by the Whisper backends of `speech_to_text`: the shortest input it takes."""

N_FFT = 400
RATE = 16_000
MIN_INPUT_NS = N_FFT * 1_000_000_000 // RATE
"""The shortest audio, in ns, whose STFT has one `N_FFT` window at `RATE`."""
