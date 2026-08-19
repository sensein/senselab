"""Smoke check: the recording loads, and both backends answer on it."""

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics.hear import HEAR_EVENT_LABELS

WAV = "/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav"

audio = Audio(filepath=WAV)
print("sr", audio.sampling_rate, "shape", tuple(audio.waveform.shape))
print("duration", audio.waveform.shape[1] / audio.sampling_rate)
print("hear labels", HEAR_EVENT_LABELS)
