"""This script is used to test the video tasks."""

from senselab.video.input_output import extract_audios_from_local_videos

files = ["../src/tests/data_for_testing/video_48khz_stereo_16bits.mp4"]
dataset = extract_audios_from_local_videos(files)

print("dataset")
print(dataset)
