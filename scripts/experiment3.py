"""This script is used to test the video tasks."""
from senselab.video.input_output import extract_audios_from_local_videos

dataset = extract_audios_from_local_videos(["/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/video_48khz_stereo_16bits.mp4"])

print("dataset")
print(dataset)
