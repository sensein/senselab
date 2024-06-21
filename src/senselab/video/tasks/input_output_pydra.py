"""This module defines a pydra API for the video input and output task."""

import pydra

from senselab.video.tasks.input_output import extract_audios_from_local_videos

extract_audios_from_local_videos_pt = pydra.mark.task(extract_audios_from_local_videos)
