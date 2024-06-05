"""This module defines a pydra API for the preprocessing task."""

import pydra

from senselab.audio.tasks.preprocessing import (
    chunk_audios,
    downmix_audios_to_mono,
    resample_audios,
    select_channel_from_audios,
)

resample_audios_pt = pydra.mark.task(resample_audios)
downmix_audios_to_mono_pt = pydra.mark.task(downmix_audios_to_mono)
chunk_audios_pt = pydra.mark.task(chunk_audios)
select_channel_from_audios_pt = pydra.mark.task(select_channel_from_audios)
