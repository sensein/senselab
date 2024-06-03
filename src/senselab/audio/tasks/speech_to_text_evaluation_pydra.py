"""This module defines a pydra API for the speech to text evaluation task."""

import pydra

from senselab.audio.tasks.speech_to_text_evaluation import (
    calculate_cer,
    calculate_mer,
    calculate_wer,
    calculate_wil,
    calculate_wip,
)

calculate_wer_pt = pydra.mark.task(calculate_wer)
calculate_mer_pt = pydra.mark.task(calculate_mer)
calculate_wil_pt = pydra.mark.task(calculate_wil)
calculate_wip_pt = pydra.mark.task(calculate_wip)
calculate_cer_pt = pydra.mark.task(calculate_cer)
