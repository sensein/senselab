"""Transcribes audio files with timestamps."""

'''
# TODO: Please double-check this because tests are failing
from typing import List

import pydra

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.forced_alignment.constants import SAMPLE_RATE
from senselab.audio.tasks.forced_alignment.forced_alignment import (
    align_transcriptions,
)
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import Language
from senselab.utils.data_structures import HFModel
from senselab.utils.data_structures import ScriptLine
from senselab.utils.tasks.batching import batch_list


def transcribe_timestamped(
    audios: List[Audio],
    model: HFModel = HFModel(path_or_uri="openai/whisper-tiny"),
    language: Language = Language(language_code="en"),
    n_batches: int = 1,
) -> List[List[ScriptLine]]:
    """Transcribes a list of audio files and timestamps them using forced alignment.

    This function processes the given list of Audio objects by performing
    necessary preprocessing steps (such as downmixing channels and resampling),
    transcribes the audio using the specified speech-to-text model, and applies
    forced alignment to generate a list of ScriptLine objects with timestamps.

    Args:
        audios (list[Audio]): List of Audio objects to be transcribed and
                              timestamped.
        model (HFModel, optional): A Huggingface model for speech-to-text.
                                   Defaults to 'whisper'.
        language (Language, optional): Language object for the transcription.
                                       If None, language detection is triggered
                                       for the 'whisper' model. Defaults to None.
        n_batches (int, optional): The number of batches to split over in the
                                   workflow.

    Returns:
        List[List[ScriptLine]]: List of ScriptLine objects resulting from the
                          transcriptions with timestamps.
    """
    if not audios:
        raise ValueError("The list of audios is empty.")

    for i in range(len(audios)):
        if audios[i].waveform.shape[0] > 1:
            audios[i] = downmix_audios_to_mono(audios=[audios[i]])[0]
        if audios[i].sampling_rate != SAMPLE_RATE:
            audios[i] = resample_audios(audios=[audios[i]], resample_rate=SAMPLE_RATE)[0]

    batched_audios = batch_list(items=audios, n_batches=n_batches)

    wf = pydra.Workflow(
        name="wf",
        input_spec=["batched_audios", "model", "language"],
        batched_audios=batched_audios,
        model=model,
        language=language,
        cache_dir=None,
    )

    @pydra.mark.task
    def transcribe_task(audios: List[Audio], model: HFModel, language: Language) -> List[tuple]:
        transcriptions = transcribe_audios(audios=audios, model=model, language=language)
        return list(zip(audios, transcriptions, [language] * len(audios)))

    wf.add(
        transcribe_task(
            name="transcribe",
            audios=wf.lzin.batched_audios,
            model=wf.lzin.model,
            language=wf.lzin.language,
        )
    ).split("batched_audios", batched_audios=wf.transcribe.lzin.batched_audios)

    align_transcriptions_task = pydra.mark.task(align_transcriptions)
    wf.add(
        align_transcriptions_task(
            name="align",
            audios_and_transcriptions_and_language=wf.transcribe.lzout.out,
        )
    )

    wf.set_output(
        [
            ("transcriptions", wf.transcribe.lzout.out),
            ("aligned_transcriptions", wf.align.lzout.out),
        ]
    )

    # Execute the workflow
    with pydra.Submitter(plugin="cf") as sub:
        sub(wf)

    return wf.result()[0].output.aligned_transcriptions
'''
