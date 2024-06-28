"""Force aligns a transcript with an audio file."""

from typing import List

import torch
from threadpoolctl import threadpool_limits
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.forced_alignment.align import (
    SingleSegment,
    align,
    convert_to_scriptline,
)
from senselab.audio.tasks.forced_alignment.constants import DEFAULT_ALIGN_MODELS_HF  # TODO add support for TORCH
from senselab.utils.data_structures.script_line import ScriptLine


def align_transcriptions(
    audios: List[Audio], transcriptions: List[ScriptLine], language: str = "en"
) -> List[List[ScriptLine]]:
    """Aligns transcriptions with the given audio using a wav2vec2.0 model.

    Args:
        audios (List[Audio]): The list of audio objects to be aligned.
        transcriptions (List[ScriptLine]): The list of transcriptions corresponding to the audio objects.
        language (str): The language of the audio (default is "en").

    Returns:
        List[List[ScriptLine]]: The list of aligned script lines for each audio.
    """
    aligned_script_lines = []

    # Define the language code and load model
    language_code = language
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = DEFAULT_ALIGN_MODELS_HF.get(language_code, "facebook/wav2vec2-base-960h")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    for audio, transcription in zip(audios, transcriptions):
        print(transcription.text)
        # Ensure start and end are not None
        start = transcription.start if transcription.start is not None else 0.0
        end = transcription.end if transcription.end is not None else audio.waveform.shape[1] / audio.sampling_rate

        # Ensure text is not None
        text = transcription.text if transcription.text is not None else ""

        # Align each segment of the transcription
        segments = [
            SingleSegment(
                start=start, end=end, text=text, clean_char=None, clean_cdx=None, clean_wdx=None, sentence_spans=None
            )
        ]

        with threadpool_limits(limits=1, user_api="blas"):
            alignment = align(
                transcript=segments,
                model=model,
                align_model_metadata={
                    "dictionary": processor.tokenizer.get_vocab(),
                    "language": language_code,
                    "type": "huggingface",
                },
                audio=audio,
                device=device,
                return_char_alignments=True,
            )
            aligned_script_lines.append(convert_to_scriptline(alignment))

    return aligned_script_lines
