"""This module implements some utilities for the speech-to-text task."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationInfo, field_validator
from transformers import pipeline

from senselab.utils.data_structures.audio import Audio
from senselab.utils.data_structures.language import Language
from senselab.utils.device import DeviceType, _select_device_and_dtype
from senselab.utils.hf import HFModel, SenselabModel


class Transcript(BaseModel):
    """A class to represent a transcript.

    Attributes:
        text (str): The full text of the transcript.
        chunks (Optional[List['Transcript.Chunk']]): A list of chunks of the transcript.
    """

    text: str
    chunks: Optional[List["Transcript.Chunk"]] = None

    def get_text(self) -> str:
        """Get the full text of the transcript.

        Returns:
            str: The full text of the transcript.
        """
        return self.text

    def get_chunks(self) -> Optional[List["Transcript.Chunk"]]:
        """Get the list of chunks in the transcript.

        Returns:
            Optional[List['Transcript.Chunk']]: The list of chunks.
        """
        return self.chunks

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transcript":
        """Create a Transcript instance from a dictionary.

        Args:
            d (Dict[str, Any]): The dictionary containing the transcript data.

        Returns:
            Transcript: An instance of Transcript.
        """
        return cls(
            text=d["text"].strip(), chunks=[cls.Chunk.from_dict(c) for c in d["chunks"]] if "chunks" in d else None
        )

    class Chunk(BaseModel):
        """A class to represent a chunk of the transcript.

        Attributes:
            text (str): The text of the chunk.
            start (float): The start timestamp of the chunk.
            end (float): The end timestamp of the chunk.
        """

        text: str
        start: float
        end: float

        @field_validator("text")
        def text_must_be_non_empty(cls, v: str, _: ValidationInfo) -> str:
            """Validate that the text is non-empty.

            Args:
                v (str): The text of the chunk.

            Returns:
                str: The validated text of the chunk.
            """
            if not v.strip():
                raise ValueError("Chunk text must be non-empty")
            return v

        @field_validator("start", "end")
        def timestamps_must_be_positive(cls, v: float, _: ValidationInfo) -> float:
            """Validate that the start and end timestamps are positive.

            Args:
                v (float): The timestamp of the chunk.

            Returns:
                float: The validated timestamp.
            """
            if v < 0:
                raise ValueError("Timestamps must be non-negative")
            return v

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> "Transcript.Chunk":
            """Create a Chunk instance from a dictionary.

            Args:
                d (Dict[str, Any]): The dictionary containing the chunk data.

            Returns:
                Transcript.Chunk: An instance of Chunk.
            """
            return cls(text=d["text"].strip(), start=d["timestamp"][0], end=d["timestamp"][1])


class ASRPipelineFactory:
    """A factory for managing ASR pipelines."""

    _pipelines: Dict[str, pipeline] = {}

    @classmethod
    def _get_hf_asr_pipeline(
        cls,
        model: HFModel,
        return_timestamps: Optional[str],
        max_new_tokens: int,
        chunk_length_s: int,
        batch_size: int,
        device: Optional[DeviceType] = None,
    ) -> pipeline:
        """Get or create a Hugging Face ASR pipeline.

        Args:
            model (HFModel): The Hugging Face model.
            return_timestamps (Optional[str]): The level of timestamp details.
            max_new_tokens (int): The maximum number of new tokens.
            chunk_length_s (int): The length of audio chunks in seconds.
            batch_size (int): The batch size for processing.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            pipeline: The ASR pipeline.
        """
        key = f"{model.path_or_uri}-{device}-{return_timestamps}-{max_new_tokens}-{chunk_length_s}-{batch_size}"
        if key not in cls._pipelines:
            device, torch_dtype = _select_device_and_dtype(
                user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
            )
            cls._pipelines[key] = pipeline(
                "automatic-speech-recognition",
                model=model.path_or_uri,
                revision=model.revision,
                return_timestamps=return_timestamps,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                device=device.value,
                torch_dtype=torch_dtype,
            )
        return cls._pipelines[key]


def transcribe_audios(
    audios: List[Audio], model: SenselabModel, language: Optional[Language] = None, device: Optional[DeviceType] = None
) -> List[Transcript]:
    """Transcribes all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be transcribed.
        model (SenselabModel): The model used for transcription.
        language (Optional[Language]): The language of the audio (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Transcript]: The list of transcriptions.
    """
    if isinstance(model, HFModel):
        return transcribe_audios_with_transformers(audios=audios, model=model, language=language, device=device)
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")


def transcribe_audios_with_transformers(
    audios: List[Audio],
    model: HFModel = HFModel(path_or_uri="openai/whisper-tiny"),
    language: Optional[Language] = None,
    return_timestamps: Optional[str] = "word",
    max_new_tokens: int = 128,
    chunk_length_s: int = 30,
    batch_size: int = 16,
    device: Optional[DeviceType] = None,
) -> List[Transcript]:
    """Transcribes all audio samples in the dataset.

    Args:
        audios (List[Audio]): The list of audio objects to be transcribed.
        model (HFModel): The Hugging Face model used for transcription.
        language (Optional[Language]): The language of the audio (default is None).
        return_timestamps (Optional[str]): The level of timestamp details (default is "word").
        max_new_tokens (int): The maximum number of new tokens (default is 128).
        chunk_length_s (int): The length of audio chunks in seconds (default is 30).
        batch_size (int): The batch size for processing (default is 16).
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Transcript]: The list of transcriptions.
    """

    def _audio_to_huggingface_dict(audio: Audio) -> Dict:
        """Convert an Audio object to a dictionary that can be used by the transformers pipeline.

        Args:
            audio (Audio): The audio object.

        Returns:
            Dict: The dictionary representation of the audio object.
        """
        return {
            "array": audio.waveform.squeeze().numpy(),
            "sampling_rate": audio.sampling_rate,
        }

    pipe = ASRPipelineFactory._get_hf_asr_pipeline(
        model=model,
        return_timestamps=return_timestamps,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        device=device,
    )

    formatted_audios = [_audio_to_huggingface_dict(audio) for audio in audios]
    transcriptions = pipe(
        formatted_audios, generate_kwargs={"language": f"{language.name.lower()}"} if language else {}
    )

    return [Transcript.from_dict(t) for t in transcriptions]
