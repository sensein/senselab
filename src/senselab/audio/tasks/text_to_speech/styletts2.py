"""This module provides the implementation of Style-TTS-based text-to-speech pipelines."""

import time
from typing import Any, Dict, List, Literal, Optional

import nltk
import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, TorchModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


class StyleTTS2:
    """A class for managing Torch-based StyleTTS2 models."""

    _models: Dict[str, "StyleTTS2"] = {}
    DEF_ALPHA = 0.3
    DEF_BETA = 0.1
    DEF_EMBEDDING_SCALE = 1

    LF_ALPHA = 0.5
    LF_BETA = 0.9
    LF_EMBEDDING_SCALE = 1.5

    ST_ALPHA = 0.5
    ST_BETA = 0.9
    ST_EMBEDDING_SCALE = 1.5
    DIFFUSION_STEPS = 10

    @classmethod
    def _get_style_tts_2_model(
        cls,
        model: TorchModel = TorchModel(path_or_uri="wilke0818/StyleTTS2-TorchHub", revision="main"),
        language: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        pretrain_data: Optional[Literal["LibriTTS", "LJSpeech"]] = "LibriTTS",
        force_reload: Optional[bool] = False,
    ) -> Any:  # noqa ANN401
        """Get or create a StyleTTS2 model.

        Args:
            model (TorchModel): The Torch model (default is "wilke0818/StyleTTS2-TorchHub:main").
            language (Optional[Language]): The language of the text (default is None).
                The only supported language is "en" for now.
            device (DeviceType): The device to run the model on (default is None). Supported devices are CPU and CUDA.
            pretrain_data (Optional[str]): the dataset that StyleTTS was trained on, currently only supports LibriTTS
                and LJSpeech
            force_reload (Optional[bool]): Whether to require a reload of the Github repository that is being
                used to load the StyleTTS model rather than using a cached version. Defaulting to False for
                efficiency, however, changes to the model and code might not be loaded as a result

        Returns:
            model: The Torch-based StyleTTS2 model.
        """
        if language == Language(language_code="en"):
            model_name: str = "styletts2"  # This is the default model they have for English.
        else:
            raise NotImplementedError("Only English is supported for now.")
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        key = f"{model.path_or_uri}-{model.revision}-{language.name}-{device.value}-{pretrain_data}"
        if key not in cls._models or force_reload:
            my_model = torch.hub.load(
                f"{model.path_or_uri}:{model.revision}",
                model_name,
                trust_repo=True,
                force_reload=force_reload,
                pretrain_data=pretrain_data,
            )
            cls._models[key] = my_model
        return cls._models[key]

    @classmethod
    def synthesize_texts_with_style_tts_2(
        cls,
        texts: List[str],
        target_audios: List[Audio],
        target_transcripts: List[Optional[str]],
        model: TorchModel = TorchModel(path_or_uri="wilke0818/StyleTTS2-TorchHub", revision="main"),
        language: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        pretrain_data: Optional[Literal["LibriTTS", "LJSpeech"]] = "LibriTTS",
        force_reload: Optional[bool] = False,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        diffusion_steps: Optional[int] = None,
        embedding_scale: Optional[float] = None,
        t: float = 0.7,
    ) -> List[Audio]:
        """Synthesizes speech from all the provided text samples and target voices+transcripts using StyleTTS2.

        Args:
            texts (List[str]): The list of text strings to be synthesized.
            target_audios (List[Audio]):
                The list of audio objects to reference.
            target_transcripts (List[Optional[str]]):
                Transcript for each target audio
            model (TorchModel): The Torch model (default is "wilke0818/StyleTTS2-TorchHub").
            language (Optional[Language]): The language of the text (default is None).
                The only supported language is "en" for now.
            device (Optional[DeviceType]): device to run model on
            pretrain_data (Optional[Literal]): The dataset (LibriTTS/LJSpeech) the model was trained on
            force_reload (Optional[bool]): whether to reload the repository that the model is coming from
            alpha (float): determines the timbre of the speaker. 0 will match the reference audio
                and 1 will match the content of the transcript. Default will vary based on interpreted TTS type:
                no reference transcripts will use 0.3 (whether or not this is longform text) and .5 otherwise.
            beta (float): beta determines the prosody of the speaker. 0 will match the reference audio
                and 1 will match the content of the transcript. Default will vary based on interpreted TTS type:
                no reference transcript and non-longform will use 0.1, longform text and transcripts will use 0.9.
            diffusion_steps (int): sampler is ancestral, the higher the stpes, the more diverse the samples
                are but the slower the synthesis. Defaults to 10 for all types.
            embedding_scale (float): classifier-free guidance scale. The higher the scale, the more conditional
                the style is to the input text and hence more emotional. Default will vary based on interpreted
                TTS type:
                no reference transcripts will use 1 (whether or not this is longform text) and 1.5 otherwise.
            t (float): how to combine previous and current styles in longform text generation.

        Returns:
            List[Audio]: The list of synthesized audio objects.

        Some tips for best quality:
            - Make sure reference audio is clean and between 2 second and 12 seconds.
            - Play around with alpha, beta, the number of diffusion steps, and the embedding scale for your
                reference audio(s) in order to find a balance between matching the reference voice and having
                the voice sound natural with the text.
            - Default values for these parameters are based off of LibriTTS pre-trained model examples the original
                authors provided.

        The original repo of the model is: https://github.com/yl4579/StyleTTS2.
        """
        nltk.download("punkt")
        nltk.download("punkt_tab")
        # Take the start time of the model initialization
        start_time_model = time.time()
        my_model = cls._get_style_tts_2_model(model, language, device, pretrain_data, force_reload)

        # Take the end time of the model initialization
        end_time_model = time.time()
        # Print the time taken for initialize the StyleTTS2 model
        elapsed_time_pipeline = end_time_model - start_time_model
        logger.info(f"Time taken to initialize the StyleTTS2 model: {elapsed_time_pipeline:.2f} seconds")

        # Take the start time of text-to-speech synthesis
        start_time_tts = time.time()
        audios = []
        diffusion_steps = diffusion_steps if diffusion_steps else cls.DIFFUSION_STEPS

        for i, text in enumerate(texts):
            sentences = text.split(".")
            target_audio = target_audios[i]

            duration = target_audio.waveform.shape[1] / target_audio.sampling_rate
            if duration < 1 or duration > 12:
                logger.warning(
                    f"Warning: Reference audio at index {i} has a duration of {duration} seconds. "
                    "It is recommended to be between 1 second and 12 seconds."
                )

            if len(sentences) > 2 or len(text) > 200:  # arbitrary but tries to handle false positives and negatives
                # longform text generation
                alpha = alpha if alpha else cls.LF_ALPHA
                beta = beta if beta else cls.LF_BETA
                embedding_scale = embedding_scale if embedding_scale else cls.LF_EMBEDDING_SCALE

                s_ref = my_model._compute_style(target_audio.waveform.squeeze(), target_audio.sampling_rate)
                wavs = []
                s_prev = None
                for sentence in sentences:
                    if sentence.strip() == "":
                        continue
                    sentence += "."  # add it back

                    wav, s_prev = my_model._LFinference(
                        sentence,
                        s_prev,
                        s_ref,
                        alpha=alpha,
                        beta=beta,  # make it more suitable for the text
                        t=t,
                        diffusion_steps=diffusion_steps,
                        embedding_scale=embedding_scale,
                    )
                    wavs.append(wav)
                wav_out = np.concatenate(wavs)

            elif target_transcripts[i]:
                # Considered Style Transfer rather than regular TTS
                alpha = alpha if alpha else cls.ST_ALPHA
                beta = beta if beta else cls.ST_BETA
                embedding_scale = embedding_scale if embedding_scale else cls.ST_EMBEDDING_SCALE

                target_transcript = target_transcripts[i]
                ref_style = my_model._compute_style(target_audio.waveform.squeeze(), target_audio.sampling_rate)
                wav_out = my_model._STinference(
                    text,
                    ref_style,
                    target_transcript,
                    alpha=alpha,
                    beta=beta,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                )
            else:
                alpha = alpha if alpha else cls.DEF_ALPHA
                beta = beta if beta else cls.DEF_BETA
                embedding_scale = embedding_scale if embedding_scale else cls.DEF_EMBEDDING_SCALE

                ref_style = my_model._compute_style(target_audio.waveform.squeeze(), target_audio.sampling_rate)
                wav_out = my_model._inference(
                    text,
                    ref_style,
                    alpha=alpha,
                    beta=beta,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                )
            audios.append(Audio(waveform=wav_out, sampling_rate=24_000))

        # Take the end time of the text-to-speech synthesis
        end_time_tts = time.time()
        # Print the time taken for text-to-speech synthesis
        elapsed_time_tts = end_time_tts - start_time_tts
        logger.info(f"Time taken for synthesizing audios: {elapsed_time_tts:.2f} seconds")

        return audios
