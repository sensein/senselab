"""This module provides the Speechbrain interface for language identification."""

from typing import Dict, List, Optional

from speechbrain.inference.classifiers import EncoderClassifier

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, SpeechBrainModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger

# lang-id-voxlingua107-ecapa and speechbrain/lang-id-commonlanguage_ecapa were both trained on 16kHz audio
# and these are the only official models available
TRAINING_SAMPLE_RATE = 16000


class SpeechBrainLanguageIdentifier:
    """A factory for managing SpeechBrain language identification pipelines."""

    _models: Dict[str, EncoderClassifier] = {}

    @classmethod
    def _get_language_identifier(
        cls,
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> EncoderClassifier:
        """Get or create a SpeechBrain language identification model."""
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            cls._models[key] = EncoderClassifier.from_hparams(
                source=model.path_or_uri, run_opts={"device": device.value}
            )
        return cls._models[key]

    @classmethod
    def identify_languages(
        cls,
        audios: List[Audio],
        model: SpeechBrainModel = SpeechBrainModel(
            path_or_uri="speechbrain/lang-id-voxlingua107-ecapa", revision="main"
        ),
        device: Optional[DeviceType] = None,
    ) -> List[Language | None]:
        """Identifies the language for all provided audio samples.

        Args:
            audios (List[Audio]): List of audio objects.
            model (SpeechBrainModel): SpeechBrain language identification model.
            device (Optional[DeviceType]): Device to run the model on (default is None).

        Returns:
            List[Language]: List of identified Language objects.
        """
        logger.info("Initializing SpeechBrain language identifier model...")
        identifier = cls._get_language_identifier(model, device)

        results: List[Language | None] = []
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError("Audio waveform must be mono (1 channel).")
            if audio.sampling_rate != TRAINING_SAMPLE_RATE:
                raise ValueError(f"{model.path_or_uri} trained on {TRAINING_SAMPLE_RATE} \
                                sample audio, but audio has sampling rate {audio.sampling_rate}.")

            logger.info(f"Processing audio with sampling rate {audio.sampling_rate}...")
            prediction = identifier.classify_batch(audio.waveform)
            if prediction is None or len(prediction) == 0:
                results.append(None)
            else:
                lang_code = prediction[3][0].split(":")[0].strip()  # Extract language code from model output
                print("Identified language:", lang_code)
                logger.info(f"Identified language: {lang_code}")
                results.append(Language(language_code=lang_code))

        return results
