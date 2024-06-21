"""Provides for extracting speaker embeddings from a list of audios using speechbrain."""

from typing import List, Optional

from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, stack

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


class SpeechBrainEmbeddings:
    """A class for extracting speaker embeddings using speechbrain models."""

    @classmethod
    def extract_speaker_embeddings_from_audios(
        self, audios: List[Audio], model: HFModel, device: Optional[DeviceType] = None
    ) -> Tensor:
        """Compute the speaker embedding of an audio signal.

        Args:
            audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
            model (SenselabModel): The model used to compute the embeddings. Must be a speechbrain model.
            device (Optional[DeviceType]): The device on which to run the model (e.g., 'cpu', 'cuda').
                Defaults to None, which means the default device will be used.

        Returns:
            List[Tensor]: A list of tensors containing the speaker embeddings for each audio file.

        """
        # Extract device name from Device object
        if device is not None:
            device_name = device.value

        classifier = EncoderClassifier.from_hparams(source=model.path_or_uri, run_opts={"device": device_name})
        embeddings = classifier.encode_batch(stack([audio.waveform for audio in audios]).squeeze())
        return embeddings.squeeze()
