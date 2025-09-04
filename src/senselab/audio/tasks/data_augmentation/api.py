"""This module provides the API for data augmentation tasks."""

from typing import List, Optional, Union

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.data_augmentation.audiomentations import augment_audios_with_audiomentations
from senselab.audio.tasks.data_augmentation.torch_audiomentations import augment_audios_with_torch_audiomentations
from senselab.utils.data_structures import DeviceType

try:
    from audiomentations import Compose as AudiomentationsCompose

    AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOMENTATIONS_AVAILABLE = False

try:
    from torch_audiomentations import Compose as TorchAudiomentationsCompose

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False


def augment_audios(
    audios: List[Audio],
    augmentation: Union["TorchAudiomentationsCompose", "AudiomentationsCompose"],
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Apply audio data augmentation to a list of `Audio` objects.

    This function provides a unified interface for applying augmentations
    using either [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
    or [audiomentations](https://github.com/iver56/audiomentations).

    - If the augmentation is a ``TorchAudiomentationsCompose``, the operation
      can run on CPU or CUDA (GPU), as specified by ``device``.
    - If the augmentation is an ``AudiomentationsCompose``, augmentation runs
      only on CPU. For efficiency, multiple files are processed in parallel
      using Pydra with the concurrent futures plugin.

    Args:
        audios (list[Audio]):
            List of `Audio` objects to augment.
        augmentation (TorchAudiomentationsCompose | AudiomentationsCompose):
            A composition of augmentations. Must be created from either
            ``torch_audiomentations.Compose`` or ``audiomentations.Compose``.
        device (DeviceType, optional):
            Target device for augmentation (relevant only for torch-audiomentations).
            Accepted values:
              * ``DeviceType.CPU`` → Run on CPU.
              * ``DeviceType.CUDA`` → Run on GPU (if CUDA is available).
            Defaults to ``None`` (equivalent to ``DeviceType.CUDA``, if available
            ``DeviceType.CPU`` otherwise).

    Returns:
        list[Audio]: A list of augmented `Audio` objects, matching the order of input.

    Raises:
        ModuleNotFoundError:
            If the selected augmentation backend is not installed.
        ValueError:
            If an unsupported augmentation type is provided.

    Example (torch-audiomentations):
        >>> from torch_audiomentations import Compose, AddColoredNoise
        >>> from senselab.audio.tasks.data_augmentation.api import augment_audios
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> aug = Compose([AddColoredNoise(p=1.0)])
        >>> audio = Audio(filepath="/absolute/path/to/sample.wav")
        >>> augmented = augment_audios([audio], aug, device=DeviceType.CPU)
        >>> len(augmented)
        1

    Example (audiomentations):
        >>> from audiomentations import Compose, PitchShift
        >>> from senselab.audio.tasks.data_augmentation.api import augment_audios
        >>> from senselab.audio.data_structures import Audio
        >>> aug = Compose([PitchShift(min_semitones=-2, max_semitones=2, p=1.0)])
        >>> audio = Audio(filepath="/absolute/path/to/sample.wav")
        >>> augmented = augment_audios([audio], aug)
        >>> len(augmented)
        1
    """
    if not AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`"
        )
    if not TORCH_AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`torch-audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`"
        )
    if isinstance(augmentation, TorchAudiomentationsCompose):
        return augment_audios_with_torch_audiomentations(audios, augmentation, device)
    elif isinstance(augmentation, AudiomentationsCompose):
        return augment_audios_with_audiomentations(audios, augmentation, plugin="cf")
    else:
        raise ValueError(
            "Unsupported augmentation type." "Use either torch_audiomentations.Compose or audiomentations.Compose."
        )
